# backend.py
# Flask API for Physics AI Coach
# - GET /                -> health
# - GET /problems        -> list problems from Notion (NOW FULLY PAGINATED)
# - GET /problem/<id>    -> full fields for one problem
# - GET /steps/<id>      -> step-by-step plan (Notion or auto-LLM)
# - POST /chat           -> checks a student's step / next hint
# 
# ========================================
# *** MULTIUSER SYSTEM - ADDED 2 FEB 2026 ***
# - POST /api/register           -> register new user
# - POST /api/login              -> login user
# - GET  /api/verify-session     -> verify JWT token
# - POST /api/track-activity     -> track user activity
# - GET  /api/user-progress      -> get current user progress
# - GET  /api/all-users          -> get all users (admin only)
# ========================================

import os, re, json
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from functools import wraps
from datetime import datetime, timedelta

# ========================================
# *** NEW: Import for authentication ***
# ========================================
try:
    import bcrypt
    import jwt
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    print("[WARNING] bcrypt or PyJWT not installed. Auth endpoints will not work.")
    print("[WARNING] Install with: pip install bcrypt PyJWT")

# ==============================
# ENV
# ==============================
NOTION_API_KEY      = os.environ.get("NOTION_API_KEY", "")
NOTION_DB_ID        = os.environ.get("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
FRONTEND_ORIGIN     = os.environ.get("FRONTEND_ORIGIN", "https://aiclub.com.mx")

# ========================================
# *** NEW: Environment variables for multiuser system ***
# ========================================
USERS_DB_ID         = os.environ.get("USERS_DB_ID", "")
ACTIVITY_DB_ID      = os.environ.get("ACTIVITY_DB_ID", "")
JWT_SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "change-this-secret-key-in-production")

# ==============================
# Flask
# ==============================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://aiclub.com.mx", "http://aiclub.com.mx"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],  # *** ADDED Authorization header ***
        "supports_credentials": False
    }
})

# ==============================
# Notion helpers
# ==============================
NOTION_BASE = "https://api.notion.com/v1"
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

UUID_RE = re.compile(r"[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

def sanitize_uuid(value: str) -> str:
    """Return a Notion-acceptable UUID (with hyphens) or empty if not found."""
    match = UUID_RE.search(value)
    if not match:
        return ""
    raw = match.group(0).replace("-", "")
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

def fetch_page(page_id: str) -> Dict[str, Any]:
    """Retrieve a single Notion page by ID."""
    pid = sanitize_uuid(page_id)
    if not pid:
        raise ValueError("Invalid page ID")
    
    url = f"{NOTION_BASE}/pages/{pid}"
    resp = requests.get(url, headers=NOTION_HEADERS, timeout=10)
    resp.raise_for_status()
    page_data = resp.json()
    
    props = page_data.get("properties", {})
    out = {"id": pid}
    
    for k, v in props.items():
        ptype = v.get("type")
        if ptype == "title":
            arr = v.get("title", [])
            out[k] = arr[0].get("plain_text", "") if arr else ""
        elif ptype == "rich_text":
            arr = v.get("rich_text", [])
            out[k] = arr[0].get("plain_text", "") if arr else ""
        elif ptype == "number":
            out[k] = v.get("number")
        elif ptype == "select":
            sel = v.get("select")
            out[k] = sel.get("name", "") if sel else ""
        elif ptype == "multi_select":
            out[k] = [ms.get("name", "") for ms in v.get("multi_select", [])]
        elif ptype == "url":
            out[k] = v.get("url", "")
        elif ptype == "files":
            files = v.get("files", [])
            out[k] = [f.get("file", {}).get("url") or f.get("external", {}).get("url") for f in files]
        elif ptype == "email":  # *** NEW: Support for email type ***
            out[k] = v.get("email", "")
        elif ptype == "date":  # *** NEW: Support for date type ***
            date_obj = v.get("date")
            if date_obj:
                out[k] = date_obj.get("start", "")
            else:
                out[k] = None
    
    return out

# ====================================================================
# *** CAMBIO BLINDADO CRÍTICO: IMPLEMENTACIÓN DE PAGINACIÓN COMPLETA ***
# ====================================================================

def query_database(database_id: str, filter_obj: dict = None, sorts: list = None) -> List[Dict[str, Any]]:
    """Query a Notion database with full pagination (brings ALL results)."""
    db_id = sanitize_uuid(database_id)
    if not db_id:
        raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/databases/{db_id}/query"
    
    all_pages = []
    has_more = True
    start_cursor = None
    
    while has_more:
        payload = {}
        if filter_obj:
            payload["filter"] = filter_obj
        if sorts:
            payload["sorts"] = sorts
        if start_cursor:
            payload["start_cursor"] = start_cursor

        resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        all_pages.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")
    
    results = []
    for page in all_pages:
        pid = page.get("id", "")
        props = page.get("properties", {})
        obj = {"id": pid}
        for k, v in props.items():
            ptype = v.get("type")
            if ptype == "title":
                arr = v.get("title", [])
                obj[k] = arr[0].get("plain_text", "") if arr else ""
            elif ptype == "rich_text":
                arr = v.get("rich_text", [])
                obj[k] = arr[0].get("plain_text", "") if arr else ""
            elif ptype == "number":
                obj[k] = v.get("number")
            elif ptype == "select":
                sel = v.get("select")
                obj[k] = sel.get("name", "") if sel else ""
            elif ptype == "multi_select":
                obj[k] = [ms.get("name", "") for ms in v.get("multi_select", [])]
            elif ptype == "email":  # *** NEW ***
                obj[k] = v.get("email", "")
            elif ptype == "date":  # *** NEW ***
                date_obj = v.get("date")
                if date_obj:
                    obj[k] = date_obj.get("start", "")
                else:
                    obj[k] = None
        results.append(obj)
    
    return results

# ====================================================================
# *** FIN DEL CAMBIO BLINDADO CRÍTICO ***
# ====================================================================

# ========================================
# *** NEW: Helper function to create Notion pages ***
# ========================================
def create_page_in_database(database_id: str, properties: dict) -> dict:
    """Create a new page in a Notion database."""
    db_id = sanitize_uuid(database_id)
    if not db_id:
        raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/pages"
    
    payload = {
        "parent": {"database_id": db_id},
        "properties": properties
    }
    
    resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def parse_steps(text: str) -> List[Dict[str, str]]:
    """Parse step-by-step text into structured steps."""
    if not text or not text.strip():
        return []
    
    lines = text.strip().split("\n")
    steps = []
    
    for i, ln in enumerate(lines, start=1):
        ln = ln.strip()
        if not ln:
            continue
        
        clean_ln = re.sub(r'^(\d+[\)\.:]?\s*|Step\s+\d+[\)\.:]?\s*)', '', ln, flags=re.IGNORECASE).strip()
        
        if clean_ln and len(clean_ln) > 2:
            steps.append({
                "id": str(i),
                "description": clean_ln,
                "rubric": ""
            })
    
    if steps and all(len(s["description"]) < 5 for s in steps):
        return []
    
    return steps

# ==============================
# NEW: Notion Blocks API helpers
# ==============================
def fetch_page_blocks(page_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all blocks (content) from a Notion page.
    This includes toggles, paragraphs, etc.
    """
    pid = sanitize_uuid(page_id)
    if not pid:
        raise ValueError("Invalid page ID")
    
    url = f"{NOTION_BASE}/blocks/{pid}/children"
    
    all_blocks = []
    has_more = True
    start_cursor = None
    
    while has_more:
        params = {}
        if start_cursor:
            params["start_cursor"] = start_cursor
        
        resp = requests.get(url, headers=NOTION_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        all_blocks.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")
    
    return all_blocks

def extract_text_from_rich_text(rich_text_array):
    """Extract plain text from Notion's rich_text format."""
    if not rich_text_array:
        return ""
    return "".join([item.get("plain_text", "") for item in rich_text_array])

def parse_steps_from_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Parse step-by-step solution from Notion blocks (toggles).
    Looks for toggle blocks that start with "Step X:" or similar.
    Reads the content INSIDE each toggle (children blocks).
    """
    steps = []
    step_counter = 1
    
    for block in blocks:
        block_type = block.get("type")
        
        if block_type == "toggle":
            toggle_data = block.get("toggle", {})
            rich_text = toggle_data.get("rich_text", [])
            title_text = extract_text_from_rich_text(rich_text)
            
            if title_text and (title_text.lower().startswith("step") or any(title_text.startswith(f"{n}.") for n in range(1, 20))):
                clean_title = re.sub(r'^(Step\s+\d+:\s*)', '', title_text, flags=re.IGNORECASE).strip()
                
                has_children = block.get("has_children", False)
                block_id = block.get("id", "")
                
                content_lines = []
                
                if has_children and block_id:
                    try:
                        children_url = f"{NOTION_BASE}/blocks/{block_id}/children"
                        children_resp = requests.get(children_url, headers=NOTION_HEADERS, timeout=10)
                        
                        if children_resp.status_code == 200:
                            children_data = children_resp.json()
                            children_blocks = children_data.get("results", [])
                            
                            for child_block in children_blocks:
                                child_type = child_block.get("type")
                                
                                if child_type == "paragraph":
                                    para_data = child_block.get("paragraph", {})
                                    para_text = extract_text_from_rich_text(para_data.get("rich_text", []))
                                    if para_text.strip():
                                        content_lines.append(para_text)
                                
                                elif child_type == "bulleted_list_item":
                                    bullet_data = child_block.get("bulleted_list_item", {})
                                    bullet_text = extract_text_from_rich_text(bullet_data.get("rich_text", []))
                                    if bullet_text.strip():
                                        content_lines.append("- " + bullet_text)
                                
                                elif child_type == "numbered_list_item":
                                    numbered_data = child_block.get("numbered_list_item", {})
                                    numbered_text = extract_text_from_rich_text(numbered_data.get("rich_text", []))
                                    if numbered_text.strip():
                                        content_lines.append("- " + numbered_text)
                    
                    except Exception as e:
                        print(f"[WARN] Could not fetch children for block {block_id}: {e}")
                
                if content_lines:
                    full_description = clean_title + "\n\n" + "\n".join(content_lines)
                else:
                    full_description = clean_title
                
                if full_description and len(full_description.strip()) > 5:
                    steps.append({
                        "id": str(step_counter),
                        "description": full_description,
                        "rubric": ""
                    })
                    step_counter += 1
    
    return steps

# ==============================
# Anthropic helper with improved error handling
# ==============================
def call_anthropic_api(prompt: str, max_tokens: int = 1024) -> str:
    """
    Call Anthropic API directly via HTTP request.
    Returns the text response or raises an exception.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")
    
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": "You are a helpful IB Physics tutor. Always respond with valid JSON only.",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        print(f"[API] Calling Anthropic API...")
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        
        print(f"[API] Status: {resp.status_code}")
        print(f"[API] Response: {resp.text[:200]}")
        
        if resp.status_code != 200:
            error_data = resp.json() if resp.text else {}
            error_type = error_data.get("error", {}).get("type", "unknown")
            error_msg = error_data.get("error", {}).get("message", resp.text)
            raise Exception(f"Anthropic API error ({error_type}): {error_msg}")
        
        data = resp.json()
        content = data.get("content", [])
        
        if not content:
            raise Exception("Empty response from Anthropic API")
        
        text = content[0].get("text", "")
        return text
        
    except requests.exceptions.Timeout:
        raise Exception("Anthropic API timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Anthropic API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Anthropic API error: {str(e)}")

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate step-by-step solution using LLM if not in Notion."""
    title = problem.get("title") or problem.get("name", "")
    statement = problem.get("statement", "")
    given = problem.get("given_values", "")
    find = problem.get("find", "")
    
    prompt = f"""You are an IB Physics HL teacher creating a step-by-step solution guide.

Problem: {title}
Given: {given}
Find: {find}
Statement: {statement}

Create a clear 4-6 step solution plan for this problem. Each step should be ONE sentence describing what the student needs to do.

Format: Write each step on a new line, numbered like this:
1. First step description
2. Second step description
3. Third step description

Do NOT use JSON. Do NOT use markdown. Just plain numbered steps.

Example format:
1. Identify which object has mass distributed farther from the rotation axis
2. Recall the moment of inertia formula I = Σmr²
3. Compare how mass distribution affects I in both cases
4. Determine which configuration has larger moment of inertia
5. Explain your reasoning using the formula

Now create the steps for the problem above:"""
    
    try:
        text = call_anthropic_api(prompt, max_tokens=512)
        
        text = text.strip()
        
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                lines = text.split("\n")
                if lines[0].strip() in ["json", "text", "plaintext", ""]:
                    text = "\n".join(lines[1:])
            else:
                text = text.replace("```", "")
        
        if text.strip().startswith("{") or text.strip().startswith("["):
            print(f"[ERROR] Claude returned JSON instead of plain text steps: {text[:100]}")
            raise ValueError("Invalid format from Claude")
        
        steps = parse_steps(text)
        
        if steps:
            print(f"[STEPS] Generated {len(steps)} steps with LLM")
            return steps
        else:
            print(f"[STEPS] Failed to parse Claude's response, using fallback")
            raise ValueError("No valid steps parsed")
            
    except Exception as e:
        print(f"[ERROR] Failed to generate steps with LLM: {e}")
        return [
            {"id": "1", "description": "Identify the given data and what needs to be found", "rubric": ""},
            {"id": "2", "description": "Determine which physics principles and formulas apply", "rubric": ""},
            {"id": "3", "description": "Set up the equation with the given values", "rubric": ""},
            {"id": "4", "description": "Solve for the unknown variable", "rubric": ""},
            {"id": "5", "description": "Check units and verify the answer makes physical sense", "rubric": ""},
        ]

# ========================================
# *** NEW: Authentication Middleware ***
# ========================================
def require_auth(f):
    """
    Decorator to require authentication for endpoints.
    Checks for valid JWT token in Authorization header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not AUTH_AVAILABLE:
            return jsonify({"error": "Authentication not available (missing dependencies)"}), 500
        
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({"error": "No token provided"}), 401
        
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            request.user = payload  # Add user info to request object
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
    
    return decorated_function

# ==============================
# Routes (EXISTING - NO CHANGES)
# ==============================
@app.get("/")
def index():
    return jsonify({"status": "ok", "service": "ai-coach-backend"})

@app.get("/problems")
def list_problems():
    if not NOTION_DB_ID:
        return jsonify({"error": "NOTION_DATABASE_ID not set"}), 500
    try:
        items = query_database(NOTION_DB_ID)
        return jsonify({"problems": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/problem/<problem_id>")
def get_problem(problem_id: str):
    try:
        page = fetch_page(problem_id)
        return jsonify({"problem": page})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/steps/<problem_id>")
def get_steps(problem_id: str):
    try:
        page = fetch_page(problem_id)
        
        steps_list = []
        try:
            blocks = fetch_page_blocks(problem_id)
            steps_list = parse_steps_from_blocks(blocks)
            
            if steps_list:
                print(f"[STEPS] Found {len(steps_list)} steps from Notion blocks (toggles)")
                return jsonify({"steps": steps_list})
        except Exception as e:
            print(f"[STEPS] Could not read blocks: {e}")
        
        steps_text = page.get("step_by_step", "")
        steps = parse_steps(steps_text) if steps_text else []
        
        if not steps:
            print(f"[STEPS] No valid steps in Notion, generating with LLM...")
            steps = generate_steps_with_llm(page)
        else:
            print(f"[STEPS] Found {len(steps)} steps from Notion text field")
        
        return jsonify({"steps": steps})
    except Exception as e:
        print(f"[ERROR] Failed to get steps: {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/solution/<problem_id>")
def get_solution(problem_id: str):
    """
    Get the complete official solution for a problem.
    Used when student chooses "View solution" option.
    """
    try:
        problem = fetch_page(problem_id)
        
        steps_list = []
        try:
            blocks = fetch_page_blocks(problem_id)
            steps_list = parse_steps_from_blocks(blocks)
            
            if steps_list:
                print(f"[SOLUTION] Found {len(steps_list)} steps from Notion blocks (toggles)")
        except Exception as e:
            print(f"[SOLUTION] Could not read blocks: {e}")
        
        if not steps_list:
            steps_text = problem.get("step_by_step", "")
            steps_list = parse_steps(steps_text) if steps_text else []
        
        if not steps_list:
            print(f"[SOLUTION] No valid steps found, generating with LLM...")
            steps_list = generate_steps_with_llm(problem)
        
        full_solution_lines = []
        for i, step in enumerate(steps_list, 1):
            full_solution_lines.append(f"{i}. {step.get('description', step.get('text', ''))}")
        
        full_solution = "\n\n".join(full_solution_lines)
        
        final_answer = problem.get("final_answer", "")
        
        return jsonify({
            "steps": steps_list,
            "full_solution": full_solution,
            "final_answer": final_answer,
            "problem_title": problem.get("title") or problem.get("name", "")
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to get solution: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/submit-solution")
def submit_solution():
    """
    Evaluate a student's complete solution to a problem.
    Used in the new flow where student attempts the problem solo first.
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    solution_text = (data.get("solution_text") or "").strip()
    solution_image = data.get("solution_image")
    time_spent_seconds = data.get("time_spent_seconds", 0)
    
    print(f"[SUBMIT] problem_id={problem_id}, time={time_spent_seconds}s, has_image={bool(solution_image)}")
    
    if not problem_id:
        return jsonify({"error": "missing problem_id"}), 400
    
    if not solution_text and not solution_image:
        return jsonify({"error": "No solution provided"}), 400
    
    try:
        problem = fetch_page(problem_id)
    except Exception as e:
        return jsonify({"error": f"Cannot read problem: {e}"}), 500
    
    official_answer = problem.get("final_answer", "")
    problem_statement = problem.get("statement", "")
    problem_title = problem.get("title") or problem.get("name", "")
    problem_given = problem.get("given_values", "")
    problem_find = problem.get("find", "")
    
    if not ANTHROPIC_API_KEY:
        is_correct = len(solution_text) > 50 or bool(solution_image)
        return jsonify({
            "correct": is_correct,
            "score": 50 if is_correct else 0,
            "feedback": "AI evaluation not available. Your answer was recorded.",
            "time_taken": f"{time_spent_seconds // 60}:{time_spent_seconds % 60:02d}"
        })
    
    try:
        if solution_image:
            print(f"[IMAGE] Evaluating solution with Vision API")
            
            evaluation = evaluate_solution_with_image(
                problem_title=problem_title,
                problem_statement=problem_statement,
                problem_given=problem_given,
                problem_find=problem_find,
                solution_text=solution_text,
                solution_image=solution_image,
                official_answer=official_answer
            )
        else:
            evaluation = evaluate_solution_with_text(
                problem_title=problem_title,
                problem_statement=problem_statement,
                problem_given=problem_given,
                problem_find=problem_find,
                solution_text=solution_text,
                official_answer=official_answer
            )
        
        minutes = time_spent_seconds // 60
        seconds = time_spent_seconds % 60
        time_str = f"{minutes}:{seconds:02d}"
        
        return jsonify({
            "correct": bool(evaluation.get("correct", False)),
            "score": int(evaluation.get("score", 0)),
            "feedback": evaluation.get("feedback", ""),
            "specific_mistakes": evaluation.get("specific_mistakes", []),
            "time_taken": time_str
        })
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return jsonify({
            "correct": False,
            "score": 0,
            "feedback": f"Error evaluating solution: {str(e)[:100]}",
            "specific_mistakes": [],
            "time_taken": f"{time_spent_seconds // 60}:{time_spent_seconds % 60:02d}"
        }), 500

def evaluate_solution_with_text(problem_title, problem_statement, problem_given, problem_find, solution_text, official_answer):
    """
    Evaluate a text-only solution (existing logic extracted into function)
    """
    eval_prompt = f"""You are an IB Physics HL examiner evaluating a student's complete solution.

Problem: {problem_title}
Statement: {problem_statement}
Given: {problem_given}
Find: {problem_find}

Official Answer: {official_answer}

Student's Solution:
{solution_text}

Evaluate the student's solution:
1. Is the final answer correct? (yes/no)
2. Is the reasoning/methodology correct?
3. What did they do well?
4. What mistakes did they make (if any)?

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "correct": true/false,
  "score": 0-100,
  "feedback": "Brief encouraging feedback (2-3 sentences)",
  "specific_mistakes": ["mistake 1", "mistake 2"] or []
}}

Be encouraging even if wrong. Focus on what they can improve."""
    
    response_text = call_anthropic_api(eval_prompt, max_tokens=1024)
    
    response_text = response_text.strip()
    
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    response_text = response_text.strip()
    
    json_match = re.search(r'\{[^{}]*"correct"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as je:
        print(f"[ERROR] JSON parse failed: {je}")
        print(f"[ERROR] Response was: {response_text[:200]}")
        parsed = {
            "correct": False,
            "score": 30,
            "feedback": "Your solution was recorded. Consider reviewing the key concepts and trying again.",
            "specific_mistakes": []
        }
    
    return parsed

def resize_image_if_needed(base64_data, max_size=1500000):
    """
    Resize image if base64 data is too large.
    Anthropic API has limits on image size.
    """
    try:
        if len(base64_data) <= max_size:
            return base64_data
        
        print(f"[IMAGE] Image too large ({len(base64_data)} chars), resizing...")
        
        import base64
        from PIL import Image
        from io import BytesIO
        
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_bytes))
        
        original_size = image.size
        new_size = (int(original_size[0] * 0.5), int(original_size[1] * 0.5))
        
        print(f"[IMAGE] Resizing from {original_size} to {new_size}")
        
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        resized_image.save(buffered, format="JPEG", quality=85)
        resized_bytes = buffered.getvalue()
        resized_base64 = base64.b64encode(resized_bytes).decode('utf-8')
        
        print(f"[IMAGE] Resized to {len(resized_base64)} chars")
        
        if len(resized_base64) > max_size:
            buffered = BytesIO()
            resized_image.save(buffered, format="JPEG", quality=60)
            resized_bytes = buffered.getvalue()
            resized_base64 = base64.b64encode(resized_bytes).decode('utf-8')
            print(f"[IMAGE] Further compressed to {len(resized_base64)} chars")
        
        return resized_base64
        
    except Exception as e:
        print(f"[IMAGE] Resize failed: {e}, using original")
        return base64_data

def evaluate_solution_with_image(problem_title, problem_statement, problem_given, problem_find, solution_text, solution_image, official_answer):
    """
    NEW: Evaluate a solution that includes an image using Claude's Vision API
    FIXED: Detect media type BEFORE splitting base64 + Auto-resize large images
    """
    try:
        media_type = "image/jpeg"
        if 'image/png' in solution_image:
            media_type = "image/png"
        elif 'image/jpeg' in solution_image or 'image/jpg' in solution_image:
            media_type = "image/jpeg"
        elif 'image/webp' in solution_image:
            media_type = "image/webp"
        elif 'image/gif' in solution_image:
            media_type = "image/gif"
        
        print(f"[IMAGE] Detected media type: {media_type}")
        
        if 'base64,' in solution_image:
            image_data = solution_image.split('base64,')[1]
            print(f"[IMAGE] Extracted base64 data, length: {len(image_data)}")
        else:
            image_data = solution_image
            print(f"[IMAGE] Using raw data, length: {len(image_data)}")
        
        image_data = resize_image_if_needed(image_data)
        
        eval_prompt = f"""You are an IB Physics HL examiner evaluating a student's solution.

Problem: {problem_title}
Statement: {problem_statement}
Given: {problem_given}
Find: {problem_find}

Official Answer: {official_answer}

The student has provided an image of their written solution.
{"Additional text: " + solution_text if solution_text else "No additional text provided."}

Please analyze the image carefully and evaluate the solution:
1. Is the final answer correct? (yes/no)
2. Is the reasoning/methodology correct?
3. Can you read the student's work clearly?
4. What did they do well?
5. What mistakes did they make (if any)?

Respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "correct": true/false,
  "score": 0-100,
  "feedback": "Brief encouraging feedback about their work (2-3 sentences)",
  "specific_mistakes": ["mistake 1", "mistake 2"] or []
}}

Be encouraging even if wrong. Focus on what they can improve."""
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2000,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": eval_prompt
                    }
                ]
            }]
        }
        
        print(f"[IMAGE] Calling Anthropic Vision API...")
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        
        print(f"[IMAGE] Status: {resp.status_code}")
        
        if resp.status_code != 200:
            error_data = resp.json() if resp.text else {}
            error_msg = error_data.get("error", {}).get("message", resp.text)
            raise Exception(f"Vision API error: {error_msg}")
        
        data = resp.json()
        content = data.get("content", [])
        
        if content:
            response_text = content[0].get("text", "")
        else:
            response_text = ""
        
        print(f"[IMAGE] Response: {response_text[:200]}")
        
        response_text = response_text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        response_text = response_text.strip()
        
        json_match = re.search(r'\{[^{}]*"correct"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as je:
            print(f"[ERROR] JSON parse failed: {je}")
            print(f"[ERROR] Response was: {response_text[:200]}")
            parsed = {
                "correct": False,
                "score": 40,
                "feedback": "I could see your work in the image. Try to explain your reasoning more clearly in your next attempt.",
                "specific_mistakes": []
            }
        
        return parsed
        
    except Exception as e:
        print(f"[ERROR] Image evaluation failed: {e}")
        return {
            "correct": False,
            "score": 0,
            "feedback": f"Error evaluating image: {str(e)[:100]}. Please try uploading a clearer image or writing your solution as text.",
            "specific_mistakes": []
        }

@app.post("/chat")
def chat():
    """
    Handle student interactions:
    - mode='tutor': Evaluate student's step answer
    - mode='hint': Provide a hint
    - mode='student': Free chat about the problem
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    step_id = str(data.get("step_id") or "1")
    student_answer = (data.get("student_answer") or "").strip()
    mode = data.get("mode", "tutor")
    message_text = data.get("message", "").strip()
    
    print(f"[CHAT] mode={mode}, problem_id={problem_id}, step_id={step_id}")
    print(f"[CHAT] student_answer={student_answer[:100]}")
    print(f"[CHAT] message={message_text[:100]}")
    
    if not problem_id:
        return jsonify({"error": "missing problem_id"}), 400
    
    try:
        problem = fetch_page(problem_id)
    except Exception as e:
        return jsonify({"error": f"Cannot read problem: {e}"}), 500
    
    steps_list = []
    try:
        blocks = fetch_page_blocks(problem_id)
        steps_list = parse_steps_from_blocks(blocks)
    except Exception as e:
        print(f"[CHAT] Could not read blocks: {e}")
    
    if not steps_list:
        steps_text = problem.get("step_by_step", "")
        steps_list = parse_steps(steps_text) if steps_text else []

    if not steps_list:
        print(f"[CHAT] No valid steps found, generating with LLM...")
        steps_list = generate_steps_with_llm(problem)

    current_step = None
    for s in steps_list:
        if s["id"] == step_id:
            current_step = s
            break
    
    if not current_step and steps_list:
        current_step = steps_list[0]
        step_id = current_step["id"]
    
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "ok": True,
            "feedback": "AI Coach is not configured. Your answer was recorded.",
            "next_step": str(int(step_id) + 1) if step_id.isdigit() else step_id,
            "message": "Continue to next step.",
            "reply": "AI Coach is not configured."
        })
    
    if mode == "student" and message_text:
        prompt = f"""You are an IB Physics tutor helping with this problem:

Problem: {problem.get('title', '')}
Statement: {problem.get('statement', '')}

Student asks: {message_text}

Provide a brief, helpful response (2-3 sentences). Guide them towards understanding without giving the full answer."""
        
        try:
            reply = call_anthropic_api(prompt, max_tokens=256)
            return jsonify({"reply": reply.strip()})
        except Exception as e:
            print(f"[ERROR] Free chat failed: {e}")
            return jsonify({"reply": f"Sorry, I encountered an error: {str(e)[:100]}"})
    
    if mode == "hint":
        prompt = f"""You are an IB Physics tutor. Provide a helpful hint for this step.

Problem: {problem.get('title', '')}
Current Step: {current_step.get('description', '')}

Provide ONE specific hint (1-2 sentences) that guides the student without giving the answer directly."""
        
        try:
            hint = call_anthropic_api(prompt, max_tokens=256)
            return jsonify({"hint": hint.strip()})
        except Exception as e:
            print(f"[ERROR] Hint failed: {e}")
            return jsonify({"hint": f"Try to think about the physics concepts involved in: {current_step.get('description', '')}"})
    
    if not student_answer:
        return jsonify({
            "ok": False,
            "feedback": "Please provide an answer to evaluate.",
            "next_step": step_id,
            "message": "Write your reasoning above."
        })
    
    eval_prompt = f"""You are an IB Physics tutor evaluating a student's step.

Problem: {problem.get('title', '')}
Step: {current_step.get('description', '')}
Student's Answer: {student_answer}

Evaluate if the student's reasoning is correct for this step.

Respond with ONLY valid JSON (no markdown, no explanations):
{{
  "ok": true/false,
  "feedback": "brief feedback message",
  "ready_to_advance": true/false,
  "next_hint": "hint if needed, or empty string"
}}

Rules:
- ok: true if answer shows understanding of the step
- feedback: 1-2 sentences about their answer
- ready_to_advance: true if they can move to next step
- next_hint: suggest what to include if incomplete"""
    
    try:
        response_text = call_anthropic_api(eval_prompt, max_tokens=512)
        
        response_text = response_text.strip()
        
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        response_text = response_text.strip()
        
        json_match = re.search(r'\{[^{}]*"ok"[^{}]*"feedback"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as je:
            print(f"[ERROR] JSON parse failed: {je}")
            print(f"[ERROR] Response was: {response_text[:200]}")
            parsed = {
                "ok": len(student_answer) > 15,
                "feedback": "Your answer was recorded. Try to be more specific with physics concepts.",
                "ready_to_advance": len(student_answer) > 15,
                "next_hint": "Explain which formulas or principles apply."
            }
        
        can_advance = bool(parsed.get("ready_to_advance") or parsed.get("ok"))
        next_step = str(int(step_id) + 1) if (can_advance and step_id.isdigit()) else step_id
        message = parsed.get("next_hint", "") if not can_advance else "Great! Continue to the next step."
        
        return jsonify({
            "ok": can_advance,
            "feedback": parsed.get("feedback", ""),
            "next_step": next_step,
            "message": message
        })
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return jsonify({
            "ok": False,
            "feedback": f"Error: {str(e)[:100]}",
            "next_step": step_id,
            "message": "Please try again."
        }), 500

# ========================================
# *** NEW: MULTIUSER AUTHENTICATION ENDPOINTS ***
# ========================================

@app.post("/api/register")
def register():
    """
    Register a new user.
    Creates user in Notion Users database.
    """
    if not AUTH_AVAILABLE:
        return jsonify({"error": "Authentication not available (install bcrypt and PyJWT)"}), 500
    
    if not USERS_DB_ID:
        return jsonify({"error": "USERS_DB_ID not configured"}), 500
    
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    password = data.get("password", "")
    
    print(f"[REGISTER] Attempting to register: {email}")
    
    # Validate inputs
    if not email or not name or not password:
        return jsonify({"error": "Email, name, and password are required"}), 400
    
    if not email.endswith("@SEP.com"):
        return jsonify({"error": "Must use @SEP.com email"}), 400
    
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    # Check if user already exists
    try:
        existing_users = query_database(
            USERS_DB_ID,
            filter_obj={
                "property": "Email",
                "email": {"equals": email}
            }
        )
        
        if existing_users:
            print(f"[REGISTER] User already exists: {email}")
            return jsonify({"error": "Email already registered"}), 400
    except Exception as e:
        print(f"[ERROR] Failed to check existing user: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    
    # Hash password
    try:
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to hash password: {e}")
        return jsonify({"error": "Failed to secure password"}), 500
    
    # Create user in Notion
    try:
        properties = {
            "Name": {
                "title": [{"text": {"content": name}}]
            },
            "Email": {
                "email": email
            },
            "password_hash": {
                "rich_text": [{"text": {"content": password_hash}}]
            },
            "created_at": {
                "date": {"start": datetime.utcnow().isoformat()}
            },
            "status": {
                "select": {"name": "active"}
            }
        }
        
        create_page_in_database(USERS_DB_ID, properties)
        
        print(f"[REGISTER] User created successfully: {email}")
        
        return jsonify({
            "success": True,
            "message": "Account created successfully! Please login."
        }), 201
        
    except Exception as e:
        print(f"[ERROR] Failed to create user: {e}")
        return jsonify({"error": f"Failed to create account: {str(e)}"}), 500

@app.post("/api/login")
def login():
    """
    Login user.
    Validates credentials and returns JWT token.
    """
    if not AUTH_AVAILABLE:
        return jsonify({"error": "Authentication not available (install bcrypt and PyJWT)"}), 500
    
    if not USERS_DB_ID:
        return jsonify({"error": "USERS_DB_ID not configured"}), 500
    
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password", "")
    
    print(f"[LOGIN] Attempting login: {email}")
    
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    
    # Find user
    try:
        users = query_database(
            USERS_DB_ID,
            filter_obj={
                "property": "Email",
                "email": {"equals": email}
            }
        )
        
        if not users:
            print(f"[LOGIN] User not found: {email}")
            return jsonify({"error": "Invalid email or password"}), 401
        
        user = users[0]
        
    except Exception as e:
        print(f"[ERROR] Failed to find user: {e}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    
    # Verify password
    try:
        stored_hash = user.get("password_hash", "")
        
        if not stored_hash:
            print(f"[ERROR] No password hash for user: {email}")
            return jsonify({"error": "Invalid account configuration"}), 500
        
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            print(f"[LOGIN] Invalid password for user: {email}")
            return jsonify({"error": "Invalid email or password"}), 401
        
    except Exception as e:
        print(f"[ERROR] Password verification failed: {e}")
        return jsonify({"error": "Authentication failed"}), 500
    
    # Update last_login
    try:
        # Note: Notion API doesn't support PATCH, so we'd need to use the pages endpoint
        # For now, we'll skip this to keep it simple
        pass
    except Exception as e:
        print(f"[WARN] Could not update last_login: {e}")
    
    # Create JWT token
    try:
        token_payload = {
            "email": email,
            "name": user.get("Name", ""),
            "exp": datetime.utcnow() + timedelta(days=7)
        }
        
        token = jwt.encode(token_payload, JWT_SECRET_KEY, algorithm='HS256')
        
        print(f"[LOGIN] Login successful: {email}")
        
        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "email": email,
                "name": user.get("Name", "")
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to create token: {e}")
        return jsonify({"error": "Failed to create session"}), 500

@app.get("/api/verify-session")
@require_auth
def verify_session():
    """
    Verify if JWT token is valid.
    Requires authentication (automatically validated by decorator).
    """
    user = request.user  # Set by @require_auth decorator
    
    return jsonify({
        "valid": True,
        "user": {
            "email": user.get("email"),
            "name": user.get("name")
        }
    }), 200

@app.post("/api/track-activity")
@require_auth
def track_activity():
    """
    Track user activity (opening problems, submitting solutions, etc).
    Creates entry in Activity Log database.
    """
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500
    
    user = request.user
    data = request.get_json(force=True, silent=True) or {}
    
    problem_id = data.get("problem_id", "")
    problem_name = data.get("problem_name", "")
    action = data.get("action", "opened")  # opened, started, submitted, completed
    time_spent = data.get("time_spent", 0)
    score = data.get("score", 0)
    attempt_number = data.get("attempt_number", 1)
    
    print(f"[TRACK] user={user.get('email')}, problem={problem_id}, action={action}")
    
    # Create activity log entry
    try:
        properties = {
            "problem_name": {
                "title": [{"text": {"content": problem_name or problem_id}}]
            },
            "user_email": {
                "email": user.get("email")
            },
            "problem_id": {
                "rich_text": [{"text": {"content": problem_id}}]
            },
            "action": {
                "select": {"name": action}
            },
            "timestamp": {
                "date": {"start": datetime.utcnow().isoformat()}
            },
            "time_spent_seconds": {
                "number": int(time_spent)
            },
            "score": {
                "number": int(score)
            },
            "attempt_number": {
                "number": int(attempt_number)
            }
        }
        
        create_page_in_database(ACTIVITY_DB_ID, properties)
        
        return jsonify({"success": True}), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to track activity: {e}")
        return jsonify({"error": f"Failed to track activity: {str(e)}"}), 500

@app.get("/api/user-progress")
@require_auth
def user_progress():
    """
    Get progress for current user.
    Returns statistics from Activity Log.
    """
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500
    
    user = request.user
    email = user.get("email")
    
    print(f"[PROGRESS] Getting progress for: {email}")
    
    # Get all activities for this user
    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={
                "property": "user_email",
                "email": {"equals": email}
            },
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        # Calculate statistics
        total_time = 0
        problems_attempted = set()
        problems_completed = set()
        scores = []
        
        for activity in activities:
            problem_id = activity.get("problem_id", "")
            action = activity.get("action", "")
            time_spent = activity.get("time_spent_seconds", 0) or 0
            score = activity.get("score", 0) or 0
            
            if action == "started" or action == "opened":
                problems_attempted.add(problem_id)
            
            if action == "completed":
                problems_completed.add(problem_id)
                total_time += time_spent
                if score > 0:
                    scores.append(score)
        
        # Get recent activities (last 10)
        recent = activities[:10] if len(activities) > 10 else activities
        
        return jsonify({
            "email": email,
            "name": user.get("name", ""),
            "problems_attempted": len(problems_attempted),
            "problems_completed": len(problems_completed),
            "total_time_minutes": total_time // 60,
            "average_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "recent_activities": recent
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get progress: {e}")
        return jsonify({"error": f"Failed to get progress: {str(e)}"}), 500

@app.get("/api/all-users")
def all_users():
    """
    Get all users and their progress.
    FOR ADMIN/PROFESSOR USE ONLY.
    
    NOTE: In production, this should require admin authentication.
    For MVP, we're keeping it open for simplicity.
    """
    if not USERS_DB_ID or not ACTIVITY_DB_ID:
        return jsonify({"error": "Databases not configured"}), 500
    
    print(f"[ADMIN] Getting all users")
    
    try:
        # Get all users
        users = query_database(USERS_DB_ID)
        
        # Get all activities
        all_activities = query_database(ACTIVITY_DB_ID)
        
        # Build stats per user
        user_stats = []
        
        for user in users:
            email = user.get("Email", "")
            name = user.get("Name", "")
            
            # Filter activities for this user
            user_activities = [a for a in all_activities if a.get("user_email") == email]
            
            # Calculate stats
            total_time = 0
            problems_attempted = set()
            problems_completed = set()
            scores = []
            last_active = None
            
            for activity in user_activities:
                problem_id = activity.get("problem_id", "")
                action = activity.get("action", "")
                time_spent = activity.get("time_spent_seconds", 0) or 0
                score = activity.get("score", 0) or 0
                timestamp = activity.get("timestamp")
                
                if action in ["started", "opened"]:
                    problems_attempted.add(problem_id)
                
                if action == "completed":
                    problems_completed.add(problem_id)
                    total_time += time_spent
                    if score > 0:
                        scores.append(score)
                
                if timestamp and (not last_active or timestamp > last_active):
                    last_active = timestamp
            
            user_stats.append({
                "email": email,
                "name": name,
                "last_active": last_active,
                "problems_attempted": len(problems_attempted),
                "problems_completed": len(problems_completed),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "total_time_minutes": total_time // 60
            })
        
        # Calculate overall stats
        total_students = len(users)
        active_today = sum(1 for u in user_stats if u.get("last_active") and 
                          (datetime.utcnow() - datetime.fromisoformat(u["last_active"].replace("Z", ""))).days == 0)
        problems_solved = sum(u["problems_completed"] for u in user_stats)
        avg_score = round(sum(u["avg_score"] for u in user_stats if u["avg_score"] > 0) / 
                         len([u for u in user_stats if u["avg_score"] > 0]), 1) if user_stats else 0
        
        return jsonify({
            "total_students": total_students,
            "active_today": active_today,
            "problems_solved": problems_solved,
            "avg_score": avg_score,
            "students": user_stats
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get all users: {e}")
        return jsonify({"error": f"Failed to get users: {str(e)}"}), 500

# ========================================
# *** END OF MULTIUSER SYSTEM ***
# ========================================

# ==============================
# Gunicorn entry
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
