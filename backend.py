# backend.py - AI Coach Physics Backend v3
# ==========================================
# ACTUALIZADO: 17 Feb 2026
# v3 CHANGES (Quick Wins):
# - QUICK WIN 1: Marks + expected time calculation
# - QUICK WIN 2: Robust evaluation prompt with official solution reference,
#   structured feedback (error_type, missing_steps, key_concept, marks_awarded)
# - QUICK WIN 3: Attempt tracking (counts previous submissions per user/problem)
# - New fields in Activity DB: attempt_number, error_type, marks_awarded, marks_total
# - New Notion field required in Problems DB: marks (Number)
# ==========================================

import os, re, json, uuid
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from functools import wraps
from datetime import datetime, timedelta, timezone
import time

try:
    import bcrypt
    import jwt
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    print("[WARNING] bcrypt or PyJWT not installed.")

# ==============================
# ENV
# ==============================
NOTION_API_KEY      = os.environ.get("NOTION_API_KEY", "")
NOTION_DB_ID        = os.environ.get("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
FRONTEND_ORIGIN     = os.environ.get("FRONTEND_ORIGIN", "https://aiclub.com.mx")
USERS_DB_ID         = os.environ.get("USERS_DB_ID", "")
ACTIVITY_DB_ID      = os.environ.get("ACTIVITY_DB_ID", "")
HOMEWORK_DB_ID      = os.environ.get("HOMEWORK_DB_ID", "")
RESOURCES_DB_ID     = os.environ.get("RESOURCES_DB_ID", "")  # Resources by Topic
JWT_SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "change-this-secret-key-in-production")

# ==============================
# Flask
# ==============================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max for image uploads
CORS(app, resources={
    r"/*": {
        "origins": ["https://aiclub.com.mx", "http://aiclub.com.mx"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# ==============================
# Simple in-memory cache (avoids hammering Notion on repeated loads)
# ==============================
_cache = {}
CACHE_TTL = 60  # seconds

def cache_get(key):
    """Get cached value if not expired."""
    entry = _cache.get(key)
    if entry and time.time() - entry["time"] < CACHE_TTL:
        return entry["data"]
    return None

def cache_set(key, data):
    """Cache data with TTL."""
    _cache[key] = {"data": data, "time": time.time()}

def cache_clear(prefix=""):
    """Clear cache entries matching prefix."""
    if not prefix:
        _cache.clear()
    else:
        keys_to_remove = [k for k in _cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del _cache[k]

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
    match = UUID_RE.search(value)
    if not match: return ""
    raw = match.group(0).replace("-", "")
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

def fetch_page(page_id: str) -> Dict[str, Any]:
    pid = sanitize_uuid(page_id)
    if not pid: raise ValueError("Invalid page ID")
    
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
        elif ptype == "email":
            out[k] = v.get("email", "")
        elif ptype == "date":
            date_obj = v.get("date")
            out[k] = date_obj.get("start", "") if date_obj else None
        elif ptype == "checkbox":
            out[k] = v.get("checkbox", False)
    
    return out

def query_database(database_id: str, filter_obj: dict = None, sorts: list = None, max_pages: int = 0) -> List[Dict[str, Any]]:
    """Query a Notion database. max_pages=0 means unlimited (fetch all)."""
    if database_id in [USERS_DB_ID, ACTIVITY_DB_ID, HOMEWORK_DB_ID]:
        db_id = database_id
    else:
        db_id = sanitize_uuid(database_id)
        if not db_id: raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/databases/{db_id}/query"
    all_pages = []
    has_more = True
    start_cursor = None
    pages_fetched = 0
    
    while has_more:
        payload = {}
        if filter_obj: payload["filter"] = filter_obj
        if sorts: payload["sorts"] = sorts
        if start_cursor: payload["start_cursor"] = start_cursor

        resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        all_pages.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")
        pages_fetched += 1
        
        # Limit pagination if max_pages is set
        if max_pages > 0 and pages_fetched >= max_pages:
            if has_more:
                print(f"[QUERY] Stopped after {pages_fetched} pages ({len(all_pages)} results). More data exists.")
            break
    
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
            elif ptype == "email":
                obj[k] = v.get("email", "")
            elif ptype == "date":
                date_obj = v.get("date")
                obj[k] = date_obj.get("start", "") if date_obj else None
            elif ptype == "checkbox":
                obj[k] = v.get("checkbox", False)
            elif ptype == "url":
                obj[k] = v.get("url", "")
        results.append(obj)
    return results

def create_page_in_database(database_id: str, properties: dict) -> dict:
    if database_id in [USERS_DB_ID, ACTIVITY_DB_ID, HOMEWORK_DB_ID]:
        db_id = database_id
    else:
        db_id = sanitize_uuid(database_id)
        if not db_id: raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/pages"
    payload = {"parent": {"database_id": db_id}, "properties": properties}
    resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def update_page_properties(page_id: str, properties: dict) -> dict:
    pid = sanitize_uuid(page_id)
    if not pid: raise ValueError("Invalid page ID")
    
    url = f"{NOTION_BASE}/pages/{pid}"
    payload = {"properties": properties}
    resp = requests.patch(url, headers=NOTION_HEADERS, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def parse_steps(text: str) -> List[Dict[str, str]]:
    if not text or not text.strip(): return []
    lines = text.strip().split("\n")
    steps = []
    for i, ln in enumerate(lines, start=1):
        ln = ln.strip()
        if not ln: continue
        clean_ln = re.sub(r'^(\d+[\)\.:]?\s*|Step\s+\d+[\)\.:]?\s*)', '', ln, flags=re.IGNORECASE).strip()
        if clean_ln and len(clean_ln) > 2:
            steps.append({"id": str(i), "description": clean_ln, "rubric": ""})
    return steps

def fetch_page_blocks(page_id: str) -> List[Dict[str, Any]]:
    pid = sanitize_uuid(page_id)
    if not pid: raise ValueError("Invalid page ID")
    url = f"{NOTION_BASE}/blocks/{pid}/children"
    all_blocks = []
    has_more = True
    start_cursor = None
    while has_more:
        params = {}
        if start_cursor: params["start_cursor"] = start_cursor
        resp = requests.get(url, headers=NOTION_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        all_blocks.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")
    return all_blocks

def extract_text_from_rich_text(rich_text_array):
    if not rich_text_array: return ""
    return "".join([item.get("plain_text", "") for item in rich_text_array])

def parse_steps_from_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    steps = []
    step_counter = 1
    for block in blocks:
        block_type = block.get("type")
        if block_type == "toggle":
            toggle_data = block.get("toggle", {})
            title_text = extract_text_from_rich_text(toggle_data.get("rich_text", []))
            if title_text and (title_text.lower().startswith("step") or any(title_text.startswith(f"{n}.") for n in range(1, 20))):
                clean_title = re.sub(r'^(Step\s+\d+:\s*)', '', title_text, flags=re.IGNORECASE).strip()
                block_id = block.get("id", "")
                content_lines = []
                if block.get("has_children", False) and block_id:
                    try:
                        children_url = f"{NOTION_BASE}/blocks/{block_id}/children"
                        children_resp = requests.get(children_url, headers=NOTION_HEADERS, timeout=10)
                        if children_resp.status_code == 200:
                            for child_block in children_resp.json().get("results", []):
                                ctype = child_block.get("type")
                                if ctype in ["paragraph", "bulleted_list_item", "numbered_list_item"]:
                                    txt = extract_text_from_rich_text(child_block.get(ctype, {}).get("rich_text", []))
                                    if txt.strip(): content_lines.append(("- " if ctype != "paragraph" else "") + txt)
                    except: pass
                full_description = clean_title + ("\n\n" + "\n".join(content_lines) if content_lines else "")
                steps.append({"id": str(step_counter), "description": full_description, "rubric": ""})
                step_counter += 1
    return steps

def call_anthropic_api(prompt: str, max_tokens: int = 1024) -> str:
    if not ANTHROPIC_API_KEY: raise ValueError("ANTHROPIC_API_KEY not configured")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": "claude-sonnet-4-20250514", "max_tokens": max_tokens, "temperature": 0, "system": "You are a helpful IB Physics tutor. Always respond with valid JSON only.", "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("content", [])[0].get("text", "")

def call_anthropic_vision(prompt: str, image_base64: str, media_type: str = "image/jpeg", max_tokens: int = 1500) -> str:
    """Call Claude API with an image (vision) for evaluating handwritten solutions."""
    if not ANTHROPIC_API_KEY: raise ValueError("ANTHROPIC_API_KEY not configured")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    
    # Clean base64 data - remove any whitespace/newlines and data URL prefix if present
    clean_b64 = image_base64.strip()
    if clean_b64.startswith('data:'):
        clean_b64 = clean_b64.split(',', 1)[1] if ',' in clean_b64 else clean_b64
    clean_b64 = clean_b64.replace('\n', '').replace('\r', '').replace(' ', '')
    
    # Validate media type
    valid_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    if media_type not in valid_types:
        media_type = 'image/jpeg'
    
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": clean_b64}},
        {"type": "text", "text": prompt}
    ]
    
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": "You are a helpful IB Physics tutor. You can read handwritten solutions from photos. Always respond with valid JSON only.",
        "messages": [{"role": "user", "content": content}]
    }
    
    print(f"[VISION] Sending image ({media_type}, ~{len(clean_b64)//1024}KB base64) to Claude API...")
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    
    if not resp.ok:
        error_body = resp.text
        print(f"[VISION ERROR] Status: {resp.status_code}, Body: {error_body[:500]}")
        resp.raise_for_status()
    
    return resp.json().get("content", [])[0].get("text", "")

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    title = problem.get("title") or problem.get("name", "")
    prompt = f"Create a clear 4-6 step solution plan for this problem: {title}\nStatement: {problem.get('statement', '')}\nFormat: Numbered lines only."
    try:
        text = call_anthropic_api(prompt, max_tokens=512)
        steps = parse_steps(text)
        return steps if steps else [{"id": "1", "description": "Analyze the problem statement.", "rubric": ""}]
    except:
        return [{"id": "1", "description": "Identify given data.", "rubric": ""}]

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not AUTH_AVAILABLE: return jsonify({"error": "Auth dependency missing"}), 500
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token: return jsonify({"error": "No token"}), 401
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            request.user = payload
            return f(*args, **kwargs)
        except: return jsonify({"error": "Invalid/Expired token"}), 401
    return decorated_function

# ==============================
# Base Routes
# ==============================
@app.get("/")
def index(): return jsonify({"status": "ok", "service": "ai-coach-backend", "version": "2.0"})

@app.get("/problems")
def list_problems():
    try:
        items = query_database(NOTION_DB_ID)
        return jsonify({"problems": items})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.get("/problem/<problem_id>")
def get_problem(problem_id: str):
    try: return jsonify({"problem": fetch_page(problem_id)})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.get("/steps/<problem_id>")
def get_steps(problem_id: str):
    try:
        page = fetch_page(problem_id)
        blocks = fetch_page_blocks(problem_id)
        steps = parse_steps_from_blocks(blocks)
        if not steps: steps = generate_steps_with_llm(page)
        return jsonify({"steps": steps})
    except Exception as e: return jsonify({"error": str(e)}), 500

# ========================================
# HELPER: Count previous attempts for a problem by user
# ========================================
def count_previous_attempts(user_email: str, problem_reference: str) -> int:
    """Count how many times a user has submitted/completed a specific problem."""
    if not ACTIVITY_DB_ID or not user_email or not problem_reference:
        return 0
    try:
        filter_obj = {
            "and": [
                {"property": "user_email", "email": {"equals": user_email}},
                {"property": "problem_reference", "rich_text": {"equals": problem_reference}},
                {"property": "action", "select": {"equals": "submitted"}}
            ]
        }
        results = query_database(ACTIVITY_DB_ID, filter_obj)
        # Also count "completed" actions
        filter_obj2 = {
            "and": [
                {"property": "user_email", "email": {"equals": user_email}},
                {"property": "problem_reference", "rich_text": {"equals": problem_reference}},
                {"property": "action", "select": {"equals": "completed"}}
            ]
        }
        results2 = query_database(ACTIVITY_DB_ID, filter_obj2)
        return len(results) + len(results2)
    except Exception as e:
        print(f"[WARNING] Could not count attempts: {e}")
        return 0

# ========================================
# HELPER: Calculate expected time from marks
# ========================================
def calculate_expected_time(marks, paper_type="P2"):
    """
    IB Physics timing rules:
    - Paper 1: ~1.5 min per mark (but MCQ, not applicable here)
    - Paper 2: ~1.5 min per mark (95 marks in 135 min)
    - Paper 3: ~2.0 min per mark (45 marks in 75 min + 10 min reading)
    """
    if not marks or marks <= 0:
        return None
    if paper_type == "P3":
        return int(marks * 2.0 * 60)  # seconds
    else:
        return int(marks * 1.5 * 60)  # seconds

@app.post("/submit-solution")
def submit_solution():
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    solution_text = data.get("solution_text", "")
    time_spent = data.get("time_spent_seconds", 0)
    user_email = data.get("user_email", "")
    image_base64 = data.get("image_base64", "")
    image_media_type = data.get("image_media_type", "image/jpeg")
    
    has_image = bool(image_base64)
    
    if not problem_id or (not solution_text and not has_image):
        return jsonify({"error": "Missing problem_id or solution"}), 400
    
    print(f"[SUBMIT] Evaluating solution for problem: {problem_id} | Image: {has_image}" + (f" | Image size: ~{len(image_base64)//1024}KB" if has_image else ""))
    
    try:
        problem = fetch_page(problem_id)
        print(f"[SUBMIT] Problem loaded: name='{problem.get('name','')}', ref='{problem.get('problem_reference','')}', statement='{(problem.get('problem_statement','') or '')[:80]}'...")
        
        # --- QUICK WIN 3: Count previous attempts ---
        problem_ref = problem.get("problem_reference", "")
        attempt_number = count_previous_attempts(user_email, problem_ref) + 1
        
        # --- QUICK WIN 1: Get marks and expected time ---
        marks = problem.get("marks", None)
        # Infer paper type from reference (e.g., 2018-Z1-P2-Q1-A → P2)
        paper_type = "P2"
        if problem_ref:
            parts = problem_ref.split("-")
            if len(parts) >= 3:
                paper_type = parts[2]  # "P2" or "P3"
        expected_time_secs = calculate_expected_time(marks, paper_type)
        
        # --- QUICK WIN 2: Robust evaluation prompt ---
        # Get official solution if available
        # Standard: full_solution = detailed step-by-step (for evaluation reference)
        #           final_answer = short answer with units (for quick display)
        #           Fallback chain for backward compatibility with existing problems
        official_solution = problem.get("full_solution", "") or problem.get("final_answer", "") or problem.get("step_by_step", "")
        
        marks_context = ""
        if marks:
            marks_context = f"""
This problem is worth [{marks} marks] in the IB exam.
Award marks according to IB marking standards:
- Each mark corresponds to a specific step, concept, or correct value
- Partial credit is expected: award marks for correct intermediate steps even if final answer is wrong
- ECF (Error Carried Forward): If a previous step has an error but subsequent steps are correctly applied, award those marks
"""
        
        solution_reference = ""
        if official_solution:
            solution_reference = f"""
OFFICIAL SOLUTION (for reference - do NOT share this with the student):
{official_solution}

Compare the student's work against this reference solution.
"""
        
        attempt_context = ""
        if attempt_number > 1:
            attempt_context = f"\nThis is the student's attempt #{attempt_number} at this problem. Be encouraging about improvement while still being precise about errors."
        
        prompt = f"""You are an expert IB Physics HL examiner evaluating a student's solution.
{attempt_context}

PROBLEM DETAILS:
Name: {problem.get('name', '')}
Statement: {problem.get('problem_statement', '')}
Given values: {problem.get('given_values', '')}
What to find: {problem.get('find', '')}
Topic: {problem.get('topic', '')}
{marks_context}
{solution_reference}

STUDENT'S SOLUTION:
{solution_text}
{"" if not has_image else "The student has also attached a PHOTO of their handwritten solution. Read and evaluate the handwritten work in the image carefully. The image contains their actual work — evaluate what you see in the photo."}

TIME SPENT: {time_spent // 60}:{time_spent % 60:02d}

EVALUATE and respond with ONLY a valid JSON object (no markdown, no backticks):
{{
  "score": <number 0-100>,
  "correct": <true if score >= 70, false otherwise>,
  "marks_awarded": <number of IB marks earned out of {marks or 'total'}>,
  "marks_total": {marks or 'null'},
  "feedback": "<detailed feedback: start with what the student did well, then explain specific errors>",
  "error_type": "<classify the main error: 'conceptual' | 'calculation' | 'units' | 'method' | 'incomplete' | 'none'>",
  "missing_steps": "<list any key steps or concepts the student missed>",
  "key_concept": "<the most important physics concept tested in this problem>",
  "time_taken": "{time_spent // 60}:{time_spent % 60:02d}"
}}
"""
        
        if not ANTHROPIC_API_KEY:
            return jsonify({
                "score": 50, "correct": False,
                "feedback": "Solution submitted but cannot evaluate (API key missing)",
                "time_taken": f"{time_spent // 60}:{time_spent % 60:02d}",
                "attempt_number": attempt_number,
                "expected_time_seconds": expected_time_secs
            })
        
        # Use vision API if image is attached, otherwise text-only
        if has_image:
            response_text = call_anthropic_vision(prompt, image_base64, image_media_type, max_tokens=1500)
        else:
            response_text = call_anthropic_api(prompt, max_tokens=1500)
        
        clean_response = response_text.strip()
        if clean_response.startswith('```'):
            clean_response = '\n'.join(clean_response.split('\n')[1:-1])
        
        result = json.loads(clean_response)
        result.setdefault('score', 50)
        result.setdefault('correct', result.get('score', 0) >= 70)
        result.setdefault('feedback', 'Solution evaluated.')
        result.setdefault('time_taken', f"{time_spent // 60}:{time_spent % 60:02d}")
        result.setdefault('error_type', 'none')
        result.setdefault('missing_steps', '')
        result.setdefault('key_concept', '')
        result.setdefault('marks_awarded', None)
        result.setdefault('marks_total', marks)
        
        # --- Inject attempt and timing metadata ---
        result['attempt_number'] = attempt_number
        result['expected_time_seconds'] = expected_time_secs
        if expected_time_secs and time_spent:
            ratio = time_spent / expected_time_secs
            if ratio <= 1.0:
                result['time_assessment'] = 'on_pace'
            elif ratio <= 1.5:
                result['time_assessment'] = 'slightly_slow'
            else:
                result['time_assessment'] = 'too_slow'
        
        return jsonify(result), 200
        
    except Exception as e:
        error_detail = str(e)
        # Try to get actual API error message
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_json = e.response.json()
                error_detail = err_json.get('error', {}).get('message', str(e))
                print(f"[ERROR] Anthropic API error: {err_json}")
            except:
                error_detail = e.response.text[:300] if e.response.text else str(e)
        print(f"[ERROR] Submit solution failed: {error_detail}")
        return jsonify({
            "score": 50, "correct": False,
            "feedback": f"Error evaluating solution: {error_detail}",
            "time_taken": f"{time_spent // 60}:{time_spent % 60:02d}",
            "attempt_number": 1, "expected_time_seconds": None
        }), 200

@app.post("/chat")
def chat():
    return jsonify({"reply": "I am your AI Physics tutor."})

# ========================================
# MULTIUSER AUTHENTICATION
# ========================================

@app.post("/api/register")
def register():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get("email", "").strip().lower()
    name = data.get("name", "").strip()
    password = data.get("password", "")
    group = data.get("group", "").strip()  # NEW: group field
    
    if not email or not name or not password: 
        return jsonify({"error": "Missing fields"}), 400
    if not email.endswith("@asf.edu.mx"): 
        return jsonify({"error": "Use ASF email"}), 400
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        properties = {
            "Name": {"title": [{"text": {"content": name}}]},
            "Email": {"email": email},
            "password_hash": {"rich_text": [{"text": {"content": password_hash}}]},
            "created_at": {"date": {"start": datetime.utcnow().isoformat()}}
        }
        
        # Add group if provided
        if group:
            properties["group"] = {"select": {"name": group}}
        
        create_page_in_database(USERS_DB_ID, properties)
        return jsonify({"success": True}), 201
    except Exception as e: 
        return jsonify({"error": str(e)}), 500

@app.post("/api/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get("email", "").lower()
    password = data.get("password", "")
    users = query_database(USERS_DB_ID, {"property": "Email", "email": {"equals": email}})
    if not users or not bcrypt.checkpw(password.encode('utf-8'), users[0]["password_hash"].encode('utf-8')):
        return jsonify({"error": "Invalid credentials"}), 401
    token = jwt.encode({"email": email, "name": users[0]["Name"], "exp": datetime.utcnow() + timedelta(days=7)}, JWT_SECRET_KEY)
    return jsonify({"success": True, "token": token, "user": {"email": email, "name": users[0]["Name"]}})

@app.get("/api/verify-session")
@require_auth
def verify_session():
    return jsonify({"valid": True, "user": {"email": request.user["email"], "name": request.user["name"]}})

# ========================================
# TRACK ACTIVITY - UPDATED WITH solution_text
# ========================================

@app.post("/api/track-activity")
@require_auth
def track_activity():
    """
    Track student activity with full solution text.
    UPDATED: Now saves solution_text for professor review.
    """
    data = request.get_json(force=True, silent=True) or {}
    
    properties = {
        "problem_name": {"title": [{"text": {"content": data.get("problem_name", "")}}]},
        "user_email": {"email": request.user["email"]},
        "action": {"select": {"name": data.get("action", "opened")}},
        "timestamp": {"date": {"start": datetime.utcnow().isoformat()}}
    }
    
    if data.get("problem_reference"):
        properties["problem_reference"] = {"rich_text": [{"text": {"content": data.get("problem_reference", "")}}]}
    
    if data.get("score") is not None:
        properties["score"] = {"number": data.get("score", 0)}
    
    if data.get("time_spent_seconds") is not None:
        properties["time_spent_seconds"] = {"number": data.get("time_spent_seconds", 0)}
    
    # ========================================
    # NEW: Save solution_text for professor review
    # ========================================
    if data.get("solution_text"):
        # Notion rich_text has a 2000 character limit per block
        solution = data.get("solution_text", "")[:2000]
        properties["solution_text"] = {"rich_text": [{"text": {"content": solution}}]}
    
    # ========================================
    # QUICK WIN 3: Save attempt_number for tracking
    # ========================================
    if data.get("attempt_number") is not None:
        properties["attempt_number"] = {"number": data.get("attempt_number", 1)}
    
    # ========================================
    # QUICK WIN 2: Save error_type and marks for analytics
    # ========================================
    if data.get("error_type"):
        properties["error_type"] = {"rich_text": [{"text": {"content": data.get("error_type", "")[:200]}}]}
    
    if data.get("marks_awarded") is not None:
        properties["marks_awarded"] = {"number": data.get("marks_awarded", 0)}
    
    if data.get("marks_total") is not None:
        properties["marks_total"] = {"number": data.get("marks_total", 0)}
    
    try:
        create_page_in_database(ACTIVITY_DB_ID, properties)
        cache_clear()  # Invalidate all caches when new data comes in
        return jsonify({"success": True})
    except Exception as e: 
        print(f"[ERROR] Track activity failed: {e}")
        return jsonify({"error": str(e)}), 500

# ========================================
# USER STATS & PROGRESS
# ========================================

@app.get("/api/user-stats")
@require_auth
def user_stats():
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500
    
    user = request.user
    email = user.get("email")
    
    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        total_time = 0
        problems_completed = set()
        scores = []
        
        for activity in activities:
            problem_ref = activity.get("problem_reference") or activity.get("problem_name", "")
            action = activity.get("action", "")
            time_spent = activity.get("time_spent_seconds", 0) or 0
            score = activity.get("score", 0) or 0
            
            if action == "completed":
                problems_completed.add(problem_ref)
                total_time += time_spent
                if score > 0:
                    scores.append(score)
        
        # Build problem reference → topic lookup from Problems DB
        problem_topic_map = {}
        if NOTION_DB_ID:
            try:
                all_problems = query_database(NOTION_DB_ID, max_pages=5)
                for p in all_problems:
                    pref = p.get("reference") or p.get("Reference") or ""
                    pname = p.get("Name") or p.get("name") or ""
                    kc = p.get("key_concepts") or p.get("Key Concepts") or ""
                    if isinstance(kc, list):
                        kc = kc[0] if kc else ""
                    topic = kc.split(";")[0].strip() if kc else ""
                    if pref and topic:
                        problem_topic_map[pref] = topic
                    if pname and topic:
                        problem_topic_map[pname] = topic
            except Exception as e:
                print(f"[WARN] Could not load problems for topic map: {e}")
        
        recent_activity = []
        all_completed = []
        for activity in activities:
            if activity.get("action") in ["completed", "submitted"]:
                display_name = activity.get("problem_reference") or activity.get("problem_name", "Unknown")
                # Use key_concept from activity, fallback to problems DB lookup
                key_concept = (activity.get("key_concept") or "").strip()
                if not key_concept:
                    key_concept = problem_topic_map.get(display_name, "")
                entry = {
                    "problem_name": display_name,
                    "score": activity.get("score", 0) or 0,
                    "timestamp": activity.get("timestamp", ""),
                    "action": activity.get("action", ""),
                    "professor_feedback": activity.get("professor_feedback", ""),
                    "error_type": activity.get("error_type", ""),
                    "key_concept": key_concept
                }
                all_completed.append(entry)
                if len(recent_activity) < 10:
                    recent_activity.append(entry)
        
        return jsonify({
            "solved_count": len(problems_completed),
            "average_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "total_time_seconds": total_time,
            "recent_activity": recent_activity,
            "all_completed": all_completed
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get user stats: {e}")
        return jsonify({"error": f"Failed to get stats: {str(e)}"}), 500

@app.get("/api/user-progress")
@require_auth
def user_progress():
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500
    
    user = request.user
    email = user.get("email")
    
    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        total_time = 0
        problems_attempted = set()
        problems_completed = set()
        scores = []
        
        for activity in activities:
            problem_id = activity.get("problem_id", "")
            action = activity.get("action", "")
            time_spent = activity.get("time_spent_seconds", 0) or 0
            score = activity.get("score", 0) or 0
            
            if action in ["started", "opened"]:
                problems_attempted.add(problem_id)
            
            if action == "completed":
                problems_completed.add(problem_id)
                total_time += time_spent
                if score > 0:
                    scores.append(score)
        
        recent = activities[:10] if len(activities) > 10 else activities
        
        return jsonify({
            "email": email,
            "name": user.get("name", "Student"),
            "problems_attempted": len(problems_attempted),
            "problems_completed": len(problems_completed),
            "total_time_minutes": total_time // 60,
            "average_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "recent_activities": recent
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get progress: {e}")
        return jsonify({"error": f"Failed to get progress: {str(e)}"}), 500

# ========================================
# ALL USERS (PROFESSOR VIEW) - WITH SOLUTION TEXT
# ========================================

@app.get("/api/all-users")
def all_users():
    """
    Get all users with their progress and solution texts.
    OPTIMIZED: Cached for 60s, sorted by timestamp desc, max 1000 activities.
    """
    if not USERS_DB_ID or not ACTIVITY_DB_ID:
        return jsonify({"error": "Databases not configured"}), 500
    
    # Check cache first
    cached = cache_get("all-users")
    if cached:
        print("[CACHE HIT] all-users")
        return jsonify(cached), 200
    
    try:
        users = query_database(USERS_DB_ID)
        all_activities = query_database(
            ACTIVITY_DB_ID,
            sorts=[{"property": "timestamp", "direction": "descending"}],
            max_pages=10  # max ~1000 activities, recent first
        )
        
        user_stats_list = []
        
        for user in users:
            email = user.get("Email", "")
            name = user.get("Name", "")
            group = user.get("group") or user.get("Group") or ""
            
            if not email: continue
            
            user_activities = [a for a in all_activities if a.get("user_email") == email]
            user_activities.sort(key=lambda x: x.get("timestamp", "") or "", reverse=True)
            
            total_time = 0
            problems_attempted = set()
            problems_completed = set()
            scores = []
            last_active = None
            
            for activity in user_activities:
                problem_name = activity.get("problem_name", "")
                action = activity.get("action", "")
                time_spent = activity.get("time_spent_seconds", 0) or 0
                score = activity.get("score", 0) or 0
                timestamp = activity.get("timestamp")
                
                if action in ["started", "opened"]:
                    problems_attempted.add(problem_name)
                
                if action == "completed":
                    problems_completed.add(problem_name)
                    total_time += time_spent
                    if score > 0:
                        scores.append(score)
                
                if timestamp and (not last_active or timestamp > last_active):
                    last_active = timestamp
            
            # Include solution_text in recent activities for professor review
            recent_activities = []
            error_counts = {}
            for activity in user_activities[:10]:
                display_name = activity.get("problem_reference") or activity.get("problem_name", "Unknown")
                recent_activities.append({
                    "id": activity.get("id", ""),
                    "problem_name": display_name,
                    "action": activity.get("action", "opened"),
                    "score": activity.get("score", 0) or 0,
                    "time_spent_seconds": activity.get("time_spent_seconds", 0) or 0,
                    "timestamp": activity.get("timestamp", ""),
                    "solution_text": activity.get("solution_text", ""),
                    "error_type": activity.get("error_type", ""),
                    "marks_awarded": activity.get("marks_awarded"),
                    "marks_total": activity.get("marks_total"),
                    "attempt_number": activity.get("attempt_number"),
                    "professor_feedback": activity.get("professor_feedback", "")
                })
            
            # Aggregate error patterns for this student
            for activity in user_activities:
                if activity.get("action") == "completed":
                    etype = activity.get("error_type", "").strip().lower()
                    if etype and etype != "none":
                        error_counts[etype] = error_counts.get(etype, 0) + 1
            
            user_stats_list.append({
                "email": email,
                "name": name,
                "group": group,
                "last_active": last_active,
                "problems_attempted": len(problems_attempted),
                "problems_completed": len(problems_completed),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "total_time_minutes": total_time // 60,
                "recent_activities": recent_activities,
                "error_patterns": error_counts
            })
        
        total_students = len(user_stats_list)
        
        active_today = 0
        for u in user_stats_list:
            if u.get("last_active"):
                try:
                    last_active_str = u["last_active"].replace("Z", "+00:00")
                    last_active_dt = datetime.fromisoformat(last_active_str)
                    now_utc = datetime.now(timezone.utc)
                    diff = now_utc - last_active_dt
                    if diff.days == 0:
                        active_today += 1
                except: continue
        
        problems_solved = sum(u["problems_completed"] for u in user_stats_list)
        
        users_with_scores = [u for u in user_stats_list if u["avg_score"] > 0]
        avg_score = round(sum(u["avg_score"] for u in users_with_scores) / len(users_with_scores), 1) if users_with_scores else 0
        
        result = {
            "total_students": total_students,
            "active_today": active_today,
            "problems_solved": problems_solved,
            "avg_score": avg_score,
            "students": user_stats_list
        }
        cache_set("all-users", result)
        print(f"[ALL-USERS] Returned {total_students} students, {len(all_activities)} activities processed")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get all users: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to get users: {str(e)}"}), 500

# ============================================================================
# ERROR PATTERNS ANALYTICS
# ============================================================================

@app.get("/api/professor/error-patterns")
def error_patterns():
    """
    Aggregate error pattern analytics across all students.
    Returns: error distribution, per-student breakdown, per-problem breakdown, insights.
    """
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500
    
    try:
        # Check cache first
        cached = cache_get("error-patterns")
        if cached:
            print("[CACHE HIT] error-patterns")
            return jsonify(cached), 200
        
        all_activities = query_database(
            ACTIVITY_DB_ID,
            sorts=[{"property": "timestamp", "direction": "descending"}],
            max_pages=10  # max ~1000 activities
        )
        
        # Global error distribution
        global_errors = {}
        # Per-student errors
        student_errors = {}
        # Per-problem errors  
        problem_errors = {}
        # Score by error type
        error_scores = {}
        # Timeline (last 30 activities with errors)
        error_timeline = []
        
        for activity in all_activities:
            if activity.get("action") != "completed":
                continue
            
            etype = (activity.get("error_type") or "").strip().lower()
            if not etype or etype == "none":
                continue
            
            email = activity.get("user_email", "")
            problem = activity.get("problem_reference") or activity.get("problem_name", "Unknown")
            score = activity.get("score", 0) or 0
            timestamp = activity.get("timestamp", "")
            
            # Global counts
            global_errors[etype] = global_errors.get(etype, 0) + 1
            
            # Per-student
            if email not in student_errors:
                student_errors[email] = {}
            student_errors[email][etype] = student_errors[email].get(etype, 0) + 1
            
            # Per-problem
            if problem not in problem_errors:
                problem_errors[problem] = {}
            problem_errors[problem][etype] = problem_errors[problem].get(etype, 0) + 1
            
            # Score by error type
            if etype not in error_scores:
                error_scores[etype] = []
            error_scores[etype].append(score)
            
            # Timeline
            error_timeline.append({
                "email": email,
                "problem": problem,
                "error_type": etype,
                "score": score,
                "timestamp": timestamp
            })
        
        # Sort timeline by timestamp descending
        error_timeline.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Calculate avg score per error type
        avg_by_error = {}
        for etype, scores_list in error_scores.items():
            avg_by_error[etype] = round(sum(scores_list) / len(scores_list), 1) if scores_list else 0
        
        # Generate insights
        insights = []
        total_errors = sum(global_errors.values())
        if total_errors > 0:
            # Most common error
            most_common = max(global_errors, key=global_errors.get)
            insights.append({
                "type": "most_common",
                "icon": "🔴",
                "text": f"Most common error: {most_common} ({global_errors[most_common]} occurrences, {round(global_errors[most_common]/total_errors*100)}% of all errors)"
            })
            
            # Lowest scoring error type
            if avg_by_error:
                lowest = min(avg_by_error, key=avg_by_error.get)
                insights.append({
                    "type": "lowest_score",
                    "icon": "⚠️",
                    "text": f"Hardest error type: {lowest} (avg score {avg_by_error[lowest]}%)"
                })
            
            # Students with most errors
            if student_errors:
                student_totals = {e: sum(errs.values()) for e, errs in student_errors.items()}
                struggling = max(student_totals, key=student_totals.get)
                insights.append({
                    "type": "needs_help",
                    "icon": "🆘",
                    "text": f"Student needing most support: {struggling} ({student_totals[struggling]} total errors)"
                })
        
        result = {
            "global_distribution": global_errors,
            "avg_score_by_error": avg_by_error,
            "student_errors": student_errors,
            "problem_errors": problem_errors,
            "recent_errors": error_timeline[:30],
            "insights": insights,
            "total_errors": total_errors if total_errors else 0
        }
        cache_set("error-patterns", result)
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Error patterns failed: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# PROFESSOR ANALYTICS
# ============================================================================

@app.get("/api/professor/analytics")
def professor_analytics():
    """
    Comprehensive analytics for professor dashboard.
    Returns: score trends, topic performance, student rankings, time analysis.
    Cross-references activities with problems DB for topic data.
    Optional query param: ?group=1A to filter by student group.
    """
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500
    
    group_filter = request.args.get("group", "").strip()
    cache_key = f"analytics-{group_filter}" if group_filter else "analytics"
    
    cached = cache_get(cache_key)
    if cached:
        print(f"[CACHE HIT] {cache_key}")
        return jsonify(cached), 200
    
    try:
        # If group filter, get emails for that group first
        group_emails = None
        available_groups = set()
        if USERS_DB_ID:
            try:
                users = query_database(USERS_DB_ID)
                for u in users:
                    g = u.get("group") or u.get("Group") or ""
                    if g:
                        available_groups.add(g)
                if group_filter:
                    group_emails = set()
                    for u in users:
                        g = u.get("group") or u.get("Group") or ""
                        if g == group_filter:
                            email = u.get("Email") or u.get("email") or ""
                            if email:
                                group_emails.add(email)
                    print(f"[ANALYTICS] Group '{group_filter}': {len(group_emails)} students")
            except Exception as e:
                print(f"[ANALYTICS] Could not load users for group filter: {e}")
        
        all_activities = query_database(
            ACTIVITY_DB_ID,
            sorts=[{"property": "timestamp", "direction": "descending"}],
            max_pages=10
        )
        
        # Build topic lookup from problems DB
        topic_map = {}
        if NOTION_DB_ID:
            try:
                problems = query_database(NOTION_DB_ID, max_pages=5)
                for p in problems:
                    pname = p.get("Name") or p.get("name") or p.get("title") or ""
                    pref = p.get("reference") or p.get("Reference") or ""
                    # Extract first topic from key_concepts
                    kc = p.get("key_concepts") or p.get("Key Concepts") or ""
                    if isinstance(kc, list):
                        kc = kc[0] if kc else ""
                    topic = kc.split(";")[0].strip() if kc else "Other"
                    if pname:
                        topic_map[pname] = topic
                    if pref:
                        topic_map[pref] = topic
                print(f"[ANALYTICS] Built topic map with {len(topic_map)} entries")
            except Exception as e:
                print(f"[ANALYTICS] Could not build topic map: {e}")
        
        # ---- Aggregate data ----
        # Score trends by date
        daily_scores = {}
        # Topic performance
        topic_stats = {}
        # Student rankings
        student_data = {}
        # Time analysis
        time_by_day = {}
        # Marks analysis
        marks_data = []
        
        for act in all_activities:
            if act.get("action") != "completed":
                continue
            
            email = act.get("user_email", "")
            
            # Filter by group if specified
            if group_emails is not None and email not in group_emails:
                continue
            
            score = act.get("score", 0) or 0
            timestamp = act.get("timestamp", "")
            time_spent = act.get("time_spent_seconds", 0) or 0
            problem = act.get("problem_reference") or act.get("problem_name", "")
            marks_aw = act.get("marks_awarded")
            marks_tot = act.get("marks_total")
            error_type = (act.get("error_type") or "").strip().lower()
            
            # Date key (YYYY-MM-DD)
            date_key = timestamp[:10] if timestamp else "unknown"
            
            # Topic lookup
            topic = topic_map.get(problem, "")
            if not topic:
                # Try matching by problem_name
                pname = act.get("problem_name", "")
                topic = topic_map.get(pname, "Other")
            
            # --- Daily scores ---
            if date_key != "unknown":
                if date_key not in daily_scores:
                    daily_scores[date_key] = {"total": 0, "count": 0}
                daily_scores[date_key]["total"] += score
                daily_scores[date_key]["count"] += 1
            
            # --- Topic stats ---
            if topic:
                if topic not in topic_stats:
                    topic_stats[topic] = {"total_score": 0, "count": 0, "errors": {}, "marks_earned": 0, "marks_possible": 0}
                topic_stats[topic]["total_score"] += score
                topic_stats[topic]["count"] += 1
                if error_type and error_type != "none":
                    topic_stats[topic]["errors"][error_type] = topic_stats[topic]["errors"].get(error_type, 0) + 1
                if marks_aw is not None and marks_tot:
                    topic_stats[topic]["marks_earned"] += marks_aw
                    topic_stats[topic]["marks_possible"] += marks_tot
            
            # --- Student rankings ---
            if email:
                if email not in student_data:
                    student_data[email] = {"scores": [], "time": 0, "count": 0}
                student_data[email]["scores"].append(score)
                student_data[email]["time"] += time_spent
                student_data[email]["count"] += 1
            
            # --- Time by day of week ---
            if date_key != "unknown" and time_spent > 0:
                if date_key not in time_by_day:
                    time_by_day[date_key] = 0
                time_by_day[date_key] += time_spent
            
            # --- Marks data for export ---
            marks_data.append({
                "date": date_key,
                "student": email,
                "problem": problem,
                "topic": topic,
                "score": score,
                "error_type": error_type if error_type != "none" else "",
                "marks_awarded": marks_aw,
                "marks_total": marks_tot,
                "time_seconds": time_spent
            })
        
        # Process daily scores into sorted list
        score_trend = []
        for date_key in sorted(daily_scores.keys()):
            d = daily_scores[date_key]
            score_trend.append({
                "date": date_key,
                "avg_score": round(d["total"] / d["count"], 1) if d["count"] else 0,
                "submissions": d["count"]
            })
        
        # Process topic stats
        topic_performance = []
        for topic, stats in topic_stats.items():
            topic_performance.append({
                "topic": topic,
                "avg_score": round(stats["total_score"] / stats["count"], 1) if stats["count"] else 0,
                "attempts": stats["count"],
                "top_error": max(stats["errors"], key=stats["errors"].get) if stats["errors"] else None,
                "marks_pct": round(stats["marks_earned"] / stats["marks_possible"] * 100, 1) if stats["marks_possible"] else None
            })
        topic_performance.sort(key=lambda x: x["avg_score"])
        
        # Process student rankings
        student_rankings = []
        for email, data in student_data.items():
            avg = round(sum(data["scores"]) / len(data["scores"]), 1) if data["scores"] else 0
            # Calculate improvement: compare first half vs second half of scores
            scores = data["scores"]
            improvement = 0
            if len(scores) >= 4:
                mid = len(scores) // 2
                first_half = sum(scores[:mid]) / mid
                second_half = sum(scores[mid:]) / (len(scores) - mid)
                improvement = round(second_half - first_half, 1)
            student_rankings.append({
                "email": email,
                "name": email.split("@")[0],
                "avg_score": avg,
                "problems_solved": data["count"],
                "total_time_min": round(data["time"] / 60, 1),
                "improvement": improvement
            })
        student_rankings.sort(key=lambda x: x["avg_score"], reverse=True)
        
        # Time trend
        time_trend = []
        for date_key in sorted(time_by_day.keys()):
            time_trend.append({
                "date": date_key,
                "minutes": round(time_by_day[date_key] / 60, 1)
            })
        
        result = {
            "score_trend": score_trend,
            "topic_performance": topic_performance,
            "student_rankings": student_rankings,
            "time_trend": time_trend,
            "export_data": marks_data,
            "available_groups": sorted(list(available_groups)),
            "current_group": group_filter or "all",
            "summary": {
                "total_submissions": len(marks_data),
                "total_students": len(student_data),
                "overall_avg": round(sum(s["avg_score"] for s in student_rankings) / len(student_rankings), 1) if student_rankings else 0,
                "topics_covered": len(topic_performance)
            }
        }
        cache_set(cache_key, result)
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Analytics failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================================
# PROFESSOR FEEDBACK ON STUDENT ATTEMPTS
# ============================================================================

@app.post("/api/professor/feedback")
def save_professor_feedback():
    """
    Save professor's feedback comment on a specific student activity.
    Expects: { activity_id, feedback }
    """
    data = request.get_json(force=True, silent=True) or {}
    activity_id = data.get("activity_id", "").strip()
    feedback = data.get("feedback", "").strip()
    
    if not activity_id:
        return jsonify({"error": "Missing activity_id"}), 400
    if not feedback:
        return jsonify({"error": "Missing feedback"}), 400
    
    try:
        update_page_properties(activity_id, {
            "professor_feedback": {"rich_text": [{"text": {"content": feedback[:2000]}}]}
        })
        print(f"[FEEDBACK] Saved feedback for activity {activity_id}: {feedback[:50]}...")
        cache_clear()  # Invalidate caches
        return jsonify({"ok": True, "message": "Feedback saved"}), 200
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Save feedback failed: {error_msg}")
        # If property doesn't exist in Notion, give helpful error
        if "professor_feedback" in error_msg or "validation" in error_msg.lower():
            return jsonify({"error": "Please add a 'professor_feedback' Rich Text property to your Activity database in Notion first."}), 400
        return jsonify({"error": error_msg}), 500

@app.get("/api/professor/feedback/<activity_id>")
def get_professor_feedback(activity_id):
    """Get feedback for a specific activity."""
    try:
        pid = sanitize_uuid(activity_id)
        if not pid:
            return jsonify({"error": "Invalid ID"}), 400
        url = f"{NOTION_BASE}/pages/{pid}"
        resp = requests.get(url, headers=NOTION_HEADERS, timeout=10)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        fb = props.get("professor_feedback", {})
        fb_text = ""
        if fb.get("type") == "rich_text":
            arr = fb.get("rich_text", [])
            fb_text = arr[0].get("plain_text", "") if arr else ""
        return jsonify({"feedback": fb_text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# PRACTICE MODE — INTELLIGENT REINFORCEMENT
# ============================================================================

@app.get("/api/student/practice-recommendations")
@require_auth
def practice_recommendations():
    """
    Analyze student weaknesses and recommend problems.
    Returns 2 categories:
    1. "topic_practice" — more problems on topics they've worked on
    2. "weakness_reinforcement" — targeted problems for specific error patterns
    
    Uses: error_type, key_concept, topic, score from activity history.
    Excludes problems already solved with score >= 70.
    """
    if not ACTIVITY_DB_ID or not NOTION_DB_ID:
        return jsonify({"error": "Databases not configured"}), 500
    
    user = request.user
    email = user.get("email")
    
    try:
        # 1. Get student's activity history
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        # 2. Get all available problems
        all_problems = query_database(NOTION_DB_ID, max_pages=5)
        
        # 3. Analyze student data
        solved_well = set()  # problems scored >= 85 (truly mastered)
        solved_any = set()   # all attempted problems
        topic_scores = {}    # topic → [scores]
        error_patterns = {}  # (topic, error_type) → count
        weak_concepts = {}   # key_concept → {count, avg_score, error_types}
        recent_topics = []   # last 10 topics worked on
        
        for act in activities:
            if act.get("action") != "completed":
                continue
            
            problem_ref = act.get("problem_reference") or act.get("problem_name", "")
            score = act.get("score", 0) or 0
            error_type = (act.get("error_type") or "").strip().lower()
            key_concept = (act.get("key_concept") or "").strip()
            
            solved_any.add(problem_ref)
            if score >= 85:
                solved_well.add(problem_ref)
            
            # Find topic for this problem
            topic = ""
            for p in all_problems:
                pref = p.get("reference") or p.get("Reference") or ""
                pname = p.get("Name") or p.get("name") or ""
                if pref == problem_ref or pname == problem_ref:
                    kc = p.get("key_concepts") or p.get("Key Concepts") or ""
                    if isinstance(kc, list):
                        kc = kc[0] if kc else ""
                    topic = kc.split(";")[0].strip() if kc else ""
                    break
            
            if not topic and key_concept:
                topic = key_concept
            
            if topic:
                if topic not in topic_scores:
                    topic_scores[topic] = []
                topic_scores[topic].append(score)
                
                if len(recent_topics) < 10 and topic not in recent_topics:
                    recent_topics.append(topic)
                
                # Track error patterns per topic
                if error_type and error_type != "none":
                    key = f"{topic}|{error_type}"
                    error_patterns[key] = error_patterns.get(key, 0) + 1
            
            # Track weak concepts (broadened: any score below 85 OR any error)
            if key_concept and (score < 85 or (error_type and error_type != "none")):
                if key_concept not in weak_concepts:
                    weak_concepts[key_concept] = {"count": 0, "total_score": 0, "error_types": {}}
                weak_concepts[key_concept]["count"] += 1
                weak_concepts[key_concept]["total_score"] += score
                if error_type and error_type != "none":
                    weak_concepts[key_concept]["error_types"][error_type] = \
                        weak_concepts[key_concept]["error_types"].get(error_type, 0) + 1
        
        # 4. Build problem lookup by topic
        problems_by_topic = {}
        for p in all_problems:
            pid = p.get("id", "")
            pname = p.get("Name") or p.get("name") or ""
            pref = p.get("reference") or p.get("Reference") or ""
            kc = p.get("key_concepts") or p.get("Key Concepts") or ""
            if isinstance(kc, list):
                kc = kc[0] if kc else ""
            topic = kc.split(";")[0].strip() if kc else "Other"
            marks = p.get("marks") or p.get("Marks") or 0
            
            identifier = pref or pname
            
            prob_obj = {
                "id": pid,
                "name": pname,
                "reference": pref,
                "topic": topic,
                "marks": marks,
                "already_solved": identifier in solved_well,
                "attempted": identifier in solved_any
            }
            
            if topic not in problems_by_topic:
                problems_by_topic[topic] = []
            problems_by_topic[topic].append(prob_obj)
        
        # 5. Generate TOPIC PRACTICE recommendations
        topic_practice = []
        
        # Include ALL topics that have available problems (not just recent)
        all_topic_names = set(list(topic_scores.keys()) + list(problems_by_topic.keys()))
        
        for topic in all_topic_names:
            if topic == "Other" or not topic:
                continue
            
            scores = topic_scores.get(topic, [])
            avg_score = round(sum(scores) / len(scores), 1) if scores else 0
            
            # Find unsolved problems in this topic
            available = [p for p in problems_by_topic.get(topic, []) if not p["already_solved"]]
            
            if available:
                # Determine status
                if not scores:
                    status = "not_started"
                elif avg_score >= 85:
                    status = "strong"
                elif avg_score >= 65:
                    status = "developing"
                else:
                    status = "needs_work"
                
                topic_practice.append({
                    "topic": topic,
                    "avg_score": avg_score,
                    "problems_solved": len(scores),
                    "problems_available": len(available),
                    "status": status,
                    "recommended_problems": available[:5]  # top 5 unsolved
                })
        
        # Sort: needs_work first, then developing, not_started, strong
        status_order = {"needs_work": 0, "developing": 1, "not_started": 2, "strong": 3}
        topic_practice.sort(key=lambda x: (status_order.get(x["status"], 4), x["avg_score"]))
        
        # 6. Generate WEAKNESS REINFORCEMENT recommendations
        weakness_reinforcement = []
        
        # Find topics with specific error patterns
        for key, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            topic, error_type = key.split("|")
            if count < 1:
                continue
            
            # Find unsolved problems in this topic
            available = [p for p in problems_by_topic.get(topic, []) if not p["already_solved"]]
            
            if available:
                scores = topic_scores.get(topic, [])
                avg = round(sum(scores) / len(scores), 1) if scores else 0
                
                weakness_reinforcement.append({
                    "topic": topic,
                    "error_type": error_type,
                    "occurrences": count,
                    "avg_score_in_topic": avg,
                    "description": get_weakness_description(topic, error_type, count),
                    "recommended_problems": available[:3]
                })
        
        # Limit to top 5 weaknesses
        weakness_reinforcement = weakness_reinforcement[:5]
        
        # 7. Summary
        total_available = sum(1 for p in all_problems 
                           if (p.get("reference") or p.get("Reference") or p.get("Name") or p.get("name") or "") not in solved_well)
        
        return jsonify({
            "student_email": email,
            "problems_solved": len(solved_well),
            "problems_attempted": len(solved_any),
            "total_problems": len(all_problems),
            "problems_remaining": total_available,
            "topic_practice": topic_practice,
            "weakness_reinforcement": weakness_reinforcement,
            "weak_concepts": [
                {
                    "concept": concept,
                    "times_failed": data["count"],
                    "avg_score": round(data["total_score"] / data["count"], 1),
                    "main_error": max(data["error_types"], key=data["error_types"].get) if data["error_types"] else None
                }
                for concept, data in sorted(weak_concepts.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            ]
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Practice recommendations failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def get_weakness_description(topic, error_type, count):
    """Generate a human-readable description of the weakness."""
    error_descriptions = {
        "conceptual": f"You've made conceptual errors in {topic} {count} time(s). Focus on understanding the underlying theory before solving.",
        "calculation": f"You've made calculation mistakes in {topic} {count} time(s). Double-check your arithmetic and unit conversions.",
        "units": f"You've had unit errors in {topic} {count} time(s). Always write units at each step and verify dimensional consistency.",
        "method": f"You've used incorrect methods in {topic} {count} time(s). Review which formulas and approaches apply to this type of problem.",
        "incomplete": f"You've submitted incomplete solutions in {topic} {count} time(s). Make sure to show all steps and reach a final answer."
    }
    return error_descriptions.get(error_type, f"You've had {error_type} errors in {topic} {count} time(s). Review this topic carefully.")

# ============================================================================
# AI PROBLEM GENERATION ENGINE
# ============================================================================

@app.post("/api/generate-problem")
@require_auth
def generate_problem():
    """
    Generate a new problem variant using AI.
    
    Input JSON:
    - topic (required): e.g. "Magnetism", "Optics"
    - marks (optional): target marks, default 4
    - error_type (optional): weakness to target, e.g. "calculation"
    - template_id (optional): specific problem to use as template
    
    Returns a generated problem ready for the chatbot to display.
    """
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "API key not configured"}), 500
    
    data = request.get_json(force=True, silent=True) or {}
    topic = (data.get("topic") or "").strip()
    target_marks = data.get("marks", 4)
    error_type = (data.get("error_type") or "").strip()
    template_id = (data.get("template_id") or "").strip()
    
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    
    try:
        # 1. Find template problem(s) from the same topic
        template_problem = None
        template_examples = []
        
        if template_id:
            # Use specific template
            try:
                template_problem = fetch_page(template_id)
            except:
                pass
        
        if not template_problem and NOTION_DB_ID:
            # Find problems in the same topic
            all_problems = query_database(NOTION_DB_ID, max_pages=5)
            topic_problems = []
            for p in all_problems:
                kc = p.get("key_concepts") or p.get("Key Concepts") or ""
                if isinstance(kc, list):
                    kc = kc[0] if kc else ""
                p_topic = kc.split(";")[0].strip().lower() if kc else ""
                if topic.lower() in p_topic or p_topic in topic.lower():
                    topic_problems.append(p)
            
            if topic_problems:
                # Pick up to 2 as examples for style reference
                import random
                samples = random.sample(topic_problems, min(2, len(topic_problems)))
                for s in samples:
                    pid = s.get("id", "")
                    try:
                        full = fetch_page(pid)
                        template_examples.append(full)
                    except:
                        template_examples.append(s)
                template_problem = template_examples[0] if template_examples else None
        
        # 2. Build generation prompt
        examples_text = ""
        if template_examples:
            for i, ex in enumerate(template_examples):
                examples_text += f"""
--- EXAMPLE PROBLEM {i+1} ---
Name: {ex.get('name') or ex.get('Name', '')}
Statement: {ex.get('problem_statement', '') or ex.get('statement', '')}
Given values: {ex.get('given_values', '')}
What to find: {ex.get('find', '')}
Marks: {ex.get('marks') or ex.get('Marks', target_marks)}
Solution: {ex.get('full_solution', '') or ex.get('step_by_step', '')}
"""
        elif template_problem:
            examples_text = f"""
--- TEMPLATE PROBLEM ---
Name: {template_problem.get('name') or template_problem.get('Name', '')}
Statement: {template_problem.get('problem_statement', '') or template_problem.get('statement', '')}
Given values: {template_problem.get('given_values', '')}
What to find: {template_problem.get('find', '')}
Marks: {template_problem.get('marks') or template_problem.get('Marks', target_marks)}
Solution: {template_problem.get('full_solution', '') or template_problem.get('step_by_step', '')}
"""
        
        weakness_context = ""
        if error_type:
            weakness_context = f"""
IMPORTANT: This problem should specifically test areas where students commonly make {error_type} errors.
- If error_type is "calculation": include multi-step calculations with unit conversions
- If error_type is "conceptual": test understanding of when/why to apply specific formulas
- If error_type is "units": require careful unit tracking (SI conversions, prefixes)
- If error_type is "method": require choosing between similar approaches
- If error_type is "incomplete": require showing ALL steps for full marks
"""
        
        prompt = f"""You are an expert IB Physics HL examiner. Generate a NEW, ORIGINAL problem for the topic "{topic}" worth [{target_marks} marks].

REQUIREMENTS:
1. The problem must be at IB Physics HL standard (difficulty and style)
2. It must be DIFFERENT from the examples below but test similar concepts
3. Include realistic numerical values with proper units
4. The problem should be solvable in approximately {int(target_marks * 1.5)} minutes
5. Provide a complete step-by-step solution with mark allocation
{weakness_context}

{examples_text if examples_text else f"Generate an IB-style problem on {topic} worth {target_marks} marks."}

Respond with ONLY a valid JSON object (no markdown, no backticks):
{{
  "name": "<short descriptive name, e.g. 'Magnetic Force on Moving Charge'>",
  "problem_statement": "<the complete problem text as it would appear on an IB exam>",
  "given_values": "<list the given numerical values>",
  "find": "<what the student needs to calculate/determine>",
  "marks": {target_marks},
  "key_concept": "{topic}",
  "steps": [
    {{
      "id": "1",
      "title": "<step title, e.g. 'Identify the formula'>",
      "content": "<what the student should do>",
      "expected_answer": "<the correct result for this step>",
      "marks": <marks for this step>
    }}
  ],
  "final_answer": "<the complete final answer with units>",
  "full_solution": "<complete worked solution as one text block>"
}}
"""
        
        # 3. Call Claude API
        response_text = call_anthropic(prompt, max_tokens=2000)
        
        # 4. Parse response
        import re
        cleaned = response_text.strip()
        # Remove markdown code fences if present
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        generated = json.loads(cleaned)
        
        # Add metadata
        generated["source"] = "ai_generated"
        generated["topic"] = topic
        generated["template_reference"] = template_problem.get("problem_reference", "") if template_problem else ""
        generated["generated_at"] = datetime.utcnow().isoformat()
        
        # Generate a temporary ID for tracking
        import hashlib
        temp_id = "gen_" + hashlib.md5(
            f"{topic}_{target_marks}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        generated["id"] = temp_id
        generated["problem_reference"] = f"AI-{topic[:3].upper()}-{target_marks}m-{temp_id[-6:]}"
        
        print(f"[GENERATE] Created problem: {generated['problem_reference']} | Topic: {topic} | Marks: {target_marks}")
        
        return jsonify(generated), 200
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse generated problem: {e}")
        print(f"[ERROR] Raw response: {response_text[:500]}")
        return jsonify({"error": "Failed to parse generated problem. Please try again."}), 500
    except Exception as e:
        print(f"[ERROR] Problem generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/api/evaluate-generated")
@require_auth
def evaluate_generated():
    """
    Evaluate a student's solution to an AI-generated problem.
    Similar to /submit but works with in-memory problem data.
    Saves result to Activity with source='ai_generated'.
    """
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "API key not configured"}), 500
    
    user = request.user
    user_email = user.get("email", "")
    
    data = request.get_json(force=True, silent=True) or {}
    
    # Problem details (from generate-problem response)
    problem_name = data.get("problem_name", "")
    problem_statement = data.get("problem_statement", "")
    problem_reference = data.get("problem_reference", "")
    full_solution = data.get("full_solution", "")
    marks = data.get("marks", 4)
    key_concept = data.get("key_concept", "")
    topic = data.get("topic", "")
    
    # Student solution
    solution_text = (data.get("solution") or "").strip()
    time_spent = int(data.get("time_spent", 0) or 0)
    
    # Image support
    image_base64 = data.get("image_base64", "")
    image_media_type = data.get("image_media_type", "image/jpeg")
    has_image = bool(image_base64)
    
    if not solution_text and not has_image:
        return jsonify({"error": "No solution provided"}), 400
    
    try:
        marks_context = ""
        if marks:
            marks_context = f"""
This problem is worth [{marks} marks] in the IB exam.
Award marks according to IB marking standards:
- Each mark corresponds to a specific step, concept, or correct value
- Partial credit is expected
- ECF (Error Carried Forward) applies
"""
        
        solution_reference = ""
        if full_solution:
            solution_reference = f"""
OFFICIAL SOLUTION (for reference - do NOT share with student):
{full_solution}

Compare the student's work against this reference.
"""
        
        prompt = f"""You are an expert IB Physics HL examiner evaluating a student's solution.

PROBLEM DETAILS:
Name: {problem_name}
Statement: {problem_statement}
Topic: {topic}
{marks_context}
{solution_reference}

STUDENT'S SOLUTION:
{solution_text}
{"The student has attached a PHOTO of their handwritten solution. Evaluate the handwritten work carefully." if has_image else ""}

TIME SPENT: {time_spent // 60}:{time_spent % 60:02d}

EVALUATE and respond with ONLY a valid JSON object (no markdown, no backticks):
{{
  "score": <number 0-100>,
  "correct": <true if score >= 70, false otherwise>,
  "marks_awarded": <number of IB marks earned out of {marks}>,
  "marks_total": {marks},
  "feedback": "<detailed feedback>",
  "error_type": "<classify: 'conceptual' | 'calculation' | 'units' | 'method' | 'incomplete' | 'none'>",
  "missing_steps": "<key steps missed>",
  "key_concept": "{key_concept}",
  "time_taken": "{time_spent // 60}:{time_spent % 60:02d}"
}}
"""
        
        if has_image:
            response_text = call_anthropic_vision(prompt, image_base64, image_media_type, max_tokens=1500)
        else:
            response_text = call_anthropic(prompt, max_tokens=1500)
        
        # Parse result
        import re
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        result = json.loads(cleaned)
        result.setdefault('error_type', 'none')
        result.setdefault('marks_awarded', 0)
        result.setdefault('marks_total', marks)
        
        # Save to Activity DB with ai_generated marker
        if ACTIVITY_DB_ID:
            try:
                properties = {
                    "user_email": {"email": user_email},
                    "action": {"select": {"name": "completed"}},
                    "problem_name": {"title": [{"text": {"content": f"[AI] {problem_name}"[:100]}}]},
                    "problem_reference": {"rich_text": [{"text": {"content": problem_reference[:100]}}]},
                    "score": {"number": result.get("score", 0)},
                    "time_spent_seconds": {"number": time_spent},
                    "solution_text": {"rich_text": [{"text": {"content": (solution_text or "")[:2000]}}]},
                    "timestamp": {"date": {"start": datetime.utcnow().isoformat() + "Z"}}
                }
                
                # Add structured fields
                if result.get("error_type"):
                    properties["error_type"] = {"rich_text": [{"text": {"content": result["error_type"][:200]}}]}
                if result.get("marks_awarded") is not None:
                    properties["marks_awarded"] = {"number": result["marks_awarded"]}
                if result.get("marks_total") is not None:
                    properties["marks_total"] = {"number": result["marks_total"]}
                if result.get("key_concept"):
                    properties["key_concept"] = {"rich_text": [{"text": {"content": result["key_concept"][:200]}}]}
                
                requests.post(
                    f"{NOTION_BASE}/pages",
                    headers=NOTION_HEADERS,
                    json={"parent": {"database_id": ACTIVITY_DB_ID}, "properties": properties},
                    timeout=10
                )
                print(f"[EVAL-GEN] Saved activity for {user_email}: {problem_reference} = {result.get('score', 0)}%")
            except Exception as e:
                print(f"[WARN] Could not save generated problem activity: {e}")
        
        result["source"] = "ai_generated"
        result["problem_reference"] = problem_reference
        
        return jsonify(result), 200
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse evaluation: {e}")
        return jsonify({"score": 50, "correct": False, "feedback": "Could not parse evaluation. Please try again.", "error_type": "none"}), 200
    except Exception as e:
        print(f"[ERROR] Generated problem evaluation failed: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# HOMEWORK SYSTEM - UPDATED WITH time_limit_minutes
# ============================================================================

@app.post("/api/assign-homework")
def assign_homework():
    """
    Assign homework with optional time limit.
    UPDATED: Now supports time_limit_minutes field.
    """
    if not HOMEWORK_DB_ID: 
        return jsonify({"error": "HOMEWORK_DB_ID missing"}), 500
    
    data = request.get_json(force=True, silent=True) or request.form.to_dict()
    student_id = data.get("student_id")
    title = data.get("title")
    description = data.get("description", "")
    due_date = data.get("due_date")
    points = int(data.get("points", 0))
    topic = data.get("topic", "")
    problem_references = data.get("problem_references", "")
    time_limit_minutes = int(data.get("time_limit_minutes", 0))  # NEW
    
    if not title or not student_id or not due_date:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        def build_props(email):
            props = {
                "Title": {"title": [{"text": {"content": title}}]},
                "student_email": {"email": email},
                "description": {"rich_text": [{"text": {"content": description}}]},
                "due_date": {"date": {"start": due_date}},
                "points": {"number": points},
                "topic": {"rich_text": [{"text": {"content": topic}}]},
                "problem_references": {"rich_text": [{"text": {"content": problem_references}}]},
                "completed": {"checkbox": False},
                "created_at": {"date": {"start": datetime.utcnow().isoformat()}},
                "created_by": {"rich_text": [{"text": {"content": "professor"}}]}
            }
            
            # NEW: Add time limit if specified
            if time_limit_minutes > 0:
                props["time_limit_minutes"] = {"number": time_limit_minutes}
            
            return props
        
        if student_id == "all":
            students = query_database(USERS_DB_ID)
            count = 0
            for s in students:
                if s.get("Email"):
                    create_page_in_database(HOMEWORK_DB_ID, build_props(s["Email"]))
                    count += 1
            return jsonify({"success": True, "message": f"Assigned to {count} students"})
        
        elif student_id.startswith("group:"):
            group_name = student_id.replace("group:", "")
            students = query_database(USERS_DB_ID)
            count = 0
            for s in students:
                student_group = s.get("group") or s.get("Group") or ""
                if s.get("Email") and student_group == group_name:
                    create_page_in_database(HOMEWORK_DB_ID, build_props(s["Email"]))
                    count += 1
            if count == 0:
                return jsonify({"error": f"No students found in group '{group_name}'"}), 400
            return jsonify({"success": True, "message": f"Assigned to {count} students in group {group_name}"})
        
        else:
            create_page_in_database(HOMEWORK_DB_ID, build_props(student_id))
            return jsonify({"success": True, "message": "Homework assigned"})
    
    except Exception as e: 
        return jsonify({"error": str(e)}), 500

@app.get("/api/student-homework")
@require_auth
def get_student_homework():
    """
    Get student homework with progress calculation.
    FIXED: Only counts completions AFTER homework was assigned.
    """
    try:
        student_email = request.user["email"]
        
        hw_list = query_database(
            HOMEWORK_DB_ID, 
            {"property": "student_email", "email": {"equals": student_email}}, 
            sorts=[{"property": "due_date", "direction": "ascending"}]
        )
        
        # Get ALL completed activities for this student with timestamps
        activities = query_database(
            ACTIVITY_DB_ID,
            {"property": "user_email", "email": {"equals": student_email}}
        )
        
        # Build dict of completed problems with their timestamps
        # Key: problem_reference, Value: list of completion timestamps
        completed_problems = {}
        for act in activities:
            if act.get("action") == "completed":
                ref = act.get("problem_reference") or act.get("problem_name")
                timestamp = act.get("timestamp")
                if ref and timestamp:
                    if ref not in completed_problems:
                        completed_problems[ref] = []
                    completed_problems[ref].append(timestamp)
        
        print(f"[HW] Student {student_email} has completed: {list(completed_problems.keys())}")
        
        processed_homework = []
        now = datetime.now(timezone.utc)
        
        for hw in hw_list:
            problem_refs_str = hw.get("problem_references", "") or ""
            hw_created_at = hw.get("created_at") or ""
            
            # Parse homework creation date
            hw_created_datetime = None
            if hw_created_at:
                try:
                    hw_created_datetime = datetime.fromisoformat(hw_created_at.replace("Z", "+00:00"))
                except:
                    pass
            
            # ALWAYS calculate progress, regardless of completed flag
            refs = [r.strip() for r in problem_refs_str.split(",") if r.strip()]
            total_problems = len(refs)
            
            # Count only completions AFTER homework was assigned
            solved_count = 0
            problem_solutions = []
            
            for ref in refs:
                if ref in completed_problems:
                    # Check if any completion is after homework creation
                    completion_after_hw = False
                    latest_completion = None
                    
                    for ts in completed_problems[ref]:
                        try:
                            completion_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            # If no hw_created_datetime, accept any completion
                            if hw_created_datetime is None or completion_time > hw_created_datetime:
                                completion_after_hw = True
                                if latest_completion is None or completion_time > latest_completion:
                                    latest_completion = completion_time
                        except:
                            pass
                    
                    if completion_after_hw:
                        solved_count += 1
                        problem_solutions.append({"problem_reference": ref, "completed": True})
                    else:
                        problem_solutions.append({"problem_reference": ref, "completed": False})
                else:
                    problem_solutions.append({"problem_reference": ref, "completed": False})
            
            # Determine completion: ALL problems must be solved AFTER homework was assigned
            is_actually_completed = (total_problems > 0 and solved_count >= total_problems)
            
            hw_data = {
                "id": hw.get("id"),
                "title": hw.get("Title") or hw.get("title") or hw.get("name") or "Assignment",
                "description": hw.get("description", ""),
                "due_date": hw.get("due_date"),
                "points": hw.get("points", 0),
                "topic": hw.get("topic", ""),
                "problem_references": problem_refs_str,
                "completed": is_actually_completed,  # Based on actual progress, not Notion flag
                "completed_at": hw.get("completed_at") if is_actually_completed else None,
                "time_limit_minutes": hw.get("time_limit_minutes") or 0,
                "is_late": False,
                "solved_count": solved_count,
                "total_problems": total_problems,
                "problem_solutions": problem_solutions
            }
            
            # Check if submission is late (only for completed homework)
            if is_actually_completed and hw_data["due_date"]:
                try:
                    due = datetime.fromisoformat(hw_data["due_date"].replace("Z", "+00:00"))
                    if hw.get("completed_at"):
                        completed = datetime.fromisoformat(hw.get("completed_at").replace("Z", "+00:00"))
                        hw_data["is_late"] = completed > due
                    else:
                        # If no completed_at, check if now is past due
                        hw_data["is_late"] = now > due
                except Exception as e:
                    print(f"[HW] Error parsing dates: {e}")
            
            # Check if currently past due (for pending homework)
            if not is_actually_completed and hw_data["due_date"]:
                try:
                    due = datetime.fromisoformat(hw_data["due_date"].replace("Z", "+00:00"))
                    hw_data["is_overdue"] = now > due
                except:
                    hw_data["is_overdue"] = False
            
            print(f"[HW] '{hw_data['title']}': {solved_count}/{total_problems} -> {'COMPLETED' if is_actually_completed else 'PENDING'}")
            
            processed_homework.append(hw_data)
        
        return jsonify({"homework": processed_homework})
    except Exception as e: 
        return jsonify({"error": str(e)}), 500

@app.post("/api/complete-homework/<homework_id>")
@require_auth
def complete_homework(homework_id):
    try:
        hw = fetch_page(homework_id)
        if hw.get("student_email") != request.user["email"]: 
            return jsonify({"error": "Forbidden"}), 403
        
        # Check if late
        is_late = False
        if hw.get("due_date"):
            try:
                due = datetime.fromisoformat(hw["due_date"].replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                is_late = now > due
            except:
                pass
        
        update_page_properties(homework_id, {
            "completed": {"checkbox": True},
            "completed_at": {"date": {"start": datetime.utcnow().isoformat()}}
        })
        
        return jsonify({
            "success": True, 
            "points_earned": hw.get("points", 0),
            "is_late": is_late
        })
    except Exception as e: 
        return jsonify({"error": str(e)}), 500

@app.get("/api/students-list")
def get_students_list():
    try:
        users = query_database(USERS_DB_ID)
        students = []
        groups = set()
        
        for u in users:
            if u.get("Email"):
                group = u.get("group") or u.get("Group") or ""
                students.append({
                    "id": u.get("Email"),
                    "name": u.get("Name"),
                    "email": u.get("Email"),
                    "group": group
                })
                if group:
                    groups.add(group)
        
        return jsonify({
            "students": students,
            "groups": sorted(list(groups))
        })
    except Exception as e: 
        return jsonify({"error": str(e)}), 500

# ============================================================================
# NEW: Get homework details for professor (with student solutions)
# ============================================================================

@app.get("/api/homework-details/<homework_id>")
def get_homework_details(homework_id):
    """
    Get detailed homework info including student's solution text.
    For professor to review student work.
    """
    try:
        hw = fetch_page(homework_id)
        student_email = hw.get("student_email")
        problem_refs = hw.get("problem_references", "")
        
        # Get student's activities for these problems
        if student_email and problem_refs:
            activities = query_database(
                ACTIVITY_DB_ID,
                {"property": "user_email", "email": {"equals": student_email}}
            )
            
            refs = [r.strip() for r in problem_refs.split(",") if r.strip()]
            problem_solutions = []
            
            for ref in refs:
                # Find the latest completed activity for this problem
                for act in sorted(activities, key=lambda x: x.get("timestamp", ""), reverse=True):
                    act_ref = act.get("problem_reference") or act.get("problem_name", "")
                    if act_ref == ref and act.get("action") == "completed":
                        problem_solutions.append({
                            "problem_reference": ref,
                            "score": act.get("score", 0),
                            "time_spent_seconds": act.get("time_spent_seconds", 0),
                            "solution_text": act.get("solution_text", ""),
                            "timestamp": act.get("timestamp", "")
                        })
                        break
                else:
                    # Problem not completed yet
                    problem_solutions.append({
                        "problem_reference": ref,
                        "score": None,
                        "time_spent_seconds": None,
                        "solution_text": None,
                        "timestamp": None
                    })
            
            hw["problem_solutions"] = problem_solutions
        
        return jsonify({"homework": hw})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================================
# RESOURCES BY TOPIC
# ========================================

@app.get("/api/resources")
def get_all_resources():
    """
    Get all resources by topic.
    Returns a dictionary with topic as key and resources as value.
    """
    if not RESOURCES_DB_ID:
        return jsonify({"error": "RESOURCES_DB_ID not configured"}), 500
    
    try:
        resources = query_database(RESOURCES_DB_ID)
        
        result = {}
        for r in resources:
            topic = r.get("topic") or r.get("Topic") or r.get("name") or ""
            if not topic:
                continue
            
            result[topic] = {
                "study_guide_url": r.get("study_guide_url") or None,
                "formula_sheet_url": r.get("formula_sheet_url") or None,
                "audio_url": r.get("audio_url") or None,
                "hints": r.get("hints") or None,
                "common_mistakes": r.get("common_mistakes") or None
            }
        
        return jsonify({"resources": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/resources/<topic>")
def get_resources_by_topic(topic):
    """
    Get resources for a specific topic.
    """
    if not RESOURCES_DB_ID:
        return jsonify({"error": "RESOURCES_DB_ID not configured"}), 500
    
    try:
        # Query with filter for specific topic
        resources = query_database(
            RESOURCES_DB_ID,
            filter_obj={
                "property": "topic",
                "title": {"equals": topic}
            }
        )
        
        if not resources:
            # Try case-insensitive search
            all_resources = query_database(RESOURCES_DB_ID)
            for r in all_resources:
                r_topic = r.get("topic") or r.get("Topic") or r.get("name") or ""
                if r_topic.lower() == topic.lower():
                    return jsonify({
                        "topic": r_topic,
                        "study_guide_url": r.get("study_guide_url") or None,
                        "formula_sheet_url": r.get("formula_sheet_url") or None,
                        "audio_url": r.get("audio_url") or None,
                        "hints": r.get("hints") or None,
                        "common_mistakes": r.get("common_mistakes") or None
                    })
            return jsonify({"error": "Topic not found"}), 404
        
        r = resources[0]
        return jsonify({
            "topic": topic,
            "study_guide_url": r.get("study_guide_url") or None,
            "formula_sheet_url": r.get("formula_sheet_url") or None,
            "audio_url": r.get("audio_url") or None,
            "hints": r.get("hints") or None,
            "common_mistakes": r.get("common_mistakes") or None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================================
# STUDENT HOMEWORK (for professor view)
# ========================================

@app.get("/api/professor/student-homework")
def get_student_homework_for_professor():
    """
    Get all homework assigned to a specific student with their progress.
    Query param: email (student's email)
    Used by professor dashboard.
    FIXED: Only counts completions AFTER homework was assigned.
    """
    email = request.args.get("email", "")
    if not email:
        return jsonify({"error": "Email parameter required"}), 400
    
    if not HOMEWORK_DB_ID or not ACTIVITY_DB_ID:
        return jsonify({"error": "Database not configured"}), 500
    
    try:
        # Get all homework
        all_homework = query_database(HOMEWORK_DB_ID)
        
        # Get student's activity
        student_activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}}
        )
        
        # Build a dict of completed problems with ALL their completions (with timestamps)
        completed_problems = {}
        for act in student_activities:
            if act.get("action") == "completed":
                ref = act.get("problem_reference") or act.get("problem_name", "")
                timestamp = act.get("timestamp")
                if ref and timestamp:
                    if ref not in completed_problems:
                        completed_problems[ref] = []
                    completed_problems[ref].append({
                        "score": act.get("score"),
                        "time_spent_seconds": act.get("time_spent_seconds"),
                        "timestamp": timestamp
                    })
        
        # Filter homework for this student
        student_homework = []
        for hw in all_homework:
            student_email_field = hw.get("student_email", "")
            # Check if assigned to this specific student
            if email == student_email_field or email in str(student_email_field):
                # Parse problem references
                refs_str = hw.get("problem_references", "")
                refs = [r.strip() for r in refs_str.split(",") if r.strip()]
                
                # Get homework creation date
                hw_created_at = hw.get("created_at") or ""
                hw_created_datetime = None
                if hw_created_at:
                    try:
                        hw_created_datetime = datetime.fromisoformat(hw_created_at.replace("Z", "+00:00"))
                    except:
                        pass
                
                # Calculate progress - only count completions AFTER homework was assigned
                solved_count = 0
                problem_solutions = []
                
                for ref in refs:
                    if ref in completed_problems:
                        # Find completion AFTER homework was assigned
                        valid_completion = None
                        for completion in completed_problems[ref]:
                            try:
                                completion_time = datetime.fromisoformat(completion["timestamp"].replace("Z", "+00:00"))
                                if hw_created_datetime is None or completion_time > hw_created_datetime:
                                    # This is a valid completion (after homework was assigned)
                                    if valid_completion is None or completion_time > datetime.fromisoformat(valid_completion["timestamp"].replace("Z", "+00:00")):
                                        valid_completion = completion
                            except:
                                pass
                        
                        if valid_completion:
                            solved_count += 1
                            problem_solutions.append({
                                "problem_reference": ref,
                                "score": valid_completion["score"],
                                "time_spent_seconds": valid_completion["time_spent_seconds"],
                                "timestamp": valid_completion["timestamp"]
                            })
                        else:
                            problem_solutions.append({
                                "problem_reference": ref,
                                "score": None,
                                "time_spent_seconds": None,
                                "timestamp": None
                            })
                    else:
                        problem_solutions.append({
                            "problem_reference": ref,
                            "score": None,
                            "time_spent_seconds": None,
                            "timestamp": None
                        })
                
                is_completed = solved_count >= len(refs) and len(refs) > 0
                
                # Check for late submission
                due_date = hw.get("due_date")
                late_submission = False
                if is_completed and due_date:
                    # Find the latest completion timestamp
                    completion_times = [
                        ps["timestamp"] for ps in problem_solutions 
                        if ps["timestamp"]
                    ]
                    if completion_times:
                        last_completion = max(completion_times)
                        if last_completion > due_date:
                            late_submission = True
                
                student_homework.append({
                    "id": hw.get("id"),
                    "title": hw.get("Title") or hw.get("title") or hw.get("name") or "Homework",
                    "description": hw.get("description", ""),
                    "due_date": due_date,
                    "points": hw.get("points"),
                    "problem_references": refs_str,
                    "total_problems": len(refs),
                    "solved_count": solved_count,
                    "completed": is_completed,
                    "late_submission": late_submission,
                    "problem_solutions": problem_solutions
                })
        
        # Sort by due date (most recent first)
        student_homework.sort(key=lambda x: x.get("due_date") or "", reverse=True)
        
        return jsonify({"homework": student_homework})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
