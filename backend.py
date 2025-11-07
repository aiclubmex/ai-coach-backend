# backend.py
# Flask API for Physics AI Coach
# - GET /                -> health
# - GET /problems        -> list problems from Notion
# - GET /problem/<id>    -> full fields for one problem
# - GET /steps/<id>      -> step-by-step plan (Notion or auto-LLM)
# - POST /chat           -> checks a student's step / next hint

import os, re, json
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# ==============================
# ENV
# ==============================
NOTION_API_KEY      = os.environ.get("NOTION_API_KEY", "")
NOTION_DB_ID        = os.environ.get("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
FRONTEND_ORIGIN     = os.environ.get("FRONTEND_ORIGIN", "https://aiclub.com.mx")

# ==============================
# Flask
# ==============================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://aiclub.com.mx", "http://aiclub.com.mx"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
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
    
    return out

def query_database(database_id: str, filter_obj: dict = None) -> List[Dict[str, Any]]:
    """Query a Notion database."""
    db_id = sanitize_uuid(database_id)
    if not db_id:
        raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/databases/{db_id}/query"
    payload = {}
    if filter_obj:
        payload["filter"] = filter_obj
    
    resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    for page in data.get("results", []):
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
        results.append(obj)
    
    return results

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
        
        # Remove numbering like "1)", "1.", "Step 1:", etc.
        clean_ln = re.sub(r'^(\d+[\)\.:]?\s*|Step\s+\d+[\)\.:]?\s*)', '', ln, flags=re.IGNORECASE).strip()
        
        # Only add if there's actual content (not just a number)
        if clean_ln and len(clean_ln) > 2:
            steps.append({
                "id": str(i),
                "description": clean_ln,
                "rubric": ""
            })
    
    # If we got steps but they're all too short (just numbers), return empty
    if steps and all(len(s["description"]) < 5 for s in steps):
        return []
    
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
        resp = requests.post(url, headers=headers, json=payload, timeout=60)  # Increased to 60 seconds
        
        # Log status and response for debugging
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
        
        # Clean up the response
        text = text.strip()
        
        # Remove any markdown code blocks
        if "```" in text:
            # Extract content between ``` markers
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                # Remove language identifier like "json" or "text"
                lines = text.split("\n")
                if lines[0].strip() in ["json", "text", "plaintext", ""]:
                    text = "\n".join(lines[1:])
            else:
                # Just remove the markers
                text = text.replace("```", "")
        
        # Remove any JSON-like content
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
        # Fallback to generic steps
        return [
            {"id": "1", "description": "Identify the given data and what needs to be found", "rubric": ""},
            {"id": "2", "description": "Determine which physics principles and formulas apply", "rubric": ""},
            {"id": "3", "description": "Set up the equation with the given values", "rubric": ""},
            {"id": "4", "description": "Solve for the unknown variable", "rubric": ""},
            {"id": "5", "description": "Check units and verify the answer makes physical sense", "rubric": ""},
        ]

# ==============================
# Routes
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
        steps_text = page.get("step_by_step", "")
        
        # Try to parse steps from Notion
        steps = parse_steps(steps_text) if steps_text else []
        
        # If no valid steps found, generate with LLM
        if not steps:
            print(f"[STEPS] No valid steps in Notion, generating with LLM...")
            steps = generate_steps_with_llm(page)
        else:
            print(f"[STEPS] Found {len(steps)} steps from Notion")
        
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
        
        # Get steps
        steps_text = problem.get("step_by_step", "")
        steps_list = parse_steps(steps_text) if steps_text else []
        
        # If no valid steps, generate with LLM
        if not steps_list:
            print(f"[SOLUTION] No valid steps found, generating with LLM...")
            steps_list = generate_steps_with_llm(problem)
        
        # Build full solution text from steps
        full_solution_lines = []
        for i, step in enumerate(steps_list, 1):
            full_solution_lines.append(f"{i}. {step.get('description', step.get('text', ''))}")
        
        full_solution = "\n\n".join(full_solution_lines)
        
        # Get final answer
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
    solution_image_base64 = data.get("solution_image_base64")  # For future Phase 3
    time_spent_seconds = data.get("time_spent_seconds", 0)
    
    print(f"[SUBMIT] problem_id={problem_id}, time={time_spent_seconds}s")
    
    if not problem_id:
        return jsonify({"error": "missing problem_id"}), 400
    
    if not solution_text and not solution_image_base64:
        return jsonify({"error": "No solution provided"}), 400
    
    # Fetch problem
    try:
        problem = fetch_page(problem_id)
    except Exception as e:
        return jsonify({"error": f"Cannot read problem: {e}"}), 500
    
    # Get official answer
    official_answer = problem.get("final_answer", "")
    problem_statement = problem.get("statement", "")
    problem_title = problem.get("title") or problem.get("name", "")
    
    # Check if AI is available
    if not ANTHROPIC_API_KEY:
        # Fallback: basic length check
        is_correct = len(solution_text) > 50
        return jsonify({
            "correct": is_correct,
            "score": 50 if is_correct else 0,
            "feedback": "AI evaluation not available. Your answer was recorded.",
            "time_taken": f"{time_spent_seconds // 60}:{time_spent_seconds % 60:02d}"
        })
    
    # Build evaluation prompt
    eval_prompt = f"""You are an IB Physics HL examiner evaluating a student's complete solution.

Problem: {problem_title}
Statement: {problem_statement}

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
    
    try:
        response_text = call_anthropic_api(eval_prompt, max_tokens=1024)
        
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        response_text = response_text.strip()
        
        # Extract JSON with regex
        json_match = re.search(r'\{[^{}]*"correct"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        # Parse JSON
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as je:
            print(f"[ERROR] JSON parse failed: {je}")
            print(f"[ERROR] Response was: {response_text[:200]}")
            # Fallback
            parsed = {
                "correct": False,
                "score": 30,
                "feedback": "Your solution was recorded. Consider reviewing the key concepts and trying again.",
                "specific_mistakes": []
            }
        
        # Format time
        minutes = time_spent_seconds // 60
        seconds = time_spent_seconds % 60
        time_str = f"{minutes}:{seconds:02d}"
        
        return jsonify({
            "correct": bool(parsed.get("correct", False)),
            "score": int(parsed.get("score", 0)),
            "feedback": parsed.get("feedback", ""),
            "specific_mistakes": parsed.get("specific_mistakes", []),
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
    
    # Fetch problem
    try:
        problem = fetch_page(problem_id)
    except Exception as e:
        return jsonify({"error": f"Cannot read problem: {e}"}), 500
    
    # Get steps
    steps_text = problem.get("step_by_step", "")
    steps_list = parse_steps(steps_text) if steps_text else []
    
    # If no valid steps, generate with LLM
    if not steps_list:
        print(f"[CHAT] No valid steps found, generating with LLM...")
        steps_list = generate_steps_with_llm(problem)
    
    # Find current step
    current_step = None
    for s in steps_list:
        if s["id"] == step_id:
            current_step = s
            break
    
    if not current_step and steps_list:
        current_step = steps_list[0]
        step_id = current_step["id"]
    
    # Check if API is available
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "ok": True,
            "feedback": "AI Coach is not configured. Your answer was recorded.",
            "next_step": str(int(step_id) + 1) if step_id.isdigit() else step_id,
            "message": "Continue to next step.",
            "reply": "AI Coach is not configured."
        })
    
    # MODE: Free chat (student asking questions)
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
    
    # MODE: Hint
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
    
    # MODE: Tutor (evaluate answer)
    if not student_answer:
        return jsonify({
            "ok": False,
            "feedback": "Please provide an answer to evaluate.",
            "next_step": step_id,
            "message": "Write your reasoning above."
        })
    
    # Build evaluation prompt
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
        
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        response_text = response_text.strip()
        
        # Try to extract JSON with regex
        json_match = re.search(r'\{[^{}]*"ok"[^{}]*"feedback"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        # Parse JSON
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as je:
            print(f"[ERROR] JSON parse failed: {je}")
            print(f"[ERROR] Response was: {response_text[:200]}")
            # Fallback response
            parsed = {
                "ok": len(student_answer) > 15,
                "feedback": "Your answer was recorded. Try to be more specific with physics concepts.",
                "ready_to_advance": len(student_answer) > 15,
                "next_hint": "Explain which formulas or principles apply."
            }
        
        # Determine next step
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

# ==============================
# Gunicorn entry
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
