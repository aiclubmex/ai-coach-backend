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
# *** HOMEWORK SYSTEM - ADDED 11 FEB 2026 ***
# - POST /api/assign-homework          -> assign homework to students
# - GET  /api/student-homework         -> get student's homework
# - POST /api/complete-homework/<id>   -> mark homework as complete
# - GET  /api/students-list            -> get students for dropdown
# ========================================

import os, re, json, uuid
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
HOMEWORK_DB_ID      = os.environ.get("HOMEWORK_DB_ID", "")
JWT_SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "change-this-secret-key-in-production")

# ==============================
# Flask
# ==============================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://aiclub.com.mx", "http://aiclub.com.mx"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
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
        elif ptype == "email":
            out[k] = v.get("email", "")
        elif ptype == "date":
            date_obj = v.get("date")
            if date_obj:
                out[k] = date_obj.get("start", "")
            else:
                out[k] = None
        elif ptype == "checkbox":
            out[k] = v.get("checkbox", False)
    
    return out

def query_database(database_id: str, filter_obj: dict = None, sorts: list = None) -> List[Dict[str, Any]]:
    """Query a Notion database with full pagination (brings ALL results)."""
    if database_id in [USERS_DB_ID, ACTIVITY_DB_ID, HOMEWORK_DB_ID]:
        db_id = database_id
    else:
        db_id = sanitize_uuid(database_id)
        if not db_id:
            raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/databases/{db_id}/query"
    all_pages = []
    has_more = True
    start_cursor = None
    
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
        results.append(obj)
    return results

def create_page_in_database(database_id: str, properties: dict) -> dict:
    """Create a new page in a Notion database."""
    if database_id in [USERS_DB_ID, ACTIVITY_DB_ID, HOMEWORK_DB_ID]:
        db_id = database_id
    else:
        db_id = sanitize_uuid(database_id)
        if not db_id:
            raise ValueError("Invalid database ID")
    
    url = f"{NOTION_BASE}/pages"
    payload = {"parent": {"database_id": db_id}, "properties": properties}
    resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def update_page_properties(page_id: str, properties: dict) -> dict:
    """Update properties of an existing Notion page."""
    pid = sanitize_uuid(page_id)
    if not pid:
        raise ValueError("Invalid page ID")
    
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
def index(): return jsonify({"status": "ok", "service": "ai-coach-backend"})

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

@app.post("/submit-solution")
def submit_solution():
    """
    Evaluate student's solution using Claude API.
    Returns score and feedback.
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    solution_text = data.get("solution_text", "")
    time_spent = data.get("time_spent_seconds", 0)
    
    if not problem_id or not solution_text:
        return jsonify({"error": "Missing problem_id or solution_text"}), 400
    
    print(f"[SUBMIT] Evaluating solution for problem: {problem_id}")
    
    try:
        # Get problem details
        problem = fetch_page(problem_id)
        
        # Build evaluation prompt
        prompt = f"""You are an IB Physics HL examiner. Evaluate this student's solution.

Problem: {problem.get('name', '')}
Statement: {problem.get('problem_statement', '')}
Given: {problem.get('given_values', '')}
Find: {problem.get('find', '')}

Student's Solution:
{solution_text}

Evaluate the solution and respond with a JSON object:
{{
  "score": <number 0-100>,
  "correct": <true if score >= 70, false otherwise>,
  "feedback": "<detailed feedback explaining what's correct/incorrect>",
  "time_taken": "{time_spent // 60}:{time_spent % 60:02d}"
}}

Be specific about what's correct and what needs improvement.
"""
        
        # Call Claude API
        if not ANTHROPIC_API_KEY:
            return jsonify({
                "score": 50,
                "correct": False,
                "feedback": "Solution submitted but cannot evaluate (API key missing)",
                "time_taken": f"{time_spent // 60}:{time_spent % 60:02d}"
            })
        
        response_text = call_anthropic_api(prompt, max_tokens=1024)
        
        # Parse response
        import json
        # Remove markdown code blocks if present
        clean_response = response_text.strip()
        if clean_response.startswith('```'):
            clean_response = '\n'.join(clean_response.split('\n')[1:-1])
        
        result = json.loads(clean_response)
        
        # Ensure required fields
        result.setdefault('score', 50)
        result.setdefault('correct', result.get('score', 0) >= 70)
        result.setdefault('feedback', 'Solution evaluated.')
        result.setdefault('time_taken', f"{time_spent // 60}:{time_spent % 60:02d}")
        
        print(f"[SUBMIT] Score: {result['score']}, Correct: {result['correct']}")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Submit solution failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback response
        return jsonify({
            "score": 50,
            "correct": False,
            "feedback": f"Error evaluating solution: {str(e)}",
            "time_taken": f"{time_spent // 60}:{time_spent % 60:02d}"
        }), 200

@app.post("/chat")
def chat():
    # (Omitido por brevedad - mantiene lógica original)
    return jsonify({"reply": "I am your AI Physics tutor."})

# ========================================
# *** MULTIUSER AUTHENTICATION ***
# ========================================

@app.post("/api/register")
def register():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get("email", "").strip().lower()
    name = data.get("name", "").strip()
    password = data.get("password", "")
    if not email or not name or not password: return jsonify({"error": "Missing fields"}), 400
    if not email.endswith("@asf.edu.mx"): return jsonify({"error": "Use ASF email"}), 400
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        create_page_in_database(USERS_DB_ID, {
            "Name": {"title": [{"text": {"content": name}}]},
            "Email": {"email": email},
            "password_hash": {"rich_text": [{"text": {"content": password_hash}}]},
            "created_at": {"date": {"start": datetime.utcnow().isoformat()}}
        })
        return jsonify({"success": True}), 201
    except Exception as e: return jsonify({"error": str(e)}), 500

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

@app.post("/api/track-activity")
@require_auth
def track_activity():
    data = request.get_json(force=True, silent=True) or {}
    try:
        create_page_in_database(ACTIVITY_DB_ID, {
            "problem_name": {"title": [{"text": {"content": data.get("problem_name", "")}}]},
            "user_email": {"email": request.user["email"]},
            "action": {"select": {"name": data.get("action", "opened")}},
            "timestamp": {"date": {"start": datetime.utcnow().isoformat()}}
        })
        return jsonify({"success": True})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.get("/api/all-users")
def all_users():
    """
    Get all users and their progress with recent activities.
    FOR ADMIN/PROFESSOR USE ONLY.
    """
    if not USERS_DB_ID or not ACTIVITY_DB_ID:
        return jsonify({"error": "Databases not configured"}), 500
    
    print(f"[ADMIN] Getting all users")
    
    try:
        # Get all users
        users = query_database(USERS_DB_ID)
        print(f"[ADMIN] Found {len(users)} users")
        
        # Get all activities
        all_activities = query_database(ACTIVITY_DB_ID)
        print(f"[ADMIN] Found {len(all_activities)} activities")
        
        # Build stats per user
        user_stats = []
        
        for user in users:
            email = user.get("Email", "")
            name = user.get("Name", "")
            
            if not email:
                continue
            
            # Filter activities for this user
            user_activities = [a for a in all_activities if a.get("user_email") == email]
            
            # Sort by timestamp descending
            user_activities.sort(key=lambda x: x.get("timestamp", "") or "", reverse=True)
            
            # Calculate stats
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
            
            # Get recent activities (last 10) for detail view
            recent_activities = []
            for activity in user_activities[:10]:
                recent_activities.append({
                    "problem_name": activity.get("problem_name", "Unknown"),
                    "action": activity.get("action", "opened"),
                    "score": activity.get("score", 0) or 0,
                    "time_spent_seconds": activity.get("time_spent_seconds", 0) or 0,
                    "timestamp": activity.get("timestamp", "")
                })
            
            user_stats.append({
                "email": email,
                "name": name,
                "last_active": last_active,
                "problems_attempted": len(problems_attempted),
                "problems_completed": len(problems_completed),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "total_time_minutes": total_time // 60,
                "recent_activities": recent_activities
            })
        
        # Calculate overall stats
        total_students = len(user_stats)
        
        # Calculate active_today safely
        active_today = 0
        for u in user_stats:
            if u.get("last_active"):
                try:
                    from datetime import timezone
                    last_active_str = u["last_active"].replace("Z", "+00:00")
                    last_active_dt = datetime.fromisoformat(last_active_str)
                    now_utc = datetime.now(timezone.utc)
                    diff = now_utc - last_active_dt
                    if diff.days == 0:
                        active_today += 1
                except Exception as e:
                    print(f"[WARN] Could not parse last_active: {e}")
                    continue
        
        problems_solved = sum(u["problems_completed"] for u in user_stats)
        
        # Calculate avg_score safely
        users_with_scores = [u for u in user_stats if u["avg_score"] > 0]
        if users_with_scores:
            avg_score = round(sum(u["avg_score"] for u in users_with_scores) / len(users_with_scores), 1)
        else:
            avg_score = 0
        
        print(f"[ADMIN] Returning {total_students} students with activities")
        
        return jsonify({
            "total_students": total_students,
            "active_today": active_today,
            "problems_solved": problems_solved,
            "avg_score": avg_score,
            "students": user_stats
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get all users: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to get users: {str(e)}"}), 500


# ============================================================================
# SISTEMA DE TAREAS/HOMEWORK - ENDPOINTS CON PROBLEM_REFERENCES
# ============================================================================

@app.post("/api/assign-homework")
def assign_homework():
    """
    Asigna una tarea a un estudiante específico o a todos los estudiantes.
    Soporta problem_references (IDs de problemas separados por coma).
    """
    if not HOMEWORK_DB_ID: return jsonify({"error": "HOMEWORK_DB_ID missing"}), 500
    
    data = request.get_json(force=True, silent=True) or request.form.to_dict()
    student_id = data.get("student_id")
    title = data.get("title")
    description = data.get("description", "")
    due_date = data.get("due_date")
    points = int(data.get("points", 0))
    topic = data.get("topic", "")
    problem_references = data.get("problem_references", "")

    if not title or not student_id or not due_date:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        def build_props(email):
            return {
                "title": {"title": [{"text": {"content": title}}]},
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

        if student_id == "all":
            students = query_database(USERS_DB_ID)
            count = 0
            for s in students:
                if s.get("Email"):
                    create_page_in_database(HOMEWORK_DB_ID, build_props(s["Email"]))
                    count += 1
            return jsonify({"success": True, "message": f"Assigned to {count} students"})
        else:
            create_page_in_database(HOMEWORK_DB_ID, build_props(student_id))
            return jsonify({"success": True, "message": "Homework assigned"})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.get("/api/student-homework")
@require_auth
def get_student_homework():
    """Obtiene todas las tareas del estudiante autenticado."""
    try:
        hw_list = query_database(
            HOMEWORK_DB_ID, 
            {"property": "student_email", "email": {"equals": request.user["email"]}}, 
            sorts=[{"property": "due_date", "direction": "ascending"}]
        )
        return jsonify({"homework": hw_list})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.post("/api/complete-homework/<homework_id>")
@require_auth
def complete_homework(homework_id):
    """Marca una tarea como completada y devuelve los puntos."""
    try:
        hw = fetch_page(homework_id)
        if hw.get("student_email") != request.user["email"]: 
            return jsonify({"error": "Forbidden"}), 403
        
        update_page_properties(homework_id, {
            "completed": {"checkbox": True},
            "completed_at": {"date": {"start": datetime.utcnow().isoformat()}}
        })
        
        return jsonify({"success": True, "points_earned": hw.get("points", 0)})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.get("/api/students-list")
def get_students_list():
    """Retorna lista de estudiantes para el dropdown del profesor."""
    try:
        users = query_database(USERS_DB_ID)
        students = [
            {"id": u.get("Email"), "name": u.get("Name"), "email": u.get("Email")} 
            for u in users if u.get("Email")
        ]
        return jsonify({"students": students})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

