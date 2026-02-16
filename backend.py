# backend.py - AI Coach Physics Backend v2
# ==========================================
# ACTUALIZADO: 16 Feb 2026
# - Agregado: solution_text en track-activity
# - Agregado: time_limit_minutes en assign-homework
# - Agregado: late_submission detection
# ==========================================

import os, re, json, uuid
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from functools import wraps
from datetime import datetime, timedelta, timezone

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

def query_database(database_id: str, filter_obj: dict = None, sorts: list = None) -> List[Dict[str, Any]]:
    if database_id in [USERS_DB_ID, ACTIVITY_DB_ID, HOMEWORK_DB_ID]:
        db_id = database_id
    else:
        db_id = sanitize_uuid(database_id)
        if not db_id: raise ValueError("Invalid database ID")
    
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

@app.post("/submit-solution")
def submit_solution():
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    solution_text = data.get("solution_text", "")
    time_spent = data.get("time_spent_seconds", 0)
    
    if not problem_id or not solution_text:
        return jsonify({"error": "Missing problem_id or solution_text"}), 400
    
    print(f"[SUBMIT] Evaluating solution for problem: {problem_id}")
    
    try:
        problem = fetch_page(problem_id)
        
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
"""
        
        if not ANTHROPIC_API_KEY:
            return jsonify({
                "score": 50,
                "correct": False,
                "feedback": "Solution submitted but cannot evaluate (API key missing)",
                "time_taken": f"{time_spent // 60}:{time_spent % 60:02d}"
            })
        
        response_text = call_anthropic_api(prompt, max_tokens=1024)
        
        clean_response = response_text.strip()
        if clean_response.startswith('```'):
            clean_response = '\n'.join(clean_response.split('\n')[1:-1])
        
        result = json.loads(clean_response)
        result.setdefault('score', 50)
        result.setdefault('correct', result.get('score', 0) >= 70)
        result.setdefault('feedback', 'Solution evaluated.')
        result.setdefault('time_taken', f"{time_spent // 60}:{time_spent % 60:02d}")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] Submit solution failed: {e}")
        return jsonify({
            "score": 50,
            "correct": False,
            "feedback": f"Error evaluating solution: {str(e)}",
            "time_taken": f"{time_spent // 60}:{time_spent % 60:02d}"
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
    
    try:
        create_page_in_database(ACTIVITY_DB_ID, properties)
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
        
        recent_activity = []
        for activity in activities[:10]:
            if activity.get("action") in ["completed", "submitted"]:
                display_name = activity.get("problem_reference") or activity.get("problem_name", "Unknown")
                recent_activity.append({
                    "problem_name": display_name,
                    "score": activity.get("score", 0) or 0,
                    "timestamp": activity.get("timestamp", ""),
                    "action": activity.get("action", "")
                })
        
        return jsonify({
            "solved_count": len(problems_completed),
            "average_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "total_time_seconds": total_time,
            "recent_activity": recent_activity
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
    UPDATED: Now includes solution_text in recent_activities.
    """
    if not USERS_DB_ID or not ACTIVITY_DB_ID:
        return jsonify({"error": "Databases not configured"}), 500
    
    try:
        users = query_database(USERS_DB_ID)
        all_activities = query_database(ACTIVITY_DB_ID)
        
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
            for activity in user_activities[:10]:
                display_name = activity.get("problem_reference") or activity.get("problem_name", "Unknown")
                recent_activities.append({
                    "problem_name": display_name,
                    "action": activity.get("action", "opened"),
                    "score": activity.get("score", 0) or 0,
                    "time_spent_seconds": activity.get("time_spent_seconds", 0) or 0,
                    "timestamp": activity.get("timestamp", ""),
                    "solution_text": activity.get("solution_text", "")  # NEW
                })
            
            user_stats_list.append({
                "email": email,
                "name": name,
                "group": group,
                "last_active": last_active,
                "problems_attempted": len(problems_attempted),
                "problems_completed": len(problems_completed),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "total_time_minutes": total_time // 60,
                "recent_activities": recent_activities
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
        
        return jsonify({
            "total_students": total_students,
            "active_today": active_today,
            "problems_solved": problems_solved,
            "avg_score": avg_score,
            "students": user_stats_list
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Failed to get all users: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to get users: {str(e)}"}), 500

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
    FIXED: Always calculates solved_count and total_problems.
    """
    try:
        student_email = request.user["email"]
        
        hw_list = query_database(
            HOMEWORK_DB_ID, 
            {"property": "student_email", "email": {"equals": student_email}}, 
            sorts=[{"property": "due_date", "direction": "ascending"}]
        )
        
        # Get ALL completed activities for this student
        activities = query_database(
            ACTIVITY_DB_ID,
            {"property": "user_email", "email": {"equals": student_email}}
        )
        
        # Build set of completed problem references
        completed_problems = set()
        for act in activities:
            if act.get("action") == "completed":
                ref = act.get("problem_reference") or act.get("problem_name")
                if ref:
                    completed_problems.add(ref)
        
        print(f"[HW] Student {student_email} has completed: {completed_problems}")
        
        processed_homework = []
        now = datetime.now(timezone.utc)
        
        for hw in hw_list:
            problem_refs_str = hw.get("problem_references", "") or ""
            
            # ALWAYS calculate progress, regardless of completed flag
            refs = [r.strip() for r in problem_refs_str.split(",") if r.strip()]
            total_problems = len(refs)
            solved_count = sum(1 for ref in refs if ref in completed_problems)
            
            # Determine completion: ALL problems must be solved
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
                "total_problems": total_problems
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
