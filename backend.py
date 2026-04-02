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
from collections import defaultdict, Counter
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
# Oxford Physics 2023 — Deep Link Reference Map
# Maps IB 2025 topic codes → Oxford section + page range
# Used to give students precise page references when they make errors
# ==============================
OXFORD_REFERENCES = {
    "A.1": {"section": "Chapter 1 — Kinematics",            "pages": "2–35",   "data_booklet": "p.4 (kinematic equations)"},
    "A.2": {"section": "Chapter 2 — Forces & Momentum",     "pages": "36–89",  "data_booklet": "p.4 (Newton's laws, momentum)"},
    "A.3": {"section": "Chapter 3 — Work, Energy & Power",  "pages": "90–125", "data_booklet": "p.5 (energy equations)"},
    "A.4": {"section": "Chapter 4 — Rigid Body Mechanics",  "pages": "126–161","data_booklet": "p.5 (rotational equations)"},
    "A.5": {"section": "Chapter 5 — Special Relativity",    "pages": "162–195","data_booklet": "p.6 (Lorentz factor)"},
    "B.1": {"section": "Chapter 8 — Thermal Energy",        "pages": "270–305","data_booklet": "p.6 (specific heat, latent heat)"},
    "B.2": {"section": "Chapter 9 — Greenhouse Effect",     "pages": "306–325","data_booklet": "p.6 (Stefan-Boltzmann, Wien's law)"},
    "B.3": {"section": "Chapter 10 — Gas Laws",             "pages": "326–355","data_booklet": "p.6 (ideal gas law, Boltzmann)"},
    "B.4": {"section": "Chapter 11 — Thermodynamics",       "pages": "356–395","data_booklet": "p.6 (1st & 2nd law, entropy)"},
    "B.5": {"section": "Chapter 7 — Electric Circuits",     "pages": "230–269","data_booklet": "p.5 (Ohm's law, power, Kirchhoff)"},
    "C.1": {"section": "Chapter 12 — Simple Harmonic Motion","pages":"396–427","data_booklet": "p.7 (SHM equations)"},
    "C.2": {"section": "Chapter 13 — Wave Model",           "pages": "428–457","data_booklet": "p.7 (wave speed, frequency)"},
    "C.3": {"section": "Chapter 14 — Wave Phenomena",       "pages": "458–499","data_booklet": "p.7 (diffraction, single-slit, Rayleigh)"},
    "C.4": {"section": "Chapter 15 — Standing Waves",       "pages": "500–527","data_booklet": "p.7 (harmonics, resonance)"},
    "C.5": {"section": "Chapter 16 — Doppler Effect",       "pages": "528–545","data_booklet": "p.7 (Doppler equations)"},
    "D.1": {"section": "Chapter 17 — Gravitational Fields", "pages": "546–585","data_booklet": "p.8 (gravitational potential, Kepler)"},
    "D.2": {"section": "Chapter 18 — Electric & Magnetic Fields","pages":"586–635","data_booklet": "p.8 (Coulomb, magnetic force)"},
    "D.3": {"section": "Chapter 19 — Motion in EM Fields",  "pages": "636–665","data_booklet": "p.9 (Lorentz force, cyclotron)"},
    "D.4": {"section": "Chapter 20 — Electromagnetic Induction","pages":"666–703","data_booklet": "p.9 (Faraday, Lenz, flux linkage)"},
    "E.1": {"section": "Chapter 21 — Atomic Structure",     "pages": "704–735","data_booklet": "p.9 (Bohr model, Rutherford)"},
    "E.2": {"section": "Chapter 22 — Quantum Physics",      "pages": "736–771","data_booklet": "p.10 (Planck, de Broglie, photoelectric)"},
    "E.3": {"section": "Chapter 23 — Radioactive Decay",    "pages": "772–809","data_booklet": "p.10 (decay constant, half-life, binding energy)"},
    "E.4": {"section": "Chapter 24 — Nuclear Fission",      "pages": "810–833","data_booklet": "p.10 (fission, chain reaction)"},
    "E.5": {"section": "Chapter 25 — Stellar Properties",   "pages": "834–875","data_booklet": "p.11 (HR diagram, stellar parallax, luminosity)"},
}

def get_oxford_reference(topic: str) -> str:
    """Returns a formatted Oxford reference string for a given IB topic code."""
    if not topic:
        return ""
    # Handle multi_select arrays stored as lists
    if isinstance(topic, list):
        topic = topic[0] if topic else ""
    topic = str(topic).strip()
    # Match exact code (A.1, B.4, etc.) or first two chars
    ref = OXFORD_REFERENCES.get(topic) or OXFORD_REFERENCES.get(topic[:3] if len(topic) >= 3 else "")
    if not ref:
        return ""
    return (
        f"Oxford Physics 2023 Reference: {ref['section']}, pages {ref['pages']}. "
        f"Data Booklet: {ref['data_booklet']}."
    )


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
CACHE_TTL = 60  # seconds (default for most caches)
PROBLEMS_CACHE_TTL = 1800  # 30 minutes for problems (they rarely change)

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
    """Clear cache entries matching prefix. Never clears problems cache (use /api/refresh-problems)."""
    if not prefix:
        # Save problems cache and restore after clear
        problems_cache = _cache.get("all-problems")
        _cache.clear()
        if problems_cache:
            _cache["all-problems"] = problems_cache
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
    statement = problem.get("problem_statement", "") or problem.get("statement", "")
    given = problem.get("given_values", "")
    find = problem.get("find", "")
    solution = problem.get("full_solution", "") or problem.get("step_by_step", "") or problem.get("final_answer", "")
    marks = problem.get("marks", "")

    prompt = f"""Break down this IB Physics HL problem into clear, specific steps that guide a student to the solution.
Each step must reference the ACTUAL values and equations from this specific problem.

Problem: {title}
Statement: {statement}
Given values: {given}
What to find: {find}
Marks: {marks}
{('Reference solution: ' + solution) if solution else ''}

Write 4-8 numbered steps. Each step should:
- Tell the student exactly what to calculate or identify
- Reference specific values from the problem (not generic placeholders)
- Include the relevant equation or principle

Format: One numbered step per line. No JSON, no markdown, just plain numbered lines.
Example:
1. Identify the given values: mass m = 2.5 kg, velocity v = 3.0 m/s
2. Apply the kinetic energy formula: Ek = ½mv²
3. Calculate: Ek = ½(2.5)(3.0)² = 11.25 J
"""
    try:
        if not ANTHROPIC_API_KEY:
            return [{"id": "1", "description": "Identify the given values from the problem.", "rubric": ""}]
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "temperature": 0,
            "system": "You are an expert IB Physics HL tutor. Generate clear, specific solution steps. Respond with numbered lines only — no JSON, no markdown code blocks, no backticks.",
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        text = resp.json().get("content", [])[0].get("text", "")
        steps = parse_steps(text)
        return steps if steps else [{"id": "1", "description": "Analyze the problem statement and identify given values.", "rubric": ""}]
    except Exception as e:
        print(f"[WARNING] generate_steps_with_llm failed: {e}")
        return [{"id": "1", "description": "Identify the given data and what you need to find.", "rubric": ""}]

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


def get_professor_groups(professor_group: str) -> list:
    """
    Devuelve lista de grupos de un profesor.
    Soporta múltiples grupos separados por coma.
    Ej: "ASF-PHY-12A,ASF-PHY-12B" → ["ASF-PHY-12A", "ASF-PHY-12B"]
    """
    if not professor_group:
        return []
    return [g.strip() for g in professor_group.split(",") if g.strip()]


def professor_can_see_group(professor_group: str, student_group: str) -> bool:
    """
    Verifica si un profesor puede ver a un alumno según sus grupos.
    Retorna True si el grupo del alumno está en la lista de grupos del profesor.
    """
    if not professor_group or not student_group:
        return False
    return student_group.strip() in get_professor_groups(professor_group)


# ==============================
# Base Routes
# ==============================
@app.get("/")
def index(): return jsonify({"status": "ok", "service": "ai-coach-backend", "version": "2.0"})

@app.get("/problems")
def list_problems():
    try:
        # Check cache first (30 min TTL — problems rarely change)
        cached = _cache.get("all-problems")
        if cached and time.time() - cached["time"] < PROBLEMS_CACHE_TTL:
            return jsonify({"problems": cached["data"], "cached": True})
        
        items = query_database(NOTION_DB_ID)
        
        # Cache the results
        _cache["all-problems"] = {"data": items, "time": time.time()}
        
        return jsonify({"problems": items, "cached": False})
    except Exception as e: return jsonify({"error": str(e)}), 500


@app.get("/api/refresh-problems")
def refresh_problems():
    """Force refresh the problems cache. Call after adding/editing problems in Notion."""
    try:
        # Clear problems cache
        if "all-problems" in _cache:
            del _cache["all-problems"]
        
        # Reload from Notion
        items = query_database(NOTION_DB_ID)
        _cache["all-problems"] = {"data": items, "time": time.time()}
        
        return jsonify({"success": True, "count": len(items), "message": f"Refreshed {len(items)} problems"})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.get("/problem/<problem_id>")
def get_problem(problem_id: str):
    try: return jsonify({"problem": fetch_page(problem_id)})
    except Exception as e: return jsonify({"error": str(e)}), 500


# ========================================
# FAST PROBLEM LOOKUP BY REFERENCE (for Study Plan)
# ========================================
@app.get("/api/problem-by-ref")
def problem_by_ref():
    """
    Fast lookup: find a problem by its problem_reference field.
    Used by Study Plan Tracker to load a specific problem without fetching all 497.
    Returns the full problem data ready for the chatbot.
    """
    ref = request.args.get("ref", "").strip()
    if not ref:
        return jsonify({"error": "ref parameter required"}), 400
    
    try:
        # Search Notion for this specific reference
        results = query_database(
            NOTION_DB_ID,
            filter_obj={"property": "problem_reference", "rich_text": {"equals": ref}},
            max_pages=1
        )
        if results:
            # Fetch full page data
            page = fetch_page(results[0]["id"])
            return jsonify({"problem": page}), 200
        
        # Fallback: try partial match (e.g., ref might have slight format differences)
        results = query_database(
            NOTION_DB_ID,
            filter_obj={"property": "problem_reference", "rich_text": {"contains": ref}},
            max_pages=1
        )
        if results:
            page = fetch_page(results[0]["id"])
            return jsonify({"problem": page}), 200
        
        return jsonify({"error": f"Problem with reference '{ref}' not found"}), 404
    except Exception as e:
        print(f"[ERROR] problem-by-ref: {e}")
        return jsonify({"error": str(e)}), 500


# ========================================
# STEP HINT — Claude gives a pedagogical hint for current step
# ========================================
@app.post("/api/step-hint")
def step_hint():
    """
    Give a hint for the current step without revealing the answer.
    Used in step-by-step mode when student is stuck.
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id", "")
    step_text = data.get("step_text", "")
    step_number = data.get("step_number", 1)
    total_steps = data.get("total_steps", 1)
    student_attempt = data.get("student_attempt", "")
    
    if not problem_id or not step_text:
        return jsonify({"error": "Missing problem_id or step_text"}), 400
    
    try:
        problem = fetch_page(problem_id)
        
        hint_prompt = f"""You are an expert IB Physics HL tutor. A student is stuck on step {step_number} of {total_steps} while solving a problem.

PROBLEM: {problem.get('name', '')}
Statement: {problem.get('problem_statement', '')}
Given values: {problem.get('given_values', '')}
What to find: {problem.get('find', '')}
Topic: {problem.get('topic', '')}

THE STEP THEY'RE STUCK ON:
{step_text}

{"THEIR ATTEMPT SO FAR: " + student_attempt if student_attempt else "They haven't written anything yet."}

Give a HINT that helps them think through this step WITHOUT giving the answer directly.
Your hint should:
- Ask a guiding QUESTION that leads the student to think (not tell them the answer)
- Name the relevant physics concept but ask THEM to recall the equation
- Reference specific values from the problem they should look at
- Be encouraging — "Think about..." or "What happens when..." style
- NEVER state the equation or calculation directly — make them think

Keep your hint to 2-3 sentences maximum. Be specific to THIS problem, not generic.

Respond with ONLY valid JSON (no markdown, no backticks):
{{
  "hint": "<your hint text>",
  "concept": "<the key physics concept for this step>"
}}"""

        if not ANTHROPIC_API_KEY:
            return jsonify({"hint": "Think about what physics principle applies to this step. What equation connects the given values?", "concept": ""}), 200
        
        response_text = call_anthropic_api(hint_prompt, max_tokens=400)
        clean = response_text.strip()
        if clean.startswith('```'): clean = '\n'.join(clean.split('\n')[1:-1])
        result = json.loads(clean)
        
        return jsonify({
            "hint": result.get("hint", "Think about what equation applies here."),
            "concept": result.get("concept", "")
        }), 200
        
    except Exception as e:
        print(f"[ERROR] step-hint: {e}")
        return jsonify({"hint": "Try to identify which physics equation connects the values given in this step.", "concept": ""}), 200


# ========================================
# AI COACH GUIDE — Socratic conversational tutor
# ========================================
@app.post("/api/coach-guide")
def coach_guide():
    """
    Socratic tutoring: Claude guides the student through the problem with questions,
    never giving the answer directly. Uses the official solution as hidden reference.
    
    Receives: problem_id, conversation (array of {role: 'coach'|'student', text})
    Returns: {message, is_complete, progress}
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id", "")
    conversation = data.get("conversation", [])
    
    if not problem_id:
        return jsonify({"error": "Missing problem_id"}), 400
    
    try:
        problem = fetch_page(problem_id)
        
        # Get the official solution (HIDDEN from student — only Claude sees it)
        official_solution = problem.get("full_solution", "") or problem.get("step_by_step", "") or problem.get("final_answer", "")
        
        # Build conversation history for Claude
        conv_text = ""
        for turn in conversation:
            role_label = "TUTOR" if turn.get("role") == "coach" else "STUDENT"
            conv_text += f"\n{role_label}: {turn.get('text', '')}"
        
        turn_number = len([t for t in conversation if t.get("role") == "student"]) + 1
        
        coach_prompt = f"""You are an expert IB Physics HL tutor using the Socratic method. 
Your job is to guide the student to discover the answer through questions — NEVER give the answer directly.

PROBLEM (student can see this):
Name: {problem.get('name', '')}
Statement: {problem.get('problem_statement', '')}
Given values: {problem.get('given_values', '')}
What to find: {problem.get('find', '')}
Topic: {problem.get('topic', '')}
Marks: {problem.get('marks', '')}

OFFICIAL SOLUTION (HIDDEN — only you can see this, NEVER reveal it):
{official_solution}

CONVERSATION SO FAR:{conv_text if conv_text else " (This is the start of the conversation)"}

RULES:
1. Ask ONE question at a time that guides the student toward the next logical step
2. If the student answers correctly, confirm briefly ("✅ Exactly!") then ask the next guiding question
3. If the student answers incorrectly, say what's not quite right and rephrase the question with a smaller hint
4. If the student says "I don't know", give a more specific hint (mention the relevant concept/equation name) but still ask THEM to apply it
5. NEVER state the final answer — guide them to say it themselves
6. Use the actual values from this specific problem, not generic examples
7. When the student has worked through all the steps and reached the correct answer, congratulate them and summarize what they learned
8. Keep each response to 2-4 sentences maximum
9. This is turn {turn_number} — {"start by asking what information the problem gives them" if turn_number == 1 else "continue from where the conversation left off"}

Respond with ONLY valid JSON (no markdown, no backticks):
{{
  "message": "<your Socratic question or response — use ✅ or 💭 emoji to indicate if their answer was right/wrong>",
  "is_complete": <true if the student has successfully reached the final answer, false otherwise>,
  "progress": "<brief label: 'Identifying given values' or 'Applying the equation' or 'Calculating' or 'Checking answer' etc>"
}}"""

        if not ANTHROPIC_API_KEY:
            return jsonify({
                "message": "What values does the problem give you? Start by identifying all the given information.",
                "is_complete": False,
                "progress": "Starting"
            }), 200
        
        response_text = call_anthropic_api(coach_prompt, max_tokens=500)
        clean = response_text.strip()
        if clean.startswith('```'): clean = '\n'.join(clean.split('\n')[1:-1])
        result = json.loads(clean)
        
        return jsonify({
            "message": result.get("message", "What information does the problem give you?"),
            "is_complete": result.get("is_complete", False),
            "progress": result.get("progress", "")
        }), 200
        
    except Exception as e:
        print(f"[ERROR] coach-guide: {e}")
        return jsonify({
            "message": "Let's start with the basics — what values does the problem give you? What are you asked to find?",
            "is_complete": False,
            "progress": "Starting"
        }), 200


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
        
        # ── STEP CHECK MODE ──────────────────────────────────────
        # When the student is in step-by-step guided mode, evaluate only this step
        step_check = data.get("step_check", False)
        step_expected = data.get("step_expected", "")
        if step_check and step_expected:
            step_prompt = f"""You are an expert IB Physics HL tutor helping a student work through a problem step by step.

PROBLEM: {problem.get('name', '')}
Statement: {problem.get('problem_statement', '')}
Given values: {problem.get('given_values', '')}

CURRENT STEP (what the student should do):
{step_expected}

STUDENT'S REASONING FOR THIS STEP:
{solution_text}

Evaluate ONLY whether the student's reasoning correctly addresses this specific step.
Do NOT evaluate the entire problem — just this one step.

Respond with ONLY valid JSON (no markdown, no backticks):
{{
  "score": <0-100 for this step only>,
  "feedback": "<2-3 sentences: what they got right, what to fix, a specific hint if wrong>"
}}"""
            try:
                if ANTHROPIC_API_KEY:
                    step_response = call_anthropic_api(step_prompt, max_tokens=500)
                    clean = step_response.strip()
                    if clean.startswith('```'): clean = '\n'.join(clean.split('\n')[1:-1])
                    step_result = json.loads(clean)
                    return jsonify({
                        "score": step_result.get("score", 50),
                        "feedback": step_result.get("feedback", "Step evaluated."),
                        "step_check": True
                    }), 200
            except Exception as se:
                print(f"[WARNING] Step check failed, falling back: {se}")
            # Fallback: just return the expected step content
            return jsonify({
                "score": 50,
                "feedback": f"Compare your reasoning with the expected approach for this step.",
                "step_check": True
            }), 200
        
        # ── FULL SOLUTION EVALUATION ─────────────────────────────
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
        
        # Oxford Deep Link: get reference for this problem's topic
        problem_topic = problem.get('topic', '')
        oxford_ref = get_oxford_reference(problem_topic)
        oxford_block = f"""
OXFORD PHYSICS 2023 REFERENCE:
{oxford_ref}
When giving feedback on errors, reference the specific Oxford section and page above.
If the student made a calculation error, also mention the relevant Data Booklet page.
""" if oxford_ref else ""

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
{oxford_block}
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
  "feedback": "<detailed feedback: start with what the student did well, then explain errors. If the student made mistakes, end with: 'To review this topic, see [Oxford section] pages [X-Y].'>",
  "error_type": "<classify the main error: 'conceptual' | 'calculation' | 'units' | 'method' | 'incomplete' | 'none'>",
  "missing_steps": "<list any key steps or concepts the student missed>",
  "key_concept": "<the most important physics concept tested in this problem>",
  "oxford_reference": "{oxford_ref}",
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
    data          = request.get_json(force=True, silent=True) or {}
    email         = data.get("email", "").strip().lower()
    name          = data.get("name", "").strip()
    password      = data.get("password", "")
    group         = data.get("group", "").strip()
    role          = data.get("role", "student").strip().lower()        # student | professor
    prof_type     = data.get("professor_type", "").strip().lower()    # independent | institutional

    if not email or not name or not password:
        return jsonify({"error": "Missing fields"}), 400

    # Students must use ASF email; professors can use any email
    if role == "student" and not email.endswith("@asf.edu.mx"):
        return jsonify({"error": "Students must use ASF email (@asf.edu.mx)"}), 400

    # Check for duplicate email
    try:
        existing = query_database(USERS_DB_ID, {"property": "Email", "email": {"equals": email}})
        if existing:
            return jsonify({"error": "Email already registered"}), 400
    except Exception:
        pass

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        properties = {
            "Name":          {"title":     [{"text": {"content": name}}]},
            "Email":         {"email":     email},
            "password_hash": {"rich_text": [{"text": {"content": password_hash}}]},
            "role":          {"select":    {"name": role}},
            "created_at":    {"date":      {"start": datetime.utcnow().isoformat()}},
        }

        if group:
            properties["group"] = {"select": {"name": group}}

        if prof_type and role == "professor":
            properties["professor_type"] = {"select": {"name": prof_type}}

        create_page_in_database(USERS_DB_ID, properties)
        return jsonify({"success": True, "role": role}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/api/login")
def login():
    data     = request.get_json(force=True, silent=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    users = query_database(USERS_DB_ID, {"property": "Email", "email": {"equals": email}})
    if not users:
        return jsonify({"error": "Invalid credentials"}), 401

    user    = users[0]
    pw_hash = user.get("password_hash", "")
    if not pw_hash or not bcrypt.checkpw(password.encode("utf-8"), pw_hash.encode("utf-8")):
        return jsonify({"error": "Invalid credentials"}), 401

    role  = user.get("role", "student")
    group = user.get("group") or user.get("Group") or ""
    name  = user.get("Name") or user.get("name") or email.split("@")[0]

    token = jwt.encode(
        {
            "email": email,
            "name":  name,
            "role":  role,
            "group": group,
            "exp":   datetime.utcnow() + timedelta(days=30),
        },
        JWT_SECRET_KEY,
        algorithm="HS256",
    )

    return jsonify({
        "success": True,
        "token":   token,
        "role":    role,
        "group":   group,
        "user":    {"email": email, "name": name, "role": role, "group": group},
    })

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
        
        # Get professor's groups from token if available
        prof_groups = []
        auth_header = request.headers.get("Authorization", "")
        if auth_header:
            try:
                token = auth_header.replace("Bearer ", "")
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
                if payload.get("role") == "professor":
                    prof_groups = get_professor_groups(payload.get("group", ""))
            except Exception:
                pass

        for user in users:
            email = user.get("Email", "")
            name = user.get("Name", "")
            group = user.get("group") or user.get("Group") or ""
            role = user.get("role", "student")

            if not email: continue
            # Skip professors from student list
            if role == "professor": continue
            # If professor is authenticated, only show their groups
            if prof_groups and group not in prof_groups: continue
            
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
# AI STUDENT ANALYSIS (Claude-powered) — NEW
# ============================================================================

@app.get("/api/professor/ai-student-analysis")
def ai_student_analysis():
    """AI-powered deep analysis of a student's performance."""
    email = request.args.get("email", "").strip()
    if not email:
        return jsonify({"error": "missing email parameter"}), 400
    
    # Period filtering: "30d", "custom", or "all" (default)
    period = request.args.get("period", "all").strip()
    from_date = request.args.get("from", "").strip()  # ISO format: 2026-01-15
    to_date = request.args.get("to", "").strip()       # ISO format: 2026-03-01
    
    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        completed = [a for a in activities if a.get("action") in ("completed", "submitted")]
        
        # Apply time filter
        period_label = "all time"
        if period == "30d":
            from datetime import datetime, timedelta
            cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
            completed = [a for a in completed if (a.get("timestamp") or "") >= cutoff]
            period_label = "last 30 days"
        elif period == "custom" and from_date:
            completed = [a for a in completed if (a.get("timestamp") or "") >= from_date]
            if to_date:
                # Add T23:59:59 to include the full end day
                to_cutoff = to_date + "T23:59:59" if "T" not in to_date else to_date
                completed = [a for a in completed if (a.get("timestamp") or "") <= to_cutoff]
            period_label = f"{from_date} to {to_date}" if to_date else f"since {from_date}"
        
        if len(completed) < 2:
            return jsonify({
                "analysis": {
                    "summary": "Not enough data for AI analysis. The student needs to complete at least 2 problems.",
                    "patterns": [],
                    "predictions": [],
                    "recommendations": ["Encourage the student to complete more practice problems to enable AI analysis."]
                },
                "activities_analyzed": len(completed)
            })
        
        # Build structured student data for Claude
        student_data_lines = []
        for a in completed[:50]:
            sol = (a.get("solution_text") or "")[:100]
            line = (
                f"Problem: {a.get('problem_name', 'Unknown')} ({a.get('problem_reference', '')})\n"
                f"  Score: {a.get('score', 0)}% | "
                f"Marks: {a.get('marks_awarded', '?')}/{a.get('marks_total', '?')} | "
                f"Time: {a.get('time_spent_seconds', 0)}s | "
                f"Error: {a.get('error_type', 'none')} | "
                f"Concept: {a.get('key_concept', '')} | "
                f"Attempt #{a.get('attempt_number', 1)}"
            )
            if sol:
                line += f"\n  Solution preview: {sol}"
            student_data_lines.append(line)
        
        # Error type distribution
        error_counts = {}
        for a in completed:
            et = (a.get("error_type") or "").strip()
            if et and et != "none":
                error_counts[et] = error_counts.get(et, 0) + 1
        
        # Topic/concept performance
        topic_scores = {}
        for a in completed:
            concept = (a.get("key_concept") or "").strip() or "Unknown"
            if concept not in topic_scores:
                topic_scores[concept] = []
            topic_scores[concept].append(a.get("score", 0) or 0)
        
        topic_summary = []
        for topic, scores in topic_scores.items():
            avg = round(sum(scores) / len(scores)) if scores else 0
            topic_summary.append(f"{topic}: avg {avg}% ({len(scores)} problems)")
        
        overall_avg = round(sum(a.get("score", 0) or 0 for a in completed) / len(completed))
        
        prompt = f"""You are an expert IB Physics teacher analyzing a student's performance data.

STUDENT DATA ({len(completed)} completed problems):

{chr(10).join(student_data_lines[:50])}

ERROR TYPE DISTRIBUTION:
{chr(10).join(f'  {k}: {v} occurrences' for k, v in sorted(error_counts.items(), key=lambda x: -x[1])) if error_counts else '  No errors recorded yet'}

TOPIC/CONCEPT PERFORMANCE:
{chr(10).join(f'  {t}' for t in sorted(topic_summary))}

OVERALL: Avg score = {overall_avg}%, {len(completed)} problems completed (period: {period_label})

Analyze this student and respond ONLY with a JSON object (no markdown, no backticks, no extra text):
{{
  "summary": "2-3 sentences summarizing overall performance, strengths, and main areas of concern",
  "patterns": ["pattern 1 - be specific about topics and error types", "pattern 2", "pattern 3"],
  "predictions": ["prediction about upcoming exam performance based on the data"],
  "recommendations": ["specific actionable recommendation for the PROFESSOR", "recommendation 2", "recommendation 3"]
}}

RULES:
- Be specific, reference actual topics and error types from the data
- Patterns: identify RECURRING issues (e.g. 'Consistently makes unit conversion errors in Circuits')
- Predictions: be concrete (e.g. 'Likely to struggle with EM Induction if Magnetism weakness not addressed')
- Recommendations: actionable for the PROFESSOR (e.g. 'Schedule review session on free-body diagrams')
- Keep each item to 1-2 sentences max
- Write in English
- Return ONLY the JSON object, nothing else"""

        response_text = call_anthropic_api(prompt, max_tokens=1000)
        
        # Clean response
        clean_response = response_text.strip()
        if clean_response.startswith('```'):
            clean_response = '\n'.join(clean_response.split('\n')[1:-1])
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
        
        try:
            analysis = json.loads(clean_response)
        except json.JSONDecodeError:
            analysis = {
                "summary": clean_response[:500],
                "patterns": [],
                "predictions": [],
                "recommendations": []
            }
        
        return jsonify({
            "analysis": analysis,
            "activities_analyzed": len(completed),
            "period": period_label
        })
        
    except Exception as e:
        print(f"[AI ANALYSIS ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)[:200]}), 500

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
        response_text = call_anthropic_api(prompt, max_tokens=2000)
        
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
            response_text = call_anthropic_api(prompt, max_tokens=1500)
        
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
            students   = query_database(USERS_DB_ID)
            count      = 0
            for s in students:
                student_group = s.get("group") or s.get("Group") or ""
                role          = s.get("role", "student")
                if s.get("Email") and student_group == group_name and role == "student":
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
        groups   = set()

        # Get professor's groups from token
        prof_groups = []
        auth_header = request.headers.get("Authorization", "")
        if auth_header:
            try:
                token = auth_header.replace("Bearer ", "")
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
                if payload.get("role") == "professor":
                    prof_groups = get_professor_groups(payload.get("group", ""))
            except Exception:
                pass

        for u in users:
            email = u.get("Email", "")
            group = u.get("group") or u.get("Group") or ""
            role  = u.get("role", "student")

            if not email: continue
            if role == "professor": continue                          # excluir profesores
            if prof_groups and group not in prof_groups: continue    # solo grupos del profesor

            students.append({
                "id":    email,
                "name":  u.get("Name", ""),
                "email": email,
                "group": group,
            })
            if group:
                groups.add(group)

        return jsonify({
            "students": students,
            "groups":   sorted(list(groups)),
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


# ========================================
# STUDENT PROGRESS (for Study Plan Tracker)
# ========================================

@app.get("/api/professor/student-progress")
def professor_student_progress():
    """
    Get a specific student's completed problems for the Study Plan Tracker.
    Professors/parents can view any student's progress.

    Query params:
      - email (required): student email

    Returns: { completed_references: ["2022-Z1-P1-Q2", ...], total_completed, avg_score, total_time_minutes }
    """
    if not ACTIVITY_DB_ID:
        return jsonify({"error": "ACTIVITY_DB_ID not configured"}), 500

    student_email = request.args.get("email", "").strip()
    if not student_email:
        return jsonify({"error": "Email parameter required"}), 400

    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": student_email}},
            sorts=[{"property": "timestamp", "direction": "descending"}],
            max_pages=10
        )

        completed_refs = set()
        scores = []
        total_time = 0

        for act in activities:
            action = act.get("action", "")
            if action == "completed":
                ref = act.get("problem_reference") or act.get("problem_name", "")
                if ref:
                    completed_refs.add(ref)
                score = act.get("score", 0) or 0
                if score > 0:
                    scores.append(score)
                total_time += act.get("time_spent_seconds", 0) or 0

        return jsonify({
            "student_email": student_email,
            "completed_references": sorted(list(completed_refs)),
            "total_completed": len(completed_refs),
            "average_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "total_time_minutes": total_time // 60
        }), 200

    except Exception as e:
        print(f"[ERROR] professor/student-progress: {e}")
        return jsonify({"error": str(e)}), 500


# ==========================================
# MOCK EXAM ENDPOINTS — v2 (IB 2025 topics + multi-theme + Paper filter)
# ==========================================

# Syllabus 2025 exclusions
SYLLABUS_EXCLUDE_RE = re.compile(r'capacitor|capacitance|transformer', re.IGNORECASE)

def extract_ib_topic(p: dict) -> str:
    """Extract IB 2025 topic code (A.1-E.5) from a Notion problem record."""
    raw = p.get("topic") or p.get("Topic") or ""
    # multi_select comes as list from query_database
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    raw = str(raw).strip()
    if re.match(r"^[A-E]\.[1-5]$", raw):
        return raw
    # Fallback: infer from key_concepts
    kc = p.get("key_concepts") or p.get("Key Concepts") or ""
    if isinstance(kc, list):
        kc = kc[0] if kc else ""
    return ""  # caller handles inference

def extract_paper_from_ref(ref: str) -> str:
    """Extract P1 or P2 from reference string like 2018-Z1-P2-Q3-A."""
    m = re.search(r"-(P[123])-", ref or "", re.IGNORECASE)
    return m.group(1).upper() if m else ""

def is_excluded(p: dict) -> bool:
    """Returns True if problem should be excluded (Capacitors/Transformers)."""
    text = f"{p.get('name','')} {p.get('key_concepts','')}"
    return bool(SYLLABUS_EXCLUDE_RE.search(text))

@app.route("/api/mock/available-topics", methods=["GET"])
@require_auth
def mock_available_topics():
    """
    Returns IB 2025 topics (A.1-E.5) with problem counts.
    Excludes Capacitors/Transformers automatically.
    """
    user = request.user
    email = user.get("email", "")
    try:
        all_problems = query_database(NOTION_DB_ID, max_pages=5)

        solved_well = set()
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        ) if ACTIVITY_DB_ID else []

        for act in activities:
            if act.get("action") == "completed":
                score = act.get("score", 0) or 0
                ref = act.get("problem_reference") or act.get("problem_name", "")
                if score >= 85 and ref:
                    solved_well.add(ref)

        # IB_THEMES structure for response
        IB_THEMES_META = {
            "A": "Space, Time & Motion",
            "B": "Particulate Nature of Matter",
            "C": "Wave Behaviour",
            "D": "Fields",
            "E": "Nuclear & Quantum Physics"
        }

        topics = {}
        for p in all_problems:
            if is_excluded(p): continue
            ib_topic = extract_ib_topic(p)
            if not ib_topic: continue

            ref = p.get("problem_reference") or p.get("reference") or p.get("name") or ""
            paper = p.get("paper") or extract_paper_from_ref(ref)
            marks = p.get("marks") or p.get("Marks") or 0
            try: marks = int(marks)
            except: marks = 0

            if ib_topic not in topics:
                theme = ib_topic[0]
                topics[ib_topic] = {
                    "topic": ib_topic,
                    "theme": theme,
                    "theme_name": IB_THEMES_META.get(theme, ""),
                    "total_problems": 0, "total_marks": 0,
                    "p1_count": 0, "p2_count": 0,
                    "unsolved": 0, "unsolved_marks": 0
                }

            topics[ib_topic]["total_problems"] += 1
            topics[ib_topic]["total_marks"] += marks
            if paper == "P1": topics[ib_topic]["p1_count"] += 1
            if paper == "P2": topics[ib_topic]["p2_count"] += 1
            if ref not in solved_well:
                topics[ib_topic]["unsolved"] += 1
                topics[ib_topic]["unsolved_marks"] += marks

        topic_list = sorted(topics.values(), key=lambda t: t["topic"])

        # Group by theme for frontend
        themes = {}
        for t in topic_list:
            th = t["theme"]
            if th not in themes:
                themes[th] = {"theme": th, "name": IB_THEMES_META.get(th,""), "subtopics": [], "total_problems": 0}
            themes[th]["subtopics"].append(t)
            themes[th]["total_problems"] += t["total_problems"]

        return jsonify({
            "topics": topic_list,
            "themes": list(themes.values()),
            "total_problems": sum(t["total_problems"] for t in topic_list),
            "total_marks": sum(t["total_marks"] for t in topic_list)
        })
    except Exception as e:
        print(f"Error in available-topics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/mock/build", methods=["POST"])
@require_auth
def mock_build():
    """
    Build a mock exam from real Notion problems.

    Input JSON:
    - type: "topic" | "themes" | "final"
    - topic: "A.3" (single topic, for type=topic)
    - themes: ["A","B","C"] (list of theme letters, for type=themes)
    - paper: "P1" | "P2" | "" (optional paper filter)
    - target_problems: int (number of problems, default 30)
    - target_marks: int (alternative to target_problems, default None)

    Professor use case: type=themes, themes=["A","B","C"], paper="P1", target_problems=30
    """
    user = request.user
    email = user.get("email", "")
    data = request.get_json(force=True) if request.data else {}

    mock_type     = data.get("type", "themes")
    target_topic  = data.get("topic", "")
    target_themes = data.get("themes", [])   # ["A","B","C"]
    paper_filter  = (data.get("paper") or "").upper().strip()  # "P1" | "P2" | ""
    target_probs  = data.get("target_problems", 30)
    target_marks  = data.get("target_marks", None)

    try:
        import random as rand_mod

        all_problems = query_database(NOTION_DB_ID, max_pages=5)

        # Get activity
        solved_well = set()
        solved_any  = {}
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        ) if ACTIVITY_DB_ID else []

        for act in activities:
            if act.get("action") == "completed":
                score = act.get("score", 0) or 0
                ref = act.get("problem_reference") or act.get("problem_name", "")
                if ref:
                    if ref not in solved_any or score > solved_any[ref]:
                        solved_any[ref] = score
                    if score >= 85:
                        solved_well.add(ref)

        # Parse & filter problems
        parsed = []
        for p in all_problems:
            if is_excluded(p): continue

            ib_topic = extract_ib_topic(p)
            if not ib_topic: continue

            ref   = p.get("problem_reference") or p.get("reference") or p.get("name") or ""
            paper = (p.get("paper") or extract_paper_from_ref(ref)).upper()
            name  = p.get("Name") or p.get("name") or ""
            marks = p.get("marks") or p.get("Marks") or 0
            try: marks = int(marks)
            except: marks = 0

            # Paper filter
            if paper_filter and paper != paper_filter:
                continue

            parsed.append({
                "id": p.get("id",""),
                "reference": ref,
                "name": name,
                "ib_topic": ib_topic,
                "theme": ib_topic[0],
                "paper": paper,
                "marks": marks,
                "solved_well": ref in solved_well,
                "best_score": solved_any.get(ref, 0)
            })

        if not parsed:
            return jsonify({"error": "No problems found with the selected filters"}), 404

        # ── SELECTION LOGIC ──────────────────────────────────────────────
        selected = []

        if mock_type == "topic":
            # Single subtopic (A.3, E.5, etc.)
            pool = [p for p in parsed if p["ib_topic"] == target_topic]
            if not pool:
                return jsonify({"error": f"No problems found for topic {target_topic}"}), 404
            pool.sort(key=lambda p: (p["solved_well"], -p["marks"]))
            if target_marks:
                total = 0
                for p in pool:
                    if total >= target_marks: break
                    selected.append(p); total += p["marks"]
            else:
                selected = pool[:target_probs]

        elif mock_type == "themes":
            # Multiple themes (professor use case: A+B+C, Paper 1, 30 problems)
            if not target_themes:
                return jsonify({"error": "themes list required for type=themes"}), 400
            pool = [p for p in parsed if p["theme"] in target_themes]
            if not pool:
                return jsonify({"error": f"No problems found for themes {target_themes}"}), 404

            if target_marks:
                # Fill by marks
                pool.sort(key=lambda p: (p["solved_well"], -p["marks"]))
                total = 0
                for p in pool:
                    if total >= target_marks: break
                    selected.append(p); total += p["marks"]
            else:
                # Distribute evenly across themes
                by_theme = {}
                for p in pool:
                    by_theme.setdefault(p["theme"], []).append(p)
                per_theme = max(1, target_probs // len(by_theme))
                for theme, probs in by_theme.items():
                    probs.sort(key=lambda p: (p["solved_well"], -p["marks"]))
                    selected.extend(probs[:per_theme])
                # Top up to target
                used_ids = {p["id"] for p in selected}
                remaining = [p for p in pool if p["id"] not in used_ids]
                rand_mod.shuffle(remaining)
                needed = target_probs - len(selected)
                selected.extend(remaining[:needed])
            rand_mod.shuffle(selected)

        elif mock_type == "final":
            # Full exam across all themes proportionally
            by_theme = {}
            for p in parsed:
                by_theme.setdefault(p["theme"], []).append(p)
            per_theme = max(1, target_probs // max(1, len(by_theme)))
            for theme, probs in by_theme.items():
                probs.sort(key=lambda p: (p["solved_well"], -p["marks"]))
                selected.extend(probs[:per_theme])
            rand_mod.shuffle(selected)
            selected = selected[:target_probs]

        if not selected:
            return jsonify({"error": "Could not select problems with these filters"}), 404

        # Fetch full details
        problems = []
        for sp in selected:
            try:
                full = fetch_page(sp["id"])
                ib_topic = extract_ib_topic(full) or sp["ib_topic"]
                marks = sp["marks"] or 0

                problems.append({
                    "name": full.get("Name") or full.get("name") or "",
                    "problem_statement": full.get("problem_statement") or "",
                    "given_values": full.get("given_values") or "",
                    "find": full.get("find") or "",
                    "marks": marks,
                    "ib_topic": ib_topic,
                    "topic": ib_topic,
                    "paper": sp["paper"],
                    "key_concept": ib_topic,
                    "full_solution": full.get("full_solution") or full.get("step_by_step") or "",
                    "final_answer": full.get("final_answer") or "",
                    "source": "real",
                    "problem_id": sp["id"],
                    "problem_reference": sp["reference"],
                    "pdf_page_number": full.get("pdf_page_number") or None,
                    "oxford_reference": get_oxford_reference(ib_topic)
                })
            except Exception as e:
                print(f"Error fetching {sp['id']}: {e}")
                continue

        if not problems:
            return jsonify({"error": "Could not fetch problem details"}), 500

        topics_covered = list(set(p["ib_topic"] for p in problems))
        p1_count = sum(1 for p in problems if p["paper"] == "P1")
        p2_count = sum(1 for p in problems if p["paper"] == "P2")

        return jsonify({
            "problems": problems,
            "total_marks": sum(p["marks"] for p in problems),
            "num_problems": len(problems),
            "topics_covered": sorted(topics_covered),
            "mock_type": mock_type,
            "paper_filter": paper_filter or "all",
            "p1_count": p1_count,
            "p2_count": p2_count,
            "themes_requested": target_themes if mock_type == "themes" else None,
            "target_topic": target_topic if mock_type == "topic" else None
        })

    except Exception as e:
        print(f"Error building mock: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/mock/create-weakness", methods=["POST"])
@require_auth
def create_weakness_mock():
    """
    Generate a weakness-based mock exam.
    Uses activity history to find weak topics, then generates problems for each.
    
    Optional input JSON:
    - num_problems: 3-5 (default 4)
    - marks_per_problem: default 4
    
    Returns: { type, title, problems[], total_marks, topics, weakness_data, created_at }
    """
    user = request.user
    email = user.get("email", "")
    data = request.get_json(force=True, silent=True) or {}
    num_problems = min(5, max(3, int(data.get("num_problems", 4))))
    marks_per = int(data.get("marks_per_problem", 4))
    
    try:
        # 1. Get student's activity to find weaknesses
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": email}},
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        all_problems = query_database(NOTION_DB_ID, max_pages=5)
        
        # 2. Analyze weaknesses
        topic_scores = {}
        error_patterns = {}
        
        for act in activities:
            if act.get("action") != "completed":
                continue
            score = act.get("score", 0) or 0
            error_type = (act.get("error_type") or "").strip().lower()
            key_concept = (act.get("key_concept") or "").strip()
            
            problem_ref = act.get("problem_reference") or act.get("problem_name", "")
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
                
                if error_type and error_type != "none":
                    key = (topic, error_type)
                    error_patterns[key] = error_patterns.get(key, 0) + 1
        
        # 3. Rank topics by weakness (lowest avg score first)
        weak_topics = []
        for topic, scores in topic_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 85:
                weak_topics.append({
                    "topic": topic,
                    "avg_score": round(avg, 1),
                    "attempts": len(scores),
                    "main_error": ""
                })
        
        weak_topics.sort(key=lambda x: x["avg_score"])
        
        # Add main error type per topic
        for wt in weak_topics:
            topic_errors = {k: v for k, v in error_patterns.items() if k[0] == wt["topic"]}
            if topic_errors:
                top_error = max(topic_errors, key=topic_errors.get)
                wt["main_error"] = top_error[1]
        
        # Fallback if no weak topics found
        if not weak_topics:
            all_topics = set()
            for p in all_problems:
                kc = p.get("key_concepts") or p.get("Key Concepts") or ""
                if isinstance(kc, list):
                    kc = kc[0] if kc else ""
                t = kc.split(";")[0].strip() if kc else ""
                if t:
                    all_topics.add(t)
            weak_topics = [{"topic": t, "avg_score": 50, "attempts": 0, "main_error": ""} for t in list(all_topics)[:num_problems]]
        
        # 4. Select topics for mock
        selected = weak_topics[:num_problems]
        
        # 5. Generate problems
        import random as rand_mod
        import json as json_mod
        generated = []
        
        for wt in selected:
            try:
                topic_problems = []
                for p in all_problems:
                    kc = p.get("key_concepts") or p.get("Key Concepts") or ""
                    if isinstance(kc, list):
                        kc = kc[0] if kc else ""
                    p_topic = kc.split(";")[0].strip().lower() if kc else ""
                    if wt["topic"].lower() in p_topic or p_topic in wt["topic"].lower():
                        topic_problems.append(p)
                
                template_text = ""
                if topic_problems:
                    samples = rand_mod.sample(topic_problems, min(2, len(topic_problems)))
                    for s in samples:
                        try:
                            full = fetch_page(s.get("id", ""))
                            template_text += f"\nName: {full.get('name','')}\nStatement: {full.get('problem_statement','')}\nMarks: {full.get('marks', marks_per)}\nSolution: {full.get('full_solution','')}\n---\n"
                        except:
                            pass
                
                error_focus = f"\nFocus on testing areas where students commonly make {wt['main_error']} errors." if wt.get("main_error") else ""
                
                gen_prompt = f"""You are an expert IB Physics HL exam creator.

Create a NEW problem about {wt['topic']} worth {marks_per} marks.{error_focus}

{('REFERENCE EXAMPLES:\n' + template_text) if template_text else ''}

REQUIREMENTS:
- IB Physics HL standard difficulty
- Include all given values with SI units
- Must be solvable step-by-step
- Marks must follow IB conventions

Respond with ONLY valid JSON:
{{
  "name": "[AI Mock] {wt['topic']} Problem",
  "problem_statement": "...",
  "given_values": "...",
  "find": "...",
  "marks": {marks_per},
  "key_concept": "{wt['topic']}",
  "full_solution": "Complete step-by-step solution...",
  "final_answer": "..."
}}"""
                
                result = call_anthropic_api(gen_prompt, max_tokens=2000)
                
                clean = result.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    clean = clean.strip()
                
                problem_data = json_mod.loads(clean)
                problem_data["topic"] = wt["topic"]
                problem_data["weakness_score"] = wt["avg_score"]
                problem_data["target_error"] = wt.get("main_error", "")
                generated.append(problem_data)
                
            except Exception as e:
                print(f"Error generating mock problem for {wt['topic']}: {e}")
                continue
        
        if not generated:
            return jsonify({"error": "Could not generate problems"}), 500
        
        mock_result = {
            "type": "weakness",
            "title": "Weakness Practice",
            "problems": generated,
            "topics": [g["topic"] for g in generated],
            "total_marks": sum(g.get("marks", marks_per) for g in generated),
            "weakness_data": selected,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(mock_result)
        
    except Exception as e:
        print(f"Error creating weakness mock: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/mock/history", methods=["GET"])
@require_auth
def mock_history():
    """
    Get mock exam history for the current student.
    Looks for activities with source='mock_weakness' and groups them by session.
    """
    user = request.user
    email = user.get("email", "")
    
    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={
                "and": [
                    {"property": "user_email", "email": {"equals": email}},
                    {"property": "action", "select": {"equals": "completed"}}
                ]
            },
            sorts=[{"property": "timestamp", "direction": "descending"}]
        )
        
        mocks = {}
        for act in activities:
            ref = act.get("problem_reference") or ""
            name = act.get("problem_name") or ""
            
            if ref.startswith("mock-") or name.startswith("[Mock]"):
                # All mock problems share session: mock-{timestamp}-{index}
                parts = ref.split("-")
                if len(parts) >= 2:
                    mock_id = "-".join(parts[:2])  # mock-{timestamp}
                else:
                    mock_id = ref
                
                if mock_id not in mocks:
                    mocks[mock_id] = {
                        "mock_id": mock_id,
                        "date": act.get("timestamp", ""),
                        "problems": [],
                        "total_score": 0,
                        "total_marks_awarded": 0,
                        "total_marks": 0
                    }
                
                score = act.get("score", 0) or 0
                marks_awarded = act.get("marks_awarded", 0) or 0
                marks_total = act.get("marks_total", 0) or 0
                
                mocks[mock_id]["problems"].append({
                    "topic": act.get("key_concept") or "",
                    "score": score,
                    "marks_awarded": marks_awarded,
                    "marks_total": marks_total,
                    "error_type": act.get("error_type") or "",
                    "time_spent": act.get("time_spent_seconds", 0) or 0
                })
                mocks[mock_id]["total_score"] += score
                mocks[mock_id]["total_marks_awarded"] += marks_awarded
                mocks[mock_id]["total_marks"] += marks_total
        
        mock_list = []
        for m in mocks.values():
            if m["problems"]:
                m["avg_score"] = round(m["total_score"] / len(m["problems"]), 1)
                m["num_problems"] = len(m["problems"])
                mock_list.append(m)
        
        mock_list.sort(key=lambda x: x["date"], reverse=True)
        
        return jsonify({"mocks": mock_list[:20]})
        
    except Exception as e:
        print(f"Error fetching mock history: {e}")
        return jsonify({"mocks": []})


# ============================================================================
# STUDY PLAN — Plan de Estudio Semanal (6 semanas al examen)
# ============================================================================

@app.get("/api/study-plan")
@require_auth
def get_study_plan():
    """
    Obtiene el plan de estudio completo de un alumno (6 semanas).
    El profesor puede consultar cualquier alumno; el alumno solo ve el suyo.
    Combina plan del profesor (si existe) con sugerencias IA para semanas vacías.
    """

    user  = request.user
    role  = user.get("role", "student")
    email = request.args.get("email", "").strip()

    # Alumnos solo pueden ver su propio plan
    if role == "student" or not email:
        email = user.get("email", "")

    if not email:
        return jsonify({"error": "Missing email"}), 400

    try:
        # ── 1. ACTIVIDAD DEL ALUMNO ───────────────────────────
        activities = []
        if ACTIVITY_DB_ID:
            activities = query_database(
                ACTIVITY_DB_ID,
                filter_obj={"property": "user_email", "email": {"equals": email}},
                sorts=[{"property": "timestamp", "direction": "descending"}],
            )
        completed = [a for a in activities if a.get("action") in ("completed", "submitted")]

        # ── 2. PLAN DEL PROFESOR (en Homework DB con prefijo [PLAN]) ──
        plan_items_raw = []
        if HOMEWORK_DB_ID:
            all_hw = query_database(
                HOMEWORK_DB_ID,
                filter_obj={"property": "student_email", "email": {"equals": email}},
            )
            plan_items_raw = [h for h in all_hw if (h.get("Title") or "").startswith("[PLAN]")]

        # ── 3. ANÁLISIS DE DEBILIDADES ────────────────────────
        topic_scores = defaultdict(list)
        error_counts = defaultdict(int)
        for a in completed:
            concept = (a.get("key_concept") or "").strip() or "General"
            score   = a.get("score", 0) or 0
            topic_scores[concept].append(score)
            et = (a.get("error_type") or "").strip()
            if et and et not in ("none", ""):
                error_counts[et] += 1

        topic_avgs     = {t: round(sum(s) / len(s)) for t, s in topic_scores.items() if s}
        dominant_error = max(error_counts, key=error_counts.get) if error_counts else None

        # ── 4. LISTA COMPLETA DE TOPICS IB PHYSICS HL ─────────
        ib_topics = [
            "Astrophysics", "Quantum", "Nuclear", "EM Induction", "Magnetism",
            "Circuits", "Electric Fields", "Waves", "Optics", "Sound",
            "Thermodynamics", "SHM", "Ideal Gases", "Gravitation", "Circular Motion",
            "Momentum", "Work & Energy", "Forces", "Kinematics", "Rotational Dynamics",
        ]
        unpracticed     = [t for t in ib_topics if t not in topic_avgs]
        priority_topics = [t for t, _ in sorted(topic_avgs.items(), key=lambda x: x[1])] + unpracticed

        exam_date  = datetime(2026, 4, 28)
        plan_start = datetime(2026, 3, 17)

        # ── 5. CONSTRUIR LAS 6 SEMANAS ────────────────────────
        weeks = []
        for i in range(6):
            week_start = plan_start + timedelta(weeks=i)
            week_end   = week_start + timedelta(days=6)
            week_num   = i + 1

            # Plan del profesor para esta semana
            plan_key   = f"[PLAN] Week {week_num}"
            prof_item  = next(
                (p for p in plan_items_raw if (p.get("Title") or "").startswith(plan_key)),
                None,
            )
            prof_data = {}
            if prof_item:
                desc  = prof_item.get("description", "") or ""
                parts = desc.split("|") if "|" in desc else [desc, "", ""]
                prof_data = {
                    "topic":    parts[0].strip() if len(parts) > 0 else "",
                    "note":     parts[1].strip() if len(parts) > 1 else "",
                    "problems": parts[2].strip() if len(parts) > 2 else "",
                    "hw_id":    prof_item.get("id", ""),
                }

            # Sugerencia IA
            ai_topic  = priority_topics[i] if i < len(priority_topics) else ib_topics[i % len(ib_topics)]
            avg_topic = topic_avgs.get(ai_topic)
            if avg_topic is not None:
                if avg_topic < 50:
                    ai_reason = f"Score de {avg_topic}% — prioridad alta para el examen"
                elif avg_topic < 70:
                    ai_reason = f"Score de {avg_topic}% — necesita refuerzo"
                else:
                    ai_reason = f"Score de {avg_topic}% — mantenimiento"
            else:
                ai_reason = "Topic no practicado todavía — cubrir antes del examen"

            # Fase y metas por semana
            if week_num <= 2:
                phase, phase_label = "diagnosis",     "🔍 Diagnóstico"
            elif week_num <= 4:
                phase, phase_label = "reinforcement", "🎯 Refuerzo"
            elif week_num == 5:
                phase, phase_label = "consolidation", "📊 Consolidación"
            else:
                phase, phase_label = "exam",          "🏆 Semana del Examen"

            rec_problems = {1: 5, 2: 6, 3: 8, 4: 8, 5: 10, 6: 4}.get(week_num, 6)
            rec_mock     = {
                1: "Weakness Mock al final de la semana",
                2: "Topic Mock del topic asignado",
                3: "Topic Mock del topic asignado",
                4: "IB Final Mock (Exam Mode, 2h30m)",
                5: "Segundo IB Final Mock",
                6: "Weakness Mock final",
            }.get(week_num, "Topic Mock")

            # Estado de la semana
            now = datetime.utcnow()
            if now < week_start:
                status = "upcoming"
            elif now > week_end:
                status = "completed"
            else:
                status = "active"

            # Progreso real del alumno esta semana
            week_acts   = [
                a for a in completed
                if week_start.isoformat()[:10] <= (a.get("timestamp") or "")[:10] <= week_end.isoformat()[:10]
            ]
            week_scores = [a.get("score", 0) or 0 for a in week_acts if (a.get("score", 0) or 0) > 0]
            week_avg    = round(sum(week_scores) / len(week_scores)) if week_scores else None

            weeks.append({
                "week_number":  week_num,
                "week_start":   week_start.strftime("%d %b"),
                "week_end":     week_end.strftime("%d %b"),
                "phase":        phase,
                "phase_label":  phase_label,
                "status":       status,
                "days_to_exam": max(0, (exam_date - week_end).days),
                "professor": {
                    "set":      bool(prof_data),
                    "topic":    prof_data.get("topic", ""),
                    "note":     prof_data.get("note", ""),
                    "problems": prof_data.get("problems", ""),
                    "hw_id":    prof_data.get("hw_id", ""),
                },
                "ai": {
                    "topic":          ai_topic,
                    "reason":         ai_reason,
                    "rec_problems":   rec_problems,
                    "rec_mock":       rec_mock,
                    "dominant_error": dominant_error,
                },
                "progress": {
                    "problems_completed": len(week_acts),
                    "avg_score":          week_avg,
                    "on_track":           len(week_acts) >= int(rec_problems * 0.6) if status != "upcoming" else None,
                },
            })

        # ── 6. RESUMEN GENERAL ────────────────────────────────
        all_scores  = [a.get("score", 0) or 0 for a in completed if (a.get("score", 0) or 0) > 0]
        overall_avg = round(sum(all_scores) / len(all_scores)) if all_scores else 0
        boundaries  = [(75, 7), (63, 6), (52, 5), (40, 4), (28, 3), (16, 2), (0, 1)]
        predicted   = next((g for b, g in boundaries if overall_avg >= b), 1)

        return jsonify({
            "student_email": email,
            "exam_date":     "28 de abril 2026",
            "weeks":         weeks,
            "overview": {
                "total_problems":     len(completed),
                "overall_avg":        overall_avg,
                "predicted_grade":    predicted,
                "topics_covered":     len(topic_avgs),
                "topics_total":       len(ib_topics),
                "topics_strong":      sum(1 for s in topic_avgs.values() if s >= 85),
                "topics_developing":  sum(1 for s in topic_avgs.values() if 65 <= s < 85),
                "topics_weak":        sum(1 for s in topic_avgs.values() if s < 65),
                "priority_topics":    priority_topics[:6],
            },
        }), 200

    except Exception as e:
        print(f"[STUDY PLAN ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)[:300]}), 500


@app.post("/api/professor/study-plan")
@require_auth
def set_study_plan():
    """
    El profesor establece o actualiza el plan de una semana para un alumno.
    Guarda en Homework DB con prefijo [PLAN] para no requerir nueva DB.
    Body: { student_email, week_number, topic, note, problems }
    """
    if not HOMEWORK_DB_ID:
        return jsonify({"error": "HOMEWORK_DB_ID not configured"}), 500

    data          = request.get_json() or {}
    student_email = data.get("student_email", "").strip()
    week_number   = int(data.get("week_number", 1))
    topic         = data.get("topic", "").strip()
    note          = data.get("note", "").strip()
    problems      = data.get("problems", "").strip()

    if not student_email or not topic:
        return jsonify({"error": "Missing student_email or topic"}), 400

    try:
        plan_start  = datetime(2026, 3, 17)
        week_start  = plan_start + timedelta(weeks=int(week_number) - 1)
        week_end    = week_start + timedelta(days=6)
        title       = f"[PLAN] Week {week_number} — {topic}"
        description = f"{topic}|{note}|{problems}"

        # Buscar si ya existe plan para esta semana
        existing_hw  = query_database(
            HOMEWORK_DB_ID,
            filter_obj={"property": "student_email", "email": {"equals": student_email}},
        )
        plan_key      = f"[PLAN] Week {week_number}"
        existing_item = next(
            (h for h in existing_hw if (h.get("Title") or "").startswith(plan_key)),
            None,
        )

        if existing_item:
            page_id = existing_item.get("id", "")
            if page_id:
                update_page_properties(page_id, {
                    "Title":       {"title":     [{"text": {"content": title}}]},
                    "description": {"rich_text": [{"text": {"content": description}}]},
                })
                cache_clear()
                return jsonify({"status": "updated", "week": week_number}), 200

        create_page_in_database(HOMEWORK_DB_ID, {
            "Title":         {"title":     [{"text": {"content": title}}]},
            "student_email": {"email":     student_email},
            "description":   {"rich_text": [{"text": {"content": description}}]},
            "due_date":      {"date":      {"start": week_end.strftime("%Y-%m-%d")}},
            "completed":     {"checkbox":  False},
        })
        cache_clear()
        return jsonify({"status": "created", "week": week_number, "topic": topic}), 200

    except Exception as e:
        print(f"[SET STUDY PLAN ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)[:300]}), 500


@app.get("/api/study-plan/ai-generate-week")
@require_auth
def ai_generate_week_plan():
    """
    IA genera el plan detallado día a día para una semana específica.
    Params: email, week_number
    """

    student_email = request.args.get("email", "").strip()
    week_number   = int(request.args.get("week_number", 1))

    if not student_email:
        return jsonify({"error": "Missing email"}), 400

    try:
        activities = query_database(
            ACTIVITY_DB_ID,
            filter_obj={"property": "user_email", "email": {"equals": student_email}},
            sorts=[{"property": "timestamp", "direction": "descending"}],
        )
        completed = [a for a in activities if a.get("action") in ("completed", "submitted")]

        topic_scores = defaultdict(list)
        for a in completed[:30]:
            concept = (a.get("key_concept") or "").strip() or "General"
            topic_scores[concept].append(a.get("score", 0) or 0)

        topic_summary = "\n".join([
            f"- {t}: {round(sum(s)/len(s))}% avg ({len(s)} problems)"
            for t, s in sorted(topic_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))[:8]
        ]) or "No practice data yet"

        plan_start = datetime(2026, 3, 17)
        week_start = plan_start + timedelta(weeks=week_number - 1)
        week_end   = week_start + timedelta(days=6)
        days_left  = max(0, (datetime(2026, 4, 28) - week_end).days)

        prompt = f"""You are an expert IB Physics HL tutor. Generate a specific weekly study plan.

Week {week_number} of 6 ({week_start.strftime('%B %d')} to {week_end.strftime('%B %d, %Y')})
Days to IB exam after this week: {days_left}
Problems completed so far: {len(completed)}

Topic performance:
{topic_summary}

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "topic_focus": "Main topic to work on this week",
  "why": "1 sentence explaining why this topic is the priority",
  "daily_plan": [
    {{"day": "Monday-Tuesday", "action": "specific action", "time": "30 min"}},
    {{"day": "Wednesday-Thursday", "action": "specific action", "time": "30 min"}},
    {{"day": "Friday-Weekend", "action": "specific action", "time": "45 min"}}
  ],
  "mock_recommendation": "Which mock to do and when",
  "success_metric": "How to know if the week was successful",
  "professor_note": "What the teacher should focus on in the next session"
}}"""

        response = call_anthropic_api(prompt, max_tokens=700)
        clean    = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rstrip("`").strip()
        plan = json.loads(clean)
        return jsonify({"week_number": week_number, "plan": plan}), 200

    except Exception as e:
        print(f"[AI WEEK PLAN ERROR] {e}")
        return jsonify({"error": str(e)[:200]}), 500

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — STUDY PLAN GENERATOR
# Agregar este código AL FINAL del archivo backend.py, ANTES de if __name__ == "__main__"
# ══════════════════════════════════════════════════════════════════════════════
# Features:
# - ROI calculation (exam_weight × gap)
# - CRITICAL / WILDCARD / IMPORTANT topic classification
# - Countdown phases (BUILD / REINFORCE / SIMULATE / PEAK)
# - Quick Review vs Deep Dive modes
# - What-If predictor (show future if plan completed)
# - Specific problem selection
# - Integrity detection
# - Class rank calculation
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# IB EXAM WEIGHTS — Based on 192 real IB problems analysis (2022-2024)
# ─────────────────────────────────────────────────────────────────────────────
TOPIC_WEIGHTS = {
    # CRITICAL — ~80% of exam, must master
    'A.2': 0.118,  # Forces & Momentum — #1 most frequent
    'B.5': 0.114,  # Electric Circuits
    'C.1': 0.105,  # Simple Harmonic Motion
    'B.1': 0.095,  # Thermal Energy
    'E.3': 0.092,  # Radioactive Decay
    'E.2': 0.087,  # Quantum Physics
    'D.4': 0.077,  # Electromagnetic Induction
    'C.3': 0.070,  # Wave Phenomena
    
    # WILDCARD — New Core 2025, never tested, MUST appear
    'A.4': 0.060,  # Rigid Body Mechanics
    'A.5': 0.050,  # Special Relativity
    'B.4': 0.050,  # Thermodynamics
    
    # IMPORTANT — Regular appearance
    'B.3': 0.045,  # Gas Laws
    'D.2': 0.040,  # Electric & Magnetic Fields
    'D.1': 0.035,  # Gravitational Fields
    'B.2': 0.030,  # Greenhouse Effect
    'E.1': 0.025,  # Nuclear Structure
    'A.1': 0.025,  # Kinematics
    'A.3': 0.020,  # Work, Energy & Power
    
    # LOW PRIORITY — Rare, cut first if time is short
    'D.3': 0.008,  # Motion in Fields
    'C.4': 0.006,  # Standing Waves
    'C.5': 0.005,  # Doppler Effect
    'E.4': 0.005,  # Fission & Fusion
    'C.2': 0.005,  # Wave Properties
    'E.5': 0.015,  # Particle Physics
}

TOPIC_NAMES = {
    'A.1': 'Kinematics', 'A.2': 'Forces & Momentum', 'A.3': 'Work, Energy & Power',
    'A.4': 'Rigid Body Mechanics', 'A.5': 'Special Relativity',
    'B.1': 'Thermal Energy', 'B.2': 'Greenhouse Effect', 'B.3': 'Gas Laws',
    'B.4': 'Thermodynamics', 'B.5': 'Electric Circuits',
    'C.1': 'Simple Harmonic Motion', 'C.2': 'Wave Properties', 'C.3': 'Wave Phenomena',
    'C.4': 'Standing Waves', 'C.5': 'Doppler Effect',
    'D.1': 'Gravitational Fields', 'D.2': 'Electric & Magnetic Fields',
    'D.3': 'Motion in Fields', 'D.4': 'Electromagnetic Induction',
    'E.1': 'Nuclear Structure', 'E.2': 'Quantum Physics', 'E.3': 'Radioactive Decay',
    'E.4': 'Fission & Fusion', 'E.5': 'Particle Physics',
}

CRITICAL_TOPICS = {'A.2', 'B.5', 'C.1', 'B.1', 'E.3', 'E.2', 'D.4', 'C.3'}
WILDCARD_TOPICS = {'A.4', 'A.5', 'B.4'}
LOW_TOPICS = {'D.3', 'C.4', 'C.5', 'E.4', 'C.2'}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE STRATEGY — Based on days to exam
# ─────────────────────────────────────────────────────────────────────────────
def get_countdown_phase(days_to_exam: int) -> dict:
    """Returns phase configuration based on countdown to exam."""
    if days_to_exam >= 21:
        return {
            'name': 'build',
            'label': '🔨 Build Phase',
            'focus': 'Cover CRITICAL topic gaps — mistakes are learning',
            'strategy': 'Focus on weakest CRITICAL topics first',
            'quick_problems': 6,
            'deep_problems': 10,
            'max_minutes_daily': 30,
            'allow_new_topics': True,
        }
    elif days_to_exam >= 8:
        return {
            'name': 'reinforce',
            'label': '🎯 Reinforcement Phase',
            'focus': 'WILDCARD topics (A.4, A.5, B.4) + consolidate criticals',
            'strategy': 'Balance new wildcards with proven strengths',
            'quick_problems': 5,
            'deep_problems': 8,
            'max_minutes_daily': 25,
            'allow_new_topics': True,
        }
    elif days_to_exam >= 4:
        return {
            'name': 'simulate',
            'label': '📝 Simulation Phase',
            'focus': 'Full mocks under exam conditions',
            'strategy': 'Practice exam timing, no new material',
            'quick_problems': 4,
            'deep_problems': 4,
            'max_minutes_daily': 45,
            'allow_new_topics': False,
        }
    else:
        return {
            'name': 'peak',
            'label': '🏆 Peak Phase',
            'focus': 'Light review + REST — trust your preparation',
            'strategy': 'Quick formula review, sleep well, no cramming',
            'quick_problems': 2,
            'deep_problems': 2,
            'max_minutes_daily': 15,
            'allow_new_topics': False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ROI CALCULATION — Prioritize topics for study
# ─────────────────────────────────────────────────────────────────────────────
def calculate_topic_roi(student_topic_data: dict) -> list:
    """
    Calculate Return on Investment for each topic.
    ROI = exam_weight × (1 - student_mark_rate_for_topic) × boost_factor
    """
    roi_list = []
    
    for topic, weight in TOPIC_WEIGHTS.items():
        data = student_topic_data.get(topic, {})
        mark_rate = data.get('mark_rate', 0)
        n_attempts = data.get('n', 0)
        
        gap = 1 - mark_rate
        roi = weight * gap * 10  # Scale for readability
        
        # Boost wildcards
        if topic in WILDCARD_TOPICS:
            roi *= 1.5
            reason = 'WILDCARD — New Core 2025, never tested'
            priority = 'high'
        elif topic in CRITICAL_TOPICS:
            reason = 'CRITICAL — High exam frequency'
            priority = 'high' if gap > 0.4 else 'medium'
        elif topic in LOW_TOPICS:
            reason = 'LOW — Rare in exams'
            priority = 'low'
        else:
            reason = 'IMPORTANT — Regular appearance'
            priority = 'medium'
        
        roi_list.append({
            'topic': topic,
            'name': TOPIC_NAMES.get(topic, topic),
            'weight_pct': round(weight * 100, 1),
            'mark_rate_pct': round(mark_rate * 100, 1),
            'gap_pct': round(gap * 100, 1),
            'roi': round(roi, 2),
            'n_attempts': n_attempts,
            'reason': reason,
            'priority': priority,
            'is_wildcard': topic in WILDCARD_TOPICS,
            'is_critical': topic in CRITICAL_TOPICS,
        })
    
    roi_list.sort(key=lambda x: x['roi'], reverse=True)
    return roi_list


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT IB GRADE — Honest prediction based on practice
# ─────────────────────────────────────────────────────────────────────────────
def predict_ib_grade_v2(mark_rate: float, n_problems: int, n_topics: int, minutes_spent: float) -> dict:
    """Honest IB grade prediction with confidence interval."""
    if n_problems < 5 or n_topics < 2:
        return {'grade': None, 'confidence': 'insufficient_data', 'reason': f'{n_problems}/5 problems, {n_topics}/2 topics needed'}
    
    pct = mark_rate * 100
    
    # Grade boundaries (IB Physics HL typical)
    if pct >= 75: grade = 7
    elif pct >= 65: grade = 6
    elif pct >= 55: grade = 5
    elif pct >= 45: grade = 4
    elif pct >= 35: grade = 3
    elif pct >= 25: grade = 2
    else: grade = 1
    
    # Confidence based on sample size
    if n_problems >= 30 and n_topics >= 8:
        confidence, margin = 'high', 0
    elif n_problems >= 15 and n_topics >= 5:
        confidence, margin = 'medium', 1
    else:
        confidence, margin = 'low', 2
    
    return {
        'grade': grade,
        'confidence': confidence,
        'low': max(1, grade - margin),
        'high': min(7, grade + margin),
        'mark_rate_pct': round(pct, 1),
        'sample_size': n_problems,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WHAT-IF PREDICTOR — Show impact of completing the plan
# ─────────────────────────────────────────────────────────────────────────────
def calculate_what_if(current_marks: int, current_possible: int, selected_problems: list, target_score: float = 0.75) -> dict:
    """Predict stats change if student completes the plan."""
    additional_marks_possible = sum(p.get('marks', 6) for p in selected_problems)
    additional_marks_awarded = int(additional_marks_possible * target_score)
    
    new_marks = current_marks + additional_marks_awarded
    new_possible = current_possible + additional_marks_possible
    
    current_rate = current_marks / current_possible if current_possible > 0 else 0
    new_rate = new_marks / new_possible if new_possible > 0 else 0
    
    return {
        'current': {
            'mark_rate_pct': round(current_rate * 100, 1),
            'total_marks': current_marks,
        },
        'projected': {
            'mark_rate_pct': round(new_rate * 100, 1),
            'total_marks': new_marks,
        },
        'improvement': {
            'mark_rate_delta': round((new_rate - current_rate) * 100, 1),
            'additional_problems': len(selected_problems),
        },
        'assumption': f'Assumes {int(target_score*100)}% score on {len(selected_problems)} problems',
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRITY CHECK — Detect potential copy-paste
# ─────────────────────────────────────────────────────────────────────────────
def check_integrity(activities: list) -> dict:
    """Check for suspicious submission patterns (not accusation — coaching signal)."""
    valid = [a for a in activities if (a.get('marks_total') or 0) > 0 and (a.get('time_spent_seconds') or 0) > 0]
    
    if len(valid) < 3:
        return {'level': 'insufficient_data', 'pct': 0, 'message': None}
    
    suspicious = [a for a in valid if (a['time_spent_seconds'] / a['marks_total']) < 30]
    rate = len(suspicious) / len(valid)
    
    if rate > 0.30:
        return {
            'level': 'verify',
            'pct': round(rate * 100),
            'message': 'Consider in-class demonstration to verify understanding',
        }
    elif rate > 0.10:
        return {
            'level': 'review',
            'pct': round(rate * 100),
            'message': 'Monitor next few submissions',
        }
    else:
        return {'level': 'ok', 'pct': 0, 'message': None}


# ─────────────────────────────────────────────────────────────────────────────
# SELECT PROBLEMS — Choose specific problems for the plan
# ─────────────────────────────────────────────────────────────────────────────
def select_problems_for_plan(priority_topics: list, problems_db: list, completed_refs: set, phase: dict, mode: str = 'quick') -> list:
    """Select specific problems based on ROI and phase."""
    max_problems = phase.get('quick_problems', 5) if mode == 'quick' else phase.get('deep_problems', 10)
    
    # Build topic → problems mapping
    topic_problems = {}
    for p in problems_db:
        topic = p.get('topic', '')
        if isinstance(topic, list):
            topic = topic[0] if topic else ''
        if not topic:
            continue
        
        ref = p.get('problem_reference', '')
        if not ref or ref in completed_refs:
            continue
        
        if topic not in topic_problems:
            topic_problems[topic] = []
        
        topic_problems[topic].append({
            'ref': ref,
            'id': p.get('id', ''),
            'name': p.get('name', p.get('Name', ref)),
            'topic': topic,
            'topic_name': TOPIC_NAMES.get(topic, topic),
            'marks': p.get('marks') or p.get('Marks') or 6,
            'difficulty': p.get('difficulty', 'medium'),
        })
    
    # Sort by difficulty within each topic
    difficulty_order = {'easy': 0, 'medium': 1, 'hard': 2}
    for topic in topic_problems:
        topic_problems[topic].sort(key=lambda x: difficulty_order.get(x['difficulty'], 1))
    
    # Select from priority topics
    selected = []
    problems_per_topic = max(1, max_problems // min(len(priority_topics), 4))
    
    for topic_info in priority_topics[:6]:
        topic = topic_info['topic']
        available = topic_problems.get(topic, [])
        
        for p in available[:problems_per_topic]:
            if len(selected) >= max_problems:
                break
            
            marks = p.get('marks', 6)
            est_minutes = round(marks * 1.5)  # IB standard
            
            selected.append({
                **p,
                'estimated_minutes': est_minutes,
                'reason': topic_info['reason'],
                'roi': topic_info['roi'],
            })
        
        if len(selected) >= max_problems:
            break
    
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DAILY PLAN — Micro-commitments
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_plan(selected_problems: list) -> list:
    """Split problems into daily micro-commitments (10-20 min each)."""
    daily_plan = []
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes-Domingo']
    problems_per_day = max(1, len(selected_problems) // 5)
    
    for i, day in enumerate(days):
        start_idx = i * problems_per_day
        end_idx = start_idx + problems_per_day
        day_problems = selected_problems[start_idx:end_idx]
        
        if day_problems:
            total_mins = sum(p.get('estimated_minutes', 10) for p in day_problems)
            daily_plan.append({
                'day': day,
                'problems': [{'ref': p['ref'], 'topic': p['topic'], 'marks': p['marks']} for p in day_problems],
                'estimated_minutes': total_mins,
                'action': f"{len(day_problems)} problema(s) — {total_mins} min",
            })
    
    # Remaining problems go to weekend
    remaining_idx = len(days) * problems_per_day
    if remaining_idx < len(selected_problems) and daily_plan:
        remaining = selected_problems[remaining_idx:]
        daily_plan[-1]['problems'].extend([{'ref': p['ref'], 'topic': p['topic'], 'marks': p['marks']} for p in remaining])
        daily_plan[-1]['estimated_minutes'] += sum(p.get('estimated_minutes', 10) for p in remaining)
    
    return daily_plan


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: GET /api/study-plan/generate
# Main Agent 4 endpoint — generates personalized plan with all features
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/study-plan/generate")
def generate_study_plan_v2():
    """
    Agent 4 — Personalized Study Plan Generator
    
    Features:
    - ROI-based topic prioritization
    - CRITICAL / WILDCARD classification
    - Countdown phase strategy
    - Quick/Deep mode
    - What-If predictor
    - Specific problem selection
    - Integrity check
    
    Params:
    - email: student email (professors can query any student)
    - mode: 'quick' (5-6 problems) or 'deep' (8-10 problems)
    - days: override days to exam (optional)
    """
    # Try to get user from auth, but don't require it
    email = request.args.get("email", "").strip()
    mode = request.args.get("mode", "quick").lower()
    days_override = request.args.get("days", "")
    
    # If no email provided, try to get from auth token
    if not email:
        try:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                if AUTH_AVAILABLE and token:
                    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
                    email = payload.get("email", "")
        except:
            pass
    
    if not email:
        return jsonify({"error": "Missing email parameter"}), 400
    
    if mode not in ('quick', 'deep'):
        mode = 'quick'
    
    try:
        # ── 1. CALCULATE DAYS TO EXAM ─────────────────────────────
        exam_date = datetime(2026, 4, 28)
        if days_override and days_override.isdigit():
            days_to_exam = int(days_override)
        else:
            days_to_exam = max(0, (exam_date - datetime.utcnow()).days)
        
        phase = get_countdown_phase(days_to_exam)
        
        # ── 2. FETCH STUDENT ACTIVITY ─────────────────────────────
        activities = []
        if ACTIVITY_DB_ID:
            activities = query_database(
                ACTIVITY_DB_ID,
                filter_obj={"property": "user_email", "email": {"equals": email}},
                sorts=[{"property": "timestamp", "direction": "descending"}],
            )
        
        completed = [a for a in activities if a.get("action") in ("completed", "submitted")]
        
        # ── 3. AGGREGATE BY TOPIC ─────────────────────────────────
        topic_data = {}
        total_marks_awarded = 0
        total_marks_possible = 0
        error_types = []
        
        for a in completed:
            topic = a.get("topic") or a.get("key_concept") or ""
            if isinstance(topic, list):
                topic = topic[0] if topic else ""
            
            # Normalize topic to IB code (A.1, B.2, etc.)
            topic_code = ""
            for code in TOPIC_WEIGHTS.keys():
                if code.lower() in str(topic).lower() or TOPIC_NAMES.get(code, "").lower() in str(topic).lower():
                    topic_code = code
                    break
            if not topic_code:
                topic_code = topic[:3] if len(topic) >= 3 else "Other"
            
            marks_awarded = a.get("marks_awarded") or 0
            marks_total = a.get("marks_total") or a.get("marks") or 1
            
            if topic_code not in topic_data:
                topic_data[topic_code] = {'marks_awarded': 0, 'marks_total': 0, 'n': 0, 'scores': []}
            
            topic_data[topic_code]['marks_awarded'] += marks_awarded
            topic_data[topic_code]['marks_total'] += marks_total
            topic_data[topic_code]['n'] += 1
            if a.get('score'):
                topic_data[topic_code]['scores'].append(a['score'])
            
            total_marks_awarded += marks_awarded
            total_marks_possible += marks_total
            
            if a.get('error_type'):
                error_types.append(a['error_type'])
        
        # Calculate mark_rate per topic
        for topic, data in topic_data.items():
            data['mark_rate'] = data['marks_awarded'] / data['marks_total'] if data['marks_total'] > 0 else 0
        
        # ── 4. CALCULATE ROI ──────────────────────────────────────
        priority_topics = calculate_topic_roi(topic_data)
        
        # ── 5. FETCH PROBLEMS DB ──────────────────────────────────
        problems = []
        cache_key = "problems_all"
        cached = cache_get(cache_key)
        if cached:
            problems = cached
        elif NOTION_DB_ID:
            problems = query_database(NOTION_DB_ID)
            cache_set(cache_key, problems)
        
        # ── 6. SELECT PROBLEMS ────────────────────────────────────
        completed_refs = {a.get('problem_reference', '') for a in completed if a.get('score', 0) >= 70}
        selected_problems = select_problems_for_plan(priority_topics, problems, completed_refs, phase, mode)
        
        # ── 7. BUILD DAILY PLAN ───────────────────────────────────
        daily_plan = build_daily_plan(selected_problems)
        
        # ── 8. WHAT-IF PREDICTOR ──────────────────────────────────
        what_if = calculate_what_if(total_marks_awarded, total_marks_possible, selected_problems)
        
        # ── 9. INTEGRITY CHECK ────────────────────────────────────
        integrity = check_integrity(completed)
        
        # ── 10. PREDICTED IB ──────────────────────────────────────
        mark_rate = total_marks_awarded / total_marks_possible if total_marks_possible > 0 else 0
        minutes_spent = sum(a.get('time_spent_seconds', 0) for a in completed) / 60
        predicted_ib = predict_ib_grade_v2(mark_rate, len(completed), len(topic_data), minutes_spent)
        
        # ── 11. DOMINANT ERROR ────────────────────────────────────
        dominant_error = Counter(error_types).most_common(1)[0][0] if error_types else None
        
        # ── 12. BUILD RESPONSE ────────────────────────────────────
        return jsonify({
            "student_email": email,
            "generated_at": datetime.utcnow().isoformat(),
            "days_to_exam": days_to_exam,
            "mode": mode,
            "phase": phase,
            
            "diagnosis": {
                "n_problems": len(completed),
                "n_topics": len(topic_data),
                "mark_rate_pct": round(mark_rate * 100, 1),
                "predicted_ib": predicted_ib,
                "dominant_error": dominant_error,
                "integrity": integrity,
            },
            
            "priority_topics": priority_topics[:8],
            
            "recommended_problems": [
                {
                    "ref": p['ref'],
                    "id": p.get('id', ''),
                    "name": p.get('name', ''),
                    "topic": p['topic'],
                    "topic_name": p['topic_name'],
                    "marks": p['marks'],
                    "difficulty": p.get('difficulty', 'medium'),
                    "estimated_minutes": p['estimated_minutes'],
                    "reason": p['reason'],
                    "roi": p['roi'],
                }
                for p in selected_problems
            ],
            
            "daily_plan": daily_plan,
            
            "what_if": what_if,
            
            "total_estimated_minutes": sum(p.get('estimated_minutes', 10) for p in selected_problems),
            
            "success_metric": f"Score ≥70% on {len(selected_problems)} problems → +{what_if['improvement']['mark_rate_delta']}% mark rate",
        }), 200
    
    except Exception as e:
        print(f"[AGENT 4 ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)[:300]}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: GET /api/professor/class-plan
# Generate class-wide summary for the professor
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/professor/class-plan")
def generate_class_plan():
    """
    Generate class-wide summary with:
    - At-risk students
    - Topics to review
    - Integrity alerts
    - Suggested class action
    """
    # No auth required - matches rest of professor dashboard
    days_override = request.args.get("days", "")
    
    try:
        exam_date = datetime(2026, 4, 28)
        if days_override and days_override.isdigit():
            days_to_exam = int(days_override)
        else:
            days_to_exam = max(0, (exam_date - datetime.utcnow()).days)
        
        phase = get_countdown_phase(days_to_exam)
        
        # Fetch all students
        all_users = []
        if USERS_DB_ID:
            all_users = query_database(
                USERS_DB_ID,
                filter_obj={"property": "role", "select": {"equals": "student"}},
            )
        
        # Fetch all activities
        all_activities = []
        if ACTIVITY_DB_ID:
            all_activities = query_database(ACTIVITY_DB_ID)
        
        # Aggregate by student
        student_data = {}
        for a in all_activities:
            email = a.get("user_email", "")
            if not email or email == "demo":
                continue
            
            if email not in student_data:
                student_data[email] = {
                    'email': email,
                    'marks_awarded': 0,
                    'marks_total': 0,
                    'n_problems': 0,
                    'activities': [],
                }
            
            if a.get("action") in ("completed", "submitted"):
                student_data[email]['marks_awarded'] += a.get("marks_awarded") or 0
                student_data[email]['marks_total'] += a.get("marks_total") or 1
                student_data[email]['n_problems'] += 1
                student_data[email]['activities'].append(a)
        
        # Calculate stats and find at-risk
        at_risk = []
        all_mark_rates = []
        integrity_alerts = []
        
        for email, data in student_data.items():
            if data['marks_total'] > 0:
                mark_rate = data['marks_awarded'] / data['marks_total']
            else:
                mark_rate = 0
            
            data['mark_rate'] = mark_rate
            all_mark_rates.append(mark_rate)
            
            # Predict IB
            pred = predict_ib_grade_v2(mark_rate, data['n_problems'], 5, 0)
            data['predicted_ib'] = pred
            
            # At-risk: IB ≤ 3 or mark_rate < 40%
            if pred.get('grade') and pred['grade'] <= 3:
                at_risk.append({
                    'email': email,
                    'predicted_ib': pred['grade'],
                    'mark_rate_pct': round(mark_rate * 100, 1),
                    'n_problems': data['n_problems'],
                })
            elif mark_rate < 0.4 and data['n_problems'] > 0:
                at_risk.append({
                    'email': email,
                    'predicted_ib': pred.get('grade'),
                    'mark_rate_pct': round(mark_rate * 100, 1),
                    'n_problems': data['n_problems'],
                })
            
            # Integrity check
            integrity = check_integrity(data['activities'])
            if integrity['level'] in ('verify', 'review'):
                integrity_alerts.append({
                    'email': email,
                    'flag': integrity['level'],
                    'pct': integrity['pct'],
                    'message': integrity['message'],
                })
        
        at_risk.sort(key=lambda x: x.get('predicted_ib') or 0)
        
        # Class averages
        class_avg = sum(all_mark_rates) / len(all_mark_rates) if all_mark_rates else 0
        
        # Aggregate topics across class
        topic_scores = {}
        for email, data in student_data.items():
            for a in data['activities']:
                topic = a.get("topic") or a.get("key_concept") or "Other"
                if isinstance(topic, list):
                    topic = topic[0] if topic else "Other"
                
                if topic not in topic_scores:
                    topic_scores[topic] = []
                if a.get('score'):
                    topic_scores[topic].append(a['score'])
        
        weak_topics = []
        for topic, scores in topic_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                if avg < 60:
                    weak_topics.append({
                        'topic': topic,
                        'class_avg': round(avg, 1),
                        'n_attempts': len(scores),
                    })
        
        weak_topics.sort(key=lambda x: x['class_avg'])
        
        return jsonify({
            "generated_at": datetime.utcnow().isoformat(),
            "days_to_exam": days_to_exam,
            "phase": phase,
            
            "class_summary": {
                "n_students": len(student_data),
                "class_avg_mark_rate": round(class_avg * 100, 1),
                "n_at_risk": len(at_risk),
            },
            
            "at_risk_students": at_risk[:10],
            
            "topics_to_review": weak_topics[:5],
            
            "integrity_alerts": integrity_alerts[:5],
            
            "suggested_action": f"Phase: {phase['label']} — {phase['focus']}. " +
                               (f"Focus on {weak_topics[0]['topic']} this week." if weak_topics else "All topics on track."),
        }), 200
    
    except Exception as e:
        print(f"[CLASS PLAN ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)[:300]}), 500


# ══════════════════════════════════════════════════════════════════════════════
# END OF AGENT 4 CODE — Add this before if __name__ == "__main__":
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
