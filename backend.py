import os
import json
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

# ========= Config =========
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_DB_ID   = os.getenv("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # e.g. https://aiclub.com.mx

# ========= App =========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})

# ========= External Clients =========
# Anthropic
try:
    import anthropic
except Exception:
    anthropic = None

def get_claude():
    if anthropic is None:
        raise RuntimeError("anthropic package not installed. Add 'anthropic' to requirements.txt")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return client

# Notion (simple via requests)
NOTION_BASE = "https://api.notion.com/v1"
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def notion_query_database(db_id: str, **kwargs) -> Dict[str, Any]:
    url = f"{NOTION_BASE}/databases/{db_id}/query"
    r = requests.post(url, headers=NOTION_HEADERS, json=kwargs, timeout=30)
    r.raise_for_status()
    return r.json()

def notion_retrieve_page(page_id: str) -> Dict[str, Any]:
    url = f"{NOTION_BASE}/pages/{page_id}"
    r = requests.get(url, headers=NOTION_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

# ========= Helpers =========
def rich_to_text(rich: List[Dict[str,Any]]) -> str:
    return " ".join([(x.get("plain_text") or "").strip() for x in (rich or [])]).strip()

def get_prop_text(props: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in props:
            p = props[k]
            t = p.get("type")
            if t == "title":
                return rich_to_text(p.get("title", []))
            if t == "rich_text":
                return rich_to_text(p.get("rich_text", []))
            if t == "number":
                v = p.get("number")
                return "" if v is None else str(v)
            if t == "select":
                x = p.get("select")
                return (x or {}).get("name","")
            if t == "multi_select":
                arr = p.get("multi_select") or []
                return ", ".join([(x or {}).get("name","") for x in arr])
            if t == "url":
                return p.get("url") or ""
            if t == "checkbox":
                return "true" if p.get("checkbox") else "false"
            if t == "people":
                arr = p.get("people", [])
                return ", ".join([x.get("name","") for x in arr if isinstance(x, dict)])
            if t == "email":
                return p.get("email") or ""
            if t == "phone_number":
                return p.get("phone_number") or ""
            if t == "date":
                dt = p.get("date", {})
                return (dt.get("start") or "") + (" — " + (dt.get("end") or "") if dt.get("end") else "")
            # fallback for any plain_text-like
            v = p.get("plain_text")
            if isinstance(v, str):
                return v.strip()
    return ""

def extract_number_and_units(s: str) -> Tuple[Optional[float], Optional[str]]:
    if not s:
        return None, None
    # crude extractor (supports scientific notation)
    m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z·/*^μΩ%°\-\s]+)?", s)
    if not m:
        return None, None
    try:
        val = float(m.group(1))
    except Exception:
        val = None
    units = (m.group(2) or "").strip() or None
    return val, units

def close_enough(a: float, b: float, tol_pct: float = 1.0) -> bool:
    try:
        tol = max(abs(b) * tol_pct / 100.0, 1e-12)
        return abs(a - b) <= tol
    except Exception:
        return False

# ========= Problem DTOs =========
def normalize_page(page: Dict[str, Any]) -> Dict[str, Any]:
    props = page.get("properties", {})
    return {
        "id": page.get("id"),
        "title": get_prop_text(props, ["Name","name","Title","title"]),
        "topic": get_prop_text(props, ["topic","Topic","subject","Subject"]),
        "difficulty": get_prop_text(props, ["difficulty","Difficulty"]),
    }

def full_problem(page_id: str) -> Dict[str, Any]:
    page = notion_retrieve_page(page_id)
    props = page.get("properties", {})
    return {
        "id": page.get("id"),
        "title": get_prop_text(props, ["Name","name","Title","title"]),
        "topic": get_prop_text(props, ["topic","Topic"]),
        "difficulty": get_prop_text(props, ["difficulty","Difficulty"]),
        "statement": get_prop_text(props, ["problem_statement","Statement"]),
        "given": get_prop_text(props, ["given_values","Given"]),
        "find": get_prop_text(props, ["find","Find"]),
        "key_concepts": get_prop_text(props, ["key_concepts","Key Concepts"]),
        "common_mistakes": get_prop_text(props, ["common_mistakes","Common mistakes"]),
        "final_answer": get_prop_text(props, ["final_answer","Final Answer"]),
        "full_solution": get_prop_text(props, ["full_solution","Solution"]),
    }

# ========= Routes =========
@app.get("/")
def ping():
    return jsonify({"ok": True})

@app.get("/problems")
def list_problems():
    if not NOTION_API_KEY or not NOTION_DB_ID:
        return jsonify({"error":"Missing Notion credentials"}), 500
    results: List[Dict[str,Any]] = []
    payload = {}
    while True:
        data = notion_query_database(NOTION_DB_ID, **payload)
        for page in data.get("results", []):
            results.append(normalize_page(page))
        if not data.get("has_more"):
            break
        payload = {"start_cursor": data.get("next_cursor")}
    # order by title asc
    results.sort(key=lambda x: (x.get("title") or "").lower())
    return jsonify(results)

@app.get("/problems/<page_id>")
def get_problem(page_id):
    try:
        return jsonify(full_problem(page_id))
    except requests.HTTPError as e:
        return jsonify({"error": str(e), "details": getattr(e, "response", None).text if hasattr(e, "response") else ""}), 500

# ========= Tutor Chat / Hint / Check =========
TUTOR_SYSTEM = (
    "You are a patient IB Physics tutor. Explain with precise equations, units, "
    "and insightful guidance. Keep answers compact unless asked to go deeper."
)

def build_context_for(page_id: str) -> str:
    p = full_problem(page_id)
    ctx = (
        f"Title: {p.get('title')}\n"
        f"Topic: {p.get('topic')}\n"
        f"Difficulty: {p.get('difficulty')}\n\n"
        f"Statement: {p.get('statement')}\n"
        f"Given: {p.get('given')}\n"
        f"Find: {p.get('find')}\n"
        f"Final Answer (if present): {p.get('final_answer')}\n"
    )
    return ctx

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    message = (data.get("message") or "").strip()
    if not page_id or not message:
        return jsonify({"error":"Missing problemId or message"}), 400
    try:
        client = get_claude()
        ctx = build_context_for(page_id)
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=700,
            system=TUTOR_SYSTEM,
            messages=[
                {"role": "user", "content": ctx + "\nStudent says:\n" + message}
            ]
        )
        text = "".join([p.text for p in msg.content if getattr(p, "type", "")=="text"]).strip()
        return jsonify({"answer": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/hint")
def hint():
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    note = (data.get("note") or "").strip()
    if not page_id:
        return jsonify({"error":"Missing problemId"}), 400
    try:
        client = get_claude()
        ctx = build_context_for(page_id)
        prompt = ctx + "\nGive a single helpful hint (one or two sentences). If the student's note is present, tailor to it.\nStudent note: " + note
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=250,
            system=TUTOR_SYSTEM,
            messages=[{"role":"user","content": prompt}]
        )
        text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"]).strip()
        return jsonify({"hint": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/check")
def check():
    """Lightweight numeric checker against final_answer with LLM fallback."""
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    answer = (data.get("answer") or "").strip()
    if not page_id or not answer:
        return jsonify({"error":"Missing problemId or answer"}), 400
    p = full_problem(page_id)
    final_answer = p.get("final_answer") or ""
    stu_val, stu_units = extract_number_and_units(answer)
    fin_val, fin_units = extract_number_and_units(final_answer)
    if (stu_val is not None) and (fin_val is not None):
        ok = close_enough(stu_val, fin_val, tol_pct=1.0)  # 1% by default
        units_ok = (not fin_units) or ((stu_units or "").strip().lower()==(fin_units or "").strip().lower())
        fb = "✅ Correct!" if (ok and units_ok) else f"❌ Not yet. Expected about {fin_val:g} {fin_units or ''} (±1%)."
        return jsonify({"correct": ok and units_ok, "feedback": fb})

    # fall back to tutor judgment
    try:
        client = get_claude()
        ctx = build_context_for(page_id)
        prompt = ctx + f"\nJudge if the student's final numeric answer is correct (tolerance 1%). Answer 'CORRECT' or 'INCORRECT' and then one helpful sentence.\nStudent answer: {answer}"
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=200,
            system=TUTOR_SYSTEM,
            messages=[{"role":"user","content": prompt}]
        )
        text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"]).strip()
        verdict = "CORRECT" in text.upper()
        fb = ("✅ Correct. " if verdict else "❌ Not yet. ") + text
        return jsonify({"correct": verdict, "feedback": fb})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========= Step-by-step engine (robust with fallback) =========
STEPS_SYSTEM = (
    "You are a rigorous yet friendly physics tutor. Split the problem into 3–10 "
    "small, sequential steps. Each step is one actionable move. If a step expects "
    "a number, include 'expects_numeric': true, an 'expected_value' (float), "
    "a 'tolerance_pct' (e.g., 1.0), and 'expected_units' when relevant. "
    "If numeric target can't be determined, set expects_numeric=false. "
    "Return ONLY valid JSON with this schema: "
    "{\"steps\":[{\"title\":str,\"instruction\":str,\"expects_numeric\":bool,"
    "\"expected_value\":number|null,\"tolerance_pct\":number|null,\"expected_units\":str|null}]}"
)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

@lru_cache(maxsize=256)
def plan_steps_for(page_id: str) -> List[Dict[str,Any]]:
    p = full_problem(page_id)
    base_user = (
        f"Problem title: {p.get('title')}\n"
        f"Statement: {p.get('statement')}\n"
        f"Given: {p.get('given')}\n"
        f"Find: {p.get('find')}\n"
        f"Verified final answer (if present): {p.get('final_answer')}\n"
    )
    full_solution = p.get("full_solution")
    if full_solution:
        guidance = "Use the verified full solution to derive precise steps."
        base_user += "Teacher's verified full solution:\n" + full_solution + "\n"
    else:
        guidance = (
            "The teacher's detailed solution is NOT available. "
            "Infer the steps from physics principles so the student can reach the final answer."
        )

    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=900,
        system=STEPS_SYSTEM,
        messages=[{"role":"user","content": base_user + guidance}]
    )
    text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"]).strip()
    try:
        data = json.loads(text)
        steps = data.get("steps", [])
    except Exception:
        steps = []

    # emergency scaffold
    if not steps:
        fin_val, fin_units = extract_number_and_units(p.get("final_answer") or "")
        steps = [
            {"title":"Identify knowns","instruction":"List all given values with units.","expects_numeric":False,"expected_value":None,"tolerance_pct":None,"expected_units":None},
            {"title":"Choose governing principle","instruction":"State the law/equation connecting unknown(s) to given quantities.","expects_numeric":False,"expected_value":None,"tolerance_pct":None,"expected_units":None},
            {"title":"Set up equation","instruction":"Write the equation with symbols, then substitute numerical values (keep units).","expects_numeric":False,"expected_value":None,"tolerance_pct":None,"expected_units":None},
            {"title":"Compute","instruction":"Compute the unknown with proper significant figures.","expects_numeric":True,"expected_value":_safe_float(fin_val),"tolerance_pct":1.0,"expected_units":fin_units or None},
        ]

    cleaned = []
    for s in steps:
        cleaned.append({
            "title": str(s.get("title","Step")),
            "instruction": str(s.get("instruction","")),
            "expects_numeric": bool(s.get("expects_numeric", False)),
            "expected_value": _safe_float(s.get("expected_value")) if s.get("expected_value") is not None else None,
            "tolerance_pct": _safe_float(s.get("tolerance_pct")) if s.get("tolerance_pct") is not None else None,
            "expected_units": (s.get("expected_units") or None)
        })
    return cleaned

@app.get("/steps/<page_id>")
def get_steps(page_id):
    try:
        steps = plan_steps_for(page_id)
        return jsonify({"steps": steps, "count": len(steps)})
    except Exception as e:
        return jsonify({"steps": [], "count": 0, "error": str(e)}), 500

@app.post("/grade_step")
def grade_step():
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    idx = int(data.get("stepIndex", 0))
    student = (data.get("work") or "").strip()
    if not page_id:
        return jsonify({"error":"Missing problemId"}), 400
    steps = plan_steps_for(page_id)
    if idx < 0 or idx >= len(steps):
        return jsonify({"error":"Invalid step index"}), 400
    step = steps[idx]

    # numeric check first
    if step.get("expects_numeric"):
        stu_val, stu_units = extract_number_and_units(student)
        exp_val = step.get("expected_value")
        tol_pct = step.get("tolerance_pct") or 1.0
        exp_units = (step.get("expected_units") or "").strip().lower() or None
        if (stu_val is not None) and (exp_val is not None):
            tol = abs(exp_val) * tol_pct / 100.0
            numeric_ok = abs(stu_val - exp_val) <= max(tol, 1e-12)
            units_ok = (not exp_units) or ((stu_units or "").strip().lower()==exp_units)
            ok = numeric_ok and units_ok
            fb = "✅ Correct step" if ok else f"❌ Check your value: expected ≈ {exp_val:g} {exp_units or ''} (±{tol_pct}%)."
            return jsonify({"correct": ok, "feedback": fb})

    # semantic LLM check
    try:
        client = get_claude()
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=200,
            system="Judge only this step; respond 'CORRECT' or 'INCORRECT' and one actionable sentence.",
            messages=[{"role":"user","content": f"Step instruction: {step.get('instruction')}\nStudent work: {student}"}]
        )
        text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"])
        verdict = "CORRECT" in text.upper()
        fb = ("✅ Correct" if verdict else "❌ Not yet") + " — " + text.strip()
        return jsonify({"correct": verdict, "feedback": fb})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========= Run local =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

