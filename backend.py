# backend.py
# Flask API for Physics AI Coach
# - GET /               -> health
# - GET /problems       -> list problems from Notion (lightweight)
# - GET /problem/<id>   -> full problem
# - GET /steps/<id>     -> step plan (from Notion or auto-generated). Nunca 500.
# - POST /chat          -> evalúa el paso del alumno y da feedback/siguiente pista

import os, re, json
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# ---------- ENV ----------
NOTION_API_KEY    = os.environ.get("NOTION_API_KEY", "")
NOTION_DB_ID      = os.environ.get("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRONTEND_ORIGIN   = os.environ.get("FRONTEND_ORIGIN")  # ej: https://aiclub.com.mx

# ---------- Flask ----------
app = Flask(__name__)
if FRONTEND_ORIGIN:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_ORIGIN]}})
else:
    # desarrollo: permitir todos (puedes endurecerlo luego)
    CORS(app)

# ---------- Notion helpers ----------
NOTION_BASE = "https://api.notion.com/v1"
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

def _rt_to_plain(rt: Any) -> str:
    if not rt:
        return ""
    parts = []
    for x in rt:
        if isinstance(x, dict):
            parts.append(x.get("plain_text") or x.get("text", {}).get("content", ""))
    return "".join(parts).strip()

def _get_prop(props: Dict[str, Any], key: str) -> str:
    p = props.get(key)
    if not p:
        return ""
    t = p.get("type")
    if t == "title":
        return _rt_to_plain(p.get("title"))
    if t == "rich_text":
        return _rt_to_plain(p.get("rich_text"))
    if t == "select":
        sl = p.get("select")
        return (sl or {}).get("name", "") if sl else ""
    if t == "multi_select":
        return ", ".join([x.get("name","") for x in p.get("multi_select", [])])
    return ""

def fetch_problems() -> List[Dict[str, Any]]:
    url = f"{NOTION_BASE}/databases/{NOTION_DB_ID}/query"
    acc: List[Dict[str, Any]] = []
    payload = {"page_size": 100}
    while True:
        r = requests.post(url, headers=NOTION_HEADERS, data=json.dumps(payload))
        r.raise_for_status()
        data = r.json()
        for page in data.get("results", []):
            props = page.get("properties", {})
            acc.append({
                "id": page.get("id"),
                "name": _get_prop(props, "Name") or _get_prop(props, "name") or "Untitled",
                "difficulty": _get_prop(props, "difficulty") or _get_prop(props, "Difficulty"),
            })
        if not data.get("has_more"): break
        payload["start_cursor"] = data.get("next_cursor")
    return acc

def fetch_page(page_id: str) -> Dict[str, Any]:
    url = f"{NOTION_BASE}/pages/{page_id}"
    r = requests.get(url, headers=NOTION_HEADERS); r.raise_for_status()
    page = r.json(); props = page.get("properties", {})
    return {
        "id": page.get("id"),
        "name": _get_prop(props, "Name") or _get_prop(props, "name") or "Untitled",
        "difficulty": _get_prop(props, "difficulty") or _get_prop(props, "Difficulty"),
        "problem_statement": _get_prop(props, "problem_statement"),
        "given_values": _get_prop(props, "given_values"),
        "find": _get_prop(props, "find"),
        "key_concepts": _get_prop(props, "key_concepts"),
        "common_mistakes": _get_prop(props, "common_mistakes"),
        "final_answer": _get_prop(props, "final_answer"),
        "step_by_step": _get_prop(props, "step_by_step"),
    }

# ---------- Steps parsing / generation ----------
STEP_RE = re.compile(r"^\s*(\d+)[\).:-]\s*(.+)$")

def parse_steps(raw: str) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if not raw: return steps
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        m = STEP_RE.match(line)
        if m:
            sid, rest = m.group(1), m.group(2)
            rubric = ""
            if "| rubric:" in rest:
                part, rub = rest.split("| rubric:", 1)
                text = part.strip(); rubric = rub.strip()
            else:
                text = rest.strip()
            steps.append({"id": sid, "text": text, "rubric": rubric})
    if not steps and raw.strip():
        steps.append({"id":"1","text":raw.strip(),"rubric":""})
    return steps

def anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=ANTHROPIC_API_KEY)

ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"  # robusto; evita 404 por alias viejo

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    if not ANTHROPIC_API_KEY:
        return []
    try:
        client = anthropic_client()
        prompt = f"""
You are a patient IB Physics tutor. Break this problem into 4–6 short steps.
Each step: one concrete action. After it add ' | rubric: ...' with what to check.
Return plain text lines ONLY:

1) ...
2) ...
3) ...

Problem:
Statement: {problem.get('problem_statement')}
Given: {problem.get('given_values')}
Find: {problem.get('find')}
Key concepts: {problem.get('key_concepts')}
"""
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            temperature=0.2,
            messages=[{"role":"user","content":prompt}]
        )
        text = ""
        for b in msg.content:
            if getattr(b, "type", "") == "text":
                text += b.text
        return parse_steps(text)
    except Exception:
        # Nunca tumbar el endpoint por fallas del proveedor
        return []

# ---------- Routes ----------
@app.get("/")
def health():
    return jsonify({"ok": True})

@app.get("/problems")
def problems():
    try:
        return jsonify({"problems": fetch_problems()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/problem/<page_id>")
def problem_detail(page_id):
    try:
        return jsonify({"problem": fetch_page(page_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/steps/<page_id>")
def steps(page_id):
    """
    Devuelve siempre 200 con el plan (aunque sea [] si no se pudo generar).
    Frontend puede mostrar “No steps generated yet”.
    """
    info = {"mode": "tutor", "steps": [], "note": ""}
    try:
        p = fetch_page(page_id)
        raw = p.get("step_by_step","")
        s = parse_steps(raw)
        if not s:
            s = generate_steps_with_llm(p)
            if not s:
                info["note"] = "Could not auto-generate steps (LLM unavailable or model not enabled)."
        info["steps"] = s
        return jsonify(info)
    except Exception as e:
        # Última defensa: tampoco 500 aquí
        info["note"] = f"Error reading steps: {e}"
        return jsonify(info)

@app.post("/chat")
def chat():
    """
    Body:
    {
      "problem_id": "...", "step_id": "1",
      "student_answer": "text", "mode": "tutor"|"student"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    step_id = str(data.get("step_id") or "1")
    student_answer = (data.get("student_answer") or "").strip()

    if not problem_id:
        return jsonify({"error":"missing problem_id"}), 400

    p = fetch_page(problem_id)
    steps_list = parse_steps(p.get("step_by_step","")) or generate_steps_with_llm(p)

    current = None
    for s in steps_list:
        if s.get("id") == step_id:
            current = s; break
    if not current and steps_list:
        current = steps_list[0]
        step_id = current["id"]

    if not ANTHROPIC_API_KEY:
        return jsonify({
            "ok": True, "feedback":"(No LLM) Answer received.",
            "next_step": str(int(step_id)+1) if step_id.isdigit() else "",
            "message":"Move to the next step."
        })

    # Construir prompt
    rub = (current or {}).get("rubric","")
    eval_prompt = f"""
You are an IB Physics tutor. Evaluate the student's response for this step.

Problem (short):
Statement: {p.get('problem_statement')}
Given: {p.get('given_values')}
Find: {p.get('find')}

Current step {step_id}:
"{(current or {}).get('text','')}"
Grading rubric (if any): {rub or "(none)"}

Student answer:
\"\"\"{student_answer}\"\"\"

Return JSON ONLY with keys:
ok (true/false), feedback (string), next_hint (string), ready_to_advance (true/false)
"""

    try:
        client = anthropic_client()
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            temperature=0.2,
            messages=[{"role":"user","content":eval_prompt}]
        )
        text = ""
        for b in msg.content:
            if getattr(b, "type", "") == "text":
                text += b.text
        try:
            parsed = json.loads(text.strip())
        except Exception:
            parsed = {"ok": False, "feedback": text, "ready_to_advance": False, "next_hint": ""}

        advance = parsed.get("ready_to_advance") or parsed.get("ok")
        next_step = str(int(step_id) + 1) if (advance and step_id.isdigit()) else step_id
        message = parsed.get("next_hint") if not advance else "Great! Let's go on."

        return jsonify({
            "ok": bool(parsed.get("ok") or parsed.get("ready_to_advance")),
            "feedback": parsed.get("feedback",""),
            "next_step": next_step,
            "message": message
        })
    except Exception as e:
        # No romper UX si Anthropic falla
        return jsonify({
            "ok": False,
            "feedback": f"(Tutor offline) I couldn't evaluate this step: {e}",
            "next_step": step_id,
            "message": "Try rechecking or move on."
        })

# ---------- Gunicorn entry ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

