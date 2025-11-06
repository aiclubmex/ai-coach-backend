# backend.py
# Flask API for Physics AI Coach
# - GET  /                 -> health
# - GET  /problems         -> list problems from Notion (id, name, difficulty)
# - GET  /problem/<id>     -> full problem detail
# - GET  /problem?id=...   -> alias for the above
# - GET  /problems/<id>    -> alias for the above
# - GET  /steps/<id>       -> step-by-step plan (from Notion or auto-generated)
# - POST /generate-steps   -> alias that returns {"steps":[...]} from id
# - POST /chat             -> evaluate a student's step / give hint / free chat
# - POST /check-step       -> alias that forwards to /chat

import os, re, json
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# ---------- ENV ----------
NOTION_API_KEY    = os.environ.get("NOTION_API_KEY", "")
NOTION_DB_ID      = os.environ.get("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FRONTEND_ORIGIN   = os.environ.get("FRONTEND_ORIGIN")  # e.g., https://aiclub.com.mx
NOTION_TIMEOUT    = int(os.environ.get("NOTION_TIMEOUT_SEC", "15"))

# ---------- Flask ----------
app = Flask(__name__)
if FRONTEND_ORIGIN:
    CORS(app, resources={r"/*": {"origins": [FRONTEND_ORIGIN]}})
else:
    CORS(app)  # dev: allow all

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
            parts.append(x.get("plain_text") or (x.get("text") or {}).get("content", ""))
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
        sel = p.get("select")
        return (sel or {}).get("name", "") if sel else ""
    if t == "multi_select":
        return ", ".join([x.get("name", "") for x in p.get("multi_select", [])])
    if t == "number":
        return str(p.get("number") or "")
    return ""

def fetch_problems() -> List[Dict[str, Any]]:
    url = f"{NOTION_BASE}/databases/{NOTION_DB_ID}/query"
    out: List[Dict[str, Any]] = []
    payload = {"page_size": 100}
    while True:
        r = requests.post(url, headers=NOTION_HEADERS, data=json.dumps(payload), timeout=NOTION_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for page in data.get("results", []):
            props = page.get("properties", {})
            out.append({
                "id": page.get("id"),
                "name": _get_prop(props, "Name") or _get_prop(props, "name") or "Untitled",
                "difficulty": _get_prop(props, "difficulty") or _get_prop(props, "Difficulty"),
                # extra fields (para filtros/heurísticas del front)
                "problem_statement": _get_prop(props, "problem_statement"),
                "given_values": _get_prop(props, "given_values"),
                "find": _get_prop(props, "find"),
                "key_concepts": _get_prop(props, "key_concepts"),
                "common_mistakes": _get_prop(props, "common_mistakes"),
                "final_answer": _get_prop(props, "final_answer"),
                "step_by_step": _get_prop(props, "step_by_step"),
            })
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data.get("next_cursor")
    return out

def fetch_page(page_id: str) -> Dict[str, Any]:
    url = f"{NOTION_BASE}/pages/{page_id}"
    r = requests.get(url, headers=NOTION_HEADERS, timeout=NOTION_TIMEOUT)
    r.raise_for_status()
    page = r.json()
    props = page.get("properties", {})
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

# ---------- Steps (igual a tu versión que funcionaba) ----------
STEP_RE = re.compile(r"^\s*(\d+)[\).:-]\s*(.+)$")

def parse_steps(raw: str) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if not raw:
        return steps
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = STEP_RE.match(line)
        if m:
            sid = m.group(1)
            rest = m.group(2)
            rubric = ""
            if "| rubric:" in rest:
                part, rub = rest.split("| rubric:", 1)
                text = part.strip()
                rubric = rub.strip()
            else:
                text = rest.strip()
            steps.append({"id": sid, "text": text, "rubric": rubric})
    # fallback: si el campo tiene texto pero sin numeración, al menos un paso
    if not steps and raw.strip():
        steps.append({"id": "1", "text": raw.strip(), "rubric": ""})
    return steps

# ---------- Anthropic (SDK oficial como el tuyo) ----------
def anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    if not ANTHROPIC_API_KEY:
        return []
    client = anthropic_client()
    prompt = f"""
You are a patient IB Physics tutor. Break this problem into 4–6 small steps.
Each step is an action the student should do next.
Append a short grading rubric after a pipe: " | rubric: ...".
Return plain text lines like:
1) ...
2) ...
3) ...

Problem:
Statement: {problem.get('problem_statement')}
Given: {problem.get('given_values')}
Find: {problem.get('find')}
Key concepts: {problem.get('key_concepts')}
"""
    try:
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for b in msg.content:
            if getattr(b, "type", "") == "text":
                text += b.text
        return parse_steps(text)
    except Exception as e:
        print(f"[LLM steps error] {e}")
        # fallback simple para no romper el front
        return [
            {"id": "1", "text": "Read and understand the problem", "rubric": ""},
            {"id": "2", "text": "Identify relevant physics concepts", "rubric": ""},
            {"id": "3", "text": "Set up equations", "rubric": ""},
            {"id": "4", "text": "Solve", "rubric": ""},
        ]

# ---------- Routes ----------
@app.get("/")
def health():
    return jsonify({"ok": True})

@app.get("/problems")
def problems():
    try:
        items = fetch_problems()
        simplified = [{"id": p["id"], "name": p["name"], "difficulty": p["difficulty"]} for p in items]
        return jsonify({"problems": simplified})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/problem/<page_id>")
def problem_detail(page_id):
    try:
        p = fetch_page(page_id)
        return jsonify({"problem": p})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# alias para tu WP si llamas /problem?id=... o /problems/<id>
@app.get("/problem")
def problem_detail_query():
    pid = request.args.get("id")
    if not pid:
        return jsonify({"error": "missing id"}), 400
    try:
        p = fetch_page(pid)
        return jsonify({"problem": p})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/problems/<pid>")
def problem_detail_alt(pid):
    try:
        p = fetch_page(pid)
        return jsonify({"problem": p})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/steps/<page_id>")
def steps(page_id):
    try:
        p = fetch_page(page_id)
        raw = p.get("step_by_step", "")
        steps_list = parse_steps(raw)
        if not steps_list:
            steps_list = generate_steps_with_llm(p)
        return jsonify({"mode": "tutor", "steps": steps_list})
    except Exception as e:
        print(f"[steps error] {e}")
        # no 500: respuesta mínima para que el UI no muera
        return jsonify({"mode": "tutor", "steps": [
            {"id": "1", "text": "Read and understand the problem", "rubric": ""},
            {"id": "2", "text": "Identify relevant physics concepts", "rubric": ""},
            {"id": "3", "text": "Set up equations", "rubric": ""},
            {"id": "4", "text": "Solve", "rubric": ""},
        ]}), 200

@app.route("/generate-steps", methods=["OPTIONS", "POST"])
def generate_steps_alias():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    pid = data.get("problem_id") or request.args.get("problem_id") or request.args.get("id")
    if not pid:
        return jsonify({"error": "missing problem_id"}), 400
    try:
        p = fetch_page(pid)
        raw = p.get("step_by_step", "")
        steps_list = parse_steps(raw)
        if not steps_list:
            steps_list = generate_steps_with_llm(p)
        return jsonify({"steps": steps_list})
    except Exception as e:
        print(f"[generate-steps error] {e}")
        return jsonify({"steps": [
            {"id": "1", "text": "Read and understand the problem", "rubric": ""},
            {"id": "2", "text": "Identify relevant physics concepts", "rubric": ""},
            {"id": "3", "text": "Set up equations", "rubric": ""},
            {"id": "4", "text": "Solve", "rubric": ""},
        ]}), 200

@app.post("/chat")
def chat():
    """
    Handle student interactions:
    - mode='tutor': Evaluate student's step
    - mode='hint': Provide a hint
    - mode='student': Free chat about the problem
    """
    data = request.get_json(force=True, silent=True) or {}
    problem_id = data.get("problem_id")
    step_id = str(data.get("step_id") or "")
    student_answer = (data.get("student_answer") or "").strip()
    mode = data.get("mode", "tutor")
    message_text = (data.get("message") or "").strip()

    if not problem_id:
        return jsonify({"error": "missing problem_id"}), 400

    # Cargamos problema y pasos DESDE NOTION (como lo tenías)
    try:
        p = fetch_page(problem_id)
    except Exception as e:
        p = {"id": problem_id}
        print(f"[WARN] fetch_page error: {e}")

    steps_list = parse_steps(p.get("step_by_step", ""))
    if not steps_list:
        steps_list = generate_steps_with_llm(p)

    # Paso actual
    current = None
    for s in steps_list:
        if s["id"] == step_id:
            current = s
            break
    if not current and steps_list:
        current = steps_list[0]
        step_id = current["id"]

    # Si no hay API key, NO rompemos
    if not ANTHROPIC_API_KEY:
        ok = (mode != "tutor") or (len(student_answer) > 5)
        return jsonify({
            "ok": ok,
            "feedback": "(No LLM) Answer received.",
            "next_step": str(int(step_id) + 1) if (ok and step_id.isdigit()) else step_id,
            "message": "Move to the next step." if ok else "Try adding the formula and units.",
            "reply": "(No LLM)"
        })

    # --- Modo student (free chat) ---
    if mode == "student" and message_text:
        prompt = f"""
You are an IB Physics tutor helping with this problem:

Statement: {p.get('problem_statement')}
Given: {p.get('given_values')}
Find: {p.get('find')}

Student asks: {message_text}

Provide a brief, helpful response (2-3 sentences). Guide them towards understanding without giving the full answer.
"""
        try:
            client = anthropic_client()
            msg = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=300,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for b in msg.content:
                if getattr(b, "type", "") == "text":
                    text += b.text
            return jsonify({"reply": text.strip() or "Let’s focus on the key formula and units for this step."})
        except Exception as e:
            print(f"[chat student error] {e}")
            return jsonify({"reply": "I couldn't reach the AI right now, but try focusing on the key formula and units for this step."})

    # --- Modo hint ---
    if mode == "hint":
        prompt = f"""
You are an IB Physics tutor. Provide ONE helpful hint for this step, 1-2 sentences, not the full answer.

Current step {step_id}: "{current.get('text')}"
"""
        try:
            client = anthropic_client()
            msg = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=200,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for b in msg.content:
                if getattr(b, "type", "") == "text":
                    text += b.text
            return jsonify({"hint": text.strip() or "Focus on the relevant formula and check units."})
        except Exception as e:
            print(f"[chat hint error] {e}")
            return jsonify({"hint": f"Think about the concepts needed for: {current.get('text')}"}), 200

    # --- Modo tutor (evaluación del paso) ---
    if mode == "tutor":
        if not student_answer:
            return jsonify({
                "ok": False,
                "feedback": "Please provide an answer to evaluate.",
                "next_step": step_id,
                "message": "Write your reasoning above."
            })
        rub = current.get("rubric", "")
        eval_prompt = f"""
You are an IB Physics tutor. Evaluate the student's response JUST for this step.

Problem (short):
Statement: {p.get('problem_statement')}
Given: {p.get('given_values')}
Find: {p.get('find')}

Current step {step_id}:
"{current.get('text')}"

Grading rubric (if any): {rub or "(none)"}

Student answer:
\"\"\"{student_answer}\"\"\"

Return compact JSON ONLY:
{{"ok": true/false, "feedback": "...", "next_hint": "...", "ready_to_advance": true/false}}
"""
        try:
            client = anthropic_client()
            msg = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=400,
                temperature=0.2,
                messages=[{"role": "user", "content": eval_prompt}],
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
                "feedback": parsed.get("feedback", ""),
                "next_step": next_step,
                "message": message,
            })
        except Exception as e:
            print(f"[chat tutor error] {e}")
            # Fallback 200: no rompemos el front
            ok = len(student_answer.strip()) > 5
            return jsonify({
                "ok": ok,
                "feedback": "Recorded. Be explicit about the formula and why it applies.",
                "next_step": str(int(step_id) + 1) if (ok and step_id.isdigit()) else step_id,
                "message": "" if ok else "Try naming the law/formula and include units."
            })

    return jsonify({"error": f"Unsupported mode: {mode}"}), 400

@app.route("/check-step", methods=["OPTIONS", "POST"])
def check_step_alias():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(force=True, silent=True) or {}
    data.setdefault("mode", "tutor")
    # Reuse /chat handler
    with app.test_request_context("/chat", method="POST", json=data, headers=request.headers):
        return chat()

# ---------- Gunicorn entry ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


