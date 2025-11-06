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
CORS(app, resources={r"/*": {"origins": [FRONTEND_ORIGIN]}})

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
    if not value:
        return ""
    m = UUID_RE.search(value)
    if not m:
        return ""
    uid = m.group(0).lower()
    if len(uid) == 32:
        # insert hyphens 8-4-4-4-12
        uid = f"{uid[0:8]}-{uid[8:12]}-{uid[12:16]}-{uid[16:20]}-{uid[20:32]}"
    return uid

def _rt_to_plain(rt: Any) -> str:
    if not rt: return ""
    parts = []
    for x in rt:
        if isinstance(x, dict):
            parts.append(x.get("plain_text") or (x.get("text") or {}).get("content", ""))
    return "".join(parts).strip()

def _get_prop(props: Dict[str, Any], key: str) -> str:
    p = props.get(key)
    if not p: return ""
    t = p.get("type")
    if t == "title":       return _rt_to_plain(p.get("title"))
    if t == "rich_text":   return _rt_to_plain(p.get("rich_text"))
    if t == "select":
        sl = p.get("select")
        return (sl or {}).get("name", "") if sl else ""
    if t == "multi_select":
        return ", ".join([x.get("name","") for x in p.get("multi_select", [])])
    return ""

def fetch_problems() -> List[Dict[str, Any]]:
    url = f"{NOTION_BASE}/databases/{NOTION_DB_ID}/query"
    payload = {"page_size": 100}
    items: List[Dict[str, Any]] = []
    while True:
        r = requests.post(url, headers=NOTION_HEADERS, data=json.dumps(payload))
        r.raise_for_status()
        data = r.json()
        for page in data.get("results", []):
            props = page.get("properties", {})
            items.append({
                "id": page.get("id"),
                "name": _get_prop(props, "Name") or _get_prop(props, "name") or "Untitled",
                "difficulty": _get_prop(props, "difficulty") or _get_prop(props, "Difficulty"),
            })
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data.get("next_cursor")
    return items

def fetch_page(page_id: str) -> Dict[str, Any]:
    pid = sanitize_uuid(page_id)
    if not pid:
        raise ValueError(f"Invalid Notion page_id: {page_id!r}")
    url = f"{NOTION_BASE}/pages/{pid}"
    r = requests.get(url, headers=NOTION_HEADERS)
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

# ==============================
# Steps parsing / generation
# ==============================
STEP_RE = re.compile(r"^\s*(\d+)[\).:-]\s*(.+)$")

def parse_steps(raw: str) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    if not raw: return steps
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
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
    if not steps and raw.strip():
        steps.append({"id": "1", "text": raw.strip(), "rubric": ""})
    return steps

def anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    if not ANTHROPIC_API_KEY:
        return []
    client = anthropic_client()
    prompt = f"""
You are a patient IB Physics tutor. Break this problem into 4–6 small steps.
Each step: concise action the student should do next.
Append a short grading rubric after a pipe like this: " | rubric: ...".
Return plain text lines:

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
        model="claude-3-5-sonnet-latest",
        max_tokens=500,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for b in msg.content:
        if getattr(b, "type", "") == "text":
            text += b.text
    return parse_steps(text)

# ==============================
# Routes
# ==============================
@app.get("/")
def health():
    return jsonify({"ok": True})

@app.get("/problems")
def problems():
    try:
        simplified = fetch_problems()
        return jsonify({"problems": simplified})
    except Exception as e:
        return jsonify({"error": f"Error listing problems: {e}"}), 500

@app.get("/problem/<page_id>")
def problem_detail(page_id):
    try:
        p = fetch_page(page_id)
        return jsonify({"problem": p})
    except Exception as e:
        return jsonify({"error": f"Error reading problem: {e}"}), 500

@app.get("/steps/<page_id>")
def steps(page_id):
    try:
        p = fetch_page(page_id)
        raw = p.get("step_by_step", "")
        steps_list = parse_steps(raw)
        note = None
        if not steps_list:
            # auto-generate with LLM (tutor leads)
            try:
                steps_list = generate_steps_with_llm(p)
                if not steps_list:
                    note = "No steps in Notion. LLM not available or returned nothing."
            except Exception as llm_err:
                note = f"Could not generate steps: {llm_err}"
        return jsonify({"mode": "tutor", "note": note, "steps": steps_list})
    except requests.HTTPError as he:
        return jsonify({"mode": "tutor", "note": f"Error reading steps: {he}", "steps": []}), 500
    except Exception as e:
        return jsonify({"mode": "tutor", "note": f"Unexpected: {e}", "steps": []}), 500

# ==============================
# 🔥 FUNCIÓN CHAT MEJORADA
# ==============================
@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    problem_id      = data.get("problem_id")
    step_id         = str(data.get("step_id") or "")
    student_answer  = (data.get("student_answer") or "").strip()
    mode            = data.get("mode", "tutor")

    if not problem_id:
        return jsonify({"error": "missing problem_id"}), 400

    try:
        p = fetch_page(problem_id)
    except Exception as e:
        return jsonify({"error": f"Cannot read problem: {e}"}), 500

    steps_list = parse_steps(p.get("step_by_step", ""))
    if not steps_list:
        steps_list = generate_steps_with_llm(p)

    current = None
    for s in steps_list:
        if s["id"] == step_id:
            current = s
            break
    if not current and steps_list:
        current = steps_list[0]
        step_id = current["id"]

    # No LLM? Return basic echo
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "ok": True,
            "feedback": "(No LLM) Answer received.",
            "next_step": str(int(step_id) + 1) if step_id.isdigit() else "",
            "message": "Move to the next step."
        })

    rub = current.get("rubric", "")
    
    # 🔥 PROMPT MEJORADO - MÁS ESTRICTO
    eval_prompt = f"""You are an IB Physics tutor. Evaluate this student's step.

Problem:
{p.get('problem_statement', '')[:300]}

Given: {p.get('given_values', '')}
Find: {p.get('find', '')}

Current step {step_id}: {current.get('text')}
Rubric: {rub or 'Check if reasoning is sound and physics concepts are correct'}

Student wrote: "{student_answer}"

CRITICAL: Respond with ONLY valid JSON. No markdown, no extra text.

Format:
{{
  "ok": true,
  "feedback": "Brief supportive message (1-2 sentences)",
  "next_hint": "One concrete hint if wrong, empty string if correct",
  "ready_to_advance": true
}}

JSON response:"""

    try:
        client = anthropic_client()
        msg = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=400,
            temperature=0.2,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        text = ""
        for b in msg.content:
            if getattr(b, "type", "") == "text":
                text += b.text
        
        # 🔥 LIMPIEZA ROBUSTA DE MARKDOWN
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # 🔥 PARSING CON FALLBACK
        try:
            parsed = json.loads(text)
        except:
            # Fallback: buscar JSON con regex
            json_match = re.search(r'\{[^{}]*"ok"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except:
                    parsed = {
                        "ok": len(student_answer) > 5,
                        "feedback": "Your answer was recorded. The AI response couldn't be parsed properly.",
                        "ready_to_advance": len(student_answer) > 5,
                        "next_hint": "Try being more specific with formulas and reasoning."
                    }
            else:
                # Última opción: usar la respuesta como feedback
                parsed = {
                    "ok": len(student_answer) > 5,
                    "feedback": text[:150] if len(text) < 150 else text[:150] + "...",
                    "ready_to_advance": len(student_answer) > 5,
                    "next_hint": ""
                }

        advance = parsed.get("ready_to_advance") or parsed.get("ok")
        next_step = str(int(step_id) + 1) if (advance and step_id.isdigit()) else step_id
        message = parsed.get("next_hint", "") if not advance else "Great! Let's continue."

        return jsonify({
            "ok": bool(parsed.get("ok") or parsed.get("ready_to_advance")),
            "feedback": parsed.get("feedback", ""),
            "next_step": next_step,
            "message": message
        })
        
    except Exception as e:
        return jsonify({
            "ok": False, 
            "feedback": f"Error evaluating: {str(e)[:100]}", 
            "next_step": step_id, 
            "message": "Try rephrasing your answer."
        }), 500

# ==============================
# Gunicorn entry
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
