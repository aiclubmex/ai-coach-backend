# ============================ backend.py (COMPLETO) ============================
# Flask backend para Physics AI Coach: Notion (problemas) + Claude (tutor)
# Endpoints:
#   GET  /health
#   GET  /problems                 -> lista completa (con paginación de Notion)
#   GET  /problems/<page_id>       -> detalle de un problema
#   POST /chat                     -> chat libre con el tutor
#   POST /hint                     -> siguiente pista según solución verificada
#   POST /check                    -> valida respuesta del alumno (num/semántico)
#
# Variables de entorno requeridas en Render:
#   NOTION_API_KEY
#   NOTION_DATABASE_ID
#   ANTHROPIC_API_KEY
# Opcional:
#   ALLOWED_ORIGIN  (p.ej. https://aiclub.com.mx)    # CORS

import os, re
from math import isfinite
from flask import Flask, jsonify, request
from flask_cors import CORS
from notion_client import Client as NotionClient
from anthropic import Anthropic

# ------------------- Config -------------------
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DB_ID = os.getenv("NOTION_DATABASE_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})

# ------------------- Helpers -------------------
def get_notion():
    if not NOTION_API_KEY:
        raise RuntimeError("Missing NOTION_API_KEY")
    return NotionClient(auth=NOTION_API_KEY)

def get_claude():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY")
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def rich_to_text(prop):
    if not prop:
        return ""
    t = prop.get("type")
    if t not in ("rich_text", "title"):
        return ""
    arr = prop.get(t, [])
    return "".join([x.get("plain_text", "") for x in arr])

def get_prop_text(props, keys):
    """Busca la primera propiedad de texto/rich válida por alias."""
    for k in keys:
        p = props.get(k)
        if p and p.get("type") in ("rich_text", "title"):
            txt = rich_to_text(p).strip()
            if txt:
                return txt
    return ""

def get_prop_select(props, keys):
    for k in keys:
        p = props.get(k)
        if p and p.get("type") == "select" and p.get("select"):
            return p["select"]["name"]
    return None

SYSTEM_PROMPT = (
    "You are a patient, step-by-step physics tutor for an IB student. "
    "Explain with clarity, small steps, and check units. Avoid revealing the final answer unless asked. "
    "When providing hints, give only the next minimal hint."
)

# ------------------- Endpoints -------------------
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/problems")
def problems():
    """Trae TODA la base (paginated) para el sidebar."""
    notion = get_notion()
    if not DB_ID:
        raise RuntimeError("Missing NOTION_DATABASE_ID")

    items = []
    cursor = None
    while True:
        resp = notion.databases.query(
            database_id=DB_ID,
            **({"start_cursor": cursor} if cursor else {})
        )
        for r in resp.get("results", []):
            props = r.get("properties", {})
            title = get_prop_text(props, ["Name", "name", "Título", "Titulo"])
            topic = get_prop_select(props, ["Topic", "topic", "Tema"])
            difficulty = get_prop_select(props, ["Difficulty", "difficulty", "Dificultad"])
            items.append({
                "id": r.get("id"),
                "title": title or "Untitled",
                "topic": topic,
                "difficulty": difficulty
            })
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    items.sort(key=lambda x: (x.get("topic") or "", x.get("title") or ""))
    return jsonify(items)

@app.get("/problems/<page_id>")
def problem_detail(page_id):
    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})

    data = {
        "id": page_id,
        "title": get_prop_text(props, ["Name", "name", "Título", "Titulo"]),
        "statement": get_prop_text(props, ["problem_statement", "Statement", "Enunciado"]),
        "given": get_prop_text(props, ["given_values", "Given", "Datos"]),
        "find": get_prop_text(props, ["find", "Find", "Busca"]),
        "final_answer": get_prop_text(props, ["final_answer", "Answer", "Respuesta"]),
        "key_concepts": get_prop_text(props, ["key_concepts", "Concepts", "Conceptos"]),
        "common_mistakes": get_prop_text(props, ["common_mistakes", "Mistakes", "Errores"]),
        "full_solution": get_prop_text(props, ["full_solution", "Solution", "Solución"]),
    }
    return jsonify(data)

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    user_msg = (data.get("message") or "").strip()
    if not page_id or not user_msg:
        return jsonify({"error": "Missing problemId or message"}), 400

    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})
    context = {
        "title": get_prop_text(props, ["Name", "name"]),
        "statement": get_prop_text(props, ["problem_statement"]),
        "given": get_prop_text(props, ["given_values"]),
        "find": get_prop_text(props, ["find"]),
        "final_answer": get_prop_text(props, ["final_answer"]),
        "key_concepts": get_prop_text(props, ["key_concepts"]),
        "common_mistakes": get_prop_text(props, ["common_mistakes"]),
        "full_solution": get_prop_text(props, ["full_solution"]),
    }

    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Problem: {context['title']}\n"
                f"Statement: {context['statement']}\n"
                f"Given: {context['given']}\n"
                f"Find: {context['find']}\n"
                f"Verified final answer: {context['final_answer']}\n"
                f"Key concepts: {context['key_concepts']}\n"
                f"Common mistakes: {context['common_mistakes']}\n\n"
                f"Student question: {user_msg}"
            )
        }]
    )
    text = "".join([p.text for p in msg.content if getattr(p, "type", "") == "text"])
    return jsonify({"answer": text.strip()})

# ---------- Hints y Check ----------
def extract_number_and_units(text):
    if not text:
        return None, None
    m = re.search(r'([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([A-Za-z°/^*\-\s]*?)\s*$', text.strip())
    if not m:
        return None, text.strip()
    try:
        val = float(m.group(1))
    except:
        val = None
    units = (m.group(2) or "").strip() or None
    return val, units

@app.post("/hint")
def hint():
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    student_note = (data.get("note") or "").strip()
    if not page_id:
        return jsonify({"error": "Missing problemId"}), 400

    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})

    title = get_prop_text(props, ["Name", "name"])
    statement = get_prop_text(props, ["problem_statement"])
    full_solution = get_prop_text(props, ["full_solution"])
    key_concepts = get_prop_text(props, ["key_concepts"])

    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role":"user",
            "content":(
                f"Student is solving '{title}'.\n"
                f"Statement: {statement}\n"
                f"Verified solution:\n{full_solution}\n"
                f"Key concepts: {key_concepts}\n\n"
                "Give ONLY the next hint (one short step) without revealing the final answer. "
                "If the student note is provided, tailor the hint.\n"
                f"Student note: {student_note}"
            )
        }]
    )
    text = "".join([p.text for p in msg.content if getattr(p, "type", "") == "text"])
    return jsonify({"hint": text.strip()})

@app.post("/check")
def check():
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    student_answer = (data.get("answer") or "").strip()
    if not page_id or not student_answer:
        return jsonify({"error":"Missing problemId or answer"}), 400

    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})
    final_answer = get_prop_text(props, ["final_answer"])
    sol = get_prop_text(props, ["full_solution"])

    stu_val, stu_units = extract_number_and_units(student_answer)
    fin_val, fin_units = extract_number_and_units(final_answer)

    if stu_val is not None and fin_val is not None and isfinite(stu_val) and isfinite(fin_val):
        tol = max(1e-3*abs(fin_val), 1e-6)  # 0.1% o 1e-6 mínimo
        numeric_ok = abs(stu_val - fin_val) <= tol
        units_ok = (not fin_units) or (stu_units or "").lower().strip() == (fin_units or "").lower().strip()
        result = numeric_ok and units_ok
        feedback = "✅ Correcto" if result else f"❌ Aún no. Esperado≈ {fin_val} {fin_units or ''} (tol {tol:g})."
        return jsonify({"correct": bool(result), "feedback": feedback.strip()})

    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=250,
        system=SYSTEM_PROMPT,
        messages=[{
            "role":"user",
            "content":(
                "Given the verified final answer and solution, judge if student's answer is equivalent.\n"
                f"Verified final answer: {final_answer}\n"
                f"Verified solution:\n{sol}\n"
                f"Student answer: {student_answer}\n"
                "Reply with 'CORRECT' or 'INCORRECT' and one short feedback line."
            )
        }]
    )
    text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"])
    verdict = "CORRECT" in text.upper()
    feedback = ("✅ Correcto" if verdict else "❌ Aún no") + " — " + text.strip()
    return jsonify({"correct": verdict, "feedback": feedback})

# Nota: en Render usa el comando de inicio:  gunicorn backend:app
# ===============================================================================
# ====== STEP-BY-STEP ENGINE (añadir al final de backend.py) ==================
from functools import lru_cache

STEPS_SYSTEM = (
    "You are a physics tutor. Break the verified full solution into small steps "
    "(2-10). Each step should be a single actionable move. If a step expects a "
    "numeric result, include expected_value (float), tolerance_pct (e.g., 1.0 for 1%), "
    "and expected_units if relevant. Keep JSON concise."
)

def _safe_float(x):
    try:
        return float(x)
    except:
        return None

@lru_cache(maxsize=256)
def plan_steps_for(page_id:str):
    """Genera y cachea los pasos para un problema (en memoria)."""
    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})
    title = get_prop_text(props, ["Name","name"])
    statement = get_prop_text(props, ["problem_statement"])
    given = get_prop_text(props, ["given_values"])
    fin = get_prop_text(props, ["final_answer"])
    solution = get_prop_text(props, ["full_solution"])

    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=800,
        system=STEPS_SYSTEM,
        messages=[{
          "role":"user",
          "content":(
            "Return ONLY valid JSON with this schema:\n"
            "{ \"steps\": [\n"
            "  {\"title\": str, \"instruction\": str, "
            "\"expects_numeric\": bool, "
            "\"expected_value\": number|null, \"tolerance_pct\": number|null, "
            "\"expected_units\": str|null }\n"
            "]}\n\n"
            f"Problem: {title}\nStatement: {statement}\nGiven: {given}\n"
            f"Verified final answer: {fin}\nFull solution:\n{solution}\n"
            "Make steps minimal and student-friendly."
          )
        }]
    )
    text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"]).strip()
    import json
    try:
        data = json.loads(text)
        steps = data.get("steps", [])
    except Exception:
        steps = [{"title":"Start","instruction":"State knowns and what is required.","expects_numeric":False,
                  "expected_value":None,"tolerance_pct":None,"expected_units":None}]
    # Sanea tipos
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
    """Devuelve el plan de pasos para renderizar el modo 'Tutor lidera'."""
    steps = plan_steps_for(page_id)
    return jsonify({"steps": steps, "count": len(steps)})

@app.post("/grade_step")
def grade_step():
    """Valida el trabajo del paso actual. Si no es numérico, usa evaluación semántica breve."""
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    idx = int(data.get("stepIndex", 0))
    student = (data.get("work") or "").strip()
    if not page_id:
        return jsonify({"error":"Missing problemId"}), 400

    steps = plan_steps_for(page_id)
    if idx<0 or idx>=len(steps):
        return jsonify({"error":"Invalid step index"}), 400
    step = steps[idx]

    # Numérico con tolerancia
    if step.get("expects_numeric"):
        stu_val, stu_units = extract_number_and_units(student)
        exp_val = step.get("expected_value")
        tol_pct = step.get("tolerance_pct") or 1.0  # % por defecto
        exp_units = (step.get("expected_units") or "").strip().lower() or None
        if stu_val is None or exp_val is None:
            # caemos a evaluación textual
            pass
        else:
            tol = abs(exp_val)*tol_pct/100.0
            numeric_ok = abs(stu_val - exp_val) <= max(tol, 1e-9)
            units_ok = (not exp_units) or ((stu_units or "").strip().lower()==exp_units)
            ok = numeric_ok and units_ok
            fb = "✅ Paso correcto" if ok else f"❌ Revisa: esperado≈ {exp_val:g} {exp_units or ''} (±{tol_pct}%)."
            return jsonify({"correct": ok, "feedback": fb})

    # Semántico/explicativo
    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=220,
        system=SYSTEM_PROMPT,
        messages=[{
          "role":"user",
          "content":(
            "Judge if the student's work fulfills THIS step only. "
            "Reply 'CORRECT' or 'INCORRECT' and one short line of actionable feedback.\n"
            f"Step instruction: {step.get('instruction')}\n"
            f"Student work: {student}"
          )
        }]
    )
    text = "".join([p.text for p in msg.content if getattr(p,"type","")=="text"])
    verdict = "CORRECT" in text.upper()
    fb = ("✅ Paso correcto" if verdict else "❌ Aún no") + " — " + text.strip()
    return jsonify({"correct": verdict, "feedback": fb})
# ============================================================================ 

