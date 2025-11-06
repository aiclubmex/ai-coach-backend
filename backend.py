# backend.py
# Flask API for Physics AI Coach
# Endpoints:
# - GET  /                 -> health
# - GET  /problems         -> list problems from Notion
# - GET  /problem/<id>     -> full fields for one problem
# - GET  /steps/<id>       -> step-by-step plan (Notion or auto-LLM)
# - POST /chat             -> checks a student's step / next hint
#
# Notas:
# - Incluye autogeneración de pasos con LLM cuando Notion no los tenga.
# - Aumenta timeouts y agrega manejo de errores para respuestas JSON del modelo.
# - /steps ahora devuelve también el "source": "notion" o "llm".
# - CORS soporta múltiples orígenes (env FRONTEND_ORIGINS, separados por coma).
# - call_anthropic_api es tolerante a variaciones de payload/response.

import os, re, json
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# ==============================
# ENV / Config
# ==============================
NOTION_API_KEY        = os.environ.get("NOTION_API_KEY", "")
NOTION_DB_ID          = os.environ.get("NOTION_DATABASE_ID", "")
ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")

# Un solo origen (compat) y/o lista separada por comas
FRONTEND_ORIGIN       = os.environ.get("FRONTEND_ORIGIN", "https://aiclub.com.mx")
FRONTEND_ORIGINS_CSV  = os.environ.get("FRONTEND_ORIGINS", "")  # p.ej. "https://aiclub.com.mx,http://localhost:5173"
EXTRA_ORIGINS         = [o.strip() for o in FRONTEND_ORIGINS_CSV.split(",") if o.strip()]
ALLOWED_ORIGINS       = [FRONTEND_ORIGIN] + EXTRA_ORIGINS

# Timeouts
NOTION_TIMEOUT_SEC    = int(os.environ.get("NOTION_TIMEOUT_SEC", "15"))
ANTHROPIC_TIMEOUT_SEC = int(os.environ.get("ANTHROPIC_TIMEOUT_SEC", "60"))

# ==============================
# Flask
# ==============================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

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
    match = UUID_RE.search(value)
    if not match:
        return ""
    raw = match.group(0).replace("-", "")
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

def fetch_page(page_id: str) -> Dict[str, Any]:
    """Retrieve a single Notion page by ID and flatten main property types."""
    pid = sanitize_uuid(page_id)
    if not pid:
        raise ValueError("Invalid page ID")
    url = f"{NOTION_BASE}/pages/{pid}"
    resp = requests.get(url, headers=NOTION_HEADERS, timeout=NOTION_TIMEOUT_SEC)
    resp.raise_for_status()
    page_data = resp.json()

    props = page_data.get("properties", {})
    out: Dict[str, Any] = {"id": pid}

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

def query_database(database_id: str, filter_obj: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Query a Notion database and return flattened rows."""
    db_id = sanitize_uuid(database_id)
    if not db_id:
        raise ValueError("Invalid database ID")

    url = f"{NOTION_BASE}/databases/{db_id}/query"
    payload: Dict[str, Any] = {}
    if filter_obj:
        payload["filter"] = filter_obj

    resp = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=NOTION_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict[str, Any]] = []
    for page in data.get("results", []):
        pid = page.get("id", "")
        props = page.get("properties", {})
        obj: Dict[str, Any] = {"id": pid}
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
    steps: List[Dict[str, str]] = []

    for i, ln in enumerate(lines, start=1):
        ln = ln.strip()
        if not ln:
            continue
        # Remove numbering like "1)", "1.", "Step 1:", etc.
        clean_ln = re.sub(r'^(\d+[\)\.:]?\s*|Step\s+\d+[\)\.:]?\s*)', '', ln, flags=re.IGNORECASE).strip()
        # Only add if there's actual content (not just a number)
        if clean_ln and len(clean_ln) > 2:
            steps.append({"id": str(i), "description": clean_ln, "rubric": ""})

    # If we got steps but they're all too short (just numbers), return empty
    if steps and all(len(s["description"]) < 5 for s in steps):
        return []
    return steps

# ==============================
# Anthropic helper with improved error handling
# ==============================
def call_anthropic_api(prompt: str, max_tokens: int = 1024) -> str:
    """
    Call Anthropic API directly via HTTP.
    Returns the text response or raises an exception.
    Tolerant a distintos formatos de payload/respuesta.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Contenido como bloques de texto (más robusto para versiones recientes)
    payload = {
        "model": "claude-3-5-sonnet-20241022",  # equivalente reciente; ajusta si tu cuenta usa otro ID
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": "You are a helpful IB Physics tutor. Always respond with valid JSON only when explicitly asked; otherwise provide concise text.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
    }

    try:
        print("[API] Calling Anthropic API…")
        resp = requests.post(url, headers=headers, json=payload, timeout=ANTHROPIC_TIMEOUT_SEC)
        print(f"[API] Status: {resp.status_code}")
        # recorta para logs
        preview = resp.text[:300].replace("\n", " ")
        print(f"[API] Response preview: {preview}")

        if resp.status_code != 200:
            try:
                error_data = resp.json()
            except Exception:
                error_data = {}
            error_type = (error_data.get("error") or {}).get("type", "unknown")
            error_msg = (error_data.get("error") or {}).get("message", resp.text)
            raise Exception(f"Anthropic API error ({error_type}): {error_msg}")

        data = resp.json()

        # Formatos posibles:
        # - {"content":[{"type":"text","text":"..."}], ...}
        # - {"content":[{"text":"..."}], ...}  (menos común)
        # Aseguramos extraer el primer bloque de texto.
        content = data.get("content", [])
        if not content:
            raise Exception("Empty response content from Anthropic API")

        first = content[0]
        if isinstance(first, dict):
            text = first.get("text") or first.get("content") or ""
        else:
            text = str(first)

        if not text:
            raise Exception("No text content in Anthropic API response")
        return text

    except requests.exceptions.Timeout:
        raise Exception("Anthropic API timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Anthropic API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Anthropic API error: {str(e)}")

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate step-by-step solution using LLM if not in Notion."""
    title = problem.get("title", "") or problem.get("Name", "")
    statement = problem.get("statement", "")

    prompt = f"""Given this IB Physics problem, create a step-by-step solution plan.

Problem: {title}
Statement: {statement}

Provide 4-6 steps that a student should follow to solve this problem.
Format each step on a new line.
Be specific about physics concepts and formulas needed.

Example:
1) Identify the relevant physics principles (Newton's laws, conservation of energy, etc.)
2) Draw a diagram and label all forces/variables
3) Write down the relevant equations
4) Substitute known values with units
5) Solve for the unknown
6) Check units and reasonableness

Steps:"""

    try:
        text = call_anthropic_api(prompt, max_tokens=512)
        return parse_steps(text)
    except Exception as e:
        print(f"[ERROR] Failed to generate steps: {e}")
        # Fallback genérico
        return [
            {"id": "1", "description": "Read and understand the problem", "rubric": ""},
            {"id": "2", "description": "Identify relevant physics concepts", "rubric": ""},
            {"id": "3", "description": "Set up equations", "rubric": ""},
            {"id": "4", "description": "Solve", "rubric": ""},
        ]

# ==============================
# Utilidades para parsing seguro de JSON del modelo
# ==============================
def try_extract_json_block(text: str) -> Optional[str]:
    """Intenta extraer un bloque JSON con claves conocidas."""
    if not text:
        return None
    s = text.strip()

    # Remueve fences de markdown si aparecieran
    if "```json" in s:
        try:
            s = s.split("```json", 1)[1].split("```", 1)[0]
        except Exception:
            pass
    elif "```" in s:
        try:
            s = s.split("```", 1)[1].split("```", 1)[0]
        except Exception:
            pass

    s = s.strip()

    # Búsqueda conservadora de objeto con "ok" y "feedback"
    try:
        m = re.search(r'\{[^{}]*"ok"[^{}]*"feedback"[^{}]*\}', s, re.DOTALL)
        if m:
            return m.group(0)
    except Exception:
        pass

    # Si el texto parece ser un JSON completo
    if s.startswith("{") and s.endswith("}"):
        return s

    return None

# ==============================
# Routes
# ==============================
@app.get("/")
def index():
    return jsonify({"status": "ok", "service": "ai-coach-backend", "origins": ALLOWED_ORIGINS})

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
        # Intentar parsear pasos desde Notion
        steps = parse_steps(steps_text) if steps_text else []
        source = "notion" if steps else "llm"
        # Si no hay pasos válidos, generar con LLM
        if not steps:
            print("[STEPS] No valid steps in Notion, generating with LLM…")
            steps = generate_steps_with_llm(page)
        else:
            print(f"[STEPS] Found {len(steps)} steps from Notion")
        return jsonify({"steps": steps, "source": source})
    except Exception as e:
        print(f"[ERROR] Failed to get steps: {e}")
        return jsonify({"error": str(e)}), 500

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
    message_text = (data.get("message") or "").strip()

    print(f"[CHAT] mode={mode}, problem_id={problem_id}, step_id={step_id}")
    if student_answer:
        print(f"[CHAT] student_answer sample: {student_answer[:120]}")
    if message_text:
        print(f"[CHAT] message sample: {message_text[:120]}")

    if not problem_id:
        return jsonify({"error": "missing problem_id"}), 400

    # Fetch problem
    try:
        problem = fetch_page(problem_id)
    except Exception as e:
        return jsonify({"error": f"Cannot read problem: {e}"}), 500

    # Get steps (from Notion or LLM)
    steps_text = problem.get("step_by_step", "")
    steps_list = parse_steps(steps_text) if steps_text else []
    if not steps_list:
        print("[CHAT] No valid steps found, generating with LLM…")
        steps_list = generate_steps_with_llm(problem)

    # Find current step
    current_step = None
    for s in steps_list:
        if s.get("id") == step_id:
            current_step = s
            break
    if not current_step and steps_list:
        current_step = steps_list[0]
        step_id = current_step.get("id", "1")

    # Si no hay clave de Anthropic, responder "sin IA" pero funcional
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "ok": True,
            "feedback": "AI Coach is not configured. Your answer was recorded.",
            "ready_to_advance": True,
            "next_step": str(int(step_id) + 1) if step_id.isdigit() else step_id,
            "message": "Continue to next step.",
            "reply": "AI Coach is not configured."
        })

    # MODE: Free chat (student asking questions)
    if mode == "student" and message_text:
        prompt = f"""You are an IB Physics tutor helping with this problem:

Problem: {problem.get('title', '') or problem.get('Name', '')}
Statement: {problem.get('statement', '')}

Student asks: {message_text}

Provide a brief, helpful response (2-3 sentences). Guide them towards understanding without giving the full answer."""
        try:
            reply = call_anthropic_api(prompt, max_tokens=256)
            return jsonify({"reply": reply.strip()})
        except Exception as e:
            print(f"[ERROR] Free chat failed: {e}")
            return jsonify({"reply": f"Sorry, I encountered an error: {str(e)[:120]}"}), 500

    # MODE: Hint
    if mode == "hint":
        prompt = f"""You are an IB Physics tutor. Provide a helpful hint for this step.

Problem: {problem.get('title', '') or problem.get('Name', '')}
Current Step: {current_step.get('description', '')}

Provide ONE specific hint (1-2 sentences) that guides the student without giving the answer directly."""
        try:
            hint = call_anthropic_api(prompt, max_tokens=256)
            return jsonify({"hint": hint.strip()})
        except Exception as e:
            print(f"[ERROR] Hint failed: {e}")
            # Fallback neutro usando el propio paso
            return jsonify({"hint": f"Think about the physics concepts involved in: {current_step.get('description', '')}"}), 200

    # MODE: Tutor (evaluate answer)
    if mode == "tutor":
        if not student_answer:
            return jsonify({
                "ok": False,
                "feedback": "Please provide an answer to evaluate.",
                "ready_to_advance": False,
                "next_step": step_id,
                "message": "Write your reasoning above."
            })

        eval_prompt = f"""You are an IB Physics tutor evaluating a student's step.

Problem: {problem.get('title', '') or problem.get('Name', '')}
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
            response_text = response_text.strip()

            # Intentar extraer JSON válido
            json_block = try_extract_json_block(response_text)
            if json_block is None:
                # Último intento: si todo el texto ya es JSON, úsalo
                json_block = response_text if response_text.startswith("{") else None

            if json_block:
                try:
                    parsed = json.loads(json_block)
                except json.JSONDecodeError as je:
                    print(f"[ERROR] JSON parse failed: {je}")
                    print(f"[ERROR] Response was: {response_text[:300]}")
                    parsed = None
            else:
                parsed = None

            # Fallback si no parsea
            if not parsed or not isinstance(parsed, dict):
                parsed = {
                    "ok": len(student_answer) > 15,
                    "feedback": "Your answer was recorded. Try to be more specific with physics concepts and units.",
                    "ready_to_advance": len(student_answer) > 15,
                    "next_hint": "Explain which formulas or principles apply."
                }

            can_advance = bool(parsed.get("ready_to_advance") or parsed.get("ok"))
            next_step = str(int(step_id) + 1) if (can_advance and step_id.isdigit()) else step_id
            message = parsed.get("next_hint", "") if not can_advance else "Great! Continue to the next step."

            return jsonify({
                "ok": can_advance,
                "feedback": parsed.get("feedback", ""),
                "ready_to_advance": bool(parsed.get("ready_to_advance", can_advance)),
                "next_step": next_step,
                "message": message
            })

        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return jsonify({
                "ok": False,
                "feedback": f"Error: {str(e)[:200]}",
                "ready_to_advance": False,
                "next_step": step_id,
                "message": "Please try again."
            }), 500

    # Default fallback
    return jsonify({"error": f"Unsupported mode: {mode}"}), 400

# ==============================
# Gunicorn / Dev entry
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"Starting AI Coach backend on 0.0.0.0:{port} (debug={debug})")
    print(f"Allowed CORS origins: {ALLOWED_ORIGINS}")
    app.run(host="0.0.0.0", port=port, debug=debug)

