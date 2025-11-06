# backend.py
# Flask API for Physics AI Coach
# - GET  /                 -> health
# - GET  /problems         -> list problems from Notion
# - GET  /problem/<id>     -> full fields for one problem
# - GET  /steps/<id>       -> step-by-step plan (Notion or auto-LLM)
# - POST /chat             -> checks a student's step / hint / student chat

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

# Modelo configurable: usa el que ya tenías habilitado antes si quieres
ANTHROPIC_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

FRONTEND_ORIGIN       = os.environ.get("FRONTEND_ORIGIN", "https://aiclub.com.mx")
FRONTEND_ORIGINS_CSV  = os.environ.get("FRONTEND_ORIGINS", "")
EXTRA_ORIGINS         = [o.strip() for o in FRONTEND_ORIGINS_CSV.split(",") if o.strip()]
ALLOWED_ORIGINS       = [FRONTEND_ORIGIN] + EXTRA_ORIGINS

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
    if not value:
        return ""
    m = UUID_RE.search(value)
    if not m:
        return ""
    raw = m.group(0).replace("-", "")
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"

def fetch_page(page_id: str) -> Dict[str, Any]:
    """Retrieve a single Notion page and flatten common properties."""
    pid = sanitize_uuid(page_id)
    if not pid:
        raise ValueError("Invalid page ID")
    url = f"{NOTION_BASE}/pages/{pid}"
    r = requests.get(url, headers=NOTION_HEADERS, timeout=NOTION_TIMEOUT_SEC)
    r.raise_for_status()
    page_data = r.json()
    props = page_data.get("properties", {})

    out: Dict[str, Any] = {"id": pid}
    for k, v in props.items():
        t = v.get("type")
        if t == "title":
            arr = v.get("title", [])
            out[k] = arr[0].get("plain_text", "") if arr else ""
        elif t == "rich_text":
            arr = v.get("rich_text", [])
            out[k] = arr[0].get("plain_text", "") if arr else ""
        elif t == "number":
            out[k] = v.get("number")
        elif t == "select":
            sel = v.get("select")
            out[k] = sel.get("name", "") if sel else ""
        elif t == "multi_select":
            out[k] = [ms.get("name", "") for ms in v.get("multi_select", [])]
        elif t == "url":
            out[k] = v.get("url", "")
        elif t == "files":
            files = v.get("files", [])
            out[k] = [f.get("file", {}).get("url") or f.get("external", {}).get("url") for f in files]
    return out

def query_database(database_id: str, filter_obj: Optional[dict] = None) -> List[Dict[str, Any]]:
    dbid = sanitize_uuid(database_id)
    if not dbid:
        raise ValueError("Invalid database ID")
    url = f"{NOTION_BASE}/databases/{dbid}/query"
    payload: Dict[str, Any] = {}
    if filter_obj:
        payload["filter"] = filter_obj
    r = requests.post(url, headers=NOTION_HEADERS, json=payload, timeout=NOTION_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    results: List[Dict[str, Any]] = []
    for page in data.get("results", []):
        pid = page.get("id", "")
        props = page.get("properties", {})
        obj: Dict[str, Any] = {"id": pid}
        for k, v in props.items():
            t = v.get("type")
            if t == "title":
                arr = v.get("title", [])
                obj[k] = arr[0].get("plain_text", "") if arr else ""
            elif t == "rich_text":
                arr = v.get("rich_text", [])
                obj[k] = arr[0].get("plain_text", "") if arr else ""
            elif t == "number":
                obj[k] = v.get("number")
            elif t == "select":
                sel = v.get("select")
                obj[k] = sel.get("name", "") if sel else ""
            elif t == "multi_select":
                obj[k] = [ms.get("name", "") for ms in v.get("multi_select", [])]
        results.append(obj)
    return results

# ---------- Extraer pasos desde bloques ----------
def fetch_page_blocks(page_id: str) -> list:
    pid = sanitize_uuid(page_id)
    url = f"{NOTION_BASE}/blocks/{pid}/children?page_size=100"
    r = requests.get(url, headers=NOTION_HEADERS, timeout=NOTION_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])

    # Cargar sub-bloques si existen
    expanded = []
    for b in results:
        expanded.append(b)
        if b.get("has_children"):
            bid = b.get("id")
            try:
                sub = requests.get(f"{NOTION_BASE}/blocks/{bid}/children?page_size=100",
                                   headers=NOTION_HEADERS, timeout=NOTION_TIMEOUT_SEC)
                sub.raise_for_status()
                b["children"] = sub.json().get("results", [])
            except Exception as e:
                print(f"[BLOCKS] child fetch error: {e}")
    return results

def extract_steps_from_blocks(blocks: list) -> List[str]:
    items: List[str] = []
    for b in blocks:
        t = b.get("type")
        if t in ("numbered_list_item", "bulleted_list_item", "to_do", "paragraph"):
            rich = b.get(t, {}).get("rich_text", [])
            txt = "".join([x.get("plain_text", "") for x in rich]).strip()
            if txt:
                # Evita párrafos genéricos muy cortos
                if t == "paragraph" and len(txt) < 4:
                    pass
                else:
                    items.append(txt)
        # Hijos
        if b.get("children"):
            items += extract_steps_from_blocks(b.get("children"))
    return items

def parse_steps(text: str) -> List[Dict[str, str]]:
    """Convierte texto con listas numeradas/viñetas en objetos paso."""
    if not text or not text.strip():
        return []
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    steps: List[Dict[str, str]] = []
    for i, ln in enumerate(lines, start=1):
        clean = re.sub(
            r'^(\d+\s*[\)\.\-:]*|step\s*\d+\s*[\)\.\-:]*|[-•]\s*)',
            '', ln, flags=re.I
        ).strip()
        if len(clean) >= 2:
            steps.append({"id": str(i), "description": clean, "rubric": ""})
    return steps

def get_steps_text_from_notion(page: Dict[str, Any]) -> str:
    """Intenta leer los pasos desde propiedades comunes o desde bloques."""
    CANDIDATES = [
        "step_by_step", "Steps", "steps", "Pasos", "Plan",
        "Solution Steps", "solution_steps", "Step Plan"
    ]
    for key in CANDIDATES:
        val = page.get(key)
        if isinstance(val, str) and val.strip():
            return val

    # Si no hay propiedad, lee listas de los bloques
    try:
        blocks = fetch_page_blocks(page.get("id", ""))
        block_texts = extract_steps_from_blocks(blocks)
        if block_texts:
            return "\n".join(block_texts)
    except Exception as e:
        print(f"[STEPS] blocks fallback error: {e}")
    return ""

# ==============================
# Anthropic helper (robusto, con retry)
# ==============================
def call_anthropic_api(prompt: str, max_tokens: int = 1024) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Intento 1: esquema moderno (content como bloques)
    payload1 = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": "You are a helpful IB Physics tutor. When asked for JSON, return ONLY valid JSON.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    # Intento 2: esquema alternativo (contenido como string plano)
    payload2 = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": "You are a helpful IB Physics tutor. When asked for JSON, return ONLY valid JSON.",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    def _extract_text(resp_json: dict) -> str:
        content = resp_json.get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return first.get("text") or first.get("content") or ""
            return str(first)
        return ""

    try:
        print(f"[API] Anthropic try #1 model={ANTHROPIC_MODEL}")
        r = requests.post(url, headers=headers, json=payload1, timeout=ANTHROPIC_TIMEOUT_SEC)
        if r.status_code == 200:
            txt = _extract_text(r.json()).strip()
            if not txt:
                raise Exception("Empty text in response")
            return txt
        else:
            print(f"[API] Try #1 failed {r.status_code}: {r.text[:200]}")

        print("[API] Anthropic try #2 (compat payload)")
        r2 = requests.post(url, headers=headers, json=payload2, timeout=ANTHROPIC_TIMEOUT_SEC)
        if r2.status_code == 200:
            txt = _extract_text(r2.json()).strip()
            if not txt:
                raise Exception("Empty text in response (retry)")
            return txt
        else:
            try:
                errj = r2.json()
            except Exception:
                errj = {}
            raise Exception(f"Anthropic error: {errj.get('error', {}).get('message', r2.text)}")

    except requests.exceptions.Timeout:
        raise Exception("Anthropic API timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Anthropic API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Anthropic API error: {str(e)}")

def generate_steps_with_llm(problem: Dict[str, Any]) -> List[Dict[str, str]]:
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
        parsed = parse_steps(text)
        if parsed:
            return parsed
    except Exception as e:
        print(f"[ERROR] Failed to generate steps: {e}")
    # Fallback muy simple (como te funcionaba antes)
    return [
        {"id": "1", "description": "Read and understand the problem", "rubric": ""},
        {"id": "2", "description": "Identify relevant physics concepts", "rubric": ""},
        {"id": "3", "description": "Set up equations", "rubric": ""},
        {"id": "4", "description": "Solve", "rubric": ""},
    ]

# ---------- Utilidad: intentar extraer JSON del modelo ----------
def try_extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
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
    try:
        m = re.search(r'\{[^{}]*"ok"[^{}]*"feedback"[^{}]*\}', s, re.DOTALL)
        if m:
            return m.group(0)
    except Exception:
        pass
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
        raw_text = get_steps_text_from_notion(page)
        steps = parse_steps(raw_text)
        source = "notion" if steps else "llm"
        if not steps:
            print("[STEPS] No valid steps in Notion, generating with LLM…")
            steps = generate_steps_with_llm(page)
        else:
            print(f"[STEPS] Found {len(steps)} steps from Notion")
        return jsonify({"steps": steps, "source": source})
    except Exception as e:
        print(f"[ERROR] Failed to get steps: {e}")
        # Aún así respondemos algo (para que el front no muera)
        return jsonify({"steps": [
            {"id": "1", "description": "Read and understand the problem", "rubric": ""},
            {"id": "2", "description": "Identify relevant physics concepts", "rubric": ""},
            {"id": "3", "description": "Set up equations", "rubric": ""},
            {"id": "4", "description": "Solve", "rubric": ""},
        ], "source": "fallback"}), 200

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
        # fallback suave
        problem = {"id": problem_id}
        print(f"[WARN] Cannot read problem: {e}")

    # Steps (Notion primero)
    steps_text = get_steps_text_from_notion(problem)
    steps_list = parse_steps(steps_text) if steps_text else []
    if not steps_list:
        print("[CHAT] No valid steps found, generating with LLM…")
        steps_list = generate_steps_with_llm(problem)

    # Paso actual
    current_step = None
    for s in steps_list:
        if s.get("id") == step_id:
            current_step = s
            break
    if not current_step and steps_list:
        current_step = steps_list[0]
        step_id = current_step.get("id", "1")

    # Si no hay clave de Anthropic, comportamiento anterior "benigno"
    if not ANTHROPIC_API_KEY:
        ok = (mode != "tutor") or (len(student_answer) > 5)
        return jsonify({
            "ok": ok,
            "feedback": "AI Coach is not configured. Your answer was recorded.",
            "ready_to_advance": ok,
            "next_step": str(int(step_id) + 1) if (ok and step_id.isdigit()) else step_id,
            "message": "Continue to next step.",
            "reply": "AI Coach is not configured."
        })

    # ===== MODO STUDENT =====
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
            # NO 500: fallback en 200
            return jsonify({"reply": "I couldn't reach the AI right now, but try focusing on the key formula and units for this step."})

    # ===== MODO HINT =====
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
            return jsonify({"hint": f"Think about the physics concepts involved in: {current_step.get('description', '')}"}), 200

    # ===== MODO TUTOR (evaluación) =====
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
            response_text = call_anthropic_api(eval_prompt, max_tokens=512).strip()
            json_block = try_extract_json_block(response_text) or (response_text if response_text.startswith("{") else None)
            parsed = None
            if json_block:
                try:
                    parsed = json.loads(json_block)
                except json.JSONDecodeError as je:
                    print(f"[ERROR] JSON parse failed: {je}; raw: {response_text[:300]}")
            if not parsed or not isinstance(parsed, dict):
                # Fallback razonable si el modelo no dio JSON bien formado
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
            # NO 500: fallback en 200
            ok = len(student_answer.strip()) > 5
            return jsonify({
                "ok": ok,
                "feedback": "Recorded. Be explicit about the formula and why it applies.",
                "ready_to_advance": ok,
                "next_step": str(int(step_id) + 1) if (ok and step_id.isdigit()) else step_id,
                "message": "" if ok else "Try naming the law/formula and include units."
            })

    return jsonify({"error": f"Unsupported mode: {mode}"}), 400

# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"Starting AI Coach backend on 0.0.0.0:{port} (debug={debug})")
    print(f"Allowed CORS origins: {ALLOWED_ORIGINS}")
    print(f"Anthropic model: {ANTHROPIC_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=debug)

