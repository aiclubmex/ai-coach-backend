import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from notion_client import Client as NotionClient
import anthropic

app = Flask(__name__)
# Ajusta origins a tu dominio de WordPress si quieres más seguridad
CORS(app, resources={r"/*": {"origins": "*"}})

def get_notion():
    key = os.environ.get("NOTION_API_KEY")
    if not key:
        raise RuntimeError("Missing NOTION_API_KEY")
    return NotionClient(auth=key)

def get_claude():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key)

DB_ID = os.environ.get("NOTION_DATABASE_ID")

SYSTEM_PROMPT = """You are an IB Physics HL tutor for Pablo.
CRITICAL RULES:
1) You are given PRE-SOLVED, verified problems from Notion.
2) NEVER calculate or solve from scratch.
3) ONLY explain the provided steps and numbers.
4) If the student asks "why", explain the concept.
5) If the student asks "how", show the calculation from the provided steps.
6) Never change numbers or methods. Your role: EXPLAINER, not SOLVER.
"""

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/problems")
def problems():
    """Devuelve lista breve (para el sidebar)."""
    notion = get_notion()
    if not DB_ID:
        raise RuntimeError("Missing NOTION_DATABASE_ID")

    res = notion.databases.query(database_id=DB_ID)
    items = []
    for r in res.get("results", []):
        props = r.get("properties", {})
        name = props.get("name") or props.get("Name")
        topic = props.get("topic") or props.get("column 2")  # según tu doc
        difficulty = props.get("difficulty")

        def get_title(p):
            if not p or p.get("type") != "title":
                return "Untitled"
            rich = p.get("title", [])
            return "".join([x.get("plain_text", "") for x in rich]) or "Untitled"

        def get_select(p):
            if not p or p.get("type") != "select":
                return None
            s = p.get("select")
            return s.get("name") if s else None

        items.append({
            "id": r.get("id"),
            "title": get_title(name),
            "topic": get_select(topic),
            "difficulty": get_select(difficulty),
        })
    return jsonify(items)

@app.get("/problems/<page_id>")
def problem_detail(page_id):
    """Devuelve el contenido listo para render (enunciado, pasos, etc.)."""
    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})

    def get_rich(pkey):
        p = props.get(pkey)
        if not p or p.get("type") not in ("rich_text", "title"):
            return ""
        arr = p.get(p.get("type"), [])
        return "".join([x.get("plain_text", "") for x in arr])

    # Según tus properties del documento
    title = get_rich("name") or get_rich("Name")
    problem_statement = get_rich("problem_statement")
    given_values = get_rich("given_values")
    find_what = get_rich("find")
    key_concepts = get_rich("key_concepts")
    common_mistakes = get_rich("common_mistakes")
    final_answer = get_rich("final_answer")
    full_solution = get_rich("full_solution")

    # (Opcional) Leer bloques si usas toggles por pasos
    # blocks = notion.blocks.children.list(block_id=page_id)

    return jsonify({
        "id": page_id,
        "title": title,
        "statement": problem_statement,
        "given": given_values,
        "find": find_what,
        "key_concepts": key_concepts,
        "common_mistakes": common_mistakes,
        "final_answer": final_answer,
        "steps": full_solution  # si guardas la solución completa aquí
    })

@app.post("/chat")
def chat():
    """Envía pregunta de Pablo + contexto del problema a Claude (modo explicador)."""
    data = request.get_json(silent=True) or {}
    page_id = data.get("problemId")
    user_msg = data.get("message", "").strip()
    if not page_id or not user_msg:
        return jsonify({"error": "Missing problemId or message"}), 400

    notion = get_notion()
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})

    def get_rich(pkey):
        p = props.get(pkey)
        if not p or p.get("type") not in ("rich_text", "title"):
            return ""
        arr = p.get(p.get("type"), [])
        return "".join([x.get("plain_text", "") for x in arr])

    title = get_rich("name") or get_rich("Name")
    statement = get_rich("problem_statement")
    given_values = get_rich("given_values")
    find_what = get_rich("find")
    final_answer = get_rich("final_answer")
    key_concepts = get_rich("key_concepts")
    common_mistakes = get_rich("common_mistakes")
    full_solution = get_rich("full_solution")

    context = f"""
Problem: {title}
Statement: {statement}
Given: {given_values}
Find: {find_what}
Verified final answer: {final_answer}
Key concepts: {key_concepts}
Common mistakes: {common_mistakes}
Verified step-by-step solution:
{full_solution}
"""

    client = get_claude()
    msg = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Student question: {user_msg}\nUse ONLY the verified solution below to explain.\n{context}"}
        ]
    )
    # Extrae texto
    text = ""
    for part in msg.content:
        if part.type == "text":
            text += part.text
    return jsonify({"answer": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

