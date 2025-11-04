"""
AI COACH - PHYSICS BACKEND
Servidor que conecta Notion (problemas) con Claude API (tutorización)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from notion_client import Client
import os
from datetime import datetime

# ============================================
# CONFIGURACIÓN
# ============================================

app = Flask(__name__)
CORS(app)  # Permite requests desde tu landing page

# API Keys (las configurarás como variables de entorno)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
NOTION_API_KEY = os.environ.get('NOTION_API_KEY')
NOTION_DATABASE_ID = os.environ.get('NOTION_DATABASE_ID')  # Tu database de problemas
24
25  # Limpiar variables de proxy que Render inyecta automáticamente
26 
27  os.environ.pop('HTTP_PROXY', None)
28  os.environ.pop('HTTPS_PROXY', None)
29  os.environ.pop('http_proxy', None)
30  os.environ.pop('https_proxy', None)
31
32  # Inicializar clientes
33  claude_client = anthropic.Anthropic(

# Inicializar clientes
claude_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    max_retries=2,
    timeout=60.0
)
notion_client = Client(auth=NOTION_API_KEY)

# ============================================
# SUPER PROMPT DEL AI COACH
# ============================================

SUPER_PROMPT = """You are an expert IB Physics HL tutor using the "Hybrid Gamified Flow" methodology.

Your mission is to help students master physics concepts through active learning, Socratic questioning, and adaptive scaffolding - NOT just by giving answers.

## TEACHING METHODOLOGY: HYBRID GAMIFIED FLOW

Every problem follows this 3-phase structure:

### PHASE 1: QUICK WIN (5-7 minutes) ⚡
Goal: Build confidence with a quick success

1. Mini-Lesson (2 min): Give a brief, visual explanation of the key concept
2. Worked Example (2 min): Show ONE similar problem solved step-by-step
3. Simple Practice (3 min): Give an easy problem to solve
4. Reward: +10 XP for completing Phase 1

Success criteria: Student solves the practice problem correctly
If struggling: Provide ONE hint, then let them try again

### PHASE 2: DEEP DIVE (10-15 minutes) 🧠
Goal: True understanding through guided discovery

Use Socratic Method:
- Ask questions to guide thinking
- Help student discover the solution themselves
- Never give direct answers unless completely stuck

Adaptive Scaffolding (4 levels):
- Level 1 (Minimal): "What principle applies here?"
- Level 2 (Conceptual): "Remember that torque = r × F..."
- Level 3 (Example): Show a similar worked problem
- Level 4 (Direct): Walk through the solution step-by-step

Start at Level 1. Only escalate if stuck after 2 attempts.

Mini-Wins System:
- +5 XP for each correct step
- +50 XP for completing Phase 2
- Celebrate small victories!

### PHASE 3: MASTERY CHECK (3-5 minutes) ✅
Goal: Confirm independent mastery

1. Give a NEW similar problem
2. Student solves it completely independently
3. NO hints allowed in this phase

Scoring:
- ≥90% correct → MASTERY ACHIEVED ✅
  - Award: +100 XP bonus
  - Unlock next topic
- <90% correct → More practice needed ⚠️
  - Error analysis: Show exactly what was misunderstood
  - Give ONE more similar problem
  - No XP penalty

## XP & LEVELING SYSTEM

XP Awards:
- Quick Win completed: +10 XP
- Correct step in Deep Dive: +5 XP
- Deep Dive completed: +50 XP
- Mastery Check passed (≥90%): +100 XP

Levels:
- Level 1: 0-499 XP
- Level 2: 500-999 XP
- Level 3: 1000-1499 XP
- Level 4: 1500-1999 XP
- Level 5: 2000+ XP

Every 500 XP = Level Up! 🎉

## CRITICAL RULES

1. NEVER give direct answers in Phase 2 - always use Socratic questioning first
2. Start with minimal scaffolding (Level 1) - only escalate if needed
3. Celebrate every small win - learning physics is hard!
4. Be encouraging but honest - if wrong, explain why clearly
5. Keep it visual - use diagrams, analogies, real-world examples
6. IB HL standard only - this is advanced physics
7. Track progress - remind student of XP earned

## RESPONSE FORMAT

After each significant milestone, display updated stats:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PROGRESS UPDATE!

XP Gained: +[amount] XP
Phase: [current phase]
Next: [what comes next]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Now, begin the session with the student."""

# ============================================
# ENDPOINT 1: OBTENER LISTA DE PROBLEMAS
# ============================================

@app.route('/api/problems', methods=['GET'])
def get_problems():
    """
    Obtiene todos los problemas de la database de Notion
    """
    try:
        # Query a Notion
        response = notion_client.databases.query(
            database_id=NOTION_DATABASE_ID,
            sorts=[
                {
                    "property": "Name",
                    "direction": "ascending"
                }
            ]
        )
        
        # Formatear problemas para el frontend
        problems = []
        for page in response['results']:
            props = page['properties']
            
            problem = {
                'id': page['id'],
                'name': props.get('name', {}).get('title', [{}])[0].get('text', {}).get('content', 'Sin título'),
                'difficulty': props.get('difficulty', {}).get('select', {}).get('name', 'Medium'),
                'topic': props.get('column 2', {}).get('select', {}).get('name', 'General'),
                'statement': props.get('problem_statement', {}).get('rich_text', [{}])[0].get('text', {}).get('content', ''),
                'status': props.get('Status', {}).get('status', {}).get('name', 'Not Started')
            }
            
            problems.append(problem)
        
        return jsonify({
            'success': True,
            'problems': problems
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ENDPOINT 2: OBTENER UN PROBLEMA ESPECÍFICO
# ============================================

@app.route('/api/problem/<problem_id>', methods=['GET'])
def get_problem(problem_id):
    """
    Obtiene los detalles completos de un problema
    """
    try:
        page = notion_client.pages.retrieve(page_id=problem_id)
        props = page['properties']
        
        problem = {
            'id': page['id'],
            'name': props.get('name', {}).get('title', [{}])[0].get('text', {}).get('content', 'Sin título'),
            'difficulty': props.get('difficulty', {}).get('select', {}).get('name', 'Medium'),
            'topic': props.get('column 2', {}).get('select', {}).get('name', 'General'),
            'statement': props.get('problem_statement', {}).get('rich_text', [{}])[0].get('text', {}).get('content', ''),
            'given_values': props.get('given_values', {}).get('rich_text', [{}])[0].get('text', {}).get('content', ''),
            'find': props.get('find', {}).get('rich_text', [{}])[0].get('text', {}).get('content', ''),
            'xp_value': 160  # Puedes sacarlo de Notion si lo agregas como propiedad
        }
        
        return jsonify({
            'success': True,
            'problem': problem
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ENDPOINT 3: CHAT CON AI COACH
# ============================================

# Almacenamiento temporal de conversaciones (en producción usarías una DB)
conversations = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Maneja la conversación con Claude
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message')
        problem_context = data.get('problem', {})
        
        # Inicializar conversación si no existe
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Construir contexto del problema
        problem_text = f"""
Problem: {problem_context.get('name', 'Unknown')}
Topic: {problem_context.get('topic', 'Unknown')}
Difficulty: {problem_context.get('difficulty', 'Medium')}

Statement: {problem_context.get('statement', '')}
Given: {problem_context.get('given_values', '')}
Find: {problem_context.get('find', '')}
"""
        
        # Agregar mensaje del estudiante
        conversations[session_id].append({
            "role": "user",
            "content": message
        })
        
        # Llamar a Claude
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SUPER_PROMPT + "\n\n" + problem_text,
            messages=conversations[session_id]
        )
        
        # Agregar respuesta de Claude a la conversación
        assistant_message = response.content[0].text
        conversations[session_id].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Calcular XP ganado (analizar la respuesta de Claude)
        xp_awarded = calculate_xp(assistant_message)
        
        return jsonify({
            'success': True,
            'message': assistant_message,
            'xp_awarded': xp_awarded,
            'conversation_length': len(conversations[session_id])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ENDPOINT 4: GUARDAR PROGRESO
# ============================================

@app.route('/api/save-progress', methods=['POST'])
def save_progress():
    """
    Guarda el progreso del estudiante en Notion
    """
    try:
        data = request.json
        problem_id = data.get('problem_id')
        xp_earned = data.get('xp_earned', 0)
        completed = data.get('completed', False)
        
        # Actualizar página en Notion
        updates = {}
        
        if completed:
            updates['Status'] = {'status': {'name': 'Completed'}}
        
        notion_client.pages.update(
            page_id=problem_id,
            properties=updates
        )
        
        return jsonify({
            'success': True,
            'message': 'Progress saved'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def calculate_xp(message):
    """
    Analiza la respuesta de Claude para determinar XP ganado
    """
    xp = 0
    
    # Detectar menciones de XP en el mensaje
    if '+10 XP' in message or 'Quick Win' in message:
        xp += 10
    if '+5 XP' in message or 'correct step' in message.lower():
        xp += 5
    if '+50 XP' in message or 'Deep Dive completed' in message:
        xp += 50
    if '+100 XP' in message or 'MASTERY ACHIEVED' in message:
        xp += 100
    
    return xp

# ============================================
# HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """
    Verifica que el servidor esté funcionando
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# INICIAR SERVIDOR
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
