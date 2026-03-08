#!/usr/bin/env python3
"""
Smart Ollama API — supercharges local LLMs with web search, URL reading,
calculations, date/time, and chain-of-thought reasoning.

Start:  python3 server.py
API:    POST http://localhost:9000/v1/chat
"""

import json
import re
import os
import math
import secrets
import sqlite3
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from urllib.parse import quote_plus

import base64
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response, stream_with_context, g

from tcg.cv_grader import grade_card_image
from tcg.card_identifier import identify_card, CardIdentity
from tcg.training import (
    init_training_tables, store_grade, submit_feedback,
    compute_corrections, build_correction_prompt, get_active_corrections,
    get_training_stats, run_training_cycle, start_trainer,
    create_ollama_model, build_modelfile,
)

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
FAST_MODEL = os.environ.get("OLLAMA_FAST_MODEL", "qwen2.5:0.5b")
SMART_MODEL = os.environ.get("OLLAMA_SMART_MODEL", "llama3.2:1b")
DB_PATH = os.path.join(os.path.dirname(__file__), "smart_ollama.db")

SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at REAL NOT NULL,
            requests_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            model TEXT,
            user_message TEXT,
            assistant_message TEXT,
            tools_used TEXT,
            elapsed REAL,
            created_at REAL NOT NULL
        );
    """)
    db.commit()
    db.close()


def gen_key():
    return f"sk_smart_{secrets.token_hex(24)}"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def require_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing API key. Use: Authorization: Bearer sk_smart_..."}), 401
        key = auth[7:]
        db = get_db()
        row = db.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        if not row:
            db.close()
            return jsonify({"error": "Invalid API key"}), 401
        db.execute("UPDATE api_keys SET requests_count = requests_count + 1 WHERE key=?", (key,))
        db.commit()
        g.api_key = key
        db.close()
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Tools — the superpowers we give the model
# ---------------------------------------------------------------------------

def tool_web_search(query: str, num_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return results."""
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=SEARCH_HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return f"Search failed (HTTP {r.status_code})"

        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for item in soup.select(".result")[:num_results]:
            title_el = item.select_one(".result__title")
            snippet_el = item.select_one(".result__snippet")
            link_el = item.select_one("a.result__a")

            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            url = link_el.get("href", "") if link_el else ""

            if title:
                results.append(f"- {title}\n  {snippet}\n  {url}")

        if not results:
            # Fallback: try Google
            return tool_google_search(query, num_results)

        return "\n\n".join(results)
    except Exception as e:
        return f"Search error: {e}"


def tool_google_search(query: str, num_results: int = 5) -> str:
    """Fallback search via Google."""
    try:
        r = requests.get(
            "https://www.google.com/search",
            params={"q": query, "num": str(num_results)},
            headers=SEARCH_HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return f"Google search failed (HTTP {r.status_code})"

        soup = BeautifulSoup(r.text, "lxml")
        results = []
        for div in soup.select("div.g, div[data-sokoban-container]")[:num_results]:
            title_el = div.select_one("h3")
            snippet_el = div.select_one("div[data-sncf], span.st, div.VwiC3b")
            link_el = div.select_one("a")

            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            url = link_el.get("href", "") if link_el else ""

            if title:
                results.append(f"- {title}\n  {snippet}\n  {url}")

        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Google search error: {e}"


def tool_read_url(url: str) -> str:
    """Fetch a URL and extract the main text content."""
    try:
        r = requests.get(url, headers=SEARCH_HEADERS, timeout=10)
        if r.status_code != 200:
            return f"Failed to fetch URL (HTTP {r.status_code})"

        soup = BeautifulSoup(r.text, "lxml")

        # Remove scripts, styles, nav, footer
        for tag in soup.select("script, style, nav, footer, header, aside, .sidebar, .nav, .menu"):
            tag.decompose()

        # Try article or main content
        main = soup.select_one("article, main, .content, .post, #content")
        if main:
            text = main.get_text("\n", strip=True)
        else:
            text = soup.get_text("\n", strip=True)

        # Trim to reasonable size
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n".join(lines[:100])
        if len(text) > 4000:
            text = text[:4000] + "\n[...truncated]"

        return text
    except Exception as e:
        return f"URL fetch error: {e}"


def tool_calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        # Allow basic math operations and functions
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "int": int, "float": float,
            "pow": pow, "sqrt": math.sqrt, "log": math.log,
            "log10": math.log10, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
        }
        # Sanitize
        clean = re.sub(r'[^0-9+\-*/().,%^ a-zA-Z_]', '', expression)
        clean = clean.replace("^", "**")
        result = eval(clean, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def tool_datetime() -> str:
    """Get current date and time."""
    now = datetime.datetime.now()
    utc = datetime.datetime.utcnow()
    return f"Local: {now.strftime('%Y-%m-%d %H:%M:%S %A')}\nUTC: {utc.strftime('%Y-%m-%d %H:%M:%S')}"


def tool_weather(location: str) -> str:
    """Get current weather for a location."""
    try:
        r = requests.get(f"https://wttr.in/{quote_plus(location)}?format=j1", timeout=10)
        if r.status_code != 200:
            return f"Weather fetch failed (HTTP {r.status_code})"
        data = r.json()
        current = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)
        country = area.get("country", [{}])[0].get("value", "")
        desc = current.get("weatherDesc", [{}])[0].get("value", "")
        temp_c = current.get("temp_C", "?")
        temp_f = current.get("temp_F", "?")
        humidity = current.get("humidity", "?")
        wind = current.get("windspeedMiles", "?")
        return f"{city}, {country}: {desc}, {temp_c}C ({temp_f}F), Humidity: {humidity}%, Wind: {wind} mph"
    except Exception as e:
        return f"Weather error: {e}"


TOOLS = {
    "web_search": tool_web_search,
    "read_url": tool_read_url,
    "calculate": tool_calculate,
    "datetime": tool_datetime,
    "weather": tool_weather,
}

# ---------------------------------------------------------------------------
# Tool detection — figure out what tools to use before calling the LLM
# ---------------------------------------------------------------------------

def detect_tools(message: str) -> list[tuple[str, dict]]:
    """Detect which tools to run based on the user's message."""
    msg = message.lower().strip()
    tools_to_run = []

    # Web search triggers
    search_patterns = [
        r"(?:search|look up|find|google|what is|who is|what are|tell me about|latest|news|how to|where is|when did|when was|when is)",
        r"(?:what(?:'s| is) (?:the |a )?(?:latest|current|new|best|price|cost|weather))",
        r"(?:how (?:much|many|long|far|old))",
        r"(?:can you (?:find|search|look))",
    ]
    needs_search = any(re.search(p, msg) for p in search_patterns)

    # URL in message
    url_match = re.search(r'(https?://\S+)', message)

    # Math triggers
    math_patterns = [
        r"(?:calculate|compute|what is \d|how much is \d|\d\s*[\+\-\*\/\^]\s*\d|solve|evaluate)",
        r"(?:square root|sqrt|log|sin|cos|tan|factorial)",
        r"(?:\d+\s*%\s*of\s*\d+)",
    ]
    needs_math = any(re.search(p, msg) for p in math_patterns)

    # Date/time triggers
    time_patterns = [r"(?:what time|what day|what date|current date|current time|today|right now)"]
    needs_time = any(re.search(p, msg) for p in time_patterns)

    # Weather triggers
    weather_match = re.search(r"weather (?:in|for|at) (.+?)(?:\?|$|\.)", msg)
    if not weather_match:
        weather_match = re.search(r"(?:temperature|forecast) (?:in|for|at) (.+?)(?:\?|$|\.)", msg)

    if url_match:
        tools_to_run.append(("read_url", {"url": url_match.group(1)}))
    if needs_search:
        # Extract the search query - use the whole message cleaned up
        query = re.sub(r"(?:please |can you |could you |search for |look up |find |google |tell me about )", "", msg).strip("?. ")
        tools_to_run.append(("web_search", {"query": query}))
    if needs_math:
        expr_match = re.search(r'[\d][\d\s\+\-\*\/\^\(\)\.%]+[\d\)]', message)
        if expr_match:
            tools_to_run.append(("calculate", {"expression": expr_match.group(0)}))
    if needs_time:
        tools_to_run.append(("datetime", {}))
    if weather_match:
        tools_to_run.append(("weather", {"location": weather_match.group(1).strip()}))

    return tools_to_run


def run_tools(tools_to_run: list[tuple[str, dict]]) -> dict[str, str]:
    """Run tools concurrently and return results."""
    results = {}

    def run_one(name, kwargs):
        fn = TOOLS[name]
        try:
            result = fn(**kwargs)
            return name, result
        except Exception as e:
            return name, f"Error: {e}"

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_one, name, kwargs) for name, kwargs in tools_to_run]
        for future in futures:
            name, result = future.result()
            results[name] = result

    return results


# ---------------------------------------------------------------------------
# System prompt — makes the small model much smarter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful, accurate, and concise AI assistant. You have access to real-time information through tools that have already been run for you.

Rules:
- Answer the user's question directly and concisely
- When tool results are provided, USE them to give an accurate, up-to-date answer
- If search results are provided, synthesize the information into a clear answer — don't just list the results
- If you don't know something and no tool results are available, say so honestly
- Keep responses focused and to the point
- For math questions, use the calculation result provided
- For date/time questions, use the datetime result provided
- When citing information from search results, be specific"""


def build_prompt(user_message: str, tool_results: dict[str, str], conversation: list[dict]) -> str:
    """Build the full prompt with tool results injected as context."""
    parts = [SYSTEM_PROMPT]

    # Add current date for context
    parts.append(f"\nCurrent date: {datetime.datetime.now().strftime('%Y-%m-%d %A')}")

    # Add tool results as context
    if tool_results:
        parts.append("\n--- TOOL RESULTS (use these to answer the user) ---")
        for tool_name, result in tool_results.items():
            parts.append(f"\n[{tool_name}]:\n{result}")
        parts.append("\n--- END TOOL RESULTS ---")

    system = "\n".join(parts)

    # Build messages for Ollama
    messages = [{"role": "system", "content": system}]

    # Add conversation history
    for msg in conversation:
        messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    return messages


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(messages: list[dict], model: str, stream: bool = False) -> str | Response:
    """Call Ollama's chat API."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
        },
    }

    if stream:
        def generate():
            full_response = ""
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=120,
                )
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            yield f"data: {json.dumps({'content': token})}\n\n"
                        if chunk.get("done"):
                            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    else:
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=120,
            )
            data = r.json()
            return data.get("message", {}).get("content", "No response from model")
        except Exception as e:
            return f"Ollama error: {e}"


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return jsonify({
        "name": "Smart Ollama API",
        "version": "1.0",
        "models": {
            "default": DEFAULT_MODEL,
            "fast": FAST_MODEL,
            "smart": SMART_MODEL,
        },
        "capabilities": ["web_search", "read_url", "calculate", "datetime", "weather", "tcg_grading"],
        "endpoints": {
            "POST /v1/chat": "Chat with the AI (supports web search, URL reading, math, etc.)",
            "POST /v1/identify-card": "Identify any card — TCG or sports (fast: ~80ms, ?ai=true for AI)",
            "POST /v1/grade-card": "Grade a TCG card image (multipart or base64)",
            "POST /v1/grade-feedback": "Submit actual grade for training",
            "GET /v1/training/stats": "View training statistics",
            "POST /v1/training/run": "Trigger a manual training cycle",
            "GET /v1/training/corrections": "View active correction rules",
            "GET /v1/models": "List available Ollama models",
            "GET /v1/usage": "View your usage stats",
            "POST /v1/keys": "Generate a new API key",
        },
        "auth": "Authorization: Bearer sk_smart_...",
    })


@app.route("/v1/chat", methods=["POST"])
@require_key
def chat():
    """
    POST /v1/chat
    {
        "message": "What's the weather in Tokyo?",
        "model": "smart",           // "fast", "smart", or model name
        "conversation": [],         // prior messages
        "stream": false,
        "tools": true               // enable/disable auto-tool use
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Missing 'message' field"}), 400

    # Model selection
    model_choice = data.get("model", "default")
    if model_choice == "fast":
        model = FAST_MODEL
    elif model_choice == "smart":
        model = SMART_MODEL
    elif model_choice == "default":
        model = DEFAULT_MODEL
    else:
        model = model_choice  # Allow raw model names

    conversation = data.get("conversation", [])
    stream = data.get("stream", False)
    use_tools = data.get("tools", True)

    t0 = time.time()

    # Detect and run tools
    tool_results = {}
    tools_used = []
    if use_tools:
        tools_to_run = detect_tools(user_message)
        if tools_to_run:
            tools_used = [t[0] for t in tools_to_run]
            tool_results = run_tools(tools_to_run)

    # Build prompt with tool context
    messages = build_prompt(user_message, tool_results, conversation)

    # Call Ollama
    if stream:
        return call_ollama(messages, model, stream=True)

    response_text = call_ollama(messages, model, stream=False)
    elapsed = time.time() - t0

    # Log
    db = get_db()
    db.execute(
        "INSERT INTO chat_log (api_key, model, user_message, assistant_message, tools_used, elapsed, created_at) VALUES (?,?,?,?,?,?,?)",
        (g.api_key, model, user_message, response_text[:500], json.dumps(tools_used), elapsed, time.time()),
    )
    db.commit()
    db.close()

    return jsonify({
        "response": response_text,
        "model": model,
        "tools_used": tools_used,
        "tool_results": {k: v[:500] for k, v in tool_results.items()} if tool_results else None,
        "elapsed_seconds": round(elapsed, 2),
    })


@app.route("/v1/models", methods=["GET"])
@require_key
def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        return jsonify({
            "models": [
                {
                    "name": m["name"],
                    "size_mb": round(m.get("size", 0) / 1024 / 1024),
                    "family": m.get("details", {}).get("family", ""),
                    "parameters": m.get("details", {}).get("parameter_size", ""),
                }
                for m in models
            ],
            "default": DEFAULT_MODEL,
            "fast": FAST_MODEL,
            "smart": SMART_MODEL,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/usage", methods=["GET"])
@require_key
def usage():
    db = get_db()
    key_row = db.execute("SELECT * FROM api_keys WHERE key=?", (g.api_key,)).fetchone()
    recent = db.execute(
        "SELECT model, user_message, tools_used, elapsed, created_at FROM chat_log WHERE api_key=? ORDER BY created_at DESC LIMIT 20",
        (g.api_key,),
    ).fetchall()
    db.close()
    return jsonify({
        "total_requests": key_row["requests_count"],
        "recent": [dict(r) for r in recent],
    })


@app.route("/v1/keys", methods=["POST"])
def create_key():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "default")
    key = gen_key()
    db = get_db()
    db.execute("INSERT INTO api_keys (key, name, created_at) VALUES (?,?,?)", (key, name, time.time()))
    db.commit()
    db.close()
    return jsonify({"api_key": key, "name": name}), 201


# ---------------------------------------------------------------------------
# TCG Grading Routes
# ---------------------------------------------------------------------------

@app.route("/v1/grade-card", methods=["POST"])
@require_key
def grade_card():
    """
    POST /v1/grade-card
    Content-Type: multipart/form-data  (field: image)
    OR
    Content-Type: application/json     (field: image_base64)
    """
    t0 = time.time()

    # Get image data
    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "Missing 'image' file field"}), 400
        img_data = f.read()
    else:
        data = request.get_json(force=True, silent=True) or {}
        b64 = data.get("image_base64", "")
        if not b64:
            return jsonify({"error": "Missing 'image_base64' or upload an image file"}), 400
        try:
            img_data = base64.b64decode(b64)
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

    # Run CV grading
    try:
        cv_result = grade_card_image(img_data)
    except Exception as e:
        return jsonify({"error": f"CV grading failed: {e}"}), 500

    cv_dict = {
        "overall": cv_result.overall,
        "centering": cv_result.centering,
        "corners": cv_result.corners,
        "edges": cv_result.edges,
        "surface": cv_result.surface,
        "confidence": cv_result.confidence,
        "defects": cv_result.defects,
    }

    # Call the TCG-grader Ollama model for AI interpretation
    ai_prompt = f"""Analyze this card grading data from computer vision and provide your expert assessment.

CV Analysis Results:
- Overall: {cv_result.overall}/10
- Centering: {cv_result.centering}/10
- Corners: {cv_result.corners}/10
- Edges: {cv_result.edges}/10
- Surface: {cv_result.surface}/10
- CV Confidence: {cv_result.confidence:.0%}
- Defects found: {', '.join(cv_result.defects) if cv_result.defects else 'None'}

Provide your grading assessment as JSON."""

    ai_result = {}
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": "tcg-grader",
            "messages": [{"role": "user", "content": ai_prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }, timeout=60)
        ai_text = r.json().get("message", {}).get("content", "")
        # Try to parse JSON from response
        json_match = re.search(r'\{[^{}]+\}', ai_text, re.DOTALL)
        if json_match:
            ai_result = json.loads(json_match.group(0))
    except Exception:
        pass  # AI enhancement is optional; CV result stands alone

    # Apply active corrections
    corrections = get_active_corrections()
    adjusted = dict(cv_dict)
    for c in corrections:
        rt = c.get("rule_type", "")
        adj = c.get("adjustment", 0) * c.get("confidence", 0)
        if rt == "ai_overall_bias" and ai_result.get("overall"):
            ai_result["overall"] = round(ai_result["overall"] + adj, 1)
        elif rt == "cv_overall_bias":
            adjusted["overall"] = round(adjusted["overall"] + adj, 1)
        elif rt.startswith("subgrade_bias_"):
            sub = rt.replace("subgrade_bias_", "")
            if sub in adjusted:
                adjusted[sub] = round(adjusted[sub] + adj, 1)

    # Clamp all values to 1-10
    for k in ["overall", "centering", "corners", "edges", "surface"]:
        if k in adjusted:
            adjusted[k] = max(1.0, min(10.0, adjusted[k]))
        if k in ai_result:
            ai_result[k] = max(1.0, min(10.0, ai_result[k]))

    # Store for training
    grade_id = f"grade_{secrets.token_hex(8)}"
    image_hash = secrets.token_hex(8)
    try:
        store_grade(grade_id, image_hash, adjusted, ai_result if ai_result else adjusted)
    except Exception:
        pass

    elapsed = time.time() - t0

    return jsonify({
        "grade_id": grade_id,
        "cv_grade": adjusted,
        "ai_grade": ai_result if ai_result else None,
        "final_grade": {
            "overall": ai_result.get("overall", adjusted["overall"]),
            "centering": ai_result.get("centering", adjusted["centering"]),
            "corners": ai_result.get("corners", adjusted["corners"]),
            "edges": ai_result.get("edges", adjusted["edges"]),
            "surface": ai_result.get("surface", adjusted["surface"]),
            "psa_equivalent": ai_result.get("psa_equivalent", _psa_label(adjusted["overall"])),
            "confidence": max(cv_result.confidence, ai_result.get("confidence", 0)),
            "defects": cv_result.defects + ai_result.get("defects", []),
        },
        "elapsed_seconds": round(elapsed, 2),
    })


def _psa_label(grade: float) -> str:
    """Convert numeric grade to PSA label."""
    labels = {
        10: "GEM MINT 10", 9: "MINT 9", 8: "NM-MT 8", 7: "NEAR MINT 7",
        6: "EX-MT 6", 5: "EXCELLENT 5", 4: "VG-EX 4", 3: "VERY GOOD 3",
        2: "GOOD 2", 1: "POOR 1",
    }
    return labels.get(round(grade), f"~{grade:.1f}")


@app.route("/v1/grade-feedback", methods=["POST"])
@require_key
def grade_feedback():
    """
    POST /v1/grade-feedback
    {"grade_id": "grade_xxx", "actual_grade": 8.5, "service": "PSA", "feedback": "optional note"}
    """
    data = request.get_json(force=True, silent=True) or {}
    grade_id = data.get("grade_id", "")
    actual = data.get("actual_grade")
    service = data.get("service", "PSA")
    feedback = data.get("feedback", "")

    if not grade_id or actual is None:
        return jsonify({"error": "Missing grade_id or actual_grade"}), 400

    try:
        submit_feedback(grade_id, float(actual), service, feedback)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "grade_id": grade_id, "actual_grade": actual, "service": service})


@app.route("/v1/training/stats", methods=["GET"])
@require_key
def training_stats():
    """GET /v1/training/stats — training statistics."""
    return jsonify(get_training_stats())


@app.route("/v1/training/run", methods=["POST"])
@require_key
def trigger_training():
    """POST /v1/training/run — trigger a manual training cycle."""
    try:
        result = run_training_cycle()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/training/corrections", methods=["GET"])
@require_key
def list_corrections():
    """GET /v1/training/corrections — list active correction rules."""
    return jsonify({"corrections": get_active_corrections()})


# ---------------------------------------------------------------------------
# Card Identification Route
# ---------------------------------------------------------------------------

@app.route("/v1/identify-card", methods=["POST"])
@require_key
def identify_card_route():
    """
    POST /v1/identify-card
    Content-Type: multipart/form-data  (field: image)
    OR application/json  (field: image_base64)
    Optional: ?ai=true for AI-enhanced identification (slower)
    """
    t0 = time.time()

    # Get image data
    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "Missing 'image' file field"}), 400
        img_data = f.read()
    else:
        data = request.get_json(force=True, silent=True) or {}
        b64 = data.get("image_base64", "")
        if not b64:
            return jsonify({"error": "Missing 'image_base64' or upload an image file"}), 400
        try:
            img_data = base64.b64decode(b64)
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

    use_ai = request.args.get("ai", "false").lower() in ("true", "1", "yes")

    try:
        result = identify_card(img_data, use_ai=use_ai)
    except Exception as e:
        return jsonify({"error": f"Identification failed: {e}"}), 500

    elapsed = time.time() - t0

    return jsonify({
        "card": {
            "name": result.name,
            "category": result.category,
            "game": result.game,
            "brand": result.brand,
            "set_name": result.set_name,
            "set_code": result.set_code,
            "card_number": result.card_number,
            "year": result.year,
            "rarity": result.rarity,
            "player": result.player,
            "team": result.team,
            "language": result.language,
            "confidence": result.confidence,
            "description": result.ai_description,
        },
        "raw_ocr": result.raw_ocr,
        "ai_enhanced": use_ai,
        "elapsed_seconds": round(elapsed, 4),
    })


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    init_training_tables()

    db = get_db()
    existing = db.execute("SELECT key FROM api_keys LIMIT 1").fetchone()
    if not existing:
        key = gen_key()
        db.execute("INSERT INTO api_keys (key, name, created_at) VALUES (?,?,?)", (key, "default", time.time()))
        db.commit()
        print(f"\n  Your API key: {key}\n")
    else:
        keys = db.execute("SELECT key, name FROM api_keys").fetchall()
        print(f"\n  API keys:")
        for k in keys:
            print(f"    {k['key']}  ({k['name']})")
        print()
    db.close()

    # Verify Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama OK — models: {', '.join(models)}")
    except Exception:
        print(f"  WARNING: Ollama not reachable at {OLLAMA_URL}")

    # Create TCG grader Ollama model
    print("  Creating TCG grader model...")
    active_corrections = get_active_corrections()
    build_modelfile(active_corrections)
    if create_ollama_model(active_corrections):
        print("  TCG grader model created successfully")
    else:
        print("  WARNING: Could not create TCG grader model")

    # Start background training loop
    start_trainer()

    print(f"  Default model: {DEFAULT_MODEL}")
    print(f"  Smart model:   {SMART_MODEL}")
    print(f"  Fast model:    {FAST_MODEL}")
    print(f"  TCG model:     tcg-grader")
    print(f"\n  Smart Ollama API running on http://localhost:9000\n")
    app.run(port=9000, debug=False)
