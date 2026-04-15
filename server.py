#!/usr/bin/env python3
"""
Smart Ollama API — supercharges local LLMs with web search, URL reading,
calculations, date/time, and chain-of-thought reasoning.

Start:  python3 server.py
API:    POST http://localhost:9000/v1/chat
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.expanduser('~'))
from load_env import load_dev_vars as _lenv; _lenv()
del _sys, _os, _lenv

import datetime
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import traceback
import base64
from functools import wraps

import requests
from flask import Flask, request, jsonify, Response, stream_with_context, g

# Shared personalization + tool registry
try:
    from user_profile import get_profile
    from tool_registry import TOOLS, execute_tool, run_with_tools, build_tool_system_prompt
    _PROFILE_AVAILABLE = True
except ImportError:
    _PROFILE_AVAILABLE = False
    def get_profile(): return None
    TOOLS = []
    def execute_tool(n, a): return ""
    def run_with_tools(msgs, model, **kw): return "", msgs
    def build_tool_system_prompt(s=""): return s

from tools import detect_tools, run_tools, build_prompt, _PROFILE_AVAILABLE as _TOOLS_PROFILE_AVAILABLE
from model_backends import (
    call_ollama, call_groq, call_cerebras, call_vllm_vision,
    _is_cloud_model, _messages_have_image,
    collect_cerebras, collect_groq, collect_ollama,
    DEFAULT_MODEL, FAST_MODEL, SMART_MODEL, VISION_LOCAL_MODEL,
    OLLAMA_URL, VLLM_VISION_URL, VLLM_VISION_MODEL,
    GROQ_API_KEY, GROQ_MODEL, CEREBRAS_API_KEY, CEREBRAS_MODEL_LARGE,
    _ollama_session, _vllm_session,
)

from tcg.cv_grader import grade_card_image
from tcg.card_identifier import identify_card, CardIdentity, search_ebay_listings, get_beckett_prices
from tcg.training import (
    init_training_tables, store_grade, submit_feedback,
    compute_corrections, build_correction_prompt, get_active_corrections,
    get_training_stats, run_training_cycle, start_trainer,
    create_ollama_model, build_modelfile, mark_activity as _mark_trainer_activity,
)

app = Flask(__name__)
from stats_logger import attach_stats_middleware

SERVER_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Topic inference for personalization
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS = {
    "trading cards": r"\b(card|tcg|pokemon|yugioh|mtg|psa|bgs|grade|slab|pack|pull)\b",
    "pricing": r"\b(price|cost|worth|value|sell|buy|market|ebay)\b",
    "sports": r"\b(nba|nfl|mlb|nhl|football|basketball|baseball|soccer|sport)\b",
    "technology": r"\b(code|programming|python|software|api|server|docker|linux)\b",
    "investing": r"\b(invest|portfolio|stock|profit|roi|flip)\b",
    "gaming": r"\b(game|gaming|playstation|xbox|nintendo|steam|rpg)\b",
}

def _infer_topics(text: str) -> list[str]:
    """Infer topic labels from a message for personalization tracking."""
    import re as _re
    text_lower = text.lower()
    return [
        topic for topic, pattern in _TOPIC_PATTERNS.items()
        if _re.search(pattern, text_lower)
    ]
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_CLOUD_URL = os.environ.get("OLLAMA_CLOUD_URL", "https://ollama.com/api")
OLLAMA_API_KEY   = os.environ.get("OLLAMA_API_KEY", "")
CEREBRAS_MODEL_FAST = os.environ.get("CEREBRAS_MODEL_FAST", "llama3.1-8b")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

DB_PATH = os.path.join(os.path.dirname(__file__), "smart_ollama.db")

# Sessions and search tools live in tools.py / model_backends.py


# ---------------------------------------------------------------------------
# Request hooks — timing, CORS, global error handler
# ---------------------------------------------------------------------------

_active_requests = 0
_last_request_at = 0.0
_IDLE_THRESHOLD = 15  # seconds of no requests before training is allowed

def is_server_idle():
    return _active_requests == 0 and (time.time() - _last_request_at) > _IDLE_THRESHOLD

@app.before_request
def _set_request_start():
    global _active_requests, _last_request_at
    g.request_start = time.time()
    _active_requests += 1
    _last_request_at = time.time()
    _mark_trainer_activity()


@app.after_request
def _add_headers(response):
    global _active_requests
    _active_requests = max(0, _active_requests - 1)
    # CORS
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    # Response-time
    elapsed_ms = round((time.time() - g.get("request_start", time.time())) * 1000, 1)
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


@app.errorhandler(Exception)
def _handle_unhandled(e):
    tb = traceback.format_exc()
    print(f"[Unhandled Error] {e}\n{tb}")
    return jsonify({"error": str(e)}), 500


@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def _options_preflight(path=""):
    resp = jsonify({})
    resp.status_code = 204
    return resp

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
            requests_count INTEGER DEFAULT 0,
            tier TEXT NOT NULL DEFAULT 'free',
            daily_requests INTEGER NOT NULL DEFAULT 0,
            daily_limit INTEGER NOT NULL DEFAULT 50,
            last_reset TEXT NOT NULL DEFAULT ''
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
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
    """)
    db.commit()
    # Safe migrations for existing deployments
    for col, sql in [
        ("tier",           "ALTER TABLE api_keys ADD COLUMN tier TEXT NOT NULL DEFAULT 'free'"),
        ("daily_requests", "ALTER TABLE api_keys ADD COLUMN daily_requests INTEGER NOT NULL DEFAULT 0"),
        ("daily_limit",    "ALTER TABLE api_keys ADD COLUMN daily_limit INTEGER NOT NULL DEFAULT 50"),
        ("last_reset",     "ALTER TABLE api_keys ADD COLUMN last_reset TEXT NOT NULL DEFAULT ''"),
    ]:
        try:
            db.execute(sql)
            db.commit()
        except Exception:
            pass
    db.close()


# ---------------------------------------------------------------------------
# Response cache (SHA256 of model+message+tools, 30-min TTL, non-streaming)
# ---------------------------------------------------------------------------

RESPONSE_CACHE_TTL = 1800  # 30 minutes

def _response_cache_key(model: str, message: str, tools_used: list) -> str:
    raw = f"{model}|{message}|{','.join(sorted(tools_used))}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_response_cache(cache_key: str):
    try:
        db = get_db()
        row = db.execute(
            "SELECT response_json, created_at FROM response_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        db.close()
        if row and (time.time() - row["created_at"]) < RESPONSE_CACHE_TTL:
            return json.loads(row["response_json"])
    except Exception:
        pass
    return None


def _set_response_cache(cache_key: str, data: dict):
    try:
        db = get_db()
        db.execute(
            "INSERT OR REPLACE INTO response_cache (cache_key, response_json, created_at) VALUES (?,?,?)",
            (cache_key, json.dumps(data), time.time()),
        )
        # Prune entries older than TTL
        db.execute(
            "DELETE FROM response_cache WHERE created_at < ?",
            (time.time() - RESPONSE_CACHE_TTL,),
        )
        db.commit()
        db.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model whitelist — fetched from Ollama /api/tags, cached 60 seconds
# ---------------------------------------------------------------------------

_model_whitelist: set = set()
_model_whitelist_ts: float = 0.0
_MODEL_CACHE_TTL = 60  # seconds


def _get_available_models() -> set:
    global _model_whitelist, _model_whitelist_ts
    now = time.time()
    if now - _model_whitelist_ts < _MODEL_CACHE_TTL and _model_whitelist:
        return _model_whitelist
    try:
        r = _ollama_session.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            names = {m["name"] for m in r.json().get("models", [])}
            _model_whitelist = names
            _model_whitelist_ts = now
            return names
    except Exception:
        pass
    return _model_whitelist  # return stale if fetch fails


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

        row_dict = dict(row)
        tier = row_dict.get("tier", "free")
        daily_limit = row_dict.get("daily_limit", 50)
        daily_requests = row_dict.get("daily_requests", 0)
        last_reset = row_dict.get("last_reset", "")
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

        # Reset counter at UTC midnight
        if last_reset != today:
            daily_requests = 0
            db.execute(
                "UPDATE api_keys SET daily_requests = 0, last_reset = ? WHERE key = ?",
                (today, key),
            )

        # Enforce free-tier daily cap
        if tier == "free" and daily_limit >= 0 and daily_requests >= daily_limit:
            db.close()
            tomorrow = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            return jsonify({
                "error": f"Daily free-tier limit reached ({daily_limit} requests/day). Resets at {tomorrow}T00:00:00Z.",
                "daily_requests": daily_requests,
                "daily_limit": daily_limit,
                "resets_at": f"{tomorrow}T00:00:00Z",
                "upgrade": "Contact us to remove the daily limit.",
            }), 429

        db.execute(
            "UPDATE api_keys SET requests_count = requests_count + 1, daily_requests = daily_requests + 1 WHERE key=?",
            (key,),
        )
        db.commit()
        g.api_key = key
        g.user_tier = tier
        db.close()
        return f(*args, **kwargs)
    return decorated

# Alias so profile/tools endpoints (which use @require_auth) work identically
require_auth = require_key


# Tools, search, model backends → tools.py and model_backends.py

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/scanner")
def scanner_page():
    """Mobile-responsive card scanner UI."""
    db = get_db()
    key_row = db.execute("SELECT key FROM api_keys LIMIT 1").fetchone()
    db.close()
    api_key = key_row["key"] if key_row else ""
    with open(os.path.join(os.path.dirname(__file__), "templates", "scanner.html")) as f:
        html = f.read().replace("{{API_KEY}}", api_key)
    return html


@app.route("/v1/profile", methods=["GET"])
@require_auth
def get_profile_endpoint():
    """GET /v1/profile — return user profile stats and memories."""
    if not _PROFILE_AVAILABLE:
        return jsonify({"error": "Profile module not available"}), 503
    profile = get_profile()
    return jsonify({
        "stats": profile.get_stats(),
        "memories": profile.all_memories(),
        "top_topics": profile.get_top_topics(10),
        "system_context": profile.system_context("smart-ollama"),
    })


@app.route("/v1/profile/remember", methods=["POST"])
@require_auth
def profile_remember():
    """POST /v1/profile/remember — store a memory. Body: {key, value, category}"""
    if not _PROFILE_AVAILABLE:
        return jsonify({"error": "Profile module not available"}), 503
    data = request.get_json(force=True) or {}
    key = data.get("key", "").strip()
    value = data.get("value", "").strip()
    category = data.get("category", "fact")
    if not key or not value:
        return jsonify({"error": "key and value required"}), 400
    get_profile().set_memory(key, value, category)
    return jsonify({"status": "saved", "key": key, "value": value})


@app.route("/v1/profile/correction", methods=["POST"])
@require_auth
def profile_correction():
    """POST /v1/profile/correction — record a model correction."""
    if not _PROFILE_AVAILABLE:
        return jsonify({"error": "Profile module not available"}), 503
    data = request.get_json(force=True) or {}
    get_profile().record_correction(
        original=data.get("original", ""),
        corrected=data.get("corrected", ""),
        service=data.get("service", "smart-ollama"),
        context=data.get("context", ""),
    )
    return jsonify({"status": "recorded"})


@app.route("/v1/tools", methods=["GET"])
@require_auth
def list_tools():
    """GET /v1/tools — list available MCP-compatible tools."""
    return jsonify({"tools": TOOLS, "count": len(TOOLS)})


@app.route("/v1/tools/call", methods=["POST"])
@require_auth
def call_tool():
    """POST /v1/tools/call — execute a tool directly. Body: {name, args}"""
    data = request.get_json(force=True) or {}
    name = data.get("name", "")
    args = data.get("args", {})
    if not name:
        return jsonify({"error": "name required"}), 400
    result = execute_tool(name, args)
    return jsonify({"tool": name, "result": result})


@app.route("/health")
def health():
    """GET /health — no auth required."""
    uptime = round(time.time() - SERVER_START_TIME, 1)

    # Try a quick ping to Ollama
    ollama_reachable = False
    model_names = []
    try:
        r = _ollama_session.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_reachable = True
            model_names = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    # Check vllm reachability
    vllm_reachable = False
    try:
        rv = _vllm_session.get(f"{VLLM_VISION_URL}/models", timeout=3)
        vllm_reachable = rv.status_code == 200
    except Exception:
        pass

    # DB size
    db_size_bytes = 0
    try:
        db_size_bytes = os.path.getsize(DB_PATH)
    except Exception:
        pass

    return jsonify({
        "status": "ok",
        "uptime_seconds": uptime,
        "ollama_reachable": ollama_reachable,
        "model_names": model_names,
        "vllm_reachable": vllm_reachable,
        "vllm_vision_model": VLLM_VISION_MODEL,
        "db_size_bytes": db_size_bytes,
    })


@app.route("/")
def index():
    return jsonify({
        "name": "Smart Ollama API",
        "version": "1.0",
        "models": {
            "default": DEFAULT_MODEL,
            "fast": FAST_MODEL,
            "smart": SMART_MODEL,
            "reasoning": f"cerebras:{CEREBRAS_MODEL_LARGE}" if CEREBRAS_API_KEY else (f"groq:{GROQ_MODEL}" if GROQ_API_KEY else DEFAULT_MODEL),
        },
        "capabilities": ["web_search", "read_url", "calculate", "datetime", "weather", "tcg_grading", "vision", "voice"],
        "endpoints": {
            "GET /health": "Health check (no auth required)",
            "POST /v1/chat": "Chat with the AI (supports web search, URL reading, math, etc.) — auto-routes fast/reasoning/vision",
            "POST /v1/vision": "Vision chat routed to InternVL2-26B via vllm (OpenAI-compatible multipart)",
            "POST /v1/voice/transcribe": "Speech-to-text (multipart audio field, uses Whisper)",
            "POST /v1/voice/speak": "Text-to-speech — returns audio/mpeg (JSON: {text, lang})",
            "POST /v1/voice/chat": "Full voice round-trip: audio in → AI response → audio out",
            "GET /v1/history": "Get last N chat messages from DB",
            "DELETE /v1/history": "Clear conversation history for your API key",
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

    conversation = data.get("conversation", [])
    stream = data.get("stream", False)
    use_tools = data.get("tools", True)

    t0 = time.time()

    # --- Smart chat routing ---
    # Patterns that suggest complex reasoning / analysis
    _REASONING_KEYWORDS = re.compile(
        r"\b(reason|explain|analyze|analyse|compare|why|how does|step.by.step|think|"
        r"logic|proof|evaluate|debate|argue|pros and cons|summarize|summarise|"
        r"write.*essay|write.*code|debug|implement|algorithm|hypothesis|research)\b",
        re.IGNORECASE,
    )
    _TOOL_KEYWORDS = re.compile(
        r"\b(search|look up|find|google|weather|calculate|compute|what is|who is|"
        r"url|http|https|news|latest|price|today|time|date)\b",
        re.IGNORECASE,
    )
    # Check if any message in the conversation contains an image
    has_image = _messages_have_image(conversation)

    model_choice = data.get("model", "auto")

    is_premium = getattr(g, "user_tier", "free") == "premium"

    if model_choice == "fast":
        model = FAST_MODEL
    elif model_choice == "smart":
        if is_premium and CEREBRAS_API_KEY:
            model = "__cerebras__"
        elif is_premium and GROQ_API_KEY:
            model = "__groq__"
        else:
            model = SMART_MODEL if OLLAMA_API_KEY else DEFAULT_MODEL
    elif model_choice in ("default", "auto"):
        if has_image:
            model = "__vllm__"
        elif _REASONING_KEYWORDS.search(user_message) or len(user_message) > 200:
            if is_premium and CEREBRAS_API_KEY:
                model = "__cerebras__"
            elif is_premium and GROQ_API_KEY:
                model = "__groq__"
            else:
                model = DEFAULT_MODEL  # free tier: local deepseek
        elif len(user_message) < 80 and not _TOOL_KEYWORDS.search(user_message):
            model = FAST_MODEL
        else:
            model = FAST_MODEL
    else:
        # Raw model name requested — block premium models for free tier
        _PREMIUM_MODELS = {"__cerebras__", "__groq__", CEREBRAS_MODEL_LARGE, CEREBRAS_MODEL_FAST, GROQ_MODEL}
        if model_choice in _PREMIUM_MODELS and not is_premium:
            return jsonify({"error": "This model requires a premium subscription."}), 403
        model = model_choice

    # Vision models must never be used for text-only chat
    _VISION_ONLY_MODELS = {"moondream:latest", "moondream", "qwen2.5vl:7b", "qwen2.5vl:3b"}
    if model in _VISION_ONLY_MODELS and not has_image:
        model = FAST_MODEL

    # Model whitelist validation — reject unknown raw model names (skip cloud + vllm pseudo-model)
    if model_choice not in ("fast", "smart", "default", "auto") and model != "__vllm__":
        if not _is_cloud_model(model):
            available = _get_available_models()
            if available and model not in available:
                return jsonify({
                    "error": f"Model '{model}' is not available. Available: {sorted(available)}",
                }), 400

    # Fast route: short messages with no tool keywords bypass tool detection
    fast_route = (
        use_tools
        and len(user_message) < 50
        and not _TOOL_KEYWORDS.search(user_message)
    )

    # Detect and run tools
    tool_results = {}
    tools_used = []
    if use_tools and not fast_route:
        tools_to_run = detect_tools(user_message)
        if tools_to_run:
            tools_used = [t[0] for t in tools_to_run]
            tool_results = run_tools(tools_to_run)

    # Non-streaming: check response cache before calling model
    if not stream:
        cache_key = _response_cache_key(model, user_message, tools_used)
        cached = _get_response_cache(cache_key)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

    # Build prompt with tool context
    messages = build_prompt(user_message, tool_results, conversation)

    # Route to Cerebras (primary) or Groq (fallback) for reasoning
    if model == "__cerebras__":
        return call_cerebras(messages, large=True, stream=stream)
    if model == "__groq__":
        return call_groq(messages, stream=stream)

    # Route to vllm for vision — fall back to qwen2.5vl:7b locally if vllm down
    if model == "__vllm__":
        if stream:
            # Try vllm first; fall back to local qwen2.5vl
            try:
                return call_vllm_vision(messages, stream=True)
            except Exception:
                return call_ollama(messages, VISION_LOCAL_MODEL, stream=True)
        try:
            response_text = call_vllm_vision(messages, stream=False)
            model = VLLM_VISION_MODEL  # label the actual model used in response
        except (TimeoutError, RuntimeError):
            # vllm unavailable — use qwen2.5vl:7b locally
            response_text = call_ollama(messages, VISION_LOCAL_MODEL, stream=False)
            model = VISION_LOCAL_MODEL
        except Exception as e:
            return jsonify({"error": str(e)}), 504
        elapsed = time.time() - t0
    # Otherwise call Ollama (local or cloud)
    elif stream:
        return call_ollama(messages, model, stream=True)
    else:
        try:
            response_text = call_ollama(messages, model, stream=False)
        except TimeoutError as e:
            return jsonify({"error": str(e)}), 504
        except PermissionError as e:
            return jsonify({"error": str(e)}), 403
        elapsed = time.time() - t0

    # Record interaction for personalization
    if _PROFILE_AVAILABLE:
        try:
            profile = get_profile()
            if profile:
                # Infer topics from message
                topics = _infer_topics(user_message)
                profile.record_interaction(
                    user_message,
                    service="smart-ollama",
                    model=model,
                    topics=topics,
                )
        except Exception:
            pass

    # Log
    db = get_db()
    db.execute(
        "INSERT INTO chat_log (api_key, model, user_message, assistant_message, tools_used, elapsed, created_at) VALUES (?,?,?,?,?,?,?)",
        (g.api_key, model, user_message, response_text[:500], json.dumps(tools_used), elapsed, time.time()),
    )
    db.commit()
    db.close()

    result = {
        "response": response_text,
        "model": model,
        "tools_used": tools_used,
        "tool_results": {k: v[:500] for k, v in tool_results.items()} if tool_results else None,
        "elapsed_seconds": round(elapsed, 2),
        "cached": False,
    }

    # Store in response cache (non-streaming only)
    _set_response_cache(cache_key, result)

    return jsonify(result)


@app.route("/v1/models", methods=["GET"])
@require_key
def list_models():
    try:
        r = _ollama_session.get(f"{OLLAMA_URL}/api/tags", timeout=5)
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
            "vision": VLLM_VISION_MODEL,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Vision endpoint — routes to InternVL2-26B via vllm
# ---------------------------------------------------------------------------

@app.route("/v1/vision", methods=["POST"])
@require_key
def vision_chat():
    """
    POST /v1/vision
    OpenAI-compatible multimodal chat routed to InternVL2-26B via vllm.

    Accepts:
      application/json:
        {
          "messages": [{"role": "user", "content": [
              {"type": "text", "text": "Describe this image"},
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
          ]}],
          "stream": false
        }

      OR shorthand:
        {
          "message": "Describe this image",
          "image_base64": "<base64>",
          "stream": false
        }
    """
    data = request.get_json(force=True, silent=True) or {}

    # Support shorthand format
    if "messages" not in data:
        user_text = data.get("message", "").strip()
        b64 = data.get("image_base64", "")
        if not user_text and not b64:
            return jsonify({"error": "Provide 'messages' or 'message'+'image_base64'"}), 400

        content: list = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        if b64:
            # Accept raw base64 or data URL
            if not b64.startswith("data:"):
                b64 = f"data:image/jpeg;base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": b64}})
        messages = [{"role": "user", "content": content}]
    else:
        messages = data["messages"]

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    stream = data.get("stream", False)
    t0 = time.time()

    if stream:
        return call_vllm_vision(messages, stream=True)

    try:
        response_text = call_vllm_vision(messages, stream=False)
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    elapsed = time.time() - t0

    # Log to DB
    user_summary = str(messages[-1].get("content", ""))[:200]
    db = get_db()
    db.execute(
        "INSERT INTO chat_log (api_key, model, user_message, assistant_message, tools_used, elapsed, created_at) VALUES (?,?,?,?,?,?,?)",
        (g.api_key, VLLM_VISION_MODEL, user_summary, response_text[:500], "[]", elapsed, time.time()),
    )
    db.commit()
    db.close()

    return jsonify({
        "response": response_text,
        "model": VLLM_VISION_MODEL,
        "elapsed_seconds": round(elapsed, 2),
    })


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
        "tier": key_row["tier"] if "tier" in key_row.keys() else "free",
        "total_requests": key_row["requests_count"],
        "recent": [dict(r) for r in recent],
    })


@app.route("/v1/history", methods=["GET"])
@require_key
def get_history():
    """GET /v1/history — returns last N chat messages for this API key."""
    limit = min(int(request.args.get("limit", 50)), 500)
    db = get_db()
    rows = db.execute(
        "SELECT id, model, user_message, assistant_message, tools_used, elapsed, created_at "
        "FROM chat_log WHERE api_key=? ORDER BY created_at DESC LIMIT ?",
        (g.api_key, limit),
    ).fetchall()
    db.close()
    messages = []
    for r in rows:
        row = dict(r)
        try:
            row["tools_used"] = json.loads(row["tools_used"] or "[]")
        except Exception:
            row["tools_used"] = []
        messages.append(row)
    return jsonify({"history": messages, "count": len(messages)})


@app.route("/v1/history", methods=["DELETE"])
@require_key
def delete_history():
    """DELETE /v1/history — clear all chat history for the current API key."""
    db = get_db()
    result = db.execute("DELETE FROM chat_log WHERE api_key=?", (g.api_key,))
    db.commit()
    db.close()
    return jsonify({"deleted": result.rowcount, "api_key": g.api_key[:20] + "..."})


@app.route("/v1/keys", methods=["POST"])
def create_key():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "default")
    key = gen_key()
    db = get_db()
    tier = data.get("tier", "free")
    if tier not in ("free", "premium"):
        tier = "free"
    db.execute("INSERT INTO api_keys (key, name, created_at, tier) VALUES (?,?,?,?)", (key, name, time.time(), tier))
    db.commit()
    db.close()
    return jsonify({"api_key": key, "name": name, "tier": tier}), 201


@app.route("/v1/keys/<key>/tier", methods=["PATCH"])
def set_key_tier(key: str):
    """PATCH /v1/keys/<key>/tier — upgrade or downgrade a key's tier.
    Body: {"tier": "premium"} or {"tier": "free"}
    Requires admin secret in Authorization header.
    """
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    auth = request.headers.get("Authorization", "")
    if not admin_secret or auth != f"Bearer {admin_secret}":
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    tier = data.get("tier", "")
    if tier not in ("free", "premium"):
        return jsonify({"error": "tier must be 'free' or 'premium'"}), 400
    db = get_db()
    result = db.execute("UPDATE api_keys SET tier=? WHERE key=?", (tier, key))
    db.commit()
    db.close()
    if result.rowcount == 0:
        return jsonify({"error": "Key not found"}), 404
    return jsonify({"key": key[:20] + "...", "tier": tier})


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

    # Step 1: verify this looks like a trading card before grading
    card_ok, card_msg = _is_card_image(img_data)
    if not card_ok:
        return jsonify({"is_card": False, "message": card_msg}), 422

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
        r = _ollama_session.post(f"{OLLAMA_URL}/api/chat", json={
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


def _is_card_image(img_data):
    """Check if img_data looks like a trading card.

    Returns:
        (True, "card detected") — proceed with grading
        (False, message)        — reject; message is user-facing
    """
    # Fast aspect-ratio heuristic (no extra deps — Pillow already used elsewhere)
    try:
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(img_data))
        w, h = img.size
        if w > 0 and h > 0:
            ratio = min(w, h) / max(w, h)
            # Standard card ratio ≈ 0.714 (2.5″ × 3.5″). Accept 0.58–0.82.
            if 0.58 <= ratio <= 0.82:
                return True, "card detected"
            # Very square or very elongated — clearly not a card
            if ratio > 0.95 or ratio < 0.38:
                return False, "That doesn't look like a trading card. Please submit a clear photo of the front or back of a card."
    except Exception:
        pass

    # Ambiguous or Pillow unavailable — try InternVL2 via vllm first, fall back to local llava
    b64 = base64.b64encode(img_data).decode()
    card_prompt = "Is this image a trading card, sports card, or collectible card? Answer with only YES or NO."

    # Try InternVL2 via vllm
    try:
        vllm_messages = [{"role": "user", "content": [
            {"type": "text", "text": card_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}]
        text = call_vllm_vision(vllm_messages, stream=False)
        if isinstance(text, str) and text.strip().upper().startswith("NO"):
            return False, "This does not appear to be a trading card."
    except Exception:
        # Fall back to local vision model
        try:
            r = _ollama_session.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": VISION_LOCAL_MODEL,
                    "prompt": card_prompt,
                    "images": [b64],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10, "num_ctx": 512},
                },
                timeout=15,
            )
            if r.status_code == 200:
                text = r.json().get("response", "").strip().upper()
                if text.startswith("NO"):
                    return False, "This does not appear to be a trading card."
        except Exception:
            pass

    # Fail open — allow grading if we can't determine
    return True, "card check skipped"


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
    Optional: ?ai=true, ?ebay=true, ?beckett=true
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
    fetch_ebay = request.args.get("ebay", "true").lower() in ("true", "1", "yes")
    fetch_beckett = request.args.get("beckett", "true").lower() in ("true", "1", "yes")

    try:
        result = identify_card(img_data, use_ai=use_ai)
    except Exception as e:
        return jsonify({"error": f"Identification failed: {e}"}), 500

    # No card detected
    if not result.card_detected:
        return jsonify({
            "card_detected": False,
            "message": "No card detected in image",
            "elapsed_seconds": round(time.time() - t0, 4),
        })

    resp = {
        "card_detected": True,
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
        "case": {
            "in_case": result.in_case,
            "case_type": result.case_type,
            "message": f"Card is in a {result.case_type}" if result.in_case else "Out of the case",
        },
        "raw_ocr": result.raw_ocr,
        "ai_enhanced": use_ai,
    }

    # eBay sold listings
    if fetch_ebay and result.name:
        resp["ebay_sold"] = search_ebay_listings(
            result.name, result.year, result.set_name,
            result.card_number, result.game
        )

    # Beckett prices
    if fetch_beckett and result.name:
        resp["beckett"] = get_beckett_prices(
            result.name, result.year, result.set_name, result.game
        )

    resp["elapsed_seconds"] = round(time.time() - t0, 4)
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Voice mode — POST /v1/voice/transcribe  &  POST /v1/voice/speak
# ---------------------------------------------------------------------------
import io as _io
import subprocess as _subp
import tempfile as _tmpf

@app.route("/v1/voice/transcribe", methods=["POST"])
@require_key
def voice_transcribe():
    """Accept audio file (wav/mp3/ogg/webm), return text transcript using Whisper.
    Multipart: field 'audio' with the file.
    Requires whisper: pip3 install openai-whisper
    """
    if "audio" not in request.files:
        return jsonify({"error": "audio file required (multipart field 'audio')"}), 400
    audio_file = request.files["audio"]
    model_size  = request.form.get("model", "base")  # tiny/base/small

    try:
        import whisper as _whisper
    except ImportError:
        return jsonify({"error": "whisper not installed: pip3 install openai-whisper"}), 503

    with _tmpf.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        try:
            model = _whisper.load_model(model_size)
            result = model.transcribe(tmp.name)
            return jsonify({"text": result["text"].strip(), "language": result.get("language", "en")})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            import os as _os; _os.unlink(tmp.name)

@app.route("/v1/voice/speak", methods=["POST"])
@require_key
def voice_speak():
    """Convert text to speech. Returns audio/mpeg binary.
    Body JSON: {"text": "...", "lang": "en"}
    Uses gTTS (Google TTS, requires internet) — fast, natural quality.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = data.get("lang", "en")
    if not text:
        return jsonify({"error": "text required"}), 400
    try:
        from gtts import gTTS as _gTTS
        from flask import Response as _Resp
        buf = _io.BytesIO()
        tts = _gTTS(text=text, lang=lang)
        tts.write_to_fp(buf)
        buf.seek(0)
        return _Resp(buf.read(), mimetype="audio/mpeg",
                     headers={"Content-Disposition": "attachment; filename=speech.mp3"})
    except ImportError:
        return jsonify({"error": "gTTS not installed: pip3 install gtts"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/voice/chat", methods=["POST"])
@require_key
def voice_chat():
    """Full voice round-trip: audio in → text → AI response → audio out.
    Multipart: field 'audio' + optional 'lang' form field.
    Returns JSON with transcript, ai_response, and audio_url for playback.
    """
    if "audio" not in request.files:
        return jsonify({"error": "audio field required"}), 400

    # Step 1: transcribe
    audio_file = request.files["audio"]
    lang        = request.form.get("lang", "en")
    model_size  = request.form.get("whisper_model", "base")

    try:
        import whisper as _whisper
    except ImportError:
        return jsonify({"error": "whisper not installed"}), 503

    with _tmpf.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        try:
            w_model  = _whisper.load_model(model_size)
            w_result = w_model.transcribe(tmp.name)
            transcript = w_result["text"].strip()
        finally:
            import os as _os; _os.unlink(tmp.name)

    if not transcript:
        return jsonify({"error": "Could not transcribe audio"}), 400

    # Step 2: get AI response
    tier  = getattr(g, "user_tier", "free")
    model = DEFAULT_MODEL
    if tier == "premium" and CEREBRAS_API_KEY:
        ai_text = collect_cerebras(transcript)
    elif GROQ_API_KEY:
        ai_text = collect_groq(transcript)
    else:
        ai_text = collect_ollama(transcript, model)

    # Step 3: TTS the response
    try:
        from gtts import gTTS as _gTTS
        import base64 as _b64
        buf = _io.BytesIO()
        _gTTS(text=ai_text, lang=lang).write_to_fp(buf)
        audio_b64 = _b64.b64encode(buf.getvalue()).decode()
    except Exception:
        audio_b64 = None

    return jsonify({
        "transcript": transcript,
        "ai_response": ai_text,
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
    })


# Voice collect helpers → model_backends.collect_cerebras / collect_groq / collect_ollama


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

attach_stats_middleware(app, 'smart-ollama')
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
        r = _ollama_session.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama OK — models: {', '.join(models)}")
    except Exception:
        print(f"  WARNING: Ollama not reachable at {OLLAMA_URL}")

    # Verify vllm
    try:
        rv = _vllm_session.get(f"{VLLM_VISION_URL}/models", timeout=3)
        if rv.status_code == 200:
            vllm_models = [m["id"] for m in rv.json().get("data", [])]
            print(f"  vllm OK — vision model: {', '.join(vllm_models) or VLLM_VISION_MODEL}")
        else:
            print(f"  WARNING: vllm returned HTTP {rv.status_code} at {VLLM_VISION_URL}")
    except Exception:
        print(f"  WARNING: vllm not reachable at {VLLM_VISION_URL} (vision endpoint will be unavailable)")

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
    print(f"  Smart model:   {SMART_MODEL} {'(cloud)' if _is_cloud_model(SMART_MODEL) else '(local)'}")
    print(f"  Fast model:    {FAST_MODEL}")
    print(f"  Vision model:  {VLLM_VISION_MODEL} via {VLLM_VISION_URL}")
    print(f"  TCG model:     tcg-grader")
    port = int(os.environ.get("PORT", 9000))
    print(f"\n  Smart Ollama API running on http://localhost:{port}\n")
    app.run(port=port, debug=False, threaded=True)
