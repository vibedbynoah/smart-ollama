"""
Model backends — Ollama (local + cloud), Groq, Cerebras, vLLM.
Extracted from server.py to keep it manageable.
"""
import json
import os
import time

import requests
from flask import Response, stream_with_context
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Config (read from env — same vars as server.py)
# ---------------------------------------------------------------------------

OLLAMA_URL          = os.environ.get("OLLAMA_URL",          "http://localhost:11434")
OLLAMA_CLOUD_URL    = os.environ.get("OLLAMA_CLOUD_URL",    "https://ollama.com/api")
OLLAMA_API_KEY      = os.environ.get("OLLAMA_API_KEY",      "")
OLLAMA_TIMEOUT      = int(os.environ.get("OLLAMA_TIMEOUT",  "60"))
VLLM_VISION_URL     = os.environ.get("VLLM_VISION_URL",     "http://localhost:8000/v1")
VLLM_VISION_MODEL   = os.environ.get("VLLM_VISION_MODEL",   "OpenGVLab/InternVL2-26B")

_UNIFIED            = os.environ.get("OLLAMA_MODEL",        "qwen2.5vl:7b")
DEFAULT_MODEL       = _UNIFIED
FAST_MODEL          = os.environ.get("OLLAMA_FAST_MODEL",   _UNIFIED)
SMART_MODEL         = os.environ.get("OLLAMA_SMART_MODEL",  "minimax-m2.5:cloud")
VISION_LOCAL_MODEL  = os.environ.get("OLLAMA_VISION_MODEL", _UNIFIED)


def resolve_best_local_model() -> str:
    """Return the best text/vision model currently installed in Ollama.

    Priority order ensures we use a capable model that's actually available
    rather than blindly defaulting to a model that may not be installed.
    """
    _candidates = [
        os.environ.get("OLLAMA_MODEL", ""),
        "qwen2.5vl:7b",
        "qwen2.5vl:7b-q4_K_M",
        "noahmrauch/qwen2.5vl:7b-q4_K_M",
        "qwen2.5:0.5b",
        "moondream:latest",
        "tcg-grader:latest",
    ]
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.ok:
            installed = {m["name"] for m in resp.json().get("models", [])}
            for c in _candidates:
                if c and c in installed:
                    return c
            # Fuzzy: any qwen2.5 variant
            for name in installed:
                if "qwen2.5vl" in name:
                    return name
            if installed:
                return next(iter(installed))
    except Exception:
        pass
    return _UNIFIED

GROQ_API_KEY        = os.environ.get("GROQ_API_KEY",        "")
GROQ_MODEL          = os.environ.get("GROQ_MODEL",          "llama-3.3-70b-versatile")
GROQ_URL            = "https://api.groq.com/openai/v1/chat/completions"

CEREBRAS_API_KEY        = os.environ.get("CEREBRAS_API_KEY",        "")
CEREBRAS_MODEL_LARGE    = os.environ.get("CEREBRAS_MODEL_LARGE",    "qwen-3-235b-a22b-instruct-2507")
CEREBRAS_MODEL_FAST     = os.environ.get("CEREBRAS_MODEL_FAST",     "llama3.1-8b")
CEREBRAS_URL            = "https://api.cerebras.ai/v1/chat/completions"

# ---------------------------------------------------------------------------
# Connection pools — one per backend
# ---------------------------------------------------------------------------

def _make_session(retries: int = 2, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=[502, 503, 504],
                  allowed_methods=["GET", "POST"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

_ollama_session = _make_session()
_cloud_session  = _make_session()
_vllm_session   = _make_session()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cloud_model(model: str) -> bool:
    return model.endswith(":cloud")


def _ollama_headers(cloud: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if cloud and OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers


def _messages_have_image(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def call_ollama(messages: list[dict], model: str, stream: bool = False) -> "str | Response":
    cloud = _is_cloud_model(model)
    base_url = OLLAMA_CLOUD_URL if cloud else OLLAMA_URL
    session = _cloud_session if cloud else _ollama_session
    headers = _ollama_headers(cloud)

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.7, "num_predict": 1024},
    }

    if stream:
        def generate():
            try:
                r = session.post(f"{base_url}/api/chat", json=payload,
                                 headers=headers, stream=True, timeout=OLLAMA_TIMEOUT)
                try:
                    for line in r.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield f"data: {json.dumps({'content': token})}\n\n"
                            if chunk.get("done"):
                                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
                except Exception:
                    yield f"data: {json.dumps({'error': 'Stream interrupted'})}\n\n"
            except requests.exceptions.Timeout:
                yield f"data: {json.dumps({'error': 'Ollama request timed out'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    last_err = None
    for attempt in range(3):
        try:
            r = session.post(f"{base_url}/api/chat", json=payload,
                             headers=headers, timeout=OLLAMA_TIMEOUT)
            if r.status_code == 503 and attempt < 2:
                time.sleep(1)
                last_err = "Ollama returned 503"
                continue
            if r.status_code == 401:
                raise PermissionError(f"Ollama auth failed (cloud={cloud}). Check OLLAMA_API_KEY.")
            return r.json().get("message", {}).get("content", "No response from model")
        except (PermissionError, TimeoutError):
            raise
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out")
        except requests.exceptions.ConnectionError as e:
            last_err = str(e)
            if attempt < 2:
                time.sleep(1)
                continue
        except Exception as e:
            return f"Ollama error: {e}"
    raise TimeoutError(f"Ollama unreachable after 3 attempts: {last_err}")


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def call_groq(messages: list[dict], stream: bool = False) -> "str | Response":
    if not GROQ_API_KEY:
        return call_ollama(messages, DEFAULT_MODEL, stream=stream)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"model": GROQ_MODEL, "messages": messages, "stream": stream,
               "max_tokens": 4096, "temperature": 0.6}

    if stream:
        def generate():
            try:
                r = requests.post(GROQ_URL, json=payload, headers=headers,
                                  stream=True, timeout=30)
                for line in r.iter_lines():
                    if not line:
                        continue
                    raw = line.decode() if isinstance(line, bytes) else line
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            yield json.dumps({"message": {"content": token}}) + "\n"
                    except Exception:
                        continue
            except Exception as e:
                yield json.dumps({"message": {"content": f"[Groq error: {e}]"}}) + "\n"
        return Response(generate(), mimetype="application/x-ndjson")

    try:
        r = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return call_ollama(messages, DEFAULT_MODEL, stream=False)


# ---------------------------------------------------------------------------
# Cerebras
# ---------------------------------------------------------------------------

def call_cerebras(messages: list[dict], large: bool = True, stream: bool = False) -> "str | Response":
    if not CEREBRAS_API_KEY:
        return call_groq(messages, stream=stream)

    model = CEREBRAS_MODEL_LARGE if large else CEREBRAS_MODEL_FAST
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {CEREBRAS_API_KEY}"}
    payload = {"model": model, "messages": messages, "stream": stream,
               "max_completion_tokens": 4096, "temperature": 0.6}

    if stream:
        def generate():
            try:
                r = requests.post(CEREBRAS_URL, json=payload, headers=headers,
                                  stream=True, timeout=30)
                for line in r.iter_lines():
                    if not line:
                        continue
                    raw = line.decode() if isinstance(line, bytes) else line
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            yield json.dumps({"message": {"content": token}}) + "\n"
                    except Exception:
                        continue
            except Exception:
                yield from _groq_stream_fallback(messages)
        return Response(generate(), mimetype="application/x-ndjson")

    try:
        r = requests.post(CEREBRAS_URL, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return call_groq(messages, stream=False)


def _groq_stream_fallback(messages):
    try:
        result = call_groq(messages, stream=False)
        yield json.dumps({"message": {"content": result}}) + "\n"
    except Exception:
        pass


# ---------------------------------------------------------------------------
# vLLM / InternVL2
# ---------------------------------------------------------------------------

def call_vllm_vision(messages: list[dict], stream: bool = False) -> "str | Response":
    payload = {"model": VLLM_VISION_MODEL, "messages": messages, "stream": stream,
               "max_tokens": 1024, "temperature": 0.7}

    if stream:
        def generate():
            try:
                r = _vllm_session.post(f"{VLLM_VISION_URL}/chat/completions",
                                       json=payload, stream=True, timeout=OLLAMA_TIMEOUT)
                for line in r.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8") if isinstance(line, bytes) else line
                    if text.startswith("data: "):
                        text = text[6:]
                    if text.strip() == "[DONE]":
                        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
                        break
                    try:
                        chunk = json.loads(text)
                        token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if token:
                            yield f"data: {json.dumps({'content': token})}\n\n"
                    except Exception:
                        pass
            except requests.exceptions.Timeout:
                yield f"data: {json.dumps({'error': 'vllm request timed out'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    try:
        r = _vllm_session.post(f"{VLLM_VISION_URL}/chat/completions",
                                json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"vllm returned HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]
    except (RuntimeError, KeyError) as e:
        raise RuntimeError(f"vllm error: {e}") from e
    except requests.exceptions.Timeout:
        raise TimeoutError("vllm request timed out")


# ---------------------------------------------------------------------------
# Voice collect helpers (used by voice_chat route in server.py)
# ---------------------------------------------------------------------------

def collect_cerebras(prompt: str) -> str:
    try:
        import urllib.request as _ur
        payload = json.dumps({"model": CEREBRAS_MODEL_LARGE,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 1024, "stream": False}).encode()
        req = _ur.Request(CEREBRAS_URL, data=payload,
                          headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}",
                                   "Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception:
        return collect_ollama(prompt, DEFAULT_MODEL)


def collect_groq(prompt: str) -> str:
    try:
        import urllib.request as _ur
        payload = json.dumps({"model": GROQ_MODEL,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": 1024}).encode()
        req = _ur.Request(GROQ_URL, data=payload,
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                                   "Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception:
        return collect_ollama(prompt, DEFAULT_MODEL)


def collect_ollama(prompt: str, model: str) -> str:
    try:
        import urllib.request as _ur
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = _ur.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "")
    except Exception as e:
        return f"Error: {e}"
