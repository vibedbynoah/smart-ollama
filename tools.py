"""
Tools module — web search, URL reading, math, datetime, weather.
Extracted from server.py to keep it manageable.
"""
import ast
import datetime
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Profile / tool-registry (optional soft deps)
# ---------------------------------------------------------------------------
try:
    from user_profile import get_profile
    from tool_registry import build_tool_system_prompt
    _PROFILE_AVAILABLE = True
except ImportError:
    _PROFILE_AVAILABLE = False
    def get_profile(): return None
    def build_tool_system_prompt(s=""): return s

# ---------------------------------------------------------------------------
# Session for web requests
# ---------------------------------------------------------------------------

def _make_web_session():
    s = requests.Session()
    retry = Retry(total=1, backoff_factor=0.5, status_forcelist=[502, 503, 504],
                  allowed_methods=["GET", "POST"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

_web_session = _make_web_session()

SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def tool_web_search(query: str, num_results: int = 5) -> str:
    """Search the web via DuckDuckGo with Google fallback."""
    ddg_result = _duckduckgo_search(query, num_results)
    if ddg_result and not ddg_result.startswith("Search"):
        return ddg_result
    google_result = tool_google_search(query, num_results)
    if google_result and not google_result.startswith("Google search"):
        return google_result
    return _duckduckgo_api_search(query, num_results)


def _duckduckgo_search(query: str, num_results: int = 5) -> str:
    try:
        r = _web_session.get(
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
            if title and snippet:
                results.append(f"- {title}\n  {snippet}\n  {url}"[:500])
        return "\n\n".join(results) if results else ""
    except Exception as e:
        return f"Search error: {e}"


def _duckduckgo_api_search(query: str, num_results: int = 5) -> str:
    try:
        r = _web_session.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            headers=SEARCH_HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return "No results found."
        data = r.json()
        results = []
        abstract = data.get("AbstractText", "")
        abstract_url = data.get("AbstractURL", "")
        if abstract:
            results.append(f"- {abstract}\n  {abstract_url}")
        for topic in data.get("RelatedTopics", [])[:num_results]:
            text = topic.get("Text", "")
            url = topic.get("FirstURL", "")
            if text:
                results.append(f"- {text[:400]}\n  {url}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search API error: {e}"


def tool_google_search(query: str, num_results: int = 5) -> str:
    try:
        r = _web_session.get(
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
            if title and snippet:
                results.append(f"- {title}\n  {snippet}\n  {url}"[:500])
        return "\n\n".join(results) if results else ""
    except Exception as e:
        return f"Google search error: {e}"


# ---------------------------------------------------------------------------
# URL reader
# ---------------------------------------------------------------------------

def tool_read_url(url: str) -> str:
    try:
        r = _web_session.get(url, headers=SEARCH_HEADERS, timeout=10)
        if r.status_code != 200:
            return f"Failed to fetch URL (HTTP {r.status_code})"
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup.select("script, style, nav, footer, header, aside, .sidebar, .nav, .menu"):
            tag.decompose()
        main = soup.select_one("article, main, .content, .post, #content")
        text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n".join(lines[:100])
        if len(text) > 4000:
            text = text[:4000] + "\n[...truncated]"
        return text
    except Exception as e:
        return f"URL fetch error: {e}"


# ---------------------------------------------------------------------------
# Math evaluator (safe AST sandbox)
# ---------------------------------------------------------------------------

_SAFE_MATH_NAMES = {"abs", "round", "min", "max", "pow", "sqrt", "log", "log10",
                    "sin", "cos", "tan", "ceil", "floor", "pi", "e"}

_SAFE_MATH_VALS = {
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e,
}


class _SafeMathVisitor(ast.NodeVisitor):
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.FloorDiv, ast.UAdd, ast.USub, ast.Load,
    )

    def visit(self, node):
        if not isinstance(node, self._ALLOWED_NODES):
            if isinstance(node, ast.Name):
                if node.id not in _SAFE_MATH_NAMES:
                    raise ValueError(f"Name '{node.id}' is not allowed")
            else:
                raise ValueError(f"Unsafe node type: {type(node).__name__}")
        return self.generic_visit(node)


def tool_calculate(expression: str) -> str:
    try:
        expression = expression.replace("^", "**").strip()
        tree = ast.parse(expression, mode="eval")
        _SafeMathVisitor().visit(tree)
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_MATH_VALS)
        return str(result)
    except ValueError as e:
        return f"Calculation rejected: {e}"
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Date/time and weather
# ---------------------------------------------------------------------------

def tool_datetime() -> str:
    now = datetime.datetime.now()
    utc = datetime.datetime.utcnow()
    return f"Local: {now.strftime('%Y-%m-%d %H:%M:%S %A')}\nUTC: {utc.strftime('%Y-%m-%d %H:%M:%S')}"


def tool_weather(location: str) -> str:
    try:
        r = _web_session.get(f"https://wttr.in/{quote_plus(location)}?format=j1", timeout=10)
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


# ---------------------------------------------------------------------------
# Tool registry and dispatch
# ---------------------------------------------------------------------------

_LOCAL_TOOLS = {
    "web_search": tool_web_search,
    "read_url": tool_read_url,
    "calculate": tool_calculate,
    "datetime": tool_datetime,
    "weather": tool_weather,
}

_SYNC_TOOLS = {"calculate", "datetime"}
_ASYNC_TOOLS = {"web_search", "read_url", "weather"}


def detect_tools(message: str) -> list[tuple[str, dict]]:
    """Detect which tools to run based on the user's message."""
    msg = message.lower().strip()
    tools_to_run = []

    search_patterns = [
        r"(?:search|look up|find|google|what is|who is|what are|tell me about|latest|news|how to|where is|when did|when was|when is)",
        r"(?:what(?:'s| is) (?:the |a )?(?:latest|current|new|best|price|cost|weather))",
        r"(?:how (?:much|many|long|far|old))",
        r"(?:can you (?:find|search|look))",
    ]
    needs_search = any(re.search(p, msg) for p in search_patterns)

    url_match = re.search(r'(https?://\S+)', message)

    math_patterns = [
        r"(?:calculate|compute|what is \d|how much is \d|\d\s*[\+\-\*\/\^]\s*\d|solve|evaluate)",
        r"(?:square root|sqrt|log|sin|cos|tan|factorial)",
        r"(?:\d+\s*%\s*of\s*\d+)",
    ]
    needs_math = any(re.search(p, msg) for p in math_patterns)

    time_patterns = [r"(?:what time|what day|what date|current date|current time|today|right now)"]
    needs_time = any(re.search(p, msg) for p in time_patterns)

    weather_match = re.search(r"weather (?:in|for|at) (.+?)(?:\?|$|\.)", msg)
    if not weather_match:
        weather_match = re.search(r"(?:temperature|forecast) (?:in|for|at) (.+?)(?:\?|$|\.)", msg)

    if url_match:
        tools_to_run.append(("read_url", {"url": url_match.group(1)}))
    if needs_search:
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
    """Run tools — math/date sync, web/fetch parallel."""
    results = {}

    def run_one(name, kwargs):
        fn = _LOCAL_TOOLS[name]
        try:
            result = fn(**kwargs)
        except Exception as e:
            result = f"Tool error: {e}"
        return name, result

    sync_tools = [(n, kw) for n, kw in tools_to_run if n in _SYNC_TOOLS]
    async_tools = [(n, kw) for n, kw in tools_to_run if n in _ASYNC_TOOLS]

    for name, kwargs in sync_tools:
        name_out, result = run_one(name, kwargs)
        results[name_out] = result

    if async_tools:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_one, name, kwargs) for name, kwargs in async_tools]
            for future in futures:
                name_out, result = future.result()
                results[name_out] = result

    return results


# ---------------------------------------------------------------------------
# System prompt + message builder
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


def build_prompt(user_message: str, tool_results: dict[str, str], conversation: list[dict]) -> list[dict]:
    """Build the full prompt with tool results and personalised user context injected."""
    parts = [SYSTEM_PROMPT]

    if _PROFILE_AVAILABLE:
        profile = get_profile()
        user_ctx = profile.system_context(service="smart-ollama") if profile else ""
        if user_ctx:
            parts.append(f"\n{user_ctx}")
        parts.append(build_tool_system_prompt())

    parts.append(f"\nCurrent date: {datetime.datetime.now().strftime('%Y-%m-%d %A')}")

    if tool_results:
        parts.append("\n--- TOOL RESULTS (use these to answer the user) ---")
        for tool_name, result in tool_results.items():
            parts.append(f"\n[{tool_name}]:\n{result}")
        parts.append("\n--- END TOOL RESULTS ---")

    system = "\n".join(parts)
    messages = [{"role": "system", "content": system}]
    for msg in conversation:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    return messages
