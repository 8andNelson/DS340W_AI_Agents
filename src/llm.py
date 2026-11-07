"""
Shared LLM call for every agent, routed to one of two providers by
config.LLM_PROVIDER:

  - "openrouter" (default): OpenRouter's chat-completions API.
  - "anthropic": Claude through the official Anthropic SDK -- implemented
    and unit-tested, not yet run live (no Claude API access yet); select
    it with LLM_PROVIDER=anthropic.

Both paths take the same arguments and return the response text, so no
agent knows which provider served it.

OpenRouter's free model returns HTTP 429 (Too Many Requests) often enough to
halt the pipeline -- the Intake Agent failed both supervised attempts three
runs in a row in logs/master_agent_log.json -- so that path retries 429s with
exponential backoff. The Anthropic SDK retries 429/5xx itself (max_retries).
"""
import json
import re
import time

import requests

from src.config import (
    ANTHROPIC_API_KEY, CLAUDE_EFFORT, CLAUDE_MODEL, DEFAULT_MODEL, LLM_PROVIDER, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

RATE_LIMIT_RETRIES = 5  # 5+10+20+40+80s: the free tier's per-minute window often outlasts 35s
RATE_LIMIT_BACKOFF_SECONDS = 5

# Claude requests always stream: generated scripts can be long, and
# streaming keeps a large max_tokens clear of HTTP timeouts.
CLAUDE_MAX_TOKENS = 64000
CLAUDE_MIN_TIMEOUT_SECONDS = 600
CLAUDE_MAX_RETRIES = 4
# Server-side fallback: if Claude's safety classifiers decline a request
# (e.g. a benign security-sounding prompt), the API re-runs it on the model
# Anthropic recommends for that refusal category instead of failing the
# stage. Only these models accept the "default" form.
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

_claude_client = None


def chat(prompt: str, system: str = "", timeout: int = 60, model: str = None, effort: str = None) -> str:
    """Send one user prompt (plus optional system prompt) and return the
    response text. model overrides the active provider's configured model;
    effort applies to Claude only (low|medium|high|xhigh|max)."""
    if LLM_PROVIDER == "openrouter":
        return _chat_openrouter(prompt, system, timeout, model or DEFAULT_MODEL)
    if LLM_PROVIDER == "anthropic":
        return _chat_claude(prompt, system, timeout, model or CLAUDE_MODEL, effort or CLAUDE_EFFORT)
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' (expected 'anthropic' or 'openrouter').")


def _client() -> "anthropic.Anthropic":
    # Imported here so the default OpenRouter path never depends on the
    # anthropic package being installed.
    import anthropic

    global _claude_client
    if _claude_client is None:
        # Without an explicit key the SDK resolves credentials itself
        # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).
        kwargs = {"api_key": ANTHROPIC_API_KEY} if ANTHROPIC_API_KEY else {}
        _claude_client = anthropic.Anthropic(max_retries=CLAUDE_MAX_RETRIES, **kwargs)
    return _claude_client


def _chat_claude(prompt: str, system: str, timeout: int, model: str, effort: str) -> str:
    kwargs = {
        "model": model,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": effort},
    }
    if system:
        kwargs["system"] = system
    if model in CLAUDE_FALLBACK_MODELS:
        kwargs["betas"] = [CLAUDE_FALLBACK_BETA]
        kwargs["fallbacks"] = "default"

    client = _client().with_options(timeout=float(max(timeout, CLAUDE_MIN_TIMEOUT_SECONDS)))
    with client.beta.messages.stream(**kwargs) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        category = getattr(message.stop_details, "category", None) if message.stop_details else None
        raise RuntimeError(f"Claude declined the request (refusal category: {category}).")
    if message.stop_reason == "max_tokens":
        print(f"  [LLM] Claude response hit max_tokens ({CLAUDE_MAX_TOKENS}); output may be truncated.")
    return "".join(block.text for block in message.content if block.type == "text")


class DailyQuotaExceeded(RuntimeError):
    """OpenRouter's free-tier daily request cap is used up. Backoff cannot
    help (the window is a day, not a minute), so this is raised at once
    instead of waiting through retries."""


def _error_metadata(response) -> dict:
    try:
        return (response.json().get("error") or {}).get("metadata") or {}
    except Exception:
        return {}


def _is_daily_quota(response) -> bool:
    # Live: 429 {"error": {"message": "Rate limit exceeded: free-models-per-day ...",
    #   "metadata": {"limit_source": "openrouter_free_tier_daily", "headers": {"X-RateLimit-Reset": ...}}}}
    return "daily" in str(_error_metadata(response).get("limit_source", ""))


def _daily_quota_message(response) -> str:
    headers = _error_metadata(response).get("headers") or {}
    reset = headers.get("X-RateLimit-Reset")
    when = ""
    try:
        when = " It resets at " + time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(int(reset) / 1000)) + "."
    except (TypeError, ValueError):
        pass
    return (f"OpenRouter free-tier daily limit reached ({headers.get('X-RateLimit-Limit', '?')} requests/day).{when} "
            "Adding credits raises the free-model limit to 1000 requests/day.")


def _chat_openrouter(prompt: str, system: str, timeout: int, model: str) -> str:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in .env")

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})

    for attempt in range(RATE_LIMIT_RETRIES + 1):
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages},
            timeout=timeout,
        )
        if response.status_code == 429 and _is_daily_quota(response):
            raise DailyQuotaExceeded(_daily_quota_message(response))
        if response.status_code == 429 and attempt < RATE_LIMIT_RETRIES:
            wait = RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [LLM] Rate limited (429). Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


_PYTHON_BLOCK = re.compile(r"```(?:python|py)[^\n]*\n([\s\S]*?)```", re.IGNORECASE)
_JSON_BLOCK = re.compile(r"```json[^\n]*\n([\s\S]*?)```", re.IGNORECASE)


def parse_json(content: str, default=None):
    """Parse a JSON value from an LLM response: the whole text, else a
    ```json block, else the outermost {...} or [...] span. Returns default
    (never raises) when nothing parses."""
    candidates = [content or ""]
    block = _JSON_BLOCK.search(content or "")
    if block:
        candidates.append(block.group(1))
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, content or "")
        if match:
            candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return default


def parse_code(content: str) -> str:
    """Return the Python script from an LLM response.

    Generated scripts are requested as a fenced ```python block rather than
    a string inside JSON: a whole script escaped into one JSON string is
    the part free models most often get wrong (unescaped quotes or
    newlines), which made the entire response unparseable. Falls back to a
    JSON {"code": ...} response for callers or models that still use it.
    Returns "" when no script is found."""
    blocks = _PYTHON_BLOCK.findall(content or "")
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    data = parse_json(content, default={})
    code = data.get("code") if isinstance(data, dict) else None
    return code if isinstance(code, str) and code.strip() else ""
