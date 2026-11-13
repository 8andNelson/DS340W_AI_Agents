"""
Shared LLM call for every agent, routed to one provider by
config.LLM_PROVIDER:

  - "openrouter" (default): OpenRouter's chat-completions API.
  - "freellmapi": a local FreeLLMAPI router (OpenAI-compatible) that pools
    many providers' free tiers and fails over between them.
  - "anthropic": Claude through the official Anthropic SDK -- implemented
    and unit-tested, not yet run live (no Claude API access yet).

Every path takes the same arguments and returns the response text, so no
agent knows which provider served it. Every call -- whichever provider --
is appended to logs/llm_calls.jsonl with the model that actually answered:
a router like FreeLLMAPI can serve each call from a different model, and a
replication must be able to say which model generated its code.

OpenRouter's free model returns HTTP 429 (Too Many Requests) often enough to
halt the pipeline -- the Intake Agent failed both supervised attempts three
runs in a row in logs/master_agent_log.json -- so that path retries 429s with
exponential backoff. The Anthropic SDK retries 429/5xx itself (max_retries).
"""
import json
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import (
    ANTHROPIC_API_KEY, CLAUDE_EFFORT, CLAUDE_MODEL, DEFAULT_MODEL, FREELLMAPI_AGENT_MODELS, FREELLMAPI_API_KEY,
    FREELLMAPI_BASE_URL, FREELLMAPI_MODEL, FREELLMAPI_SMART_MODEL, LLM_PROVIDER, OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from src.orchestration import run_context

CALL_LOG = Path(__file__).parent.parent / "logs" / "llm_calls.jsonl"
_log_lock = threading.Lock()
_last_call = threading.local()  # per thread: research/validation call concurrently

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
_supervisor_notes = ContextVar("supervisor_notes", default=())


HIGH_EFFORT = {"high", "xhigh", "max"}


def chat(prompt: str, system: str = "", timeout: int = 60, model: str = None, effort: str = None,
         agent: str = None) -> str:
    """Send one user prompt (plus optional system prompt) and return the
    response text. model overrides the active provider's configured model.
    effort (low|medium|high|xhigh|max) sets Claude's effort; on FreeLLMAPI,
    high effort and above requests FREELLMAPI_SMART_MODEL. agent (e.g.
    "research_agent") selects that agent's primary/backup models from
    FREELLMAPI_AGENT_MODELS on FreeLLMAPI; other providers ignore it.
    While Master Agent notes are active (supervisor_notes), they are added
    to the system prompt."""
    system = _with_supervisor_notes(system)
    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY is not set in .env")
        return _chat_openai_compatible("openrouter", OPENROUTER_BASE_URL, OPENROUTER_API_KEY,
                                       model or DEFAULT_MODEL, prompt, system, timeout)
    if LLM_PROVIDER == "freellmapi":
        if not FREELLMAPI_API_KEY:
            raise EnvironmentError("FREELLMAPI_API_KEY is not set in .env (copy it from FreeLLMAPI's Keys page)")
        auto_model = FREELLMAPI_SMART_MODEL if effort in HIGH_EFFORT else FREELLMAPI_MODEL
        if model:
            chain = [model]
        else:
            chain = [m for m in FREELLMAPI_AGENT_MODELS.get(agent, ()) if m] + [auto_model]
        return _chat_freellmapi_chain(chain, prompt, system, timeout, agent)
    if LLM_PROVIDER == "anthropic":
        return _chat_claude(prompt, system, timeout, model or CLAUDE_MODEL, effort or CLAUDE_EFFORT)
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' (expected 'openrouter', 'freellmapi' or 'anthropic').")


@contextmanager
def supervisor_notes(notes: list):
    """
    While active, every chat() call adds these notes to its system prompt.
    Master Agent wraps an agent's retry in this (master_agent._supervise),
    so each LLM decision of the retried agent knows what failed on the
    earlier attempts and what to change. Notes belong to this invocation;
    child workers inherit them through submit_with_context().
    """
    token = _supervisor_notes.set(tuple(str(n) for n in notes or [] if n))
    try:
        yield
    finally:
        _supervisor_notes.reset(token)


def submit_with_context(pool, fn, *args, **kwargs):
    """Submit a worker with a separate copy of the invocation's context."""
    return pool.submit(copy_context().run, fn, *args, **kwargs)


def _with_supervisor_notes(system: str) -> str:
    notes = _supervisor_notes.get()
    if not notes:
        return system
    block = ("MASTER AGENT NOTES -- earlier attempts of this task failed. Use these notes to avoid the same "
             "failure. They never override the rules or the output format given above.\n"
             + "\n".join(f"- {n}" for n in notes))
    return f"{system}\n\n{block}" if system else block


def last_served_by() -> str:
    """The model that answered this thread's most recent chat() call,
    e.g. "groq/llama-3.3-70b-versatile" -- for recording in a replication's
    environment notes. "" before any call."""
    return getattr(_last_call, "served_by", "")


def _record_call(provider: str, requested: str, served_by: str, started: float, usage: dict = None,
                 agent: str = None, error: str = None) -> None:
    """Append one call to CALL_LOG. A failed attempt (error set) is logged
    too, so a fallback from an agent's primary model stays visible."""
    if not error:
        _last_call.served_by = served_by
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider,
        "agent": agent,
        "requested_model": requested,
        "served_by": served_by,
        "duration_seconds": round(time.monotonic() - started, 2),
        "usage": usage or {},
    }
    if error:
        entry["error"] = error
    run_context.tag(entry)  # the pipeline run in progress, for run metrics
    try:
        with _log_lock:
            CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(CALL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass  # the call log must never fail the call itself


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

    started = time.monotonic()
    client = _client().with_options(timeout=float(max(timeout, CLAUDE_MIN_TIMEOUT_SECONDS)))
    with client.beta.messages.stream(**kwargs) as stream:
        message = stream.get_final_message()
    usage = getattr(message, "usage", None)
    _record_call("anthropic", model, str(getattr(message, "model", "") or model), started,
                 {"input_tokens": getattr(usage, "input_tokens", None),
                  "output_tokens": getattr(usage, "output_tokens", None)} if usage else None)

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


def _served_by(response, requested: str) -> str:
    """Which model answered: FreeLLMAPI's X-Routed-Via header
    ("<platform>/<model>"), else the response body's own "model" field."""
    routed = (getattr(response, "headers", None) or {}).get("X-Routed-Via")
    if isinstance(routed, str) and routed:
        return routed
    try:
        model = response.json().get("model")
        return model if isinstance(model, str) and model else requested
    except Exception:
        return requested


def _chat_freellmapi_chain(chain: list, prompt: str, system: str, timeout: int, agent: str = None) -> str:
    """Try each FreeLLMAPI model in order and return the first answer.

    A pinned model id is not failed over by the router itself, so a
    timeout, connection error, HTTP error or empty answer moves on to the
    next model here. Every model except the last skips the long 429
    backoff -- switching to the backup is faster than waiting out a
    per-minute window. The last model's error is raised if all fail."""
    for i, model in enumerate(chain):
        is_last = i == len(chain) - 1
        started = time.monotonic()
        try:
            content = _chat_openai_compatible(
                "freellmapi", FREELLMAPI_BASE_URL, FREELLMAPI_API_KEY, model, prompt, system, timeout,
                rate_limit_retries=RATE_LIMIT_RETRIES if is_last else 0, agent=agent,
            )
        except requests.RequestException as e:
            if is_last:
                raise
            _record_call("freellmapi", model, "", started, agent=agent, error=_short_error(e))
            print(f"  [LLM] {model} failed for {agent or 'call'} ({_short_error(e)}); trying {chain[i + 1]}...")
            continue
        if (content and content.strip()) or is_last:
            return content
        print(f"  [LLM] {model} returned an empty answer for {agent or 'call'}; trying {chain[i + 1]}...")


def _short_error(e: Exception) -> str:
    response = getattr(e, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}"
    return type(e).__name__


def _chat_openai_compatible(provider: str, base_url: str, api_key: str, model: str,
                            prompt: str, system: str, timeout: int,
                            rate_limit_retries: int = RATE_LIMIT_RETRIES, agent: str = None) -> str:
    """POST /chat/completions on an OpenAI-compatible API (OpenRouter,
    FreeLLMAPI). Per-minute 429s are retried with backoff; OpenRouter's
    daily free-tier cap fails fast (backoff cannot outlast a day)."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    started = time.monotonic()

    for attempt in range(rate_limit_retries + 1):
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages},
            timeout=timeout,
        )
        if response.status_code == 429 and _is_daily_quota(response):
            raise DailyQuotaExceeded(_daily_quota_message(response))
        if response.status_code == 429 and attempt < rate_limit_retries:
            wait = RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [LLM] Rate limited (429). Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        body = response.json()
        _record_call(provider, model, _served_by(response, model), started,
                     body.get("usage") if isinstance(body.get("usage"), dict) else None, agent=agent)
        return body["choices"][0]["message"]["content"]


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
