"""
Secret scrubber: runs on every row before it is written to the memory
store (docs/CLAUDE_BACKEND_PLAN.md, section 5.3). The team's store is
shared and the repository is public, so memory content must never carry a
credential.

It replaces with "[REDACTED:<kind>]":
  * the value of every secret variable set in .env (a name containing
    KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL or DATABASE_URL; 6 characters
    or longer). Other .env values (LLM_PROVIDER=freellmapi, ...) are
    settings, not secrets, and appear in run records on purpose;
  * known key and token formats (OpenAI/Anthropic/OpenRouter "sk-...",
    Slack "xox?-...", GitHub "ghp_..."/"github_pat_...", Google "AIza...",
    AWS "AKIA...", Hugging Face "hf_...", JWTs);
  * "Bearer <token>" headers, private-key blocks, connection strings with
    a password ("postgres://user:pass@host"), and NAME=value assignments
    whose name ends in KEY, TOKEN, SECRET or PASSWORD;
  * email addresses.

Best effort: a pattern list cannot prove that no secret is left. It is
one of several controls (no dataset rows are stored; the pipeline's
database role cannot delete; memories can be quarantined).
"""
import re
from functools import lru_cache
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent.parent / ".env"
MIN_ENV_VALUE_CHARS = 6
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|DATABASE_URL", re.I)

_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("connection_string", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s@/]+@[^\s/]+")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{8,}=*")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("secret_assignment", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD))(\s*[=:]\s*)(['\"]?)([^\s'\"]{4,})\3")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
)


@lru_cache(maxsize=1)
def _env_values() -> tuple:
    """Values of secret variables set in .env, longest first (so a value
    containing another is replaced whole). Read once per process."""
    if not ENV_FILE.exists():
        return ()
    try:
        from dotenv import dotenv_values
        values = dotenv_values(ENV_FILE)
    except Exception:
        return ()
    found = {str(v).strip() for name, v in values.items()
             if v and _SECRET_NAME.search(name) and len(str(v).strip()) >= MIN_ENV_VALUE_CHARS}
    return tuple(sorted(found, key=len, reverse=True))


def scrub_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    for value in _env_values():
        if value in text:
            text = text.replace(value, "[REDACTED:env_value]")
    for kind, pattern in _PATTERNS:
        if kind == "secret_assignment":
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED:{kind}]", text)
        else:
            text = pattern.sub(f"[REDACTED:{kind}]", text)
    return text


def scrub(value):
    """Scrub every string inside a value (dicts, lists and nested JSON)."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value
