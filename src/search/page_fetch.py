"""Direct retrieval first, then a bounded local Camoufox browser if blocked."""
import json
import threading
from itertools import chain
from datetime import datetime, timezone
from pathlib import Path

import requests
from src.search import content_validator
from src.search import camoufox_fetch

FETCH_LOG = Path(__file__).parent.parent.parent / "logs" / "page_fetch_calls.jsonl"
CHALLENGE_SCAN_BYTES = 128 * 1024
_lock = threading.Lock()
REJECTED_VERDICTS = (*content_validator.BLOCKED_VERDICTS, "LOGIN_REQUIRED", "RATE_LIMITED", "FORBIDDEN")


class BlockedPageError(requests.RequestException):
    """The server responded, but did not provide accessible content."""
    def __init__(self, response, assessment):
        super().__init__(assessment["reason"], response=response)
        self.verdict = assessment["verdict"]


def blocked_reason(status_code: int, content_type: str = "", body: bytes = b"",
                   headers: dict = None) -> str:
    result = content_validator.classify(status_code, {**(headers or {}), "Content-Type": content_type}, body)
    return result["reason"] if result["verdict"] in REJECTED_VERDICTS else ""


def response_prefix(response, streamed: bool = False) -> bytes:
    """Inspect bounded bytes and replay them so downloads lose no content."""
    if not streamed:
        return response.content[:CHALLENGE_SCAN_BYTES]
    iterator = response.iter_content(chunk_size=8192)
    chunks, size = [], 0
    for chunk in iterator:
        chunks.append(chunk)
        size += len(chunk)
        if size >= CHALLENGE_SCAN_BYTES:
            break
    original_iter_content = response.iter_content
    used = False

    def replay(chunk_size=1, decode_unicode=False):
        nonlocal used
        if used:
            yield from original_iter_content(chunk_size, decode_unicode=decode_unicode)
            return
        used = True
        for chunk in chain(chunks, iterator):
            yield chunk.decode(response.encoding or "utf-8", errors="replace") if decode_unicode else chunk

    response.iter_content = replay
    return b"".join(chunks)[:CHALLENGE_SCAN_BYTES]



def _log(url: str, status: int, verdict: str, reason: str, size: int = 0,
         provider: str = "direct") -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "url": url, "status": status, "verdict": verdict, "reason": reason,
             "inspected_bytes": size, "provider": provider}
    try:
        with _lock:
            FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
            with FETCH_LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def get(url: str, timeout: int = 30, headers: dict = None,
        expect_pdf: bool = False, **kwargs) -> requests.Response:
    """Fetch directly, then use Camoufox for a blocked public page if enabled.

    HTTP errors such as 404 remain available for callers to classify.
    Login pages and rate limits are not retried with a browser. Authenticated
    requests stay direct; caller credentials are never copied to Camoufox.
    """
    response = None
    try:
        response = requests.get(url, timeout=timeout, headers=headers, **kwargs)
        prefix = response_prefix(response, streamed=bool(kwargs.get("stream")))
        assessment = content_validator.classify(response.status_code, response.headers, prefix)
    except Exception as error:
        if response is not None:
            response.close()
        _log(url, 0, "FETCH_FAILED", type(error).__name__)
        raise
    _log(url, response.status_code, assessment["verdict"], assessment["reason"], len(prefix))
    credentials = (kwargs.get("auth") is not None or bool(kwargs.get("cookies")) or
                   any(str(name).lower() in ("authorization", "cookie", "proxy-authorization", "x-api-key")
                       for name in (headers or {})))
    retryable = (assessment["verdict"] in (*content_validator.BLOCKED_VERDICTS, "FORBIDDEN") or
                 response.status_code == 503)
    if retryable and camoufox_fetch.CAMOUFOX_ENABLED and not credentials:
        response.close()
        try:
            print("  [Fetch] Page blocked; trying local Camoufox browser...")
            unblocked = camoufox_fetch.fetch(url, expect_pdf=expect_pdf)
            verified = content_validator.classify(unblocked.status_code, unblocked.headers,
                                                  unblocked.content,
                                                  expected="pdf" if expect_pdf else "any")
            if verified["verdict"] != "ACCESSIBLE":
                unblocked.close()
                raise requests.RequestException("Camoufox: " + verified["reason"])
            _log(url, unblocked.status_code, verified["verdict"], verified["reason"],
                 len(unblocked.content), provider="camoufox")
            response.close()
            return unblocked
        except requests.RequestException as error:
            _log(url, 0, "FETCH_FAILED", str(error)[:300], provider="camoufox")
            print("  [Fetch] Camoufox could not retrieve accessible content; keeping the blocked verdict.")
    if assessment["verdict"] in REJECTED_VERDICTS:
        response.close()
        raise BlockedPageError(response, assessment)
    return response
