"""
What did a web request actually return? An HTTP 200 is not proof that the
paper, dataset or repository was retrieved: the body can be a CAPTCHA, a
JavaScript bot check, a login page or an unrelated page. Every retrieval
-- fetched directly -- is classified here, in three steps:

  1. Reachability: did the server respond at all?
  2. Accessibility: did it serve content, or a challenge / login page?
  3. Validity: is the content the expected kind (a PDF, tabular data, an
     HTML page) and, when expected_text is given, the expected item?

classify() returns one verdict:

  ACCESSIBLE     expected content found                -> accept, then verify it
  CAPTCHA        CAPTCHA widget or text                 -> reject and seek an alternate source
  JS_CHALLENGE   JavaScript bot check ("Just a moment") -> reject and seek an alternate source
  BOT_BLOCKED    403/503 with bot-protection evidence   -> reject and seek an alternate source
  RATE_LIMITED   HTTP 429                               -> honor Retry-After; seek an alternate source
  LOGIN_REQUIRED sign-in page                           -> restricted; never bypassed
  FORBIDDEN      403 with no bot-protection evidence    -> restricted
  NOT_FOUND      404/410                                -> dead link
  HTTP_ERROR     any other 4xx/5xx                      -> failed
  WRONG_CONTENT  not the expected kind or item          -> reject

This is a heuristic. A page that only mentions CAPTCHAs, or that loads a
CAPTCHA script for a login form, must not count as a challenge, so text
and widget evidence only counts when the page has little other content or
the expected content is missing.
"""
import re

from bs4 import BeautifulSoup

BLOCKED_VERDICTS = ("CAPTCHA", "JS_CHALLENGE", "BOT_BLOCKED")

CAPTCHA_MARKERS = ("verify you are human", "complete the captcha", "solve the captcha", "are you a robot",
                   "automated access detected", "unusual traffic from your", "please complete the security check",
                   "press & hold to confirm", "confirm you are not a robot")
JS_CHALLENGE_MARKERS = ("checking your browser", "just a moment", "enable javascript and cookies to continue",
                        "attention required", "ddos protection by", "please wait while we verify")
# DOM and script evidence of a challenge (Google reCAPTCHA, hCaptcha,
# Cloudflare Turnstile and challenge pages, PerimeterX, DataDome, Incapsula).
CHALLENGE_SELECTORS = (".g-recaptcha, .h-captcha, .cf-turnstile, #challenge-form, #challenge-running, "
                       "#cf-challenge-running, #px-captcha, #captcha-container")
CHALLENGE_SCRIPT_MARKERS = ("/cdn-cgi/challenge-platform", "cf-chl-", "captcha-delivery.com", "px-captcha",
                            "_incapsula_resource", "hcaptcha.com/1/api.js", "challenges.cloudflare.com")
# Response headers set by bot-protection layers on a challenge response.
BOT_HEADERS = ("cf-mitigated", "x-datadome", "x-px-block", "x-amzn-waf-action", "x-sucuri-block")
LOGIN_MARKERS = ("sign in", "log in", "login", "sign-in", "single sign-on")
# A real article or dataset page has much more text than a challenge page.
SHORT_PAGE_CHARS = 3000
MAX_PARSE_BYTES = 2_000_000


def _is_html(content_type: str, body: bytes) -> bool:
    if "html" in (content_type or "").lower():
        return True
    head = (body or b"")[:300].lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def expected_text_found(text: str, expected_text: str, min_share: float = 0.8) -> bool:
    """True when at least min_share of the expected text's words (e.g. a
    paper title's) occur in text. Words of 3 or fewer letters are ignored."""
    words = [w for w in _normalize(expected_text).split() if len(w) > 3]
    if not words:
        return True
    haystack = f" {_normalize(text)} "
    return sum(f" {w} " in haystack for w in words) / len(words) >= min_share


def detect_challenge(html: str, headers: dict = None) -> tuple[str, str]:
    """("captcha" | "js_challenge" | "", evidence) for an HTML page. Strong
    evidence (challenge widgets, challenge scripts, bot-protection headers,
    a challenge page title) always counts; body text only counts on a short
    page, so an article that discusses CAPTCHAs is not a challenge."""
    headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    header = next((h for h in BOT_HEADERS if h in headers), "")
    soup = BeautifulSoup(html or "", "html.parser")
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    for tag in soup(["style", "noscript"]):
        tag.decompose()
    scripts = " ".join((s.get("src") or "") + " " + (s.string or "") for s in soup.find_all("script")).lower()
    for tag in soup(["script"]):
        tag.decompose()
    body = " ".join(soup.get_text(" ", strip=True).lower().split())
    short = len(body) < SHORT_PAGE_CHARS

    if header:
        return "js_challenge", f"response header '{header}'"
    if "captcha" in title or "verify you are human" in title:
        return "captcha", f"page title '{title[:80]}'"
    if "just a moment" in title or "attention required" in title or "checking your browser" in title:
        return "js_challenge", f"page title '{title[:80]}'"
    script = next((m for m in CHALLENGE_SCRIPT_MARKERS if m in scripts or m in (html or "").lower()), "")
    widgets = soup.select(CHALLENGE_SELECTORS)
    if short and widgets:
        return "captcha", f"challenge element '{widgets[0].name}.{' '.join(widgets[0].get('class') or [])}'"
    if short and script:
        return "js_challenge", f"challenge script '{script}'"
    if short:
        marker = next((m for m in CAPTCHA_MARKERS if m in body), "")
        if marker:
            return "captcha", f"text '{marker}'"
        marker = next((m for m in JS_CHALLENGE_MARKERS if m in body), "")
        if marker:
            return "js_challenge", f"text '{marker}'"
    return "", ""


def detect_login(html: str) -> bool:
    """A sign-in page: a password field on a short page, or a sign-in title."""
    soup = BeautifulSoup(html or "", "html.parser")
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    short = len(" ".join(soup.get_text(" ", strip=True).split())) < SHORT_PAGE_CHARS
    has_password = bool(soup.select("input[type=password]"))
    return (has_password and short) or any(title == m or title.startswith(m + " ") or title.startswith(m + " |") for m in LOGIN_MARKERS)


def classify(status_code: int, headers: dict = None, body: bytes = b"", expected: str = "any",
             expected_text: str = "") -> dict:
    """{"verdict", "challenge_type", "reason"} for one response (see the
    module docstring). expected: "pdf", "tabular", "html" or "any".
    expected_text: e.g. a paper title that must appear in an HTML page."""
    headers = headers or {}
    content_type = next((str(v) for k, v in headers.items() if str(k).lower() == "content-type"), "")
    body = body or b""
    html = _is_html(content_type, body)
    text = body[:MAX_PARSE_BYTES].decode("utf-8", errors="ignore") if html else ""

    def verdict(name: str, reason: str, challenge: str = "") -> dict:
        return {"verdict": name, "challenge_type": challenge, "reason": reason}

    if not 200 <= status_code < 300 and status_code < 400:
        return verdict("HTTP_ERROR", f"HTTP {status_code}, no final successful response")
    if status_code == 429:
        return verdict("RATE_LIMITED", "HTTP 429 (rate limited)")
    if status_code in (403, 503):
        challenge, evidence = detect_challenge(text, headers) if html else ("", "")
        if not challenge:
            lower_headers = {str(k).lower() for k in headers}
            header = next((h for h in BOT_HEADERS if h in lower_headers), "")
            challenge, evidence = ("js_challenge", f"response header '{header}'") if header else ("", "")
        if challenge:
            return verdict("BOT_BLOCKED", f"HTTP {status_code} with bot protection ({evidence})", challenge)
        return verdict("FORBIDDEN" if status_code == 403 else "HTTP_ERROR",
                       f"HTTP {status_code} with no sign of a bot check")
    if status_code in (404, 410):
        return verdict("NOT_FOUND", f"HTTP {status_code}")
    if status_code >= 400:
        return verdict("HTTP_ERROR", f"HTTP {status_code}")

    if html:
        challenge, evidence = detect_challenge(text, headers)
        if challenge == "captcha":
            return verdict("CAPTCHA", f"HTTP {status_code}, CAPTCHA detected ({evidence})", challenge)
        if challenge:
            return verdict("JS_CHALLENGE", f"HTTP {status_code}, JavaScript bot check detected ({evidence})",
                           challenge)
        if detect_login(text):
            return verdict("LOGIN_REQUIRED", f"HTTP {status_code}, a sign-in page")
    if not body.strip():
        return verdict("WRONG_CONTENT", "an empty response")
    if expected == "pdf" and not body.startswith(b"%PDF"):
        return verdict("WRONG_CONTENT", "not a PDF" + (" (an HTML page)" if html else ""))
    if expected == "tabular" and html:
        return verdict("WRONG_CONTENT", "an HTML page, not a data file")
    if expected == "html" and not html and not body.strip():
        return verdict("WRONG_CONTENT", "an empty response")
    if html and expected_text and not expected_text_found(text, expected_text):
        return verdict("WRONG_CONTENT", "the page does not contain the expected item ("
                       f"'{expected_text[:60]}')")
    return verdict("ACCESSIBLE", f"HTTP {status_code}, content found")
