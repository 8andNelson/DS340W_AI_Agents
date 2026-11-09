"""Evidence-based paper access checks; search snippets never prove access."""
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.search import content_validator, page_fetch, web_search

MAX_ALTERNATIVES = 3


def _identity(body: bytes, title: str, doi: str) -> tuple[bool, str]:
    if not (title or doi):
        # Data and Code Discovery reuse the generic reachability helper. A
        # reachable CSV or repository is a resource, not verified paper text.
        return bool(body.strip()), "RESOURCE"
    if body.startswith(b"%PDF"):
        # A PDF signature alone is insufficient evidence of the paper's identity.
        from src.search.paper_fulltext import extract_text, MIN_FULL_TEXT_CHARS
        text, _ = extract_text(body)
        if len(text) < MIN_FULL_TEXT_CHARS:
            return False, ""
        identity_text = text[:6000]
        kind = "FULL_TEXT"
    else:
        if not content_validator._is_html("", body):
            return False, ""
        soup = BeautifulSoup(body, "html.parser")
        # A title buried in another article's references is not identity evidence.
        headings = [tag.get_text(" ", strip=True) for tag in soup.select("title, h1")]
        metadata = [m.get("content", "") for m in soup.select("meta[content]")
                    if (m.get("name") or m.get("property") or "").lower() in (
                        "citation_title", "dc.title", "dcterms.title", "og:title",
                        "citation_doi", "dc.identifier", "dcterms.identifier")]
        identity_text = " ".join(headings + metadata)
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        kind = "LANDING_PAGE"
    def normalized(value):
        return re.sub(r"[^a-z0-9]", "", value.lower())
    title_match = bool(title and normalized(title) in normalized(identity_text))
    doi_value = doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:")
    doi_match = bool(doi_value and re.search(re.escape(doi_value) + r"(?![a-z0-9./_-])", identity_text.lower()))
    return bool(text.strip()) and (title_match or doi_match or not (title or doi)), kind


def check(url: str, title: str = "", doi: str = "", timeout: int = 15, source_check=None) -> dict:
    """Fetch original, then at most three SerpAPI alternatives if necessary.
    Returned evidence distinguishes a landing page from readable full text.
    An alternate must contain the title or DOI; a result snippet never suffices.
    """
    attempts = []

    def attempt(candidate):
        response = None
        try:
            if urlparse(candidate).scheme not in ("http", "https"):
                raise ValueError("URL must use HTTP or HTTPS")
            if source_check is not None and not source_check(candidate):
                raise ValueError("prohibited paper source")
            response = page_fetch.get(candidate, timeout=timeout, stream=True,
                                      allow_redirects=True,
                                      headers={"User-Agent": "DS340W-AI-Agents-Pipeline/1.0"})
            chunks, size = [], 0
            # Bound PDFs and HTML alike. Truncation cannot prove PDF validity.
            limit = 50 * 1024 * 1024 if "pdf" in response.headers.get("Content-Type", "").lower() else 2 * 1024 * 1024
            for chunk in response.iter_content(8192):
                size += len(chunk)
                if size > limit:
                    raise ValueError("response exceeds paper inspection size limit")
                chunks.append(chunk)
            body = b"".join(chunks)
            result = content_validator.classify(response.status_code, response.headers, body)
            matched, kind = _identity(body, title, doi) if result["verdict"] == "ACCESSIBLE" else (False, "")
            if result["verdict"] == "ACCESSIBLE" and not matched:
                result = {"verdict": "WRONG_CONTENT", "reason": "paper title/DOI not found in fetched content"}
            final_url = getattr(response, "url", "")
            if not isinstance(final_url, str) or not final_url:
                final_url = candidate
            if source_check is not None and not source_check(final_url):
                raise ValueError("redirected to a prohibited paper source")
            record = {"url": candidate, "final_url": final_url, "status_code": response.status_code,
                      **result, "access_level": kind if matched else ""}
        except (requests.RequestException, ValueError) as error:
            failed_response = getattr(error, "response", None)
            record = {"url": candidate,
                      "status_code": failed_response.status_code if failed_response is not None else 0,
                      "verdict": getattr(error, "verdict", "FETCH_FAILED"),
                      "reason": str(error), "access_level": ""}
        finally:
            if response is not None:
                response.close()
        attempts.append(record)
        return record

    found = attempt(url)
    if found["verdict"] != "ACCESSIBLE" and (title or doi):
        try:
            results = web_search.search(f'"{title or doi}"', count=MAX_ALTERNATIVES,
                                        provider_name="serpapi")
        except Exception as error:
            results = []
            attempts.append({"verdict": "SEARCH_FAILED", "reason": str(error), "provider": "serpapi"})
        seen = {url}
        for result in results[:MAX_ALTERNATIVES]:
            candidate = result.get("url", "")
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            found = attempt(candidate)
            if found["verdict"] == "ACCESSIBLE":
                break
    return {"reachable": found["verdict"] == "ACCESSIBLE",
            "status_code": found["status_code"], "source_url": found.get("final_url", found["url"]),
            "access_level": found["access_level"], "verdict": found["verdict"],
            "reason": found["reason"], "attempts": attempts}
