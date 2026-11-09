"""
Finds and downloads a paper's legal open-access PDF for the dashboard.

This is a dashboard feature, not a pipeline stage, so it lives here and
saves into the dashboard's own folder (dashboard_files/pdfs/), never into
papers/ or any other folder the pipeline owns.
"""
import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
TIMEOUT_SECONDS = 30
MAX_BYTES = 50 * 1024 * 1024
# Identify honestly as what we are, as the Data Agent does for Kaggle.
USER_AGENT = "DS340W-AI-Agents-Dashboard/1.0 (open-access paper fetch)"

# OpenAlex sometimes lists a publisher's reference-list PDF as the work's
# pdf_url (seen live: Nature's "<article>_reference.pdf"). That file is a
# valid PDF but not the paper, so such links are skipped.
_NOT_THE_PAPER = ("_reference.pdf",)


class PdfError(Exception):
    """No usable PDF; the message is shown to the user."""


def _pdf_url(work: dict) -> str:
    locations = [work.get("best_oa_location"), work.get("primary_location"), *(work.get("locations") or [])]
    for location in locations:
        url = (location or {}).get("pdf_url") or ""
        if url and not url.lower().endswith(_NOT_THE_PAPER):
            return url
    return ""


def _same_title(a: str, b: str) -> bool:
    def norm(t: str) -> str:
        return "".join(ch for ch in (t or "").lower() if ch.isalnum())
    return bool(norm(a)) and norm(a) == norm(b)


def open_access_pdf_url(doi: str = "", title: str = "") -> str:
    """OpenAlex's open-access PDF link by DOI, else by exact title; "" if none.
    A paywalled paper is never fetched any other way."""
    try:
        if doi:
            response = requests.get(f"{OPENALEX_WORKS_URL}/doi:{doi}", timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                return _pdf_url(response.json())
        if title:
            response = requests.get(OPENALEX_WORKS_URL, params={"search": title, "per_page": 1},
                                    timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                results = response.json().get("results") or []
                if results and _same_title(results[0].get("title", ""), title):
                    return _pdf_url(results[0])
    except (requests.RequestException, ValueError):
        return ""
    return ""


def download_pdf(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS, stream=True, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise PdfError("The PDF is larger than the 50 MB limit.")
            chunks.append(chunk)
    except requests.RequestException as e:
        raise PdfError(f"The open-access PDF could not be downloaded: {e}") from e
    content = b"".join(chunks)
    # Sniff, don't trust: an "open-access PDF" link can still serve an HTML page.
    if not content.startswith(b"%PDF"):
        raise PdfError("The open-access link did not return a PDF file (it may need a browser login).")
    return content
