"""
The Parent Paper's own full text, for the agents that must read it
directly (Data Agent: which dataset, where, how large; Replication Agent:
the methodology; Evaluation Agent: the published results and procedure).

Only a legal open-access copy is ever fetched: OpenAlex's open-access PDF
links for the paper (by DOI, else by exact title), and the paper's own URL.
Article pages are inspected for individual View PDF links or citation PDF
metadata; full-issue and bulk downloads are excluded. Blocked links are
retried through the local Camoufox browser by src/search/page_fetch.py.
Content is sniffed (%PDF), never
trusted, and the text is extracted with pypdf. The PDF and its text are
kept in papers/parent/ (gitignored), so each agent reads the same copy and
the paper is downloaded once per Parent Paper.

When no open-access PDF exists, the agents fall back to the paper's stored
metadata (official abstract, methodology, results summary), and the result
says so ("ABSTRACT_ONLY") -- the agents must not present that as having
read the full paper.

This is a pipeline module. The dashboard has its own PDF fetcher and must
not import this one (they are separate programs).
"""
import io
import json
import re
from pathlib import Path

import requests

from src.search import page_fetch, content_validator
from src.search.pdf_links import article_pdf_links, is_bulk_download

REPO_ROOT = Path(__file__).parent.parent.parent
PARENT_PAPER_DIR = REPO_ROOT / "papers" / "parent"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
TIMEOUT_SECONDS = 30
MAX_PDF_BYTES = 50 * 1024 * 1024
MIN_FULL_TEXT_CHARS = 2000  # less than this is a cover page or a failed extraction
# Identify direct requests honestly; Camoufox is used only when blocked.
USER_AGENT = "DS340W-AI-Agents-Pipeline/1.0 (open-access paper fetch)"

# OpenAlex sometimes lists a publisher's reference-list PDF as a work's
# pdf_url (seen live: Nature's "<article>_reference.pdf"); it is not the paper.
_NOT_THE_PAPER = ("_reference.pdf",)
_ARXIV_ABS = re.compile(r"arxiv\.org/abs/([^\s?#]+)")


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:80] or "parent-paper"


def _same_title(a: str, b: str) -> bool:
    def norm(t: str) -> str:
        return "".join(ch for ch in (t or "").lower() if ch.isalnum())
    return bool(norm(a)) and norm(a) == norm(b)


def _openalex_work(doi: str, title: str) -> dict | None:
    try:
        if doi:
            response = requests.get(f"{OPENALEX_WORKS_URL}/doi:{doi}", timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()
        if title:
            response = requests.get(OPENALEX_WORKS_URL, params={"search": title, "per_page": 1},
                                    timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                results = response.json().get("results") or []
                if results and _same_title(results[0].get("title", ""), title):
                    return results[0]
    except (requests.RequestException, ValueError):
        return None
    return None


def candidate_pdf_urls(parent_paper: dict) -> list:
    """PDF links and article pages from which an individual PDF can be found."""
    urls = []
    work = _openalex_work(parent_paper.get("doi", ""), parent_paper.get("title", ""))
    if work:
        locations = [work.get("best_oa_location"), work.get("primary_location"), *(work.get("locations") or [])]
        urls += [(loc or {}).get("pdf_url") or "" for loc in locations]
        urls.append((work.get("open_access") or {}).get("oa_url") or "")
    access = parent_paper.get("reachability") or {}
    if access.get("reachable") and access.get("access_level") == "FULL_TEXT":
        urls.append(access.get("source_url", ""))
    paper_url = parent_paper.get("paper_url", "") or ""
    arxiv = _ARXIV_ABS.search(paper_url)
    if arxiv:
        urls.append(f"https://arxiv.org/pdf/{arxiv.group(1)}")
    elif paper_url:
        urls.append(paper_url)

    unique = []
    for url in urls:
        if url and not is_bulk_download(url) and not url.lower().endswith(_NOT_THE_PAPER) and url not in unique:
            unique.append(url)
    return unique


def _download_pdf(url: str, _seen: set | None = None, _depth: int = 0) -> bytes | None:
    """The PDF at url, or None. Blocked responses can use local Camoufox.
    Accessible HTML is inspected for individual article PDF controls, with
    bounded recursion and no bulk downloads."""
    if is_bulk_download(url) or _depth > 2:
        return None
    seen = _seen if _seen is not None else set()
    if url in seen:
        return None
    seen.add(url)
    response = None
    links = []
    try:
        response = page_fetch.get(url, timeout=TIMEOUT_SECONDS, stream=True, expect_pdf=True,
                                  headers={"User-Agent": USER_AGENT})
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_PDF_BYTES:
                return None
            chunks.append(chunk)
        content = b"".join(chunks)
        result = content_validator.classify(response.status_code, response.headers, content, expected="pdf")
        if result["verdict"] == "ACCESSIBLE":
            return content
        page = content_validator.classify(response.status_code, response.headers, content)
        if page["verdict"] == "ACCESSIBLE" and content_validator._is_html(
                response.headers.get("Content-Type", ""), content):
            resolved = getattr(response, "url", url)
            links = article_pdf_links(content, resolved if isinstance(resolved, str) else url)
    except requests.RequestException:
        return None
    finally:
        if response is not None:
            response.close()
    for link in links[:3]:
        content = _download_pdf(link, seen, _depth + 1)
        if content:
            return content
    return None


def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """(text, page count); ("", 0) when the PDF cannot be read."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return "", 0
    text = "\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)   # rejoin words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip(), len(pages)


def _metadata_text(parent_paper: dict) -> str:
    fields = (("Title", "title"), ("Venue", "venue"), ("Year", "year"),
              ("Abstract", "official_abstract"), ("Abstract (search snippet)", "abstract"),
              ("Methodology", "methodology"), ("Dataset", "dataset"), ("Dataset source", "dataset_source"),
              ("Results summary", "results_summary"))
    parts, seen = [], set()
    for label, key in fields:
        value = str(parent_paper.get(key) or "").strip()
        if value and value not in seen:
            seen.add(value)
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def get_parent_paper_text(parent_paper: dict, use_cache: bool = True) -> dict:
    """
    The Parent Paper's text, from its open-access PDF when one exists.

    Returns {"status": "FULL_TEXT" | "ABSTRACT_ONLY", "text", "source_url",
    "pdf_path", "text_path", "pages", "notes"}. The PDF and text are cached
    in papers/parent/<slug>.*, so every agent reads the same copy.
    """
    parent_paper = parent_paper or {}
    slug = _slug(parent_paper.get("title", ""))
    pdf_path = PARENT_PAPER_DIR / f"{slug}.pdf"
    text_path = PARENT_PAPER_DIR / f"{slug}.txt"
    meta_path = PARENT_PAPER_DIR / f"{slug}.json"

    if use_cache and text_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return {**meta, "text": text_path.read_text(encoding="utf-8"),
                    "pdf_path": str(pdf_path) if pdf_path.is_file() else "", "text_path": str(text_path)}
        except (OSError, ValueError):
            pass

    # A PDF the user saved there themselves (e.g. downloaded in a browser
    # from a publisher that serves PDFs only to browsers) is read first.
    if pdf_path.is_file():
        text, pages = extract_text(pdf_path.read_bytes())
        if len(text) >= MIN_FULL_TEXT_CHARS:
            text_path.write_text(text, encoding="utf-8")
            meta = {"status": "FULL_TEXT", "source_url": "", "pages": pages,
                    "notes": f"Read the Parent Paper's PDF saved in {pdf_path} ({pages} pages)."}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[Paper Text] {meta['notes']}")
            return {**meta, "text": text, "pdf_path": str(pdf_path), "text_path": str(text_path)}

    tried = []
    for url in candidate_pdf_urls(parent_paper):
        tried.append(url)
        content = _download_pdf(url)
        if not content:
            continue
        text, pages = extract_text(content)
        if len(text) < MIN_FULL_TEXT_CHARS:
            continue
        if parent_paper.get("title") and not content_validator.expected_text_found(text[:6000], parent_paper["title"]):
            continue
        PARENT_PAPER_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(content)
        text_path.write_text(text, encoding="utf-8")
        meta = {"status": "FULL_TEXT", "source_url": url, "pages": pages,
                "notes": f"Read the open-access PDF ({pages} pages) from {url}."}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[Paper Text] Read the Parent Paper's full text ({pages} pages) from {url}.")
        return {**meta, "text": text, "pdf_path": str(pdf_path), "text_path": str(text_path)}

    note = ("No open-access PDF could be downloaded" + (f" (tried {len(tried)} link(s))" if tried else
            " (none is listed)") + "; only the paper's stored metadata (abstract, methodology, results "
            "summary) is available. This is NOT the full paper.")
    print(f"[Paper Text] {note}")
    return {"status": "ABSTRACT_ONLY", "text": _metadata_text(parent_paper), "source_url": "",
            "pdf_path": "", "text_path": "", "pages": 0, "notes": note}


# Keywords that mark the paragraphs each agent needs most.
DATA_KEYWORDS = ("dataset", "data set", "data source", "collected", "samples", "records", "instances",
                 "transactions", "rows", "kaggle", "uci", "repository", "download", "http", "available at",
                 "features", "attributes", "class distribution", "imbalance", "label")
METHOD_KEYWORDS = ("method", "model", "architecture", "algorithm", "preprocess", "normaliz", "standardiz",
                   "scal", "encod", "feature", "smote", "oversampl", "undersampl", "resampl", "split",
                   "train", "test", "validation", "cross-validation", "fold", "hyperparameter", "learning rate",
                   "epoch", "batch", "layer", "estimator", "depth", "regulariz", "loss", "optimizer", "seed",
                   "pipeline", "procedure", "experiment")
RESULT_KEYWORDS = ("result", "accuracy", "precision", "recall", "f1", "auc", "roc", "table", "outperform",
                   "baseline", "compared", "performance", "score", "%")


def excerpt(text: str, keywords: tuple, max_chars: int = 12000, head_chars: int = 1500) -> str:
    """
    The parts of a paper's text an agent needs, within max_chars:
    the opening (title and abstract) plus the paragraphs with the most
    keyword hits, kept in the paper's own order. Deterministic -- the same
    text always gives the same excerpt.
    """
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[:head_chars]
    paragraphs = []
    for block in re.split(r"\n\s*\n|(?<=[.:])\n(?=[A-Z0-9])", text[head_chars:]):
        block = block.strip()
        # PDF text often has no paragraph breaks; long blocks become windows.
        paragraphs += [block[i:i + 1500] for i in range(0, len(block), 1500)] if block else []
    scored = []
    for index, paragraph in enumerate(paragraphs):
        lower = paragraph.lower()
        hits = sum(lower.count(k) for k in keywords)
        if hits:
            scored.append((hits / (1 + len(paragraph) / 1500), index, paragraph))
    budget = max_chars - len(head)
    chosen = []
    for _score, index, paragraph in sorted(scored, key=lambda s: (-s[0], s[1])):
        if len(paragraph) + 5 > budget:
            continue
        chosen.append((index, paragraph))
        budget -= len(paragraph) + 5
    body = "\n[...]\n".join(p for _i, p in sorted(chosen))
    return f"{head}\n[...]\n{body}" if body else head
