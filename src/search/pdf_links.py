"""Discover individual article PDFs; never follow full-issue/bulk downloads."""
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup


def is_bulk_download(url: str, label: str = "") -> bool:
    value = (urlsplit(url).path + " " + label).lower()
    return any(marker in value for marker in (
        "full issue", "full-issue", "full_issue", "fullissue", "download all",
        "download-all", "/science/issue/", "/science/journal/", "/science/journals/",
    )) or urlsplit(url).path.lower().endswith((".zip", ".tar", ".gz"))


def article_pdf_links(html: bytes, page_url: str) -> list[str]:
    """Resolve PDF controls and metadata, preserving fresh signed URLs.

    ScienceDirect targets must identify the current article's PII, including
    links to sciencedirectassets.com. Similar articles and issue files fail.
    JavaScript-only buttons cannot be executed by this HTTP fetcher.
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for meta in soup.select('meta[name="citation_pdf_url"], meta[property="citation_pdf_url"]'):
        candidates.append((meta.get("content", ""), "citation_pdf_url"))
    for element in soup.select("a[href], iframe[src], embed[src], object[data]"):
        link = element.get("href") or element.get("src") or element.get("data") or ""
        label = " ".join((element.get_text(" ", strip=True), element.get("title", ""),
                          element.get("aria-label", "")))
        path = urlsplit(link).path.lower()
        if "pdf" in label.lower() or path.endswith(".pdf") or "/pdfft" in path:
            candidates.append((link, label))

    pii = re.search(r"/pii/(S\d{16})", urlsplit(page_url).path)
    links = []
    for link, label in candidates:
        target = urljoin(page_url, link)
        parsed = urlsplit(target)
        if parsed.scheme not in ("https", "http") or not link or link.startswith("#"):
            continue
        if is_bulk_download(target, label):
            continue
        if any(marker in parsed.path.lower() for marker in ("_reference.pdf", "supplement", "mmc")):
            continue
        if pii:
            host = (parsed.hostname or "").lower()
            if not any(host == domain or host.endswith("." + domain)
                       for domain in ("sciencedirect.com", "sciencedirectassets.com")):
                continue
            if pii.group(1) not in parsed.path:
                continue
        if target not in links:
            links.append(target)
    return links
