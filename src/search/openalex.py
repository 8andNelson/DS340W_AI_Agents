"""
OpenAlex lookups (https://api.openalex.org) -- free, no API key, and far
less rate-limited than Semantic Scholar's keyless tier (which returned 429
on most live requests). Used to confirm metadata of papers already found by
search, never to discover new ones: a work's official abstract, year and
venue by DOI or by exact title.
"""
import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
TIMEOUT_SECONDS = 20


def _abstract_from_inverted_index(index: dict) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    positions = [(pos, word) for word, poss in (index or {}).items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def _normalize(work: dict) -> dict:
    source = ((work.get("primary_location") or {}).get("source") or {})
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    return {
        "title": work.get("title") or "",
        "year": work.get("publication_year") or 0,
        "venue": source.get("display_name") or "",
        "doi": doi,
        "abstract": _abstract_from_inverted_index(work.get("abstract_inverted_index")),
        "is_retracted": bool(work.get("is_retracted")),
    }


def work_by_doi(doi: str) -> dict | None:
    """The work with this DOI, or None if not found / unavailable."""
    if not doi:
        return None
    try:
        response = requests.get(f"{OPENALEX_WORKS_URL}/doi:{doi}", timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    return _normalize(response.json())


def match_title(title: str) -> dict | None:
    """OpenAlex's best title-search match, or None. Callers must still
    confirm the returned title is the same paper -- search returns the
    closest match, not an exact one."""
    if not title:
        return None
    try:
        response = requests.get(OPENALEX_WORKS_URL, params={"search": title, "per_page": 1},
                                timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    results = response.json().get("results") or []
    return _normalize(results[0]) if results else None
