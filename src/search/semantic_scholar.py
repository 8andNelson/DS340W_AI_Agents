import time
import requests

SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = (
    "title,authors,year,venue,externalIds,abstract,"
    "isOpenAccess,openAccessPdf,citationCount,fieldsOfStudy"
)


def search_papers(query: str, limit: int = 10, min_year: int = 2022, retries: int = 3) -> list:
    """Search Semantic Scholar and return raw records for papers published >= min_year.

    Retries with exponential backoff on 429 rate-limit responses.
    """
    params = {
        "query": query,
        "fields": SS_FIELDS,
        "limit": limit,
        "offset": 0,
    }
    for attempt in range(retries):
        response = requests.get(SS_SEARCH_URL, params=params, timeout=15)
        if response.status_code == 429:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
            print(f"  [Semantic Scholar] Rate limited. Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        raise RuntimeError("Semantic Scholar rate limit exceeded after retries.")

    results = []
    for p in response.json().get("data", []):
        if p.get("year") and p["year"] >= min_year:
            results.append(p)
    return results


def normalize_paper(raw: dict) -> dict:
    """Convert a raw Semantic Scholar record to the project paper schema."""
    ext = raw.get("externalIds") or {}
    doi = ext.get("DOI", "")
    arxiv_id = ext.get("ArXiv", "")

    if doi:
        paper_url = f"https://doi.org/{doi}"
    elif arxiv_id:
        paper_url = f"https://arxiv.org/abs/{arxiv_id}"
    else:
        paper_url = ""

    pdf_info = raw.get("openAccessPdf") or {}

    return {
        "title": raw.get("title", ""),
        "authors": [a["name"] for a in raw.get("authors", [])],
        "year": raw.get("year", 0),
        "venue": raw.get("venue", ""),
        "peer_reviewed": bool(raw.get("venue")),
        "doi": doi,
        "paper_url": paper_url,
        "pdf_url": pdf_info.get("url", ""),
        "abstract": raw.get("abstract", "") or "",
        "citation_count": raw.get("citationCount", 0),
        "fields_of_study": raw.get("fieldsOfStudy") or [],
        # Fields populated by LLM assessment
        "dataset": "",
        "dataset_source": "",
        "methodology": "",
        "results_summary": "",
        "has_tables": False,
        "has_figures": False,
        "has_graphs": False,
        "has_code": False,
        "code_url": "",
        "parent_paper_score": 0,
        "notes": "",
        "source": "semantic_scholar",
        "verified": False,
    }
