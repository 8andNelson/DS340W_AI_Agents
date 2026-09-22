import requests
from src.config import BRAVE_API_KEY

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Academic site targets for paper searches
ACADEMIC_SITES = (
    "site:arxiv.org OR site:ieeexplore.ieee.org OR site:dl.acm.org "
    "OR site:springer.com OR site:sciencedirect.com OR site:semanticscholar.org "
    "OR site:researchgate.net OR site:mdpi.com OR site:nature.com"
)


def search(query: str, count: int = 10) -> list:
    """Run a Brave web search and return raw result records."""
    if not BRAVE_API_KEY:
        raise EnvironmentError("BRAVE_API_KEY is not set in .env")

    response = requests.get(
        BRAVE_SEARCH_URL,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
        params={"q": query, "count": count},
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("web", {}).get("results", [])


def search_academic_papers(query: str, count: int = 10) -> list:
    """Search for academic papers by appending site filters to the query."""
    academic_query = f"{query} {ACADEMIC_SITES}"
    return search(academic_query, count=count)


# Curated dataset-directory targets for the Data Agent's last-resort
# exploration search (used only when paper-driven dataset search fails).
# Hugging Face and UCI are included alongside CLAUDE.md's three named sites
# because they publish structured, machine-readable size metadata (a
# Hugging Face dataset's row counts are available via its public API; UCI
# pages state "Number of Instances" directly) -- unlike Kaggle, which gates
# automated page access behind a reCAPTCHA challenge, so its entry counts
# usually can't be confirmed without manual Kaggle API credentials.
DATASET_EXPLORATION_SITES = (
    "site:datasetsearch.research.google.com",
    "site:kaggle.com/datasets",
    "site:github.com/awesomedata/awesome-public-datasets",
    "site:huggingface.co/datasets",
    "site:archive.ics.uci.edu",
)


def search_dataset_sites(query: str, count: int = 10) -> list:
    """Search each curated dataset directory separately and merge results.

    A single query combining all sites with OR lets Brave's ranking crowd
    the result page with whichever one site it favors for the topic (in
    practice, Kaggle) -- so each site gets its own query and a fair share
    of the result budget instead of being drowned out.
    """
    per_site_count = max(2, count // len(DATASET_EXPLORATION_SITES))
    seen_urls = set()
    merged = []
    for site_filter in DATASET_EXPLORATION_SITES:
        try:
            results = search(f"{query} {site_filter}", count=per_site_count)
        except Exception:
            continue
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(r)
    return merged
