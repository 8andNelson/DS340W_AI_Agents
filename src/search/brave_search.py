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
