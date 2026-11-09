"""
Brave Search API: the first web-search provider (1,000 requests/month,
reset on the 1st). Agents do not call this module directly; they use
src/search/web_search.py, which moves on to SerpApi when Brave fails.
"""
import requests
from src.config import BRAVE_API_KEY

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Brave's web search API rejects count > 20 with HTTP 422 (verified live:
# count=20 succeeds, count=21 fails). Callers that swallow exceptions would
# otherwise silently get zero results, so search() clamps instead.
BRAVE_MAX_COUNT = 20


def is_configured() -> bool:
    return bool(BRAVE_API_KEY)


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
        params={"q": query, "count": min(count, BRAVE_MAX_COUNT)},
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("web", {}).get("results", [])
