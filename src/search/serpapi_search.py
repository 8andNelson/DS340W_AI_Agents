"""
SerpApi (serpapi.com): Google results as JSON -- the second web-search
provider, used when Brave fails or reaches its monthly limit (see
src/search/web_search.py). Results are mapped to the record shape the
agents read from Brave: {"title", "url", "description"}.
"""
import requests

from src.config import SERPENT_API_KEY

SERPAPI_URL = "https://serpapi.com/search.json"
TIMEOUT_SECONDS = 30
GOOGLE_MAX_NUM = 100


def is_configured() -> bool:
    return bool(SERPENT_API_KEY)


def search(query: str, count: int = 10) -> list:
    """Google results for query through SerpApi; at most count records."""
    if not is_configured():
        raise EnvironmentError("SERPENT_API_KEY is not set in .env")
    response = requests.get(
        SERPAPI_URL,
        params={"engine": "google", "q": query, "num": max(1, min(count, GOOGLE_MAX_NUM)),
                "api_key": SERPENT_API_KEY},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("SerpApi returned an unexpected reply (not a JSON object).")
    if body.get("error"):
        # SerpApi reports some failures (e.g. an exhausted plan) in the body.
        raise RuntimeError(f"SerpApi error: {body['error']}")
    results = []
    for item in body.get("organic_results") or []:
        url = (item or {}).get("link") or ""
        if url:
            results.append({"title": item.get("title") or "", "url": url,
                            "description": item.get("snippet") or ""})
    return results[:count]
