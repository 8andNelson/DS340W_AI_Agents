"""
Dataset caching for the Data Agent.

Two caches, both plain JSON files under logs/ (gitignored, like
project_state.json):

  1. validated_papers_cache.json -- a snapshot of the Validation Agent's
     composite-score-ranked candidate pool, written once per pipeline run so
     the Data Agent can traverse it without re-deriving the ranking.
  2. dataset_search_cache.json -- a lookup from normalized dataset name to
     the last verification result found for it, so re-encountering the same
     dataset name (from a different paper, or a re-run) skips a repeat web
     search.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
VALIDATED_PAPERS_CACHE = REPO_ROOT / "logs" / "validated_papers_cache.json"
DATASET_SEARCH_CACHE = REPO_ROOT / "logs" / "dataset_search_cache.json"


def cache_validated_papers(ranked_pool: list) -> None:
    """Persist the composite-score-ranked candidate pool for this run."""
    VALIDATED_PAPERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    VALIDATED_PAPERS_CACHE.write_text(json.dumps(ranked_pool, indent=2))


def load_validated_papers_cache() -> list:
    if VALIDATED_PAPERS_CACHE.exists():
        try:
            return json.loads(VALIDATED_PAPERS_CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def normalize_dataset_name(name: str) -> str:
    """Lowercase and strip whitespace/punctuation so cache lookups match
    despite minor phrasing differences (e.g. 'CIFAR-10' vs 'cifar 10')."""
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def normalize_dataset_link(url: str) -> str:
    """Canonicalize a dataset's downloadable link for identity/dedup: this
    is the ground truth for 'is this the same dataset', since two papers
    can name the same dataset differently but must link to the same file."""
    if not url:
        return ""
    url = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    url = url.split("?", 1)[0].split("#", 1)[0]
    return url.rstrip("/").lower()


def _load_search_cache() -> dict:
    if DATASET_SEARCH_CACHE.exists():
        try:
            return json.loads(DATASET_SEARCH_CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def get_cached_dataset(name: str) -> dict | None:
    key = normalize_dataset_name(name)
    if not key:
        return None
    return _load_search_cache().get(key)


def cache_dataset(name: str, candidate: dict) -> None:
    key = normalize_dataset_name(name)
    if not key:
        return
    cache = _load_search_cache()
    cache[key] = candidate
    DATASET_SEARCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_SEARCH_CACHE.write_text(json.dumps(cache, indent=2))
