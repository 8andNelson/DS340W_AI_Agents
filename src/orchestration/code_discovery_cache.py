"""
Code-repository caching for the Code Discovery Agent.

One cache, a plain JSON file under logs/ (gitignored, like
dataset_search_cache.json): a lookup from normalized parent-paper title to
the repository that was found for it, so a retry of this stage (Master
Agent's own retry-once, or a future re-run request) doesn't re-search.

Keyed by paper title rather than by URL (dataset_cache.py's key) because
the lookup direction here is "given this paper, what repo did we already
find" -- the paper is the stable input, not a dataset name.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
CODE_DISCOVERY_CACHE = REPO_ROOT / "logs" / "code_discovery_cache.json"


def normalize_paper_title(title: str) -> str:
    """Lowercase and strip whitespace/punctuation, same normalization as
    dataset_cache.normalize_dataset_name, so minor phrasing differences
    don't cause a cache miss."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _load_cache() -> dict:
    if CODE_DISCOVERY_CACHE.exists():
        try:
            return json.loads(CODE_DISCOVERY_CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def get_cached_repo(paper_title: str) -> dict | None:
    key = normalize_paper_title(paper_title)
    if not key:
        return None
    return _load_cache().get(key)


def cache_repo(paper_title: str, repository: dict) -> None:
    """Only ever called with a verified, reachable FOUND result -- a
    failed search is never cached, so a retry gets a genuine re-search
    rather than a stale negative."""
    key = normalize_paper_title(paper_title)
    if not key:
        return
    cache = _load_cache()
    cache[key] = repository
    CODE_DISCOVERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CODE_DISCOVERY_CACHE.write_text(json.dumps(cache, indent=2))
