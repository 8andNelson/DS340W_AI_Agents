"""
Data Agent (Milestone 6).

Builds a dataset pool of at least TARGET_ENTRIES total entries to support
the Parent Paper's replication, escalating through three stages:

  1. Parent Paper's own stated dataset. If it alone reaches the target,
     stop there.
  2. The full ranked candidate pool (composite-score order, starting at the
     second-highest paper), each paper's own stated dataset -- no cap on
     how many papers are checked.
  3. If nothing usable was found in stages 1-2, do NOT retry those same
     (already-failed) dataset names again -- instead search curated dataset
     directories (Google Dataset Search, Kaggle Datasets,
     awesome-public-datasets) generally for anything relevant to the
     project's scope, scoring each candidate on usability (entry count,
     license, format, access method) since there's no single paper claim to
     match against here.

A dataset's identity is its resolved downloadable link (`name`), not any
text name mentioned in a paper -- two papers naming the same dataset
differently must not be double-counted, and a paper's stated name is only
ever used to drive the search itself (`display_name` keeps that text for
human-readable reports). Every candidate is also checked against the
Parent Paper's own scope (title/abstract/methodology) before being
accepted, so a search returning a superficially similar but unrelated
dataset gets rejected rather than silently used.

Never invents a dataset a paper doesn't claim to use, never invents an
entry count that isn't explicitly stated by the source, and never accepts
a dataset without confirming it is both reachable and in scope, per
CLAUDE.md's Source Verification / No Hallucination Policy.
"""
import json
import re
import requests
from bs4 import BeautifulSoup

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL
from src.search import brave_search
from src.orchestration import dataset_cache
from src.agents.validation_agent import _check_url_reachable

TARGET_ENTRIES = 10000
SINGLE_DATASET_THRESHOLD = 10000
MIN_ENTRIES_TO_CONSIDER = 100
RESULTS_PER_QUERY = 10
PAGE_FETCH_TIMEOUT = 15
PAGE_TEXT_CHAR_LIMIT = 6000

# A realistic browser UA -- some dataset hosts (e.g. Kaggle) return 403s to
# an identifying bot User-Agent even for pages that are genuinely public.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

HUGGINGFACE_DATASET_RE = re.compile(r"huggingface\.co/datasets/([^/?#]+(?:/[^/?#]+)?)")

_CANDIDATE_DEFAULTS = {
    "name": "",             # canonical identity: the resolved downloadable link
    "display_name": "",     # human-readable name, for reports only
    "matches_parent_paper": False,
    "match_explanation": "",
    "in_scope": False,
    "scope_explanation": "",
    "source": "",
    "source_url": "",
    "access_method": "",
    "license": "",
    "format": "",
    "entry_count": None,
    "usability_score": None,
    "verified": False,
    "verification_method": "",
    "notes": "",
}


def _default_candidate(**overrides) -> dict:
    candidate = dict(_CANDIDATE_DEFAULTS)
    candidate.update(overrides)
    return candidate


def _scope_description(parent_paper: dict) -> str:
    """The Parent Paper's own scope -- every candidate dataset, at every
    escalation stage, is checked against this same description."""
    if not parent_paper:
        return ""
    parts = [parent_paper.get("title", "")]
    context = parent_paper.get("abstract") or parent_paper.get("methodology") or ""
    if context:
        parts.append(context[:400])
    return " -- ".join(p for p in parts if p)


def _build_query(dataset_name: str, dataset_source: str) -> str:
    parts = [dataset_name]
    if dataset_source:
        parts.append(dataset_source)
    parts.append("dataset")
    return " ".join(parts)


def _extract_dataset_candidates_from_results(
    results: list, dataset_name: str, paper_title: str, scope_description: str
) -> list:
    """Use LLM to extract structured dataset candidates from search results,
    including whether each one fits the project's scope."""

    snippets = []
    for i, r in enumerate(results):
        snippets.append(
            f"[{i + 1}] Title: {r.get('title', '')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    Description: {r.get('description', '')}"
        )

    target_desc = (
        f'Dataset the paper claims to use: "{dataset_name}"' if dataset_name else
        "No specific dataset name is targeted -- find any dataset relevant to the project scope below."
    )

    prompt = f"""You are extracting dataset metadata from web search results.

Paper title: "{paper_title}"
{target_desc}
Project scope (what the dataset must be relevant to): "{scope_description}"

Search results:
{chr(10).join(snippets)}

For each result that appears to be a real, accessible page for a relevant dataset (e.g. Kaggle, UCI ML Repository, HuggingFace, GitHub, Google Dataset Search, an official project site), extract its metadata.

Skip results that are unrelated to the project scope, or that are just articles/tutorials/blog posts about a dataset rather than the dataset's actual source.

For each valid dataset page, return a JSON object:
{{
  "name": "<dataset name as it appears at the source>",
  "matches_parent_paper": <true if this is confirmed to be the exact dataset named above, false if only similar/related or if no name was targeted>,
  "match_explanation": "<1 sentence on how you confirmed the match>",
  "in_scope": <true only if this dataset's subject matter genuinely fits the project scope described above>,
  "scope_explanation": "<1 sentence on why it does or doesn't fit the project scope>",
  "source": "<e.g. Kaggle, UCI ML Repository, HuggingFace, GitHub, official project site>",
  "source_url": "<the actual downloadable link for the dataset itself, not a landing/article page if avoidable>",
  "access_method": "<direct download, API, request/application required, manual registration>",
  "license": "<license as stated, empty string if not stated>",
  "format": "<e.g. CSV, images, COCO-style JSON, empty string if not stated>",
  "entry_count": <integer number of rows/samples/records ONLY if explicitly stated by the source, else null -- never estimate or guess>,
  "notes": "<any observations or concerns>"
}}

Return a JSON array of valid dataset candidates, best match first. If none are found, return an empty array [].
Return ONLY valid JSON. No explanation or markdown."""

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    try:
        candidates = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        candidates = json.loads(match.group(0)) if match else []

    return [_default_candidate(**c) for c in candidates]


def _fetch_page_text(url: str) -> str:
    """Fetch a dataset page's visible text so entry-count extraction has
    more to work with than a one-line search snippet. Never raises --
    returns "" on any failure, so callers can treat it like "nothing found"."""
    if not url:
        return ""
    try:
        resp = requests.get(url, timeout=PAGE_FETCH_TIMEOUT, headers={"User-Agent": BROWSER_USER_AGENT})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
    except Exception:
        return ""
    return text[:PAGE_TEXT_CHAR_LIMIT]


def _extract_entry_count_from_page(page_text: str, dataset_name: str) -> int | None:
    """One LLM call restricted to finding a stated row/sample/record count
    on the dataset's own page -- same no-guessing discipline as
    _extract_dataset_candidates_from_results, just given fuller context
    than a search snippet."""
    if not page_text:
        return None

    prompt = f"""You are reading the content of a dataset's own page to find its size.

Dataset: "{dataset_name}"
Page text:
{page_text}

Find the number of rows/samples/records/instances/examples in this dataset, ONLY if it is explicitly stated somewhere in the text above. Do not estimate, infer, or guess from file size, category counts, or any other indirect signal.

Return ONLY valid JSON: {{"entry_count": <integer if explicitly stated, else null>}}"""

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else json.loads(content)
        count = data.get("entry_count")
        return count if isinstance(count, int) else None
    except Exception:
        return None


def _lookup_huggingface_entry_count(source_url: str) -> int | None:
    """Hugging Face publishes exact split sizes via a public API -- no LLM
    guessing needed when the dataset is hosted there. Falls through to None
    (and the page-text fallback) on any failure or unrecognized shape."""
    if not source_url:
        return None
    match = HUGGINGFACE_DATASET_RE.search(source_url)
    if not match:
        return None
    dataset_id = match.group(1).strip("/")

    try:
        resp = requests.get(
            f"https://huggingface.co/api/datasets/{dataset_id}",
            params={"full": "true"},
            timeout=10,
            headers={"User-Agent": BROWSER_USER_AGENT},
        )
        resp.raise_for_status()
        info = resp.json().get("dataset_info")
    except Exception:
        return None

    if not info:
        return None

    # dataset_info is either a single {"splits": [...]} dict, or a dict of
    # config_name -> {"splits": [...]} for multi-config datasets.
    configs = [info] if isinstance(info, dict) and "splits" in info else list(info.values())
    total = 0
    for config in configs:
        if not isinstance(config, dict):
            continue
        for split in config.get("splits", []) or []:
            if isinstance(split, dict) and isinstance(split.get("num_examples"), int):
                total += split["num_examples"]
    return total if total > 0 else None


def _resolve_entry_count(candidate: dict, dataset_name: str) -> None:
    """Best-effort fill-in for entry_count when the search snippet didn't
    state one: try Hugging Face's structured API first (exact, no LLM),
    then fall back to reading the actual dataset page. Never fabricates a
    value -- if neither finds a stated count, entry_count stays None."""
    if isinstance(candidate.get("entry_count"), int):
        return
    source_url = candidate.get("source_url", "")
    count = _lookup_huggingface_entry_count(source_url)
    if count is None:
        page_text = _fetch_page_text(source_url)
        count = _extract_entry_count_from_page(page_text, dataset_name or candidate.get("name", ""))
    if count is not None:
        candidate["entry_count"] = count


def _resolve_dataset(dataset_name: str, dataset_source: str, paper_title: str,
                      scope_description: str, query_builder, top_k: int = 1) -> dict:
    """
    Search, extract, verify, and scope-check candidates for a dataset name,
    checking the cache first. Only a reachable, in-scope candidate is
    cached or returned as usable -- failures are never cached, so a later
    retry (Stage 3, with a different query_builder) genuinely re-searches
    instead of replaying a cached miss.
    """
    cached = dataset_cache.get_cached_dataset(dataset_name)
    if cached is not None:
        served = dict(cached)
        served["verification_method"] = "cache_hit"
        return served

    try:
        query = query_builder(dataset_name, dataset_source)
        results = brave_search.search(query, count=RESULTS_PER_QUERY)
        candidates = _extract_dataset_candidates_from_results(
            results, dataset_name, paper_title, scope_description
        )
    except Exception as e:
        return _default_candidate(display_name=dataset_name, notes=f"Dataset search/extraction failed: {e}")

    best_in_scope = None
    for c in candidates[:top_k]:
        reachable, _status = _check_url_reachable(c.get("source_url", ""))
        c["verified"] = reachable
        c["verification_method"] = "brave_search"
        c["display_name"] = c.get("name", "") or dataset_name
        if not reachable:
            c["entry_count"] = None
            continue
        c["name"] = c.get("source_url", "")
        if not c.get("in_scope"):
            continue
        _resolve_entry_count(c, dataset_name)
        if isinstance(c.get("entry_count"), int):
            dataset_cache.cache_dataset(dataset_name, c)
            return c
        # Reachable and in scope, but no entry count found anywhere -- keep
        # it as a fallback answer, but deliberately don't cache it, so a
        # later run (or a different paper naming the same dataset) gets a
        # genuine retry instead of replaying a permanent unknown count.
        best_in_scope = c

    if best_in_scope is not None:
        return best_in_scope

    return _default_candidate(
        display_name=dataset_name,
        notes="No verified, in-scope dataset candidate found.",
    )


def _find_and_verify_dataset_for_paper(paper: dict, scope_description: str) -> dict | None:
    """Find and verify the dataset a specific paper claims to use. Returns
    None if the paper states no dataset name at all."""
    dataset_name = (paper.get("dataset") or "").strip()
    if not dataset_name:
        return None
    dataset_source = (paper.get("dataset_source") or "").strip()
    return _resolve_dataset(
        dataset_name, dataset_source, paper.get("title", ""), scope_description,
        query_builder=_build_query, top_k=3,
    )


def _usability_score(candidate: dict) -> float:
    """Deterministic usability score for exploration-site candidates (no
    single paper claim to match against, so grounded fields decide instead
    of an LLM guess). Never fabricates a value not already present."""
    score = 0.0

    entries = candidate.get("entry_count")
    if isinstance(entries, int) and entries > 0:
        score += min(4.0, entries / 5000)

    license_ = (candidate.get("license") or "").lower()
    if any(k in license_ for k in ("cc0", "public domain", "mit", "cc-by", "apache")):
        score += 2.0
    elif license_:
        score += 1.0

    fmt = (candidate.get("format") or "").lower()
    if any(k in fmt for k in ("csv", "json", "parquet", "tsv")):
        score += 2.0
    elif fmt:
        score += 1.0

    access = (candidate.get("access_method") or "").lower()
    if "direct" in access:
        score += 2.0
    elif "api" in access:
        score += 1.0

    return round(score, 2)


def _search_exploration_sites(scope_description: str) -> list:
    """Stage 3: last-resort generic search against curated dataset
    directories, scored on usability since there's no paper to match."""
    if not scope_description:
        return []
    try:
        results = brave_search.search_dataset_sites(scope_description, count=RESULTS_PER_QUERY)
        candidates = _extract_dataset_candidates_from_results(results, "", "", scope_description)
    except Exception:
        return []

    resolved = []
    for c in candidates:
        reachable, _status = _check_url_reachable(c.get("source_url", ""))
        c["verified"] = reachable
        c["verification_method"] = "exploration_site"
        c["display_name"] = c.get("name", "") or c.get("source", "")
        if not reachable:
            c["entry_count"] = None
            continue
        c["name"] = c.get("source_url", "")
        if c.get("in_scope"):
            _resolve_entry_count(c, c.get("display_name", ""))
        c["usability_score"] = _usability_score(c)
        resolved.append(c)

    resolved.sort(key=lambda c: c["usability_score"], reverse=True)
    return resolved


def _usable(candidate: dict | None, min_entries: int) -> bool:
    return bool(
        candidate
        and candidate.get("verified")
        and candidate.get("in_scope")
        and isinstance(candidate.get("entry_count"), int)
        and candidate["entry_count"] >= min_entries
    )


def run(parent_paper: dict, ranked_pool: list) -> dict:
    """
    Build a dataset pool of >= TARGET_ENTRIES total entries for the Parent
    Paper's replication, escalating through the three stages described in
    this module's docstring.
    """
    print(f"\n[Data Agent] Building dataset pool (target: {TARGET_ENTRIES} entries)...")

    scope_description = _scope_description(parent_paper)
    stated_dataset = (parent_paper.get("dataset", "") if parent_paper else "") or ""
    selected_datasets = []
    seen_links = set()
    total_entries = 0
    datasets_considered = 0
    warnings = []
    search_stage = "parent_paper"

    def try_add(candidate: dict | None) -> bool:
        nonlocal total_entries
        if not _usable(candidate, MIN_ENTRIES_TO_CONSIDER):
            return False
        link_key = dataset_cache.normalize_dataset_link(candidate["name"])
        if not link_key or link_key in seen_links:
            return False
        selected_datasets.append(candidate)
        seen_links.add(link_key)
        total_entries += candidate["entry_count"]
        print(f"[Data Agent] Added '{candidate.get('display_name') or candidate['name']}' "
              f"({candidate['entry_count']} entries); running total {total_entries}.")
        return True

    # --- Stage 1: Parent Paper's own dataset ---
    parent_candidate = None
    if parent_paper:
        parent_candidate = _find_and_verify_dataset_for_paper(parent_paper, scope_description)
        datasets_considered += 1

    if _usable(parent_candidate, SINGLE_DATASET_THRESHOLD):
        print(f"[Data Agent] Parent Paper's own dataset alone meets the target "
              f"({parent_candidate['entry_count']} entries). Using it as the sole dataset.")
        return {
            "status": "OK",
            "target_entries": TARGET_ENTRIES,
            "total_entries": parent_candidate["entry_count"],
            "target_met": True,
            "datasets_considered": datasets_considered,
            "selected_datasets": [parent_candidate],
            "master_dataset": parent_candidate,
            "is_combined": False,
            "search_stage": search_stage,
            "warnings": warnings,
            "notes": "Parent Paper's own dataset alone met the target; no combination needed.",
        }

    if parent_candidate is None:
        warnings.append("Parent Paper does not state a dataset.")
    elif not try_add(parent_candidate):
        warnings.append(
            f"Parent Paper's stated dataset ('{stated_dataset}') could not be verified, "
            f"was not in scope, or has fewer than {MIN_ENTRIES_TO_CONSIDER} entries."
        )

    # --- Stage 2: full ranked candidate pool, no cap on how many papers are checked ---
    for paper in ranked_pool[1:]:
        if total_entries >= TARGET_ENTRIES:
            break
        datasets_considered += 1
        try_add(_find_and_verify_dataset_for_paper(paper, scope_description))

    if selected_datasets:
        search_stage = "paper_traversal"

    # --- Stage 3: nothing usable in any paper's own named dataset -- search
    # the web generally for datasets that fit the project's scope, instead
    # of retrying the specific (already-failed) dataset names again. ---
    if not selected_datasets:
        search_stage = "exploration_sites"
        print("[Data Agent] No usable dataset found among candidate papers; "
              "searching generally for datasets relevant to the project scope...")
        for candidate in _search_exploration_sites(scope_description):
            if total_entries >= TARGET_ENTRIES:
                break
            try_add(candidate)

    target_met = total_entries >= TARGET_ENTRIES
    if not selected_datasets:
        status = "NOT_FOUND"
    elif target_met:
        status = "OK"
    else:
        status = "DEGRADED"
        warnings.append(
            f"Stopped after {len(selected_datasets)} unique dataset(s) with {total_entries} total "
            f"entries -- below the {TARGET_ENTRIES}-entry target, even after all escalation stages."
        )

    print(f"[Data Agent] Done. Status={status}, total_entries={total_entries}, "
          f"datasets_used={len(selected_datasets)}, stage={search_stage}.")

    return {
        "status": status,
        "target_entries": TARGET_ENTRIES,
        "total_entries": total_entries,
        "target_met": target_met,
        "datasets_considered": datasets_considered,
        "selected_datasets": selected_datasets,
        "master_dataset": selected_datasets[0] if len(selected_datasets) == 1 else None,
        "is_combined": len(selected_datasets) > 1,
        "search_stage": search_stage,
        "warnings": warnings,
        "notes": "",
    }
