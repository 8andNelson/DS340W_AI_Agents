"""
Code Discovery Agent (Milestone 9).

Given the approved Parent Paper, tries to locate the implementation that
the (future) Replication Agent will reproduce, in CLAUDE.md's preferred
order:

  1. The paper's own stated code_url, if any -- still verified, never
     trusted outright (CLAUDE.md: "Do not assume a repository implements
     the paper simply because it has a similar name").
  2. A GitHub / code-index search built from the paper's title, each
     result LLM-verified against the paper's own scope.
  3. Kaggle "code" (notebooks) and other credible implementations, folded
     into the same search rather than a separate authenticated path --
     unlike Data Agent's datasets, downloading isn't this agent's job, so
     there's no need for Kaggle API credentials here.

Unlike Data Agent/Cleaning Agent, finding nothing is an explicitly valid
outcome: CLAUDE.md's Preferred Implementation Order's last resort is to
"reconstruct the implementation from the paper only if usable code cannot
be located." This agent never blocks the pipeline -- it reports what it
found (or didn't) for the Replication Agent to act on.

Every verified+reachable GitHub candidate gets its license/stars/
dependency files pulled from GitHub's public REST API (no token
required); non-GitHub matches (Kaggle, GitLab, Papers with Code) are
recorded with whatever the search result itself provided, since this
agent's job is identification, not deep repo inspection.
"""
import json
import re

import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL
from src.search import brave_search
from src.orchestration import code_discovery_cache
from src.agents.validation_agent import _check_url_reachable

RESULTS_PER_QUERY = 10
GITHUB_API_TIMEOUT = 15

DEPENDENCY_FILENAMES = {"requirements.txt", "environment.yml", "setup.py", "pyproject.toml", "Pipfile"}
README_FILENAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}

GITHUB_REPO_RE = re.compile(r"github\.com/([^/?#]+)/([^/?#]+)", re.IGNORECASE)

_REPO_DEFAULTS = {
    "repo_url": "",
    "display_name": "",
    "owner": "",
    "source": "",                  # "paper_stated" | "github_search" | "kaggle" | "other"
    "matches_parent_paper": False,
    "match_explanation": "",
    "verified": False,
    "verification_method": "",     # "llm_match" | "cache_hit"
    "reachable": False,
    "license": "",
    "dependency_files": [],
    "has_setup_instructions": False,
    "stars": None,
    "default_branch": "",
    "notes": "",
}


def _default_repo(**overrides) -> dict:
    repo = dict(_REPO_DEFAULTS)
    repo.update(overrides)
    return repo


def _scope_description(parent_paper: dict) -> str:
    """Same shape as data_agent._scope_description -- every candidate repo
    is checked against this same description."""
    if not parent_paper:
        return ""
    parts = [parent_paper.get("title", "")]
    context = parent_paper.get("abstract") or parent_paper.get("methodology") or ""
    if context:
        parts.append(context[:400])
    return " -- ".join(p for p in parts if p)


def _build_query(paper_title: str) -> str:
    return f'"{paper_title}" implementation code'


def _parse_github_repo(url: str) -> tuple | None:
    """Return (owner, repo) if url is a GitHub repo link, else None."""
    match = GITHUB_REPO_RE.search(url or "")
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    repo = repo.split("/", 1)[0]  # drop any trailing path (e.g. /tree/main)
    return owner, repo[:-4] if repo.lower().endswith(".git") else repo


def _fetch_github_metadata(owner: str, repo: str) -> dict:
    """Best-effort GitHub repo metadata via the public REST API -- no
    token required, 60 req/hr is plenty for one paper's handful of
    candidates. Never raises; returns {} on any failure."""
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers={"Accept": "application/vnd.github+json"},
            timeout=GITHUB_API_TIMEOUT,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception:
        return {}

    return {
        "license": (data.get("license") or {}).get("spdx_id") or "",
        "stars": data.get("stargazers_count"),
        "default_branch": data.get("default_branch", ""),
    }


def _fetch_github_contents_listing(owner: str, repo: str) -> list:
    """Root-directory file listing via GitHub's contents API. Never
    raises; returns [] on any failure (private repo, rate limit, etc.)."""
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents",
            headers={"Accept": "application/vnd.github+json"},
            timeout=GITHUB_API_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        if not isinstance(data, list):
            return []
        return [item.get("name", "") for item in data if isinstance(item, dict)]
    except Exception:
        return []


def _enrich_github_repo(repo_dict: dict, owner: str, repo: str) -> None:
    """Mutates repo_dict in place with GitHub metadata + dependency-file
    detection. has_setup_instructions is a deterministic check (README AND
    at least one known dependency file both present) -- no LLM judgment
    call for this, matching Data Agent's reproducibility-fit check
    precedent of using deterministic Python for a hard accept/reject fact,
    not fuzzy LLM judgment."""
    metadata = _fetch_github_metadata(owner, repo)
    repo_dict["license"] = metadata.get("license", "")
    repo_dict["stars"] = metadata.get("stars")
    repo_dict["default_branch"] = metadata.get("default_branch", "")

    filenames = _fetch_github_contents_listing(owner, repo)
    lower_names = {f.lower() for f in filenames}
    repo_dict["dependency_files"] = sorted(f for f in filenames if f in DEPENDENCY_FILENAMES)
    has_readme = bool(lower_names & README_FILENAMES)
    repo_dict["has_setup_instructions"] = has_readme and bool(repo_dict["dependency_files"])


def _extract_repo_candidates_from_results(results: list, paper_title: str, scope_description: str) -> list:
    """LLM call verifying which search results are plausibly THIS paper's
    own implementation, not just a similarly-named repo -- mirrors
    data_agent._extract_dataset_candidates_from_results's prompt/parsing
    shape exactly."""
    if not results:
        return []

    snippets = []
    for i, r in enumerate(results):
        snippets.append(
            f"[{i + 1}] Title: {r.get('title', '')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    Description: {r.get('description', '')}"
        )

    prompt = f"""You are verifying whether code repository search results actually implement a specific academic paper.

Paper title: "{paper_title}"
Project scope: "{scope_description}"

Search results:
{chr(10).join(snippets)}

For each result that is a real code repository (GitHub, GitLab, Kaggle code/notebook, Papers with Code) -- NOT an article, forum post, or unrelated tool -- assess whether it is genuinely an implementation of the paper above. Do not assume a match just because the name or topic sounds similar; look for concrete signals (paper title/authors mentioned, matching method names, explicit "official implementation of..." language).

For each candidate repository, return a JSON object:
{{
  "repo_url": "<the repository URL>",
  "display_name": "<repo name as it appears>",
  "matches_parent_paper": <true only if there is concrete evidence this implements the specific paper above, false otherwise>,
  "match_explanation": "<1 sentence on the evidence for or against the match>",
  "source": "<github_search, kaggle, or other>",
  "notes": "<any observations or concerns>"
}}

Return a JSON array of candidate repositories, best match first. If none are found, return an empty array [].
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

    return [_default_repo(**c) for c in candidates]


def _verify_single_match(repo_url: str, paper_title: str, scope_description: str) -> dict | None:
    """Used for the paper-stated code_url path, where there's exactly one
    candidate (not a batch of search results) to confirm against the
    paper. Reuses the same extraction prompt/parsing by wrapping the one
    URL as a single-item result list."""
    fake_result = [{"title": repo_url, "url": repo_url, "description": ""}]
    candidates = _extract_repo_candidates_from_results(fake_result, paper_title, scope_description)
    return candidates[0] if candidates else None


def run(parent_paper: dict) -> dict:
    """
    Locate the Parent Paper's implementation, in CLAUDE.md's preferred
    order (author-stated code -> GitHub/code-index search -> Kaggle/other),
    verifying every candidate against the paper's own scope rather than
    trusting a name match. Finding nothing is a valid, non-blocking
    outcome -- Replication Agent falls back to reconstructing from the
    paper itself.
    """
    paper_title = (parent_paper or {}).get("title", "")
    print(f"\n[Code Discovery Agent] Locating implementation for '{paper_title}'...")

    cached = code_discovery_cache.get_cached_repo(paper_title)
    if cached:
        print(f"[Code Discovery Agent] Using cached repository: {cached.get('repo_url', '')}.")
        return {
            "status": "FOUND",
            "repository": {**cached, "verification_method": "cache_hit"},
            "candidates_considered": [cached],
            "source_stage": "cache_hit",
            "warnings": [],
            "notes": "",
        }

    scope_description = _scope_description(parent_paper)
    candidates_considered = []
    warnings = []
    chosen = None
    source_stage = "none"

    # --- Stage 1: author-stated code, if any ---
    stated_url = (parent_paper or {}).get("code_url", "") or ""
    if stated_url:
        candidate = _verify_single_match(stated_url, paper_title, scope_description)
        if candidate:
            candidate["source"] = "paper_stated"
            candidate["verification_method"] = "llm_match"
            reachable, _ = _check_url_reachable(candidate["repo_url"] or stated_url)
            candidate["reachable"] = reachable
            candidates_considered.append(candidate)
            if candidate["matches_parent_paper"] and reachable:
                chosen = candidate
                source_stage = "paper_stated"
        if not chosen:
            warnings.append("Parent Paper's stated code_url could not be verified or reached.")
    else:
        warnings.append("Parent Paper does not state a code_url.")

    # --- Stage 2 & 3: GitHub / code-index search (includes Kaggle code,
    # GitLab, Papers with Code via brave_search.search_code_sites) ---
    if not chosen:
        try:
            results = brave_search.search_code_sites(_build_query(paper_title), count=RESULTS_PER_QUERY)
        except Exception as e:
            results = []
            warnings.append(f"Code search failed: {e}")

        found_candidates = _extract_repo_candidates_from_results(results, paper_title, scope_description)
        for candidate in found_candidates:
            reachable, _ = _check_url_reachable(candidate["repo_url"])
            candidate["reachable"] = reachable
            candidate["verification_method"] = "llm_match"
            candidates_considered.append(candidate)

        verified_matches = [c for c in found_candidates if c["matches_parent_paper"] and c["reachable"]]
        github_matches = [c for c in verified_matches if "github.com" in c["repo_url"].lower()]
        other_matches = [c for c in verified_matches if "github.com" not in c["repo_url"].lower()]

        if github_matches:
            # Enrich every GitHub match with metadata before ranking by
            # stars, so the pick isn't just "first LLM-returned result".
            for c in github_matches:
                parsed = _parse_github_repo(c["repo_url"])
                if parsed:
                    c["owner"] = parsed[0]
                    c["source"] = c["source"] or "github_search"
                    _enrich_github_repo(c, *parsed)
            github_matches.sort(key=lambda c: c.get("stars") or 0, reverse=True)
            chosen = github_matches[0]
            source_stage = "github_search"
        elif other_matches:
            other_matches[0]["source"] = other_matches[0]["source"] or "kaggle"
            chosen = other_matches[0]
            source_stage = "kaggle"

    if chosen:
        print(f"[Code Discovery Agent] Found implementation: {chosen['repo_url']} (via {source_stage}).")
        code_discovery_cache.cache_repo(paper_title, chosen)
        status = "FOUND"
    else:
        print("[Code Discovery Agent] No verified implementation found; "
              "Replication Agent will need to reconstruct from the paper.")
        status = "NOT_FOUND"
        warnings.append("No verified, reachable implementation found across all search stages.")

    return {
        "status": status,
        "repository": chosen,
        "candidates_considered": candidates_considered,
        "source_stage": source_stage,
        "warnings": warnings,
        "notes": "",
    }
