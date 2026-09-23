"""
Data Agent (Milestone 6).

Builds a dataset pool of at least TARGET_ENTRIES total entries to support
the Parent Paper's replication, escalating through three stages:

  1. Parent Paper's own stated dataset.
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

Collection does NOT stop the moment the entry-count target is reached --
it keeps adding further qualifying, non-duplicate datasets (across all
three stages) up to MAX_DATASETS, so a pool of several smaller datasets is
preferred over stopping at the first single dataset big enough to meet the
target alone. It still stops early once MAX_DATASETS is reached, or once
a stage's candidates are exhausted, whichever comes first.

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

A candidate is also only ever counted once it has actually been
downloaded and converted to a local CSV file (data/raw/) -- a dataset that
merely resolves to a real, in-scope, reachable page (e.g. most Kaggle
pages, which gate real access behind a reCAPTCHA challenge) but can't
actually be fetched is rejected the same way an unreachable one is. This
is what lets the downstream Cleaning Agent assume every dataset it's
handed is a real local file, with no downloading of its own to do.
"""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL, KAGGLE_USERNAME, KAGGLE_KEY
from src.search import brave_search
from src.orchestration import dataset_cache
from src.agents.validation_agent import _check_url_reachable

TARGET_ENTRIES = 10000
MIN_ENTRIES_TO_CONSIDER = 1  # any real, non-empty download counts -- the
                             # 10,000+ target is enforced on the merged
                             # master CSV by the Cleaning Agent, not per
                             # individual dataset here
MAX_DATASETS = 5
RESULTS_PER_QUERY = 25
PAGE_FETCH_TIMEOUT = 15
PAGE_TEXT_CHAR_LIMIT = 6000
DOWNLOAD_TIMEOUT = 30
KAGGLE_DOWNLOAD_TIMEOUT = 90  # Kaggle dataset zips are often large (tens of MB)

REPO_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

# A realistic browser UA -- some dataset hosts (e.g. Kaggle) return 403s to
# an identifying bot User-Agent even for pages that are genuinely public.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Kaggle's own API endpoint, counterintuitively, must NOT use a
# browser-spoofing UA -- verified live that a Chrome-impersonating UA on
# non-browser traffic trips Kaggle's reCAPTCHA gate even with valid
# credentials, while identifying honestly as the API client does not.
KAGGLE_USER_AGENT = "kaggle-api/1.6.17"

HUGGINGFACE_DATASET_RE = re.compile(r"huggingface\.co/datasets/([^/?#]+(?:/[^/?#]+)?)")
KAGGLE_DATASET_RE = re.compile(r"kaggle\.com/datasets/([^/?#]+)/([^/?#]+)")

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
    "entry_count_source": "",  # how entry_count was determined -- see _resolve_entry_count
    "usability_score": None,
    "verified": False,
    "verification_method": "",
    "local_csv_path": "",   # set once downloaded and converted -- see _download_and_convert_to_csv
    "download_status": "",  # "" until attempted; "ok" or "failed: <reason>" after
    "reproducibility_fit": None,  # None until checked -- see _check_reproducibility_fit
    "reproducibility_notes": "",
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
    value -- if neither finds a stated count, entry_count stays None.

    Also records entry_count_source, so the console log and saved reports
    can say exactly how each dataset's count was determined -- this varies
    per candidate, since one dataset's count might already be in the search
    snippet while another's only turns up on Hugging Face or the page
    itself."""
    if isinstance(candidate.get("entry_count"), int):
        if not candidate.get("entry_count_source"):
            candidate["entry_count_source"] = "search result"
        return
    source_url = candidate.get("source_url", "")
    count = _lookup_huggingface_entry_count(source_url)
    if count is not None:
        candidate["entry_count"] = count
        candidate["entry_count_source"] = "Hugging Face API"
        return
    page_text = _fetch_page_text(source_url)
    count = _extract_entry_count_from_page(page_text, dataset_name or candidate.get("name", ""))
    if count is not None:
        candidate["entry_count"] = count
        candidate["entry_count_source"] = "the dataset's own page"


def _slugify(candidate: dict) -> str:
    """Filesystem-safe, stable filename for a candidate's downloaded CSV --
    a short slug of its display name plus a hash of its resolved link, so
    two differently-named candidates never collide and re-downloading the
    same link produces the same filename."""
    base = candidate.get("display_name") or candidate.get("name") or "dataset"
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:60]
    link_hash = hashlib.sha1((candidate.get("name") or "").encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{link_hash}" if slug else link_hash


def _parse_tabular_content(resp: "requests.Response", url: str) -> "pd.DataFrame | None":
    """Sniff a downloaded response and try to read it as tabular data.
    Never raises -- returns None if nothing works. Rejects immediately on
    an HTML content type: a landing page (the single most common failure
    mode observed against real dataset hosts) can otherwise "succeed" as a
    one-column garbage CSV if read blindly."""
    content_type = resp.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        return None

    lower_url = url.lower()
    content = resp.content

    if "zip" in content_type or lower_url.endswith(".zip"):
        return _parse_zip_for_csv(content)

    if "json" in content_type or lower_url.endswith((".json", ".jsonl")):
        try:
            return pd.read_json(io.BytesIO(content), lines=lower_url.endswith(".jsonl"))
        except Exception:
            pass

    if lower_url.endswith(".parquet") or "parquet" in content_type:
        try:
            return pd.read_parquet(io.BytesIO(content))
        except Exception:
            pass

    # Default / fallback: most direct-download dataset links are CSV/TSV
    # even when the server doesn't set an accurate Content-Type.
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
        if df.shape[1] > 0 and df.shape[0] > 0:
            return df
    except Exception:
        pass

    return None


def _parse_zip_for_csv(content: bytes) -> "pd.DataFrame | None":
    """Extract the largest CSV member from a downloaded zip (Kaggle-style
    packaging) and read it. Never raises."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None
            largest = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
            with zf.open(largest) as f:
                return pd.read_csv(f)
    except Exception:
        return None


def _download_huggingface_csv(candidate: dict) -> "pd.DataFrame | None":
    """A Hugging Face dataset's own page is HTML (the generic path would
    reject it) -- instead resolve a real Parquet file via HF's public
    datasets-server API and read that directly. Never raises."""
    match = HUGGINGFACE_DATASET_RE.search(candidate.get("source_url") or candidate.get("name", ""))
    if not match:
        return None
    dataset_id = match.group(1).strip("/")

    try:
        resp = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": dataset_id},
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": BROWSER_USER_AGENT},
        )
        resp.raise_for_status()
        parquet_files = resp.json().get("parquet_files", [])
        if not parquet_files:
            return None
        file_url = parquet_files[0].get("url", "")
        if not file_url:
            return None
        file_resp = requests.get(file_url, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": BROWSER_USER_AGENT})
        file_resp.raise_for_status()
        return pd.read_parquet(io.BytesIO(file_resp.content))
    except Exception:
        return None


def _download_kaggle_dataset(candidate: dict) -> "pd.DataFrame | None":
    """Kaggle blocks plain unauthenticated access behind a reCAPTCHA
    challenge (confirmed repeatedly this session) -- when KAGGLE_USERNAME/
    KAGGLE_KEY are configured, use the classic, long-stable REST endpoint
    instead. Tries HTTP Basic Auth (the scheme that endpoint has used for
    years) first, then falls back to a Bearer token (the newer per-account
    API-token format) on a 401/403, since either could be what a given
    token expects. Critically, this must use a User-Agent that identifies
    as the Kaggle API client rather than the browser-spoofing
    BROWSER_USER_AGENT used elsewhere -- verified live that a
    Chrome-impersonating UA on non-browser traffic trips the same
    reCAPTCHA gate even with valid credentials, while identifying honestly
    as the API client does not. Never raises; returns None on any failure
    (including no credentials configured, or a non-Kaggle URL), so the
    generic download path still applies."""
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        return None
    match = KAGGLE_DATASET_RE.search(candidate.get("source_url") or candidate.get("name", ""))
    if not match:
        return None
    owner, slug = match.group(1), match.group(2)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"

    resp = None
    try:
        resp = requests.get(
            url, timeout=KAGGLE_DOWNLOAD_TIMEOUT, headers={"User-Agent": KAGGLE_USER_AGENT},
            auth=(KAGGLE_USERNAME, KAGGLE_KEY),
        )
        if resp.status_code in (401, 403):
            resp = requests.get(
                url, timeout=KAGGLE_DOWNLOAD_TIMEOUT,
                headers={"User-Agent": KAGGLE_USER_AGENT, "Authorization": f"Bearer {KAGGLE_KEY}"},
            )
        resp.raise_for_status()
    except Exception:
        return None

    return _parse_tabular_content(resp, url)


def _search_kaggle_datasets(query: str, count: int = RESULTS_PER_QUERY) -> list:
    """Query Kaggle's own dataset search API directly (when credentials are
    configured) instead of depending on Brave having indexed a matching
    Kaggle page. Verified live: GET /api/v1/datasets/list with the same
    auth as downloading returns real, structured results (title, url,
    subtitle, license, etc.) -- more reliable and far less run-to-run
    variable than a general web search's ranking. Returns results shaped
    like Brave's (title/url/description) so they flow through the existing
    _extract_dataset_candidates_from_results pipeline unchanged. Never
    raises; returns [] on any failure or when credentials aren't
    configured, so this is a pure addition on top of the existing Brave
    search, never a replacement."""
    if not KAGGLE_USERNAME or not KAGGLE_KEY or not query:
        return []
    try:
        resp = requests.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": query, "sortBy": "hottest", "page": 1},
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": KAGGLE_USER_AGENT},
            auth=(KAGGLE_USERNAME, KAGGLE_KEY),
        )
        resp.raise_for_status()
        datasets = resp.json()
    except Exception:
        return []

    if not isinstance(datasets, list):
        return []

    results = []
    for d in datasets[:count]:
        ref = d.get("ref", "")
        if not ref:
            continue
        results.append({
            "title": d.get("title", ""),
            "url": f"https://www.kaggle.com/datasets/{ref}",
            "description": d.get("subtitle") or d.get("description", ""),
        })
    return results


def _download_and_convert_to_csv(candidate: dict) -> bool:
    """Download the candidate's resolved link and convert it to a local
    CSV under data/raw/. Tries the Hugging Face special case first (its
    dataset pages are HTML and would otherwise fail outright), then the
    Kaggle special case (if credentials are configured), then a plain GET
    with content-sniffing. Never raises. On success, sets local_csv_path
    and derives entry_count directly from the real file (ground truth --
    overrides any earlier guess). On failure, sets download_status to a
    short reason and prints why, so a rejected dataset's cause is visible
    in the console log the same way an added one's provenance is."""
    url = candidate.get("source_url") or candidate.get("name", "")
    label = candidate.get("display_name") or candidate.get("name", "")

    df = _download_huggingface_csv(candidate)
    if df is None:
        df = _download_kaggle_dataset(candidate)
    if df is None:
        if not url:
            candidate["download_status"] = "failed: no source URL"
            return False
        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": BROWSER_USER_AGENT})
            resp.raise_for_status()
        except Exception as e:
            candidate["download_status"] = f"failed: could not fetch URL ({e})"
            print(f"[Data Agent] Could not download '{label}' from {url}: {e}")
            return False
        df = _parse_tabular_content(resp, url)

    if df is None or df.empty or df.shape[1] == 0:
        candidate["download_status"] = (
            "failed: content is not parseable tabular data "
            "(likely an HTML landing page, a zip with no CSV inside, or an unsupported format)"
        )
        print(f"[Data Agent] '{label}' from {url} did not resolve to usable tabular data; skipped.")
        return False

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / f"{_slugify(candidate)}.csv"
    df.to_csv(path, index=False)

    candidate["local_csv_path"] = str(path)
    candidate["download_status"] = "ok"
    candidate["entry_count"] = len(df)
    candidate["entry_count_source"] = "downloaded file"
    return True


_OPAQUE_COLUMN_RE = re.compile(r"^(v|col|column|feature|feat|f|x|pc)_?\d+$", re.IGNORECASE)


def _check_reproducibility_fit(candidate: dict, parent_paper: dict) -> tuple[bool, str]:
    """Check whether the actual downloaded dataset's columns are readable
    enough to be usable -- deterministic, not an LLM judgment call. An
    earlier LLM-based version of this check (comparing columns against the
    Parent Paper's methodology text) proved non-deterministic on borderline
    cases -- the same dataset against the same paper got opposite verdicts
    on different runs, which is unacceptable for a hard accept/reject gate.
    Anonymized/opaque columns (PCA components, generic V1/col_3/feature_7
    names) make a dataset unusable regardless of topical fit, since the
    actual features can't be understood or verified -- so this rejects
    unconditionally when a majority of columns match that pattern, with no
    exception for what the paper's methodology says.

    `parent_paper` is accepted but unused -- kept so this stays a drop-in
    replacement for the call sites in _resolve_dataset and
    _search_exploration_sites. Fails open (True) only on a technical
    failure (can't re-read the file), never fabricating a rejection it
    can't support."""
    try:
        sample = pd.read_csv(candidate["local_csv_path"], nrows=5)
    except Exception as e:
        return True, f"not checked -- could not re-read downloaded file ({e})"

    columns = list(sample.columns)
    opaque = [c for c in columns if _OPAQUE_COLUMN_RE.match(str(c).strip())]
    if columns and len(opaque) / len(columns) >= 0.5:
        return False, (
            f"{len(opaque)}/{len(columns)} columns are anonymized/opaque "
            f"(e.g. {', '.join(str(c) for c in opaque[:5])}) with no stated meaning -- "
            "rejected regardless of topical fit, since the actual features "
            "can't be verified or understood."
        )
    return True, ""


_VERIFICATION_METHOD_DESCRIPTIONS = {
    "cache_hit": "reused from an earlier search",
    "brave_search": "found via Brave search for the paper's stated dataset name",
    "exploration_site": "found via a curated dataset-directory search",
}

_ENTRY_COUNT_SOURCE_DESCRIPTIONS = {
    "search result": "count stated in the search result",
    "Hugging Face API": "count read from Hugging Face's dataset API",
    "the dataset's own page": "count found by reading the dataset's own page",
}


def _describe_provenance(candidate: dict) -> str:
    """Human-readable account of how THIS candidate, specifically, was
    located and how its entry count was determined. Varies per candidate --
    a cache hit, a fresh Brave search, or a curated exploration search each
    describe differently, and the entry count itself might come from the
    search snippet, Hugging Face's API, or the dataset's own page. Also
    surfaces the reproducibility-fit explanation when one was actually
    checked (mainly relevant for a rejected candidate, so its reason
    doesn't silently disappear into "no usable dataset found")."""
    parts = [
        _VERIFICATION_METHOD_DESCRIPTIONS.get(candidate.get("verification_method", ""), ""),
        _ENTRY_COUNT_SOURCE_DESCRIPTIONS.get(candidate.get("entry_count_source", ""), ""),
    ]
    notes = candidate.get("reproducibility_notes", "")
    if notes and not notes.startswith("not checked"):
        parts.append(f"reproducibility fit: {notes}")
    return "; ".join(p for p in parts if p)


def _resolve_dataset(dataset_name: str, dataset_source: str, paper_title: str,
                      scope_description: str, query_builder, parent_paper: dict = None,
                      top_k: int = 1) -> dict:
    """
    Search, extract, verify, and scope-check candidates for a dataset name,
    checking the cache first. Only a reachable, in-scope, downloadable
    candidate that also passes the reproducibility-fit check is cached or
    returned as usable -- failures are never cached, so a later retry
    (Stage 3, with a different query_builder) genuinely re-searches instead
    of replaying a cached miss.
    """
    cached = dataset_cache.get_cached_dataset(dataset_name)
    if cached is not None:
        served = dict(cached)
        served["verification_method"] = "cache_hit"
        return served

    query = query_builder(dataset_name, dataset_source)
    try:
        results = brave_search.search(query, count=RESULTS_PER_QUERY)
    except Exception:
        results = []
    results = results + _search_kaggle_datasets(query)
    if not results:
        return _default_candidate(display_name=dataset_name, notes="Dataset search failed: no results from any source.")

    try:
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
        if _download_and_convert_to_csv(c):
            fit, explanation = _check_reproducibility_fit(c, parent_paper)
            c["reproducibility_fit"] = fit
            c["reproducibility_notes"] = explanation
            if fit:
                dataset_cache.cache_dataset(dataset_name, c)
                return c
            label = c.get("display_name") or c.get("name", "")
            print(f"[Data Agent] '{label}' downloaded but rejected: {explanation}")
            best_in_scope = c
            continue
        # Reachable and in scope, but couldn't actually be downloaded --
        # still try to report an approximate entry count for transparency,
        # but keep it uncached (so a later run, or a different paper naming
        # the same dataset, gets a genuine retry) and unusable (_usable()
        # requires local_csv_path, which download failure left empty).
        _resolve_entry_count(c, dataset_name)
        best_in_scope = c

    if best_in_scope is not None:
        return best_in_scope

    return _default_candidate(
        display_name=dataset_name,
        notes="No verified, in-scope dataset candidate found.",
    )


def _find_and_verify_dataset_for_paper(paper: dict, scope_description: str, parent_paper: dict = None) -> dict | None:
    """Find and verify the dataset a specific paper claims to use. Returns
    None if the paper states no dataset name at all."""
    dataset_name = (paper.get("dataset") or "").strip()
    if not dataset_name:
        return None
    dataset_source = (paper.get("dataset_source") or "").strip()
    return _resolve_dataset(
        dataset_name, dataset_source, paper.get("title", ""), scope_description,
        query_builder=_build_query, parent_paper=parent_paper, top_k=3,
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


def _search_exploration_sites(scope_description: str, parent_paper: dict = None) -> list:
    """Stage 3: last-resort generic search against curated dataset
    directories, scored on usability since there's no paper to match."""
    if not scope_description:
        return []
    try:
        results = brave_search.search_dataset_sites(scope_description, count=RESULTS_PER_QUERY)
    except Exception:
        results = []
    results = results + _search_kaggle_datasets(scope_description)
    if not results:
        return []

    try:
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
            if _download_and_convert_to_csv(c):
                fit, explanation = _check_reproducibility_fit(c, parent_paper)
                c["reproducibility_fit"] = fit
                c["reproducibility_notes"] = explanation
                if not fit:
                    print(f"[Data Agent] '{c['display_name']}' downloaded but rejected: {explanation}")
            else:
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
        and candidate.get("local_csv_path")
        and candidate.get("reproducibility_fit", True)
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
        provenance = _describe_provenance(candidate)
        detail = f" ({provenance})" if provenance else ""
        print(f"[Data Agent] Added '{candidate.get('display_name') or candidate['name']}' "
              f"({candidate['entry_count']} entries) from {candidate['name']}{detail}; "
              f"running total {total_entries}.")
        return True

    # --- Stage 1: Parent Paper's own dataset ---
    parent_candidate = None
    if parent_paper:
        parent_candidate = _find_and_verify_dataset_for_paper(parent_paper, scope_description, parent_paper)
        datasets_considered += 1

    if parent_candidate is None:
        warnings.append("Parent Paper does not state a dataset.")
    elif not try_add(parent_candidate):
        warnings.append(
            f"Parent Paper's stated dataset ('{stated_dataset}') could not be verified, "
            f"was not in scope, or has fewer than {MIN_ENTRIES_TO_CONSIDER} entries."
        )

    # --- Stage 2: full ranked candidate pool, no cap on how many papers are
    # checked for candidacy -- keeps going even after the entry-count target
    # is already met, stopping only once MAX_DATASETS unique datasets have
    # been collected (or the pool is exhausted), so more than one dataset
    # gets combined whenever more than one genuinely qualifies. ---
    for paper in ranked_pool[1:]:
        if len(selected_datasets) >= MAX_DATASETS:
            break
        datasets_considered += 1
        try_add(_find_and_verify_dataset_for_paper(paper, scope_description, parent_paper))

    if selected_datasets:
        search_stage = "paper_traversal"

    # --- Stage 3: nothing usable in any paper's own named dataset -- search
    # the web generally for datasets that fit the project's scope, instead
    # of retrying the specific (already-failed) dataset names again. ---
    if not selected_datasets:
        search_stage = "exploration_sites"
        print("[Data Agent] No usable dataset found among candidate papers; "
              "searching generally for datasets relevant to the project scope...")
        for candidate in _search_exploration_sites(scope_description, parent_paper):
            if len(selected_datasets) >= MAX_DATASETS:
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
