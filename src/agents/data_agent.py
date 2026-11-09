"""
Data Agent (Milestone 6).

Builds the dataset pool for the Parent Paper's replication. It reads the
Parent Paper itself -- the open-access full text when one exists
(src/search/paper_fulltext.py), else its stored abstract and metadata --
to find the exact dataset the paper used: its name, its source, the links
the paper gives, and its stated size (plan_data_needs). Two modes:

  parent_only (the first run): collect ONLY the Parent Paper's own dataset
     -- each link written in the paper, then a search for the paper's
     dataset name. There is no entry minimum; the plan's suggested size is
     recorded as advice for the Evaluation Agent. Only when the paper's own
     dataset cannot be obtained at all does the agent search other sources
     (stages 2-3 below) for a SUBSTITUTE, logged as such (status DEGRADED).

  expand (only on an Evaluation Agent request, relayed and run by Master
     Agent): keep the datasets already collected and add datasets from
     sources away from the Parent Paper until the request's goal
     (target_entries) is met or MAX_DATASETS is reached:
       2. the ranked candidate papers' own datasets (composite-score order,
          starting at the second-highest paper);
       3. curated dataset directories (Google Dataset Search, Kaggle,
          awesome-public-datasets, Hugging Face, UCI) for anything relevant
          to the project's scope, scored on usability.

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

from src import llm
from src.config import KAGGLE_USERNAME, KAGGLE_KEY
from src.search import page_fetch, paper_fulltext, web_search
from src.orchestration import dataset_cache
from src.orchestration.dataset_budget import (
    MAX_MASTER_ROWS, MAX_ROWS_PER_DATASET, check_target_fit, find_label_column, read_csv_capped, stratified_cap,
)
from src.agents.validation_agent import _check_url_reachable
from src.orchestration.handoff import request as handoff_request

MIN_ENTRIES_TO_CONSIDER = 1  # any real, non-empty download counts
# The agent's own entry target (plan_data_needs) is capped at the local row
# budget: the Cleaning Agent samples the master CSV down to MAX_MASTER_ROWS,
# so a larger target could never be met.
MAX_TARGET_ENTRIES = MAX_MASTER_ROWS
MAX_DATASETS = 5
RESULTS_PER_QUERY = 20  # the search maximum (web_search.MAX_COUNT, Brave's limit)
PAGE_FETCH_TIMEOUT = 15
PAGE_TEXT_CHAR_LIMIT = 6000
DOWNLOAD_TIMEOUT = 30
KAGGLE_DOWNLOAD_TIMEOUT = 90  # Kaggle dataset zips are often large (tens of MB)
# A download larger than this is skipped before it is read: even capped to
# MAX_ROWS_PER_DATASET rows, a multi-gigabyte file can stall a local run.
MAX_DOWNLOAD_BYTES = 3 * 1024 ** 3

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
    "reproducibility_fit": None,  # None until checked -- see _check_fit
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


_PAPER_FIELDS_FOR_PLAN = ("title", "venue", "year", "abstract", "methodology", "dataset", "dataset_source",
                          "results_summary", "notes")


def _paper_text(parent_paper: dict) -> str:
    return "\n".join(f"{k}: {parent_paper.get(k)}" for k in _PAPER_FIELDS_FOR_PLAN if parent_paper.get(k))


PAPER_EXCERPT_CHARS = 10000
_URL_RE = re.compile(r"https?://[^\s)\]>,;\"']+")


def _stated_in(number: int, text: str) -> bool:
    """Is this number written in the paper's own text (e.g. 284807,
    284,807 or 284 807)? Used to drop a dataset size the LLM did not read
    from the paper."""
    digits = str(number)
    return any(digits == re.sub(r"[,\s.]", "", m) for m in re.findall(r"\d[\d,\s.]*\d|\d", text))


def _positive_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _urls_in(text: str) -> list:
    return [u.rstrip(".") for u in _URL_RE.findall(text or "")]


def plan_data_needs(parent_paper: dict, target_terms: list = None, paper: dict = None) -> dict:
    """
    Read the Parent Paper itself (its open-access full text when one exists
    -- src/search/paper_fulltext.py -- else its stored abstract and
    metadata) and return what the Data Agent needs to collect its dataset:
    the dataset's name, source and the links the paper gives, its stated
    size, the model the paper used, and how many entries the replication
    needs. One LLM call; the code then checks the answer:

      - a dataset link is kept only if it is written in the paper's text;
      - paper_dataset_size is kept only if that number is written in the
        paper's text (never a guess -- No Hallucination Policy);
      - target_entries must be a positive integer, capped at
        MAX_TARGET_ENTRIES (the local row budget).

    target_entries is advice for the Evaluation Agent: the first Data Agent
    run collects only the Parent Paper's own dataset, whatever its size.
    """
    plan = {"target_entries": None, "paper_dataset_size": None, "paper_model": "", "paper_dataset": "",
            "dataset_source": "", "dataset_urls": [], "reasoning": "", "capped": False, "source": "none",
            "paper_text_status": (paper or {}).get("status", "")}
    if not parent_paper:
        plan["reasoning"] = "No Parent Paper, so no entry target was set."
        return plan

    paper_text = (paper or {}).get("text") or _paper_text(parent_paper)
    text = paper_fulltext.excerpt(paper_text, paper_fulltext.DATA_KEYWORDS + ("model", "classifier"),
                                  max_chars=PAPER_EXCERPT_CHARS)
    full_text_note = ("This is text from the paper's own PDF." if (paper or {}).get("status") == "FULL_TEXT"
                      else "Only the paper's abstract and metadata are available, not the full paper.")
    target_line = f"Prediction target: {', '.join(target_terms)}\n" if target_terms else ""
    prompt = f"""You are the Data Agent of a research pipeline that must replicate a Parent Paper's experiment and
reach results as close to (or better than) the paper's as possible. Read the paper below and find the exact
dataset it used, and decide how many dataset entries (rows) the replication needs.

Parent Paper ({full_text_note}):
{text}
{target_line}
Reason from the paper itself:
- Which dataset did it use, from which source, and does the text give a link (URL) to it? Copy links exactly.
- How many rows does the paper state? If it states a size, replicating at that scale is the strongest choice.
- Which model(s) did it use? Deep learning models usually need far more rows than tree ensembles, linear models or
  SVMs.
- How was it evaluated (splits, cross-validation, metrics)? For an imbalanced target, the minority class needs
  enough examples in every split.

Return ONE JSON object:
{{
  "paper_dataset": "<the dataset's name exactly as the paper gives it, else empty string>",
  "dataset_source": "<where the paper says it comes from (e.g. Kaggle, UCI, a company), else empty string>",
  "dataset_urls": ["<every link to the dataset written in the text above, copied exactly; [] if none>"],
  "paper_model": "<model(s) the paper used, as stated, else empty string>",
  "paper_dataset_size": <integer number of rows ONLY if the paper text above states it, else null -- never guess>,
  "target_entries": <integer: the number of rows this replication should have>,
  "reasoning": "<2-4 sentences: why this number is enough to reproduce the paper's results>"
}}
Return ONLY valid JSON. No explanation or markdown."""

    try:
        content = llm.chat(prompt, timeout=300, agent="data_agent")
        try:
            answer = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            answer = json.loads(match.group(0)) if match else {}
        if not isinstance(answer, dict):
            answer = {}
    except Exception as e:
        plan["reasoning"] = f"Could not read the paper for its dataset ({e})."
        return plan

    plan["source"] = "llm"
    plan["paper_model"] = str(answer.get("paper_model") or "")
    plan["paper_dataset"] = str(answer.get("paper_dataset") or "")
    plan["dataset_source"] = str(answer.get("dataset_source") or "")
    plan["reasoning"] = str(answer.get("reasoning") or "")

    written = set(_urls_in(paper_text))
    urls = answer.get("dataset_urls") if isinstance(answer.get("dataset_urls"), list) else []
    kept = [str(u).rstrip(".") for u in urls if str(u).rstrip(".") in written]
    if len(kept) < len(urls):
        plan["reasoning"] += f" ({len(urls) - len(kept)} dataset link(s) not written in the paper were dropped.)"
    plan["dataset_urls"] = list(dict.fromkeys(kept))

    size = _positive_int(answer.get("paper_dataset_size"))
    if size is not None and not _stated_in(size, paper_text):
        plan["reasoning"] += f" (A paper dataset size of {size} was given but is not in the paper's text; dropped.)"
        size = None
    plan["paper_dataset_size"] = size

    target = _positive_int(answer.get("target_entries"))
    if target is None:
        plan["reasoning"] += " (No valid entry target was returned.)"
    elif target > MAX_TARGET_ENTRIES:
        plan["reasoning"] += (f" (Target {target} is above the local row budget; capped at "
                              f"{MAX_TARGET_ENTRIES}.)")
        target = MAX_TARGET_ENTRIES
        plan["capped"] = True
    plan["target_entries"] = target
    return plan


def _extract_dataset_candidates_from_results(
    results: list, dataset_name: str, paper_title: str, scope_description: str, target_terms: list = None
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

    target_line = (
        f"Prediction target: {', '.join(target_terms)} -- a dataset is in scope only if it labels this outcome "
        "(a dataset on a related topic that predicts something else, e.g. loan default for a fraud project, "
        "is NOT in scope).\n" if target_terms else ""
    )

    prompt = f"""You are extracting dataset metadata from web search results.

Paper title: "{paper_title}"
{target_desc}
Project scope (what the dataset must be relevant to): "{scope_description}"
{target_line}
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

    content = llm.chat(prompt, timeout=300, agent="data_agent")

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
        resp = page_fetch.get(url, timeout=PAGE_FETCH_TIMEOUT, headers={"User-Agent": BROWSER_USER_AGENT})
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
        content = llm.chat(prompt, timeout=300, agent="data_agent")
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


def _parse_tabular_content(resp: "requests.Response", url: str, target_terms: list = None) -> "pd.DataFrame | None":
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
        return _parse_zip_for_csv(content, target_terms)

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
    # even when the server doesn't set an accurate Content-Type. Read in
    # chunks under the row budget so a huge file is never fully loaded.
    try:
        df, _info = read_csv_capped(io.BytesIO(content), MAX_ROWS_PER_DATASET, target_terms,
                                    sep=None, engine="python")
        if df.shape[1] > 0 and df.shape[0] > 0:
            df.attrs["sampling"] = _info
            return df
    except Exception:
        pass

    return None


def _parse_zip_for_csv(content: bytes, target_terms: list = None) -> "pd.DataFrame | None":
    """Extract the largest CSV member from a downloaded zip (Kaggle-style
    packaging) and read it in chunks under the row budget. Never raises."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None
            largest = max(csv_names, key=lambda n: zf.getinfo(n).file_size)
            with zf.open(largest) as f:
                df, info = read_csv_capped(f, MAX_ROWS_PER_DATASET, target_terms)
                df.attrs["sampling"] = info
                return df
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


def _download_kaggle_dataset(candidate: dict, target_terms: list = None) -> "pd.DataFrame | None":
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

    return _parse_tabular_content(resp, url, target_terms)


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


def _too_large(resp) -> bool:
    try:
        return int(resp.headers.get("Content-Length", "0")) > MAX_DOWNLOAD_BYTES
    except (TypeError, ValueError):
        return False


def _download_and_convert_to_csv(candidate: dict, target_terms: list = None) -> bool:
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
        df = _download_kaggle_dataset(candidate, target_terms)
    if df is None:
        if not url:
            candidate["download_status"] = "failed: no source URL"
            return False
        try:
            resp = page_fetch.get(url, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": BROWSER_USER_AGENT})
            resp.raise_for_status()
        except Exception as e:
            candidate["download_status"] = f"failed: could not fetch URL ({e})"
            print(f"[Data Agent] Could not download '{label}' from {url}: {e}")
            return False
        if _too_large(resp):
            candidate["download_status"] = f"failed: larger than the {MAX_DOWNLOAD_BYTES // 1024 ** 3} GB download ceiling"
            print(f"[Data Agent] '{label}' from {url} exceeds the download ceiling; skipped.")
            return False
        df = _parse_tabular_content(resp, url, target_terms)

    if df is None or df.empty or df.shape[1] == 0:
        candidate["download_status"] = (
            "failed: content is not parseable tabular data "
            "(likely an HTML landing page, a zip with no CSV inside, or an unsupported format)"
        )
        print(f"[Data Agent] '{label}' from {url} did not resolve to usable tabular data; skipped.")
        return False

    # CSV/ZIP were already capped while reading; JSON/Parquet arrive whole.
    sampling = df.attrs.get("sampling")
    if sampling is None:
        df, sampling = stratified_cap(df, MAX_ROWS_PER_DATASET, find_label_column(df.head(5000), target_terms))
    if sampling["sampled"]:
        print(f"[Data Agent] '{label}': kept a stratified sample of {sampling['sampled_rows']} of "
              f"{sampling['original_rows']} rows (row budget {MAX_ROWS_PER_DATASET}).")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / f"{_slugify(candidate)}.csv"
    df.to_csv(path, index=False)

    candidate["local_csv_path"] = str(path)
    candidate["download_status"] = "ok"
    candidate["entry_count"] = len(df)
    candidate["entry_count_source"] = "downloaded file"
    candidate["original_entry_count"] = sampling["original_rows"]
    candidate["sampled"] = sampling["sampled"]
    return True


_OPAQUE_COLUMN_RE = re.compile(r"^(v|col|column|feature|feat|f|x|pc)_?\d+$", re.IGNORECASE)


def _describe_columns(candidate: dict) -> str:
    """A note (never a rejection) when columns have no descriptive names,
    e.g. PCA components V1..V28. Such datasets are accepted -- many papers,
    including the ULB credit-card fraud papers, use exactly these -- but the
    report should say the features cannot be interpreted."""
    try:
        columns = list(pd.read_csv(candidate["local_csv_path"], nrows=5).columns)
    except Exception:
        return ""
    opaque = [str(c) for c in columns if _OPAQUE_COLUMN_RE.match(str(c).strip())]
    if not opaque:
        return ""
    return (f"{len(opaque)}/{len(columns)} columns have no descriptive name (e.g. {', '.join(opaque[:5])}), "
            "such as anonymized or PCA features; accepted, but these features cannot be interpreted.")


def _dataset_text(candidate: dict) -> str:
    return " ".join(str(candidate.get(k) or "") for k in ("display_name", "name", "source_url", "notes",
                                                          "scope_explanation"))


def _check_fit(candidate: dict, parent_paper: dict, ctx: dict = None) -> tuple[bool, str]:
    """The deterministic fit check on a downloaded dataset: a label for the
    project's prediction target (dataset_budget.check_target_fit). Column
    names are not checked -- anonymized/PCA columns are accepted, with a
    note (`column_notes`)."""
    candidate["column_notes"] = _describe_columns(candidate)
    target_fit, target_explanation = check_target_fit(
        candidate["local_csv_path"], (ctx or {}).get("target_terms"), _dataset_text(candidate))
    candidate["target_fit_notes"] = target_explanation
    return target_fit, "" if target_fit else target_explanation


def _already_rejected(candidate: dict, ctx: dict = None) -> bool:
    """A link already downloaded and rejected in this run is not
    downloaded again (in a live run the same ~150 MB anonymized dataset
    was fetched and rejected under six different names)."""
    key = dataset_cache.normalize_dataset_link(candidate.get("source_url") or candidate.get("name", ""))
    rejected = (ctx or {}).get("rejected_links")
    if key and rejected is not None and key in rejected:
        label = candidate.get("display_name") or candidate.get("name", "")
        print(f"[Data Agent] '{label}' was already rejected this run; not downloading again.")
        return True
    return False


def _remember_rejection(candidate: dict, ctx: dict = None) -> None:
    key = dataset_cache.normalize_dataset_link(candidate.get("source_url") or candidate.get("name", ""))
    if key and ctx is not None and ctx.get("rejected_links") is not None:
        ctx["rejected_links"].add(key)


_VERIFICATION_METHOD_DESCRIPTIONS = {
    "cache_hit": "reused from an earlier search",
    "brave_search": "found via a web search (Brave or SerpAPI) for the paper's stated dataset name",
    "exploration_site": "found via a curated dataset-directory search",
    "parent_paper_text": "linked in the Parent Paper's own text",
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
    if candidate.get("column_notes"):
        parts.append(candidate["column_notes"])
    return "; ".join(p for p in parts if p)


def _resolve_dataset(dataset_name: str, dataset_source: str, paper_title: str,
                      scope_description: str, query_builder, parent_paper: dict = None,
                      top_k: int = 1, ctx: dict = None) -> dict:
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
        results = web_search.search(query, count=RESULTS_PER_QUERY)
    except Exception:
        results = []
    results = results + _search_kaggle_datasets(query)
    if not results:
        return _default_candidate(display_name=dataset_name, notes="Dataset search failed: no results from any source.")

    try:
        candidates = _extract_dataset_candidates_from_results(
            results, dataset_name, paper_title, scope_description, (ctx or {}).get("target_terms")
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
        if _already_rejected(c, ctx):
            continue
        if _download_and_convert_to_csv(c, (ctx or {}).get("target_terms")):
            fit, explanation = _check_fit(c, parent_paper, ctx)
            c["reproducibility_fit"] = fit
            c["reproducibility_notes"] = explanation
            if fit:
                dataset_cache.cache_dataset(dataset_name, c)
                return c
            label = c.get("display_name") or c.get("name", "")
            print(f"[Data Agent] '{label}' downloaded but rejected: {explanation}")
            _remember_rejection(c, ctx)
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


def _find_and_verify_dataset_for_paper(paper: dict, scope_description: str, parent_paper: dict = None,
                                       ctx: dict = None) -> dict | None:
    """Find and verify the dataset a specific paper claims to use. Returns
    None if the paper states no dataset name at all."""
    dataset_name = (paper.get("dataset") or "").strip()
    if not dataset_name:
        return None
    dataset_source = (paper.get("dataset_source") or "").strip()
    return _resolve_dataset(
        dataset_name, dataset_source, paper.get("title", ""), scope_description,
        query_builder=_build_query, parent_paper=parent_paper, top_k=3, ctx=ctx,
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


def _search_exploration_sites(scope_description: str, parent_paper: dict = None,
                              search_query: str = "", ctx: dict = None) -> list:
    """Stage 3: last-resort generic search against curated dataset
    directories, scored on usability since there's no paper to match.

    search_query (normally the user's project topic) is what's actually
    sent to Brave/Kaggle; scope_description is only used for the in-scope
    check. Verified live: sending the full scope description (paper title
    + 400 chars of abstract) as the query returns zero Kaggle results and
    only the paper's own pages from Brave, while a short topic query
    returns relevant datasets. Falls back to the scope description when no
    short query is given."""
    if not scope_description:
        return []
    query = search_query.strip() or scope_description
    try:
        results = web_search.search_dataset_sites(query, count=RESULTS_PER_QUERY)
    except Exception:
        results = []
    results = results + _search_kaggle_datasets(query)
    if not results:
        return []

    try:
        candidates = _extract_dataset_candidates_from_results(results, "", "", scope_description,
                                                              (ctx or {}).get("target_terms"))
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
        if c.get("in_scope") and not _already_rejected(c, ctx):
            if _download_and_convert_to_csv(c, (ctx or {}).get("target_terms")):
                fit, explanation = _check_fit(c, parent_paper, ctx)
                c["reproducibility_fit"] = fit
                c["reproducibility_notes"] = explanation
                if not fit:
                    print(f"[Data Agent] '{c['display_name']}' downloaded but rejected: {explanation}")
                    _remember_rejection(c, ctx)
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


def _dataset_from_paper_link(url: str, plan: dict, parent_paper: dict, ctx: dict) -> dict | None:
    """Try one dataset link written in the Parent Paper itself. The paper
    names it, so it is in scope by definition; it must still be reachable,
    download as real data and pass the target-fit check."""
    candidate = _default_candidate(
        name=url, source_url=url, display_name=plan.get("paper_dataset") or url,
        matches_parent_paper=True, match_explanation="The link is written in the Parent Paper's own text.",
        in_scope=True, scope_explanation="Named by the Parent Paper itself.",
        source=plan.get("dataset_source", ""), verification_method="parent_paper_text",
    )
    reachable, _status = _check_url_reachable(url)
    candidate["verified"] = reachable
    if not reachable or _already_rejected(candidate, ctx):
        return None
    if not _download_and_convert_to_csv(candidate, ctx.get("target_terms")):
        print(f"[Data Agent] The paper's link {url} did not download as tabular data.")
        return None
    fit, explanation = _check_fit(candidate, parent_paper, ctx)
    candidate["reproducibility_fit"] = fit
    candidate["reproducibility_notes"] = explanation
    if not fit:
        print(f"[Data Agent] The paper's link {url} downloaded but was rejected: {explanation}")
        _remember_rejection(candidate, ctx)
        return None
    return candidate


def _find_parent_dataset(parent_paper: dict, plan: dict, scope_description: str, ctx: dict) -> list:
    """Stage 1: the Parent Paper's own dataset, in this order -- each link
    the paper's text gives, then a search for the name the paper's text
    gives, then a search for the name stored with the paper's search record
    (when it differs). Returns every candidate tried, best last."""
    tried = []
    for url in plan.get("dataset_urls") or []:
        candidate = _dataset_from_paper_link(url, plan, parent_paper, ctx)
        if candidate:
            return tried + [candidate]

    names = []
    for name, source in ((plan.get("paper_dataset"), plan.get("dataset_source")),
                         (parent_paper.get("dataset"), parent_paper.get("dataset_source"))):
        name = (name or "").strip()
        if name and name.lower() not in [n.lower() for n, _s in names]:
            names.append((name, (source or "").strip()))
    for name, source in names:
        tried.append(_resolve_dataset(name, source, parent_paper.get("title", ""), scope_description,
                                      query_builder=_build_query, parent_paper=parent_paper, top_k=3, ctx=ctx))
        if _usable(tried[-1], MIN_ENTRIES_TO_CONSIDER):
            break
    return tried


def run(parent_paper: dict, ranked_pool: list, target_entries: int | None = None,
        search_query: str = "", target_terms: list = None, mode: str = "parent_only",
        existing_datasets: list = None) -> dict:
    """
    Build the dataset pool for the Parent Paper's replication.

    mode="parent_only" (the first run): read the Parent Paper itself
    (plan_data_needs) and collect only the Parent Paper's own dataset.
    Other papers' datasets and general dataset directories are searched
    only as a logged substitute when the paper's own dataset cannot be
    obtained at all. The plan's target_entries is recorded as advice; the
    Evaluation Agent decides later whether more data is needed.

    mode="expand" (only when Master Agent relays an Evaluation Agent
    request): keep existing_datasets and add datasets from sources away
    from the Parent Paper -- the ranked candidate papers' datasets, then
    curated dataset directories -- until target_entries (the goal set by
    the request) is met or MAX_DATASETS is reached.

    search_query is the short query for the directory search (normally the
    user's project topic). target_terms (the Intake Agent's
    prediction-target keywords) drive the target-fit check every accepted
    dataset must pass, cache hits included.
    """
    if mode not in ("parent_only", "expand"):
        raise ValueError(f"Unknown Data Agent mode: {mode}")
    if mode == "expand" and target_entries is None:
        raise ValueError("mode='expand' needs a target_entries goal.")

    paper = paper_fulltext.get_parent_paper_text(parent_paper) if parent_paper else {}
    if mode == "parent_only":
        plan = plan_data_needs(parent_paper, target_terms, paper)
    else:
        plan = {"target_entries": target_entries, "paper_dataset_size": None, "paper_model": "",
                "paper_dataset": "", "dataset_source": "", "dataset_urls": [],
                "reasoning": f"Goal of {target_entries} entries set by an Evaluation Agent request, "
                             "relayed by Master Agent.",
                "capped": False, "source": "request", "paper_text_status": paper.get("status", "")}
    goal = target_entries if mode == "expand" else None
    goal_label = goal if goal is not None else "the Parent Paper's own dataset"
    print(f"\n[Data Agent] Building dataset pool (mode: {mode}, goal: {goal_label})...")
    if paper.get("notes"):
        print(f"[Data Agent] {paper['notes']}")
    if plan["reasoning"]:
        print(f"[Data Agent] Data plan: {plan['reasoning']}")

    scope_description = _scope_description(parent_paper)
    selected_datasets = []
    seen_links = set()
    total_entries = 0
    datasets_considered = 0
    warnings = []
    search_stage = "parent_paper"
    parent_dataset_found = False
    ctx = {"target_terms": target_terms or [], "rejected_links": set()}

    def try_add(candidate: dict | None) -> bool:
        nonlocal total_entries
        if not _usable(candidate, MIN_ENTRIES_TO_CONSIDER):
            return False
        link_key = dataset_cache.normalize_dataset_link(candidate["name"])
        if not link_key or link_key in seen_links or link_key in ctx["rejected_links"]:
            return False
        # Applies to every candidate, cache hits included: a cached result
        # predates this check and may predict something else.
        target_fit, explanation = check_target_fit(candidate["local_csv_path"], ctx["target_terms"],
                                                   _dataset_text(candidate))
        if not target_fit:
            label = candidate.get("display_name") or candidate["name"]
            print(f"[Data Agent] '{label}' rejected: {explanation}")
            warnings.append(f"Rejected '{label}': {explanation}")
            ctx["rejected_links"].add(link_key)
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

    def keep_searching() -> bool:
        if len(selected_datasets) >= MAX_DATASETS:
            return False
        return total_entries < goal if goal is not None else not selected_datasets

    if mode == "parent_only":
        # --- Stage 1: the Parent Paper's own dataset, read from the paper ---
        datasets_considered += 1
        for candidate in _find_parent_dataset(parent_paper, plan, scope_description, ctx) if parent_paper else []:
            if try_add(candidate):
                parent_dataset_found = True
                break
        if not parent_dataset_found:
            stated = plan.get("paper_dataset") or (parent_paper or {}).get("dataset") or ""
            warnings.append(
                f"The Parent Paper's own dataset ('{stated or 'not stated'}') could not be found, downloaded, "
                "or did not fit the target; searching other sources for a SUBSTITUTE (logged, not the paper's "
                "own data)."
            )
    else:
        for candidate in existing_datasets or []:
            if try_add(candidate):
                parent_dataset_found = parent_dataset_found or bool(candidate.get("matches_parent_paper"))
        search_stage = "expansion"
        print(f"[Data Agent] Kept {len(selected_datasets)} existing dataset(s) ({total_entries} entries); "
              f"collecting more from sources away from the Parent Paper to reach {goal}...")

    # --- Stage 2: other ranked candidate papers' datasets (expansion, or a
    # substitute when the Parent Paper's own dataset was not obtained) ---
    if keep_searching():
        search_stage = "expansion" if mode == "expand" else "substitute_paper_traversal"
        for other in (ranked_pool or [])[1:]:
            if not keep_searching():
                break
            datasets_considered += 1
            try_add(_find_and_verify_dataset_for_paper(other, scope_description, parent_paper, ctx))

    # --- Stage 3: curated dataset directories for anything in scope ---
    if keep_searching():
        search_stage = (search_stage + "+exploration_sites") if search_stage != "parent_paper" else "exploration_sites"
        print(f"[Data Agent] Searching dataset directories for datasets relevant to the project "
              f"({total_entries} entries so far, goal {goal_label})...")
        for candidate in _search_exploration_sites(scope_description, parent_paper, search_query, ctx):
            if not keep_searching():
                break
            try_add(candidate)

    target_met = None if goal is None else total_entries >= goal
    if not selected_datasets:
        status = "NOT_FOUND"
    elif mode == "parent_only" and not parent_dataset_found:
        status = "DEGRADED"
    elif target_met is False:
        status = "DEGRADED"
        warnings.append(
            f"Stopped after {len(selected_datasets)} unique dataset(s) with {total_entries} total "
            f"entries -- below the requested goal of {goal}, even after all sources were searched."
        )
    else:
        status = "OK"

    print(f"[Data Agent] Done. Status={status}, total_entries={total_entries}, goal={goal_label}, "
          f"datasets_used={len(selected_datasets)}, stage={search_stage}.")

    return {
        "status": status,
        "mode": mode,
        "target_entries": goal,
        "data_plan": plan,
        "parent_dataset_found": parent_dataset_found,
        "paper_text_status": paper.get("status", ""),
        "paper_text_notes": paper.get("notes", ""),
        "parent_paper_pdf_path": paper.get("pdf_path", ""),
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


def handoff(result: dict, context: dict) -> list:
    """The Data Agent's request to Master Agent when it finishes: run the
    Cleaning Agent on the datasets it downloaded."""
    datasets = (result or {}).get("selected_datasets") or []
    return [handoff_request("data_agent", "Cleaning Agent",
                            {"selected_datasets": datasets, "target_rows": (result or {}).get("target_entries")},
                            f"{len(datasets)} dataset(s) downloaded ({(result or {}).get('total_entries', 0)} "
                            "entries); they need merging into the master CSV.")]
