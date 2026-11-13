import json
import re
from concurrent.futures import ThreadPoolExecutor

from src import llm
from src.search import openalex, semantic_scholar, web_search
from src.orchestration.handoff import request as handoff_request

MIN_YEAR = 2022
RESULTS_PER_QUERY = 10

# LLM extraction dominates this stage's runtime (measured live: ~589s of a
# ~599s run across 4 queries, Brave itself ~1.5s/query), so extraction calls
# run concurrently. Kept small because the free model rate-limits (429s);
# llm.chat retries those with backoff.
LLM_WORKERS = 3

# Search-result titles often carry the hosting site's name, e.g.
# "... - ScienceDirect" or "... | IEEE Conference Publication | IEEE Xplore".
# Left in, the same paper found via two queries escapes title de-duplication.
_TITLE_SITE_SUFFIX = re.compile(
    r"\s+[-|]\s+(ScienceDirect|ResearchGate|arXiv|IEEE Xplore|Springer(?: Nature Link)?|"
    r"MDPI|Nature|Semantic Scholar|ACM Digital Library)\s*$",
    re.IGNORECASE,
)

_RETRACTION_MARKER = re.compile(r"^\s*(retracted|retraction(?: notice)?)\b|\bthis article has been retracted\b",
                                re.IGNORECASE)

_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>?#]+")


def clean_title(title: str) -> str:
    """Strip HTML tags and trailing site-name suffixes from a search-result
    title. Only " | " segments and known site names are removed, so a title
    with an ordinary hyphen is left intact."""
    title = re.sub(r"<[^>]+>", "", title or "").strip()
    if " | " in title:
        title = title.split(" | ")[0].strip()
    previous = None
    while previous != title:
        previous = title
        title = _TITLE_SITE_SUFFIX.sub("", title).strip()
    return title


def is_retracted(title: str, description: str = "") -> bool:
    """True if the title or snippet marks the paper as retracted. A
    retracted paper is not valid peer-reviewed literature."""
    return bool(_RETRACTION_MARKER.search(title or "") or _RETRACTION_MARKER.search(description or ""))


def _doi_from_text(*texts: str) -> str:
    """Extract a DOI that literally appears in the given text, never one
    the LLM produced -- a DOI is a citation, so it must be traceable."""
    for text in texts:
        match = _DOI_PATTERN.search(text or "")
        if match:
            return match.group(0).rstrip(".,;)")
    return ""


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _extract_papers_from_results(results: list, topic: str, query: str) -> list:
    """Use the LLM to classify Brave search results and extract the fields a
    snippet can't give directly.

    The LLM returns only per-result classification fields keyed by result
    number. Title, URL, and abstract are copied verbatim from the Brave
    result itself, and the DOI comes from a regex over the URL/snippet --
    the LLM never rewrites an identifier. This also keeps the response
    short, which is what dominates a call's latency."""
    if not results:
        return []

    snippets = []
    for i, r in enumerate(results):
        snippets.append(
            f"[{i + 1}] Title: {clean_title(r.get('title', ''))}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    Description: {r.get('description', '')}"
        )

    prompt = f"""You are classifying web search results for an academic literature search.

Research topic: "{topic}"
Search query used: "{query}"

Search results:
{chr(10).join(snippets)}

Include only results that appear to be a real peer-reviewed academic paper (journal article or conference paper published {MIN_YEAR} or later).

Skip results that are:
- Blog posts, news articles, or tutorials
- Kaggle notebooks or Medium articles
- Non-peer-reviewed preprints with no venue
- Books or textbooks
- Retracted papers
- Papers published before {MIN_YEAR}

For each included result return a JSON object. Use only information visible in the result; never guess:
{{
  "index": <the result number in brackets above>,
  "authors": ["<author names if visible, else empty list>"],
  "year": <publication year as integer, 0 if not visible>,
  "venue": "<journal or conference name, empty string if not visible>",
  "peer_reviewed": <true if published in a named journal or conference>,
  "dataset": "<dataset name if mentioned, else empty string>",
  "dataset_source": "<dataset origin if mentioned, else empty string>",
  "methodology": "<1 sentence summary of the ML method if described, else empty string>",
  "results_summary": "<key quantitative results if mentioned, else empty string>",
  "has_code": <true if a code repository is mentioned>,
  "code_url": "<code URL only if it appears in the result, else empty string>",
  "parent_paper_score": <integer 1-10, overall suitability as a parent paper for replication>,
  "notes": "<one short observation or concern>"
}}

Return a JSON array. If no results qualify, return [].
Return ONLY valid JSON. No explanation or markdown."""

    content = llm.chat(prompt, timeout=300, agent="research_agent")

    try:
        extracted = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        extracted = json.loads(match.group(0)) if match else []

    papers = []
    seen_indices = set()
    for item in extracted:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(results) or idx in seen_indices:
            continue
        seen_indices.add(idx)
        r = results[idx]
        url = r.get("url", "")
        description = r.get("description", "")

        try:
            year = int(item.get("year") or 0)
        except (TypeError, ValueError):
            year = 0
        try:
            score = int(item.get("parent_paper_score") or 0)
        except (TypeError, ValueError):
            score = 0

        papers.append({
            "title": clean_title(r.get("title", "")),
            "authors": item.get("authors") or [],
            "year": year,
            "venue": item.get("venue") or "",
            "peer_reviewed": bool(item.get("peer_reviewed", False)),
            "doi": _doi_from_text(url, description),
            "paper_url": url,
            "abstract": re.sub(r"<[^>]+>", "", description),
            "dataset": item.get("dataset") or "",
            "dataset_source": item.get("dataset_source") or "",
            "methodology": item.get("methodology") or "",
            "results_summary": item.get("results_summary") or "",
            "has_tables": False,
            "has_figures": False,
            "has_graphs": False,
            "has_code": bool(item.get("has_code", False)),
            "code_url": item.get("code_url") or "",
            "citation_count": 0,
            "parent_paper_score": score,
            "notes": item.get("notes") or "",
            "source": "brave_search",
            "verified": False,
        })

    return papers


def _lookup_metadata(paper: dict) -> tuple:
    """(record, source name) for this paper from OpenAlex (by DOI, else by
    title), falling back to Semantic Scholar's title match. A record is
    returned only when its title is this paper's title (after
    normalization) -- title search returns the closest match, which may be
    a different paper. (None, "") when nothing matches."""
    title_key = _title_key(paper.get("title", ""))
    candidates = []
    if paper.get("doi"):
        candidates.append((lambda: openalex.work_by_doi(paper["doi"]), "OpenAlex"))
    candidates.append((lambda: openalex.match_title(paper.get("title", "")), "OpenAlex"))
    candidates.append((lambda: semantic_scholar.match_title(paper.get("title", "")), "Semantic Scholar"))
    for lookup, source in candidates:
        record = lookup()
        if record and _title_key(record.get("title", "")) == title_key:
            return record, source
    return None, ""


def _fill_missing_year(paper: dict) -> None:
    """For a paper whose year couldn't be read from its search snippet, look
    it up by DOI/title (see _lookup_metadata). Best-effort: on no match or
    an API error the year stays 0 (unknown)."""
    match, source = _lookup_metadata(paper)
    if not match:
        return
    if match.get("year"):
        paper["year"] = int(match["year"])
        paper["notes"] = (paper.get("notes", "") + f" Year confirmed via {source} title match.").strip()
    if not paper.get("venue") and match.get("venue"):
        paper["venue"] = match["venue"]
    if not paper.get("doi"):
        paper["doi"] = match.get("doi") or (match.get("externalIds") or {}).get("DOI", "") or ""


def run(structured_request: dict) -> list:
    """Search academic literature via web search and return scored paper candidates."""

    topic = structured_request.get("domain", "")
    project_topic = structured_request.get("project_topic", topic)
    search_queries = structured_request.get("search_queries", [])

    if not search_queries:
        search_queries = structured_request.get("keywords", [])[:3]

    print(f"\n[Research Agent] Searching for papers on: {project_topic}")
    print(f"[Research Agent] Running {len(search_queries)} queries via web search...")

    # Brave searches stay sequential (fast, and Brave's plan is rate-limited
    # per second); each query's LLM extraction starts as soon as its search
    # returns, and the extractions run concurrently.
    futures = []
    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        for query in search_queries:
            print(f"  > '{query}'")
            try:
                results = web_search.search_academic_papers(query, count=RESULTS_PER_QUERY)
            except Exception as e:
                print(f"  WARNING: Search failed for '{query}' - {e}")
                continue
            futures.append((query, llm.submit_with_context(pool, _extract_papers_from_results,
                                                         results, project_topic, query)))

        # Collect in query order so de-duplication is deterministic.
        extracted = []
        for query, future in futures:
            try:
                extracted.append(future.result())
            except Exception as e:
                print(f"  WARNING: Extraction failed for '{query}' - {e}")

    all_papers = []
    seen_urls = set()
    seen_titles = set()

    for papers in extracted:
        for p in papers:
            if is_retracted(p.get("title", ""), p.get("abstract", "")):
                print(f"  SKIPPED (retracted): {p['title'][:70]}")
                continue

            url = p.get("paper_url", "")
            title_key = _title_key(p.get("title", ""))

            if url and url in seen_urls:
                continue
            if title_key and title_key in seen_titles:
                continue

            if url:
                seen_urls.add(url)
            if title_key:
                seen_titles.add(title_key)

            all_papers.append(p)

    missing_year = [p for p in all_papers if not p.get("year")]
    if missing_year:
        print(f"[Research Agent] Looking up {len(missing_year)} missing publication year(s) on Semantic Scholar...")
        for p in missing_year:
            _fill_missing_year(p)

    # Filter to 2022+. A year still unknown (0) is kept -- the Master Agent
    # doesn't count it as qualifying, and Validation flags it.
    all_papers = [p for p in all_papers if not p.get("year") or p["year"] >= MIN_YEAR]

    print(f"\n[Research Agent] {len(all_papers)} unique candidate papers found (>= {MIN_YEAR}).")

    if not all_papers:
        print("[Research Agent] ERROR: No papers found. Try different search terms.")
        return []

    # Sort by parent paper score
    all_papers.sort(key=lambda p: p.get("parent_paper_score", 0), reverse=True)

    return all_papers


def handoff(result: list, context: dict) -> list:
    """The Research Agent's request to Master Agent when it finishes: run
    the Validation Agent on the candidate papers it found."""
    return [handoff_request("research_agent", "Validation Agent", {"papers": result},
                            f"{len(result or [])} candidate papers found; they need independent validation "
                            "and a Parent Paper recommendation.")]
