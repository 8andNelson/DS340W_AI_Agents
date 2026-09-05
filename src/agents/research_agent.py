import json
import re
import time
import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL
from src.search import brave_search

MIN_YEAR = 2022
RESULTS_PER_QUERY = 10


def _extract_papers_from_results(results: list, topic: str, query: str) -> list:
    """Use LLM to extract structured paper records from Brave search results."""

    snippets = []
    for i, r in enumerate(results):
        snippets.append(
            f"[{i + 1}] Title: {r.get('title', '')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    Description: {r.get('description', '')}"
        )

    prompt = f"""You are extracting academic paper metadata from web search results.

Research topic: "{topic}"
Search query used: "{query}"

Search results:
{chr(10).join(snippets)}

For each result that appears to be a real peer-reviewed academic paper (journal article or conference paper published 2022 or later), extract its metadata.

Skip results that are:
- Blog posts, news articles, or tutorials
- Kaggle notebooks or Medium articles
- Non-peer-reviewed preprints with no venue
- Books or textbooks
- Papers published before 2022

For each valid paper, return a JSON object:
{{
  "title": "<full paper title>",
  "authors": ["<author names if visible, else empty list>"],
  "year": <publication year as integer, 0 if unknown>,
  "venue": "<journal or conference name, empty string if unknown>",
  "peer_reviewed": <true if published in a named journal or conference>,
  "doi": "<DOI if visible in URL or description, else empty string>",
  "paper_url": "<best URL for the paper>",
  "abstract": "<abstract or description snippet>",
  "dataset": "<dataset name if mentioned, else empty string>",
  "dataset_source": "<dataset origin if mentioned, else empty string>",
  "methodology": "<1-2 sentence summary of the ML method if described>",
  "results_summary": "<key quantitative results if mentioned, else empty string>",
  "has_code": <true if a code repository is mentioned>,
  "code_url": "<GitHub or code URL if mentioned, else empty string>",
  "parent_paper_score": <integer 1-10, overall suitability as a parent paper for replication>,
  "notes": "<any observations or concerns>"
}}

Return a JSON array of valid papers only. If no valid papers are found, return an empty array [].
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
        papers = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        papers = json.loads(match.group(0)) if match else []

    # Fill in any missing schema fields with defaults
    defaults = {
        "authors": [], "year": 0, "venue": "", "peer_reviewed": False,
        "doi": "", "paper_url": "", "abstract": "", "dataset": "",
        "dataset_source": "", "methodology": "", "results_summary": "",
        "has_tables": False, "has_figures": False, "has_graphs": False,
        "has_code": False, "code_url": "", "citation_count": 0,
        "parent_paper_score": 0, "notes": "", "source": "brave_search",
        "verified": False,
    }
    for p in papers:
        for k, v in defaults.items():
            p.setdefault(k, v)

    return papers


def run(structured_request: dict) -> list:
    """Search academic literature via Brave and return scored paper candidates."""

    topic = structured_request.get("domain", "")
    project_topic = structured_request.get("project_topic", topic)
    search_queries = structured_request.get("search_queries", [])

    if not search_queries:
        search_queries = structured_request.get("keywords", [])[:3]

    print(f"\n[Research Agent] Searching for papers on: {project_topic}")
    print(f"[Research Agent] Running {len(search_queries)} queries via Brave Search...")

    all_papers = []
    seen_urls = set()
    seen_titles = set()

    for query in search_queries:
        print(f"  > '{query}'")
        try:
            results = brave_search.search_academic_papers(query, count=RESULTS_PER_QUERY)
            papers = _extract_papers_from_results(results, project_topic, query)

            for p in papers:
                # Filter to 2022+
                if p.get("year") and p["year"] < MIN_YEAR:
                    continue

                url = p.get("paper_url", "")
                title_key = p.get("title", "").lower().strip()

                if url and url in seen_urls:
                    continue
                if title_key and title_key in seen_titles:
                    continue

                if url:
                    seen_urls.add(url)
                if title_key:
                    seen_titles.add(title_key)

                all_papers.append(p)

            time.sleep(1)

        except Exception as e:
            print(f"  WARNING: Query failed - {e}")

    print(f"\n[Research Agent] {len(all_papers)} unique candidate papers found (>= {MIN_YEAR}).")

    if not all_papers:
        print("[Research Agent] ERROR: No papers found. Try different search terms.")
        return []

    # Sort by parent paper score
    all_papers.sort(key=lambda p: p.get("parent_paper_score", 0), reverse=True)

    return all_papers
