import json
import re
import time
import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL

# Domains that are prohibited as academic paper sources
PROHIBITED_DOMAINS = [
    "medium.com",
    "towardsdatascience.com",
    "kaggle.com",
    "github.com",
    "twitter.com",
    "linkedin.com",
    "reddit.com",
    "facebook.com",
    "youtube.com",
    "blog.",
    "wordpress.com",
    "substack.com",
    "towards-datascience",
    "analyticsvidhya.com",
    "kdnuggets.com",
    "hackernoon.com",
]

# Domains that are credible academic sources
CREDIBLE_DOMAINS = [
    "doi.org",
    "arxiv.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "springer.com",
    "sciencedirect.com",
    "semanticscholar.org",
    "nature.com",
    "mdpi.com",
    "wiley.com",
    "tandfonline.com",
    "aaai.org",
    "neurips.cc",
    "proceedings.mlr.press",
    "aclanthology.org",
    "researchgate.net",
    "pubmed.ncbi.nlm.nih.gov",
]

MIN_YEAR = 2022
URL_TIMEOUT = 10
ASSESSMENT_BATCH = 4


def _check_prohibited_source(url: str) -> str:
    """Return rejection reason if URL is from a prohibited domain, else empty string."""
    url_lower = url.lower()
    for domain in PROHIBITED_DOMAINS:
        if domain in url_lower:
            return f"Prohibited source: {domain}"
    return ""


def _check_url_reachable(url: str) -> tuple[bool, int]:
    """Return (reachable, status_code) for a URL."""
    if not url:
        return False, 0
    try:
        resp = requests.head(url, timeout=URL_TIMEOUT, allow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
        return resp.status_code < 400, resp.status_code
    except Exception:
        try:
            # Fallback: some servers reject HEAD, try GET with short timeout
            resp = requests.get(url, timeout=URL_TIMEOUT, allow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0 (research-bot)"},
                                stream=True)
            resp.close()
            return resp.status_code < 400, resp.status_code
        except Exception:
            return False, 0


def _is_credible_domain(url: str) -> bool:
    url_lower = url.lower()
    return any(d in url_lower for d in CREDIBLE_DOMAINS)


def _llm_validate_batch(papers: list) -> list:
    """Ask the LLM to independently validate a batch of papers."""

    entries = []
    for i, p in enumerate(papers):
        entries.append(
            f"[{i + 1}]\n"
            f"  Title: {p.get('title', '')}\n"
            f"  Year: {p.get('year', 0)}\n"
            f"  Venue: {p.get('venue', 'Unknown')}\n"
            f"  URL: {p.get('paper_url', '')}\n"
            f"  Abstract: {(p.get('abstract', '') or '')[:400]}"
        )

    prompt = f"""You are an independent academic paper validator for a machine learning research project.

Evaluate each paper below WITHOUT using the scores from whoever found them.
Your job is to catch non-academic sources, non-peer-reviewed work, and papers too vague to replicate.

Papers to validate:
{chr(10).join(entries)}

For each paper return a JSON object:
{{
  "index": <1-based integer matching the list above>,
  "validated_peer_reviewed": <true only if published in a named journal or named conference proceedings — false for preprints with no venue or blog posts>,
  "venue_credible": <true if the venue is a real, recognizable academic journal or conference>,
  "year_valid": <true if year >= {MIN_YEAR}>,
  "validated_has_dataset": <true if the title/abstract mentions a specific dataset or data source>,
  "validated_has_results": <true if the abstract mentions quantitative results like accuracy, F1, AUC>,
  "validated_reproducibility": <integer 1-5, 5 = very likely reproducible based on detail visible>,
  "validation_notes": "<one sentence: key strength or concern about this paper>",
  "recommendation": "VERIFIED" | "UNVERIFIED" | "REJECTED",
  "rejection_reason": "<if REJECTED, explain why — else empty string>"
}}

Rules:
- VERIFIED: peer-reviewed venue, year >= {MIN_YEAR}, not a prohibited source
- UNVERIFIED: real paper but venue/year unclear or cannot be confirmed from available info
- REJECTED: blog post, non-peer-reviewed, year < {MIN_YEAR}, or prohibited source

Return ONLY a valid JSON array. No explanation or markdown."""

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
        assessments = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", content)
        assessments = json.loads(match.group(0)) if match else []

    for a in assessments:
        idx = a.get("index", 0) - 1
        if 0 <= idx < len(papers):
            papers[idx].update({
                "validated_peer_reviewed": a.get("validated_peer_reviewed", False),
                "venue_credible": a.get("venue_credible", False),
                "year_valid": a.get("year_valid", False),
                "validated_has_dataset": a.get("validated_has_dataset", False),
                "validated_has_results": a.get("validated_has_results", False),
                "validated_reproducibility": a.get("validated_reproducibility", 0),
                "validation_notes": a.get("validation_notes", ""),
                "validation_status": a.get("recommendation", "UNVERIFIED"),
                "rejection_reason": a.get("rejection_reason", ""),
            })

    return papers


def run(papers: list) -> dict:
    """
    Validate a list of research papers through three layers.
    Returns a dict with 'verified', 'unverified', and 'rejected' lists.
    """
    print(f"\n[Validation Agent] Validating {len(papers)} papers...")
    print("[Validation Agent] Layer 1: Checking for prohibited sources...")

    # Layer 1 — prohibited source check (instant, no network)
    for p in papers:
        url = p.get("paper_url", "")
        rejection = _check_prohibited_source(url)
        if rejection:
            p["validation_status"] = "REJECTED"
            p["rejection_reason"] = rejection
            p["url_reachable"] = False
            print(f"  REJECTED: {p['title'][:60]} | {rejection}")
        else:
            p.setdefault("validation_status", "PENDING")
            p.setdefault("rejection_reason", "")

    pending = [p for p in papers if p.get("validation_status") == "PENDING"]

    # Layer 2 — URL reachability check
    print(f"[Validation Agent] Layer 2: Checking URL reachability for {len(pending)} papers...")
    for p in pending:
        url = p.get("paper_url", "")
        reachable, status = _check_url_reachable(url)
        p["url_reachable"] = reachable
        credible = _is_credible_domain(url)

        status_str = f"HTTP {status}" if status else "unreachable"
        domain_str = "credible domain" if credible else "unknown domain"
        flag = "[OK]" if reachable else "[WARN]"
        print(f"  {flag} {status_str} | {domain_str} | {p['title'][:55]}")
        time.sleep(0.3)

    # Layer 3 — LLM independent re-assessment (batched)
    print(f"[Validation Agent] Layer 3: LLM independent assessment...")
    for i in range(0, len(pending), ASSESSMENT_BATCH):
        batch = pending[i: i + ASSESSMENT_BATCH]
        try:
            _llm_validate_batch(batch)
        except Exception as e:
            print(f"  WARNING: LLM assessment failed for batch — {e}")
            for p in batch:
                # Layer 1 already set validation_status="PENDING" via setdefault,
                # so setdefault here would be a no-op — force the fallback status
                # explicitly so papers don't stay stuck at PENDING.
                p["validation_status"] = "UNVERIFIED"
                p.setdefault("validated_peer_reviewed", p.get("peer_reviewed", False))
                p.setdefault("validated_has_dataset", bool(p.get("dataset")))
                p.setdefault("validated_has_results", bool(p.get("results_summary")))
                p.setdefault("validated_reproducibility", 0)
                p.setdefault("validation_notes", "LLM assessment unavailable")
        if i + ASSESSMENT_BATCH < len(pending):
            time.sleep(1)

    # Tally results
    verified = [p for p in papers if p.get("validation_status") == "VERIFIED"]
    unverified = [p for p in papers if p.get("validation_status") == "UNVERIFIED"]
    rejected = [p for p in papers if p.get("validation_status") == "REJECTED"]

    # Sort verified by reproducibility then parent paper score
    verified.sort(
        key=lambda p: (p.get("validated_reproducibility", 0), p.get("parent_paper_score", 0)),
        reverse=True,
    )

    return {
        "verified": verified,
        "unverified": unverified,
        "rejected": rejected,
        "all": papers,
    }
