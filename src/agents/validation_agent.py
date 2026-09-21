"""
Paper Validation Agent (Milestone 3) + Parent Paper Selection (Milestone 4).

Validation Agent independently checks the Research Agent's findings through
three layers (prohibited-source check, URL reachability, LLM re-assessment),
then ranks the resulting candidate pool and recommends a single Parent Paper.

Ranking (`rank_and_recommend`) is fully deterministic (plain Python over
fields already gathered by research_agent + this module's own validation
layers) -- no LLM call, no network dependency, and no risk of fabricating a
score or claim not grounded in the paper data. The Master Agent supervises
this agent's output (see master_agent.py); this agent only recommends.
"""
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

# --- Parent Paper selection (moved from paper_selection_agent.py) ---
MIN_QUALIFYING = 5
JUSTIFICATION_TOP_N = 3  # chosen paper + up to 2 runner-ups discussed in prose

COMPUTE_HEAVY_KEYWORDS = [
    "transformer", "bert", "gpt", "diffusion", "gan",
    "reinforcement learning", "large language model",
    "cnn", "resnet", "lstm", "deep learning", "gpu",
]

CLARITY_KEYWORDS = [
    "hyperparameter", "training procedure", "architecture",
    "preprocessing", "feature", "baseline", "evaluation metric",
    "cross-validation", "ablation",
]


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


def _dataset_availability(paper: dict) -> str:
    """Label how clearly the dataset is identified."""
    dataset = (paper.get("dataset") or "").strip()
    source = (paper.get("dataset_source") or "").strip()
    if dataset and source:
        return "Named + sourced"
    if dataset or paper.get("validated_has_dataset"):
        return "Mentioned"
    return "Not specified"


def _clarity_label(text: str, keyword_list: list) -> str:
    """Derive a High/Medium/Low clarity label from keyword density + length."""
    text = (text or "").lower()
    if not text:
        return "Low"
    hits = sum(1 for kw in keyword_list if kw in text)
    if hits >= 3 or len(text) > 300:
        return "High"
    if hits >= 1 or len(text) > 100:
        return "Medium"
    return "Low"


def _compute_estimate(paper: dict) -> str:
    """Estimate compute requirements from methodology text keywords."""
    text = (paper.get("methodology") or "").lower()
    hits = sum(1 for kw in COMPUTE_HEAVY_KEYWORDS if kw in text)
    if hits >= 2:
        return "High"
    if hits == 1:
        return "Medium"
    return "Low"


def _reproducibility_label(paper: dict) -> tuple[str, bool]:
    """
    Return (label, is_estimated).

    is_estimated=True means Layer 3 validation never actually scored this
    paper (validated_reproducibility is 0/missing) and the label below was
    derived from a heuristic fallback -- never presented as if it were a real
    validated assessment.
    """
    score = paper.get("validated_reproducibility", 0) or 0
    if score > 0:
        if score >= 4:
            return "High", False
        if score >= 2:
            return "Medium", False
        return "Low", False

    # Fallback estimate: completeness of dataset/methodology/results/code
    signals = [
        bool((paper.get("dataset") or "").strip()),
        bool((paper.get("methodology") or "").strip()),
        bool((paper.get("results_summary") or "").strip()),
        bool(paper.get("has_code")),
    ]
    hits = sum(signals)
    if hits >= 3:
        return "Medium (estimated)", True
    if hits >= 1:
        return "Low (estimated)", True
    return "Unknown (estimated)", True


def _visuals_summary(paper: dict) -> str:
    parts = []
    if paper.get("has_tables"):
        parts.append("tables")
    if paper.get("has_figures"):
        parts.append("figures")
    if paper.get("has_graphs"):
        parts.append("graphs")
    return ", ".join(parts) if parts else "none reported"


def _expected_difficulty(reproducibility_label: str, compute_estimate: str, impl_clarity: str) -> str:
    """Composite Low/Medium/High difficulty label for reproducing this paper."""
    points = 0
    repro_base = reproducibility_label.split(" ")[0]  # strip "(estimated)" suffix
    points += {"High": 0, "Medium": 1, "Unknown": 2, "Low": 2}.get(repro_base, 1)
    points += {"Low": 0, "Medium": 1, "High": 2}.get(compute_estimate, 1)
    points += {"High": 0, "Medium": 1, "Low": 2}.get(impl_clarity, 1)

    if points <= 1:
        return "Low"
    if points <= 3:
        return "Medium"
    return "High"


def _compute_criteria_row(paper: dict) -> dict:
    """Assemble one full comparison-table row covering the CLAUDE.md criteria."""
    methodology_clarity = _clarity_label(paper.get("methodology"), CLARITY_KEYWORDS)
    implementation_clarity = _clarity_label(
        (paper.get("methodology") or "") + " " + (paper.get("results_summary") or ""),
        CLARITY_KEYWORDS,
    )
    repro_label, repro_estimated = _reproducibility_label(paper)
    compute_estimate = _compute_estimate(paper)
    dataset_availability = _dataset_availability(paper)
    has_evaluation_metrics = bool((paper.get("results_summary") or "").strip()) or bool(
        paper.get("validated_has_results")
    )

    row = {
        "title": paper.get("title", ""),
        "publication_year": paper.get("year", 0),
        "peer_reviewed": bool(paper.get("validated_peer_reviewed", paper.get("peer_reviewed", False))),
        "research_topic": paper.get("venue", "") or "",
        "dataset_availability": dataset_availability,
        "dataset_source": paper.get("dataset_source", "") or "",
        "methodology_clarity": methodology_clarity,
        "implementation_clarity": implementation_clarity,
        "code_availability": bool(paper.get("has_code")),
        "computing_resources_estimate": compute_estimate,
        "reproducibility_difficulty": repro_label,
        "reproducibility_estimated": repro_estimated,
        "visuals": _visuals_summary(paper),
        "has_evaluation_metrics": has_evaluation_metrics,
        "upstream_parent_paper_score": paper.get("parent_paper_score", 0),
    }
    row["expected_difficulty"] = _expected_difficulty(
        repro_label, compute_estimate, implementation_clarity
    )
    row["composite_score"] = _composite_score(row)
    return row


_CLARITY_POINTS = {"Low": 0, "Medium": 1, "High": 2}


def _repro_points(repro_label: str, estimated: bool) -> float:
    base = repro_label.split(" ")[0]
    mapping = {"High": 5, "Medium": 3, "Low": 1, "Unknown": 0}
    raw = mapping.get(base, 0)
    # Real Layer-3 scores are weighted higher than heuristic estimates so
    # validated data always outranks a guess of the same label.
    return raw * (0.6 if not estimated else 0.4)


def _composite_score(row: dict, code_weight: float = 0.5) -> float:
    """
    Deterministic weighted composite score (~0-15 pt scale). Code availability
    is deliberately a small flat bonus so it can never dominate the ranking
    (CLAUDE.md: only one qualifying paper may be chosen on code-availability
    grounds -- this weighting makes that structurally hard to violate).
    """
    score = 0.0
    score += 2.5 if row["peer_reviewed"] else 0.0

    year = row["publication_year"] or 0
    if year >= 2022:
        score += min(1.0, (year - 2021) * 0.25)

    if row["dataset_availability"] == "Named + sourced":
        score += 2.0
    elif row["dataset_availability"] == "Mentioned":
        score += 1.0

    score += _CLARITY_POINTS.get(row["methodology_clarity"], 0)
    score += _CLARITY_POINTS.get(row["implementation_clarity"], 0)
    score += _repro_points(row["reproducibility_difficulty"], row["reproducibility_estimated"])

    visuals_count = 0 if row["visuals"] == "none reported" else len(row["visuals"].split(", "))
    score += min(1.5, visuals_count * 0.5)

    if row["has_evaluation_metrics"]:
        score += 1.0

    score += (row["upstream_parent_paper_score"] or 0) * 0.3

    if row["code_availability"]:
        score += code_weight

    return round(score, 3)


def _select_candidate_pool(validation_results: dict) -> dict:
    """
    Choose which bucket of papers to rank from, degrading gracefully.
    Priority: verified -> unverified -> anything not REJECTED in 'all'.
    Never selects from REJECTED papers.
    """
    verified = validation_results.get("verified") or []
    if verified:
        return {"pool": verified, "source": "verified", "degraded": False, "reason": ""}

    unverified = validation_results.get("unverified") or []
    if unverified:
        return {
            "pool": unverified,
            "source": "unverified",
            "degraded": True,
            "reason": (
                "No VERIFIED papers were available (Layer 3 validation did not confirm "
                "any candidate). Ranking falls back to UNVERIFIED papers -- this "
                "recommendation needs extra scrutiny before Master Agent / human approval."
            ),
        }

    all_papers = validation_results.get("all") or []
    non_rejected = [p for p in all_papers if p.get("validation_status") != "REJECTED"]
    if non_rejected:
        return {
            "pool": non_rejected,
            "source": "all_non_rejected",
            "degraded": True,
            "reason": (
                "No VERIFIED or UNVERIFIED papers were available -- validation could not be "
                "confirmed for any candidate (all still PENDING or unlabeled). Ranking falls "
                "back to every non-rejected paper. This recommendation is unconfirmed and "
                "must not advance without Master Agent / human review."
            ),
        }

    return {"pool": [], "source": "none", "degraded": True, "reason": "All candidate papers were rejected during validation."}


def _code_availability_audit(pool: list) -> dict:
    with_code = [p for p in pool if p.get("has_code")]
    note = ""
    if len(with_code) > 1:
        note = (
            f"{len(with_code)} of {len(pool)} qualifying papers report available code; "
            "project policy allows at most one paper's selection to be justified by code "
            "availability. This does not block selection but must not be used as the "
            "primary justification for more than one candidate."
        )
    return {
        "count_with_code": len(with_code),
        "titles_with_code": [p.get("title", "") for p in with_code],
        "policy_violation": len(with_code) > 1,
        "note": note,
    }


def _is_code_deciding_factor(rows: list, chosen_index: int) -> bool:
    """
    True only if removing the code-availability bonus would change the #1
    pick -- i.e. the claim is computed, not asserted.
    """
    if not rows[chosen_index]["code_availability"]:
        return False
    zeroed = [dict(r, composite_score=_composite_score(r, code_weight=0.0)) for r in rows]
    top_without_code = max(range(len(zeroed)), key=lambda i: zeroed[i]["composite_score"])
    return top_without_code != chosen_index


def _template_justify(chosen_row: dict, runner_up_rows: list) -> str:
    lines = [
        f"'{chosen_row['title']}' is recommended as the Parent Paper "
        f"(composite score {chosen_row['composite_score']:.2f})."
    ]
    reasons = []
    if chosen_row["peer_reviewed"]:
        reasons.append("it is peer-reviewed")
    if chosen_row["dataset_availability"] != "Not specified":
        reasons.append(f"its dataset is {chosen_row['dataset_availability'].lower()}")
    if chosen_row["methodology_clarity"] == "High":
        reasons.append("its methodology is clearly described")
    if chosen_row["implementation_clarity"] == "High":
        reasons.append("implementation details are clear enough to realistically replicate")
    if chosen_row["has_evaluation_metrics"]:
        reasons.append("it reports concrete evaluation metrics")
    if chosen_row["visuals"] != "none reported":
        reasons.append(f"it includes {chosen_row['visuals']}")
    if reasons:
        lines.append("Academic quality reasons: " + "; ".join(reasons) + ".")
    if chosen_row["code_availability"]:
        lines.append(
            "It also has an available code implementation, which is a secondary "
            "convenience factor, not the primary justification."
        )
    if chosen_row["reproducibility_estimated"]:
        lines.append(
            "Note: reproducibility for this paper is an estimate -- Layer 3 validation "
            "did not produce a confirmed score."
        )

    if runner_up_rows:
        lines.append("Runner-ups considered:")
        for r in runner_up_rows:
            lines.append(
                f"  - '{r['title']}' (score {r['composite_score']:.2f}): "
                f"methodology clarity {r['methodology_clarity']}, "
                f"dataset {r['dataset_availability'].lower()}, "
                f"reproducibility {r['reproducibility_difficulty']}."
            )
    return "\n".join(lines)


def rank_and_recommend(validation_results: dict) -> dict:
    """
    Rank qualifying papers and recommend one Parent Paper.

    Input: a dict shaped like this module's own tally
           ({"verified", "unverified", "rejected", "all"}).
    Output: a structured recommendation dict (see keys below). Never raises
    on missing/degraded data and never fabricates a score or claim not
    grounded in the paper dicts already gathered upstream.
    """
    print("\n[Validation Agent] Comparing candidate papers for Parent Paper selection...")

    pool_info = _select_candidate_pool(validation_results)
    pool = pool_info["pool"]

    if not pool:
        print("[Validation Agent] No non-rejected candidates available.")
        return {
            "status": "NO_CANDIDATES",
            "candidate_pool_source": pool_info["source"],
            "candidate_pool_size": 0,
            "degraded": True,
            "degraded_reason": pool_info["reason"],
            "comparison_table": [],
            "code_availability_audit": {"count_with_code": 0, "titles_with_code": [], "policy_violation": False, "note": ""},
            "recommended_parent_paper": None,
            "runner_ups": [],
            "justification": "",
            "expected_difficulty": "",
            "code_availability_deciding_factor": False,
        }

    if pool_info["degraded"]:
        print(f"[Validation Agent] WARNING: {pool_info['reason']}")
    if len(pool) < MIN_QUALIFYING:
        print(f"[Validation Agent] WARNING: Only {len(pool)} candidates in pool (need >= {MIN_QUALIFYING}).")

    rows = [_compute_criteria_row(p) for p in pool]
    ranked_indices = sorted(range(len(pool)), key=lambda i: rows[i]["composite_score"], reverse=True)

    chosen_idx = ranked_indices[0]
    runner_up_indices = ranked_indices[1:JUSTIFICATION_TOP_N]

    audit = _code_availability_audit(pool)
    deciding = _is_code_deciding_factor(rows, chosen_idx)
    justification = _template_justify(rows[chosen_idx], [rows[i] for i in runner_up_indices])

    print(f"[Validation Agent] Recommended: {pool[chosen_idx]['title'][:70]}")

    return {
        "status": "OK" if not pool_info["degraded"] else "DEGRADED",
        "candidate_pool_source": pool_info["source"],
        "candidate_pool_size": len(pool),
        "degraded": pool_info["degraded"],
        "degraded_reason": pool_info["reason"],
        "comparison_table": [rows[i] for i in ranked_indices],
        "code_availability_audit": audit,
        "recommended_parent_paper": pool[chosen_idx],
        "runner_ups": [pool[i] for i in runner_up_indices],
        "justification": justification,
        "expected_difficulty": rows[chosen_idx]["expected_difficulty"],
        "code_availability_deciding_factor": deciding,
    }


def run(papers: list) -> dict:
    """
    Validate a list of research papers through three layers, then rank the
    resulting candidate pool and recommend a single Parent Paper.
    Returns a dict with 'verified', 'unverified', 'rejected', 'all', plus
    the Parent Paper recommendation keys from rank_and_recommend().
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

    tally = {
        "verified": verified,
        "unverified": unverified,
        "rejected": rejected,
        "all": papers,
    }

    selection = rank_and_recommend(tally)
    return {**tally, **selection}
