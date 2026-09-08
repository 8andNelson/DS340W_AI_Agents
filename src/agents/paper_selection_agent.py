"""
Parent Paper Selection Agent (Milestone 4).

Ranks the qualifying papers produced by validation_agent.run() against the
CLAUDE.md comparison criteria and recommends a single Parent Paper, with a
justification explaining why it beats the runner-ups.

Ranking is fully deterministic (plain Python over fields already gathered by
research_agent + validation_agent) — no LLM call, no network dependency, and
no risk of fabricating a score or claim not grounded in the paper data. The
Master Agent (Milestone 5) makes the final human/agent approval; this agent
only recommends.
"""

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
    derived from a heuristic fallback — never presented as if it were a real
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
    grounds — this weighting makes that structurally hard to violate).
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
                "any candidate). Ranking falls back to UNVERIFIED papers — this "
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
                "No VERIFIED or UNVERIFIED papers were available — validation could not be "
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
    pick — i.e. the claim is computed, not asserted.
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
            "Note: reproducibility for this paper is an estimate — Layer 3 validation "
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


def run(validation_results: dict) -> dict:
    """
    Rank qualifying papers and recommend one Parent Paper.

    Input: the dict returned by validation_agent.run()
           ({"verified", "unverified", "rejected", "all"}).
    Output: a structured recommendation dict (see keys below). Never raises
    on missing/degraded data and never fabricates a score or claim not
    grounded in the paper dicts already gathered upstream.
    """
    print("\n[Paper Selection Agent] Comparing candidate papers for Parent Paper selection...")

    pool_info = _select_candidate_pool(validation_results)
    pool = pool_info["pool"]

    if not pool:
        print("[Paper Selection Agent] No non-rejected candidates available.")
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
        print(f"[Paper Selection Agent] WARNING: {pool_info['reason']}")
    if len(pool) < MIN_QUALIFYING:
        print(f"[Paper Selection Agent] WARNING: Only {len(pool)} candidates in pool (need >= {MIN_QUALIFYING}).")

    rows = [_compute_criteria_row(p) for p in pool]
    ranked_indices = sorted(range(len(pool)), key=lambda i: rows[i]["composite_score"], reverse=True)

    chosen_idx = ranked_indices[0]
    runner_up_indices = ranked_indices[1:JUSTIFICATION_TOP_N]

    audit = _code_availability_audit(pool)
    deciding = _is_code_deciding_factor(rows, chosen_idx)
    justification = _template_justify(rows[chosen_idx], [rows[i] for i in runner_up_indices])

    print(f"[Paper Selection Agent] Recommended: {pool[chosen_idx]['title'][:70]}")

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
