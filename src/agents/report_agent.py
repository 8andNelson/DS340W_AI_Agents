"""
Report Agent (Milestone 13).

Writes the final research report from the run's own records -- never from
memory of other runs, and never with a number the records do not hold.

  1. gather_facts(project): every fact the report may state, copied
     deterministically from the approved results Master Agent holds
     (papers, Parent Paper, data, cleaning, code search, replication,
     evaluation, reruns, Experiment Agent analyses) and the run's LLM call
     log. Saved next to the report (report_facts.json), so every value can
     be traced.
  2. Tables, lists, figures and references are rendered from the facts by
     code: the metrics comparison, the literature, the datasets, the
     reruns, the environment. An LLM never writes a number into a table.
  3. One LLM call writes only the prose sections (abstract, introduction,
     literature review, methodology, discussion, limitations, conclusion,
     future work) from the facts.
  4. _ground() checks the prose deterministically, sentence by sentence,
     and removes a sentence that
       * states a number found nowhere in the facts (rounding allowed),
       * cites a paper key ([P3]) that is not in the literature list, or
       * claims more than the evaluation found (an "exact" reproduction
         when the status is not EXACT, a "successful" one when it FAILED or
         is INCONCLUSIVE, or the authors' code for a reconstructed baseline).
     Every removed sentence is listed in the report's provenance appendix.
     The check finds numbers that appear nowhere; it cannot prove that a
     number found in the facts is attributed correctly -- which is why the
     results themselves are only in the code-rendered tables.

If the LLM call fails or returns nothing usable, the report is still
written from the facts, with each prose section marked as not written
(status DEGRADED). Output: reports/<slug>/report.md, report_facts.json and
figures/*.svg (light surface; each bar has a tooltip, and every plotted
value is also in a table).
"""
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import llm

REPO_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
LITERATURE_START_YEAR = 2022
FACTS_PROMPT_CHARS = 24000
SECTION_MAX_CHARS = 2500
REPORT_TIMEOUT_SECONDS = 240

PROSE_SECTIONS = ("abstract", "introduction", "literature_review", "methodology", "discussion", "limitations",
                  "conclusion", "future_work")
# CLAUDE.md's report sections, in order, plus the provenance appendix.
SECTIONS = ("Abstract", "Introduction", "Research Question", "Literature Review", "Parent Paper", "Dataset",
            "Methodology", "Baseline Replication", "Proposed Changes / Experiments", "Experimental Setup",
            "Results", "Discussion", "Limitations", "Conclusion", "Future Work", "References")
APPENDICES = ("Appendix A: Memory Used", "Appendix B: Provenance")

# The chart palette (dataviz reference palette, categorical slots 1-2; both
# validated on the light surface) and its chrome.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
PUBLISHED_COLOR, REPRODUCED_COLOR, NOT_KEPT_COLOR = "#2a78d6", "#eb6834", "#c3c2b7"

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.\-])(-?\d+(?:,\d{3})*(?:\.\d+)?)(?![A-Za-z0-9_])")
_CITATION = re.compile(r"\[(P\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60]
    return slug or "report"


def _short(text, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


# --- 1. Facts ---

def _models_used(run_id: str | None) -> dict:
    """{model that answered: calls} for this run, from the LLM call log."""
    counts = Counter()
    if not run_id or not Path(llm.CALL_LOG).is_file():
        return {}
    with open(llm.CALL_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("run_id") == run_id and not entry.get("error"):
                counts[entry.get("served_by") or entry.get("requested_model") or "unknown"] += 1
    return dict(counts.most_common())


def _paper_record(p: dict, key: str, parent_title: str) -> dict:
    return {"key": key, "title": p.get("title", ""), "authors": list(p.get("authors") or []),
            "year": p.get("year"), "venue": p.get("venue", ""), "doi": p.get("doi", ""),
            "url": p.get("paper_url", ""), "status": p.get("validation_status") or "UNVERIFIED",
            "peer_reviewed": p.get("validated_peer_reviewed", p.get("peer_reviewed")),
            "dataset": p.get("dataset", ""), "methodology": _short(p.get("methodology"), 400),
            "results_summary": _short(p.get("results_summary"), 400),
            "is_parent": bool(parent_title) and p.get("title", "") == parent_title}


def gather_facts(project: dict, run_id: str = None) -> dict:
    """Every fact the report may state, from the run's approved results."""
    from src.agents.experiment_agent import score_gap, score_margin, signed_points

    intake = project.get("structured") or {}
    validation = project.get("validation_result") or {}
    parent = project.get("parent_paper") or validation.get("recommended_parent_paper") or {}
    dataset = project.get("dataset_result") or {}
    cleaning = project.get("cleaning_result") or {}
    code = project.get("code_discovery_result") or {}
    replication = project.get("replication_result") or {}
    evaluation = project.get("evaluation_result") or {}
    reruns = project.get("reruns") or []
    analyses = project.get("experiment_results") or []

    pool = [p for p in (validation.get("all") or project.get("papers") or [])
            if p.get("validation_status") != "REJECTED"]
    if parent and not any(p.get("title") == parent.get("title") for p in pool):
        pool = [parent] + pool
    papers = [_paper_record(p, f"P{i}", parent.get("title", "")) for i, p in enumerate(pool, 1)]
    parent_key = next((p["key"] for p in papers if p["is_parent"]), None)

    review = evaluation.get("review") or {}
    findings = [{"severity": f.get("severity"), "category": f.get("category"), "issue": _short(f.get("issue"), 300),
                 "evidence": _short(f.get("evidence"), 200)}
                for f in review.get("findings") or [] if f.get("verified")]
    findings += [{"severity": f.get("severity"), "category": "leakage (automated check)",
                  "issue": _short(f.get("issue"), 300), "evidence": _short(f.get("evidence"), 200)}
                 for f in review.get("static_findings") or []]
    plan = dataset.get("data_plan") or {}
    repo = code.get("repository") or {}
    replication_dir = replication.get("replication_dir") or ""
    environment = Path(replication_dir) / "ENVIRONMENT.md" if replication_dir else None

    baseline_gap = reruns[0].get("gap_before") if reruns else score_gap(evaluation)
    baseline_margin = reruns[0].get("margin_before") if reruns else score_margin(evaluation)
    # Changes the Experiment Agent declared to be beyond the Parent Paper's
    # method; the kept ones are part of the final result.
    beyond = [{"rerun": r.get("rerun"), "target_agent": r.get("target_agent"),
               "new_methods": list(r.get("new_methods") or []), "kept": bool(r.get("applied")),
               "instructions": [_short(i, 300) for i in r.get("instructions") or []]}
              for r in reruns if r.get("method_source") == "beyond_paper"]
    return {
        "topic": project.get("topic") or intake.get("project_topic") or "",
        "research_question": intake.get("research_question") or intake.get("domain") or "",
        "ml_task": intake.get("ml_task"),
        "target_terms": list(project.get("target_terms") or intake.get("target_terms") or []),
        "keywords": list(intake.get("keywords") or []),
        "input_mode": project.get("input_mode", "topic"),
        "literature": {"window_start": LITERATURE_START_YEAR, "count": len(papers),
                       "verified_count": sum(p["status"] == "VERIFIED" for p in papers), "papers": papers},
        "parent_paper": {
            "key": parent_key, "title": parent.get("title", ""), "authors": list(parent.get("authors") or []),
            "year": parent.get("year"), "venue": parent.get("venue", ""), "doi": parent.get("doi", ""),
            "url": parent.get("paper_url", ""), "status": parent.get("validation_status") or "UNVERIFIED",
            "abstract": _short(parent.get("abstract"), 1500), "methodology": _short(parent.get("methodology"), 1200),
            "dataset": parent.get("dataset", ""), "dataset_source": parent.get("dataset_source", ""),
            "results_summary": _short(parent.get("results_summary"), 800),
            "selection_justification": _short(validation.get("justification"), 2000),
            "expected_difficulty": validation.get("expected_difficulty"),
            "paper_text_status": replication.get("paper_text_status") or dataset.get("paper_text_status") or "",
        },
        "selection": {"pool_size": validation.get("candidate_pool_size"),
                      "candidates": [{"title": r.get("title"), "year": r.get("publication_year"),
                                      "composite_score": r.get("composite_score"),
                                      "reproducibility_difficulty": r.get("reproducibility_difficulty"),
                                      "dataset_availability": r.get("dataset_availability"),
                                      "code_availability": r.get("code_availability")}
                                     for r in (validation.get("comparison_table") or [])[:8]]},
        "data": {"mode": dataset.get("mode"), "parent_dataset_found": dataset.get("parent_dataset_found"),
                 "total_entries": dataset.get("total_entries"), "suggested_rows": plan.get("target_entries"),
                 "paper_dataset": plan.get("paper_dataset"), "paper_dataset_size": plan.get("paper_dataset_size"),
                 "paper_model": plan.get("paper_model"),
                 "datasets": [{"name": d.get("display_name") or d.get("name"), "link": d.get("name"),
                               "source": d.get("source"), "license": d.get("license"), "format": d.get("format"),
                               "rows": d.get("entry_count"), "original_rows": d.get("original_entry_count"),
                               "sampled": d.get("sampled"), "column_notes": d.get("column_notes")}
                              for d in dataset.get("selected_datasets") or []],
                 "warnings": list(dataset.get("warnings") or [])},
        "cleaning": {"rows_total": cleaning.get("rows_total"), "columns_total": cleaning.get("columns_total"),
                     "datasets_merged": cleaning.get("datasets_merged"),
                     "datasets_attempted": cleaning.get("datasets_attempted"),
                     "label_column": cleaning.get("label_column"), "split_rows": cleaning.get("split_rows") or {},
                     "split_seed": cleaning.get("split_seed"), "sampling": cleaning.get("sampling"),
                     "conflicts": [_short(c.get("reason"), 200) for c in cleaning.get("conflicts") or []]},
        "code": {"status": code.get("status"), "repo_url": repo.get("repo_url", ""), "license": repo.get("license"),
                 "source_stage": code.get("source_stage")},
        "replication": {"status": replication.get("status"), "source": replication.get("source"),
                        "repo_url": replication.get("repo_url"), "commit_sha": replication.get("commit_sha"),
                        "entry_point": replication.get("entry_point"), "exit_code": replication.get("exit_code"),
                        "duration_seconds": replication.get("duration_seconds"),
                        "inferred_assumptions": list(replication.get("inferred_assumptions") or []),
                        "modifications": list(replication.get("modifications") or []),
                        "generated_by": list(replication.get("generated_by") or []),
                        "revision_round": replication.get("revision_round"),
                        "fast_mode_from": replication.get("fast_mode_from"),
                        "warnings": list(replication.get("warnings") or []),
                        "pip_freeze_path": replication.get("pip_freeze_path") or "",
                        "environment_file": str(environment) if environment and environment.is_file() else "",
                        "sandbox": replication.get("sandbox"),
                        "repository_fallback": replication.get("repository_fallback"),
                        "baseline_failed": bool(replication.get("baseline_failed"))},
        "evaluation": {"status": evaluation.get("status"),
                       "comparison_basis": evaluation.get("comparison_basis") or "",
                       "compared_metrics": evaluation.get("compared_metrics"),
                       "metrics": [{**{k: m.get(k) for k in ("metric", "published_value", "reproduced_value",
                                                             "difference_points", "classification", "notes")},
                                    "signed_difference_points": signed_points(m),
                                    "published_scope": m.get("published_scope") or "paper_scenario"}
                                   for m in evaluation.get("metrics") or []],
                       "mean_gap_points": score_gap(evaluation), "mean_margin_points": score_margin(evaluation), "likely_causes": list(evaluation.get("likely_causes") or []),
                       "assessment": _short(review.get("assessment"), 1200), "verified_findings": findings,
                       "decision": (evaluation.get("decision") or {}).get("action")},
        "experiments": {"rerun_count": len(reruns), "kept_count": sum(bool(r.get("applied")) for r in reruns),
                        "baseline_gap_points": baseline_gap, "final_gap_points": score_gap(evaluation),
                        "baseline_margin_points": baseline_margin, "final_margin_points": score_margin(evaluation),
                        "beyond_paper": beyond,
                        "beyond_paper_in_final": [b for b in beyond if b["kept"]],
                        "end_reason": project.get("rerun_end_reason") or "",
                        "reruns": [{"rerun": r.get("rerun"), "requested_by": r.get("requested_by"),
                                    "target_agent": r.get("target_agent"),
                                    "instructions": [_short(i, 300) for i in r.get("instructions") or []],
                                    "goal": r.get("goal") or {}, "gap_before": r.get("gap_before"),
                                    "gap_after": r.get("gap_after"), "margin_before": r.get("margin_before"),
                                    "margin_after": r.get("margin_after"),
                                    "method_source": r.get("method_source") or "",
                                    "new_methods": list(r.get("new_methods") or []), "kept": bool(r.get("applied")),
                                    "improved": bool(r.get("improved")), "outcome": _short(r.get("outcome"), 300)}
                                   for r in reruns],
                        "analyses": [{"analysis": _short(a.get("analysis"), 600),
                                      "decision": (a.get("decision") or {}).get("action"),
                                      "target_agent": (a.get("decision") or {}).get("target_agent"),
                                      "gap_points": a.get("gap_points"), "margin_points": a.get("margin_points")}
                                     for a in analyses]},
        "run": {"run_id": run_id, "date": datetime.now(timezone.utc).date().isoformat(),
                "llm_provider": getattr(llm, "LLM_PROVIDER", ""), "models_used": _models_used(run_id)},
    }


def fact_numbers(facts) -> list:
    """Every number in the facts: numeric values and numbers inside strings."""
    found = []

    def walk(value):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            found.append(float(value))
        elif isinstance(value, str):
            found.extend(float(n.replace(",", "")) for n in _NUMBER.findall(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
    walk(facts)
    return found


# --- 4. Grounding the prose ---

def _supported(number: str, known: list) -> bool:
    """A prose number is supported when a fact rounds to it, directly or as
    a percentage (0.923 -> 92.3)."""
    value = float(number.replace(",", ""))
    decimals = len(number.split(".")[1]) if "." in number else 0
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(value - f) <= tolerance or abs(value - f * 100) <= tolerance or abs(value * 100 - f) <= tolerance
               for f in known)


def _overclaim(sentence: str, facts: dict) -> str:
    low = sentence.lower()
    status = str(facts["evaluation"].get("status") or "").upper()
    if status != "EXACT" and re.search(r"\b(exact(ly)?|perfect(ly)?)\s+(reproduc|replicat|match)", low):
        return f"claims an exact reproduction, but the evaluation status is {status or 'not set'}"
    if status in ("FAILED", "INCONCLUSIVE", "") and re.search(r"\bsuccessful(ly)?\s+(reproduc|replicat)", low):
        return f"claims a successful reproduction, but the evaluation status is {status or 'not set'}"
    surpassed = any(m.get("classification") == "SURPASSED" for m in facts["evaluation"].get("metrics") or [])
    if not surpassed and re.search(r"\b(surpass|outperform|exceed|beat|better than)\w*\b.{0,40}\b(paper|published|"
                                   r"original|authors)", low) and not re.search(r"\b(not|no|n't|did not)\b", low):
        return "claims a score above the paper, but no metric is SURPASSED"
    if facts["experiments"].get("beyond_paper_in_final") and \
            re.search(r"\b(only|solely|strictly|entirely)\b.{0,30}\b(paper's|paper|original)\s+(method|methodology)",
                      low) and not re.search(r"\b(not|no|n't)\b", low):
        return "says only the paper's method was used, but a kept rerun used a method beyond the paper"
    if facts["replication"].get("source") == "reconstructed" and \
            re.search(r"\b(authors'|official|original)\s+(code|implementation|repository)", low) and \
            not re.search(r"\b(not|no|n't|without|unavailable|could not)\b", low):
        return "presents the reconstructed baseline as the authors' code"
    return ""


def _ground(text: str, facts: dict, known: list) -> tuple[str, list]:
    """(the text without unsupported sentences, [{sentence, reason}])."""
    keys = {p["key"] for p in facts["literature"]["papers"]}
    kept, removed = [], []
    for paragraph in str(text or "").split("\n"):
        out = []
        for sentence in _SENTENCE.split(paragraph.strip()) if paragraph.strip() else []:
            unknown = [k for k in _CITATION.findall(sentence) if k not in keys]
            bare = _CITATION.sub("", sentence)
            unsupported = [n for n in _NUMBER.findall(bare) if not _supported(n, known)]
            reason = (f"cites {', '.join(unknown)}, not in the literature list" if unknown
                      else f"states {', '.join(unsupported)}, not in the run's records" if unsupported
                      else _overclaim(sentence, facts))
            if reason:
                removed.append({"sentence": sentence, "reason": reason})
            else:
                out.append(sentence)
        kept.append(" ".join(out))
    return "\n\n".join(p for p in kept if p).strip(), removed


# --- 3. The prose (one LLM call) ---

SYSTEM = (
    "You are the Report Agent of an academic reproduction project. You write the prose sections of the final "
    "research report, only from the FACTS given (JSON). Rules:\n"
    "1. Use only the facts. Never add a paper, author, dataset, metric, result or number that is not in them.\n"
    "2. Copy any number exactly as it is in the facts. Prefer to refer to the Results table over restating "
    "values.\n"
    "3. Cite papers only with their keys in square brackets, e.g. [P2]. The Parent Paper is facts.parent_paper.key.\n"
    "4. State the reproduction outcome exactly as facts.evaluation.status says (EXACT, SURPASSED, APPROXIMATE, "
    "FAILED or INCONCLUSIVE). SURPASSED means the reproduction scored above the paper. A metric marked "
    "NOT_IN_PAPER is one the paper does not state, and NOT_A_SCORE is a time, loss or cost: neither is part of the outcome. A reconstructed baseline "
    "(facts.replication.source == 'reconstructed') is not the authors' code.\n"
    "4a. When facts.experiments.beyond_paper_in_final is not empty, say plainly that the final result uses methods "
    "the Parent Paper does not describe, and name them. Never present them as the paper's method.\n"
    "5. Where a paper's status is UNVERIFIED, say it could not be verified. State uncertainty; never guess.\n"
    "6. A failed or imperfect reproduction is useful research information: report it plainly.\n"
    "Answer with one JSON object and nothing else, with these string keys: "
    + ", ".join(PROSE_SECTIONS) + f". Each value is 1-3 short paragraphs, at most {SECTION_MAX_CHARS} characters, "
    "plain text (no headings, no tables)."
)


def _prompt(facts: dict) -> str:
    text = json.dumps(facts, ensure_ascii=False, default=str)
    if len(text) > FACTS_PROMPT_CHARS:
        trimmed = json.loads(text)
        for p in trimmed["literature"]["papers"]:
            p.pop("methodology", None), p.pop("results_summary", None)
        text = json.dumps(trimmed, ensure_ascii=False, default=str)[:FACTS_PROMPT_CHARS]
    return f"FACTS:\n{text}\n\nWrite the prose sections as one JSON object."


def write_prose(facts: dict) -> tuple[dict, str, str]:
    """({section: text}, the model that wrote it, error or "")."""
    try:
        content = llm.chat(_prompt(facts), system=SYSTEM, timeout=REPORT_TIMEOUT_SECONDS, agent="report_agent")
    except Exception as e:
        return {}, "", f"{type(e).__name__}: {e}"
    parsed = llm.parse_json(content, default={})
    if not isinstance(parsed, dict):
        return {}, "", "the model's answer was not a JSON object"
    prose = {k: str(parsed.get(k) or "").strip()[:SECTION_MAX_CHARS] for k in PROSE_SECTIONS}
    if not any(prose.values()):
        return {}, "", "the model's answer had none of the prose sections"
    return prose, llm.last_served_by(), ""


# --- 2. Tables, figures, references ---

def _cell(value) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        value = f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list, rows: list) -> list:
    return (["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
            + ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows])


def _authors(authors: list) -> str:
    if not authors:
        return "Authors not recorded"
    return ", ".join(authors[:6]) + (", et al." if len(authors) > 6 else "")


def _bar(x: float, y: float, width: float, height: float, color: str, tip: str) -> str:
    """A horizontal bar anchored at x, with a 4px rounded data end."""
    width = max(width, 0.0)
    r = min(4.0, width / 2, height / 2)
    path = (f"M{x:.1f},{y:.1f} h{width - r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} v{height - 2 * r:.1f} "
            f"a{r:.1f},{r:.1f} 0 0 1 -{r:.1f},{r:.1f} h-{width - r:.1f} z")
    return f'<path d="{path}" fill="{color}"><title>{_xml(tip)}</title></path>'


def _xml(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _svg(width: int, height: int, title: str, body: list) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" '
            f'role="img" aria-label="{_xml(title)}">\n<title>{_xml(title)}</title>\n'
            f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>\n'
            f'<text x="16" y="24" font-size="14" font-weight="600" fill="{INK}">{_xml(title)}</text>\n'
            + "\n".join(body) + "\n</svg>\n")


def metrics_figure(metrics: list) -> str | None:
    """Published vs reproduced value per metric (both stated), or None."""
    rows = [m for m in metrics if isinstance(m.get("published_value"), (int, float))
            and isinstance(m.get("reproduced_value"), (int, float))]
    if not rows:
        return None
    top = max(max(m["published_value"], m["reproduced_value"]) for m in rows)
    scale_max = 1.0 if top <= 1.0 else top * 1.05
    left, plot_w, bar_h, gap = 150, 380, 14, 2
    group_h = 2 * bar_h + gap + 18
    height = 70 + group_h * len(rows) + 30
    body = [f'<rect x="{left}" y="40" width="10" height="10" rx="2" fill="{PUBLISHED_COLOR}"/>'
            f'<text x="{left + 16}" y="49" font-size="12" fill="{INK_2}">Published (Parent Paper)</text>',
            f'<rect x="{left + 190}" y="40" width="10" height="10" rx="2" fill="{REPRODUCED_COLOR}"/>'
            f'<text x="{left + 206}" y="49" font-size="12" fill="{INK_2}">Reproduced (this run)</text>']
    base_y = 66
    for i in range(5):
        x = left + plot_w * i / 4
        body.append(f'<line x1="{x:.1f}" y1="{base_y - 4}" x2="{x:.1f}" y2="{height - 30}" stroke="{GRID}" '
                    f'stroke-width="1"/><text x="{x:.1f}" y="{height - 14}" font-size="11" fill="{INK_2}" '
                    f'text-anchor="middle">{scale_max * i / 4:.3g}</text>')
    for n, m in enumerate(rows):
        y = base_y + n * group_h
        body.append(f'<text x="{left - 8}" y="{y + bar_h + 4}" font-size="12" fill="{INK}" '
                    f'text-anchor="end">{_xml(m["metric"])}</text>')
        for k, (key, color, label) in enumerate((("published_value", PUBLISHED_COLOR, "Published"),
                                                  ("reproduced_value", REPRODUCED_COLOR, "Reproduced"))):
            value = m[key]
            w = plot_w * value / scale_max
            by = y + k * (bar_h + gap)
            body.append(_bar(left, by, w, bar_h, color, f"{m['metric']}, {label}: {value}"))
            body.append(f'<text x="{left + w + 6:.1f}" y="{by + bar_h - 3}" font-size="11" fill="{INK_2}">'
                        f'{_xml(value)}</text>')
    return _svg(620, height, "Published vs reproduced metrics", body)


def gap_figure(facts: dict) -> str | None:
    """The mean difference from the paper (+ above it) after the baseline
    and each rerun, as bars around zero, or None."""
    exp = facts["experiments"]
    steps = []
    if exp.get("baseline_margin_points") is not None:
        steps.append(("Baseline", exp["baseline_margin_points"], True))
    steps += [(f"Rerun {r['rerun']} ({r['target_agent']})", r["margin_after"], r["kept"])
              for r in exp["reruns"] if isinstance(r.get("margin_after"), (int, float))]
    if len(steps) < 2:
        return None
    top = max(abs(v) for _, v, _ in steps) * 1.1 or 1.0
    left, plot_w, bar_h, row_h = 210, 330, 16, 26
    zero = left + plot_w / 2
    height = 70 + row_h * len(steps) + 30
    body = [f'<text x="16" y="48" font-size="12" fill="{INK_2}">Mean difference from the paper, percentage points '
            f'(right of the line: above the paper). Grey: not kept.</text>']
    for n, (label, value, kept) in enumerate(steps):
        y = 64 + n * row_h
        w = (plot_w / 2) * abs(value) / top
        color = PUBLISHED_COLOR if kept else NOT_KEPT_COLOR
        tip = f"{label}: {value:+g} points" + ("" if kept else " (not kept)")
        body.append(f'<text x="{left - 8}" y="{y + bar_h - 3}" font-size="12" fill="{INK}" '
                    f'text-anchor="end">{_xml(label)}</text>')
        if value >= 0:
            body.append(_bar(zero, y, w, bar_h, color, tip))
            text_x, anchor = zero + w + 6, "start"
        else:
            body.append(f'<rect x="{zero - w:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}">'
                        f'<title>{_xml(tip)}</title></rect>')
            text_x, anchor = zero - w - 6, "end"
        body.append(f'<text x="{text_x:.1f}" y="{y + bar_h - 3}" font-size="11" fill="{INK_2}" '
                    f'text-anchor="{anchor}">{value:+g}{"" if kept else " (not kept)"}</text>')
    body.append(f'<line x1="{zero:.1f}" y1="58" x2="{zero:.1f}" y2="{height - 26}" stroke="{INK_2}" stroke-width="1"/>')
    return _svg(620, height, "Difference from the Parent Paper over the reruns", body)


def _basis_line(ev: dict) -> str:
    """What the overall status was judged on (the Evaluation Agent's comparison_basis)."""
    other = sum(1 for m in ev["metrics"] if m.get("published_scope") == "other_scenario"
                and m.get("difference_points") is not None)
    count = ev.get("compared_metrics")
    if ev.get("comparison_basis") == "paper_and_other_scenarios":
        return (f"The status is judged on {count} metric(s): fewer than 3 of the paper's metrics for the matched "
                f"scenario could be compared, so {other} were compared with values the paper reports for another "
                "scenario (see the Notes column).")
    if ev.get("comparison_basis") == "paper_metrics":
        return f"The status is judged on the {count} metric(s) the paper reports for the matched scenario."
    return ""


def _signed(value) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value:+g}"


def _method_label(rerun: dict) -> str:
    if rerun.get("method_source") == "beyond_paper":
        return "beyond the paper: " + (", ".join(rerun.get("new_methods") or []) or "methods not named")
    return "paper's method" if rerun.get("method_source") == "paper" else "not stated"


def _beyond_paper_lines(exp: dict) -> list:
    """Section 9's note on changes beyond the Parent Paper's method."""
    if not exp["beyond_paper"]:
        return ["Every rerun stayed within the Parent Paper's method.", ""]
    lines = ["Changes beyond the Parent Paper's method (methods the paper does not describe):"]
    lines += [f"- Rerun {b['rerun']} ({b['target_agent']}): {', '.join(b['new_methods']) or 'methods not named'}; "
              + ("kept, so it is part of the final result." if b["kept"] else "not kept.")
              for b in exp["beyond_paper"]]
    return lines + [""]


def _environment_line(rep: dict) -> str:
    """Where the baseline ran, from the Replication Agent's own record."""
    box = rep.get("sandbox") or {}
    if rep.get("source") == "cloned_repository" and box:
        return (f"- The authors' code ran in a Docker sandbox: {box.get('image') or 'image not recorded'} "
                f"({box.get('digest') or 'digest not recorded'}), network {box.get('run_network')}, user "
                f"{box.get('run_user')}, Linux capabilities {box.get('capabilities')}, memory limit "
                f"{box.get('memory')}, {box.get('cpus')} CPUs.")
    line = "- The reconstructed baseline ran in its own virtual environment under replication/."
    if rep.get("repository_fallback"):
        line += (f" The discovered repository {rep['repository_fallback'].get('repo_url', '')} did not run "
                 f"({rep['repository_fallback'].get('reason', '')}).")
    return line


def _prose(sections: dict, key: str, why: str) -> list:
    text = sections.get(key)
    return [text if text else f"*This section was not written: {why}*", ""]


def render(facts: dict, sections: dict, removed: list, figures: dict, prose_error: str, written_by: str,
           memory_section: str) -> str:
    parent, ev, exp = facts["parent_paper"], facts["evaluation"], facts["experiments"]
    rep, data, cl = facts["replication"], facts["data"], facts["cleaning"]
    why = prose_error or "the language model returned no text for it"
    title = f"Reproducing \"{parent['title'] or 'the Parent Paper'}\""
    L = [f"# {title}", "",
         f"Topic: {facts['topic']}. Run {facts['run']['run_id'] or 'not recorded'}, {facts['run']['date']}. "
         "Every table and figure is generated from the run's records; see Appendix B.", ""]

    L += ["## 1. Abstract", ""] + _prose(sections, "abstract", why)
    L += ["## 2. Introduction", ""] + _prose(sections, "introduction", why)

    L += ["## 3. Research Question", "", f"- Topic: {facts['topic']}",
          f"- Research question: {facts['research_question'] or 'not recorded'}",
          f"- ML task: {facts['ml_task'] or 'not recorded'}",
          f"- Prediction target: {', '.join(facts['target_terms']) or 'not recorded'}"]
    if facts["keywords"]:
        L.append(f"- Search keywords: {', '.join(facts['keywords'])}")
    if facts["input_mode"] == "parent_pdf":
        L.append("- The user supplied the Parent Paper; its peer review was not independently verified.")
    L.append("")

    lit = facts["literature"]
    L += ["## 4. Literature Review", ""] + _prose(sections, "literature_review", why)
    L += [f"{lit['count']} paper(s) in the validated set ({lit['verified_count']} verified; published "
          f"{lit['window_start']} or later is a requirement):", ""]
    L += _table(["Key", "Title", "Year", "Venue", "Status", "DOI"],
                [[p["key"] + (" (Parent)" if p["is_parent"] else ""), p["title"], p["year"], p["venue"], p["status"],
                  p["doi"]] for p in lit["papers"]]) + [""]

    L += ["## 5. Parent Paper", "",
          f"**{parent['title']}** -- {_authors(parent['authors'])} ({parent['year'] or 'year not recorded'}), "
          f"{parent['venue'] or 'venue not recorded'}. Status: {parent['status']}."
          + (f" DOI: {parent['doi']}." if parent["doi"] else "") + (f" {parent['url']}" if parent["url"] else ""), "",
          f"Paper text read by the agents: {parent['paper_text_status'] or 'not recorded'}.", ""]
    if parent["selection_justification"]:
        L += ["Why it was selected (the Validation Agent's ranking, approved by Master Agent):", "",
              "> " + parent["selection_justification"].replace("\n", "\n> "), ""]
    if facts["selection"]["candidates"]:
        L += _table(["Candidate", "Year", "Composite score", "Reproducibility", "Dataset availability", "Code"],
                    [[c["title"], c["year"], c["composite_score"], c["reproducibility_difficulty"],
                      c["dataset_availability"], c["code_availability"]] for c in facts["selection"]["candidates"]])
        L.append("")

    L += ["## 6. Dataset", ""]
    if data["parent_dataset_found"] is False:
        L += ["**The Parent Paper's own dataset could not be obtained; the data below is a substitute.**", ""]
    if data["paper_dataset"]:
        L.append(f"The paper names its dataset as: {data['paper_dataset']}"
                 + (f" ({data['paper_dataset_size']} rows, as stated in the paper)." if data["paper_dataset_size"]
                    else "."))
    L.append("")
    L += _table(["Dataset", "Source", "License", "Rows used", "Rows in source", "Sampled"],
                [[d["name"], d["source"], d["license"], d["rows"], d["original_rows"], d["sampled"]]
                 for d in data["datasets"]]) + [""]
    split = cl["split_rows"]
    L += [f"The Cleaning Agent merged {cl['datasets_merged'] or 0} of {cl['datasets_attempted'] or 0} dataset(s) "
          f"into {cl['rows_total'] or 0} rows and {cl['columns_total'] or 0} columns (label column: "
          f"{cl['label_column'] or 'not recorded'}), split train/test/validation "
          f"{split.get('train', 'n/a')}/{split.get('test', 'n/a')}/{split.get('validation', 'n/a')} with seed "
          f"{cl['split_seed'] if cl['split_seed'] is not None else 'n/a'}.", ""]
    sampled_here = [s for s in cl["sampling"] or [] if isinstance(s, dict) and s.get("sampled")]
    if sampled_here:
        L += [f"Row budget sampling (stratified on the label, fixed seed): {json.dumps(sampled_here, default=str)}. "
              "The sample is not the full dataset.", ""]
    if cl["conflicts"]:
        L += ["Not merged (logged, not silently dropped):"] + [f"- {c}" for c in cl["conflicts"]] + [""]
    for d in data["datasets"]:
        if d.get("column_notes"):
            L.append(f"- Note on {d['name']}: {d['column_notes']}")

    L += ["", "## 7. Methodology", ""] + _prose(sections, "methodology", why)
    if rep["inferred_assumptions"]:
        L += ["Details inferred when the baseline was reconstructed (not stated by the authors):"]
        L += [f"- {a}" for a in rep["inferred_assumptions"]] + [""]

    L += ["## 8. Baseline Replication", ""]
    if rep["source"] == "cloned_repository":
        L.append(f"The baseline is the authors' published code: {rep['repo_url']} at commit "
                 f"{rep['commit_sha'] or 'not recorded'}.")
    else:
        L.append("No usable implementation was found"
                 + (f" (code search: {facts['code']['status']})" if facts["code"]["status"] else "")
                 + ", so the baseline was reconstructed from the paper. It is not the authors' code.")
        if rep["generated_by"]:
            L.append(f"The script was written and repaired by these models, in order: {', '.join(rep['generated_by'])}.")
    if rep["fast_mode_from"]:
        L.append(f"The script was adapted from a past run's script (memory {rep['fast_mode_from']}, fast mode), "
                 "not written anew; see Appendix A.")
    L += ["", *_table(["Item", "Value"], [["Status", rep["status"]], ["Entry point", rep["entry_point"]],
                                          ["Exit code", rep["exit_code"]], ["Duration (s)", rep["duration_seconds"]],
                                          ["Revision round", rep["revision_round"]]]), ""]
    if rep["modifications"]:
        L += ["Changes needed to run it:"] + [f"- {m}" for m in rep["modifications"]] + [""]

    L += ["## 9. Proposed Changes / Experiments", ""]
    if exp["reruns"]:
        L += [f"Master Agent ran {exp['rerun_count']} rerun(s) requested by the Evaluation and Experiment Agents; "
              f"{exp['kept_count']} were kept. The goal is the paper's score and, where possible, a higher one; a "
              "rerun that lowered the score was reverted. Difference = reproduced minus published, in percentage "
              "points (+ is above the paper).", ""]
        L += _table(["Rerun", "Requested by", "Agent rerun", "Method", "Instructions", "Difference before",
                     "Difference after", "Kept", "Outcome"],
                    [[r["rerun"], r["requested_by"], r["target_agent"], _method_label(r),
                      "; ".join(r["instructions"]) or r["goal"], _signed(r["margin_before"]),
                      _signed(r["margin_after"]), "yes" if r["kept"] else "no", r["outcome"]]
                     for r in exp["reruns"]]) + [""]
        L += _beyond_paper_lines(exp)
    else:
        L += ["No rerun was run.", ""]
    if exp["end_reason"]:
        L += [f"The rerun loop ended because {exp['end_reason']}.", ""]
    for n, a in enumerate(exp["analyses"], 1):
        if a["analysis"]:
            L.append(f"- Experiment Agent analysis {n} (difference {_signed(a.get('margin_points'))}"
                     f" points; decision {a['decision']}{' -> ' + a['target_agent'] if a['target_agent'] else ''}): "
                     f"{a['analysis']}")
    L.append("")

    L += ["## 10. Experimental Setup", "",
          f"- LLM provider: {facts['run']['llm_provider'] or 'not recorded'}; models that answered in this run: "
          + (", ".join(f"{m} ({n} calls)" for m, n in facts["run"]["models_used"].items()) or "not recorded"),
          f"- Train/test/validation split seed: {cl['split_seed'] if cl['split_seed'] is not None else 'n/a'}",
          f"- Environment record: {rep['environment_file'] or 'not recorded'}",
          f"- Installed packages (pip freeze): {rep['pip_freeze_path'] or 'not recorded'}",
          _environment_line(rep), ""]

    L += ["## 11. Results", "",
          f"Overall reproduction status: **{ev['status'] or 'not set'}**. "
          + (f"Mean difference from the paper: {_signed(ev['mean_margin_points'])} percentage points "
             "(+ is above the paper)"
             + (f" (baseline before reruns: {_signed(exp['baseline_margin_points'])})." if exp["reruns"] else ".")
             if ev["mean_margin_points"] is not None
             else "The difference from the paper is not measurable: no metric is stated on both sides."), ""]
    if _basis_line(ev):
        L += [_basis_line(ev), ""]
    if exp["beyond_paper_in_final"]:
        L += ["The final result includes changes beyond the Parent Paper's method: "
              + "; ".join(f"rerun {b['rerun']} ({', '.join(b['new_methods']) or 'methods not named'})"
                          for b in exp["beyond_paper_in_final"]) + ". It is not a pure reproduction of the paper.", ""]
    if ev["metrics"]:
        L += _table(["Metric", "Published", "Reproduced", "Difference (points, + above the paper)", "Classification",
                     "Notes"],
                    [[m["metric"], m["published_value"], m["reproduced_value"],
                      _signed(m.get("signed_difference_points")), m["classification"], m["notes"]]
                     for m in ev["metrics"]]) + [""]
        if any(m["classification"] == "NOT_IN_PAPER" for m in ev["metrics"]):
            L += ["NOT_IN_PAPER: the run printed this metric, but the Parent Paper does not state it. It is shown for "
                  "information and is not part of the overall status.", ""]
        if any(m["classification"] == "NOT_A_SCORE" for m in ev["metrics"]):
            L += ["NOT_A_SCORE: a time, loss, error or cost the paper states. It has no 0-1/0-100 scale, so it is not "
                  "compared and is not part of the overall status.", ""]
    else:
        L += ["No metric was stated on both sides, so no value can be compared.", ""]
    for key, caption in (("metrics_figure_path", "Published vs reproduced metrics"),
                         ("gap_figure_path", "Difference from the Parent Paper over the reruns")):
        if figures.get(key):
            L += [f"![{caption}](figures/{Path(figures[key]).name})", ""]
    if ev["verified_findings"]:
        L += ["Verified findings of the Evaluation Agent's review:"]
        L += [f"- [{f['severity']}, {f['category']}] {f['issue']} (evidence: \"{f['evidence']}\")"
              for f in ev["verified_findings"]] + [""]

    L += ["## 12. Discussion", ""] + _prose(sections, "discussion", why)
    if ev["likely_causes"]:
        L += ["Possible causes of the difference, from the Evaluation Agent:"]
        L += [f"- {c}" for c in ev["likely_causes"]] + [""]

    L += ["## 13. Limitations", ""] + _prose(sections, "limitations", why)
    known = []
    if data["parent_dataset_found"] is False:
        known.append("The Parent Paper's own dataset was not used (substitute data).")
    if any(d.get("sampled") for d in data["datasets"]) or sampled_here:
        known.append("The data was subsampled to the local row budget.")
    if rep["source"] != "cloned_repository":
        known.append("The baseline is a reconstruction from the paper, not the authors' code.")
    if parent["paper_text_status"] == "ABSTRACT_ONLY":
        known.append("Only the Parent Paper's abstract and metadata could be read, not its full text.")
    if lit["verified_count"] < lit["count"]:
        known.append(f"{lit['count'] - lit['verified_count']} paper(s) could not be fully verified (UNVERIFIED).")
    if len(facts["run"]["models_used"]) > 1:
        known.append("LLM calls were answered by several models, which makes the run less reproducible.")
    if known:
        L += ["Recorded limitations:"] + [f"- {k}" for k in known] + [""]

    L += ["## 14. Conclusion", ""] + _prose(sections, "conclusion", why)
    L += ["## 15. Future Work", ""] + _prose(sections, "future_work", why)

    L += ["## 16. References", ""]
    for p in lit["papers"]:
        L.append(f"- [{p['key']}] {_authors(p['authors'])} ({p['year'] or 'n.d.'}). {p['title']}. "
                 f"{p['venue'] or 'Venue not recorded'}." + (f" https://doi.org/{p['doi']}" if p["doi"]
                                                             else f" {p['url']}" if p["url"] else "")
                 + ("" if p["status"] == "VERIFIED" else f" ({p['status']})"))
    L.append("")

    L += [f"## {APPENDICES[0]}", "", (memory_section or "The memory record of this run is not available.").strip(), ""]
    L += [f"## {APPENDICES[1]}", "",
          "- Tables, lists, figures and references: generated by code from report_facts.json (the run's approved "
          "agent results and LLM call log).",
          ("- Prose sections: not written" + (f" ({prose_error})" if prose_error else "") + "."
           if not any(sections.values()) else
           f"- Prose sections: written by {written_by or 'the LLM (model not recorded)'}, then checked sentence by "
           "sentence against the facts."),
          f"- Sentences removed by the check: {len(removed)}."]
    L += [f"  - \"{_short(r['sentence'], 300)}\" -- {r['reason']}" for r in removed]
    L.append("")
    return "\n".join(L)


# --- The agent ---

def run(project: dict, run_id: str = None) -> dict:
    """Write the report. project: Master Agent's approved results (the
    keys of its run result). Returns the paths and what was checked."""
    facts = gather_facts(project, run_id)
    out_dir = REPORTS_DIR / _slugify(facts["parent_paper"]["title"] or facts["topic"])
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    for key, name, svg in (("metrics_figure_path", "metrics.svg", metrics_figure(facts["evaluation"]["metrics"])),
                           ("gap_figure_path", "gap.svg", gap_figure(facts))):
        path = figures_dir / name
        if svg:
            path.write_text(svg, encoding="utf-8")
            figures[key] = str(path)
        elif path.exists():
            path.unlink()  # a figure of an earlier run must not appear in this report

    prose, written_by, error = write_prose(facts)
    known = fact_numbers(facts)
    sections, removed = {}, []
    for key in PROSE_SECTIONS:
        text, dropped = _ground(prose.get(key, ""), facts, known)
        sections[key] = text
        removed += [dict(r, section=key) for r in dropped]

    try:
        from src.memory import report as memory_report
        memory_section = memory_report.render_markdown(run_id=run_id) if run_id else ""
        memory_section = memory_section.replace("## Memory used", "").strip()
    except Exception as e:  # the memory record is optional; the report is not
        memory_section = f"The memory record could not be read ({type(e).__name__})."

    text = render(facts, sections, removed, figures, error, written_by, memory_section)
    report_path = out_dir / "report.md"
    facts_path = out_dir / "report_facts.json"
    report_path.write_text(text, encoding="utf-8")
    facts_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    missing_prose = [k for k in PROSE_SECTIONS if not sections[k]]
    warnings = ([f"The prose was not written: {error}."] if error else []) + \
        ([f"Prose sections empty after the check: {', '.join(missing_prose)}."] if missing_prose and not error else [])
    return {
        "status": "DEGRADED" if error or missing_prose else "OK",
        "report_path": str(report_path),
        "facts_path": str(facts_path),
        **figures,
        "sections": [s for s in SECTIONS + APPENDICES if f" {s}" in text],
        "prose_sections_written": [k for k in PROSE_SECTIONS if sections[k]],
        "prose": sections,  # after the grounding check; the Presentation Agent's source text
        "removed_claims": removed,
        "written_by": written_by,
        "warnings": warnings,
    }


def handoff(result: dict, context: dict) -> list:
    """The Report Agent's request to Master Agent: the Presentation Agent
    builds the slides from this report."""
    from src.orchestration.handoff import request

    return [request("report_agent", "Presentation Agent", {"report_result": result},
                    "The report is written; build the slides from its facts and figures.")]
