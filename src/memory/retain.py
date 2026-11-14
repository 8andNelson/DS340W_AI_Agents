"""
Retain: turn a finished run's records into memories (milestone M4;
docs/CLAUDE_BACKEND_PLAN.md, sections 8.2, 9.1 and 10).

retain_run(run_id) reads the run database (runs, stages, artifacts,
stage_timings) and the run's lineage from the memory store, and writes
memories by these rules -- deterministic, no LLM:

  approved stage output   one memory per approved attempt. Only a "kept"
                          attempt is a positive solution; superseded,
                          reverted and unknown ones are neutral. A kept
                          baseline script whose evaluation found leakage is
                          a negative memory (a warning), not a solution.
  failed attempt          a failure memory (Retrying / Rejected): failed
                          checks or the exception, and Master Agent's
                          diagnosis; with a failure signature.
  diagnosis               a Master Agent decision memory: the diagnosis and
                          whether the next attempt then passed.
  rerun                   a decision memory for each rerun request Master
                          Agent opened (result["reruns"]): positive when it
                          moved the score closer, negative when undone.
  evaluation finding      each verified finding (code_checks leakage, or a
                          verified analyst finding): positive when the
                          Evaluation Agent's fix was kept, else neutral.

Not covered, because the agents do not record them: datasets and
repositories that were rejected, and column mappings (empty in every
recorded run so far).

A memory identical to one from another run (same agent and content hash)
is not stored again: the existing memory's seen_count goes up. A run is
retained once (memory_runs). For runs recorded before run recording
(milestone M3) the lineage is derived from the stages, and is "unknown"
where reruns may have replaced a result (plan section 10).

Scores come from ranking.py; every component's reason is stored.
"""
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.memory import features, ranking
from src.orchestration import run_store
from src.orchestration.handoff import AGENT_PHASE

NAMESPACE = uuid.UUID("6f3b2a4e-6d1c-4c8e-9b1a-3e7f0d2c5a91")  # deterministic memory ids per run and rule
# Version of the rules below. 2 (2026-11-13): cleaning with no label column is
# a warning; "no repository found" is neutral; an evaluation that compared
# nothing has outcome 0.5 and is neutral; the earlier controlled experiments are scored by
# their result (a regression is a warning). rescore_run applies a new version
# to runs retained under an older one.
RULES_VERSION = 2
RERUNNABLE = ("Data Agent", "Cleaning Agent", "Code Discovery Agent", "Replication Agent", "Evaluation Agent")
NON_BASELINE_AGENTS = ("Experiment Agent",)
REPRODUCTION_AGENTS = ("Validation Agent", "Replication Agent", "Evaluation Agent", "Experiment Agent")
DATA_AGENTS = ("Data Agent", "Cleaning Agent")
# The commit that added the target-fit check (plan 10): Data and Cleaning
# memories from runs before it are low_trust.
TARGET_FIT_COMMIT = "b8519b6"
REVIEW_PAPER = re.compile(r"\b(review|survey|systematic literature|meta-analysis)\b", re.I)
PHASES = list(dict.fromkeys(AGENT_PHASE.values()))
DIAGNOSIS_MARK = "--- Master Agent diagnosis ---"
REPO_ROOT = Path(run_store.REPO_ROOT)
MAX_EXCERPT_CHARS = 1500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decode(text):
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


def _portable(value):
    """Machine paths -> placeholders, so memories read the same on every
    machine and do not carry a home folder."""
    if isinstance(value, str):
        for path, placeholder in ((str(REPO_ROOT), "<repo>"), (str(Path.home()), "~")):
            for form in {path, path.replace("\\", "/"), path.replace("\\", "\\\\")}:
                value = value.replace(form, placeholder)
        return value
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def _json(value) -> str:
    return json.dumps(_portable(value), indent=1, default=str, ensure_ascii=False)


def _short(text, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


@lru_cache(maxsize=64)
def _has_target_fit(commit: str) -> bool | None:
    if not commit:
        return None
    try:
        proc = subprocess.run(["git", "merge-base", "--is-ancestor", TARGET_FIT_COMMIT, commit],
                              cwd=REPO_ROOT, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(proc.returncode)


# --- Loading a run ---

def load_run(run_id: str) -> dict | None:
    """Everything retain needs about one run, or None for an unknown run."""
    with run_store._connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        stages = [dict(r) for r in conn.execute("SELECT * FROM stages WHERE run_id = ? ORDER BY id", (run_id,))]
        timings = {(r["agent"], r["attempt"]): r["invocation_id"] for r in conn.execute(
            "SELECT agent, attempt, invocation_id FROM stage_timings WHERE run_id = ?", (run_id,))}
        artifacts = [dict(r) for r in conn.execute(
            "SELECT id, stage_id, kind, original_path, sha256 FROM artifacts WHERE run_id = ? ORDER BY id", (run_id,))]
    run = dict(run)
    for s in stages:
        s["output"] = _decode(s["output_json"])
        s["invocation_id"] = timings.get((s["agent"], s["attempt"]))
    result = _decode(run["result_json"])
    result = result if isinstance(result, dict) else None

    def kept(key: str, agent: str):
        if result is not None:
            value = result.get(key)
            return value if isinstance(value, dict) else None
        approved = [s["output"] for s in stages if s["agent"] == agent and s["decision"] == "Approved"
                    and isinstance(s["output"], dict)]
        return approved[-1] if approved else None

    status = run["status"]
    if status == "running":
        from src.orchestration import run_metrics
        status, _ = run_metrics._status(run, stages)
    validation = kept("validation_result", "Validation Agent") or {}
    parent = _decode(run["parent_paper_json"]) or validation.get("recommended_parent_paper") or {}
    return {
        "run_id": run_id, "topic": run["topic"], "status": status, "final_phase": run["final_phase"] or "",
        "git_commit": run["project_git_commit"] or "", "started_at": run["started_at"],
        "finished_at": run["finished_at"], "result": result, "stages": stages, "artifacts": artifacts,
        "intake": kept("structured", "Intake Agent") or {}, "parent_paper": parent,
        "dataset": kept("dataset_result", "Data Agent"), "cleaning": kept("cleaning_result", "Cleaning Agent"),
        "replication": kept("replication_result", "Replication Agent"),
        "final_evaluation": (result or {}).get("evaluation_result") if result else None,
        "reruns": result.get("reruns") if result is not None and "reruns" in result else None,
    }


# --- Lineage ---

def recorded_lineage(run: dict) -> dict:
    """{invocation_id: (lineage, reason)} from run recording (milestone M3)."""
    from src import memory
    c = memory.client()
    if c is None or c.local is None:
        return {}
    return {r["invocation_id"]: (r["lineage"], r.get("reason") or "")
            for r in c.local.rows("invocation_lineage", run_id=run["run_id"])}


def derived_lineage(run: dict) -> dict:
    """Lineage of a run recorded before milestone M3, from its stages:
    {stage id: (lineage, reason)}. Unknown where it cannot be known."""
    finished = run["status"] in ("success", "halted")
    reruns = run["reruns"] or []
    last_approved = {}
    for s in run["stages"]:
        if s["decision"] == "Approved":
            last_approved[s["agent"]] = s["id"]
    out = {}
    for s in run["stages"]:
        if s["decision"] != "Approved":
            out[s["id"]] = ("rejected", f"Master Agent decision: {s['decision']}")
        elif not finished:
            out[s["id"]] = ("unknown", "the run did not finish normally, so its final baseline is unknown")
        elif reruns and s["agent"] in RERUNNABLE:
            out[s["id"]] = ("unknown", "recorded before run recording: rerun boundaries are not known")
        elif s["agent"] in NON_BASELINE_AGENTS or last_approved[s["agent"]] == s["id"]:
            out[s["id"]] = ("kept", "approved; part of the final result (derived from the stages)")
        else:
            out[s["id"]] = ("superseded", "replaced by a later approved attempt (derived from the stages)")
    return out


def stage_lineage(run: dict) -> tuple[dict, str]:
    """({stage id: (lineage, reason)}, how it was found)."""
    recorded = recorded_lineage(run)
    if recorded:
        derived = derived_lineage(run)
        return ({s["id"]: recorded.get(s["invocation_id"]) or derived[s["id"]] for s in run["stages"]},
                "run recording")
    return derived_lineage(run), "derived from stages (recorded before run recording)"


# --- Helpers over a run's stages ---

def _failures_before(run: dict, stage: dict) -> int:
    count = 0
    for s in run["stages"]:
        if s["id"] >= stage["id"]:
            break
        if s["agent"] == stage["agent"]:
            count = 0 if s["decision"] == "Approved" else count + 1
    return count


def _next_of(run: dict, stage: dict, agent: str, decision: str = None) -> dict | None:
    return next((s for s in run["stages"] if s["id"] > stage["id"] and s["agent"] == agent
                 and (decision is None or s["decision"] == decision)), None)


def _evaluation_of(run: dict, replication_stage: dict) -> dict | None:
    """The approved evaluation of this baseline: the next approved
    Evaluation stage before the next approved Replication stage."""
    for s in run["stages"]:
        if s["id"] <= replication_stage["id"]:
            continue
        if s["agent"] == "Replication Agent" and s["decision"] == "Approved":
            return None
        if s["agent"] == "Evaluation Agent" and s["decision"] == "Approved":
            return s
    return None


def _script(run: dict, stage: dict) -> str | None:
    output = stage["output"] or {}
    scripts = [a for a in run["artifacts"] if a["stage_id"] == stage["id"] and a["kind"] == "script"]
    entry = output.get("entry_point") or ""
    chosen = next((a for a in scripts if Path(a["original_path"]).name == entry), scripts[0] if scripts else None)
    if not chosen:
        return None
    content = run_store.read_artifact(chosen["id"])
    return content.decode("utf-8", errors="replace") if content is not None else None


def _diagnosis(report_text: str) -> tuple[str | None, list]:
    if DIAGNOSIS_MARK not in (report_text or ""):
        return None, []
    tail = report_text.split(DIAGNOSIS_MARK, 1)[1]
    cause = re.search(r"Cause:\s*(.+)", tail)
    guidance = [line[2:].strip() for line in tail.splitlines() if line.startswith("- ")]
    return (cause.group(1).strip() if cause else None), guidance


def _attempt_events(run: dict) -> dict:
    from src import memory
    c = memory.client()
    if c is None or c.local is None:
        return {}
    return {e["invocation_id"]: e.get("payload") or {}
            for e in c.local.rows("memory_events", run_id=run["run_id"], event_type="attempt") if e.get("invocation_id")}


def _leakage(evaluation_output: dict | None) -> list:
    return list(((evaluation_output or {}).get("review") or {}).get("static_findings") or [])


def _gap(evaluation_output: dict | None):
    from src.agents.experiment_agent import score_gap
    return score_gap(evaluation_output) if evaluation_output else None


def _review_paper(run: dict) -> bool:
    paper = run["parent_paper"] or {}
    return bool(REVIEW_PAPER.search(f"{paper.get('title', '')} {paper.get('abstract', '')[:600]}"))


def _trust(run: dict, agent: str) -> tuple[str, str]:
    if agent in REPRODUCTION_AGENTS and _review_paper(run):
        return "low_trust", "the Parent Paper looks like a review or survey (no experiment of its own to reproduce)"
    if agent in DATA_AGENTS and _has_target_fit(run["git_commit"]) is False:
        return "low_trust", "recorded before the target-fit check existed"
    return "normal", ""


def _downstream(run: dict, agent: str) -> tuple[float, str]:
    if run["status"] == "success" and run["final_evaluation"]:
        return 1.0, "the run reached a kept evaluation"
    if run["status"] not in ("success", "halted"):
        return 0.5, "the run crashed; what came after is unknown"
    stopped = PHASES.index(run["final_phase"]) if run["final_phase"] in PHASES else -1
    mine = PHASES.index(AGENT_PHASE[agent]) if agent in AGENT_PHASE else -1
    if stopped > mine:
        return 0.5, f"the run went past this agent and stopped at {run['final_phase']}"
    return 0.0, "the run stopped at this agent"


# --- Rules ---

def _outcome(run: dict, stage: dict, lineage: str) -> tuple[tuple, tuple]:
    """(outcome, validity) of an approved attempt: each (value, reason)."""
    agent, output = stage["agent"], stage["output"] or {}
    valid = (1.0, "no verified leakage")
    if agent in ("Intake Agent", "Research Agent"):
        value, reason = _downstream(run, agent)
        return (value, f"same as downstream: {reason}"), valid
    if agent == "Validation Agent":
        final = run["final_evaluation"]
        if _review_paper(run):
            return (0.0, "the Parent Paper looks like a review or survey"), valid
        if final and _gap(final) is not None:
            return (1.0, "the Parent Paper reached a measurable gap"), valid
        if final:
            return (0.3, "INCONCLUSIVE: no metric stated by both the paper and the run"), valid
        return (0.5, "no evaluation was reached; the paper's reproducibility is unknown"), valid
    if agent == "Data Agent":
        cleaning = _next_of(run, stage, "Cleaning Agent", "Approved")
        if cleaning and (cleaning["output"] or {}).get("datasets_merged"):
            return (1.0, "its datasets were merged by the Cleaning Agent"), valid
        if run["final_phase"] == "DATA_CLEANING" and run["status"] == "halted":
            return (0.0, "the run halted at cleaning"), valid
        return (0.5, "whether its datasets were merged is not recorded"), valid
    if agent == "Cleaning Agent":
        attempted, merged = output.get("datasets_attempted"), output.get("datasets_merged")
        if attempted:
            return (merged / attempted, f"{merged} of {attempted} dataset(s) merged"), valid
        return (0.5, "no merge count recorded"), valid
    if agent == "Code Discovery Agent":
        if output.get("status") == "FOUND":
            replication = _next_of(run, stage, "Replication Agent", "Approved")
            if replication and (replication["output"] or {}).get("source") == "cloned_repo":
                return (1.0, "the repository it found was run"), valid
            return (0.5, "a repository was found but not run"), valid
        return (0.5, "no repository found; reconstruction from the paper is a valid outcome"), valid
    if agent == "Replication Agent":
        evaluation = _evaluation_of(run, stage)
        if evaluation is None:
            return (0.5, "this baseline was not evaluated"), valid
        leaks = _leakage(evaluation["output"])
        if leaks:
            return ((0.5, "gap not counted: the baseline had verified leakage"),
                    (0.0, f"verified leakage: {_short(leaks[0].get('issue'))}"))
        gap = _gap(evaluation["output"])
        if gap is None:
            return (0.5, "no metric stated by both the paper and the run"), valid
        return (ranking.gap_outcome(gap), f"gap to the paper: {gap} points"), valid
    if agent == "Evaluation Agent":
        review = output.get("review") or {}
        static = review.get("static_findings") or []
        verified = [f for f in review.get("findings") or [] if f.get("verified")]
        unverified = [f for f in review.get("findings") or [] if not f.get("verified")]
        fixed = any(r.get("requested_by") == "Evaluation Agent" and r.get("applied") for r in run["reruns"] or [])
        if static or verified:
            if fixed:
                return (1.0, f"{len(static) + len(verified)} real finding(s), and the fix was kept "
                             "(validity is success even if the gap grew)"), valid
            return (0.5, f"{len(static) + len(verified)} real finding(s); no kept fix recorded"), valid
        if _gap(output) is None:
            # Rules v2: an evaluation that compared nothing has not shown it judged well.
            return (0.5, "nothing was compared: no metric stated by both the paper and the run "
                         f"(status {output.get('status')})"), valid
        decision = (output.get("decision") or {}).get("action")
        if decision == "accept":
            note = f"; {len(unverified)} unverified claim(s) not acted on" if unverified else ""
            return (1.0, "accepted a measured baseline with no verified defect" + note), valid
        return (0.5, f"decision {decision or 'unknown'} without a verified finding"), valid
    if agent == "Experiment Agent":
        if "what_changed" in output:  # the earlier design: one controlled experiment
            result = str(output.get("overall") or "").upper()
            value = {"IMPROVED": 1.0, "NEUTRAL": 0.5, "REGRESSED": 0.0}.get(result, 0.5)
            return (value, f"controlled experiment result: {result or 'unknown'}"), valid
        return (0.5, "its effect is in the rerun decision memories"), valid
    return (0.5, "no outcome rule for this agent"), valid


def _approved_warning(run: dict, agent: str, output: dict) -> tuple[str, str] | None:
    """(cause, failure signature) when an approved output itself shows a
    problem worth warning about (rules v2), else None."""
    if agent == "Cleaning Agent" and (run["intake"] or {}).get("target_terms") and not output.get("label_column"):
        return ("the merged data has no label column for the target "
                f"{(run['intake'] or {}).get('target_terms')}", "cleaning:no_label_column")
    if agent == "Experiment Agent" and str(output.get("overall") or "").upper() == "REGRESSED":
        return (f"this change made the results worse: {_short(output.get('what_changed'), 200)}",
                f"experiment:{output.get('category') or 'unknown'}:regressed")
    return None


def _approved_memory(run: dict, stage: dict, lineage: tuple, script: str = None) -> dict | None:
    agent, output = stage["agent"], stage["output"]
    if not isinstance(output, dict):
        return None
    lin, lin_reason = lineage
    trust, trust_reason = _trust(run, agent)
    outcome, validity = _outcome(run, stage, lin)
    downstream = _downstream(run, agent)
    approval = (ranking.approval_score(_failures_before(run, stage)),
                f"approved after {_failures_before(run, stage)} failed attempt(s)")
    paper = (run["parent_paper"] or {}).get("title") or "the Parent Paper"
    content, title, summary = _describe(run, stage, output, script, paper)
    feats = features.task_features(run, agent, script=script)

    if agent == "Replication Agent" and lin == "kept" and validity[0] == 0.0:
        # A kept baseline with verified leakage is a warning, not a solution.
        leak = _leakage((_evaluation_of(run, stage) or {}).get("output"))[0]
        revised = any(r.get("requested_by") == "Evaluation Agent" and r.get("applied") for r in run["reruns"] or [])
        quality, reasons = ranking.warning_usefulness(leak.get("issue"), revised or None,
                                                      f"leakage:{leak.get('check') or 'unknown'}", trust)
        return _memory(run, stage, "baseline", kind="failure", polarity="negative", agent=agent,
                       title="Leaky baseline: " + title, summary=f"Verified leakage: {_short(leak.get('issue'), 300)}. "
                       + summary, content=content, feats=feats, lineage=lin, trust=trust,
                       quality=quality, reasons={**reasons, "lineage": lin_reason, "trust_note": trust_reason},
                       outcome={"decision": "Approved", "lineage": lin, "leakage": leak})
    warning = _approved_warning(run, agent, output)
    if warning:
        # Rules v2: an approved output that shows a problem is a warning, not a solution.
        cause, signature = warning
        quality, reasons = ranking.warning_usefulness(cause, None, signature, trust)
        return _memory(run, stage, "approved", kind="failure", polarity="negative", agent=agent, title=title,
                       summary=f"{cause}. {summary}", content=content, feats=feats, lineage=lin, trust=trust,
                       quality=quality, reasons={**reasons, "lineage": lin_reason, "trust_note": trust_reason},
                       outcome={"decision": "Approved", "lineage": lin, "problem": cause})
    quality, reasons = ranking.base_quality(outcome, approval, validity, downstream, trust)
    # Only a kept attempt is positive -- and only when it is something to reuse:
    # "no repository found" is a fact about the paper (rules v2), and a
    # controlled experiment is positive only when it improved the results.
    reusable = not (agent == "Code Discovery Agent" and output.get("status") != "FOUND")
    if agent == "Evaluation Agent" and _gap(output) is None and not _leakage(output):
        reusable = False  # compared nothing and found nothing: nothing was validated
    if agent == "Experiment Agent" and "what_changed" in output:
        reusable = str(output.get("overall") or "").upper() == "IMPROVED"
    polarity = "positive" if lin == "kept" and reusable else "neutral"
    return _memory(run, stage, "approved", kind="solution", polarity=polarity, agent=agent, title=title,
                   summary=summary + ("" if lin == "kept" else f" Lineage: {lin} ({lin_reason})."),
                   content=content, feats=feats, lineage=lin, trust=trust, quality=quality,
                   reasons={**reasons, "lineage": lin_reason, "trust_note": trust_reason},
                   outcome={"decision": "Approved", "lineage": lin, "attempt": stage["attempt"]})


def _describe(run: dict, stage: dict, output: dict, script: str, paper: str) -> tuple[str, str, str]:
    """(content, title, summary) of an approved output, per agent."""
    agent = stage["agent"]
    if agent == "Intake Agent":
        keep = {k: output.get(k) for k in ("domain", "ml_task", "keywords", "search_queries", "target_terms")}
        return (_json(keep), f"Intake: {output.get('domain', '')} / {output.get('ml_task', '')}",
                f"Topic '{run['topic']}' framed as {output.get('ml_task')}; target terms {output.get('target_terms')}.")
    if agent == "Research Agent":
        papers = output if isinstance(output, list) else output.get("papers") or []
        listing = [{k: p.get(k) for k in ("title", "year", "venue", "doi")} for p in papers if isinstance(p, dict)]
        return (_json(listing), f"Research: {len(listing)} candidate paper(s) for '{_short(run['topic'], 60)}'",
                f"Found {len(listing)} candidate paper(s), e.g. {_short(listing[0]['title'], 90) if listing else 'none'}.")
    if agent == "Validation Agent":
        rec = output.get("recommended_parent_paper") or {}
        keep = {"recommended": {k: rec.get(k) for k in ("title", "year", "venue", "doi", "dataset", "methodology")},
                "justification": output.get("justification"), "expected_difficulty": output.get("expected_difficulty"),
                "candidate_pool_size": output.get("candidate_pool_size")}
        return (_json(keep), f"Parent Paper: {_short(rec.get('title'), 90)}",
                f"Chose '{_short(rec.get('title'), 100)}' ({rec.get('year')}, {rec.get('venue')}) from "
                f"{output.get('candidate_pool_size')} candidates; difficulty {output.get('expected_difficulty')}.")
    if agent == "Data Agent":
        datasets = [{k: d.get(k) for k in ("name", "display_name", "source", "access_method", "license", "format",
                                           "entry_count", "verification_method", "matches_parent_paper",
                                           "target_fit_notes", "column_notes")}
                    for d in output.get("selected_datasets") or []]
        names = ", ".join(_short(d.get("display_name") or d.get("name"), 50) for d in datasets) or "none"
        return (_json({"datasets": datasets, "parent_dataset_found": output.get("parent_dataset_found"),
                       "mode": output.get("mode"), "warnings": output.get("warnings")}),
                f"Dataset for '{_short(paper, 60)}': {names}",
                f"{len(datasets)} dataset(s): {names}; the Parent Paper's own dataset "
                f"{'was' if output.get('parent_dataset_found') else 'was not'} found.")
    if agent == "Cleaning Agent":
        keep = {k: output.get(k) for k in ("label_column", "rows_total", "columns_total", "datasets_attempted",
                                           "datasets_merged", "column_mappings", "conflicts", "sampled")}
        return (_json(keep), f"Cleaning: {output.get('datasets_merged')} dataset(s) merged, label "
                f"'{output.get('label_column')}'",
                f"Merged {output.get('datasets_merged')} of {output.get('datasets_attempted')} dataset(s) into "
                f"{output.get('rows_total')} rows; {len(output.get('conflicts') or [])} conflict(s).")
    if agent == "Code Discovery Agent":
        repo = output.get("repository") or {}
        if output.get("status") == "FOUND":
            return (_json(repo), f"Repository for '{_short(paper, 60)}': {repo.get('repo_url') or repo.get('url')}",
                    f"Found {repo.get('repo_url') or repo.get('url')} via {output.get('source_stage')}.")
        return (_json({"status": output.get("status"), "notes": output.get("notes")}),
                f"No repository found for '{_short(paper, 70)}'",
                "No implementation was found; the baseline is reconstructed from the paper.")
    if agent == "Replication Agent":
        fams = ", ".join(features.model_families(script)) if script else "unknown model"
        source = output.get("source") or "unknown source"
        evaluation = _evaluation_of(run, stage)
        gap = _gap((evaluation or {}).get("output"))
        content = script if script is not None else _json({k: output.get(k) for k in ("entry_point", "notes")})
        return (content, f"{fams} baseline script ({source}) for '{_short(paper, 60)}'",
                f"{source} baseline using {fams}; exit code {output.get('exit_code')}; "
                f"{output.get('notes') or ''} Gap to the paper: {gap if gap is not None else 'not measurable'}.")
    if agent == "Evaluation Agent":
        review = output.get("review") or {}
        keep = {"status": output.get("status"), "metrics": output.get("metrics"),
                "static_findings": review.get("static_findings"),
                "verified_findings": [f for f in review.get("findings") or [] if f.get("verified")],
                "decision": output.get("decision"), "likely_causes": output.get("likely_causes")}
        gap = _gap(output)
        return (_json(keep), f"Evaluation: {output.get('status')}, decision {(output.get('decision') or {}).get('action')}",
                f"Overall {output.get('status')}; gap {gap if gap is not None else 'not measurable'}; "
                f"{len(review.get('static_findings') or [])} leakage finding(s).")
    if agent == "Experiment Agent" and "what_changed" in output:
        # The earlier design ran one controlled experiment against the baseline.
        keep = {k: output.get(k) for k in ("category", "hypothesis", "what_changed", "baseline_metrics",
                                           "experiment_metrics", "comparison", "overall")}
        return (_json(keep), f"Experiment ({output.get('category')}): {_short(output.get('what_changed'), 120)}",
                f"Changed: {_short(output.get('what_changed'), 300)} Result: {output.get('overall')} against "
                f"the baseline. Hypothesis: {_short(output.get('hypothesis'), 200)}")
    if agent == "Experiment Agent":
        decision = output.get("decision") or {}
        keep = {k: output.get(k) for k in ("analysis", "hypotheses", "decision", "gap_points")}
        return (_json(keep), f"Experiment analysis: {decision.get('action')} {decision.get('target_agent') or ''}".strip(),
                f"{_short(output.get('analysis'), 300)} Decision: {decision.get('action')} "
                f"{decision.get('target_agent') or ''}.")
    return _json(output), f"{agent} output", f"Approved output of the {agent}."


def memory_id(rule_key: str, version: int = None) -> str:
    """The deterministic id of a memory under one rules version (version 1
    ids have no suffix, so memories stored before versions keep theirs)."""
    version = version or RULES_VERSION
    return str(uuid.uuid5(NAMESPACE, rule_key if version == 1 else f"{rule_key}|rules{version}"))


def _memory(run: dict, stage: dict | None, rule: str, **m) -> dict:
    key = f"{run['run_id']}|{rule}|{stage['id'] if stage else m.get('key', '')}|{m.get('key', '')}"
    feats = m.pop("feats")
    m["reasons"] = {**m["reasons"], "rules_version": RULES_VERSION}
    return {
        "id": memory_id(key), "kind": m["kind"], "agent": m["agent"], "run_id": run["run_id"],
        "stage_attempt": stage["attempt"] if stage else None,
        "invocation_id": stage.get("invocation_id") if stage else None,
        "polarity": m["polarity"], "title": _short(_portable(m["title"]), 200),
        "summary": _portable(m["summary"]), "content": _portable(m.get("content")),
        "task": feats, "tags": features.tags(feats), "outcome": _portable(m.get("outcome") or {}),
        "evidence": {"run_id": run["run_id"], "stage_id": stage["id"] if stage else None,
                     "invocation_id": stage.get("invocation_id") if stage else None, "rule": rule,
                     "rule_key": key, "rules_version": RULES_VERSION},
        "quality": m["quality"], "quality_reasons": m["reasons"], "trust": m["trust"],
        "lineage": m.get("lineage") or "unknown",
    }


def _failure_memories(run: dict, stage: dict, event_payload: dict) -> list:
    agent = stage["agent"]
    report = stage["report_text"] or ""
    cause, guidance = _diagnosis(report)
    error = event_payload.get("error")
    checks = event_payload.get("failed_checks") or []
    if not error and report.startswith(f"{agent} raised an exception"):
        # "<agent> raised an exception:\n<message or traceback>"
        error = report.split(DIAGNOSIS_MARK)[0].split("\n", 1)[-1].strip() or None
    output = stage["output"] if isinstance(stage["output"], dict) else {}
    if not error and not checks and output.get("status"):
        # Runs recorded before run recording: the agent's own result says what failed.
        warnings = [w for w in output.get("warnings") or [] if w]
        checks = [f"status {output['status']}" + (f": {warnings[0]}" if warnings else "")]
    # The signature comes only from a real error or failed check, never from
    # the LLM's diagnosis text (which would not match across runs).
    detail = error or (checks[0] if checks else None)
    signature = features.failure_signature(detail)
    detail = detail or cause
    following = _next_of(run, stage, agent)
    fixed = None if following is None else following["decision"] == "Approved"
    trust, trust_reason = _trust(run, agent)
    feats = features.task_features(run, agent, failure=detail)
    excerpt = report.split(DIAGNOSIS_MARK)[0][-MAX_EXCERPT_CHARS:]
    content = _json({"failed_checks": checks, "error": _short(error, 1500) if error else None,
                     "diagnosis": {"cause": cause, "guidance": guidance}, "report_excerpt": excerpt})
    quality, reasons = ranking.warning_usefulness(cause, fixed, signature, trust)
    memories = [_memory(run, stage, "failure", kind="failure", polarity="negative", agent=agent,
                        title=f"{agent} attempt failed: {_short(signature or cause or 'no cause recorded', 120)}",
                        summary=f"{stage['decision']} on attempt {stage['attempt']}. "
                                + (f"Cause: {_short(cause, 400)} " if cause else "")
                                + ("The next attempt passed." if fixed else
                                   "The next attempt also failed." if fixed is False else ""),
                        content=content, feats=feats, lineage="rejected", trust=trust, quality=quality,
                        reasons={**reasons, "trust_note": trust_reason},
                        outcome={"decision": stage["decision"], "fixed_by_next_attempt": fixed})]
    if cause and not cause.lower().startswith(("no cause", "master agent could not diagnose")):
        # A diagnosis that itself failed teaches nothing; the failure memory keeps the attempt.
        positive = fixed is True
        feats = features.task_features(run, "Master Agent", failure=detail)
        if positive:
            downstream = _downstream(run, agent)
            quality, reasons = ranking.base_quality((1.0, "the next attempt passed with this diagnosis"),
                                                    (1.0, "not applicable (a diagnosis)"),
                                                    (1.0, "not applicable (a diagnosis)"), downstream, trust)
        else:
            quality, reasons = ranking.warning_usefulness(cause, fixed, signature, trust)
        memories.append(_memory(run, stage, "diagnosis", kind="decision",
                                polarity="positive" if positive else "negative", agent="Master Agent",
                                title=f"Diagnosis of a {agent} failure: {_short(cause, 110)}",
                                summary=f"Cause: {_short(cause, 400)} Guidance: {_short('; '.join(guidance), 400)} "
                                        + ("The retry then passed." if positive else
                                           "The retry also failed." if fixed is False else "No retry followed."),
                                content=_json({"failed_agent": agent, "failure": _short(detail, 1500),
                                               "cause": cause, "guidance": guidance}),
                                feats=feats, lineage="kept" if positive else "rejected", trust=trust,
                                quality=quality, reasons=reasons,
                                outcome={"next_attempt_passed": fixed}))
    return memories


def _rerun_memories(run: dict) -> list:
    out = []
    for i, r in enumerate(run["reruns"] or []):
        if not isinstance(r, dict):
            continue
        sender = r.get("requested_by") or "Experiment Agent"
        target = r.get("target_agent") or "unknown agent"
        trust, trust_reason = _trust(run, sender)
        if r.get("applied") and r.get("improved"):
            polarity, value, effect = "positive", 1.0, "kept, and the score moved closer to the paper"
        elif r.get("applied"):
            polarity, value, effect = "neutral", 0.5, "kept, with no gain in the gap"
        else:
            polarity, value, effect = "negative", 0.0, f"not kept: {_short(r.get('outcome'), 200)}"
        feats = features.task_features(run, sender)
        signature = f"rerun:{target}:{r.get('action') or 'rerun'}"
        if polarity == "negative":
            quality, reasons = ranking.warning_usefulness(r.get("outcome"), None, signature, trust)
        else:
            quality, reasons = ranking.base_quality((value, effect), (1.0, "not applicable (a request)"),
                                                    (1.0, "no verified leakage recorded"), _downstream(run, sender),
                                                    trust)
        out.append(_memory(run, None, "rerun", key=str(r.get("rerun") or i), kind="decision", polarity=polarity,
                           agent=sender, title=f"Rerun of the {target}: {effect[:60]}",
                           summary=f"Requested by the {sender}: {_short('; '.join(r.get('instructions') or []), 400)} "
                                   f"Gap {r.get('gap_before')} -> {r.get('gap_after')} points; {effect}.",
                           content=_json({k: r.get(k) for k in ("target_agent", "action", "goal", "instructions",
                                                                "reason", "gap_before", "gap_after", "status_before",
                                                                "status_after", "applied", "improved", "outcome")}),
                           feats=feats, lineage="kept" if r.get("applied") else "reverted", trust=trust,
                           quality=quality, reasons={**reasons, "trust_note": trust_reason},
                           outcome={"applied": r.get("applied"), "improved": r.get("improved"),
                                    "gap_before": r.get("gap_before"), "gap_after": r.get("gap_after")}))
    return out


def _finding_memories(run: dict, stage: dict, lineage: str) -> list:
    review = (stage["output"] or {}).get("review") or {}
    findings = [dict(f, source="code_checks") for f in review.get("static_findings") or []]
    findings += [dict(f, source="analyst review") for f in review.get("findings") or [] if f.get("verified")]
    fixed = lineage in ("kept", "superseded") and any(
        r.get("requested_by") == "Evaluation Agent" and r.get("applied") for r in run["reruns"] or [])
    out = []
    trust, trust_reason = _trust(run, "Evaluation Agent")
    for i, f in enumerate(findings):
        outcome = (1.0, "the Evaluation Agent's fix was kept") if fixed else (0.5, "no kept fix recorded")
        quality, reasons = ranking.base_quality(outcome, (1.0, "not applicable (a finding)"),
                                                (1.0, "the finding itself is verified"),
                                                _downstream(run, "Evaluation Agent"), trust)
        feats = features.task_features(run, "Evaluation Agent")
        out.append(_memory(run, stage, "finding", key=str(i), kind="solution",
                           polarity="positive" if fixed else "neutral", agent="Evaluation Agent",
                           title=f"Verified finding ({f.get('category') or 'leakage'}): {_short(f.get('issue'), 120)}",
                           summary=f"{_short(f.get('issue'), 300)} Evidence: {_short(f.get('evidence'), 200)} "
                                   f"Fix: {_short(f.get('fix'), 200)}",
                           content=_json(f), feats=feats, lineage=lineage, trust=trust, quality=quality,
                           reasons={**reasons, "trust_note": trust_reason},
                           outcome={"fix_kept": fixed, "source": f["source"]}))
    return out


def build_memories(run: dict) -> tuple[list, str]:
    """(memories, how lineage was found) for one run -- nothing is written."""
    lineage, lineage_source = stage_lineage(run)
    payloads = _attempt_events(run)
    memories = []
    for stage in run["stages"]:
        lin = lineage.get(stage["id"]) or ("unknown", "no lineage recorded")
        if stage["decision"] == "Approved":
            script = _script(run, stage) if stage["agent"] == "Replication Agent" else None
            m = _approved_memory(run, stage, lin, script)
            if m:
                memories.append(m)
            if stage["agent"] == "Evaluation Agent":
                memories.extend(_finding_memories(run, stage, lin[0]))
        else:
            memories.extend(_failure_memories(run, stage, payloads.get(stage["invocation_id"]) or {}))
    memories.extend(_rerun_memories(run))
    return memories, lineage_source


# --- Writing ---

def _run_row(run: dict) -> dict:
    from src import memory
    from src.orchestration import run_context
    usage = memory.client().local.rows("memory_usage", run_id=run["run_id"])
    # The mode recorded at the run's start; a run retained after its process
    # ended (backfill) has none recorded.
    mode = run_context.info(run["run_id"]).get("memory_mode") or "unknown"
    return {"run_id": run["run_id"], "contributor": memory.MEMORY_CONTRIBUTOR or None, "topic": run["topic"],
            "status": run["status"], "final_phase": run["final_phase"], "git_commit": run["git_commit"] or None,
            "memory_mode": mode, "memories_recalled": len(usage), "memories_used": sum(bool(u.get("used")) for u in usage),
            "trust": "low_trust" if _review_paper(run) else "normal",
            "started_at": run["started_at"], "finished_at": run["finished_at"]}


def retain_run(run_id: str, dry_run: bool = False) -> dict:
    """Make and store the memories of one finished run (once). With dry_run,
    build them and report, writing nothing. With MEMORY_MODE "read" (a
    fixed bank for a controlled test) or "off", nothing is retained.
    Never raises."""
    from src import memory
    from src.config import MEMORY_MODE
    report = {"run_id": run_id, "skipped": None, "written": [], "duplicates": [], "lineage": ""}
    try:
        c = memory.client()
        if c is None or c.backend == "off":
            report["skipped"] = "memory is off"
            return report
        if MEMORY_MODE != "read_write" and not dry_run:
            report["skipped"] = f"MEMORY_MODE={MEMORY_MODE}: the memory bank is not changed"
            return report
        if c.local.rows("memory_runs", run_id=run_id):
            report["skipped"] = "already retained"
            return report
        run = load_run(run_id)
        if run is None:
            report["skipped"] = "no such run"
            return report
        if run["status"] is None:
            report["skipped"] = "the run may still be running"
            return report
        memories, report["lineage"] = build_memories(run)
        rows = [memory.build_memory(m) for m in memories]
        shas = sorted({r["content_sha"] for r in rows if r["content_sha"]})
        existing = {}
        # Duplicates are counted against memories still in use, not superseded or retired ones.
        for found in (c.fetch(content_shas=shas, states=("active", "candidate"), limit=10_000) if shas else []):
            if found.get("run_id") != run_id:
                existing.setdefault((found["agent"], found["content_sha"]), found)
        seen = set()
        for row in rows:
            key = (row["agent"], row["content_sha"])
            if row["content_sha"] and key in seen:
                continue  # the same content twice in one run is one memory
            seen.add(key)
            if row["content_sha"] and key in existing:
                report["duplicates"].append({"of": existing[key]["id"], "title": row["title"], "agent": row["agent"]})
                if not dry_run:
                    memory.increment(existing[key]["id"], seen_count=1)
                continue
            report["written"].append(row)
            if not dry_run:
                memory.put_memory(row)
        if not dry_run:
            c.write({"op_id": str(uuid.uuid4()), "type": "insert", "table": "memory_runs",
                     "rows": [_run_row(run)]})
            # Lessons from the new memories (one LLM call; candidates until confirmed).
            from src.memory import curator
            report["curator"] = curator.curate_run(run_id, report["written"])
    except Exception as e:
        report["skipped"] = f"error: {type(e).__name__}: {e}"
        print(f"[Memory] WARNING: could not retain run {run_id}: {e}")
    return report


def _same(old: dict, new: dict) -> bool:
    return all(old.get(k) == new.get(k) for k in ("kind", "polarity", "title", "summary", "content_sha")) and \
        round(old.get("quality") or 0, 4) == round(new.get("quality") or 0, 4)


def rescore_run(run_id: str, dry_run: bool = False) -> dict:
    """Apply the current RULES_VERSION to a run retained under older rules.
    Memories are never edited: a changed memory is written under the new
    version's id and the old one is set to "superseded" (superseded_by the
    new one); an old memory the rules no longer make is set to "retired".
    Unchanged memories stay as they are. Every state change has a reason.
    Uses the local copy of the run's memories (the machine that retained it)."""
    from src import memory
    report = {"run_id": run_id, "skipped": None, "new": [], "superseded": [], "retired": [], "unchanged": 0}
    try:
        c = memory.client()
        if c is None or c.backend == "off":
            report["skipped"] = "memory is off"
            return report
        if not c.local.rows("memory_runs", run_id=run_id):
            report["skipped"] = "not retained yet (use retain)"
            return report
        run = load_run(run_id)
        if run is None:
            report["skipped"] = "no such run"
            return report
        existing = {m["id"]: m for m in c.local.rows("memories", run_id=run_id)}
        memories, _ = build_memories(run)
        matched, seen = set(), set()
        reason = f"rescored with retain rules version {RULES_VERSION}"
        for row in (memory.build_memory(m) for m in memories):
            if (row["agent"], row["content_sha"]) in seen and row["content_sha"]:
                continue
            seen.add((row["agent"], row["content_sha"]))
            if row["id"] in existing:  # already under the current version
                matched.add(row["id"])
                report["unchanged"] += 1
                continue
            older = [existing[i] for i in (memory_id(row["evidence"]["rule_key"], v) for v in range(RULES_VERSION - 1, 0, -1))
                     if i in existing]
            old = older[0] if older else None
            if old:
                matched.add(old["id"])
                if old["state"] in ("active", "candidate") and _same(old, row):
                    report["unchanged"] += 1
                    continue
            report["new"].append(row)
            if old and old["state"] in ("active", "candidate"):
                report["superseded"].append({"old": old, "new": row})
            if not dry_run:
                memory.put_memory(row)
                if old and old["state"] in ("active", "candidate"):
                    memory.set_state(old["id"], "superseded", reason, superseded_by=row["id"])
        for old in existing.values():
            if old["id"] not in matched and old["state"] in ("active", "candidate"):
                report["retired"].append(old)
                if not dry_run:
                    memory.set_state(old["id"], "retired", f"{reason}: these rules no longer make this memory")
    except Exception as e:
        report["skipped"] = f"error: {type(e).__name__}: {e}"
        print(f"[Memory] WARNING: could not rescore run {run_id}: {e}")
    return report


def format_rescore_report(reports: list, dry_run: bool) -> str:
    lines = [f"# Memory rescore report, rules version {RULES_VERSION} "
             f"({'dry run: nothing written' if dry_run else 'written'})", "", f"Made {_now()}.", ""]
    for r in reports:
        lines.append(f"## Run {r['run_id']}")
        if r["skipped"]:
            lines += [f"Skipped: {r['skipped']}", ""]
            continue
        lines.append(f"{r['unchanged']} unchanged, {len(r['superseded'])} replaced, "
                     f"{len(r['new']) - len(r['superseded'])} new, {len(r['retired'])} retired.")
        for s in r["superseded"]:
            o, n = s["old"], s["new"]
            lines.append(f"- {n['agent']}: {o['kind']}/{o['polarity']} {o['quality']:.2f} -> "
                         f"{n['kind']}/{n['polarity']} {n['quality']:.2f}: {n['title']}")
            if n["quality_reasons"].get("outcome"):
                lines.append(f"    - outcome: {n['quality_reasons']['outcome']['reason']}")
        for n in r["new"]:
            if not any(s["new"] is n for s in r["superseded"]):
                lines.append(f"- new {n['agent']}: {n['kind']}/{n['polarity']} {n['quality']:.2f}: {n['title']}")
        for o in r["retired"]:
            lines.append(f"- retired {o['agent']}: {o['title']}")
        lines.append("")
    return "\n".join(lines)


def backfill(dry_run: bool = False) -> list:
    """Retain every recorded run that has not been retained yet."""
    with run_store._connect() as conn:
        run_ids = [r[0] for r in conn.execute("SELECT run_id FROM runs ORDER BY started_at")]
    return [retain_run(run_id, dry_run=dry_run) for run_id in run_ids]


def format_report(reports: list, dry_run: bool) -> str:
    """A readable report of what retain made (plan 10: the backfill report)."""
    lines = [f"# Memory backfill report ({'dry run: nothing written' if dry_run else 'written'})", "",
             f"Made {_now()}. Scores are base quality (plan 9.1); every component's reason is listed.", ""]
    for r in reports:
        lines.append(f"## Run {r['run_id']}")
        if r["skipped"]:
            lines += [f"Skipped: {r['skipped']}", ""]
            continue
        lines.append(f"Lineage: {r['lineage']}. {len(r['written'])} memories, {len(r['duplicates'])} duplicate(s) "
                     "of memories from other runs (counted, not stored again).")
        lines.append("")
        for m in sorted(r["written"], key=lambda m: (m["agent"], -m["quality"])):
            lines.append(f"- **{m['quality']:.2f}** [{m['kind']}/{m['polarity']}, lineage {m['lineage']}, "
                         f"{m['trust']}] {m['agent']}: {m['title']}")
            for comp, info in m["quality_reasons"].items():
                if isinstance(info, dict) and "value" in info:
                    lines.append(f"    - {comp} = {info['value']}: {info['reason']}")
        for d in r["duplicates"]:
            lines.append(f"- duplicate ({d['agent']}): {d['title']} -> seen again in memory {d['of']}")
        lines.append("")
    return "\n".join(lines)
