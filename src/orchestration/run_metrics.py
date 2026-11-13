"""
Run metrics: every pipeline run's conditions and measurements, so runs can
be compared with each other -- above all runs with and without cross-run
memory (docs/CLAUDE_BACKEND_PLAN.md, section 11; milestone M1).

run_store.finish_run calls compute(run_id) when a run ends (success,
halted or crashed); `backfill` measures runs recorded before this module
existed. The rules:

  * Every value is calculated in code from the run's own records: the run
    database (runs, stages, artifacts), the LLM / search / page-fetch call
    logs and the Master Agent log. No LLM calculates or estimates a metric.
  * A value that cannot be measured is stored as NULL with a reason
    (missing_reason) -- never as 0, and never guessed.
  * Written once: a run's metrics are not changed. A changed calculation
    gets a new METRICS_VERSION, and the rows of the old version stay.
  * A log line belongs to a run by its run_id; a line written before run
    ids were logged is matched by the run's time span (a machine runs one
    pipeline run at a time). run_conditions.log_matching says which.

CLI:
  python -m src.orchestration.run_metrics compute <run_id>
  python -m src.orchestration.run_metrics backfill
  python -m src.orchestration.run_metrics show <run_id>
  python -m src.orchestration.run_metrics compare [--topic T] [--parent-paper P] [--since YYYY-MM-DD]
      [--contributor C] [--group-by FIELD] [--metrics a,b,...] [--all-tasks] [--out FILE.json|FILE.csv]
"""
import argparse
import ast
import csv
import itertools
import json
import random
import re
import sqlite3
import statistics
import sys
from datetime import datetime, timedelta, timezone
from math import comb
from pathlib import Path

from src.config import MEMORY_BACKEND, MEMORY_CONTRIBUTOR
from src.orchestration import run_context, run_store
from src.orchestration.handoff import AGENT_PHASE

# 3: memory use is measured (agents recall from milestone M5): memory mode,
# actual exposure, cases shown and used, requests, prompt size, memories made.
METRICS_VERSION = 3

# A run still marked "running" with no recorded activity for this long is
# measured as crashed (its process ended without closing the record).
STALE_AFTER = timedelta(hours=6)

# Runs measured under metrics versions 1-2 had no cross-run memory:
# "pre_memory" keeps them apart from the "off" control group of a memory
# comparison (plan section 11.7, rule 8). A run measured after its process
# ended has no recorded memory settings: "unknown", never a control.
PRE_MEMORY = "pre_memory"
NOT_YET = {"memories_helped": "use-based learning is milestone M6", "memories_hurt": "use-based learning is milestone M6",
           "fast_mode_used": "fast mode is milestone M6", "tuning_actions": "pipeline tuning is milestone M8"}

# The workflow states, in order; final_phase_reached is the 1-based index.
PHASE_ORDER = list(dict.fromkeys(AGENT_PHASE.values()))

# name -> (unit, direction, description). A name ending in "." is a family:
# one metric per evaluation metric or search provider (final_gap.accuracy).
METRICS = {
    # Result quality
    "run_status": ("status", "neutral", "success / halted / crashed"),
    "final_phase_reached": ("workflow index", "higher_better",
                           "furthest workflow state entered (1 = TOPIC_ANALYSIS ... 9 = EXPERIMENTATION)"),
    "halted_at_agent": ("agent", "neutral", "the agent whose stage halted the run"),
    "reached_measurable_gap": ("0/1", "higher_better",
                               "an evaluation compared at least one metric stated by both the paper and the run"),
    "evaluation_status": ("status", "neutral", "EXACT / APPROXIMATE / FAILED / INCONCLUSIVE of the kept evaluation"),
    "final_gap_points": ("percentage points", "lower_better", "mean gap to the paper, kept evaluation"),
    "baseline_gap_points": ("percentage points", "lower_better", "mean gap to the paper, first evaluation"),
    "gap_closed_points": ("percentage points", "higher_better", "baseline_gap_points - final_gap_points"),
    "final_gap.": ("percentage points", "lower_better", "gap of one matched metric, kept evaluation"),
    "reproduced.": ("as printed", "neutral", "value printed by the run, kept evaluation"),
    "published.": ("as stated", "neutral", "value stated by the Parent Paper, kept evaluation"),
    "leakage_findings_first": ("count", "lower_better", "code_checks.py leakage findings, first evaluated baseline"),
    "leakage_findings_final": ("count", "lower_better", "code_checks.py leakage findings, kept baseline"),
    "parent_dataset_found": ("0/1", "higher_better", "the Parent Paper's own dataset was obtained"),
    # Reliability
    "attempts": ("count", "lower_better", "recorded agent attempts"),
    "first_attempt_approved": ("0/1", "higher_better", "the agent's first attempt passed its checks"),
    "first_attempt_approval_rate": ("share", "higher_better", "share of agents approved on their first attempt"),
    "retries": ("count", "lower_better", "attempts that failed and were retried"),
    "rejections": ("count", "lower_better", "attempts rejected after the last allowed attempt"),
    "diagnoses": ("count", "lower_better", "Master Agent diagnoses of failed attempts"),
    "exceptions": ("count", "lower_better", "attempts that raised an exception"),
    "repair_attempts": ("count", "lower_better", "Replication Agent script repairs"),
    "reruns_requested": ("count", "neutral", "rerun requests: opened plus declined"),
    "reruns_declined": ("count", "neutral", "requests Master Agent declined"),
    "reruns_kept": ("count", "neutral", "reruns whose new baseline was kept"),
    "reruns_undone": ("count", "neutral", "reruns undone (failed, declined or reverted)"),
    "reruns_reverted": ("count", "neutral", "reruns reverted because the score moved away from the paper"),
    "reruns_improved": ("count", "higher_better", "kept reruns that moved the score closer to the paper"),
    "rerun_end_reason": ("text", "neutral", "why Master Agent ended the rerun loop"),
    # Cost and time
    "run_seconds": ("seconds", "lower_better", "start to finish"),
    "agent_seconds": ("seconds", "lower_better", "time attributed to the agent's attempts (see timing_method)"),
    "agent_seconds_estimate": ("seconds", "lower_better", "legacy completion-gap estimate; not measured execution time"),
    "llm_calls": ("count", "lower_better", "successful LLM calls"),
    "llm_failed_calls": ("count", "lower_better", "failed LLM call attempts (e.g. a model that fell back)"),
    "llm_calls_without_usage": ("count", "neutral", "successful calls whose provider reported no token usage"),
    "llm_input_tokens": ("tokens", "lower_better", "input tokens of calls that reported usage"),
    "llm_output_tokens": ("tokens", "lower_better", "output tokens of calls that reported usage"),
    "llm_seconds": ("seconds", "lower_better", "time spent in LLM calls"),
    "search_calls": ("count", "lower_better", "successful web searches"),
    "search_failed_calls": ("count", "lower_better", "failed web search attempts"),
    "search_calls.": ("count", "lower_better", "successful web searches of one provider"),
    "page_fetches": ("count", "lower_better", "result links opened"),
    "browser_fetches": ("count", "lower_better", "result links opened with the local browser"),
    # Memory use
    "memory_exposure": ("none/partial/full", "neutral", "memory actually supplied to the agents"),
    "memory_exposure_reason": ("text", "neutral", "why no memory was supplied"),
    "recall_errors": ("count", "lower_better", "recalls that failed (store unreachable)"),
    "recall_empty": ("count", "neutral", "recalls that found no matching past case"),
    "memories_recalled": ("count", "neutral", "memories shown to agents"),
    "memories_used": ("count", "neutral", "memories the agents reported using (self-report)"),
    "recall_precision": ("share", "higher_better", "memories used / memories shown"),
    "memory_requests": ("count", "neutral", "memory requests made by the model"),
    "memories_helped": ("count", "neutral", "used memories followed by a good outcome"),
    "memories_hurt": ("count", "neutral", "used memories followed by a bad outcome"),
    "fast_mode_used": ("0/1", "neutral", "a past solution was adapted (fast mode)"),
    "tuning_actions": ("count", "neutral", "pipeline choices changed by memory"),
    "memory_prompt_chars": ("characters", "lower_better", "memory text added to prompts"),
    "memories_retained": ("count", "neutral", "new memories made from this run"),
}
MEMORY_METRICS = ("memories_recalled", "memories_used", "recall_precision", "memory_requests", "memories_helped",
                  "memories_hurt", "fast_mode_used", "tuning_actions", "memory_prompt_chars", "memories_retained")

# Estimator names and modules -> model family (plan section 5.2 vocabulary).
MODEL_FAMILIES = {
    "xgboost": ("xgboost", "XGBClassifier", "XGBRegressor", "XGBRFClassifier"),
    "lightgbm": ("lightgbm", "LGBMClassifier", "LGBMRegressor"),
    "catboost": ("catboost", "CatBoostClassifier", "CatBoostRegressor"),
    "random_forest": ("RandomForestClassifier", "RandomForestRegressor"),
    "extra_trees": ("ExtraTreesClassifier", "ExtraTreesRegressor"),
    "gradient_boosting": ("GradientBoostingClassifier", "GradientBoostingRegressor",
                          "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"),
    "adaboost": ("AdaBoostClassifier", "AdaBoostRegressor"),
    "decision_tree": ("DecisionTreeClassifier", "DecisionTreeRegressor"),
    "logistic_regression": ("LogisticRegression", "LogisticRegressionCV"),
    "linear_model": ("LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier", "SGDRegressor"),
    "svm": ("SVC", "LinearSVC", "SVR", "OneClassSVM"),
    "knn": ("KNeighborsClassifier", "KNeighborsRegressor"),
    "naive_bayes": ("GaussianNB", "MultinomialNB", "BernoulliNB"),
    "mlp": ("MLPClassifier", "MLPRegressor"),
    "isolation_forest": ("IsolationForest",),
    "local_outlier_factor": ("LocalOutlierFactor",),
    "neural_network": ("torch", "tensorflow", "keras"),
}
# random_state=42, RANDOM_STATE = 42, SEED = 5, random_seed=1, np.random.seed(3)
_SEED_PATTERNS = (re.compile(r"\b\w*(?:random_state|seed)\w*\s*=\s*(\d+)", re.I),
                  re.compile(r"\bseed\s*\(\s*(\d+)", re.I))
_REPAIR_NOTE = re.compile(r"Repair attempts:\s*(\d+)")

# run_conditions columns compare can group by, and the ones it checks for
# differences between groups (possible confounders).
GROUP_FIELDS = ("memory_mode", "memory_exposure", "llm_provider", "git_commit", "code_source", "contributor",
                "comparison_group", "status")
CONFOUNDER_FIELDS = ("git_commit", "git_dirty", "llm_provider", "top_served_model", "rows_used", "sampled",
                     "dataset_identities", "code_source", "model_family", "memory_mode", "memory_exposure",
                     "memory_bank_size", "contributor", "log_matching", "timing_method")
DEFAULT_COMPARE_METRICS = (
    "final_phase_reached", "reached_measurable_gap", "final_gap_points", "baseline_gap_points", "gap_closed_points",
    "leakage_findings_final", "attempts", "retries", "exceptions", "first_attempt_approval_rate", "reruns_kept",
    "reruns_reverted", "run_seconds", "llm_calls", "llm_failed_calls", "llm_input_tokens", "llm_output_tokens",
    "search_calls", "page_fetches",
)
MIN_GROUP_RUNS = 3          # fewer runs in a group: "too few runs", no test
PERMUTATIONS = 20000        # random permutations when exact enumeration is larger


# --- Small helpers ---

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ts(value) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _decode(text):
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


def topic_key(topic: str) -> str:
    """'Machine learning for detecting credit-card fraud' and 'credit card
    fraud detecting, machine learning for' -> the same key."""
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", (topic or "").lower()))))


def metric_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "unnamed"


def _agent_label(name) -> str | None:
    """'code_discovery_agent' (the LLM log's agent field) -> 'Code Discovery Agent' (a stage name)."""
    words = re.sub(r"[_\-]+", " ", str(name or "")).split()
    return " ".join(w.capitalize() for w in words) or None


def model_families(script: str) -> list:
    """Model families used by a script, from its identifiers and imports
    (Python ast; regex word match when the script does not parse)."""
    try:
        names = set()
        for node in ast.walk(ast.parse(script or "")):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.update(node.name.split("."))
                if node.asname:
                    names.add(node.asname)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
    except SyntaxError:
        names = set(re.findall(r"\w+", script or ""))
    return sorted(family for family, markers in MODEL_FAMILIES.items() if names & set(markers))


def script_seeds(script: str) -> list:
    return sorted({int(v) for pattern in _SEED_PATTERNS for v in pattern.findall(script or "")})


def _unit(metric: str) -> str:
    if metric in METRICS:
        return METRICS[metric][0]
    family = metric.split(".", 1)[0] + "."
    return METRICS[family][0] if family in METRICS else ""


class _Sheet:
    """The metric rows of one run: (scope, metric) -> value or missing reason."""

    def __init__(self):
        self.rows = {}

    def num(self, metric: str, value, scope: str = "run", reason: str = None) -> None:
        if value is None and not reason:
            raise ValueError(f"{metric}: a missing value needs a reason")
        number = None if value is None else float(value)
        self.rows[(scope, metric)] = (number, None, _unit(metric), None if value is not None else reason)

    def text(self, metric: str, value, scope: str = "run", reason: str = None) -> None:
        if value in (None, "") and not reason:
            raise ValueError(f"{metric}: a missing value needs a reason")
        present = value not in (None, "")
        self.rows[(scope, metric)] = (None, str(value) if present else None, _unit(metric),
                                      None if present else reason)


# --- Log matching ---

def _read_jsonl(path: Path) -> list | None:
    """The log's entries, or None when the file does not exist."""
    if not path or not Path(path).exists():
        return None
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _lines_for_run(entries: list | None, run_id: str, start: datetime, end: datetime) -> tuple[list | None, str]:
    """(the run's log lines, how they were matched). None when the log did
    not exist yet (no line written before the run ended) -- its counts are
    then unknown, not 0."""
    if entries is None:
        return None, "log not found"
    tagged = [e for e in entries if e.get("run_id") == run_id]
    if tagged:
        return tagged, "run_id"
    stamps = [(e, _ts(e.get("timestamp"))) for e in entries]
    if not any(stamp and stamp <= end for _, stamp in stamps):
        return None, "log not kept yet"
    return [e for e, stamp in stamps if "run_id" not in e and stamp and start <= stamp <= end], "time_span"


def _log_sources() -> dict:
    """The log files, read from their own modules so tests that redirect
    them (tests/conftest.py) are honored."""
    from src import llm
    from src.agents import master_agent
    from src.search import page_fetch, web_search
    return {"llm": llm.CALL_LOG, "search": web_search.SEARCH_LOG, "page_fetch": page_fetch.FETCH_LOG,
            "master_log": master_agent.LOG_FILE}


def _read_master_log(path: Path) -> list | None:
    if not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else None


# --- Measuring one run ---

def _status(run: dict, stages: list, timings: list | None = None) -> tuple[str | None, list]:
    """The run's status for measurement, or None while it may still be running."""
    if run["status"] != "running":
        return run["status"], []
    if run_context.current_run_id() == run["run_id"]:
        return None, []
    activity = [_ts(s["recorded_at"]) for s in stages]
    activity += [_ts(t.get("finished_at") or t["started_at"]) for t in timings or []]
    last = max([_ts(run["started_at"])] + [t for t in activity if t])
    if datetime.now(timezone.utc) - last < STALE_AFTER:
        return None, []
    return "crashed", [f"Recorded as running, with no activity after {last.isoformat()}; measured as crashed."]


def _baseline_script(stages: list, replication: dict | None) -> str | None:
    """The kept baseline script's text, from the run database's stored copy."""
    if not replication:
        return None
    wanted = json.dumps(replication, sort_keys=True, default=str)
    candidates = [s for s in stages if s["agent"] == "Replication Agent" and s["decision"] == "Approved"]
    stage = next((s for s in reversed(candidates)
                  if json.dumps(_decode(s["output_json"]), sort_keys=True, default=str) == wanted),
                 candidates[-1] if candidates else None)
    if not stage:
        return None
    with run_store._connect() as conn:
        artifacts = conn.execute("SELECT id, original_path FROM artifacts WHERE stage_id = ? AND kind = 'script' "
                                 "ORDER BY id", (stage["id"],)).fetchall()
    entry = replication.get("entry_point") or ""
    chosen = next((a for a in artifacts if Path(a["original_path"]).name == entry), artifacts[0] if artifacts else None)
    content = run_store.read_artifact(chosen["id"]) if chosen else None
    return content.decode("utf-8", errors="replace") if content is not None else None


def _measure(run: dict, status: str, stages: list, next_start: str | None, notes: list,
             timings: list | None = None) -> dict:
    from src.agents.experiment_agent import score_gap

    run_id = run["run_id"]
    sheet = _Sheet()
    result = _decode(run["result_json"])
    result = result if isinstance(result, dict) else None
    start = _ts(run["started_at"])
    outputs = [(s, _decode(s["output_json"])) for s in stages]

    def approved(agent: str) -> list:
        return [o for s, o in outputs if s["agent"] == agent and s["decision"] == "Approved" and isinstance(o, dict)]

    def kept(key: str, agent: str):
        if result is not None:
            value = result.get(key)
            return value if isinstance(value, dict) else None
        found = approved(agent)
        return found[-1] if found else None

    if result is None and any(s["decision"] == "Approved" for s in stages):
        notes.append("No final result was recorded (crash): data and code conditions come from the last approved "
                     "attempt of each agent, which may belong to a rerun that was not kept.")

    # End of the run's time span.
    activity = [_ts(s["recorded_at"]) for s in stages]
    activity += [_ts(t.get("finished_at") or t["started_at"]) for t in timings or []]
    last_activity = max([start] + [t for t in activity if t])
    if run["finished_at"]:
        end = _ts(run["finished_at"])
    else:
        end = _ts(next_start) or datetime.now(timezone.utc)
        notes.append("The run has no finish time; its log lines are matched up to "
                     + ("the next run's start." if next_start else "the time of measurement."))

    validation = kept("validation_result", "Validation Agent")
    dataset = kept("dataset_result", "Data Agent")
    cleaning = kept("cleaning_result", "Cleaning Agent")
    replication = kept("replication_result", "Replication Agent")
    supplied = (approved("Parent Paper Intake") or [{}])[-1]
    parent = (validation or {}).get("recommended_parent_paper") or supplied.get("parent_paper") or {}
    input_fingerprint = (result or {}).get("input_fingerprint") or supplied.get("input_fingerprint")
    script = _baseline_script(stages, replication)

    # --- Logs ---
    sources = _log_sources()
    matching = {}
    lines = {}
    for name in ("llm", "search", "page_fetch"):
        lines[name], matching[name] = _lines_for_run(_read_jsonl(sources[name]), run_id, start, end)
    master_entries = _read_master_log(sources["master_log"])
    if master_entries is None:
        lines["master_log"], matching["master_log"] = None, "log not found"
    else:
        lines["master_log"], matching["master_log"] = _lines_for_run(master_entries, run_id, start, end)

    llm_ok = [e for e in lines["llm"] or [] if not e.get("error")]
    served = {}
    for e in llm_ok:
        served[e.get("served_by") or "unknown"] = served.get(e.get("served_by") or "unknown", 0) + 1

    info = run_context.info(run_id)
    datasets = [d.get("name") for d in (dataset or {}).get("selected_datasets") or [] if d.get("name")]
    sampled = None
    if cleaning is not None:
        sampled = int(bool(cleaning.get("sampled")) or any(d.get("sampled") for d in
                                                              (dataset or {}).get("selected_datasets") or []))
    conditions = {
        "run_id": run_id,
        "metrics_version": METRICS_VERSION,
        "status": status,
        "contributor": MEMORY_CONTRIBUTOR or None,
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
        "topic": run["topic"],
        "topic_key": topic_key(run["topic"]),
        "parent_paper_doi": run["parent_paper_doi"] or parent.get("doi") or None,
        "parent_paper_title": run["parent_paper_title"] or parent.get("title") or None,
        "dataset_identities": json.dumps(datasets) if dataset is not None else None,
        "rows_used": (cleaning or {}).get("rows_total"),
        "sampled": sampled,
        "code_source": (replication or {}).get("source") or None,
        "model_family": json.dumps(model_families(script)) if script is not None else None,
        "git_commit": run["project_git_commit"] or None,
        "git_dirty": None if info.get("git_dirty") is None else int(info["git_dirty"]),
        "llm_provider": run["llm_provider"] or None,
        "served_models": json.dumps(served) if lines["llm"] is not None else None,
        # Recorded at the run's start (run_store._memory_snapshot); unknown if measured later.
        "memory_mode": info.get("memory_mode") or "unknown",
        "memory_backend": info.get("memory_backend") or (MEMORY_BACKEND if info else None),
        "memory_reuse": None,    # fast mode: milestone M6
        "memory_tuning": None,   # pipeline tuning: milestone M8
        "memory_curator": None,  # lessons: milestone M6
        "memory_exposure": None,  # set below from the run's recorded memory use
        "memory_bank_size": json.dumps(info["memory_bank_size"]) if "memory_bank_size" in info else None,
        "memory_bank_snapshot": info.get("memory_snapshot"),
        "comparison_group": f"parent_pdf:{input_fingerprint}" if input_fingerprint else None,
        "seeds": json.dumps({"split_seed": (cleaning or {}).get("split_seed"),
                             "script_seeds": script_seeds(script) if script is not None else None}),
        "python_version": info.get("python_version"),
        "os": info.get("os"),
        "log_matching": json.dumps(matching),
        "timing_method": "invocation_monotonic" if timings else "legacy_completion_gaps",
        "notes": None,
        "computed_at": _now(),
    }
    if not info:
        notes.append("Measured after the run's process ended: git dirty state, Python version and OS are unknown.")
    if any(v == "time_span" for v in matching.values()):
        notes.append("Some log lines have no run id and were matched by the run's time span.")

    # --- Result quality ---
    sheet.text("run_status", status)
    phase = result.get("phase") if result is not None else None
    if not phase:
        entered = [AGENT_PHASE[s["agent"]] for s in stages if s["agent"] in AGENT_PHASE]
        phase = max(entered, key=PHASE_ORDER.index) if entered else None
    sheet.num("final_phase_reached", PHASE_ORDER.index(phase) + 1 if phase in PHASE_ORDER else None,
              reason="no workflow state was entered")

    if status == "halted":
        rejected = [s["agent"] for s in stages if s["decision"] == "Rejected"]
        blockers = (result or {}).get("blockers") or []
        agent = rejected[-1] if rejected else (blockers[0].get("target") if blockers else None)
        sheet.text("halted_at_agent", agent, reason="halted without a recorded rejected stage")
    else:
        sheet.text("halted_at_agent", None, reason=f"the run did not halt ({status})")

    evaluations = approved("Evaluation Agent")
    first_eval = evaluations[0] if evaluations else None
    final_eval = result.get("evaluation_result") if result is not None else None
    final_eval = final_eval if isinstance(final_eval, dict) else None
    if result is None:
        final_reason = "the run crashed before its result was recorded; the kept final evaluation is unknown"
    else:
        final_reason = "no evaluation was reached"

    sheet.num("reached_measurable_gap", int(any(score_gap(e) is not None for e in evaluations)))
    sheet.text("evaluation_status", (final_eval or {}).get("status"), reason=final_reason)
    final_gap = score_gap(final_eval) if final_eval else None
    baseline_gap = score_gap(first_eval) if first_eval else None
    no_match = "the evaluation matched no metric stated by both the paper and the run"
    sheet.num("final_gap_points", final_gap, reason=no_match if final_eval else final_reason)
    sheet.num("baseline_gap_points", baseline_gap, reason=no_match if first_eval else "no evaluation was reached")
    sheet.num("gap_closed_points", None if final_gap is None or baseline_gap is None else baseline_gap - final_gap,
              reason="needs both the first and the kept evaluation's gap")
    for m in (final_eval or {}).get("metrics") or []:
        key = metric_key(m.get("metric"))
        for family, field in (("final_gap.", "difference_points"), ("reproduced.", "reproduced_value"),
                              ("published.", "published_value")):
            if m.get(field) is not None and ("run", family + key) not in sheet.rows:
                sheet.num(family + key, m[field])

    def leakage(evaluation, label):
        if not evaluation:
            return None, final_reason if label == "final" else "no evaluation was reached"
        findings = (evaluation.get("review") or {}).get("static_findings")
        if findings is None:
            return None, "the evaluation has no leakage-check record (older code version)"
        return len(findings), None

    for label, evaluation in (("first", first_eval), ("final", final_eval)):
        value, reason = leakage(evaluation, label)
        sheet.num(f"leakage_findings_{label}", value, reason=reason)
    found = (dataset or {}).get("parent_dataset_found")
    sheet.num("parent_dataset_found", None if found is None else int(bool(found)),
              reason="the Data Agent did not run" if dataset is None else "not recorded by this code version")

    # --- Reliability ---
    by_agent = {}
    for s in stages:
        by_agent.setdefault(s["agent"], []).append(s)
    totals = {"attempts": 0, "retries": 0, "rejections": 0, "diagnoses": 0, "exceptions": 0}
    firsts = []
    for agent, rows in by_agent.items():
        counts = {
            "attempts": len(rows),
            "retries": sum(r["decision"] == "Retrying" for r in rows),
            "rejections": sum(r["decision"] == "Rejected" for r in rows),
            "diagnoses": sum("--- Master Agent diagnosis ---" in (r["report_text"] or "") for r in rows),
            "exceptions": sum((r["report_text"] or "").startswith(f"{agent} raised an exception") for r in rows),
        }
        for metric, value in counts.items():
            sheet.num(metric, value, scope=agent)
            totals[metric] += value
        firsts.append(int(rows[0]["decision"] == "Approved"))
        sheet.num("first_attempt_approved", firsts[-1], scope=agent)
    for metric, value in totals.items():
        sheet.num(metric, value)
    sheet.num("first_attempt_approval_rate", sum(firsts) / len(firsts) if firsts else None,
              reason="no agent attempt was recorded")

    repairs = [int(m.group(1)) for s, o in outputs if s["agent"] == "Replication Agent" and isinstance(o, dict)
               for m in [_REPAIR_NOTE.search(str(o.get("notes") or ""))] if m]
    if "Replication Agent" in by_agent:
        sheet.num("repair_attempts", sum(repairs) if repairs else None, scope="Replication Agent",
                  reason="no repair count recorded (cloned repository, or older code)")

    rerun_metrics = ("reruns_requested", "reruns_declined", "reruns_kept", "reruns_undone", "reruns_reverted",
                     "reruns_improved")
    if result is None or "reruns" not in result:
        reason = ("the run crashed before its rerun history was recorded" if result is None
                  else "rerun history is not recorded by this code version")
        for metric in rerun_metrics:
            sheet.num(metric, None, reason=reason)
        sheet.text("rerun_end_reason", None, reason=reason)
    else:
        history = [r for r in result.get("reruns") or [] if isinstance(r, dict)]
        declined = None
        if lines["master_log"] is not None:
            declined = sum(1 for e in lines["master_log"] if str(e.get("decision", "")).startswith("Rejected")
                           and "declined" in str(e.get("issue", "")))
        sheet.num("reruns_declined", declined, reason="the Master Agent log was not found")
        sheet.num("reruns_requested", None if declined is None else len(history) + declined,
                  reason="needs the declined count")
        sheet.num("reruns_kept", sum(bool(r.get("applied")) for r in history))
        sheet.num("reruns_undone", sum(not r.get("applied") for r in history))
        sheet.num("reruns_reverted", sum(not r.get("applied") and "moved further from the paper" in
                                         str(r.get("outcome", "")) for r in history))
        sheet.num("reruns_improved", sum(bool(r.get("improved")) for r in history))
        sheet.text("rerun_end_reason", result.get("rerun_end_reason"), reason="no rerun loop ran")

    # --- Cost and time ---
    if run["finished_at"]:
        sheet.num("run_seconds", (end - start).total_seconds())
    else:
        sheet.num("run_seconds", (last_activity - start).total_seconds())
        notes.append("run_seconds is measured to the last recorded stage (the run has no finish time).")
    if timings:
        executions = {}
        for timing in timings:
            executions.setdefault(timing["agent"], {})[timing["attempt"]] = timing
        for agent in sorted(set(by_agent) | set(executions)):
            actual = executions.get(agent, {})
            expected = {s["attempt"] for s in by_agent.get(agent, [])} | set(actual)
            complete = bool(expected) and all(
                actual.get(attempt, {}).get("finished_at") is not None
                and actual.get(attempt, {}).get("duration_seconds") is not None for attempt in expected)
            sheet.num("agent_seconds", sum(t["duration_seconds"] for t in actual.values()) if complete else None,
                      scope=agent, reason="one or more invocations have no complete execution timer")
            if not complete:
                conditions["timing_method"] = "invocation_partial"
                notes.append(f"{agent}: incomplete execution timing; duration is unknown, not estimated from peers.")
    else:
        notes.append("Legacy run: agent execution timers were not recorded. Completion-gap estimates are separate "
                     "from agent_seconds and cannot establish individual durations when work overlaps.")
        previous = start
        estimates = {}
        for s in sorted(stages, key=lambda s: s["recorded_at"]):
            stamp = _ts(s["recorded_at"]) or previous
            estimates[s["agent"]] = estimates.get(s["agent"], 0.0) + max(0.0, (stamp - previous).total_seconds())
            previous = stamp
        for agent, value in estimates.items():
            sheet.num("agent_seconds", None, scope=agent, reason="legacy run has no recorded execution timer")
            sheet.num("agent_seconds_estimate", value, scope=agent)

    def usage(entry, *keys):
        u = entry.get("usage") or {}
        return next((u[k] for k in keys if isinstance(u.get(k), (int, float))), None)

    def llm_metrics(entries, scope):
        ok = [e for e in entries if not e.get("error")]
        sheet.num("llm_calls", len(ok), scope=scope)
        sheet.num("llm_failed_calls", len(entries) - len(ok), scope=scope)
        sheet.num("llm_seconds", sum(float(e.get("duration_seconds") or 0) for e in entries), scope=scope)
        inputs = [usage(e, "prompt_tokens", "input_tokens") for e in ok]
        outputs_ = [usage(e, "completion_tokens", "output_tokens") for e in ok]
        sheet.num("llm_calls_without_usage", sum(v is None for v in inputs), scope=scope)
        reason = "the provider reported no token usage for these calls" if ok else "no successful LLM call"
        known_in, known_out = [v for v in inputs if v is not None], [v for v in outputs_ if v is not None]
        sheet.num("llm_input_tokens", sum(known_in) if known_in else None, scope=scope, reason=reason)
        sheet.num("llm_output_tokens", sum(known_out) if known_out else None, scope=scope, reason=reason)

    if lines["llm"] is None:
        for metric in ("llm_calls", "llm_failed_calls", "llm_seconds", "llm_calls_without_usage",
                       "llm_input_tokens", "llm_output_tokens"):
            sheet.num(metric, None, reason=f"LLM call log: {matching['llm']}")
    else:
        llm_metrics(lines["llm"], "run")
        per_agent = {}
        for e in lines["llm"]:
            if _agent_label(e.get("agent")):
                per_agent.setdefault(_agent_label(e.get("agent")), []).append(e)
        for agent, entries in per_agent.items():
            llm_metrics(entries, agent)

    if lines["search"] is None:
        for metric in ("search_calls", "search_failed_calls"):
            sheet.num(metric, None, reason=f"search log: {matching['search']}")
    else:
        ok = [e for e in lines["search"] if not e.get("error")]
        sheet.num("search_calls", len(ok))
        sheet.num("search_failed_calls", len(lines["search"]) - len(ok))
        for provider in sorted({str(e.get("provider") or "unknown") for e in lines["search"]}):
            sheet.num("search_calls." + metric_key(provider),
                      sum(str(e.get("provider") or "unknown") == provider for e in ok))
    if lines["page_fetch"] is None:
        for metric in ("page_fetches", "browser_fetches"):
            sheet.num(metric, None, reason=f"page-fetch log: {matching['page_fetch']}")
    else:
        sheet.num("page_fetches", len(lines["page_fetch"]))
        sheet.num("browser_fetches", sum((e.get("provider") or "direct") != "direct" for e in lines["page_fetch"]))

    # --- Memory use ---
    _memory_metrics(sheet, conditions, run_id)

    conditions["notes"] = json.dumps(notes) if notes else None
    return {"conditions": conditions, "metrics": sheet.rows}


def _memory_metrics(sheet: "_Sheet", conditions: dict, run_id: str) -> None:
    """Memory use, from the run's own records in the memory store: the
    attempt events' memory summaries and the memory_usage rows. Exposure is
    what agents actually received, apart from the assigned memory_mode
    (plan 11.3): none, partial (only some recalls found cases) or full."""
    mode = conditions["memory_mode"]
    try:
        from src import memory
        store = memory.client().local if memory.client() else None
    except Exception:
        store = None
    if store is None:
        reason = "the memory store is off"
        conditions["memory_exposure"] = "none"
        sheet.text("memory_exposure", "none")
        for metric in MEMORY_METRICS:
            sheet.num(metric, None, reason=NOT_YET.get(metric, reason))
        return

    summaries = [(e.get("payload") or {}).get("memory") for e in store.rows("memory_events", run_id=run_id,
                                                                               event_type="attempt")]
    summaries = [s for s in summaries if s]
    usage = store.rows("memory_usage", run_id=run_id)
    recalls = sum(s.get("recalls", 0) for s in summaries)
    with_cases = sum(s.get("recalls_with_cases", 0) for s in summaries)
    errors = sum(s.get("recall_errors", 0) for s in summaries)
    if mode == "off":
        exposure, why = "none", "memory mode off"
    elif not recalls and not usage:
        exposure, why = "none", ("store unreachable" if errors else "no recall point was reached")
    elif not usage:
        exposure, why = "none", "no matching past case"
    else:
        exposure, why = ("full" if with_cases >= recalls else "partial"), ""
    conditions["memory_exposure"] = exposure
    sheet.text("memory_exposure", exposure)
    if why:
        sheet.text("memory_exposure_reason", why)
    sheet.num("recall_errors", errors)
    sheet.num("recall_empty", recalls - with_cases)
    shown, used = len(usage), sum(bool(u.get("used")) for u in usage)
    sheet.num("memories_recalled", shown)
    sheet.num("memories_used", used)
    sheet.num("recall_precision", used / shown if shown else None, reason="no past case was shown")
    sheet.num("memory_requests", sum(s.get("requests", 0) for s in summaries))
    sheet.num("memory_prompt_chars", sum(s.get("prompt_chars", 0) for s in summaries))
    sheet.num("memories_retained", len(store.rows("memories", run_id=run_id)))
    for agent in sorted({u.get("agent") for u in usage if u.get("agent")}):
        mine = [u for u in usage if u.get("agent") == agent]
        sheet.num("memories_recalled", len(mine), scope=agent)
        sheet.num("memories_used", sum(bool(u.get("used")) for u in mine), scope=agent)
    for metric, reason in NOT_YET.items():
        sheet.num(metric, None, reason=reason)


def _write(measured: dict) -> bool:
    """Store one run's measurements. False when they already exist (written once)."""
    conditions = measured["conditions"]
    columns = list(conditions)
    with run_store._connect() as conn:
        try:
            conn.execute(f"INSERT INTO run_conditions ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                         [conditions[c] for c in columns])
            conn.executemany(
                "INSERT INTO run_metrics (run_id, metrics_version, scope, metric, value_num, value_text, unit, "
                "missing_reason, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(conditions["run_id"], METRICS_VERSION, scope, metric, *row, conditions["computed_at"])
                 for (scope, metric), row in measured["metrics"].items()],
            )
        except sqlite3.IntegrityError:
            conn.rollback()
            return False
    return True


def _compute(run_id: str) -> tuple[dict | None, str]:
    with run_store._connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            return None, "no such run"
        if conn.execute("SELECT 1 FROM run_conditions WHERE run_id = ? AND metrics_version = ?",
                        (run_id, METRICS_VERSION)).fetchone():
            return None, "already measured"
        stages = [dict(r) for r in conn.execute("SELECT * FROM stages WHERE run_id = ? ORDER BY id", (run_id,))]
        timings = [dict(r) for r in conn.execute("SELECT * FROM stage_timings WHERE run_id = ?", (run_id,))]
        next_start = conn.execute("SELECT MIN(started_at) FROM runs WHERE started_at > ?",
                                  (run["started_at"],)).fetchone()[0]
    run = dict(run)
    status, notes = _status(run, stages, timings)
    if status is None:
        return None, "still running"
    measured = _measure(run, status, stages, next_start, notes, timings)
    if not _write(measured):
        return None, "already measured"
    return measured, "measured"


def compute(run_id: str) -> dict | None:
    """Measure one run and store its conditions and metrics (once per
    METRICS_VERSION). Returns {"conditions", "metrics"}, or None when the
    run is unknown, may still be running, or was already measured."""
    return _compute(run_id)[0]


def backfill() -> dict:
    """Measure every recorded run that has no measurements at all. A run
    measured under an older METRICS_VERSION keeps those measurements; it is
    not measured again after its process ended (the conditions recorded at
    its start, such as the memory mode, would be missing)."""
    with run_store._connect() as conn:
        run_ids = [r[0] for r in conn.execute("SELECT run_id FROM runs ORDER BY started_at")]
        measured = {r[0] for r in conn.execute("SELECT DISTINCT run_id FROM run_conditions")}
    return {run_id: ("already measured" if run_id in measured else _compute(run_id)[1])
            for run_id in run_ids}


# --- Reading and comparing ---

def load_runs(topic: str = None, parent_paper: str = None, since: str = None, contributor: str = None) -> list:
    """Every measured run (its newest metrics version) as
    {"conditions": {...}, "metrics": {(scope, metric): {"value", "missing_reason", "unit"}}}."""
    with run_store._connect() as conn:
        rows = conn.execute(
            "SELECT c.* FROM run_conditions c JOIN (SELECT run_id, MAX(metrics_version) v FROM run_conditions "
            "GROUP BY run_id) latest ON c.run_id = latest.run_id AND c.metrics_version = latest.v "
            "ORDER BY c.started_at").fetchall()
        runs = []
        for row in rows:
            cond = dict(row)
            if topic and topic_key(topic) not in (cond["topic_key"] or ""):
                continue
            paper = f"{cond['parent_paper_doi'] or ''} {cond['parent_paper_title'] or ''}".lower()
            if parent_paper and parent_paper.lower() not in paper:
                continue
            if since and (cond["started_at"] or "") < since:
                continue
            if contributor and cond["contributor"] != contributor:
                continue
            metrics = {}
            for m in conn.execute("SELECT * FROM run_metrics WHERE run_id = ? AND metrics_version = ?",
                                  (cond["run_id"], cond["metrics_version"])):
                value = m["value_num"] if m["value_num"] is not None else m["value_text"]
                metrics[(m["scope"], m["metric"])] = {"value": value, "missing_reason": m["missing_reason"],
                                                      "unit": m["unit"]}
            if cond["metrics_version"] < 2:
                # Version 1 used completion gaps under the measured metric's
                # name. Relabel on read without rewriting immutable history.
                for (scope, metric), measurement in list(metrics.items()):
                    if metric == "agent_seconds":
                        metrics[(scope, "agent_seconds_estimate")] = measurement
                        metrics[(scope, metric)] = {**measurement, "value": None,
                                                   "missing_reason": "legacy run has no recorded execution timer"}
            runs.append({"conditions": cond, "metrics": metrics})
    return runs


def _task(cond: dict) -> str:
    if (cond.get("comparison_group") or "").startswith("parent_pdf:"):
        return f"{cond['topic_key'] or '(no topic)'} | {cond['comparison_group']}"
    paper = cond["parent_paper_doi"] or (cond["parent_paper_title"] or "").strip().lower() or "(no Parent Paper)"
    return f"{cond['topic_key'] or '(no topic)'} | {paper}"


def _confounder_value(cond: dict, field: str):
    if field == "top_served_model":
        served = _decode(cond.get("served_models")) or {}
        return max(served, key=served.get) if served else None
    return cond.get(field)


def _stats(values: list) -> dict:
    values = sorted(values)
    out = {"n": len(values)}
    if values:
        out.update(median=statistics.median(values), mean=statistics.fmean(values), min=values[0], max=values[-1])
        if len(values) >= 2:
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
            out["iqr"] = q3 - q1
    return out


def cliffs_delta(a: list, b: list) -> float:
    """P(a > b) - P(a < b): -1..1, 0 = no difference."""
    greater = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    return (greater - less) / (len(a) * len(b))


def permutation_test(a: list, b: list, seed: int = 0) -> float:
    """Two-sided p-value for a difference in means (exact when the number
    of splits is at most PERMUTATIONS, else PERMUTATIONS random splits)."""
    pooled = list(a) + list(b)
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    total = comb(len(pooled), len(a))

    def extreme(first_idx) -> bool:
        chosen = set(first_idx)
        x = [pooled[i] for i in first_idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        return abs(statistics.fmean(x) - statistics.fmean(y)) >= observed - 1e-12

    if total <= PERMUTATIONS:
        hits = sum(extreme(idx) for idx in itertools.combinations(range(len(pooled)), len(a)))
        return hits / total
    rng = random.Random(seed)
    hits = sum(extreme(rng.sample(range(len(pooled)), len(a))) for _ in range(PERMUTATIONS))
    return (hits + 1) / (PERMUTATIONS + 1)


def compare_runs(runs: list, group_by: str = "memory_mode", metrics: tuple = DEFAULT_COMPARE_METRICS,
                 all_tasks: bool = False) -> dict:
    """Group runs by task (same topic and Parent Paper, unless all_tasks)
    and, inside each task, by the run condition group_by. Per group and
    metric: n, median, mean, min, max, IQR. With exactly two groups of at
    least MIN_GROUP_RUNS values: a permutation test and Cliff's delta.
    Lists every condition that differs between the groups. Never states a
    conclusion; it shows the numbers and their limits."""
    if group_by not in GROUP_FIELDS:
        raise ValueError(f"group_by must be one of {', '.join(GROUP_FIELDS)}")
    tasks = {}
    for r in runs:
        tasks.setdefault("all tasks" if all_tasks else _task(r["conditions"]), []).append(r)

    kind = "descriptive"
    groups_seen = {r["conditions"].get("comparison_group") for r in runs}
    if len(groups_seen) == 1 and None not in groups_seen:
        kind = "controlled"

    blocks = []
    for task, task_runs in tasks.items():
        groups = {}
        for r in task_runs:
            groups.setdefault(str(r["conditions"].get(group_by)), []).append(r)
        block = {"task": task, "runs": len(task_runs), "groups": {}, "tests": {}, "confounders": {}}
        for name, members in groups.items():
            statuses = {}
            for r in members:
                statuses[r["conditions"]["status"]] = statuses.get(r["conditions"]["status"], 0) + 1
            per_metric = {}
            for metric in metrics:
                values = [r["metrics"].get(("run", metric), {}).get("value") for r in members]
                numeric = [v for v in values if isinstance(v, (int, float))]
                per_metric[metric] = {**_stats(numeric), "runs": len(members)}
            block["groups"][name] = {"runs": len(members), "statuses": statuses,
                                     "too_few_runs": len(members) < MIN_GROUP_RUNS, "metrics": per_metric}
        if len(groups) == 2:
            (name_a, a_runs), (name_b, b_runs) = groups.items()
            for metric in metrics:
                a = [r["metrics"].get(("run", metric), {}).get("value") for r in a_runs]
                b = [r["metrics"].get(("run", metric), {}).get("value") for r in b_runs]
                a = [v for v in a if isinstance(v, (int, float))]
                b = [v for v in b if isinstance(v, (int, float))]
                if len(a) >= MIN_GROUP_RUNS and len(b) >= MIN_GROUP_RUNS:
                    block["tests"][metric] = {"groups": [name_a, name_b], "p_value": permutation_test(a, b),
                                              "cliffs_delta": cliffs_delta(a, b), "n": [len(a), len(b)]}
        for field in CONFOUNDER_FIELDS:
            if field == group_by:
                continue
            values = {name: sorted({str(_confounder_value(r["conditions"], field)) for r in members})
                      for name, members in groups.items()}
            if len({v for vs in values.values() for v in vs}) > 1:
                block["confounders"][field] = values
        blocks.append(block)
    return {"kind": kind, "group_by": group_by, "metrics": list(metrics), "blocks": blocks}


# --- CLI ---

def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _print_show(run_id: str) -> int:
    runs = [r for r in load_runs() if r["conditions"]["run_id"] == run_id]
    if not runs:
        print(f"No measurements for run {run_id} (run `compute {run_id}` first).")
        return 1
    run = runs[0]
    print(f"Run {run_id} (metrics version {run['conditions']['metrics_version']})")
    for key, value in run["conditions"].items():
        if key not in ("run_id", "metrics_version") and value is not None:
            print(f"  {key}: {value}")
    print("  Metrics (scope / metric = value [unit] or missing reason):")
    for (scope, metric), m in sorted(run["metrics"].items(), key=lambda kv: (kv[0][0] != "run", kv[0])):
        shown = _fmt(m["value"]) + (f" {m['unit']}" if m["value"] is not None and m["unit"] else "")
        print(f"    {scope} / {metric} = {shown if m['value'] is not None else 'NULL (' + str(m['missing_reason']) + ')'}")
    return 0


def _print_compare(runs: list, report: dict) -> None:
    group_by = report["group_by"]
    print(f"Comparison of {len(runs)} run(s), grouped by {group_by} "
          f"({report['kind']}: {'a controlled test' if report['kind'] == 'controlled' else 'describes differences; it cannot show what caused them'}).")
    print("\nPer run:")
    shown = ("final_phase_reached", "final_gap_points", "attempts", "llm_calls", "run_seconds")
    print(f"  {'run_id':<27} {'status':<8} {group_by:<12} " + " ".join(f"{m[:18]:>18}" for m in shown))
    for r in runs:
        c = r["conditions"]
        print(f"  {c['run_id']:<27} {c['status']:<8} {str(c.get(group_by))[:12]:<12} "
              + " ".join(f"{_fmt(r['metrics'].get(('run', m), {}).get('value')):>18}" for m in shown))
    for block in report["blocks"]:
        print(f"\nTask: {block['task']}  ({block['runs']} run(s))")
        for name, group in block["groups"].items():
            note = "  [too few runs: no conclusion]" if group["too_few_runs"] else ""
            print(f"  Group {group_by}={name}: {group['runs']} run(s), statuses {group['statuses']}{note}")
            for metric, s in group["metrics"].items():
                if s["n"]:
                    print(f"    {metric:<28} n={s['n']}/{s['runs']} median={_fmt(s['median'])} mean={_fmt(s['mean'])} "
                          f"min={_fmt(s['min'])} max={_fmt(s['max'])} iqr={_fmt(s.get('iqr'))}")
                else:
                    print(f"    {metric:<28} n=0/{s['runs']} (no run has a value)")
        if block["tests"]:
            print("  Tests (two-sided permutation test on the mean; Cliff's delta, first group vs second):")
            for metric, t in block["tests"].items():
                print(f"    {metric:<28} {t['groups'][0]} vs {t['groups'][1]}: p={t['p_value']:.4f} "
                      f"delta={t['cliffs_delta']:+.2f} n={t['n']}")
        elif len(block["groups"]) == 2:
            print(f"  No test: each group needs at least {MIN_GROUP_RUNS} runs with a value.")
        if block["confounders"]:
            print("  Conditions that differ (possible confounders):")
            for field, values in block["confounders"].items():
                print(f"    {field}: " + "; ".join(f"{g}: {', '.join(v)[:120]}" for g, v in values.items()))


def _write_out(path: str, runs: list, report: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        metrics = report["metrics"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "task", report["group_by"], "status", *metrics])
            for r in runs:
                c = r["conditions"]
                writer.writerow([c["run_id"], _task(c), c.get(report["group_by"]), c["status"],
                                 *[r["metrics"].get(("run", m), {}).get("value") for m in metrics]])
    else:
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")


def _main(argv: list) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.orchestration.run_metrics")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("compute").add_argument("run_id")
    sub.add_parser("backfill")
    sub.add_parser("show").add_argument("run_id")
    cmp = sub.add_parser("compare")
    cmp.add_argument("--topic")
    cmp.add_argument("--parent-paper")
    cmp.add_argument("--since")
    cmp.add_argument("--contributor")
    cmp.add_argument("--group-by", default="memory_mode", choices=GROUP_FIELDS)
    cmp.add_argument("--metrics", help="comma-separated run-scope metrics")
    cmp.add_argument("--all-tasks", action="store_true", help="put runs of different tasks in one comparison")
    cmp.add_argument("--out", help="write the comparison to a .json or .csv file")
    args = parser.parse_args(argv)

    if args.command == "compute":
        measured, reason = _compute(args.run_id)
        print(f"{args.run_id}: {reason}")
        return 0 if measured or reason == "already measured" else 1
    if args.command == "backfill":
        for run_id, reason in backfill().items():
            print(f"  {run_id}: {reason}")
        return 0
    if args.command == "show":
        return _print_show(args.run_id)
    runs = load_runs(args.topic, args.parent_paper, args.since, args.contributor)
    if not runs:
        print("No measured runs match (run `backfill` to measure recorded runs).")
        return 1
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip()) if args.metrics else DEFAULT_COMPARE_METRICS
    report = compare_runs(runs, args.group_by, metrics, args.all_tasks)
    _print_compare(runs, report)
    if args.out:
        _write_out(args.out, runs, report)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
