"""
Pure analysis shared by the pipeline and the dashboard (milestone M9).

The dashboard may not import pipeline modules (tests/test_dashboard.py), but
it must show exactly the numbers the pipeline's CLI shows. So the parts both
need live here, with no import from the rest of the project and no I/O:

  * comparing measured runs (run_metrics.compare_runs, the dashboard's
    Compare runs view): grouping by task and condition, per-group n /
    median / mean / min / max / IQR, the permutation test and Cliff's delta,
    and the conditions that differ between groups;
  * reading run_metrics rows into values (with the version-1 relabelling);
  * a memory's effective quality (ranking.effective_quality, the dashboard's
    Memory view).

Standard library only.
"""
import itertools
import json
import random
import statistics
from math import comb

GROUP_FIELDS = ("memory_mode", "assigned_condition", "memory_exposure", "llm_provider", "git_commit", "code_source",
                "contributor", "comparison_group", "status")
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


def _decode(text):
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


# --- Run metrics rows ---

def metrics_from_rows(rows, metrics_version: int) -> dict:
    """{(scope, metric): {"value", "missing_reason", "unit"}} from run_metrics
    rows (mappings) of one run and version. Version 1 used completion gaps
    under agent_seconds; they are relabelled on read, never rewritten."""
    metrics = {}
    for m in rows:
        value = m["value_num"] if m["value_num"] is not None else m["value_text"]
        metrics[(m["scope"], m["metric"])] = {"value": value, "missing_reason": m["missing_reason"], "unit": m["unit"]}
    if metrics_version < 2:
        for (scope, metric), measurement in list(metrics.items()):
            if metric == "agent_seconds":
                metrics[(scope, "agent_seconds_estimate")] = measurement
                metrics[(scope, metric)] = {**measurement, "value": None,
                                           "missing_reason": "legacy run has no recorded execution timer"}
    return metrics


# --- Comparing runs ---

def task_of(cond: dict) -> str:
    if (cond.get("comparison_group") or "").startswith("parent_pdf:"):
        return f"{cond['topic_key'] or '(no topic)'} | {cond['comparison_group']}"
    paper = cond["parent_paper_doi"] or (cond["parent_paper_title"] or "").strip().lower() or "(no Parent Paper)"
    return f"{cond['topic_key'] or '(no topic)'} | {paper}"


def confounder_value(cond: dict, field: str):
    if field == "top_served_model":
        served = _decode(cond.get("served_models")) or {}
        return max(served, key=served.get) if served else None
    return cond.get(field)


def stats(values: list) -> dict:
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
        tasks.setdefault("all tasks" if all_tasks else task_of(r["conditions"]), []).append(r)

    # Only a planned test (src/orchestration/comparison.py, "test:<name>") is a
    # controlled comparison; runs sharing a supplied Parent Paper PDF
    # ("parent_pdf:...") are the same task, not a controlled test.
    kind = "descriptive"
    groups_seen = {r["conditions"].get("comparison_group") for r in runs}
    if len(groups_seen) == 1 and str(next(iter(groups_seen)) or "").startswith("test:"):
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
                per_metric[metric] = {**stats(numeric), "runs": len(members)}
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
            values = {name: sorted({str(confounder_value(r["conditions"], field)) for r in members})
                      for name, members in groups.items()}
            if len({v for vs in values.values() for v in vs}) > 1:
                block["confounders"][field] = values
        blocks.append(block)
    return {"kind": kind, "group_by": group_by, "metrics": list(metrics), "blocks": blocks}


# --- Memory scores ---

def effective_quality(memory: dict) -> float:
    """The score recall ranks by (plan 9.3): the base quality, moved toward
    how often the memory was followed by a good outcome."""
    base = float(memory.get("quality") or 0)
    helped, hurt = int(memory.get("times_helped") or 0), int(memory.get("times_hurt") or 0)
    judged = helped + hurt
    if not judged:
        return round(base, 4)
    weight = 1 / (1 + judged / 3)
    return round(weight * base + (1 - weight) * (helped + 1) / (judged + 2), 4)
