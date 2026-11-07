"""
Experiment Agent (Milestone 12).

Only becomes active once a baseline reproduction has succeeded -- Master
Agent only reaches this stage when Replication Agent's status was "OK",
since that stage already blocks the pipeline otherwise (see
master_agent._check_replication). Proposes and runs exactly one
experimental change per invocation ("change one major variable at a
time" per CLAUDE.md), then evaluates it against the preserved BASELINE's
own reproduced metrics -- not the paper's published numbers, which is
Evaluation Agent's comparison, not this one's.

Baseline preservation (CLAUDE.md's Baseline Preservation section) is a
fresh, separate experiment directory per attempt
(experiments/<slug>/exp_NNN/) -- the working replication/<slug>/ tree
Replication Agent produced is only ever read from, never modified or
executed in place.

Two kinds of experiment, chosen deterministically by category-exhaustion
order (never randomly, and never repeating a category already recorded
in previous_experiments for this paper):

  1. "different_dataset_split" -- always available, regardless of
     source. Re-splits the master CSV with a different seed and reruns
     the EXACT SAME entry point (the cloned repo's own script, or the
     reconstructed script) against the new split, reusing the baseline's
     already-installed virtualenv -- the environment and the code both
     stay fixed; only the split changes.
  2. The remaining code-level categories from CLAUDE.md's catalog
     (alternative model, hyperparameter change, different preprocessing,
     different feature selection, alternative loss function, alternative
     sampling technique, additional evaluation metric) -- reconstructed
     source only, since that's the only case where we actually have and
     understand the code being run. One LLM call proposes a single,
     scoped modification (same JSON-call-with-regex-fallback-parsing
     pattern as every other generation call in this project), changing
     only what that category calls for. No new dependencies are ever
     installed for an experiment -- the environment itself is one of the
     variables being held fixed, same as the code for the dataset-split
     category.

A cloned repository that has already tried "different_dataset_split" has
no further category this version knows how to attempt safely without
understanding that specific repo's own code/CLI -- reported as SKIPPED,
not silently ignored.

Never blocks the pipeline: a REGRESSED or BLOCKED experiment is itself a
valid research finding (CLAUDE.md: "a failed or imperfect reproduction
is still useful research information" applies equally here), not an
agent defect.
"""
import shutil
from pathlib import Path

import pandas as pd

from src import llm
from src.agents.cleaning_agent import _split_train_test_validation
from src.agents.replication_agent import (
    REPLICATION_DIR, _slugify, _venv_python, _run_bounded, _wire_dataset_into_repo, _repair_generated_script,
    _dataset_profile, data_env,
)
from src.agents.evaluation_agent import METRIC_LINE_FORMAT, _extract_metrics, parse_metric_lines, to_percentage_points

REPO_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

EXPERIMENT_TIMEOUT_SECONDS = 1800
EXPERIMENT_SPLIT_SEED = 7  # deliberately different from cleaning_agent.SPLIT_SEED (42)
NEUTRAL_THRESHOLD_POINTS = 1.0  # percentage points
GENERATION_TIMEOUT_SECONDS = 600
# A modified script gets up to 2 repairs. A repair is told to keep the
# experiment's change -- it fixes the script, it does not undo the
# experiment.
MAX_EXPERIMENT_ATTEMPTS = 3

DATASET_SPLIT_CATEGORY = "different_dataset_split"
# Code-level categories are only attempted for the "reconstructed" source
# -- see module docstring. Order is the exhaustion order _next_category
# walks through.
CODE_LEVEL_CATEGORIES = (
    "alternative_model", "hyperparameter_change", "different_preprocessing",
    "different_feature_selection", "alternative_loss_function",
    "alternative_sampling_technique", "additional_evaluation_metric",
)

_RESULT_DEFAULTS = {
    "status": "",                 # "OK" | "BLOCKED" | "TIMEOUT" | "SKIPPED"
    "source": "",
    "category": "",
    "hypothesis": "",
    "what_changed": "",
    "experiment_dir": "",
    "command": None,
    "exit_code": None,
    "stdout_path": "",
    "stderr_path": "",
    "duration_seconds": None,
    "baseline_metrics": {},
    "experiment_metrics": {},
    "comparison": [],
    "overall": "INCONCLUSIVE",    # "IMPROVED" | "REGRESSED" | "NEUTRAL" | "INCONCLUSIVE"
    "warnings": [],
    "notes": "",
}


def _default_result(**overrides) -> dict:
    result = dict(_RESULT_DEFAULTS)
    result.update(overrides)
    return result


def _next_category(source: str, previous_experiments: list) -> str:
    """Deterministic, exhaustion-ordered category choice -- never
    repeats a category already recorded for this paper, never random."""
    tried = {e.get("category", "") for e in (previous_experiments or [])}
    if source == "reconstructed":
        for category in CODE_LEVEL_CATEGORIES:
            if category not in tried:
                return category
    if DATASET_SPLIT_CATEGORY not in tried:
        return DATASET_SPLIT_CATEGORY
    return ""


def _baseline_metrics_from_evaluation(evaluation_result: dict) -> dict:
    """Evaluation Agent's per-metric breakdown already carries the
    reproduced value for every metric it found (whether matched against
    the paper or not) -- reused here rather than re-extracting."""
    metrics = {}
    for m in (evaluation_result or {}).get("metrics", []):
        if m.get("reproduced_value") is not None:
            metrics[m["metric"]] = m["reproduced_value"]
    return metrics


def _classify_change(baseline: float, experiment: float) -> tuple:
    """Direction-aware (unlike evaluation_agent._classify's
    distance-only comparison): an experiment either helps, hurts, or
    doesn't move the needle, which matters more here than how close it
    lands to a fixed target."""
    base, exp = to_percentage_points(baseline), to_percentage_points(experiment)
    diff = exp - base
    if diff > NEUTRAL_THRESHOLD_POINTS:
        return "IMPROVED", diff
    if diff < -NEUTRAL_THRESHOLD_POINTS:
        return "REGRESSED", diff
    return "NEUTRAL", diff


def _compare_to_baseline(baseline_metrics: dict, experiment_metrics: dict) -> list:
    comparison = []
    for name, base_value in baseline_metrics.items():
        if name in experiment_metrics:
            classification, diff = _classify_change(base_value, experiment_metrics[name])
            comparison.append({
                "metric": name, "baseline_value": base_value, "experiment_value": experiment_metrics[name],
                "difference_points": diff, "classification": classification,
            })
        else:
            comparison.append({
                "metric": name, "baseline_value": base_value, "experiment_value": None,
                "difference_points": None, "classification": "INCONCLUSIVE",
            })
    for name, exp_value in experiment_metrics.items():
        if name not in baseline_metrics:
            comparison.append({
                "metric": name, "baseline_value": None, "experiment_value": exp_value,
                "difference_points": None, "classification": "INCONCLUSIVE",
            })
    return comparison


_OVERALL_RANK = {"REGRESSED": 0, "NEUTRAL": 1, "IMPROVED": 2, "INCONCLUSIVE": 3}


def _overall_from_comparison(comparison: list) -> str:
    """Worst case wins -- a single regressed metric is never hidden
    behind an improved one, same "don't hide a real discrepancy"
    principle as evaluation_agent's own worst-case aggregation."""
    if not comparison:
        return "INCONCLUSIVE"
    return min((c["classification"] for c in comparison), key=lambda c: _OVERALL_RANK[c])


def _resplit_dataset(cleaning_result: dict, experiment_dir: Path) -> dict:
    """Re-split the master CSV with a different seed, writing the new
    train/test/validation CSVs into this experiment's own directory --
    the canonical data/processed/ files Cleaning Agent produced are
    never touched."""
    master_df = pd.read_csv((cleaning_result or {}).get("master_csv_path", ""))
    train_df, test_df, validation_df = _split_train_test_validation(master_df, seed=EXPERIMENT_SPLIT_SEED)

    data_dir = experiment_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path, validation_path = data_dir / "train.csv", data_dir / "test.csv", data_dir / "validation.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    return {
        "master_csv_path": (cleaning_result or {}).get("master_csv_path", ""),
        "train_csv_path": str(train_path), "test_csv_path": str(test_path), "validation_csv_path": str(validation_path),
    }


def _rewrite_hardcoded_csv_paths(script_text: str, cleaning_result: dict, new_split: dict) -> str:
    """The reconstructed baseline script has the original train/test CSV
    paths hardcoded as literal strings (the generation prompt explicitly
    asked for that) -- deterministic text substitution, not an LLM call,
    points it at this experiment's own re-split CSVs instead."""
    for key in ("train_csv_path", "test_csv_path", "validation_csv_path"):
        original = (cleaning_result or {}).get(key, "")
        if original:
            script_text = script_text.replace(original, new_split[key])
    return script_text


def _run_dataset_split_experiment(source: str, baseline_dir: Path, replication_result: dict,
                                   cleaning_result: dict, experiment_dir: Path) -> dict:
    new_split = _resplit_dataset(cleaning_result, experiment_dir)

    if source == "cloned_repository":
        run_cwd = experiment_dir / "repo"
        shutil.copytree(baseline_dir / "repo", run_cwd)
        _wire_dataset_into_repo(run_cwd, new_split)
        command = list(replication_result.get("command") or [])
    else:
        run_cwd = experiment_dir
        baseline_script_path = baseline_dir / "reconstructed" / "train.py"
        script_text = baseline_script_path.read_text(errors="ignore") if baseline_script_path.is_file() else ""
        script_text = _rewrite_hardcoded_csv_paths(script_text, cleaning_result, new_split)
        (run_cwd / "train.py").write_text(script_text, encoding="utf-8")
        venv_dir = baseline_dir / "venv"
        command = [str(_venv_python(venv_dir)), str(run_cwd / "train.py")]

    hypothesis = (
        f"A different random train/test/validation split (seed={EXPERIMENT_SPLIT_SEED} vs. the baseline's "
        "seed=42) should produce similar performance if the model generalizes rather than overfitting to the "
        "baseline split's particular partition; a large change would suggest split-sensitivity."
    )
    # The reconstructed script reads its data paths from environment
    # variables (replication_agent.DATA_ENV_VARS); pointing them at the new
    # split is what actually makes this run use it.
    run_result = _run_bounded(command, cwd=run_cwd, timeout=EXPERIMENT_TIMEOUT_SECONDS, env=data_env(new_split))
    return {
        "hypothesis": hypothesis,
        "what_changed": f"Train/test/validation split reseeded from 42 to {EXPERIMENT_SPLIT_SEED}.",
        "command": command,
        "run_result": run_result,
    }


def _propose_code_change(category: str, parent_paper: dict, baseline_code: str, inferred_assumptions: list) -> dict:
    """One LLM call proposing a single, scoped code modification
    implementing exactly one experiment category -- same
    JSON-call-with-regex-fallback-parsing pattern as every other
    generation call in this project. Returns {} on any failure."""
    prompt = f"""You are proposing ONE controlled experiment on an existing baseline machine learning script, changing exactly one variable.

Paper title: "{parent_paper.get('title', '')}"
Experiment category to apply: "{category}"

Baseline script (already a working reproduction):
{baseline_code}

Previously inferred assumptions in the baseline (context only -- do not change these unless the category above specifically calls for it):
{chr(10).join(inferred_assumptions) or "(none)"}

Modify ONLY what the "{category}" category calls for (e.g. "alternative_model" -> swap the model class; "hyperparameter_change" -> change one hyperparameter; "different_preprocessing" -> change one preprocessing step; "different_feature_selection" -> change which features are used; "alternative_loss_function" -> change the loss/objective; "alternative_sampling_technique" -> change how class imbalance/sampling is handled; "additional_evaluation_metric" -> add one new metric to the printed output without changing the model). Keep everything else identical to the baseline script, including how it reads its input paths (the TRAIN_CSV/TEST_CSV/VALIDATION_CSV environment variables) and its "{METRIC_LINE_FORMAT}" metric lines. Use only libraries the baseline script already imports, plus the Python standard library -- nothing new is installed for an experiment.

Respond with exactly two fenced blocks and nothing else. First, a JSON block:
```json
{{"hypothesis": "<1-2 sentences: what you expect to happen and why>", "what_changed": "<1 sentence: exactly what was changed>"}}
```
Then the full modified script:
```python
<the script>
```"""

    try:
        content = llm.chat(prompt, timeout=GENERATION_TIMEOUT_SECONDS, effort="high")
    except Exception as e:
        print(f"[Experiment Agent] Proposal call failed: {e}")
        return {}

    code = llm.parse_code(content)
    if not code:
        return {}
    meta = llm.parse_json(content, default={})
    meta = meta if isinstance(meta, dict) else {}
    return {
        "hypothesis": str(meta.get("hypothesis") or ""),
        "what_changed": str(meta.get("what_changed") or ""),
        "code": code,
    }


def _run_code_level_experiment(category: str, parent_paper: dict, baseline_dir: Path,
                                replication_result: dict, experiment_dir: Path, cleaning_result: dict = None) -> dict:
    baseline_script_path = baseline_dir / "reconstructed" / "train.py"
    baseline_code = baseline_script_path.read_text(errors="ignore") if baseline_script_path.is_file() else ""
    if not baseline_code:
        return {"error": "Baseline reconstructed script could not be read; cannot propose a code-level experiment."}

    proposal = _propose_code_change(category, parent_paper, baseline_code, replication_result.get("inferred_assumptions", []))
    if not proposal:
        return {"error": "LLM could not propose a code-level modification for this category."}

    script_path = experiment_dir / "train.py"
    venv_dir = baseline_dir / "venv"
    command = [str(_venv_python(venv_dir)), str(script_path)]
    env = data_env(cleaning_result)
    dataset_profile = _dataset_profile(cleaning_result)
    code = proposal["code"]
    run_result = None

    for attempt in range(1, MAX_EXPERIMENT_ATTEMPTS + 1):
        script_path.write_text(code, encoding="utf-8")
        print(f"[Experiment Agent] Running modified script (attempt {attempt}/{MAX_EXPERIMENT_ATTEMPTS})...")
        run_result = _run_bounded(command, cwd=experiment_dir, timeout=EXPERIMENT_TIMEOUT_SECONDS, env=env)
        if run_result["timed_out"] or (run_result["exit_code"] == 0 and parse_metric_lines(run_result["stdout"])):
            break
        if attempt == MAX_EXPERIMENT_ATTEMPTS:
            break
        feedback = run_result.get("stderr") or ""
        if run_result["exit_code"] == 0:
            feedback = f"The script exited successfully but printed no '{METRIC_LINE_FORMAT}' lines.\n" + feedback
        repaired = _repair_generated_script(
            code, f"This script is an experiment ({proposal['what_changed']}); keep that change.\n{feedback}",
            parent_paper, stdout=run_result.get("stdout", ""), dataset_profile=dataset_profile,
        )
        if not repaired:
            break
        code = repaired

    return {
        "hypothesis": proposal["hypothesis"],
        "what_changed": proposal["what_changed"],
        "command": command,
        "run_result": run_result,
    }


def run(parent_paper: dict, replication_result: dict, evaluation_result: dict, cleaning_result: dict,
        previous_experiments: list = None) -> dict:
    """
    Run exactly one experiment -- a single changed variable -- against
    the preserved baseline, and evaluate whether it improved, regressed,
    or made no meaningful difference. Only meant to be called once a
    baseline reproduction has succeeded.
    """
    parent_paper = parent_paper or {}
    replication_result = replication_result or {}
    paper_title = parent_paper.get("title", "")
    source = replication_result.get("source", "")

    category = _next_category(source, previous_experiments)
    if not category:
        print(f"\n[Experiment Agent] No remaining experiment category available for '{paper_title}' "
              f"(source={source}); every category this version knows how to attempt has already been tried.")
        return _default_result(
            status="SKIPPED", source=source,
            notes="Every experiment category available for this source has already been tried.",
        )

    print(f"\n[Experiment Agent] Running experiment '{category}' for '{paper_title}' (source={source})...")

    slug = _slugify(paper_title or source)
    baseline_dir = REPLICATION_DIR / slug
    experiment_number = len(previous_experiments or []) + 1
    experiment_dir = EXPERIMENTS_DIR / slug / f"exp_{experiment_number:03d}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _baseline_metrics_from_evaluation(evaluation_result)

    if category == DATASET_SPLIT_CATEGORY:
        outcome = _run_dataset_split_experiment(source, baseline_dir, replication_result, cleaning_result, experiment_dir)
    else:
        outcome = _run_code_level_experiment(category, parent_paper, baseline_dir, replication_result, experiment_dir,
                                             cleaning_result)

    if "error" in outcome:
        return _default_result(
            status="BLOCKED", source=source, category=category, experiment_dir=str(experiment_dir),
            warnings=[outcome["error"]], notes="Experiment setup failed before execution.",
        )

    hypothesis, what_changed, command, run_result = (
        outcome["hypothesis"], outcome["what_changed"], outcome["command"], outcome["run_result"],
    )

    stdout_path = experiment_dir / "run_stdout.txt"
    stderr_path = experiment_dir / "run_stderr.txt"
    stdout_path.write_text(run_result.get("stdout", "") or "", encoding="utf-8")
    stderr_path.write_text(run_result.get("stderr", "") or "", encoding="utf-8")

    warnings = []
    if run_result["timed_out"]:
        status = "TIMEOUT"
    elif run_result["exit_code"] == 0:
        status = "OK"
    else:
        status = "BLOCKED"
        warnings.append(f"Experiment script exited with code {run_result['exit_code']}.")

    experiment_metrics = {}
    if status == "OK":
        experiment_metrics = parse_metric_lines(run_result.get("stdout", "")) or _extract_metrics(
            (run_result.get("stdout", "") or "")[-4000:], "an experiment run's captured output")
    comparison = _compare_to_baseline(baseline_metrics, experiment_metrics) if status == "OK" else []
    overall = _overall_from_comparison(comparison) if status == "OK" else "INCONCLUSIVE"

    print(f"[Experiment Agent] Done. status={status}, category={category}, overall={overall}.")

    return _default_result(
        status=status, source=source, category=category, hypothesis=hypothesis, what_changed=what_changed,
        experiment_dir=str(experiment_dir), command=command, exit_code=run_result["exit_code"],
        stdout_path=str(stdout_path), stderr_path=str(stderr_path), duration_seconds=run_result["duration_seconds"],
        baseline_metrics=baseline_metrics, experiment_metrics=experiment_metrics, comparison=comparison,
        overall=overall, warnings=warnings,
    )
