"""
Is the Parent Paper's own score reachable without leakage? (Evaluation Agent)

A paper that oversamples (or SMOTEs) its whole dataset before the
train/test split puts copies of the same minority rows in train and test,
and reports near-perfect scores that no correct reproduction can reach. In
run 20261115T171901Z-86734b16 the paper's KBD + RFE + Over + RF row gave
F1 0.9926 and recall 1.0; the honest baseline gave F1 0.10, and the same
data oversampled before the split gave F1 0.996.

This module runs a fixed probe (leakage_probe_script.py) on the run's own
master dataset: one standard random forest, scored with a correct split
and with oversampling before the split. Then a deterministic rule:

  a published metric is "explained by leakage" when it is within
  LEAK_MATCH_POINTS of the leaky score and at least LEAK_GAP_POINTS above
  the honest score. The verdict is CONSISTENT_WITH_LEAKAGE when at least
  half of the compared metrics are.

It is evidence, not proof (the probe is not the paper's exact method), so
the Evaluation Agent reports it as "the published result may come from
leakage", and the Experiment Agent stops chasing that score.

The probe runs only when it can matter: the paper's text describes
resampling, and a published score is at least LEAK_GAP_POINTS above the
reproduction. It runs in the replication venv (scikit-learn is not in the
project's .venv), as an argv list with a timeout and no project secrets.
"""
import json
import math
import os
import re
import subprocess
from pathlib import Path

from src.orchestration import repo_review

PROBE_SCRIPT = Path(__file__).with_name("leakage_probe_script.py")
LEAK_MATCH_POINTS = 5.0   # published within this of the leaky score
LEAK_GAP_POINTS = 15.0    # and at least this above the honest score (and the reproduction)
PROBE_MAX_ROWS = 50_000
PROBE_TIMEOUT_SECONDS = 600
RESAMPLING = re.compile(r"smote|adasyn|over-?sampl|under-?sampl|resampl|randomoversampler|borderline", re.IGNORECASE)
PROBE_METRICS = ("accuracy", "precision", "recall", "f1", "auc")

_CACHE: dict = {}  # (csv, size, mtime, label) -> probe result: the Evaluation Agent runs after every rerun


def _venv_python(replication_result: dict) -> Path | None:
    folder = replication_result.get("replication_dir")
    if not folder:
        return None
    python = Path(folder) / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return python if python.is_file() else None


def run_probe(python: Path, csv_path: str, label_column: str) -> dict:
    """The probe's JSON result, or {"error": ...}. Never raises."""
    try:
        stat = Path(csv_path).stat()
    except OSError as e:
        return {"error": f"the master dataset cannot be read ({e})"}
    key = (str(csv_path), stat.st_size, stat.st_mtime_ns, label_column)
    if key in _CACHE:
        return _CACHE[key]
    try:
        proc = subprocess.run([str(python), "-I", str(PROBE_SCRIPT), str(csv_path), label_column, str(PROBE_MAX_ROWS)],
                              capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS, encoding="utf-8",
                              errors="replace", env=repo_review.child_environment({"PYTHONUTF8": "1"}))
    except subprocess.TimeoutExpired:
        return {"error": f"the probe did not finish within {PROBE_TIMEOUT_SECONDS} s"}
    except OSError as e:
        return {"error": f"the probe could not start ({e})"}
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        result = {}
    if not isinstance(result, dict) or not result:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
        result = {"error": f"the probe failed (exit {proc.returncode}): {tail[0][:300]}"}
    if "error" not in result:
        _CACHE[key] = result
    return result


def compare(published_points: dict, probe: dict) -> dict:
    """The deterministic verdict. published_points: {metric: published value
    in percentage points}, names already normalized (accuracy, precision,
    recall, f1, auc)."""
    rows = []
    for metric in PROBE_METRICS:
        if metric not in published_points or metric not in probe.get("honest", {}):
            continue
        published = published_points[metric]
        honest = round(probe["honest"][metric] * 100, 2)
        leaky = round(probe["leaky"][metric] * 100, 2)
        explained = abs(published - leaky) <= LEAK_MATCH_POINTS and published - honest >= LEAK_GAP_POINTS
        rows.append({"metric": metric, "published_points": published, "honest_points": honest,
                     "leaky_points": leaky, "explained_by_leakage": explained})
    explained = sum(r["explained_by_leakage"] for r in rows)
    if not rows:
        return {"status": "SKIPPED", "reason": "No published metric can be compared with the probe's metrics.",
                "metrics": rows}
    consistent = explained >= max(1, math.ceil(len(rows) / 2))
    return {"status": "CONSISTENT_WITH_LEAKAGE" if consistent else "NOT_CONSISTENT", "metrics": rows,
            "reason": f"{explained} of {len(rows)} published metric(s) are within {LEAK_MATCH_POINTS} points of the "
                      f"leaky probe and at least {LEAK_GAP_POINTS} points above the honest probe."}


def check(published_points: dict, reproduced_points: dict, paper_text: str, replication_result: dict,
          cleaning_result: dict) -> dict:
    """The full check for the Evaluation Agent. Always returns a dict with
    a status: CONSISTENT_WITH_LEAKAGE, NOT_CONSISTENT, SKIPPED (not
    needed) or UNAVAILABLE (needed but could not run)."""
    gaps = {m: published_points[m] - reproduced_points[m] for m in published_points if m in reproduced_points}
    if not gaps or max(gaps.values()) < LEAK_GAP_POINTS:
        return {"status": "SKIPPED", "reason": f"No published score is {LEAK_GAP_POINTS} points or more above the "
                                               "reproduction.", "metrics": []}
    match = RESAMPLING.search(paper_text or "")
    if not match:
        return {"status": "SKIPPED", "reason": "The paper's text does not describe resampling.", "metrics": []}
    csv_path = (cleaning_result or {}).get("master_csv_path")
    label = (cleaning_result or {}).get("label_column")
    if not csv_path or not label:
        return {"status": "UNAVAILABLE", "reason": "No master dataset with a label column.", "metrics": []}
    python = _venv_python(replication_result or {})
    if python is None:
        return {"status": "UNAVAILABLE", "reason": "No replication environment with scikit-learn to run the probe "
                                                   "in (a sandboxed repository has none on this machine).",
                "metrics": []}
    probe = run_probe(python, csv_path, label)
    if "error" in probe:
        return {"status": "UNAVAILABLE", "reason": probe["error"], "metrics": []}
    verdict = compare(published_points, probe)
    return {**verdict, "probe": {k: probe[k] for k in ("rows", "positive_rate", "features")},
            "resampling_quote": (paper_text or "")[max(match.start() - 80, 0):match.end() + 80].replace("\n", " ")}


def likely_cause(result: dict) -> str:
    """The Evaluation Agent's likely-cause sentence for a positive verdict."""
    rows = [r for r in result.get("metrics", []) if r["explained_by_leakage"]]
    detail = "; ".join(f"{r['metric']}: paper {r['published_points']}, leaky probe {r['leaky_points']}, "
                       f"correct split {r['honest_points']}" for r in rows)
    return ("The published result may come from leakage: the paper describes resampling, and a standard random "
            "forest on this run's data reaches the paper's scores only when the data is oversampled before the "
            f"train/test split (copies of the same rows in train and test) -- {detail} (points). If the paper "
            "resampled before splitting, its scores cannot be reached by a correct reproduction, so this gap is "
            "not a reproduction defect and should not be chased.")
