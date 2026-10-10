"""
Evaluation Agent (Milestone 11).

Compares the Replication Agent's actual captured output against the
Parent Paper's own published results, classifying each matched metric as:

  - EXACT        -- reproduced value within a tight tolerance of published.
  - APPROXIMATE  -- within a looser tolerance -- a real reproduction, with
                    expected variance (seeds/hardware/library versions).
  - FAILED       -- outside that tolerance -- a genuine discrepancy.
  - INCONCLUSIVE -- no comparable number exists on one or both sides, so
                    no verdict can be produced for that metric. Per
                    CLAUDE.md's "state uncertainty, don't guess" policy --
                    a paper that never states a concrete number doesn't
                    get a manufactured one, and neither does a run that
                    didn't print one.

Extracting "what number did the paper actually report" and "what number
did our run actually print" are each a single, narrowly-scoped LLM call
restricted to values EXPLICITLY stated in that text -- the same
no-guessing discipline as every other extraction call in this project
(e.g. data_agent's entry_count: null-unless-explicitly-stated rule). The
classification itself is fully deterministic (a fixed percentage-point
tolerance band) once both numbers are in hand -- matching this project's
established precedent (Data Agent's reproducibility-fit check) that a
hard accept/reject-style fact should never be decided by an LLM once the
facts needed to decide it are available.

This agent never blocks the pipeline: a FAILED or INCONCLUSIVE
evaluation is itself useful research information, not an agent defect --
CLAUDE.md's Results Integrity policy is explicit that "a failed or
imperfect reproduction is still useful research information."
"""
import json
import re
from pathlib import Path

import requests

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL

STDOUT_CHAR_LIMIT = 4000
EXACT_THRESHOLD_POINTS = 1.0        # percentage points
APPROXIMATE_THRESHOLD_POINTS = 5.0  # percentage points

_STATUS_RANK = {"EXACT": 0, "APPROXIMATE": 1, "FAILED": 2, "INCONCLUSIVE": 3}

_METRIC_DEFAULTS = {
    "metric": "",
    "published_value": None,
    "reproduced_value": None,
    "difference_points": None,
    "classification": "INCONCLUSIVE",
    "notes": "",
}


def _default_metric(**overrides) -> dict:
    metric = dict(_METRIC_DEFAULTS)
    metric.update(overrides)
    return metric


def _normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _read_stdout(stdout_path: str) -> str:
    if not stdout_path:
        return ""
    try:
        return Path(stdout_path).read_text(errors="ignore")[:STDOUT_CHAR_LIMIT]
    except Exception:
        return ""


def _extract_metrics(text: str, source_label: str) -> dict:
    """One LLM call restricted to metrics EXPLICITLY stated in text.
    Returns {} on any failure, malformed response, or when nothing is
    stated -- never estimates or guesses a value."""
    if not text:
        return {}

    prompt = f"""You are extracting quantitative evaluation metrics from {source_label}.

Text:
{text[:STDOUT_CHAR_LIMIT]}

Find every evaluation metric (e.g. accuracy, precision, recall, F1, AUC) with a numeric value EXPLICITLY stated in this text. Do not estimate, infer, or compute a metric that isn't directly stated.

Return ONLY a JSON object mapping metric name to its numeric value (a plain number, not a string; keep the scale exactly as stated -- percentages as 0-100, fractions as 0-1):
{{"accuracy": 94.2, "f1": 0.91}}
If no metric is explicitly stated, return {{}}.
Return ONLY valid JSON. No explanation or markdown."""

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception:
        return {}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else {}

    if not isinstance(data, dict):
        return {}

    return {name: float(value) for name, value in data.items() if isinstance(value, (int, float))}


def _classify(published: float, reproduced: float) -> tuple:
    """Deterministic classification -- a fixed percentage-point tolerance
    band, not an LLM judgment call, once both numbers are in hand. Both
    values are normalized onto a 0-100 percentage-point scale first so a
    0-1 fraction and a 0-100 percentage compare correctly against the
    same thresholds."""
    pub, rep = published, reproduced
    if abs(pub) <= 1.0 and abs(rep) <= 1.0:
        pub, rep = pub * 100, rep * 100
    diff = abs(pub - rep)
    if diff <= EXACT_THRESHOLD_POINTS:
        return "EXACT", diff
    if diff <= APPROXIMATE_THRESHOLD_POINTS:
        return "APPROXIMATE", diff
    return "FAILED", diff


def _likely_causes(parent_paper: dict, replication_result: dict, metrics: list) -> list:
    """Deterministic, fact-grounded candidate causes -- template reasoning
    over data already in hand (CLAUDE.md's own list of standard cause
    categories), not a free-form LLM guess. Only causes that actually
    apply given what we know are listed."""
    causes = []
    if replication_result.get("source") == "reconstructed":
        causes.append("Implementation was reconstructed from the paper's text, not the authors' own code -- "
                       "undocumented implementation details are the most likely source of any gap.")
    if replication_result.get("inferred_assumptions"):
        causes.append(f"{len(replication_result['inferred_assumptions'])} detail(s) had to be inferred "
                       "(hyperparameters, preprocessing, etc.) rather than taken from the paper directly.")
    if replication_result.get("modifications"):
        causes.append("The original repository's dataset wiring was modified to point at our own CSVs, "
                       "which may not exactly match the authors' original data split/version.")
    if not metrics:
        causes.append("No comparable metric could be extracted from either the paper's own text or the "
                       "run's output -- the paper may not state a concrete number, or the run didn't print one.")
    if any(m["classification"] == "FAILED" for m in metrics):
        causes.append("Possible causes for a large gap: random seed differences, hardware/library version "
                       "differences, missing preprocessing details, or hyperparameter differences not stated "
                       "in the paper.")
    return causes


def run(parent_paper: dict, replication_result: dict) -> dict:
    """
    Compare the Replication Agent's captured output against the Parent
    Paper's own stated results. Returns a per-metric breakdown plus an
    overall classification -- the worst case among matched metrics, or
    INCONCLUSIVE if none could be matched.
    """
    parent_paper = parent_paper or {}
    replication_result = replication_result or {}
    paper_title = parent_paper.get("title", "")

    print(f"\n[Evaluation Agent] Comparing reproduced results against '{paper_title}'...")

    published_text = " ".join(filter(None, [parent_paper.get("results_summary"), parent_paper.get("abstract")]))
    reproduced_text = _read_stdout(replication_result.get("stdout_path", ""))

    published_metrics = _extract_metrics(published_text, "an academic paper's own stated results")
    reproduced_metrics = _extract_metrics(reproduced_text, "a reproduction run's captured output")

    published_by_key = {_normalize_metric_name(k): (k, v) for k, v in published_metrics.items()}
    reproduced_by_key = {_normalize_metric_name(k): (k, v) for k, v in reproduced_metrics.items()}

    metrics = []
    for key, (pub_name, pub_value) in published_by_key.items():
        if key in reproduced_by_key:
            _, rep_value = reproduced_by_key[key]
            classification, diff = _classify(pub_value, rep_value)
            metrics.append(_default_metric(
                metric=pub_name, published_value=pub_value, reproduced_value=rep_value,
                difference_points=diff, classification=classification,
            ))
        else:
            metrics.append(_default_metric(
                metric=pub_name, published_value=pub_value,
                notes="Not printed by the reproduction run.",
            ))
    for key, (rep_name, rep_value) in reproduced_by_key.items():
        if key not in published_by_key:
            metrics.append(_default_metric(
                metric=rep_name, reproduced_value=rep_value,
                notes="Not stated in the paper's own text.",
            ))

    overall = max((m["classification"] for m in metrics), key=lambda c: _STATUS_RANK[c]) if metrics else "INCONCLUSIVE"
    causes = _likely_causes(parent_paper, replication_result, metrics)

    print(f"[Evaluation Agent] Done. Overall={overall}, metrics_compared={len(metrics)}.")

    return {
        "status": overall,
        "source": replication_result.get("source", ""),
        "metrics": metrics,
        "published_metrics_found": bool(published_metrics),
        "reproduced_metrics_found": bool(reproduced_metrics),
        "likely_causes": causes,
        "notes": "",
    }
