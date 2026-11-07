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
import re
from pathlib import Path

from src import llm

STDOUT_CHAR_LIMIT = 4000
EXACT_THRESHOLD_POINTS = 1.0        # percentage points
APPROXIMATE_THRESHOLD_POINTS = 5.0  # percentage points

# Reconstructed scripts are told to print each final metric in this exact
# form (see replication_agent's generation prompt), so a run's own output
# is read deterministically -- no LLM call, no chance of a misread number.
# The LLM extraction below remains the fallback for output in any other
# form (e.g. a cloned repository's own logging).
METRIC_LINE_FORMAT = "METRIC <name>=<value>"
_METRIC_LINE = re.compile(r"^\s*METRIC\s+([A-Za-z0-9_\- ]+?)\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*%?\s*$",
                          re.MULTILINE)

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


def parse_metric_lines(text: str) -> dict:
    """Metrics printed as METRIC_LINE_FORMAT lines; a later line for the
    same metric wins (the final evaluation, not an intermediate one)."""
    return {name.strip(): float(value) for name, value in _METRIC_LINE.findall(text or "")}


def _read_stdout(stdout_path: str) -> str:
    """The full captured stdout. Truncation to STDOUT_CHAR_LIMIT happens
    only for the LLM prompt -- METRIC lines are usually printed last, so
    they must be searched for in the whole text."""
    if not stdout_path:
        return ""
    try:
        return Path(stdout_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def reproduced_metrics(stdout: str, source_label: str) -> dict:
    """A run's own metrics: METRIC lines when present, else the LLM
    fallback over the end of the output (where final results are
    printed)."""
    metrics = parse_metric_lines(stdout)
    if metrics:
        return metrics
    return _extract_metrics((stdout or "")[-STDOUT_CHAR_LIMIT:], source_label)


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
        content = llm.chat(prompt, timeout=300)
    except Exception:
        return {}

    data = llm.parse_json(content, default={})
    if not isinstance(data, dict):
        return {}

    return {name: float(value) for name, value in data.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def to_percentage_points(value: float) -> float:
    """A metric on the 0-100 scale. Each value is normalized on its own: a
    paper stating "99%" and a run printing 0.9965 must compare as 99 vs.
    99.65, not 99 vs. 0.9965. (Normalizing only when both values were
    <= 1 produced a false 98-point FAILED for exactly that pair.) A value
    <= 1 is read as a fraction -- correct for accuracy-style metrics,
    which is what this project compares."""
    return value * 100 if abs(value) <= 1.0 else value


def _classify(published: float, reproduced: float) -> tuple:
    """Deterministic classification -- a fixed percentage-point tolerance
    band, not an LLM judgment call, once both numbers are in hand. Both
    values are normalized onto a 0-100 percentage-point scale first so a
    0-1 fraction and a 0-100 percentage compare correctly against the
    same thresholds."""
    pub, rep = to_percentage_points(published), to_percentage_points(reproduced)
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
    if replication_result.get("dataset_sampled"):
        causes.append("The data was subsampled to the local row budget (stratified on the label, fixed seed), "
                      "so the model trained on fewer rows than the full dataset(s).")
    if replication_result.get("dataset_semantic_mappings"):
        causes.append(f"{replication_result['dataset_semantic_mappings']} column(s) were merged across datasets by "
                      "verified semantic mapping (see logs/column_mappings.json), so the master dataset differs "
                      "from any single source the paper used.")
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

    # The official abstract, when Validation attached one, is the only
    # published text used: results_summary and abstract were read from a
    # search snippet, which can quote another paper's results.
    if parent_paper.get("official_abstract"):
        published_text = parent_paper["official_abstract"]
        published_source = "official_abstract"
    else:
        published_text = " ".join(filter(None, [parent_paper.get("results_summary"), parent_paper.get("abstract")]))
        published_source = "search_snippet"
    reproduced_text = _read_stdout(replication_result.get("stdout_path", ""))

    published_metrics = _extract_metrics(published_text, "an academic paper's own stated results")
    reproduced = reproduced_metrics(reproduced_text, "a reproduction run's captured output")

    published_by_key = {_normalize_metric_name(k): (k, v) for k, v in published_metrics.items()}
    reproduced_by_key = {_normalize_metric_name(k): (k, v) for k, v in reproduced.items()}

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
    if published_source == "search_snippet" and published_metrics:
        causes.append("The published values were read from a search-result snippet, not the paper's verified "
                      "abstract -- a snippet can quote another paper's results, so treat them as UNVERIFIED.")

    print(f"[Evaluation Agent] Done. Overall={overall}, metrics_compared={len(metrics)}.")

    return {
        "status": overall,
        "source": replication_result.get("source", ""),
        "metrics": metrics,
        "published_metrics_found": bool(published_metrics),
        "published_source": published_source,
        "reproduced_metrics_found": bool(reproduced),
        "likely_causes": causes,
        "notes": "",
    }
