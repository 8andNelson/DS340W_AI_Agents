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

The published values come from the Parent Paper's own full text when an
open-access copy exists (src/search/paper_fulltext.py), else from its
verified abstract.

Then the agent acts as a senior data analyst (review()): it reads the
paper, the baseline script, the run's output and the data actually used,
runs deterministic leakage checks over the script
(src/orchestration/code_checks.py -- e.g. SMOTE before the split), and
asks one LLM call for findings and the ONE next action most likely to
raise a valid score: collect more data (Data Agent), fix the baseline
(Replication Agent), hand over to the Experiment Agent, or accept.
Every LLM finding must quote evidence that really occurs in the script,
output, paper or data facts, else it is UNVERIFIED and cannot drive a
decision; fixed rules then check the decision itself (_decide). The agent
never runs another agent: it returns a structured request, and Master
Agent decides whether to act on it.

This agent never blocks the pipeline: a FAILED or INCONCLUSIVE
evaluation is itself useful research information, not an agent defect --
CLAUDE.md's Results Integrity policy is explicit that "a failed or
imperfect reproduction is still useful research information."
"""
import json
import re
from pathlib import Path

import pandas as pd

from src import llm
from src.orchestration.code_checks import find_leakage
from src.orchestration.dataset_budget import MAX_MASTER_ROWS
from src.search import paper_fulltext
from src.orchestration.handoff import request as handoff_request

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


def _extract_metrics(text: str, source_label: str, max_chars: int = STDOUT_CHAR_LIMIT) -> dict:
    """One LLM call restricted to metrics EXPLICITLY stated in text.
    Returns {} on any failure, malformed response, or when nothing is
    stated -- never estimates or guesses a value."""
    if not text:
        return {}

    prompt = f"""You are extracting quantitative evaluation metrics from {source_label}.

Text:
{text[:max_chars]}

Find every evaluation metric (e.g. accuracy, precision, recall, F1, AUC) with a numeric value EXPLICITLY stated in this text. Do not estimate, infer, or compute a metric that isn't directly stated.

Return ONLY a JSON object mapping metric name to its numeric value (a plain number, not a string; keep the scale exactly as stated -- percentages as 0-100, fractions as 0-1):
{{"accuracy": 94.2, "f1": 0.91}}
If no metric is explicitly stated, return {{}}.
Return ONLY valid JSON. No explanation or markdown."""

    try:
        content = llm.chat(prompt, timeout=300, agent="evaluation_agent")
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


# --- Senior data analyst review -------------------------------------------
#
# After the metric comparison, the agent reviews the replication the way a
# senior data analyst would: is the result valid (no leakage), does the
# code follow the paper's method, is there enough data, is the metric
# meaningful -- and decides the ONE next action most likely to raise the
# score. It never runs another agent itself: it returns a structured
# request, and Master Agent decides whether to act on it.

SCRIPT_CHAR_LIMIT = 16000
PAPER_REVIEW_CHARS = 9000
OUTPUT_TAIL_CHARS = 3000
REVIEW_TIMEOUT_SECONDS = 600
# A request for more data must be for meaningfully more than we have.
MIN_DATA_GROWTH = 1.2

ACTIONS = ("rerun_data_agent", "revise_replication", "run_experiment", "accept")
ACTION_TARGET = {"rerun_data_agent": "Data Agent", "revise_replication": "Replication Agent",
                 "run_experiment": "Experiment Agent"}
FINDING_CATEGORIES = ("data_leakage", "insufficient_data", "preprocessing", "method_mismatch", "hyperparameters",
                      "evaluation", "model_choice", "other")
SEVERITIES = ("critical", "major", "minor")
# Kinds of change the review may suggest with "run_experiment" -- a hint for
# the Experiment Agent, which decides the actual rerun request itself.
EXPERIMENT_CATEGORIES = ("alternative_model", "hyperparameter_change", "different_preprocessing",
                         "different_feature_selection", "alternative_loss_function",
                         "alternative_sampling_technique", "additional_evaluation_metric")


def _read_baseline_script(replication_result: dict) -> tuple[str, str]:
    """(path, text) of the script the baseline ran, or ("", "")."""
    slug_dir = Path(replication_result.get("replication_dir") or "")
    if not str(slug_dir) or not slug_dir.is_dir():
        return "", ""
    if replication_result.get("source") == "reconstructed":
        candidates = [slug_dir / "reconstructed" / "train.py"]
    else:
        entry = str(replication_result.get("entry_point") or "").split()
        script = next((part for part in entry if part.endswith(".py")), "")
        candidates = [slug_dir / "repo" / script] if script else []
    for path in candidates:
        if path.is_file():
            return str(path), path.read_text(encoding="utf-8", errors="ignore")
    return "", ""


def _label_counts(train_path: str, label_column: str) -> dict:
    if not train_path or not label_column:
        return {}
    try:
        series = pd.read_csv(train_path, usecols=[label_column])[label_column]
    except Exception:
        return {}
    return {str(k): int(v) for k, v in series.value_counts().items()}


def data_facts(cleaning_result: dict, dataset_result: dict) -> dict:
    """What the replication actually trained on, read from the Data and
    Cleaning Agents' own records (and the training CSV's label column)."""
    cleaning_result = cleaning_result or {}
    dataset_result = dataset_result or {}
    plan = dataset_result.get("data_plan") or {}
    split_rows = cleaning_result.get("split_rows") or {}
    counts = _label_counts(cleaning_result.get("train_csv_path", ""), cleaning_result.get("label_column", ""))
    return {
        "master_rows": cleaning_result.get("rows_total"),
        "train_rows": split_rows.get("train"),
        "test_rows": split_rows.get("test"),
        "validation_rows": split_rows.get("validation"),
        "label_column": cleaning_result.get("label_column") or "",
        "train_label_counts": counts,
        "minority_class_train_rows": min(counts.values()) if len(counts) >= 2 else None,
        "rows_sampled_to_budget": bool(cleaning_result.get("sampled")),
        "datasets_merged": cleaning_result.get("datasets_merged"),
        "parent_dataset_found": dataset_result.get("parent_dataset_found"),
        "data_agent_mode": dataset_result.get("mode", ""),
        "paper_dataset_size": plan.get("paper_dataset_size"),
        "data_agent_suggested_rows": plan.get("target_entries"),
        "data_agent_reasoning": plan.get("reasoning", ""),
        "max_rows_possible": MAX_MASTER_ROWS,
    }


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _verify(findings: list, corpus: str) -> list:
    """Keep the LLM's findings, each marked verified only if its evidence
    quote really occurs in the script, the run output, the paper text or
    the data facts. An unverified finding is reported (UNVERIFIED) but never
    drives a decision -- No Hallucination Policy."""
    corpus = _normalize_ws(corpus)
    checked = []
    for f in findings if isinstance(findings, list) else []:
        if not isinstance(f, dict):
            continue
        evidence = str(f.get("evidence") or "")
        category = str(f.get("category") or "other")
        severity = str(f.get("severity") or "minor").lower()
        quote = _normalize_ws(evidence)
        checked.append({
            "category": category if category in FINDING_CATEGORIES else "other",
            "severity": severity if severity in SEVERITIES else "minor",
            "issue": str(f.get("issue") or ""),
            "evidence": evidence,
            "fix": str(f.get("fix") or ""),
            "verified": len(quote) >= 4 and quote in corpus,
            "source": "llm_review",
        })
    return checked


def _review_prompt(parent_paper: dict, paper: dict, replication_result: dict, script: str, stdout: str,
                   stderr: str, metrics: list, facts: dict, static: list, history: list) -> str:
    paper_body = paper_fulltext.excerpt(paper.get("text", ""),
                                        paper_fulltext.METHOD_KEYWORDS + paper_fulltext.RESULT_KEYWORDS,
                                        max_chars=PAPER_REVIEW_CHARS)
    paper_label = ("excerpts from the paper's own full text" if paper.get("status") == "FULL_TEXT"
                   else "the paper's abstract and metadata only -- the full text was not available")
    metric_lines = "\n".join(
        f"- {m['metric']}: published={m['published_value']}, reproduced={m['reproduced_value']}, "
        f"{m['classification']}" + (f" ({m['notes']})" if m.get("notes") else "") for m in metrics) or "- none"
    static_lines = "\n".join(f"- line {f['line']}: {f['issue']} Evidence: `{f['evidence']}`" for f in static) \
        or "- none found"
    history_lines = "\n".join(
        f"- rerun {h.get('rerun', h.get('round'))}: {h.get('target_agent') or h.get('action')}"
        + (f" ({'; '.join(h['instructions'])})" if h.get("instructions") else "")
        + f" -> {h.get('outcome', '')}" for h in history or []) or "- none"
    source = replication_result.get("source", "")
    revisable = ("yes (reconstructed by this project)" if source == "reconstructed"
                 else "no (the authors' cloned repository is preserved as the baseline)")
    return f"""You are a senior data analyst reviewing a machine learning replication of an academic paper. Your
job: find what is wrong or weak in the replication, and decide the ONE next action most likely to give a valid,
higher score that is faithful to the paper. Think like a reviewer who must sign off on the result.

Check at least:
1. Data leakage: resampling (SMOTE etc.), scaling, encoding, imputation or feature selection fitted before the split
   or on test data; the test set used for training, tuning or early stopping; the target or a proxy of it used as a
   feature; duplicate rows across train and test. A leaky score is not a real score -- fixing it comes first.
2. Method fidelity: does the script do what the paper says (model, preprocessing order, split, class-imbalance
   handling, hyperparameters, evaluation)? Name each deviation.
3. Data sufficiency: compare the rows and minority-class examples used with the paper's dataset size and the model's
   needs. More data helps only when the model is data-limited.
4. Evaluation: are the metrics meaningful (e.g. accuracy alone on an imbalanced target is misleading)?

Parent Paper: "{parent_paper.get('title', '')}"
Paper text ({paper_label}):
\"\"\"
{paper_body}
\"\"\"

Baseline source: {source} -- can its code be revised: {revisable}
Baseline script:
```python
{script[:SCRIPT_CHAR_LIMIT] or '(not available)'}
```
Run output (last part of stdout):
{(stdout or '(empty)')[-OUTPUT_TAIL_CHARS:]}
Run errors/warnings (last part of stderr):
{(stderr or '(empty)')[-1500:]}

Published vs reproduced metrics:
{metric_lines}

Data actually used (from the Data and Cleaning Agents' records):
{json.dumps(facts, indent=2, default=str)}

Automated leakage checks over the script (deterministic):
{static_lines}

Earlier reruns in this run (do not repeat an action that did not help):
{history_lines}

Possible next actions:
- "rerun_data_agent": ask Master Agent to have the Data Agent collect MORE data from sources away from the Parent
  Paper. Give "target_entries" (total rows wanted, at most {MAX_MASTER_ROWS}). Only when the model is data-limited.
- "revise_replication": ask Master Agent to have the Replication Agent fix specific defects in the baseline script
  (leakage, wrong preprocessing order, deviations from the paper). Give precise "instructions". Only if the baseline
  can be revised (see above).
- "run_experiment": the baseline is valid; hand over to the Experiment Agent, which works on closing the gap to the
  paper's score. Give a suggested "experiment_category" (one of {', '.join(EXPERIMENT_CATEGORIES)}) and
  "instructions" as hints.
- "accept": the baseline is valid and faithful; nothing would clearly improve it.

Every finding must quote its evidence EXACTLY as it appears in the script, the run output, the paper text or the
data facts above (copy a short exact snippet). Do not invent facts.

Return ONE JSON object:
{{
  "assessment": "<3-5 sentences: is the result valid, how faithful is it, what limits the score>",
  "findings": [{{"category": "<one of {', '.join(FINDING_CATEGORIES)}>", "severity": "critical|major|minor",
                "issue": "<what is wrong>", "evidence": "<exact quote>", "fix": "<what to change>"}}],
  "decision": {{"action": "<one of {', '.join(ACTIONS)}>", "reason": "<why this is the best next step>",
               "target_entries": <integer or null>, "experiment_category": "<category or empty>",
               "instructions": ["<precise instruction>"]}}
}}
Return ONLY valid JSON. No explanation or markdown."""


def _decide(raw: dict, findings: list, static: list, facts: dict, source: str) -> tuple[dict, list]:
    """Turn the LLM's proposed decision into one the evidence supports.
    Returns (decision, notes on any change). Deterministic rules:

      - verified leakage in a reconstructed baseline is always fixed first
        (a leaky score is not a real score);
      - "revise_replication" needs a verified critical/major finding and a
        reconstructed baseline;
      - "rerun_data_agent" needs a verified insufficient_data finding and a
        goal meaningfully above the rows already used, within the budget;
      - anything else that does not hold becomes "accept" (or
        "run_experiment" when one was proposed with a valid category).
    """
    notes = []
    raw = raw if isinstance(raw, dict) else {}
    action = str(raw.get("action") or "accept")
    action = action if action in ACTIONS else "accept"
    instructions = [str(i) for i in raw.get("instructions") or [] if str(i).strip()] \
        if isinstance(raw.get("instructions"), list) else []
    category = str(raw.get("experiment_category") or "")
    verified = [f for f in findings if f["verified"]]
    serious = [f for f in verified + static if f["severity"] in ("critical", "major")]
    leakage = [f for f in static + verified if f["category"] == "data_leakage" and f["severity"] in ("critical", "major")]

    if leakage and source == "reconstructed" and action != "revise_replication":
        notes.append(f"Proposed '{action}' changed to 'revise_replication': verified data leakage must be fixed "
                     "before any other change, since the current score is not valid.")
        action = "revise_replication"
    if action == "revise_replication":
        if source != "reconstructed":
            notes.append("'revise_replication' not possible: the baseline is the authors' cloned code, which is "
                         "preserved. The defects are reported for the Experiment Agent and the report.")
            action = "accept"
        elif not serious:
            notes.append("'revise_replication' not supported by a verified critical/major finding.")
            action = "accept"
        else:
            fixes = [f"{f['issue']} Fix: {f['fix']}" for f in leakage + serious if f.get("fix")]
            instructions = list(dict.fromkeys(fixes + instructions))
    if action == "rerun_data_agent":
        current = facts.get("master_rows") or 0
        goal = raw.get("target_entries")
        try:
            goal = int(goal)
        except (TypeError, ValueError):
            goal = 0
        goal = min(goal, MAX_MASTER_ROWS)
        if not any(f["category"] == "insufficient_data" for f in verified):
            notes.append("'rerun_data_agent' not supported by a verified insufficient_data finding.")
            action = "accept"
        elif current >= MAX_MASTER_ROWS:
            notes.append(f"'rerun_data_agent' not possible: already at the {MAX_MASTER_ROWS}-row budget.")
            action = "accept"
        elif goal < current * MIN_DATA_GROWTH:
            notes.append(f"'rerun_data_agent' goal {goal} is not meaningfully above the {current} rows used.")
            action = "accept"
        else:
            raw = dict(raw, target_entries=goal)
    if action == "run_experiment" and category not in EXPERIMENT_CATEGORIES:
        notes.append(f"Experiment category '{category}' is not a known category; the Experiment Agent decides.")
        category = ""

    return {
        "action": action,
        "target_agent": ACTION_TARGET.get(action, ""),
        "reason": str(raw.get("reason") or ""),
        "target_entries": raw.get("target_entries") if action == "rerun_data_agent" else None,
        "experiment_category": category if action == "run_experiment" else "",
        "instructions": instructions,
    }, notes


def review(parent_paper: dict, paper: dict, replication_result: dict, metrics: list, stdout: str,
           cleaning_result: dict = None, dataset_result: dict = None, history: list = None) -> dict:
    """The senior data analyst review (see the section comment above)."""
    script_path, script = _read_baseline_script(replication_result)
    stderr = _read_stdout(replication_result.get("stderr_path", ""))
    static = [dict(f, verified=True, source="static_check") for f in find_leakage(script)]
    facts = data_facts(cleaning_result, dataset_result)
    source = replication_result.get("source", "")

    raw, reviewed_by, review_error = {}, "", ""
    try:
        content = llm.chat(_review_prompt(parent_paper, paper, replication_result, script, stdout, stderr,
                                          metrics, facts, static, history),
                           timeout=REVIEW_TIMEOUT_SECONDS, effort="high", agent="evaluation_agent")
        raw = llm.parse_json(content, default={})
        raw = raw if isinstance(raw, dict) else {}
        reviewed_by = llm.last_served_by()
    except Exception as e:
        review_error = f"The review call failed: {e}"

    corpus = "\n".join([script, stdout or "", stderr, paper.get("text", ""), json.dumps(facts, default=str)])
    findings = _verify(raw.get("findings") or [], corpus)
    decision, decision_notes = _decide(raw.get("decision"), findings, static, facts, source)
    if review_error:
        decision_notes.insert(0, review_error)

    return {
        "assessment": str(raw.get("assessment") or ""),
        "static_findings": static,
        "findings": findings,
        "data_facts": facts,
        "decision": decision,
        "decision_notes": decision_notes,
        "script_path": script_path,
        "paper_text_status": paper.get("status", ""),
        "reviewed_by": reviewed_by,
    }


def _request_for(decision: dict, review_result: dict) -> list:
    """The structured request to Master Agent (CLAUDE.md's agent message
    model), or [] when the decision is "accept"."""
    if decision["action"] == "accept":
        return []
    evidence = [f for f in review_result["static_findings"] + review_result["findings"]
                if f.get("verified", True) and f["severity"] in ("critical", "major")]
    goal = {"target_entries": decision["target_entries"]} if decision["target_entries"] else {}
    if decision["experiment_category"]:
        goal["experiment_category"] = decision["experiment_category"]
    return [{
        "sender": "evaluation_agent",
        "recipient": "master_agent",
        "message_type": "request",
        "status": "needs_review",
        "requires_review": True,
        "target_agent": decision["target_agent"],
        "action": decision["action"],
        "goal": goal,
        "instructions": decision["instructions"],
        "summary": decision["reason"],
        "evidence": [{"issue": f["issue"], "evidence": f["evidence"]} for f in evidence[:6]],
    }]


def run(parent_paper: dict, replication_result: dict, cleaning_result: dict = None, dataset_result: dict = None,
        history: list = None) -> dict:
    """
    Compare the Replication Agent's captured output against the Parent
    Paper's own stated results (a per-metric breakdown plus an overall
    classification -- the worst case among matched metrics, or INCONCLUSIVE
    if none could be matched), then review the replication as a senior data
    analyst and return the next-action request for Master Agent.

    cleaning_result/dataset_result describe the data actually used;
    history lists earlier reruns in this run (Master Agent's rerun records:
    "rerun", "target_agent", "instructions", "outcome") so the review does
    not repeat an action that did not help.
    """
    parent_paper = parent_paper or {}
    replication_result = replication_result or {}
    paper_title = parent_paper.get("title", "")

    print(f"\n[Evaluation Agent] Comparing reproduced results against '{paper_title}'...")

    # The paper's own full text is the best source of its results. Without
    # it, the official abstract (when Validation attached one) is the only
    # published text used: results_summary and abstract were read from a
    # search snippet, which can quote another paper's results.
    paper = paper_fulltext.get_parent_paper_text(parent_paper) if parent_paper else {"status": "", "text": ""}
    published_limit = STDOUT_CHAR_LIMIT
    if paper.get("status") == "FULL_TEXT":
        published_text = paper_fulltext.excerpt(paper["text"], paper_fulltext.RESULT_KEYWORDS, max_chars=8000)
        published_source = "full_text"
        published_label = ("an academic paper's own full text -- extract only the final results of the paper's own "
                           "main method, not those of baselines or of other papers it cites")
        published_limit = 8000
    elif parent_paper.get("official_abstract"):
        published_text = parent_paper["official_abstract"]
        published_source = "official_abstract"
        published_label = "an academic paper's own stated results"
    else:
        published_text = " ".join(filter(None, [parent_paper.get("results_summary"), parent_paper.get("abstract")]))
        published_source = "search_snippet"
        published_label = "an academic paper's own stated results"
    reproduced_text = _read_stdout(replication_result.get("stdout_path", ""))

    published_metrics = _extract_metrics(published_text, published_label, max_chars=published_limit)
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

    print(f"[Evaluation Agent] Compared {len(metrics)} metric(s): overall={overall}. Reviewing the replication...")
    review_result = review(parent_paper, paper, replication_result, metrics, reproduced_text,
                           cleaning_result, dataset_result, history)
    decision = review_result["decision"]
    for f in review_result["static_findings"]:
        print(f"[Evaluation Agent] Leakage check: {f['issue']} (line {f['line']}: {f['evidence']})")
    for note in review_result["decision_notes"]:
        print(f"[Evaluation Agent] Decision check: {note}")
    print(f"[Evaluation Agent] Done. Overall={overall}, decision={decision['action']}"
          + (f" -> {decision['target_agent']}" if decision["target_agent"] else "") + ".")

    return {
        "status": overall,
        "source": replication_result.get("source", ""),
        "metrics": metrics,
        "published_metrics_found": bool(published_metrics),
        "published_source": published_source,
        "reproduced_metrics_found": bool(reproduced),
        "likely_causes": causes,
        "review": review_result,
        "decision": decision,
        "requests": _request_for(decision, review_result),
        "notes": "",
    }


def handoff(result: dict, context: dict) -> list:
    """The Evaluation Agent's request to Master Agent when it finishes: its
    validity request (a rerun of the Data or Replication Agent) when its
    review made one, else run the Experiment Agent on the gap to the paper."""
    reruns = [r for r in (result or {}).get("requests") or []
              if r.get("action") in ("rerun_data_agent", "revise_replication")]
    if reruns:
        return reruns[:1]
    return [handoff_request("evaluation_agent", "Experiment Agent", {},
                            f"Evaluation done (overall {(result or {}).get('status', '')}); the gap to the paper's "
                            "score needs a data scientist's analysis.")]
