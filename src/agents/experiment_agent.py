"""
Experiment Agent: the project's expert data scientist.

Its goal is to bring the pipeline's reproduced score as close as possible
to the Parent Paper's published score. It does not run code itself. It
reads the Evaluation Agent's comparison and review (the gap per metric,
the verified findings, the leakage checks, the data facts), the Parent
Paper's own text, the baseline script and its output, and every earlier
rerun in this run with its effect on the gap. It then decides ONE next
step and returns it to Master Agent as a structured request:

  - "rerun_agent": rerun one earlier agent (Data, Cleaning, Code
    Discovery, Replication or Evaluation Agent) with new knowledge --
    precise instructions, and for the Data Agent optionally a row goal.
  - "stop": the gap is closed (within EXACT_THRESHOLD_POINTS), there is no
    published number to close a gap to, or nothing plausible is left.

Master Agent decides whether to act on the request (it can decline a
request as invalid or redundant and send the reason back, which this
agent receives as `feedback` on its next call), runs the requested agent
and every stage downstream of it, and calls this agent again with the new
evaluation. See master_agent._rerun_loop.

Deterministic checks on the answer (No Hallucination Policy): each
evidence quote is marked verified only if it occurs in the paper text,
the script, the run output, the evaluation or the data facts; the target
is normalized to a known agent name; a row goal is capped at the row
budget. Master Agent declines a request with no verified evidence.

The aim is the paper's score reached faithfully -- never by leakage,
tuning on the test set, or other changes that only move the number.
"""
import json
import re

from src import llm
from src.agents.evaluation_agent import (
    EXACT_THRESHOLD_POINTS, OUTPUT_TAIL_CHARS, SCRIPT_CHAR_LIMIT, _read_baseline_script, _read_stdout,
)
from src.orchestration.dataset_budget import MAX_MASTER_ROWS
from src.search import paper_fulltext
from src.orchestration.handoff import request as handoff_request

PAPER_CHARS = 9000
PROPOSAL_TIMEOUT_SECONDS = 600
MAX_EVIDENCE = 6

ACTIONS = ("rerun_agent", "stop")
# The agents this agent may ask Master Agent to rerun, and what each can do
# with new knowledge. Intake, Research and Validation are not listed: a
# rerun of those could replace the Parent Paper, and a gap to a different
# paper is not a smaller gap. Master Agent declines them.
RERUNNABLE_AGENTS = {
    "Data Agent": "collect the data again. With target_entries: collect MORE rows from sources away from the "
                  "Parent Paper (expand mode). Without it: search again for the paper's own dataset, following "
                  "your instructions (e.g. a dataset name or link written in the paper).",
    "Cleaning Agent": "merge the downloaded datasets into the master CSV again. Mostly deterministic; your "
                      "instructions only guide its label and column-matching decisions.",
    "Code Discovery Agent": "search again for the authors' own implementation, following your instructions "
                            "(e.g. a repository name or URL written in the paper). Its cache is skipped.",
    "Replication Agent": "change the reconstructed baseline script exactly as your instructions say (model, "
                         "hyperparameters, preprocessing order, resampling, split, features, printed metrics), "
                         "then run it. A cloned repository of the authors' code is never changed.",
    "Evaluation Agent": "compare the reproduced and published metrics again (e.g. when a published value was "
                        "misread or a metric was not matched).",
}
ALL_AGENTS = ("Intake Agent", "Research Agent", "Validation Agent", *RERUNNABLE_AGENTS)


def score_gap(evaluation_result: dict) -> float | None:
    """The mean absolute gap, in percentage points, between the published
    and reproduced values of every metric both sides state; None when no
    metric is on both sides. Smaller is closer to the Parent Paper."""
    diffs = [m["difference_points"] for m in (evaluation_result or {}).get("metrics", [])
             if m.get("difference_points") is not None]
    return round(sum(diffs) / len(diffs), 4) if diffs else None


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def canonical_agent(name: str) -> str:
    """'data_agent', 'Data agent', 'data' -> 'Data Agent'; '' when unknown."""
    key = re.sub(r"[^a-z]", "", str(name or "").lower()).removesuffix("agent")
    for agent in ALL_AGENTS:
        if re.sub(r"[^a-z]", "", agent.lower()).removesuffix("agent") == key:
            return agent
    return ""


def _history_lines(history: list) -> str:
    lines = []
    for h in history or []:
        lines.append(f"- rerun {h.get('rerun')}: {h.get('requested_by')} asked for {h.get('target_agent')} "
                     f"({'; '.join(h.get('instructions') or []) or 'no instructions'}"
                     + (f"; goal {h['goal']}" if h.get("goal") else "") + f") -> {h.get('outcome', '')}")
    return "\n".join(lines) or "- none yet"


def _prompt(parent_paper: dict, paper: dict, replication_result: dict, script: str, stdout: str,
            evaluation_result: dict, code_discovery_result: dict, gap: float | None, history: list,
            feedback: list) -> str:
    paper_body = paper_fulltext.excerpt(paper.get("text", ""),
                                        paper_fulltext.RESULT_KEYWORDS + paper_fulltext.METHOD_KEYWORDS,
                                        max_chars=PAPER_CHARS)
    paper_label = ("excerpts from the paper's own full text" if paper.get("status") == "FULL_TEXT"
                   else "the paper's abstract and metadata only -- the full text was not available")
    review = evaluation_result.get("review") or {}
    metric_lines = "\n".join(
        f"- {m['metric']}: published={m['published_value']}, reproduced={m['reproduced_value']}, "
        f"gap={m.get('difference_points')} points, {m['classification']}"
        + (f" ({m['notes']})" if m.get("notes") else "") for m in evaluation_result.get("metrics", [])) or "- none"
    finding_lines = "\n".join(
        f"- [{f['severity']}, {f.get('category', '')}] {f['issue']} Evidence: \"{f['evidence']}\""
        + ("" if f.get("verified", True) else " (UNVERIFIED)")
        for f in review.get("static_findings", []) + review.get("findings", [])) or "- none"
    decision = evaluation_result.get("decision") or {}
    repo = (code_discovery_result or {}).get("repository") or {}
    code_line = (f"FOUND: {repo.get('repo_url', '')}" if (code_discovery_result or {}).get("status") == "FOUND"
                 else "NOT_FOUND (the baseline was reconstructed from the paper)")
    agent_lines = "\n".join(f"- {name}: {what}" for name, what in RERUNNABLE_AGENTS.items())
    feedback_lines = "\n".join(f"- {f}" for f in feedback or []) or "- none"
    return f"""You are an expert data scientist on a project that reproduces an academic paper's machine learning
results. Goal: bring the pipeline's reproduced score as CLOSE AS POSSIBLE to the paper's published score, by making
the replication more faithful to the paper. Closest, not highest: a score above the paper's is also a gap. Never
reach the number by data leakage, by tuning on the test set, or by any change that is not faithful to the paper.

Find the most likely causes of the gap and choose ONE next step. You cannot run code; you ask the Master Agent to
rerun one earlier agent with new knowledge. The Master Agent runs that agent, every stage after it, and a new
evaluation, then asks you again. It may decline a request as invalid or redundant (see its feedback below).

Parent Paper: "{parent_paper.get('title', '')}"
Paper text ({paper_label}):
\"\"\"
{paper_body}
\"\"\"

Published vs reproduced metrics (mean gap: {gap if gap is not None else 'not measurable'} percentage points):
{metric_lines}

Evaluation Agent's assessment: {review.get('assessment') or 'none'}
Evaluation Agent's findings (automated leakage checks and its review):
{finding_lines}
Evaluation Agent's decision: {decision.get('action', '')} {('-- ' + decision.get('reason', '')) if decision.get('reason') else ''}
Data actually used:
{json.dumps(review.get('data_facts') or {}, indent=2, default=str)}

Implementation search: {code_line}
Baseline source: {replication_result.get('source', '')}
Inferred assumptions in the baseline: {'; '.join(replication_result.get('inferred_assumptions') or []) or 'none'}
Baseline script:
```python
{script[:SCRIPT_CHAR_LIMIT] or '(not available)'}
```
Run output (last part):
{(stdout or '(empty)')[-OUTPUT_TAIL_CHARS:]}

Earlier reruns in this run and their effect on the gap (do not repeat one that did not help):
{_history_lines(history)}

Master Agent's feedback on your earlier requests (find a NEW solution for anything declined):
{feedback_lines}

Agents you may ask the Master Agent to rerun:
{agent_lines}

Every evidence quote must be copied EXACTLY from the paper text, the script, the run output, the evaluation
findings or the data facts above. Do not invent facts. Choose "stop" when the gap is within
{EXACT_THRESHOLD_POINTS} point, when the paper states no number to compare with, or when no faithful change is left.

Return ONE JSON object:
{{
  "analysis": "<3-6 sentences: what causes the gap, ranked>",
  "hypotheses": [{{"cause": "<a cause of the gap>", "evidence": "<exact quote>", "fix": "<which agent, what change>"}}],
  "decision": {{"action": "rerun_agent|stop", "target_agent": "<one agent name from the list, or empty>",
               "instructions": ["<precise instruction for that agent>"], "target_entries": <integer or null>,
               "reason": "<why this step should close the gap>", "expected_effect": "<what should change>",
               "evidence": ["<exact quote>"]}}
}}
Return ONLY valid JSON. No explanation or markdown."""


def _decide(raw: dict, corpus: str, gap: float | None, evaluation_result: dict) -> tuple[dict, list, list]:
    """Turn the LLM's proposed decision into a checked one. Returns
    (decision, evidence [{"quote", "verified"}], notes on any change)."""
    notes = []
    raw = raw if isinstance(raw, dict) else {}
    action = str(raw.get("action") or "stop")
    action = action if action in ACTIONS else "stop"
    target = canonical_agent(raw.get("target_agent"))
    instructions = [str(i).strip() for i in raw.get("instructions") or [] if str(i).strip()] \
        if isinstance(raw.get("instructions"), list) else []
    normalized_corpus = _normalize_ws(corpus)
    quotes = raw.get("evidence") if isinstance(raw.get("evidence"), list) else []
    evidence = [{"quote": str(q), "verified": len(_normalize_ws(str(q))) >= 4
                 and _normalize_ws(str(q)) in normalized_corpus} for q in quotes[:MAX_EVIDENCE] if str(q).strip()]
    try:
        target_entries = min(int(raw.get("target_entries")), MAX_MASTER_ROWS) if raw.get("target_entries") else None
    except (TypeError, ValueError):
        target_entries = None

    published = [m for m in evaluation_result.get("metrics", []) if m.get("published_value") is not None]
    if action == "rerun_agent" and not published:
        notes.append("No published metric was extracted from the paper, so there is no gap to close; stopping.")
        action = "stop"
    elif action == "rerun_agent" and gap is not None and gap <= EXACT_THRESHOLD_POINTS:
        notes.append(f"The gap ({gap} points) is already within {EXACT_THRESHOLD_POINTS} point; stopping.")
        action = "stop"
    if action == "rerun_agent" and not target:
        notes.append(f"Unknown target agent '{raw.get('target_agent')}'; stopping.")
        action = "stop"
    if target != "Data Agent":
        target_entries = None

    return {
        "action": action,
        "target_agent": target if action == "rerun_agent" else "",
        "instructions": instructions if action == "rerun_agent" else [],
        "target_entries": target_entries if action == "rerun_agent" else None,
        "reason": str(raw.get("reason") or ""),
        "expected_effect": str(raw.get("expected_effect") or ""),
    }, evidence, notes


def _request_for(decision: dict, evidence: list) -> list:
    """The structured request to Master Agent (CLAUDE.md's agent message
    model), or [] when the decision is "stop"."""
    if decision["action"] != "rerun_agent":
        return []
    return [{
        "sender": "experiment_agent",
        "recipient": "master_agent",
        "message_type": "request",
        "status": "needs_review",
        "requires_review": True,
        "action": "rerun_agent",
        "target_agent": decision["target_agent"],
        "goal": {"target_entries": decision["target_entries"]} if decision["target_entries"] else {},
        "instructions": decision["instructions"],
        "summary": decision["reason"],
        "expected_effect": decision["expected_effect"],
        "evidence": [{"issue": "quoted by the Experiment Agent", "evidence": e["quote"]}
                     for e in evidence if e["verified"]],
    }]


def run(parent_paper: dict, replication_result: dict, evaluation_result: dict, cleaning_result: dict = None,
        dataset_result: dict = None, code_discovery_result: dict = None, history: list = None,
        feedback: list = None) -> dict:
    """
    Analyze the gap between the reproduced and published scores and return
    ONE request for Master Agent (see the module docstring).

    history lists the reruns already done in this run, with their effect
    on the gap; feedback lists Master Agent's reasons for declining earlier
    requests. cleaning_result and dataset_result are accepted for the same
    call shape as the Evaluation Agent; the data facts come from its review.
    """
    parent_paper = parent_paper or {}
    replication_result = replication_result or {}
    evaluation_result = evaluation_result or {}
    gap = score_gap(evaluation_result)
    print(f"\n[Experiment Agent] Analyzing the gap to '{parent_paper.get('title', '')}' "
          f"(mean gap: {gap if gap is not None else 'not measurable'} points)...")

    paper = paper_fulltext.get_parent_paper_text(parent_paper) if parent_paper else {"status": "", "text": ""}
    _script_path, script = _read_baseline_script(replication_result)
    stdout = _read_stdout(replication_result.get("stdout_path", ""))

    raw, proposed_by, notes = {}, "", []
    try:
        content = llm.chat(_prompt(parent_paper, paper, replication_result, script, stdout, evaluation_result,
                                   code_discovery_result, gap, history, feedback),
                           timeout=PROPOSAL_TIMEOUT_SECONDS, effort="high", agent="experiment_agent")
        raw = llm.parse_json(content, default={})
        raw = raw if isinstance(raw, dict) else {}
        proposed_by = llm.last_served_by()
    except Exception as e:
        notes.append(f"The analysis call failed: {e}")

    review = evaluation_result.get("review") or {}
    corpus = "\n".join([
        paper.get("text", ""), script, stdout, str(review.get("assessment") or ""),
        json.dumps(review.get("data_facts") or {}, default=str),
        *(f"{f.get('issue', '')} {f.get('evidence', '')} {f.get('fix', '')}"
          for f in review.get("static_findings", []) + review.get("findings", [])),
        *(f"{m['metric']} {m['published_value']} {m['reproduced_value']}" for m in evaluation_result.get("metrics", [])),
    ])
    decision, evidence, decision_notes = _decide(raw.get("decision"), corpus, gap, evaluation_result)
    hypotheses = [{"cause": str(h.get("cause") or ""), "evidence": str(h.get("evidence") or ""),
                   "fix": str(h.get("fix") or ""),
                   "verified": _normalize_ws(str(h.get("evidence") or "")) in _normalize_ws(corpus)
                   and len(_normalize_ws(str(h.get("evidence") or ""))) >= 4}
                  for h in raw.get("hypotheses") or [] if isinstance(h, dict)]

    status = "REQUEST" if decision["action"] == "rerun_agent" else "STOP"
    if decision["action"] == "rerun_agent":
        print(f"[Experiment Agent] Requesting a rerun of the {decision['target_agent']}: "
              f"{'; '.join(decision['instructions']) or 'no instructions'}")
    else:
        print(f"[Experiment Agent] No further rerun requested. {' '.join(notes + decision_notes)}")

    return {
        "status": status,
        "gap_points": gap,
        "analysis": str(raw.get("analysis") or ""),
        "hypotheses": hypotheses,
        "decision": decision,
        "evidence": evidence,
        "decision_notes": notes + decision_notes,
        "requests": _request_for(decision, evidence),
        "feedback_received": list(feedback or []),
        "paper_text_status": paper.get("status", ""),
        "proposed_by": proposed_by,
    }


def handoff(result: dict, context: dict) -> list:
    """The Experiment Agent's request to Master Agent when it finishes: its
    rerun request, or nothing when it decided to stop."""
    return list((result or {}).get("requests") or [])[:1]
