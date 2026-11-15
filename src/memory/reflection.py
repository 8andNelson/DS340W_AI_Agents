"""
Insight memories, step 3 (docs/CLAUDE_BACKEND_PLAN.md, section 18.3.3):
each agent's reflection on a finished run.

Code-made insights (setting changes, the published-score leakage warning)
cover what can be measured directly. Reflection covers the rest: what an
agent learned about its OWN job that would change what it does next time.

Once per finished run (after retain, so a retry never repeats it), one LLM
call per agent in REFLECTING_AGENTS that ran gets four blocks:

  CONTEXT    the run and task scope -- shown, never quotable;
  MEASURED   measured results: rerun metrics before/after, final metrics;
  FACTS      what was recorded: attempts, outputs, verified findings, the
             final script (MEASURED is part of the facts);
  PROPOSALS  suggestions and opinions that may never have been tried or
             tested: fixes, instructions, hypotheses, analyses, assessments,
             likely causes (PROPOSAL_KEYS).

Each lesson states its basis -- "measured" (an effect seen in MEASURED),
"observed" (a recorded fact) or "hypothesis" (a plausible cause not
tested) -- and is stored only when, deterministically:

  * every quote occurs in FACTS or PROPOSALS and has at least
    MIN_QUOTE_CHARS characters and MIN_QUOTE_WORDS words;
  * at least one quote is a FACT, and a "measured" lesson quotes MEASURED;
  * every number in its claim, conditions and action occurs in FACTS
    (rounded to the decimals written, or on the other percent/fraction
    scale);
  * its claim is specific and its polarity and basis are known.

Three live failures shaped these rules (run 20261115T171901Z): a proposed
"try 50 or 100 features" came back as a lesson that it "improved recall";
the analyst's opinion "the most likely cause" came back as "directly
caused"; and the Data Agent quoted labels such as "random_forest" from the
task scope as evidence for lessons about the model, not its own job.

A stored lesson is an active "lesson" memory of that agent, scoped by the
run's task features, with its basis in the title and "seen once" in the
summary (step 4 adds confidence from repeats). Replaces the curator
(MEMORY_REFLECTION=false brings it back). Fail-soft: a failed call or
check stores nothing and never stops retain; every outcome is reported.
"""
import json
import re
import uuid

REFLECTING_AGENTS = ("Data Agent", "Replication Agent", "Evaluation Agent", "Experiment Agent")
# Each agent reflects on its own job only.
ROLES = {
    "Data Agent": "finding, choosing and downloading the dataset(s) that reproduce the Parent Paper's experiment",
    "Replication Agent": "writing, running, repairing and revising the baseline script that reproduces the paper",
    "Evaluation Agent": "reading the paper's published results, comparing them with the reproduction and reviewing "
                        "the baseline for validity (leakage, method mismatch, too little data)",
    "Experiment Agent": "choosing the next change (which agent to rerun, with which instructions) to bring the "
                        "reproduced score closer to the paper's",
}
MAX_LESSONS = 3
MIN_QUOTE_CHARS = 25
MIN_QUOTE_WORDS = 4
MIN_CLAIM_CHARS = 25
BASES = ("measured", "observed", "hypothesis")
# Stored as candidates: recorded, not recalled, until a person approves one
# (python -m src.memory approve <id>) or step 4 confirms it in another run.
# The checks prove quotes and numbers, not that a claim's reasoning is right:
# live, a lesson said recall "dropped" from 0.0541 to 0.0324 when it rose.
LESSON_STATE = "candidate"
QUALITY = {"measured": 0.7, "observed": 0.6, "hypothesis": 0.4}  # all verified; seen in one run only
BASIS_LABEL = {"measured": "Measured", "observed": "Observed", "hypothesis": "Hypothesis (not tested)"}
SCRIPT_CHARS = 6000
OUTPUT_CHARS = 6000
REPORT_EXCERPT_CHARS = 600
TIMEOUT_SECONDS = 300
PROPOSAL_KEYS = {"fix", "instructions", "revision_instructions", "hypotheses", "expected_effect", "guidance",
                 "analysis", "assessment", "likely_causes", "reason"}

# What of each agent's final output is worth reflecting on.
OUTPUT_KEYS = {
    "Data Agent": ("mode", "data_plan", "parent_dataset_found", "paper_text_status", "selected_datasets",
                   "total_entries", "target_entries", "warnings", "notes"),
    "Replication Agent": ("status", "source", "repo_url", "exit_code", "duration_seconds", "notes",
                          "inferred_assumptions", "generated_by", "repository_fallback", "warnings",
                          "revision_round", "revision_instructions", "fast_mode_from"),
    "Evaluation Agent": ("status", "metrics", "likely_causes", "decision", "published_source",
                         "published_extraction", "published_leakage_check", "review_status"),
    "Experiment Agent": ("analysis", "hypotheses", "decision", "decision_notes", "gap_points"),
}
DATASET_KEYS = ("display_name", "source", "entry_count", "matches_parent_paper", "access_method")
# The reruns each agent's lessons can be about.
RERUNS_OF = {"Data Agent": ("Data Agent", "Cleaning Agent")}
QUOTE_MARKS = "\"'“”‘’ "
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])(\d+(?:\.\d+)?)(%?)")


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def _json(value) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)


def _quote(text) -> str:
    """A quote without the marks or trailing ellipsis a model wraps it in."""
    return str(text or "").strip().strip(QUOTE_MARKS).rstrip(".… ").strip(QUOTE_MARKS)


def _trim(output: dict, agent: str) -> dict:
    keep = {k: output.get(k) for k in OUTPUT_KEYS.get(agent, ()) if output.get(k) not in (None, "", [], {})}
    if "selected_datasets" in keep:
        keep["selected_datasets"] = [{k: d.get(k) for k in DATASET_KEYS if d.get(k) is not None}
                                     for d in keep["selected_datasets"] if isinstance(d, dict)]
    if agent == "Evaluation Agent":
        review = output.get("review") or {}
        keep["verified_findings"] = [f for f in review.get("findings") or [] if f.get("verified")]
        keep["leakage_checks"] = review.get("static_findings") or []
        keep["assessment"] = review.get("assessment")
    return keep


def _split(value, path: str, proposals: list):
    """value without its PROPOSAL_KEYS; each removed value goes to proposals."""
    if isinstance(value, dict):
        kept = {}
        for k, v in value.items():
            if k in PROPOSAL_KEYS and v not in (None, "", [], {}):
                proposals.append(f"- {path}.{k}: {v if isinstance(v, str) else _json(v)}")
            else:
                kept[k] = _split(v, f"{path}.{k}", proposals)
        return kept
    if isinstance(value, list):
        return [_split(v, f"{path}[{i}]", proposals) for i, v in enumerate(value)]
    return value


def _metric_lines(evaluation: dict) -> list:
    return [f"- {m.get('metric')}: reproduced {m.get('reproduced_value')}, published {m.get('published_value')}, "
            f"difference {m.get('difference_points')} points, {m.get('classification')}"
            for m in evaluation.get("metrics") or [] if isinstance(m, dict)]


def records(run: dict, agent: str, insight_titles: list) -> dict:
    """The blocks an agent reflects on: {"context", "measured", "facts",
    "proposals"} (strings; "facts" includes "measured")."""
    from src.memory import features, retain

    proposals, measured, facts = [], [], []
    context = [f"Run topic: {run.get('topic')}", f"Parent Paper: {(run.get('parent_paper') or {}).get('title', '')}",
               f"Run status: {run.get('status')}", f"Task scope: {_json(features.task_features(run, agent))}",
               "Insights already recorded by code (do not restate these):"]
    context += [f"- {t}" for t in insight_titles] or ["- none"]

    evaluation = run.get("final_evaluation") or {}
    if _metric_lines(evaluation):
        measured += ["Final metrics of the run:"] + _metric_lines(evaluation)
    reruns = [r for r in run.get("reruns") or [] if isinstance(r, dict)
              and (agent not in RERUNS_OF or r.get("target_agent") in RERUNS_OF[agent])]
    if reruns:
        measured += ["Reruns (requested by an agent, run by Master Agent) and their measured result:"]
        measured += [f"- rerun {r.get('rerun')}: the {r.get('requested_by')} asked for the {r.get('target_agent')}; "
                     f"metrics {r.get('metrics_before')} -> {r.get('metrics_after')}; gap {r.get('gap_before')} -> "
                     f"{r.get('gap_after')}; {'kept' if r.get('applied') else 'not kept'}" for r in reruns]
        proposals += [f"- rerun {r.get('rerun')} instructions: {'; '.join(map(str, r.get('instructions') or []))[:800]}"
                      for r in reruns if r.get("instructions")]

    stages = [s for s in run["stages"] if s["agent"] == agent]
    approved = [s for s in stages if s["decision"] == "Approved" and isinstance(s["output"], dict)]
    facts += ["Your attempts:"] + [
        f"- attempt {s['attempt']}: {s['decision']}" + (f" -- {str(s['report_text'])[:REPORT_EXCERPT_CHARS]}"
                                                         if s["decision"] != "Approved" and s.get("report_text") else "")
        for s in stages]
    if approved:
        facts += ["", "Your final output:", _json(_split(_trim(approved[-1]["output"], agent), "your final output",
                                                         proposals))[:OUTPUT_CHARS]]
    if agent == "Experiment Agent" and len(approved) > 1:
        facts += ["", "Your earlier decisions:"] + [
            f"- {_json(_split(_trim(s['output'], agent), f'decision {n}', proposals))[:1500]}"
            for n, s in enumerate(approved[:-1], 1)]
    if agent != "Data Agent":
        # What each earlier evaluation found (verified against the code), which
        # is what the reruns acted on; a finding's suggested fix is a proposal.
        evaluations = [s for s in run["stages"] if s["agent"] == "Evaluation Agent" and s["decision"] == "Approved"
                       and isinstance(s["output"], dict)][:-1]
        found = [(s["attempt"], f) for s in evaluations for f in (s["output"].get("review") or {}).get("findings") or []
                 if isinstance(f, dict) and f.get("verified")]
        found += [(s["attempt"], f) for s in evaluations
                  for f in (s["output"].get("review") or {}).get("static_findings") or [] if isinstance(f, dict)]
        if found:
            facts += ["", "Earlier evaluations' verified findings:"]
            facts += [f"- evaluation attempt {a}: {_json(_split(f, f'finding (evaluation attempt {a})', proposals))}"
                      for a, f in found]
        if agent != "Evaluation Agent" and evaluation:
            facts += ["", "Final evaluation:", _json(_split(_trim(evaluation, "Evaluation Agent"), "final evaluation",
                                                            proposals))[:OUTPUT_CHARS]]
    if agent in ("Replication Agent", "Experiment Agent"):
        replications = [s for s in run["stages"] if s["agent"] == "Replication Agent" and s["decision"] == "Approved"]
        script = retain._script(run, replications[-1]) if replications else None
        if script:
            facts += ["", "Final baseline script:", "```python", script[:SCRIPT_CHARS], "```"]
    measured_text = "\n".join(measured)
    return {"context": "\n".join(context), "measured": measured_text,
            "facts": "\n".join(filter(None, [measured_text, "\n".join(facts)])), "proposals": "\n".join(proposals)}


def prompt(agent: str, blocks: dict) -> str:
    return f"""You are the {agent} of a multi-agent pipeline that reproduces machine-learning papers. Your job is
{ROLES.get(agent, 'your stage of the pipeline')}.

Below are the records of one finished run. Write what your future self should know before a later run of a
similar task: lessons about YOUR OWN JOB that would CHANGE what you do. At most {MAX_LESSONS}; fewer is better
than weak ones; none is fine. A good lesson names the specific setting, data trait, paper trait or check. Do not
write generic advice, lessons about another agent's job, or what the insights already recorded say.

Each lesson has a basis:
- "measured": the effect was measured -- quote the result from MEASURED.
- "observed": a recorded fact, not an effect -- quote it from FACTS.
- "hypothesis": a plausible cause that was NOT tested. Say so; never write it as a proven cause.
PROPOSALS are suggestions and opinions that may never have been tried: never present one as a result.

Rules:
- Every number you write must appear in FACTS.
- evidence: 1-3 quotes copied EXACTLY (each at least {MIN_QUOTE_WORDS} words), at least one from FACTS.
- polarity: "positive" (do this again) or "negative" (avoid this / watch for it).

CONTEXT (not quotable):
{blocks['context']}

MEASURED:
\"\"\"
{blocks['measured'] or 'none'}
\"\"\"

FACTS (MEASURED is part of them):
\"\"\"
{blocks['facts']}
\"\"\"

PROPOSALS (suggested or opinion, not measured):
\"\"\"
{blocks['proposals'] or 'none'}
\"\"\"

Return ONLY this JSON:
{{"lessons": [{{"claim": "<one specific sentence>", "basis": "measured|observed|hypothesis",
  "applies_when": "<model family, data traits, paper traits>", "action": "<what to do or avoid next time>",
  "polarity": "positive|negative", "evidence": ["<exact quote>", "..."]}}]}}"""


def _numbers(text: str) -> list:
    """(value, decimals) of each number written in text; identifiers such as
    F1 or rerun ids glued to letters are not numbers."""
    return [(float(n), len(n.split(".")[1]) if "." in n else 0) for n, _pct in _NUMBER.findall(text or "")]


def numbers_grounded(text: str, source: str) -> list:
    """The numbers in text that do NOT occur in source (empty = grounded).
    A number matches when a source number rounds to it at the decimals
    written, on the same scale or the other percent/fraction scale."""
    known = [value for value, _d in _numbers(source)]
    missing = []
    for value, decimals in _numbers(text):
        if not any(abs(round(k * scale, decimals) - value) < 1e-9 for k in known for scale in (1, 100, 0.01)):
            missing.append(value)
    return missing


def check(lesson, blocks: dict) -> str:
    """Why a lesson cannot be stored, or "" when it passes every check."""
    if not isinstance(lesson, dict):
        return "not an object"
    claim = str(lesson.get("claim") or "").strip()
    if len(claim) < MIN_CLAIM_CHARS:
        return "the claim is missing or too short to be specific"
    if lesson.get("polarity") not in ("positive", "negative"):
        return "unknown polarity"
    if lesson.get("basis") not in BASES:
        return "unknown basis (measured, observed or hypothesis)"
    quotes = [_quote(q) for q in lesson.get("evidence") or [] if _quote(q)] \
        if isinstance(lesson.get("evidence"), list) else []
    if not quotes:
        return "no evidence quote"
    facts, proposals, measured = (_norm(blocks.get(k)) for k in ("facts", "proposals", "measured"))
    for quote in quotes:
        q = _norm(quote)
        if len(q) < MIN_QUOTE_CHARS or len(q.split()) < MIN_QUOTE_WORDS:
            return f"evidence too short to support a claim: {quote[:80]!r}"
        if q not in facts and q not in proposals:
            return f"evidence not found in the records: {quote[:80]!r}"
    if not any(_norm(q) in facts for q in quotes):
        return "every quote is from a proposal; a lesson needs a recorded fact"
    if lesson["basis"] == "measured" and not any(_norm(q) in measured for q in quotes):
        return "a measured lesson must quote a measured result"
    missing = numbers_grounded(" ".join(str(lesson.get(k) or "") for k in ("claim", "applies_when", "action")),
                               blocks.get("facts"))
    if missing:
        return f"number(s) not in the records: {', '.join(map(str, missing[:5]))}"
    return ""


def _row(run: dict, agent: str, n: int, lesson: dict) -> dict:
    from src.memory import features, retain
    replications = [s for s in run["stages"] if s["agent"] == "Replication Agent" and s["decision"] == "Approved"]
    script = retain._script(run, replications[-1]) if replications and \
        agent in ("Replication Agent", "Experiment Agent") else None
    task = features.task_features(run, agent, script=script)
    claim, when, action = (str(lesson.get(k) or "").strip() for k in ("claim", "applies_when", "action"))
    basis = lesson["basis"]
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"reflection|{run['run_id']}|{agent}|{n}|{claim}")),
        "kind": "lesson", "agent": agent, "run_id": run["run_id"], "polarity": lesson["polarity"],
        "title": f"Lesson ({basis}): {claim}"[:200],
        "summary": f"{BASIS_LABEL[basis]}: {claim} Applies when: {when or 'not stated'}. "
                   f"Next time: {action or 'not stated'}. Seen once (run {run['run_id']}).",
        "content": json.dumps({"claim": claim, "basis": basis, "applies_when": when, "action": action,
                               "evidence": lesson["evidence"]}, indent=1, ensure_ascii=False),
        "task": task, "tags": features.tags(task), "state": LESSON_STATE, "lineage": "kept",
        "quality": QUALITY[basis],
        "quality_reasons": {"formula": "a reflection lesson starts at 0.7 (measured), 0.6 (observed) or 0.4 "
                                       "(hypothesis) and moves only through use and repeats",
                            "basis": {"value": QUALITY[basis], "reason": BASIS_LABEL[basis]},
                            "evidence": {"value": len(lesson["evidence"]),
                                         "reason": "exact quotes and every number verified in the run's records"}},
        "evidence": {"rule": "reflection", "run_id": run["run_id"], "quotes": lesson["evidence"]},
        "outcome": {"seen_in_runs": 1, "basis": basis},
    }


def reflect_run(run: dict, written: list) -> dict:
    """One reflection per agent that ran. Returns {"lessons": [ids],
    "rejected": [{"agent", "claim", "reason"}], "errors": [...],
    "answered": {agent: lessons proposed}}."""
    from src import llm, memory
    report = {"lessons": [], "rows": [], "rejected": [], "errors": [], "answered": {}}
    ran = {s["agent"] for s in run.get("stages") or [] if s["decision"] == "Approved"}
    for agent in REFLECTING_AGENTS:
        if agent not in ran:
            continue
        try:
            titles = [m["title"] for m in written if m.get("agent") == agent and m.get("kind") in ("lesson", "failure")]
            blocks = records(run, agent, titles)
            answer = llm.parse_json(llm.chat(prompt(agent, blocks), timeout=TIMEOUT_SECONDS, agent="master_agent"),
                                    default={})
        except Exception as e:
            report["errors"].append(f"{agent}: {type(e).__name__}: {e}")
            print(f"[Memory] Reflection of the {agent} failed for run {run.get('run_id')}: {e}")
            continue
        lessons = answer.get("lessons") if isinstance(answer, dict) else None
        if not isinstance(lessons, list):
            report["errors"].append(f"{agent}: the answer had no lessons list")
            print(f"[Memory] Reflection of the {agent}: the answer could not be read as lessons.")
            continue
        report["answered"][agent] = len(lessons)
        for n, lesson in enumerate(lessons[:MAX_LESSONS]):
            if isinstance(lesson, dict) and isinstance(lesson.get("evidence"), list):
                lesson["evidence"] = [_quote(q) for q in lesson["evidence"] if _quote(q)]  # as stored
            reason = check(lesson, blocks)
            if reason:
                report["rejected"].append({"agent": agent, "claim": str(lesson.get("claim", ""))[:200]
                                           if isinstance(lesson, dict) else "", "reason": reason})
                continue
            row = _row(run, agent, n, lesson)
            memory_id = memory.put_memory(row)
            if memory_id:
                report["lessons"].append(memory_id)
                report["rows"].append(row)  # for the confidence step (confirm.py)
    for item in report["rejected"]:
        print(f"[Memory] Reflection lesson of the {item['agent']} not stored ({item['reason']}): {item['claim'][:100]}")
    return report
