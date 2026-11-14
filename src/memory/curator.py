"""
Curator (milestone M6; docs/CLAUDE_BACKEND_PLAN.md, sections 8.3 and 8.4):
after a run is retained, one LLM call reads the run's new memories and

  * writes at most MAX_LESSONS general lessons -- rules that can help a
    later run -- each citing the memories it comes from with an exact quote;
  * may say that one of its new memories SUPPORTS an existing candidate
    lesson from another run, or CONTRADICTS an existing lesson.

Every quote is checked deterministically: it must occur in the cited
memory's title, summary or content (whitespace and case aside). A lesson
with an unverified quote is not stored; an unverified support or
contradiction is ignored -- the same rule as the Evaluation Agent's _verify.

A lesson is a general claim made by an LLM, so it starts as "candidate" and
is never given to agents. It becomes "active" when a person approves it, or
when a second, independent run supports it with a verified quote (its
seen_count reaches 2; a run cannot support its own lesson). A verified
contradiction links the two memories and lowers both scores by 0.2
(quality_reasons.contradicted_by), so recall shows them together as
conflicting cases.

Lessons start with quality 0.5 and move only through use (ranking.py).
MEMORY_CURATOR=false turns this off. Every function fails soft.
"""
import json
import re
import uuid

from src.memory import features

MAX_LESSONS = 5
MAX_CASES = 40
MAX_CANDIDATES = 30
CASE_EXCERPT_CHARS = 700
LESSON_QUALITY = 0.5
CONTRADICTION_PENALTY = 0.2
ACTIVATION_RUNS = 2


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def quote_found(quote: str, memory: dict) -> bool:
    """The quote (at least 12 characters) occurs in the memory's own text."""
    q = _norm(quote)
    if len(q) < 12:
        return False
    return q in _norm(" ".join(str(memory.get(k) or "") for k in ("title", "summary", "content")))


def _prompt(cases: list, lessons: list) -> str:
    case_lines = "\n\n".join(
        f"[N{n}] {m['agent']} -- {m['kind']}/{m['polarity']}: {m['title']}\nSummary: {m['summary']}\n"
        f"Excerpt: {(m.get('content') or '')[:CASE_EXCERPT_CHARS]}" for n, m in enumerate(cases, 1))
    lesson_lines = "\n".join(f"[L{n}] {m['agent']} -- {m['title']}: {m['summary']}" for n, m in enumerate(lessons, 1)) \
        or "(none)"
    return f"""You are the curator of an AI research pipeline's shared memory. Below are the memories made from one
finished run ([N] cases) and the lessons other runs proposed that are not yet confirmed ([L] lessons).

CASES FROM THIS RUN:
{case_lines}

UNCONFIRMED LESSONS FROM OTHER RUNS:
{lesson_lines}

Tasks:
1. Write at most {MAX_LESSONS} general LESSONS that can help a LATER run of a different paper: a practice to follow
   (positive) or a pattern to avoid (negative). A lesson must be a general rule, not a summary of this run. Each
   lesson must cite the cases it comes from, with an EXACT quote copied from each cited case's text above.
2. If a case from this run shows that an unconfirmed lesson holds, list it under "supports", with an exact quote
   from the case. If a case shows that a lesson is wrong, list it under "contradicts", with an exact quote.
Do not invent facts, numbers, papers, datasets or URLs. Write nothing you cannot quote. Empty lists are fine.

Return only JSON:
{{"lessons": [{{"agent": "<the agent the lesson is for, as named in its cases>", "polarity": "positive|negative",
   "title": "<short rule>", "rule": "<one or two sentences>", "evidence": [{{"case": "N1", "quote": "<exact text>"}}]}}],
 "supports": [{{"lesson": "L1", "case": "N2", "quote": "<exact text>"}}],
 "contradicts": [{{"lesson": "L1", "case": "N3", "quote": "<exact text>"}}]}}"""


def _merged_task(cited: list, agent: str) -> dict:
    task = {"agent": agent}
    for key in ("model_family", "techniques", "metrics", "target_terms"):
        values = sorted({v for m in cited for v in (m.get("task") or {}).get(key) or []})
        if values:
            task[key] = values
    for key in ("task_type", "failure_signature", "source"):
        value = next(((m.get("task") or {}).get(key) for m in cited if (m.get("task") or {}).get(key)), None)
        if value:
            task[key] = value
    return task


def curate_run(run_id: str, new_memories: list) -> dict:
    """Run the curator over one run's new memories. Returns what it stored:
    {"lessons": [...], "rejected": [...], "supported": [...], "activated": [...], "contradicted": [...]}."""
    from src import llm, memory
    from src.config import MEMORY_CURATOR

    report = {"lessons": [], "rejected": [], "supported": [], "activated": [], "contradicted": []}
    cases = [m for m in new_memories if m.get("kind") != "lesson"][:MAX_CASES]
    if not MEMORY_CURATOR or not cases:
        return report
    try:
        agents = sorted({m["agent"] for m in cases})
        lessons = [m for m in memory.find_memories(kind="lesson", states=("candidate", "active"), limit=1000)
                   if m["agent"] in agents and m.get("run_id") != run_id][:MAX_CANDIDATES]
        try:
            answer = llm.parse_json(llm.chat(_prompt(cases, lessons), timeout=180, agent="master_agent"), default={})
        except Exception as e:
            print(f"[Memory] Curator call failed for run {run_id}: {e}")
            return report
        answer = answer if isinstance(answer, dict) else {}

        def case(label):
            m = re.fullmatch(r"N(\d+)", str(label or "").strip(), re.I)
            return cases[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(cases) else None

        def lesson(label):
            m = re.fullmatch(r"L(\d+)", str(label or "").strip(), re.I)
            return lessons[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(lessons) else None

        for i, item in enumerate((answer.get("lessons") or [])[:MAX_LESSONS]):
            if not isinstance(item, dict):
                continue
            evidence = [(case(e.get("case")), e.get("quote")) for e in item.get("evidence") or [] if isinstance(e, dict)]
            verified = [(m, q) for m, q in evidence if m and quote_found(q, m)]
            agent = str(item.get("agent") or "")
            problem = None
            if not evidence or len(verified) != len(evidence):
                problem = "a cited quote was not found in its case"
            elif agent not in {m["agent"] for m, _ in verified}:
                problem = f"agent '{agent}' is not the agent of its cited cases"
            elif item.get("polarity") not in ("positive", "negative") or not item.get("title") or not item.get("rule"):
                problem = "missing polarity, title or rule"
            if problem:
                report["rejected"].append({"title": item.get("title"), "reason": problem})
                continue
            cited = [m for m, _ in verified]
            task = _merged_task(cited, agent)
            memory_id = memory.put_memory({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"lesson|{run_id}|{i}|{item['title']}")),
                "kind": "lesson", "agent": agent, "run_id": run_id, "polarity": item["polarity"],
                "title": str(item["title"])[:200], "summary": str(item["rule"]),
                "content": json.dumps({"rule": item["rule"], "evidence": [{"memory_id": m["id"], "quote": q}
                                                                          for m, q in verified]}, indent=1),
                "task": task, "tags": features.tags(task), "state": "candidate", "lineage": "kept",
                "quality": LESSON_QUALITY,
                "quality_reasons": {"formula": "a new curator lesson starts at 0.5 and moves only through use",
                                    "evidence": {"value": len(verified), "reason": "exact quotes verified in cited memories"}},
                "evidence": {"cited": [{"memory_id": m["id"], "quote": q} for m, q in verified], "run_id": run_id,
                             "rule": "curator"},
                "outcome": {}})
            if memory_id:
                report["lessons"].append(memory_id)

        for item in answer.get("supports") or []:
            target, m = (lesson(item.get("lesson")), case(item.get("case"))) if isinstance(item, dict) else (None, None)
            if not target or not m or not quote_found(item.get("quote"), m):
                continue
            memory.increment(target["id"], seen_count=1)
            report["supported"].append(target["id"])
            if target["state"] == "candidate" and int(target.get("seen_count") or 1) + 1 >= ACTIVATION_RUNS:
                memory.set_state(target["id"], "active",
                                 f"supported by an independent run ({run_id}) with a verified quote", actor="curator")
                report["activated"].append(target["id"])

        for item in answer.get("contradicts") or []:
            target, m = (lesson(item.get("lesson")), case(item.get("case"))) if isinstance(item, dict) else (None, None)
            if not target or not m or not quote_found(item.get("quote"), m):
                continue
            for a, b in ((target, m), (m, target)):
                reasons = dict(a.get("quality_reasons") or {})
                reasons["contradicted_by"] = sorted(set(reasons.get("contradicted_by") or []) | {b["id"]})
                reasons["contradiction"] = {"value": -CONTRADICTION_PENALTY, "reason": f"contradicted by memory {b['id'][:8]}"}
                memory.client().write({"op_id": str(uuid.uuid4()), "type": "set_quality", "table": "memories",
                                       "key": {"id": a["id"]}, "quality": max(0.0, float(a.get("quality") or 0) - CONTRADICTION_PENALTY),
                                       "quality_reasons": reasons})
            report["contradicted"].append([target["id"], m["id"]])
        if report["lessons"] or report["supported"] or report["contradicted"]:
            print(f"[Memory] Curator: {len(report['lessons'])} candidate lesson(s), {len(report['supported'])} "
                  f"support(s) ({len(report['activated'])} activated), {len(report['contradicted'])} contradiction(s); "
                  f"{len(report['rejected'])} lesson(s) rejected (unverified).")
    except Exception as e:
        print(f"[Memory] WARNING: the curator failed for run {run_id}: {e}")
    return report
