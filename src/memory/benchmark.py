"""
Retrieval test set (docs/CLAUDE_BACKEND_PLAN.md, section 9.6): does recall
return the right memories?

tests/memory_retrieval_cases.json holds hand-labelled queries, each shaped
like what a recall point really sends (the Replication Agent's generation,
repair or revision; Master Agent's diagnosis), with:

  acceptable         memories that are right to show (one is enough)
  harmful            memories that must never be shown
  critical_warnings  warnings that must be shown when their conditions match
  expect_none        nothing matches: nothing should be shown

`python -m src.memory benchmark` runs every query through the real recall
code against the store, read-only (no counters, no usage rows), and checks
the release gate (proposed targets): an acceptable memory in the pack for at
least 80% of the queries that have one; no harmful memory shown; every
critical warning shown. It reports the number of queries and each miss.
"""
import json
from pathlib import Path

from src.memory import features, recall

CASES_FILE = Path(__file__).parent.parent.parent / "tests" / "memory_retrieval_cases.json"
TARGET_HIT_RATE = 0.8


def run_case(case: dict, active_ids: set) -> dict:
    query = {k: v for k, v in (case.get("query") or {}).items()}
    if case.get("failure"):
        query["failure_signature"] = features.failure_signature(case["failure"])
        query["text"] = (query.get("text") or case["failure"])[:2000]
    ctx = recall.MemoryContext(case["agent"], query, case["search_agents"],
                               kinds=tuple(case["kinds"]) if case.get("kinds") else None, max_requests=0)
    shown = [(i["group"], i["memory"]["id"]) for i in ctx.items]
    ids = {i for _, i in shown}
    warnings = {i for g, i in shown if g == "warning"}
    acceptable = set(case.get("acceptable") or [])
    hit = bool(acceptable & ids) if acceptable else None
    harmful = sorted(set(case.get("harmful") or []) & ids)
    missed = sorted(set(case.get("critical_warnings") or []) - warnings)
    none_ok = (not shown) if case.get("expect_none") else None
    labelled = acceptable | set(case.get("harmful") or []) | set(case.get("critical_warnings") or [])
    return {"id": case["id"], "description": case.get("description", ""), "shown": shown, "hit": hit,
            "harmful_shown": harmful, "critical_missed": missed, "none_ok": none_ok,
            "stale_labels": sorted(labelled - active_ids),
            "passed": hit is not False and not harmful and not missed and none_ok is not False}


def run(cases_file: Path = CASES_FILE) -> dict:
    from src import memory
    data = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    active_ids = {m["id"] for m in memory.find_memories(states=("active",), limit=100_000)}
    results = [run_case(case, active_ids) for case in data["cases"]]
    with_acceptable = [r for r in results if r["hit"] is not None]
    hit_rate = sum(r["hit"] for r in with_acceptable) / len(with_acceptable) if with_acceptable else None
    harmful = sum(len(r["harmful_shown"]) for r in results)
    missed = sum(len(r["critical_missed"]) for r in results)
    return {
        "status": data.get("status"), "labelled_by": data.get("labelled_by"), "queries": len(results),
        "queries_with_acceptable": len(with_acceptable), "hit_rate": hit_rate,
        "harmful_shown": harmful, "critical_warnings_missed": missed,
        "no_match_queries": sum(r["none_ok"] is not None for r in results),
        "no_match_failures": sum(r["none_ok"] is False for r in results),
        "gate_passed": (hit_rate is not None and hit_rate >= TARGET_HIT_RATE and harmful == 0 and missed == 0
                        and all(r["none_ok"] is not False for r in results)),
        "results": results,
    }


def format_report(report: dict) -> str:
    lines = [f"Retrieval test set: {report['queries']} queries ({report['status']}; labelled by "
             f"{report['labelled_by']})",
             f"  acceptable memory shown: {report['hit_rate']:.0%} of {report['queries_with_acceptable']} queries "
             f"(target {TARGET_HIT_RATE:.0%})" if report["hit_rate"] is not None else "  no query has an acceptable memory",
             f"  harmful memories shown: {report['harmful_shown']} (target 0)",
             f"  critical warnings missed: {report['critical_warnings_missed']} (target 0)",
             f"  no-match queries that showed something: {report['no_match_failures']} of {report['no_match_queries']}",
             f"  GATE: {'PASSED' if report['gate_passed'] else 'NOT PASSED'}", "", "Misses:"]
    for r in report["results"]:
        if not r["passed"] or r["stale_labels"]:
            problems = []
            if r["hit"] is False:
                problems.append("no acceptable memory shown")
            if r["harmful_shown"]:
                problems.append(f"harmful shown {', '.join(i[:8] for i in r['harmful_shown'])}")
            if r["critical_missed"]:
                problems.append(f"critical warning missed {', '.join(i[:8] for i in r['critical_missed'])}")
            if r["none_ok"] is False:
                problems.append("expected nothing, but something was shown")
            if r["stale_labels"]:
                problems.append(f"labels no longer active: {', '.join(i[:8] for i in r['stale_labels'])}")
            shown = ", ".join(f"{g[0]}:{i[:8]}" for g, i in r["shown"]) or "nothing"
            lines.append(f"  {r['id']}: {'; '.join(problems)} (shown: {shown}) -- {r['description']}")
    return "\n".join(lines)
