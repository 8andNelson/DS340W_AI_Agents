"""
Learning from use (milestone M6; docs/CLAUDE_BACKEND_PLAN.md, section 9.3).

At run end, every past case an agent said it used gets an outcome --
what happened after it was used, from the run's own records:

  helped     the attempt was approved, its lineage is kept, and its output
             was not itself a warning (for example a baseline with
             verified leakage);
  hurt       the attempt failed its checks or raised, or its rerun was
             undone;
  not judged the lineage is unknown (a crash) or superseded -- no outcome
             is recorded, and the memory's score does not move.

A Master Agent diagnosis is judged by the failed agent's next attempt:
helped when it passed, hurt when it failed again.

Fast mode (an adapted past script) is judged at once by the Replication
Agent when the adapted script fails its first run (report_fast_mode_failure),
before any repair -- so a repaired run does not count it as helping.

Each outcome is applied once (the usage_outcome op sets memory_usage.helped
only while it is empty, and moves times_helped / times_hurt in the same
transaction). ranking.effective_quality turns the counts into the score
recall ranks by. "Used" is the model's self-report; an outcome is what
followed, not proof that the memory caused it.
"""
import uuid


def _client():
    from src import memory
    c = memory.client()
    return c if c and c.backend != "off" else None


def record_outcome(usage_id: str, memory_id: str, helped: bool) -> bool:
    """Apply one usage outcome (once). Fail-soft."""
    try:
        c = _client()
        if c is None:
            return False
        c.write({"op_id": str(uuid.uuid4()), "type": "usage_outcome", "table": "memory_usage",
                 "key": {"id": usage_id}, "memory_id": memory_id, "helped": bool(helped)})
        return True
    except Exception as e:
        print(f"[Memory] WARNING: could not record the outcome of memory {memory_id}: {e}")
        return False


def report_fast_mode_failure(usage_id: str | None, memory_id: str | None) -> bool:
    """The adapted past script failed its first run: the memory hurt."""
    if not usage_id or not memory_id:
        return False
    print(f"[Memory] The adapted past script (memory {memory_id[:8]}) failed its first run; its rank goes down.")
    return record_outcome(usage_id, memory_id, False)


def _outcome(row: dict, attempts: dict, by_agent: dict, lineage: dict, warnings: set) -> tuple[bool | None, str]:
    """(helped, why) for one used memory_usage row."""
    attempt = attempts.get(row.get("invocation_id"))
    if attempt is None:
        return None, "the attempt that used it is not recorded"
    if row.get("agent") == "Master Agent":
        later = [a for a in by_agent.get(attempt["agent"], []) if (a.get("attempt") or 0) > (attempt.get("attempt") or 0)]
        if not later:
            return None, "no later attempt of the failed agent"
        passed = min(later, key=lambda a: a.get("attempt") or 0).get("decision") == "Approved"
        return passed, "the next attempt passed" if passed else "the next attempt failed again"
    if attempt.get("decision") != "Approved":
        return False, f"the attempt was {attempt.get('decision')}"
    state = lineage.get(row["invocation_id"])
    if state == "reverted":
        return False, "the attempt's rerun was undone"
    if state != "kept":
        return None, f"lineage {state or 'unknown'}: not judged"
    if row["invocation_id"] in warnings:
        return False, "the attempt's output is itself a warning (e.g. verified leakage)"
    return True, "approved, kept, and not a warning"


def learn_run(run_id: str) -> dict:
    """Record the outcome of every past case this run used. Returns
    {"helped": n, "hurt": n, "not_judged": n}. Never raises."""
    report = {"helped": 0, "hurt": 0, "not_judged": 0}
    try:
        c = _client()
        if c is None or not run_id:
            return report
        usage = [u for u in c.local.rows("memory_usage", run_id=run_id) if u.get("used") and u.get("helped") is None]
        if not usage:
            return report
        events = c.local.rows("memory_events", run_id=run_id, event_type="attempt")
        attempts = {e["invocation_id"]: e for e in events if e.get("invocation_id")}
        by_agent = {}
        for e in events:
            by_agent.setdefault(e.get("agent"), []).append(e)
        lineage = {r["invocation_id"]: r["lineage"] for r in c.local.rows("invocation_lineage", run_id=run_id)}
        warnings = {m.get("invocation_id") for m in c.local.rows("memories", run_id=run_id)
                    if m.get("polarity") == "negative" and m.get("invocation_id")}
        for row in usage:
            helped, _why = _outcome(row, attempts, by_agent, lineage, warnings)
            if helped is None:
                report["not_judged"] += 1
                continue
            record_outcome(row["id"], row["memory_id"], helped)
            report["helped" if helped else "hurt"] += 1
    except Exception as e:
        print(f"[Memory] WARNING: could not learn from run {run_id}: {e}")
    return report
