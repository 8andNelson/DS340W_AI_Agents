"""
Run recording (milestone M3; docs/CLAUDE_BACKEND_PLAN.md, sections 8.1 and 8.5).

Master Agent records what happens in a run as it happens -- each agent
attempt and its decision, each rerun opened and closed, each declined
request -- so a crash or a killed process loses nothing that happened
before it. Events go to the memory store like any other write (the local
file, and the hosted store's outbox when MEMORY_BACKEND=postgres).

At run end, resolve_lineage() gives every attempt its place in the run's
final result:

  kept        approved, and part of the final baseline: the last approved
              attempt of its agent that is not inside an undone rerun.
              (Every approved Experiment Agent analysis is kept: an
              analysis is a decision, not a part of the baseline.)
  superseded  approved, then replaced by a later kept attempt of the same
              agent (for example a Replication rerun that was kept).
  reverted    approved inside a rerun that was undone (reverted because the
              score moved away from the paper, or failed, or declined).
  rejected    failed its checks or raised (Retrying / Rejected).
  unknown     the record is not complete: the run crashed or was killed
              before its baseline was final, or the attempt is inside a
              rerun that never closed. Never treated as kept.

A run killed without its `finally` block is resolved when the next run
starts (recover_unfinished). Every function fails soft.
"""
import itertools
import threading
import uuid
from datetime import datetime, timezone

from src.memory import schema

NON_BASELINE_AGENTS = ("Experiment Agent",)
FINISHED_STATUSES = ("success", "halted")
MAX_TEXT_CHARS = 2000

_seq = itertools.count(1)
_seq_lock = threading.Lock()  # first-pass stages run in parallel worker threads


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _client():
    from src import memory
    c = memory.client()
    return c if c and c.backend != "off" else None


def _short(text) -> str | None:
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS] + " [cut]"


def _record(event_type: str, run_id: str | None, **fields) -> str | None:
    if not run_id:
        return None
    try:
        c = _client()
        if c is None:
            return None
        from src import memory
        with _seq_lock:
            seq = next(_seq)
        row = {"id": str(uuid.uuid4()), "run_id": run_id, "seq": seq, "event_type": event_type,
               "contributor": memory.MEMORY_CONTRIBUTOR or None, "created_at": _now(), **fields}
        row = memory._scrub({k: v for k, v in row.items() if v is not None})
        schema.check_row("memory_events", row)
        c.write({"op_id": str(uuid.uuid4()), "type": "insert", "table": "memory_events", "rows": [row]})
        return row["id"]
    except Exception as e:
        print(f"[Memory] WARNING: could not record the {event_type} event: {e}")
        return None


# --- Events (called by Master Agent and the run store) ---

def attempt(run_id, invocation_id, agent: str, attempt_no: int, decision: str,
            failed_checks: list = None, error: str = None, memory: dict = None) -> str | None:
    """One supervised agent attempt and Master Agent's decision on it
    (Approved / Retrying / Rejected); memory: the attempt's memory use
    (recalls, cases shown and used, requests), when it recalled any."""
    payload = {}
    if failed_checks:
        payload["failed_checks"] = [_short(c) for c in failed_checks][:20]
    if error:
        payload["error"] = _short(error)
    if memory:
        payload["memory"] = memory
    return _record("attempt", run_id, invocation_id=invocation_id, agent=agent, attempt=attempt_no,
                   decision=decision, payload=payload or None)


def rerun_opened(run_id, rerun: int, target: str, sender: str, action: str = "") -> str | None:
    return _record("rerun_opened", run_id, rerun=rerun, agent=target,
                   payload={"requested_by": sender, "action": action})


def rerun_closed(run_id, rerun: int, target: str, outcome: str, reason: str = "",
                 improved: bool = None) -> str | None:
    """outcome: "kept", "reverted" (score moved away) or "undone" (failed or declined)."""
    return _record("rerun_closed", run_id, rerun=rerun, agent=target, decision=outcome,
                   payload={"reason": _short(reason), "improved": improved})


def request_declined(run_id, sender: str, target: str, reason: str, redundant: bool = False) -> str | None:
    return _record("request_declined", run_id, agent=target, decision="declined",
                   payload={"requested_by": sender, "reason": _short(reason), "redundant": redundant})


def run_finished(run_id, status: str) -> list:
    """Record the run's end and resolve every attempt's lineage."""
    _record("run_finished", run_id, decision=status)
    return resolve_lineage(run_id, finished=status in FINISHED_STATUSES)


# --- Lineage ---

def lineage_for(events: list, finished: bool) -> list[dict]:
    """[{invocation_id, agent, attempt, lineage, reason}] from a run's events
    (see the module docstring for the rules). Events are ordered by time,
    then by seq: seq restarts in every process, and on Windows two events
    can share a timestamp."""
    rerun = None
    outcomes = {}
    attempts = []
    for e in sorted(events, key=lambda e: (str(e.get("created_at") or ""), e.get("seq") or 0)):
        if e["event_type"] == "rerun_opened":
            rerun = e.get("rerun")
        elif e["event_type"] == "rerun_closed":
            outcomes[e.get("rerun")] = (e.get("decision"), (e.get("payload") or {}).get("reason") or "")
            rerun = None
        elif e["event_type"] == "attempt" and e.get("invocation_id"):
            attempts.append((e, rerun))
    open_rerun = rerun
    in_open_rerun = {e["agent"] for e, r in attempts if open_rerun is not None and r == open_rerun}

    resolved = {}
    candidates = {}  # agent -> approved attempts that still count, in order
    for e, r in attempts:
        key = e["invocation_id"]
        if e.get("decision") != "Approved":
            resolved[key] = ("rejected", f"Master Agent decision: {e.get('decision')}")
        elif r is not None and r == open_rerun:
            resolved[key] = ("unknown", f"inside rerun {r}, which never closed")
        elif r is not None and outcomes.get(r, ("",))[0] != "kept":
            outcome, reason = outcomes.get(r, ("undone", ""))
            resolved[key] = ("reverted", f"rerun {r} was {outcome}: {reason}".strip(": "))
        else:
            candidates.setdefault(e["agent"], []).append((e, r))

    for agent, items in candidates.items():
        for i, (e, r) in enumerate(items):
            key = e["invocation_id"]
            where = f"rerun {r} (kept)" if r is not None else "the first pass"
            if not finished:
                resolved[key] = ("unknown", "the run did not finish normally, so its final baseline is unknown")
            elif agent in in_open_rerun:
                resolved[key] = ("unknown", f"the {agent} was rerun in rerun {open_rerun}, which never closed")
            elif agent in NON_BASELINE_AGENTS or i == len(items) - 1:
                resolved[key] = ("kept", f"approved in {where}; part of the final result")
            else:
                later = items[-1][1]
                resolved[key] = ("superseded", "replaced by a later kept attempt"
                                 + (f" (rerun {later})" if later is not None else ""))

    return [{"invocation_id": e["invocation_id"], "agent": e["agent"], "attempt": e.get("attempt"),
             "lineage": resolved[e["invocation_id"]][0], "reason": resolved[e["invocation_id"]][1]}
            for e, _ in attempts]


def resolve_lineage(run_id, finished: bool) -> list:
    """Write each attempt's lineage for one run, once. Returns the rows written."""
    if not run_id:
        return []
    try:
        c = _client()
        if c is None or c.local.rows("invocation_lineage", run_id=run_id):
            return []
        from src import memory
        events = c.local.rows("memory_events", run_id=run_id)
        now = _now()
        rows = [{**row, "run_id": run_id, "contributor": memory.MEMORY_CONTRIBUTOR or None, "resolved_at": now}
                for row in lineage_for(events, finished)]
        if rows:
            c.write({"op_id": str(uuid.uuid4()), "type": "insert", "table": "invocation_lineage",
                     "rows": [memory._scrub(r) for r in rows]})
        return rows
    except Exception as e:
        print(f"[Memory] WARNING: could not resolve the lineage of run {run_id}: {e}")
        return []


def recover_unfinished(exclude_run_id: str = None) -> list[str]:
    """Resolve runs that have events but no lineage -- a process killed
    before its `finally` block. Called when a new run starts (a machine runs
    one pipeline run at a time). Returns the run ids resolved."""
    try:
        c = _client()
        if c is None:
            return []
        with c.local._connect() as conn:
            run_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT run_id FROM memory_events WHERE run_id NOT IN "
                "(SELECT run_id FROM invocation_lineage) ORDER BY run_id")]
        recovered = []
        for run_id in run_ids:
            if run_id == exclude_run_id:
                continue
            finished = any(e.get("decision") in FINISHED_STATUSES
                           for e in c.local.rows("memory_events", run_id=run_id, event_type="run_finished"))
            if not finished:
                _record("run_finished", run_id, decision="crashed", payload={"recovered": True})
            resolve_lineage(run_id, finished=finished)
            recovered.append(run_id)
        return recovered
    except Exception as e:
        print(f"[Memory] WARNING: could not recover unfinished runs: {e}")
        return []
