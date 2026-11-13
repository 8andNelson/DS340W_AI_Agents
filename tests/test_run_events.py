"""Run recording (milestone M3): events written as they happen, and each
agent attempt's lineage at run end -- kept, superseded, reverted,
rejected or unknown."""
import json
from unittest.mock import patch

import pytest

from src import llm, memory
from src.agents import master_agent
from src.memory import events
from src.orchestration import run_store
from tests.test_master_reruns import asks, evaluation, make_ctx, stops


def _ev(event_type, n, **fields):
    return {"event_type": event_type, "seq": n, "created_at": f"2026-11-13T00:00:{n:02d}.000000+00:00", **fields}


def _attempt(n, agent, decision="Approved", attempt=1):
    return _ev("attempt", n, invocation_id=f"inv-{n}", agent=agent, attempt=attempt, decision=decision)


def _lineage(rows):
    return {r["invocation_id"]: r["lineage"] for r in rows}


# --- The lineage rules ---

def test_first_pass_attempts_are_kept_or_rejected():
    rows = events.lineage_for([_attempt(1, "Data Agent", "Retrying"), _attempt(2, "Data Agent", attempt=2),
                               _attempt(3, "Cleaning Agent")], finished=True)
    assert _lineage(rows) == {"inv-1": "rejected", "inv-2": "kept", "inv-3": "kept"}


def test_a_kept_rerun_supersedes_the_earlier_attempts_of_the_same_agents():
    rows = events.lineage_for([
        _attempt(1, "Replication Agent"), _attempt(2, "Evaluation Agent"), _attempt(3, "Experiment Agent"),
        _ev("rerun_opened", 4, rerun=1, agent="Replication Agent"),
        _attempt(5, "Replication Agent", attempt=2), _attempt(6, "Evaluation Agent", attempt=2),
        _ev("rerun_closed", 7, rerun=1, decision="kept"),
        _attempt(8, "Experiment Agent", attempt=2),
    ], finished=True)
    assert _lineage(rows) == {"inv-1": "superseded", "inv-2": "superseded", "inv-3": "kept",
                              "inv-5": "kept", "inv-6": "kept", "inv-8": "kept"}  # analyses are all kept


def test_an_undone_rerun_reverts_its_attempts_and_the_earlier_baseline_stays_kept():
    rows = events.lineage_for([
        _attempt(1, "Replication Agent"),
        _ev("rerun_opened", 2, rerun=1, agent="Replication Agent"),
        _attempt(3, "Replication Agent", attempt=2), _attempt(4, "Evaluation Agent"),
        _ev("rerun_closed", 5, rerun=1, decision="reverted", payload={"reason": "the score moved further"}),
    ], finished=True)
    assert _lineage(rows) == {"inv-1": "kept", "inv-3": "reverted", "inv-4": "reverted"}
    assert "reverted: the score moved further" in next(r["reason"] for r in rows if r["invocation_id"] == "inv-3")


def test_a_crash_makes_approved_attempts_unknown_never_kept():
    rows = events.lineage_for([_attempt(1, "Intake Agent"), _attempt(2, "Research Agent", "Retrying")],
                              finished=False)
    assert _lineage(rows) == {"inv-1": "unknown", "inv-2": "rejected"}


def test_a_rerun_that_never_closed_makes_that_agents_attempts_unknown():
    rows = events.lineage_for([
        _attempt(1, "Intake Agent"), _attempt(2, "Data Agent"),
        _ev("rerun_opened", 3, rerun=1, agent="Data Agent"), _attempt(4, "Data Agent", attempt=2),
    ], finished=True)
    assert _lineage(rows) == {"inv-1": "kept", "inv-2": "unknown", "inv-4": "unknown"}


def test_events_are_ordered_by_time_then_seq():
    # seq restarts in a new process; time decides, seq only breaks ties
    late = _attempt(1, "Data Agent", attempt=2)
    late["created_at"] = "2026-11-13T00:00:59.000000+00:00"
    rows = events.lineage_for([late, _attempt(5, "Data Agent")], finished=True)
    assert _lineage(rows) == {"inv-5": "superseded", "inv-1": "kept"}


# --- Recording through Master Agent and the run store ---

@pytest.fixture
def run_id(monkeypatch):
    rid = run_store.start_run("fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", rid)
    return rid


def _store():
    return memory.client().local


def test_supervised_attempts_are_recorded_with_their_timer_invocation_ids(run_id):
    outputs = iter([{"ok": False}, {"ok": True}])
    output, blockers = master_agent._supervise(
        "Data Agent", lambda: next(outputs),
        lambda out: [] if out["ok"] else [master_agent._log_entry("Correction", "Data Agent", "no dataset found",
                                                                  "Retrying", "retry")],
        lambda out: "report", max_attempts=2)
    assert output == {"ok": True} and not blockers

    recorded = _store().rows("memory_events", run_id=run_id, event_type="attempt")
    assert [(e["attempt"], e["decision"]) for e in sorted(recorded, key=lambda e: e["seq"])] == \
        [(1, "Retrying"), (2, "Approved")]
    retry = next(e for e in recorded if e["decision"] == "Retrying")
    assert retry["payload"]["failed_checks"] == ["no dataset found"]
    with run_store._connect() as conn:
        timer_ids = {r[0] for r in conn.execute("SELECT invocation_id FROM stage_timings WHERE run_id = ?", (run_id,))}
    assert {e["invocation_id"] for e in recorded} == timer_ids

    run_store.finish_run(run_id, "halted", "DATA_DISCOVERY", result={})
    lineage = sorted(_store().rows("invocation_lineage", run_id=run_id), key=lambda r: r["attempt"])
    assert [(r["attempt"], r["lineage"]) for r in lineage] == [(1, "rejected"), (2, "kept")]
    assert [e["decision"] for e in _store().rows("memory_events", run_id=run_id, event_type="run_finished")] == \
        ["halted"]


def test_an_exception_is_recorded_as_the_attempts_error(run_id):
    def boom():
        raise RuntimeError("API quota exhausted")

    master_agent._supervise("Research Agent", boom, lambda out: [], lambda out: "", max_attempts=1)
    event = _store().rows("memory_events", run_id=run_id, event_type="attempt")[0]
    assert event["decision"] == "Rejected" and "RuntimeError: API quota exhausted" in event["payload"]["error"]


def test_a_reverted_rerun_through_the_rerun_loop(run_id, tmp_path):
    ctx = make_ctx(tmp_path)
    with patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=9.0)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent"), stops()]):
        master_agent._rerun_loop(ctx)
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={})

    closed = _store().rows("memory_events", run_id=run_id, event_type="rerun_closed")
    assert [(e["rerun"], e["decision"]) for e in closed] == [(1, "reverted")]
    lineage = {(r["agent"], r["lineage"]) for r in _store().rows("invocation_lineage", run_id=run_id)}
    assert ("Replication Agent", "reverted") in lineage and ("Evaluation Agent", "reverted") in lineage
    assert ("Experiment Agent", "kept") in lineage


def test_a_kept_rerun_and_a_declined_request_through_the_rerun_loop(run_id, tmp_path):
    ctx = make_ctx(tmp_path)
    with patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=1.0)), \
         patch("src.agents.experiment_agent.run",
               side_effect=[asks("Replication Agent"), asks("Intake Agent"), stops()]):
        master_agent._rerun_loop(ctx)
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={})

    assert [e["decision"] for e in _store().rows("memory_events", run_id=run_id, event_type="rerun_closed")] == ["kept"]
    declined = _store().rows("memory_events", run_id=run_id, event_type="request_declined")
    assert len(declined) == 1 and declined[0]["payload"]["requested_by"] == "Experiment Agent"
    assert {r["lineage"] for r in _store().rows("invocation_lineage", run_id=run_id)
            if r["agent"] == "Replication Agent"} == {"kept"}


def test_a_killed_run_is_resolved_when_the_next_run_starts(monkeypatch):
    killed = run_store.start_run("fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", killed)
    master_agent._supervise("Intake Agent", lambda: {"x": 1}, lambda out: [], lambda out: "", max_attempts=1)
    # the process dies here: no finish_run

    run_store.start_run("fraud detection")
    lineage = _store().rows("invocation_lineage", run_id=killed)
    assert [r["lineage"] for r in lineage] == ["unknown"]
    finished = _store().rows("memory_events", run_id=killed, event_type="run_finished")
    assert finished[0]["decision"] == "crashed" and finished[0]["payload"] == {"recovered": True}
    assert events.recover_unfinished() == []  # resolved once


def test_lineage_is_written_once(run_id):
    master_agent._supervise("Intake Agent", lambda: {"x": 1}, lambda out: [], lambda out: "", max_attempts=1)
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={})
    assert events.resolve_lineage(run_id, finished=True) == []
    assert len(_store().rows("invocation_lineage", run_id=run_id)) == 1


def test_events_are_sent_to_the_hosted_store(tmp_path, monkeypatch):
    from tests.test_memory_store import FakeRemote
    remote = FakeRemote(tmp_path / "remote.db")
    memory.reset_client(memory.MemoryClient("postgres", tmp_path / "local.db", remote=remote))
    rid = run_store.start_run("fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", rid)
    master_agent._supervise("Intake Agent", lambda: {"x": 1}, lambda out: [], lambda out: "", max_attempts=1)
    run_store.finish_run(rid, "success", "EXPERIMENTATION", result={})  # syncs at the end
    assert remote.count("memory_events") == 2 and remote.count("invocation_lineage") == 1


def test_recording_fails_soft_and_does_nothing_with_memory_off(run_id):
    memory.reset_client(memory.MemoryClient("off"))
    assert events.attempt(run_id, "inv", "Intake Agent", 1, "Approved") is None
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={})
    assert run_store.get_run(run_id)["status"] == "success"


def test_bad_values_are_refused(run_id, capsys):
    assert events._record("teleported", run_id) is None
    assert "could not record the teleported event" in capsys.readouterr().out


def test_llm_calls_carry_the_run_id(run_id):
    llm._record_call("openrouter", "m", "m", 0.0)
    line = json.loads(llm.CALL_LOG.read_text(encoding="utf-8").splitlines()[-1])
    assert line["run_id"] == run_id
