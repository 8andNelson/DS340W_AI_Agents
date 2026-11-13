"""Milestone M4: memories from runs -- features, base quality, the
record -> memory rules, duplicates, states, and the backfill report."""
import json
from unittest.mock import patch

import pytest

from src import memory
from src.agents import master_agent
from src.memory import features, ranking, retain
from src.orchestration import run_store
from tests.test_master_reruns import asks, evaluation, make_ctx, stops

SCRIPT = ("import pandas as pd\nfrom imblearn.over_sampling import SMOTE\nfrom xgboost import XGBClassifier\n"
          "from sklearn.model_selection import train_test_split\n"
          "X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)\n"
          "model = XGBClassifier(scale_pos_weight=10)\n")
INTAKE = {"domain": "fraud detection", "ml_task": "classification", "keywords": ["credit card fraud"],
          "target_terms": ["fraud"], "search_queries": []}


def _evaluation(gap, leaks=(), findings=(), action="accept"):
    return {"status": "APPROXIMATE", "metrics": [{"metric": "AUC", "published_value": 0.95, "reproduced_value": 0.9,
                                                  "difference_points": gap, "classification": "APPROXIMATE"}],
            "review": {"static_findings": list(leaks), "findings": list(findings)},
            "decision": {"action": action}}


def _supervise(agent, outputs, fail_issue="check failed"):
    outputs = iter(outputs)

    def check(out):
        return [] if out.get("ok", True) else [master_agent._log_entry("Correction", agent, fail_issue, "Retrying", "")]

    return master_agent._supervise(agent, lambda: next(outputs), check, lambda out: f"{agent} report",
                                   max_attempts=3)


def make_run(tmp_path, monkeypatch, gap=2.0, leaks=(), reruns=(), script=SCRIPT, status="success", name="paper"):
    folder = tmp_path / "replication" / name
    folder.mkdir(parents=True)
    (folder / "train.py").write_text(script, encoding="utf-8", newline="\n")
    run_id = run_store.start_run("credit card fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    monkeypatch.setattr(master_agent, "_diagnose", lambda agent, attempt, failure, report, notes: {
        "cause": "xgboost was imported under a wrong module name", "guidance": ["import xgboost"],
        "note": "diagnosis"})
    _supervise("Intake Agent", [INTAKE])
    replication = {"status": "OK", "source": "reconstructed", "entry_point": "train.py",
                   "replication_dir": str(folder), "exit_code": 0, "notes": "Repair attempts: 1."}
    _supervise("Replication Agent", [{"ok": False}, replication], fail_issue="ModuleNotFoundError: No module named 'xgb'")
    final = _evaluation(gap, leaks)
    _supervise("Evaluation Agent", [final])
    run_store.finish_run(run_id, status, "EXPERIMENTATION", parent_paper={"title": "A fraud paper", "doi": "10.1/x"},
                         result={"phase": "EXPERIMENTATION", "structured": INTAKE, "replication_result": replication,
                                 "evaluation_result": final, "reruns": list(reruns)})
    return run_id


def _memories(run_id=None, **filters):
    found = memory.find_memories(states=None, **filters)
    return [m for m in found if run_id is None or m["run_id"] == run_id]


# --- Features and ranking ---

def test_features_from_a_script_and_the_intake_request():
    assert features.techniques(SCRIPT) == ["class_weight", "smote", "stratified_split"]
    assert features.task_type("classification") == "binary"
    assert features.task_type(["classification", "anomaly detection"]) == "anomaly"  # models may give a list
    assert features.task_type("") is None
    assert features.failure_signature("Traceback\n  File \"C:/a/b.py\", line 9\nKeyError: 'Class' in row 12") == \
        "keyerror: <name> in row <n>"
    assert features.failure_signature("status NOT_FOUND: the paper's dataset ('X') could not be verified") == \
        "status not_found: the paper's dataset (<name>) could not be verified"
    feats = features.task_features({"intake": INTAKE, "cleaning": {"rows_total": 50_000, "columns_total": 31}},
                                   "Replication Agent", script=SCRIPT)
    assert feats["model_family"] == ["xgboost"] and feats["dataset_traits"] == {"rows": "10k-100k", "columns": "30-99",
                                                                                "sampled": False}
    assert {"model:xgboost", "tech:smote", "task:binary", "target:fraud"} <= set(features.tags(feats))


def test_base_quality_and_warning_usefulness_explain_every_component():
    quality, reasons = ranking.base_quality((1.0, "o"), (0.6, "a"), (1.0, "v"), (0.5, "d"), "low_trust")
    assert quality == round(0.35 + 0.12 + 0.20 + 0.075 + 0.05, 4)
    assert reasons["approval"] == {"value": 0.6, "reason": "a"} and reasons["trust"]["value"] == 0.5
    assert ranking.gap_outcome(5.0) == 0.75 and ranking.gap_outcome(40) == 0.0 and ranking.gap_outcome(None) is None
    q, r = ranking.warning_usefulness("a cause", True, "keyerror", "normal")
    assert q == 1.0 and r["fixed"]["reason"] == "the next attempt passed"
    assert ranking.warning_usefulness("No cause identified.", None, None, "normal")[0] == 0.1


# --- Rules ---

def test_a_kept_baseline_is_a_positive_solution_with_its_script(tmp_path, monkeypatch):
    run_id = make_run(tmp_path, monkeypatch, gap=2.0)
    script_memory = next(m for m in _memories(run_id, agent="Replication Agent") if m["kind"] == "solution")
    assert script_memory["polarity"] == "positive" and script_memory["lineage"] == "kept"
    assert script_memory["content"] == SCRIPT
    assert script_memory["task"]["model_family"] == ["xgboost"] and "tech:smote" in script_memory["tags"]
    reasons = script_memory["quality_reasons"]
    assert reasons["outcome"] == {"value": 0.9, "reason": "gap to the paper: 2.0 points"}
    assert reasons["approval"]["value"] == 0.6  # approved after one failed attempt
    assert script_memory["invocation_id"]  # the attempt's id from the run recording


def test_a_failed_attempt_becomes_a_warning_with_its_signature(tmp_path, monkeypatch):
    run_id = make_run(tmp_path, monkeypatch)
    failure = next(m for m in _memories(run_id, agent="Replication Agent") if m["kind"] == "failure")
    assert failure["polarity"] == "negative" and failure["lineage"] == "rejected"
    assert failure["task"]["failure_signature"] == "modulenotfounderror: no module named <name>"
    assert failure["quality_reasons"]["fixed"]["value"] == 1.0  # the next attempt passed
    diagnosis = [m for m in _memories(run_id, agent="Master Agent")]
    assert len(diagnosis) == 1 and diagnosis[0]["kind"] == "decision"


def test_a_diagnosis_that_itself_failed_makes_no_memory(monkeypatch):
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    _supervise("Data Agent", [{"ok": False}, {"ok": True}])  # the diagnosis LLM call fails (no network in tests)
    run_store.finish_run(run_id, "halted", "DATA_DISCOVERY", result={"reruns": []})
    assert _memories(run_id, agent="Master Agent") == []
    assert [m["kind"] for m in _memories(run_id, agent="Data Agent")].count("failure") == 1


def test_a_kept_baseline_with_verified_leakage_is_a_warning_not_a_solution(tmp_path, monkeypatch):
    leak = {"check": "smote_before_split", "issue": "SMOTE is fitted before the split", "category": "data_leakage",
            "severity": "critical", "evidence": "SMOTE().fit_resample(X, y)"}
    run_id = make_run(tmp_path, monkeypatch, gap=0.1, leaks=[leak])
    baseline = next(m for m in _memories(run_id, agent="Replication Agent") if m["title"].startswith("Leaky"))
    assert baseline["kind"] == "failure" and baseline["polarity"] == "negative"
    assert not [m for m in _memories(run_id, agent="Replication Agent") if m["polarity"] == "positive"]
    finding = next(m for m in _memories(run_id, agent="Evaluation Agent") if m["title"].startswith("Verified finding"))
    assert finding["polarity"] == "neutral"  # no kept fix yet


def test_validity_is_success_for_the_evaluation_agent_even_when_the_gap_grows(tmp_path, monkeypatch):
    leak = {"check": "smote_before_split", "issue": "SMOTE before split", "category": "data_leakage",
            "severity": "critical"}
    fix = {"rerun": 1, "requested_by": "Evaluation Agent", "target_agent": "Replication Agent", "applied": True,
           "improved": False, "gap_before": 0.1, "gap_after": 6.0, "outcome": "gap 0.1 -> 6.0"}
    run_id = make_run(tmp_path, monkeypatch, gap=6.0, leaks=[leak], reruns=[fix])
    evaluation_memory = next(m for m in _memories(run_id, agent="Evaluation Agent") if m["title"].startswith("Evaluation"))
    assert evaluation_memory["quality_reasons"]["outcome"]["value"] == 1.0
    assert "validity is success" in evaluation_memory["quality_reasons"]["outcome"]["reason"]
    rerun = next(m for m in _memories(run_id, agent="Evaluation Agent") if m["kind"] == "decision")
    assert rerun["polarity"] == "neutral"  # kept, no gain in the gap: a validity fix, not a score change


def test_reverted_and_unknown_attempts_are_never_positive(tmp_path, monkeypatch):
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    ctx = make_ctx(tmp_path)
    with patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=9.0)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent"), stops()]):
        loop = master_agent._rerun_loop(ctx)
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={"evaluation_result": ctx["evaluation_result"],
                                                                       "reruns": loop["reruns"]})
    found = _memories(run_id)
    assert {m["lineage"] for m in found if m["agent"] == "Replication Agent"} == {"reverted"}
    assert all(m["polarity"] != "positive" for m in found if m["lineage"] in ("reverted", "unknown", "superseded"))
    rerun = next(m for m in found if m["kind"] == "decision" and m["agent"] == "Experiment Agent")
    assert rerun["polarity"] == "negative" and "further from the paper" in rerun["summary"]


def test_a_crashed_run_has_no_positive_solutions(tmp_path, monkeypatch):
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    _supervise("Intake Agent", [INTAKE])
    run_store.finish_run(run_id, "crashed")
    intake = _memories(run_id, agent="Intake Agent")[0]
    assert intake["lineage"] == "unknown" and intake["polarity"] == "neutral"


def test_derived_lineage_for_runs_recorded_before_run_recording():
    stage = lambda i, agent, decision="Approved": {"id": i, "agent": agent, "decision": decision}  # noqa: E731
    run = {"status": "success", "reruns": [], "stages": [stage(1, "Data Agent", "Retrying"), stage(2, "Data Agent"),
                                                         stage(3, "Data Agent"), stage(4, "Experiment Agent"),
                                                         stage(5, "Experiment Agent")]}
    assert {k: v[0] for k, v in retain.derived_lineage(run).items()} == {
        1: "rejected", 2: "superseded", 3: "kept", 4: "kept", 5: "kept"}
    run["reruns"] = [{"rerun": 1}]
    assert retain.derived_lineage(run)[3][0] == "unknown"  # a rerun may have replaced it
    run.update(reruns=[], status="crashed")
    assert retain.derived_lineage(run)[3][0] == "unknown"


def test_memories_carry_no_machine_paths(tmp_path, monkeypatch):
    run_id = make_run(tmp_path, monkeypatch, script=SCRIPT + f"DATA = r'{retain.REPO_ROOT}\\data\\train.csv'\n")
    content = next(m for m in _memories(run_id, agent="Replication Agent") if m["kind"] == "solution")["content"]
    assert str(retain.REPO_ROOT) not in content and "<repo>" in content


# --- Duplicates, once per run, states ---

def test_the_same_script_in_another_run_is_counted_not_stored_again(tmp_path, monkeypatch):
    first = make_run(tmp_path, monkeypatch, name="a")
    second = make_run(tmp_path, monkeypatch, name="b")
    scripts = [m for m in _memories(agent="Replication Agent") if m["kind"] == "solution"]
    assert len(scripts) == 1 and scripts[0]["run_id"] == first and scripts[0]["seen_count"] == 2
    assert retain.retain_run(second)["skipped"] == "already retained"


def test_quarantine_and_approve(tmp_path, monkeypatch):
    run_id = make_run(tmp_path, monkeypatch)
    target = next(m for m in _memories(run_id) if m["state"] == "active")
    assert memory.set_state(target["id"], "quarantined", "copied a metric from another paper", actor="tester")
    assert target["id"] not in {m["id"] for m in memory.list_memories(limit=1000)}
    assert memory.get_memory(target["id"])["state"] == "quarantined"  # never deleted
    change = memory.client().local.rows("memory_state_changes", memory_id=target["id"])[0]
    assert change["old_state"] == "active" and change["new_state"] == "quarantined" and change["actor"] == "tester"
    assert memory.set_state(target["id"], "active", "checked again")
    assert memory.set_state(target["id"], "deleted", "x") is False
    assert memory.set_state(target["id"], "retired", "") is False  # a reason is required


def test_state_changes_reach_the_hosted_store(tmp_path, monkeypatch):
    from tests.test_memory_store import FakeRemote
    remote = FakeRemote(tmp_path / "remote.db")
    memory.reset_client(memory.MemoryClient("postgres", tmp_path / "local.db", remote=remote))
    run_id = make_run(tmp_path, monkeypatch)  # finish_run retains and syncs
    assert remote.count("memories") == len(_memories(run_id)) > 0
    target = _memories(run_id)[0]
    memory.set_state(target["id"], "quarantined", "test")
    memory.sync()
    assert remote.fetch_memories(ids=[target["id"]], states=None)[0]["state"] == "quarantined"
    assert remote.count("memory_state_changes") == 1


# --- Backfill and review CLI ---

def test_backfill_dry_run_writes_nothing_and_reports_every_score(tmp_path, monkeypatch, capsys):
    from src.memory.__main__ import _main
    memory.reset_client(memory.MemoryClient("off"))
    run_id = make_run(tmp_path, monkeypatch)  # memory off: nothing retained at finish
    memory.reset_client(memory.MemoryClient("sqlite", tmp_path / "m.db"))
    report = tmp_path / "report.md"
    assert _main(["backfill", "--dry-run", "--report", str(report)]) == 0
    assert _memories() == [] and not memory.client().local.rows("memory_runs")
    text = report.read_text(encoding="utf-8")
    assert f"## Run {run_id}" in text and "dry run: nothing written" in text
    assert "outcome = 0.9: gap to the paper: 2.0 points" in text
    assert _main(["backfill"]) == 0
    assert _memories(run_id) and memory.client().local.rows("memory_runs", run_id=run_id)
    assert _main(["review", "--n", "2"]) == 0
    assert "== Replication Agent" in capsys.readouterr().out


def test_retain_never_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(retain, "load_run", lambda run_id: 1 / 0)
    assert retain.retain_run("x")["skipped"].startswith("error: ZeroDivisionError")
    assert "could not retain run x" in capsys.readouterr().out
