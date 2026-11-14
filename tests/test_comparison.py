"""Milestone M8: the controlled comparison -- a plan written once, a frozen
memory snapshot with held-out topics, runs in alternating order under their
assigned condition, and a report grouped by that condition."""
import json
import subprocess

import pytest

from src import config, memory
from src.memory import recall
from src.orchestration import comparison, run_context, run_metrics, run_store

TOPICS = ["machine learning for detecting credit card fraud", "machine learning for detecting lung cancer"]


def _put(title, topic="machine learning for churn", created_at="2026-01-01T00:00:00+00:00", **m):
    base = {"kind": "solution", "agent": "Replication Agent", "polarity": "positive", "title": title,
            "summary": title, "quality": 0.8, "lineage": "kept", "created_at": created_at,
            "task": {"model_family": ["xgboost"], "topic": topic}}
    base.update(m)
    return memory.put_memory(base)


def _recalled():
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    return {i["memory"]["id"]: i["group"] for i in ctx.items}


# --- The plan ---

def test_the_run_order_covers_every_topic_and_condition_and_alternates():
    order = comparison.run_order(["a", "b"], ["off", "read"], 3)
    assert len(order) == 12
    assert sorted(order) == sorted([(t, c) for t in ("a", "b") for c in ("off", "read")] * 3)
    assert order[:4] == [("a", "off"), ("a", "read"), ("b", "read"), ("b", "off")]
    assert order[4:6] == [("a", "read"), ("a", "off")]  # the next repeat starts the other way round


def test_a_plan_is_written_once_with_everything_fixed_before_the_first_run():
    plan = comparison.make_plan("t1", TOPICS, ["off", "read"], 3)
    assert plan["comparison_group"] == "test:t1" and len(plan["run_order"]) == 12
    assert plan["held_out_topics"] == sorted(run_metrics.topic_key(t) for t in TOPICS)
    assert plan["code_commit"] == "test-commit" and plan["llm_provider"] == config.LLM_PROVIDER
    assert plan["primary_metrics"] and plan["exclusion_rules"] and plan["memory_snapshot"]
    with pytest.raises(FileExistsError):
        comparison.make_plan("t1", TOPICS)
    with pytest.raises(ValueError):
        comparison.make_plan("t2", TOPICS, ["off"])


# --- What a test run may see ---

def test_a_test_run_sees_only_the_frozen_bank_without_its_own_topics(monkeypatch):
    other = _put("churn xgboost script")
    same_topic = _put("fraud xgboost script", topic=TOPICS[0])
    plan = comparison.make_plan("t1", TOPICS)
    later = _put("made after the plan", created_at="2999-01-01T00:00:00+00:00")
    monkeypatch.setattr(config, "MEMORY_COMPARISON_GROUP", "t1")
    run_id = run_store.start_run(TOPICS[0])
    info = run_context.info(run_id)
    assert info["memory_snapshot"] == plan["memory_snapshot"] and info["comparison_group"] == "test:t1"
    found = _recalled()
    assert other in found and same_topic not in found and later not in found
    assert info["memory_bank_size"] == {"Replication Agent": 1}  # only what the run can see
    run_store.finish_run(run_id, "halted", "TOPIC_ANALYSIS", result={})
    cond = next(r["conditions"] for r in run_metrics.load_runs() if r["conditions"]["run_id"] == run_id)
    assert cond["comparison_group"] == "test:t1" and cond["memory_bank_snapshot"] == plan["memory_snapshot"]


def test_the_no_warnings_condition_shows_good_cases_only(monkeypatch):
    good = _put("churn xgboost script")
    warning = _put("leaky churn script", kind="failure", polarity="negative")
    assert set(_recalled()) == {good, warning}
    monkeypatch.setattr(config, "MEMORY_WARNINGS", False)
    run_store.start_run("some topic")
    assert set(_recalled()) == {good}


def test_runs_on_a_supplied_paper_are_not_called_a_controlled_test():
    runs = [{"conditions": {"comparison_group": "parent_pdf:abc", "topic_key": "t", "parent_paper_doi": None,
                            "parent_paper_title": None, "memory_mode": "off", "status": "success"}, "metrics": {}}]
    assert run_metrics.compare_runs(runs)["kind"] == "descriptive"


# --- Running the plan ---

@pytest.fixture
def fake_pipeline(monkeypatch):
    """`python run.py <topic>` stand-in: runs the real run store and metrics
    under the condition the runner put in the child's environment."""
    gaps = {"off": iter([6.0, 7.0, 8.0, 6.5, 7.5, 8.5]), "read": iter([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])}

    def fake_run(cmd, cwd=None, env=None, timeout=None):
        assert cmd[1:2] == ["run.py"]
        monkeypatch.setattr(config, "MEMORY_MODE", env["MEMORY_MODE"])
        monkeypatch.setattr(config, "MEMORY_WARNINGS", env["MEMORY_WARNINGS"] == "true")
        monkeypatch.setattr(config, "MEMORY_COMPARISON_GROUP", env["MEMORY_COMPARISON_GROUP"])
        run_id = run_store.start_run(cmd[2])
        condition = "off" if env["MEMORY_MODE"] == "off" else "read"
        evaluation = {"metrics": [{"metric": "accuracy", "published_value": 0.95, "reproduced_value": 0.9,
                                   "difference_points": next(gaps[condition]), "classification": "APPROXIMATE"}]}
        run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={"evaluation_result": evaluation,
                                                                          "reruns": []})
        monkeypatch.setattr(config, "MEMORY_COMPARISON_GROUP", "")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(comparison.subprocess, "run", fake_run)
    monkeypatch.setattr(comparison, "preflight", lambda plan, allow_dirty=False: [])


def test_the_runner_runs_the_plan_in_order_and_the_report_compares_by_assigned_condition(fake_pipeline):
    comparison.make_plan("t1", TOPICS[:1], ["off", "read"], 3)
    records = []
    while (record := comparison.run_one("t1")) and not record.get("done"):
        records.append(record)
        assert record["valid"], record.get("invalid_reason")
    assert [r["condition"] for r in records] == ["off", "read", "read", "off", "off", "read"]
    assert all(r["valid"] for r in records) and comparison.next_run("t1") is None

    result, runs = comparison.report("t1")
    assert result["kind"] == "controlled" and result["valid_runs"] == 6 and result["planned_runs"] == 6
    block = result["blocks"][0]
    assert set(block["groups"]) == {"off", "read"}
    assert block["groups"]["read"]["metrics"]["final_gap_points"]["median"] == 2.0
    test = block["tests"]["final_gap_points"]
    assert test["n"] == [3, 3] and test["p_value"] == pytest.approx(0.1)
    assert set(result["exposure"].values()) == {"none"}  # an empty bank: assigned "read", but no memory shown
    assert (comparison.COMPARISONS_DIR / "t1.report.json").exists()


def test_a_run_whose_recorded_condition_differs_from_its_assignment_is_invalid(fake_pipeline, monkeypatch):
    comparison.make_plan("t1", TOPICS[:1], ["off", "read"], 1)
    original = comparison.subprocess.run

    def wrong_mode(cmd, cwd=None, env=None, timeout=None):
        return original(cmd, cwd=cwd, env={**env, "MEMORY_MODE": "read_write"}, timeout=timeout)

    monkeypatch.setattr(comparison.subprocess, "run", wrong_mode)
    record = comparison.run_one("t1")
    assert not record["valid"] and "recorded memory mode read_write" in record["invalid_reason"]
    assert comparison.next_run("t1")["n"] == 1  # an invalid run is run again
    assert comparison.report("t1")[0]["valid_runs"] == 0


def test_preflight_refuses_another_commit_provider_or_an_active_run(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")  # no router to ping
    plan = comparison.make_plan("t1", TOPICS[:1])
    assert comparison.preflight(plan) == []
    monkeypatch.setattr(run_store, "_project_git_commit", lambda: "other-commit")
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    run_store.start_run("an active run")
    problems = " | ".join(comparison.preflight(plan))
    assert "commit" in problems and "LLM_PROVIDER" in problems and "another pipeline run is active" in problems


def test_the_cli(fake_pipeline, capsys):
    assert comparison._main(["plan", "t1", "--topic", TOPICS[0], "--repeats", "1", "--no-warnings-condition"]) == 0
    plan = json.loads(comparison.plan_path("t1").read_text(encoding="utf-8"))
    assert set(plan["conditions"]) == {"off", "read", "read_no_warnings"} and len(plan["run_order"]) == 3
    assert comparison._main(["run", "t1"]) == 0
    assert comparison._main(["status", "t1"]) == 0
    assert comparison._main(["report", "t1"]) == 0
    out = capsys.readouterr().out
    assert "pending" in out and "1 valid run(s) of 3 planned" in out


def test_preflight_refuses_when_the_llm_router_is_down(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "freellmapi")  # tests block the network: the ping fails
    plan = comparison.make_plan("t1", TOPICS[:1])
    assert any("FreeLLMAPI router is not reachable" in p for p in comparison.preflight(plan))
