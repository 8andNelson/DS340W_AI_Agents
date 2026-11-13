import json
from datetime import datetime, timedelta, timezone

import pytest

from src import llm
from src.agents import master_agent
from src.orchestration import run_context, run_metrics, run_store
from src.search import page_fetch, web_search


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _metrics(run_id):
    run = next(r for r in run_metrics.load_runs() if r["conditions"]["run_id"] == run_id)
    return run["conditions"], run["metrics"]


def _value(metrics, metric, scope="run"):
    return metrics[(scope, metric)]["value"]


def _assert_no_silent_zero(metrics):
    for key, m in metrics.items():
        if m["value"] is None:
            assert m["missing_reason"], f"{key} is missing without a reason"
        else:
            assert m["missing_reason"] is None


def test_parallel_agent_durations_are_independent_of_completion_order():
    run_id = run_store.start_run("parallel timers")
    data = run_store.start_stage_attempt(run_id, "Data Agent", 1, 5)
    code = run_store.start_stage_attempt(run_id, "Code Discovery Agent", 1, 5)
    run_store.finish_stage_attempt(data, 180)
    run_store.finish_stage_attempt(code, 300)
    # Reverse the persistence order: it must not change measured execution.
    run_store.record_stage(run_id, "Code Discovery Agent", 1, "Approved", {})
    run_store.record_stage(run_id, "Data Agent", 1, "Approved", {})
    with run_store._connect() as conn:
        conn.execute("UPDATE runs SET status='success', started_at='2026-01-01T12:00:00+00:00', "
                     "finished_at='2026-01-01T12:05:00+00:00' WHERE run_id=?", (run_id,))
    run_metrics.compute(run_id)
    conditions, metrics = _metrics(run_id)
    assert conditions["timing_method"] == "invocation_monotonic"
    assert _value(metrics, "run_seconds") == 300
    assert _value(metrics, "agent_seconds", "Data Agent") == 180
    assert _value(metrics, "agent_seconds", "Code Discovery Agent") == 300


def test_interrupted_timer_is_unknown_and_complete_peers_keep_their_duration():
    run_id = run_store.start_run("interrupted parallel timers")
    data = run_store.start_stage_attempt(run_id, "Data Agent", 1, 5)
    run_store.start_stage_attempt(run_id, "Code Discovery Agent", 1, 5)
    run_store.finish_stage_attempt(data, 12)
    run_store.record_stage(run_id, "Data Agent", 1, "Approved", {})
    run_store.finish_run(run_id, "crashed")
    conditions, metrics = _metrics(run_id)
    assert conditions["timing_method"] == "invocation_partial"
    assert _value(metrics, "agent_seconds", "Data Agent") == 12
    assert _value(metrics, "agent_seconds", "Code Discovery Agent") is None
    assert "no complete" in metrics[("Code Discovery Agent", "agent_seconds")]["missing_reason"]


def test_recent_invocation_start_prevents_stale_run_classification():
    run_id = run_store.start_run("recent invocation in an older run")
    run_store.start_stage_attempt(run_id, "Data Agent", 1, 5)
    with run_store._connect() as conn:
        conn.execute("UPDATE runs SET started_at='2020-01-01T00:00:00+00:00' WHERE run_id=?", (run_id,))
    run_context.clear()
    assert run_metrics.compute(run_id) is None


def test_legacy_agent_estimates_are_separate_and_old_metric_versions_are_preserved():
    run_id = run_store.start_run("legacy timers")
    run_store.record_stage(run_id, "Data Agent", 1, "Approved", {})
    with run_store._connect() as conn:
        conn.execute("INSERT INTO run_conditions (run_id, metrics_version, computed_at) VALUES (?, 1, ?)",
                     (run_id, _now()))
        conn.execute("INSERT INTO run_metrics (run_id, metrics_version, scope, metric, value_num, computed_at) "
                     "VALUES (?, 1, 'Data Agent', 'agent_seconds', 99, ?)", (run_id, _now()))
    _, old_display = _metrics(run_id)
    assert _value(old_display, "agent_seconds", "Data Agent") is None
    assert _value(old_display, "agent_seconds_estimate", "Data Agent") == 99
    run_store.finish_run(run_id, "halted")
    conditions, metrics = _metrics(run_id)
    assert conditions["metrics_version"] == 2
    assert conditions["timing_method"] == "legacy_completion_gaps"
    assert _value(metrics, "agent_seconds", "Data Agent") is None
    assert _value(metrics, "agent_seconds_estimate", "Data Agent") is not None
    with run_store._connect() as conn:
        assert conn.execute("SELECT value_num FROM run_metrics WHERE run_id=? AND metrics_version=1",
                            (run_id,)).fetchone()[0] == 99


def _evaluation(gap, static_findings=()):
    return {"status": "APPROXIMATE", "source": "reconstructed",
            "metrics": [{"metric": "Accuracy", "published_value": 0.95, "reproduced_value": round(0.95 - gap / 100, 4),
                         "difference_points": gap, "classification": "APPROXIMATE", "notes": ""},
                        {"metric": "f1", "published_value": None, "reproduced_value": 0.7,
                         "difference_points": None, "classification": "INCONCLUSIVE", "notes": ""}],
            "review": {"static_findings": list(static_findings)}}


@pytest.fixture
def script_dir(tmp_path):
    folder = tmp_path / "replication" / "paper"
    folder.mkdir(parents=True)
    (folder / "train.py").write_text("from xgboost import XGBClassifier\n"
                                     "model = XGBClassifier(random_state=7)\n", encoding="utf-8")
    return folder


def test_finished_run_is_measured_with_all_metric_groups(script_dir, monkeypatch):
    monkeypatch.setattr(run_metrics, "MEMORY_CONTRIBUTOR", "aiden")
    run_id = run_store.start_run("Machine learning for detecting credit card fraud", llm_provider="freellmapi")
    dataset = {"parent_dataset_found": True, "selected_datasets": [{"name": "https://example.org/fraud.csv"}]}
    cleaning = {"rows_total": 1000, "sampled": False, "split_seed": 42}
    replication = {"source": "reconstructed", "entry_point": "train.py", "replication_dir": str(script_dir),
                   "notes": "Repair attempts: 2."}
    run_store.record_stage(run_id, "Intake Agent", 1, "Approved", {"target_terms": ["fraud"]})
    run_store.record_stage(run_id, "Data Agent", 1, "Retrying", None,
                           "Data Agent raised an exception:\n...\n\n--- Master Agent diagnosis ---\nCause: x")
    run_store.record_stage(run_id, "Data Agent", 2, "Approved", dataset)
    run_store.record_stage(run_id, "Cleaning Agent", 1, "Approved", cleaning)
    run_store.record_stage(run_id, "Replication Agent", 1, "Approved", replication)
    run_store.record_stage(run_id, "Evaluation Agent", 1, "Approved",
                           _evaluation(5.0, static_findings=[{"issue": "SMOTE before split"}]))
    final_eval = _evaluation(2.0)
    run_store.record_stage(run_id, "Evaluation Agent", 2, "Approved", final_eval)

    _append_jsonl(llm.CALL_LOG, [
        {"timestamp": _now(), "agent": "data_agent", "served_by": "groq/llama", "duration_seconds": 1.5,
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        {"timestamp": _now(), "agent": "data_agent", "served_by": "", "duration_seconds": 0.5, "usage": {},
         "error": "HTTP 502"},
        {"timestamp": _now(), "agent": None, "served_by": "groq/llama", "duration_seconds": 1.0, "usage": {}},
    ])
    web_search._log("brave", "fraud dataset", 3)
    web_search._log("serpapi", "fraud dataset", 0, error="HTTP 429")
    page_fetch._log("https://example.org", 200, "OK", "", provider="camoufox")
    master_agent.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    master_agent.LOG_FILE.write_text(json.dumps([
        {"decision": "Rejected (redundant)", "issue": "Request to rerun the Data Agent declined: redundant",
         "timestamp": _now()},
        {"decision": "Approved", "issue": "Replication Agent output passed all checks", "timestamp": _now()},
    ]), encoding="utf-8")

    result = {"phase": "EXPERIMENTATION", "dataset_result": dataset, "cleaning_result": cleaning,
              "replication_result": replication, "evaluation_result": final_eval,
              "reruns": [{"applied": True, "improved": True, "outcome": "gap 5.0 -> 2.0"},
                         {"applied": False, "improved": False,
                          "outcome": "the score moved further from the paper (gap 2.0 -> 9.0); the previous "
                                     "baseline was kept."}],
              "rerun_end_reason": "the Experiment Agent requested no further rerun", "blockers": []}
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", parent_paper={"title": "Parent", "doi": "10.1/x"},
                         result=result)

    conditions, metrics = _metrics(run_id)
    _assert_no_silent_zero(metrics)
    # Conditions
    assert conditions["status"] == "success"
    assert conditions["contributor"] == "aiden"
    assert conditions["topic_key"] == "card credit detecting for fraud learning machine"
    assert conditions["parent_paper_doi"] == "10.1/x"
    assert json.loads(conditions["dataset_identities"]) == ["https://example.org/fraud.csv"]
    assert conditions["rows_used"] == 1000 and conditions["sampled"] == 0
    assert conditions["code_source"] == "reconstructed"
    assert json.loads(conditions["model_family"]) == ["xgboost"]
    assert json.loads(conditions["seeds"]) == {"split_seed": 42, "script_seeds": [7]}
    assert conditions["git_commit"] == "test-commit" and conditions["git_dirty"] == 0
    assert json.loads(conditions["served_models"]) == {"groq/llama": 2}
    assert json.loads(conditions["log_matching"]) == {"llm": "time_span", "search": "run_id",
                                                     "page_fetch": "run_id", "master_log": "time_span"}
    assert conditions["memory_mode"] == "pre_memory" and conditions["memory_exposure"] == "none"
    assert conditions["python_version"] and conditions["os"]
    # Result quality
    assert _value(metrics, "run_status") == "success"
    assert _value(metrics, "final_phase_reached") == 9
    assert _value(metrics, "reached_measurable_gap") == 1
    assert _value(metrics, "final_gap_points") == 2.0
    assert _value(metrics, "baseline_gap_points") == 5.0
    assert _value(metrics, "gap_closed_points") == 3.0
    assert _value(metrics, "final_gap.accuracy") == 2.0
    assert _value(metrics, "published.accuracy") == 0.95
    assert _value(metrics, "reproduced.f1") == 0.7
    assert ("run", "final_gap.f1") not in metrics
    assert _value(metrics, "leakage_findings_first") == 1
    assert _value(metrics, "leakage_findings_final") == 0
    assert _value(metrics, "parent_dataset_found") == 1
    assert metrics[("run", "halted_at_agent")]["missing_reason"].startswith("the run did not halt")
    # Reliability
    assert _value(metrics, "attempts") == 7
    assert _value(metrics, "retries", "Data Agent") == 1
    assert _value(metrics, "diagnoses", "Data Agent") == 1
    assert _value(metrics, "exceptions", "Data Agent") == 1
    assert _value(metrics, "first_attempt_approved", "Data Agent") == 0
    assert _value(metrics, "first_attempt_approved", "Intake Agent") == 1
    assert _value(metrics, "first_attempt_approval_rate") == pytest.approx(4 / 5)
    assert _value(metrics, "repair_attempts", "Replication Agent") == 2
    assert _value(metrics, "reruns_kept") == 1
    assert _value(metrics, "reruns_undone") == 1
    assert _value(metrics, "reruns_reverted") == 1
    assert _value(metrics, "reruns_improved") == 1
    assert _value(metrics, "reruns_declined") == 1
    assert _value(metrics, "reruns_requested") == 3
    # Cost and time
    assert _value(metrics, "llm_calls") == 2
    assert _value(metrics, "llm_failed_calls") == 1
    assert _value(metrics, "llm_calls_without_usage") == 1
    assert _value(metrics, "llm_input_tokens") == 10
    assert _value(metrics, "llm_output_tokens") == 5
    assert _value(metrics, "llm_calls", "Data Agent") == 1
    assert _value(metrics, "llm_failed_calls", "Data Agent") == 1
    assert _value(metrics, "search_calls") == 1
    assert _value(metrics, "search_failed_calls") == 1
    assert _value(metrics, "search_calls.brave") == 1
    assert _value(metrics, "page_fetches") == 1 and _value(metrics, "browser_fetches") == 1
    assert _value(metrics, "run_seconds") >= 0
    assert ("Data Agent", "agent_seconds") in metrics
    # Memory use: recorded as absent, never left out
    assert _value(metrics, "memory_exposure") == "none"
    assert _value(metrics, "memories_used") is None
    assert "not implemented" in metrics[("run", "memories_used")]["missing_reason"]


def test_crashed_run_keeps_unknown_values_unknown():
    run_id = run_store.start_run("fraud detection")
    run_store.record_stage(run_id, "Intake Agent", 1, "Approved", {"target_terms": ["fraud"]})
    run_store.record_stage(run_id, "Research Agent", 1, "Retrying", None, "Research Agent raised an exception: x")
    run_store.finish_run(run_id, "crashed")

    conditions, metrics = _metrics(run_id)
    _assert_no_silent_zero(metrics)
    assert conditions["status"] == "crashed"
    assert _value(metrics, "run_status") == "crashed"
    assert _value(metrics, "final_phase_reached") == 2  # Research Agent's LITERATURE_SEARCH
    assert _value(metrics, "final_gap_points") is None
    assert "crashed" in metrics[("run", "final_gap_points")]["missing_reason"]
    assert _value(metrics, "reached_measurable_gap") == 0
    assert _value(metrics, "reruns_kept") is None
    assert _value(metrics, "exceptions") == 1
    # No LLM call log was ever written in this test: unknown, not 0.
    assert _value(metrics, "llm_calls") is None
    assert "log not found" in metrics[("run", "llm_calls")]["missing_reason"]
    assert any("No final result" in n for n in json.loads(conditions["notes"]))


def test_halted_run_names_the_halting_agent():
    run_id = run_store.start_run("fraud detection")
    run_store.record_stage(run_id, "Intake Agent", 1, "Approved", {})
    run_store.record_stage(run_id, "Data Agent", 1, "Rejected", {"status": "NOT_FOUND"}, "no data")
    run_store.finish_run(run_id, "halted", "DATA_DISCOVERY", result={"phase": "DATA_DISCOVERY", "reruns": [],
                                                                     "rerun_end_reason": "", "blockers": []})
    _, metrics = _metrics(run_id)
    _assert_no_silent_zero(metrics)
    assert _value(metrics, "halted_at_agent") == "Data Agent"
    assert _value(metrics, "final_phase_reached") == 4
    assert metrics[("run", "final_gap_points")]["missing_reason"] == "no evaluation was reached"
    assert _value(metrics, "reruns_kept") == 0  # a real count: no rerun ran


def test_rerun_counts_are_unknown_when_the_code_version_did_not_record_them():
    run_id = run_store.start_run("fraud detection")
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={"phase": "EXPERIMENTATION"})
    _, metrics = _metrics(run_id)
    assert _value(metrics, "reruns_kept") is None
    assert metrics[("run", "reruns_kept")]["missing_reason"] == "rerun history is not recorded by this code version"


def test_metrics_are_written_once():
    run_id = run_store.start_run("fraud detection")
    run_store.finish_run(run_id, "halted", "TOPIC_ANALYSIS", result={"phase": "TOPIC_ANALYSIS"})
    assert run_metrics.compute(run_id) is None  # already measured at finish
    measured, reason = run_metrics._compute(run_id)
    assert measured is None and reason == "already measured"
    with run_store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_conditions WHERE run_id = ?", (run_id,)).fetchone()[0] == 1


def test_a_metrics_failure_does_not_stop_finishing_the_run(monkeypatch, capsys):
    def boom(run_id):
        raise RuntimeError("metrics broke")

    monkeypatch.setattr(run_metrics, "compute", boom)
    run_id = run_store.start_run("fraud detection")
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", result={"success": True})
    assert run_store.get_run(run_id)["status"] == "success"
    assert "metrics broke" in capsys.readouterr().out
    assert run_context.current_run_id() is None


def test_log_lines_are_tagged_only_during_a_run():
    web_search._log("brave", "before", 1)
    run_id = run_store.start_run("fraud detection")
    web_search._log("brave", "during", 1)
    run_store.finish_run(run_id, "halted", "TOPIC_ANALYSIS", result={})
    web_search._log("brave", "after", 1)
    lines = [json.loads(line) for line in web_search.SEARCH_LOG.read_text(encoding="utf-8").splitlines()]
    assert [line.get("run_id") for line in lines] == [None, run_id, None]


def _insert_run(run_id, started_at, status="running"):
    with run_store._connect() as conn:
        conn.execute("INSERT INTO runs (run_id, topic, started_at, status) VALUES (?, ?, ?, ?)",
                     (run_id, "old topic", started_at, status))


def test_backfill_measures_old_runs_and_skips_runs_that_may_still_be_running():
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec="seconds")
    _insert_run("stale-run", old)
    _insert_run("recent-run", _now())
    _insert_run("halted-run", old, status="halted")

    outcome = run_metrics.backfill()
    assert outcome == {"stale-run": "measured", "halted-run": "measured", "recent-run": "still running"}
    conditions, metrics = _metrics("stale-run")
    assert conditions["status"] == "crashed"
    assert "measured as crashed" in conditions["notes"]
    assert conditions["git_dirty"] is None  # not recorded for a run measured after its process ended
    assert run_metrics.backfill()["halted-run"] == "already measured"


def _finished_run(provider, gap, doi="10.1/x", topic="fraud detection"):
    run_id = run_store.start_run(topic, llm_provider=provider)
    evaluation = _evaluation(gap)
    run_store.record_stage(run_id, "Evaluation Agent", 1, "Approved", evaluation)
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", parent_paper={"title": "P", "doi": doi},
                         result={"phase": "EXPERIMENTATION", "evaluation_result": evaluation, "reruns": [],
                                 "rerun_end_reason": "stop"})
    return run_id


def test_compare_groups_by_task_and_condition_and_tests_only_with_enough_runs():
    for gap in (1.0, 2.0, 3.0):
        _finished_run("openrouter", gap)
    for gap in (6.0, 7.0, 8.0):
        _finished_run("freellmapi", gap)
    _finished_run("freellmapi", 4.0, doi="10.9/other")

    report = run_metrics.compare_runs(run_metrics.load_runs(), group_by="llm_provider",
                                      metrics=("final_gap_points", "attempts"))
    assert report["kind"] == "descriptive"
    blocks = {b["task"]: b for b in report["blocks"]}
    main = blocks["detection fraud | 10.1/x"]
    other = blocks["detection fraud | 10.9/other"]
    assert main["runs"] == 6 and other["runs"] == 1
    gaps = main["groups"]["openrouter"]["metrics"]["final_gap_points"]
    assert gaps["n"] == 3 and gaps["median"] == 2.0 and gaps["min"] == 1.0 and gaps["max"] == 3.0
    test = main["tests"]["final_gap_points"]
    assert test["n"] == [3, 3]
    assert test["p_value"] == pytest.approx(0.1)  # exact: 2 of the 20 splits are as extreme
    assert abs(test["cliffs_delta"]) == 1.0
    assert other["groups"]["freellmapi"]["too_few_runs"] is True
    assert other["tests"] == {}


def test_compare_lists_conditions_that_differ_between_groups(monkeypatch):
    _finished_run("openrouter", 1.0)
    monkeypatch.setattr(run_store, "_project_git_commit", lambda: "other-commit")
    _finished_run("freellmapi", 2.0)
    report = run_metrics.compare_runs(run_metrics.load_runs(), group_by="llm_provider")
    assert report["blocks"][0]["confounders"]["git_commit"] == {"openrouter": ["test-commit"],
                                                                 "freellmapi": ["other-commit"]}


def test_compare_cli_and_show(capsys, tmp_path):
    run_id = _finished_run("openrouter", 1.0)
    out = tmp_path / "reports" / "compare.csv"
    assert run_metrics._main(["compare", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("run_id,task,memory_mode,status")
    assert run_metrics._main(["show", run_id]) == 0
    printed = capsys.readouterr().out
    assert "describes differences; it cannot show what caused them" in printed
    assert "run / final_gap_points = 1 percentage points" in printed


def test_helpers():
    assert run_metrics.topic_key("Fraud, credit-card fraud!") == "card credit fraud"
    assert run_metrics.model_families("from sklearn.ensemble import RandomForestClassifier\n"
                                      "import torch\nm = RandomForestClassifier()") == ["neural_network",
                                                                                       "random_forest"]
    assert run_metrics.model_families("not python (") == []
    assert run_metrics.script_seeds("np.random.seed(3)\nx = f(random_state=42)\nSEED = 5") == [3, 5, 42]
    assert run_metrics.script_seeds("RANDOM_STATE = 42\nm = M(random_state=RANDOM_STATE)\nif seed == 9: pass") == [42]
    assert run_metrics.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert run_metrics.permutation_test([1, 2, 3], [4, 5, 6]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        run_metrics._Sheet().num("llm_calls", None)  # a missing value needs a reason
