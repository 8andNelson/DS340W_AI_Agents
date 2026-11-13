"""Master Agent's driver: it runs agents only from their handoff()
requests, checks each request first (known agent, no skipped workflow
state, no second run outside an approved rerun), and puts reruns through
the same path."""
import json
import threading
from unittest.mock import patch

from src.agents import master_agent
from src.orchestration.handoff import request

from tests.test_master_reruns import asks, evaluation, make_ctx, script, stops


def log_entries():
    return json.loads(master_agent.LOG_FILE.read_text())


def first_pass_ctx(tmp_path, **missing):
    """A ctx after the Cleaning Agent, before the Code Discovery Agent."""
    ctx = make_ctx(tmp_path)
    ctx.update(structured={"domain": "fraud"}, papers=[{"title": "Paper"}], validation_result={"ok": True},
               code_discovery_result=None, replication_result=None, evaluation_result=None)
    ctx.update(missing)
    return ctx


def test_a_handoff_that_skips_a_workflow_state_halts_the_first_pass(tmp_path):
    ctx = first_pass_ctx(tmp_path)
    drive = master_agent._new_drive()
    with patch("src.agents.replication_agent.run") as replicate:
        master_agent._drive(ctx, [request("cleaning_agent", "Replication Agent")], drive, sender="Cleaning Agent")

    replicate.assert_not_called()  # the Code Discovery Agent has not run yet
    assert drive["blockers"] and drive["phase"] == "DATA_CLEANING"
    assert "cannot run before the Code Discovery Agent" in drive["blockers"][0]["issue"]


def test_an_agent_that_already_ran_is_not_run_again_by_a_plain_handoff(tmp_path):
    ctx = first_pass_ctx(tmp_path)
    drive = master_agent._new_drive()
    with patch("src.agents.data_agent.run") as data:
        master_agent._drive(ctx, [request("cleaning_agent", "Data Agent")], drive, sender="Cleaning Agent")
    data.assert_not_called()
    assert "already ran" in drive["blockers"][0]["issue"]


def test_an_unknown_target_is_declined(tmp_path):
    ctx = first_pass_ctx(tmp_path)
    drive = master_agent._new_drive()
    master_agent._drive(ctx, [request("cleaning_agent", "Modeling Agent")], drive, sender="Cleaning Agent")
    assert "not an agent of this pipeline" in drive["blockers"][0]["issue"]


def test_the_next_agent_is_the_one_the_handoff_requests(tmp_path):
    ctx = first_pass_ctx(tmp_path)
    drive = master_agent._new_drive()
    order = []

    def record(name, value):
        def run(*_args, **_kwargs):
            order.append(name)
            return value
        return run

    found = {"status": "NOT_FOUND", "repository": None, "source_stage": ""}
    replication = {"status": "OK", "source": "reconstructed", "inferred_assumptions": [], "stdout_path": "",
                   "stderr_path": "", "replication_dir": ""}
    with patch("src.agents.code_discovery_agent.run", side_effect=record("code", found)), \
         patch("src.agents.replication_agent.run", side_effect=record("replicate", replication)), \
         patch("src.agents.evaluation_agent.run", side_effect=record("evaluate", evaluation(gap=0.5))), \
         patch("src.agents.experiment_agent.run", side_effect=record("experiment", stops())):
        master_agent._drive(ctx, [request("cleaning_agent", "Code Discovery Agent")], drive,
                            sender="Cleaning Agent")

    assert order == ["code", "replicate", "evaluate", "experiment"]
    assert drive["blockers"] == [] and "requested no further rerun" in drive["end_reason"]
    assert ctx["evaluation_result"]["metrics"][0]["difference_points"] == 0.5


def test_a_rerun_request_before_the_baseline_is_evaluated_is_declined(tmp_path):
    ctx = first_pass_ctx(tmp_path)
    drive = master_agent._new_drive()
    with patch("src.agents.data_agent.run") as data:
        master_agent._drive(ctx, asks("Data Agent")["requests"], drive, sender="Cleaning Agent")
    data.assert_not_called()
    assert "only after the baseline is evaluated" in drive["blockers"][0]["issue"]


def test_a_rerun_whose_chain_stops_before_evaluation_is_undone(tmp_path):
    ctx = make_ctx(tmp_path)

    def changes_script(*_args, **_kwargs):
        script(ctx).write_text("changed()")
        return dict(ctx["replication_result"])

    with patch("src.agents.replication_agent.run", side_effect=changes_script), \
         patch("src.agents.replication_agent.handoff", return_value=[]), \
         patch("src.agents.evaluation_agent.run") as evaluate, \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent"), stops()]):
        loop = master_agent._rerun_loop(ctx)

    evaluate.assert_not_called()
    assert script(ctx).read_text() == "original_script()"
    assert loop["reruns"][0]["applied"] is False
    assert "requested no next agent" in loop["reruns"][0]["outcome"]


def test_a_report_with_characters_outside_cp1252_is_saved(tmp_path):
    # A live run crashed here: the Experiment Agent's report quoted the
    # PDF ligature U+FB01, which the Windows default encoding cannot write.
    path = master_agent._write_report("Experiment Agent", 1, "classi\ufb01cation \u2248 0.99")
    assert path.read_text(encoding="utf-8") == "classi\ufb01cation \u2248 0.99"


def test_parallel_handoffs_clean_while_code_runs_and_join_once(tmp_path):
    from src.agents import validation_agent

    ctx = first_pass_ctx(tmp_path, dataset_result=None, cleaning_result=None)
    drive = master_agent._new_drive()
    code_started, cleaned = threading.Event(), threading.Event()
    calls = []

    def run(target, req, snapshot, state):
        calls.append(target)
        if target == "Data Agent":
            assert code_started.wait(5), "Data and Code Discovery must overlap"
        elif target == "Code Discovery Agent":
            code_started.set()
            assert cleaned.wait(5), "Cleaning must not wait for Code Discovery"
        elif target == "Cleaning Agent":
            assert snapshot["dataset_result"] is not None
            assert snapshot["code_discovery_result"] is None
            cleaned.set()
        elif target == "Replication Agent":
            assert snapshot["cleaning_result"] is not None
            assert snapshot["code_discovery_result"] is not None
        return (stops() if target == "Experiment Agent" else {}), []

    requests = validation_agent.handoff({"recommended_parent_paper": ctx["parent_paper"]}, ctx)
    with patch.object(master_agent, "_run_stage", side_effect=run), \
         patch.object(master_agent, "_after_stage"):
        master_agent._drive(ctx, requests, drive, sender="Validation Agent")
    assert not drive["blockers"]
    assert calls.count("Code Discovery Agent") == 1
    assert calls.count("Replication Agent") == 1
    assert drive["evaluated"]


def test_parallel_failure_does_not_launch_dependents(tmp_path):
    ctx = first_pass_ctx(tmp_path, dataset_result=None, cleaning_result=None)
    drive = master_agent._new_drive()
    started = threading.Barrier(2)
    calls = []

    def run(target, req, snapshot, state):
        calls.append(target)
        started.wait(timeout=5)
        return {}, ([{"issue": "no usable data"}] if target == "Data Agent" else [])

    with patch.object(master_agent, "_run_stage", side_effect=run), \
         patch.object(master_agent, "_after_stage"):
        master_agent._drive(ctx, [request("validation_agent", "Data Agent"),
                                  request("validation_agent", "Code Discovery Agent")], drive)
    assert drive["blockers"]
    assert set(calls) == {"Data Agent", "Code Discovery Agent"}


def test_code_finishes_first_replication_waits_for_data_and_cleaning(tmp_path):
    from src.agents import validation_agent

    ctx = first_pass_ctx(tmp_path, dataset_result=None, cleaning_result=None)
    drive = master_agent._new_drive()
    code_published = threading.Event()
    calls = []

    def run(target, req, snapshot, state):
        if target == "Data Agent":
            assert code_published.wait(5)
            snapshot["parent_paper"]["title"] = "private worker mutation"
        if target == "Replication Agent":
            assert snapshot["cleaning_result"] is not None
            assert snapshot["code_discovery_result"] is not None
        calls.append(target)
        return (stops() if target == "Experiment Agent" else {}), []

    def publish(target, output, context, state):
        if target == "Code Discovery Agent":
            code_published.set()

    requests = validation_agent.handoff({"recommended_parent_paper": ctx["parent_paper"]}, ctx)
    with patch.object(master_agent, "_run_stage", side_effect=run), \
         patch.object(master_agent, "_after_stage", side_effect=publish):
        master_agent._drive(ctx, requests, drive, sender="Validation Agent")
    assert calls.index("Code Discovery Agent") < calls.index("Data Agent")
    assert calls.index("Cleaning Agent") < calls.index("Replication Agent")
    assert calls.count("Replication Agent") == 1
    assert ctx["parent_paper"]["title"] == "Paper"
    assert not drive["blockers"]


def test_parallel_supervisors_preserve_all_log_entries():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as workers:
        list(workers.map(lambda n: master_agent._append_log([{"id": n}]), range(30)))
    assert sorted(e["id"] for e in log_entries()) == list(range(30))


def test_conflicting_parallel_requests_are_rejected(tmp_path):
    ctx = first_pass_ctx(tmp_path, dataset_result=None)
    drive = master_agent._new_drive()
    with patch.object(master_agent, "_run_stage", return_value=({}, [])) as stage:
        master_agent._drive(ctx, [request("validation_agent", "Data Agent", {"query": "a"}),
                                  request("validation_agent", "Data Agent", {"query": "b"})], drive)
    assert stage.call_count == 1
    assert "conflicting handoff inputs" in drive["blockers"][0]["issue"]
