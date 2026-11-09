"""Master Agent acting on the Evaluation Agent's requests: it checks each
request, runs the requested agent and every stage downstream of it, has the
new baseline reviewed again, and undoes a re-run that fails."""
import json
from pathlib import Path
from unittest.mock import patch

from src.agents import master_agent


def request(action, goal=None, instructions=None, evidence=True):
    target = {"rerun_data_agent": "Data Agent", "revise_replication": "Replication Agent",
              "run_experiment": "Experiment Agent"}[action]
    return {"sender": "evaluation_agent", "recipient": "master_agent", "action": action, "target_agent": target,
            "goal": goal or {}, "instructions": instructions or [], "summary": "analyst summary",
            "evidence": [{"issue": "SMOTE before split", "evidence": "fit_resample(X, y)"}] if evidence else []}


def evaluation(status="EXACT", accuracy=0.999, requests=None):
    return {"status": status, "metrics": [{"metric": "accuracy", "published_value": 99.9,
                                           "reproduced_value": accuracy, "classification": status, "notes": ""}],
            "likely_causes": [], "requests": requests or [], "decision": {}, "review": {}}


def make_ctx(tmp_path: Path, first_request, source="reconstructed"):
    processed = tmp_path / "processed"
    processed.mkdir()
    cleaning = {"rows_total": 1000, "datasets_merged": 1, "conflicts": [], "split_rows": {}, "split_seed": 42,
                "target_met": None}
    for key in ("master_csv_path", "train_csv_path", "test_csv_path", "validation_csv_path"):
        path = processed / f"{key}.csv"
        path.write_text("original")
        cleaning[key] = str(path)
    slug = tmp_path / "replication" / "paper"
    (slug / "reconstructed").mkdir(parents=True)
    (slug / "reconstructed" / "train.py").write_text("original_script()")
    (slug / "run_stdout.txt").write_text("METRIC accuracy=0.999")
    replication = {"status": "OK", "source": source, "replication_dir": str(slug),
                   "stdout_path": str(slug / "run_stdout.txt"), "stderr_path": "", "inferred_assumptions": ["a"],
                   "generated_by": ["model-x"]}
    dataset = {"selected_datasets": [{"name": "https://kaggle.com/own"}], "total_entries": 1000,
               "target_entries": None, "data_plan": {}, "target_met": None, "is_combined": False}
    return {"parent_paper": {"title": "Paper"}, "ranked_pool": [{"title": "Paper"}], "topic": "fraud",
            "target_terms": ["fraud"], "dataset_result": dataset, "cleaning_result": cleaning,
            "code_discovery_result": {"status": "NOT_FOUND"}, "replication_result": replication,
            "evaluation_result": evaluation(requests=[first_request])}


def ok_replication(ctx, **overrides):
    return dict(ctx["replication_result"], **overrides)


def log_entries():
    return json.loads(master_agent.LOG_FILE.read_text())


def test_leakage_fix_is_run_and_the_new_baseline_is_reviewed(tmp_path):
    ctx = make_ctx(tmp_path, request("revise_replication", instructions=["Split before SMOTE."]))
    fixed = ok_replication(ctx, revision_round=1)
    with patch("src.agents.replication_agent.run", return_value=fixed) as replicate, \
         patch("src.agents.evaluation_agent.run", return_value=evaluation("APPROXIMATE", 0.95)) as evaluate, \
         patch("src.agents.data_agent.run") as data:
        history = master_agent._improvement_rounds(ctx)

    data.assert_not_called()
    revision = replicate.call_args.kwargs["revision"]
    assert revision == {"round": 1, "instructions": ["Split before SMOTE."], "previous_inferred_assumptions": ["a"]}
    assert evaluate.call_args.args[1] is fixed
    assert history[0]["applied"] is True
    assert "0.999 -> accuracy=0.95" in history[0]["outcome"]
    assert ctx["replication_result"] is fixed
    assert ctx["evaluation_result"]["status"] == "APPROXIMATE"
    entries = log_entries()
    assert any(e["target"] == "Replication Agent" and e["decision"] == "Approved request" for e in entries)


def test_data_request_runs_data_cleaning_replication_and_evaluation_in_order(tmp_path):
    ctx = make_ctx(tmp_path, request("rerun_data_agent", goal={"target_entries": 50000}))
    more_data = dict(ctx["dataset_result"], total_entries=48000, target_entries=50000)
    new_cleaning = dict(ctx["cleaning_result"], rows_total=48000)
    with patch("src.agents.data_agent.run", return_value=more_data) as data, \
         patch("src.agents.cleaning_agent.run", return_value=new_cleaning) as clean, \
         patch("src.agents.replication_agent.run", return_value=ok_replication(ctx)) as replicate, \
         patch("src.agents.evaluation_agent.run", return_value=evaluation()):
        history = master_agent._improvement_rounds(ctx)

    assert data.call_args.kwargs["mode"] == "expand"
    assert data.call_args.kwargs["target_entries"] == 50000
    assert data.call_args.kwargs["existing_datasets"] == [{"name": "https://kaggle.com/own"}]
    assert clean.call_args.kwargs["target_rows"] == 50000
    assert replicate.call_args.args[2] is new_cleaning
    assert replicate.call_args.kwargs["revision"]["reuse_script"] is True  # same code, more data
    assert history[0]["applied"] and ctx["cleaning_result"] is new_cleaning


def test_a_failed_rerun_restores_the_previous_baseline(tmp_path):
    ctx = make_ctx(tmp_path, request("revise_replication", instructions=["fix"]))
    previous = ctx["replication_result"]
    script = Path(previous["replication_dir"]) / "reconstructed" / "train.py"

    def broken_rerun(*_args, **_kwargs):
        script.write_text("broken()")
        return dict(previous, status="BLOCKED")

    with patch("src.agents.replication_agent.run", side_effect=broken_rerun), \
         patch("src.agents.evaluation_agent.run") as evaluate:
        history = master_agent._improvement_rounds(ctx)

    evaluate.assert_not_called()
    assert script.read_text() == "original_script()"
    assert ctx["replication_result"] is previous
    assert history[0]["applied"] is False
    assert "previous baseline was kept" in history[0]["outcome"]


def test_no_new_data_ends_the_rounds_without_touching_the_baseline(tmp_path):
    ctx = make_ctx(tmp_path, request("rerun_data_agent", goal={"target_entries": 50000}))
    with patch("src.agents.data_agent.run", return_value=dict(ctx["dataset_result"])), \
         patch("src.agents.cleaning_agent.run") as clean:
        history = master_agent._improvement_rounds(ctx)
    clean.assert_not_called()
    assert history[0]["applied"] is False


def test_rounds_stop_at_the_limit(tmp_path):
    ctx = make_ctx(tmp_path, request("revise_replication", instructions=["fix 1"]))
    again = [evaluation(requests=[request("revise_replication", instructions=[f"fix {i}"])]) for i in range(2, 9)]
    with patch("src.agents.replication_agent.run", return_value=ok_replication(ctx)) as replicate, \
         patch("src.agents.evaluation_agent.run", side_effect=again):
        history = master_agent._improvement_rounds(ctx)
    assert len(history) == master_agent.MAX_IMPROVEMENT_ROUNDS
    assert replicate.call_count == master_agent.MAX_IMPROVEMENT_ROUNDS


def test_reruns_are_numbered_after_the_first_attempts(tmp_path):
    ctx = make_ctx(tmp_path, request("revise_replication", instructions=["fix"]))
    with patch("src.agents.replication_agent.run", return_value=ok_replication(ctx)), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation()):
        master_agent._improvement_rounds(ctx)
    assert (master_agent.REPORT_DIR / f"replication_agent_attempt{master_agent.MAX_ATTEMPTS + 1}.txt").is_file()


def test_experiment_or_accept_requests_end_the_rounds(tmp_path):
    ctx = make_ctx(tmp_path, request("run_experiment", goal={"experiment_category": "alternative_model"}))
    with patch("src.agents.replication_agent.run") as replicate:
        assert master_agent._improvement_rounds(ctx) == []
    replicate.assert_not_called()
    ctx["evaluation_result"] = evaluation(requests=[])
    assert master_agent._improvement_rounds(ctx) == []


# --- Master Agent's own check of a request ---

def check(req, history=(), source="reconstructed", rows=1000):
    return master_agent._review_evaluation_request(req, list(history), {"source": source}, {"rows_total": rows})


def test_valid_requests_are_approved():
    assert check(request("revise_replication", instructions=["fix"])) == ""
    assert check(request("rerun_data_agent", goal={"target_entries": 5000})) == ""


def test_cloned_code_is_not_revised():
    assert "cloned" in check(request("revise_replication", instructions=["fix"]), source="cloned_repository")


def test_revision_needs_instructions_and_evidence():
    assert "instructions" in check(request("revise_replication"))
    assert "evidence" in check(request("revise_replication", instructions=["fix"], evidence=False))


def test_data_goal_must_exceed_the_rows_used():
    assert "not above" in check(request("rerun_data_agent", goal={"target_entries": 900}))


def test_repeats_are_declined():
    req = request("revise_replication", instructions=["fix"])
    done = [{"round": 1, "action": req["action"], "goal": req["goal"], "instructions": req["instructions"],
             "applied": True}]
    assert "already acted on" in check(req, done)
    failed_data = [{"round": 1, "action": "rerun_data_agent", "goal": {"target_entries": 3000}, "instructions": [],
                    "applied": False}]
    assert "no usable new data" in check(request("rerun_data_agent", goal={"target_entries": 9000}), failed_data)


def test_a_declined_request_is_logged_as_a_correction_to_the_evaluation_agent(tmp_path):
    ctx = make_ctx(tmp_path, request("revise_replication", instructions=["fix"]), source="cloned_repository")
    with patch("src.agents.replication_agent.run") as replicate:
        assert master_agent._improvement_rounds(ctx) == []
    replicate.assert_not_called()
    entry = log_entries()[-1]
    assert entry["target"] == "Evaluation Agent" and entry["decision"] == "Rejected"
