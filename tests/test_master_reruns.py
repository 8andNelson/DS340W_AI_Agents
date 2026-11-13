"""Master Agent's rerun loop: the Evaluation Agent (validity) and the
Experiment Agent (closing the gap to the paper) request reruns of earlier
agents; Master Agent checks each request, declines invalid or redundant
ones and tells the sender why, runs the requested agent and every stage
after it, and undoes a rerun that fails or moves the score away."""
import json
from pathlib import Path
from unittest.mock import patch

from src import llm
from src.agents import master_agent


def evaluation(gap=4.0, requests=None, status="APPROXIMATE"):
    return {"status": status, "metrics": [{"metric": "accuracy", "published_value": 99.0,
                                           "reproduced_value": round(99.0 - gap, 4), "difference_points": gap,
                                           "classification": status, "notes": ""}],
            "likely_causes": [], "requests": requests or [], "decision": {}, "review": {}}


def eval_request(action, goal=None, instructions=("Split before SMOTE.",)):
    target = {"rerun_data_agent": "Data Agent", "revise_replication": "Replication Agent"}[action]
    return {"sender": "evaluation_agent", "recipient": "master_agent", "action": action, "target_agent": target,
            "goal": goal or {}, "instructions": list(instructions), "summary": "analyst summary",
            "evidence": [{"issue": "SMOTE before split", "evidence": "fit_resample(X, y)"}]}


def asks(target, instructions=("change it",), goal=None, evidence=True):
    """An Experiment Agent analysis that requests one rerun."""
    request = {"sender": "experiment_agent", "recipient": "master_agent", "action": "rerun_agent",
               "target_agent": target, "goal": goal or {}, "instructions": list(instructions),
               "summary": "close the gap",
               "evidence": [{"issue": "quoted", "evidence": "n_estimators=50"}] if evidence else []}
    return {"status": "REQUEST", "requests": [request], "decision": {"action": "rerun_agent"}, "decision_notes": []}


def stops():
    return {"status": "STOP", "requests": [], "decision": {"action": "stop", "reason": "gap closed"},
            "decision_notes": []}


def make_ctx(tmp_path: Path, first_evaluation=None, source="reconstructed"):
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
    (slug / "run_stdout.txt").write_text("METRIC accuracy=0.95")
    replication = {"status": "OK", "source": source, "replication_dir": str(slug),
                   "stdout_path": str(slug / "run_stdout.txt"), "stderr_path": "", "inferred_assumptions": ["a"],
                   "generated_by": ["model-x"]}
    dataset = {"selected_datasets": [{"name": "https://kaggle.com/own"}], "total_entries": 1000,
               "target_entries": None, "data_plan": {}, "target_met": None, "is_combined": False}
    return {"parent_paper": {"title": "Paper"}, "ranked_pool": [{"title": "Paper"}], "topic": "fraud",
            "target_terms": ["fraud"], "dataset_result": dataset, "cleaning_result": cleaning,
            "code_discovery_result": {"status": "NOT_FOUND"}, "replication_result": replication,
            "evaluation_result": first_evaluation or evaluation()}


def log_entries():
    return json.loads(master_agent.LOG_FILE.read_text())


def script(ctx):
    return Path(ctx["replication_result"]["replication_dir"]) / "reconstructed" / "train.py"


# --- Requests from the Evaluation Agent (validity first) ---

def test_a_leakage_fix_runs_before_the_experiment_agent_and_is_kept_even_if_the_gap_widens(tmp_path):
    ctx = make_ctx(tmp_path, evaluation(gap=0.5, requests=[eval_request("revise_replication")]))
    fixed = dict(ctx["replication_result"], revision_round=1)
    with patch("src.agents.replication_agent.run", return_value=fixed) as replicate, \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=3.0)) as evaluate, \
         patch("src.agents.experiment_agent.run", return_value=stops()) as experiment, \
         patch("src.agents.data_agent.run") as data:
        loop = master_agent._rerun_loop(ctx)

    data.assert_not_called()
    assert replicate.call_args.kwargs["revision"] == {
        "round": 1, "instructions": ["Split before SMOTE."], "previous_inferred_assumptions": ["a"]}
    assert evaluate.call_args.args[1] is fixed
    assert loop["reruns"][0]["applied"] is True  # a leaky score is not a real score
    assert loop["reruns"][0]["requested_by"] == "Evaluation Agent"
    assert ctx["replication_result"] is fixed
    experiment.assert_called_once()
    assert "requested no further rerun" in loop["end_reason"]


def test_a_data_request_runs_data_cleaning_replication_and_evaluation_in_order(tmp_path):
    ctx = make_ctx(tmp_path, evaluation(requests=[eval_request("rerun_data_agent", goal={"target_entries": 50000})]))
    more_data = dict(ctx["dataset_result"], total_entries=48000, target_entries=50000)
    new_cleaning = dict(ctx["cleaning_result"], rows_total=48000)
    with patch("src.agents.data_agent.run", return_value=more_data) as data, \
         patch("src.agents.cleaning_agent.run", return_value=new_cleaning) as clean, \
         patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])) as replicate, \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=2.0)), \
         patch("src.agents.experiment_agent.run", return_value=stops()):
        loop = master_agent._rerun_loop(ctx)

    assert data.call_args.kwargs["mode"] == "expand"
    assert data.call_args.kwargs["target_entries"] == 50000
    assert data.call_args.kwargs["existing_datasets"] == [{"name": "https://kaggle.com/own"}]
    assert clean.call_args.kwargs["target_rows"] == 50000
    assert replicate.call_args.args[2] is new_cleaning
    assert replicate.call_args.kwargs["revision"]["reuse_script"] is True  # same code, more data
    assert loop["reruns"][0]["improved"] is True and ctx["cleaning_result"] is new_cleaning


# --- Requests from the Experiment Agent ---

def test_the_relayed_instructions_reach_the_requested_agents_prompts(tmp_path):
    ctx = make_ctx(tmp_path)
    seen = {}

    def replicate(*_args, **kwargs):
        seen["notes"] = list(llm._supervisor_notes.get())
        seen["revision"] = kwargs["revision"]
        return dict(ctx["replication_result"])

    with patch("src.agents.replication_agent.run", side_effect=replicate), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=1.5)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent", ["Use 500 trees."]),
                                                              stops()]):
        loop = master_agent._rerun_loop(ctx)

    assert "Use 500 trees." in seen["notes"][0] and "Experiment Agent" in seen["notes"][0]
    assert seen["revision"]["instructions"] == ["Use 500 trees."]
    assert loop["reruns"][0]["gap_before"] == 4.0 and loop["reruns"][0]["gap_after"] == 1.5
    assert llm._supervisor_notes.get() == ()


def test_a_code_discovery_rerun_skips_the_cache_and_replicates_afresh(tmp_path):
    ctx = make_ctx(tmp_path)
    found = {"status": "FOUND", "repository": {"repo_url": "https://github.com/a/b"}, "source_stage": "search"}
    with patch("src.agents.code_discovery_agent.run", return_value=found) as discover, \
         patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])) as replicate, \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=1.0)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks("code_discovery_agent"), stops()]):
        master_agent._rerun_loop(ctx)

    assert discover.call_args.kwargs["use_cache"] is False
    assert replicate.call_args.args[1] is found
    assert replicate.call_args.kwargs["revision"] is None


def test_a_rerun_that_moves_the_score_away_is_reverted_and_the_experiment_agent_is_told(tmp_path):
    ctx = make_ctx(tmp_path)
    previous = ctx["replication_result"]

    def worse(*_args, **_kwargs):
        script(ctx).write_text("worse_script()")
        return dict(previous)

    with patch("src.agents.replication_agent.run", side_effect=worse), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=9.0)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent"), stops()]) as experiment:
        loop = master_agent._rerun_loop(ctx)

    assert script(ctx).read_text() == "original_script()"
    assert ctx["replication_result"] is previous and ctx["evaluation_result"]["metrics"][0]["difference_points"] == 4.0
    assert loop["reruns"][0]["applied"] is False and "further from the paper" in loop["reruns"][0]["outcome"]
    assert any("was not kept" in f for f in experiment.call_args_list[1].args[7])
    assert any(e["decision"] == "Reverted" for e in log_entries())


def test_a_failed_rerun_restores_the_previous_baseline(tmp_path):
    ctx = make_ctx(tmp_path)

    def broken(*_args, **_kwargs):
        script(ctx).write_text("broken()")
        return dict(ctx["replication_result"], status="BLOCKED")

    with patch.object(master_agent, "MAX_ATTEMPTS", 1), \
         patch("src.agents.replication_agent.run", side_effect=broken), \
         patch("src.agents.evaluation_agent.run") as evaluate, \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent"), stops()]):
        loop = master_agent._rerun_loop(ctx)

    evaluate.assert_not_called()
    assert script(ctx).read_text() == "original_script()"
    assert "previous baseline was kept" in loop["reruns"][0]["outcome"]


# --- Master Agent decides when a rerun is redundant ---

def test_a_fourth_data_rerun_without_progress_is_declined_as_redundant(tmp_path):
    ctx = make_ctx(tmp_path)
    requests = [asks("Data Agent", [f"try source {i}"]) for i in range(1, 5)] + [asks("Replication Agent"), stops()]
    with patch("src.agents.data_agent.run", return_value=dict(ctx["dataset_result"])), \
         patch("src.agents.cleaning_agent.run", return_value=dict(ctx["cleaning_result"])), \
         patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=4.0)), \
         patch("src.agents.experiment_agent.run", side_effect=requests) as experiment:
        loop = master_agent._rerun_loop(ctx)

    assert [r["target_agent"] for r in loop["reruns"]] == ["Data Agent"] * 3 + ["Replication Agent"]
    declined = [e for e in log_entries() if e["decision"] == "Rejected (redundant)"]
    assert declined and declined[0]["target"] == "Experiment Agent"
    assert "look for a different solution" in declined[0]["next"].lower()
    feedback = experiment.call_args_list[4].args[7]
    assert any("redundant" in f and "Data Agent" in f for f in feedback)


def test_the_loop_ends_after_several_reruns_without_progress(tmp_path):
    ctx = make_ctx(tmp_path)
    targets = ["Replication Agent", "Cleaning Agent", "Evaluation Agent", "Replication Agent", "Cleaning Agent"]
    with patch("src.agents.cleaning_agent.run", return_value=dict(ctx["cleaning_result"])), \
         patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=4.0)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks(t, [f"idea {i}"]) for i, t in
                                                              enumerate(targets)]):
        loop = master_agent._rerun_loop(ctx)
    assert len(loop["reruns"]) == master_agent.NO_PROGRESS_LIMIT
    assert "did not bring the score closer" in loop["end_reason"]


def test_the_loop_ends_after_three_declined_requests_in_a_row(tmp_path):
    ctx = make_ctx(tmp_path)
    with patch("src.agents.experiment_agent.run", side_effect=[asks("Research Agent")] * 3) as experiment:
        loop = master_agent._rerun_loop(ctx)
    assert loop["reruns"] == [] and experiment.call_count == master_agent.MAX_DECLINED_IN_A_ROW
    assert "declined" in loop["end_reason"]


def test_the_loop_ends_at_the_rerun_limit(tmp_path):
    ctx = make_ctx(tmp_path)
    gaps = [4.0 - 0.3 * i for i in range(1, 20)]  # every rerun improves, so only the limit stops it
    with patch.object(master_agent, "MAX_PIPELINE_RERUNS", 3), \
         patch("src.agents.evaluation_agent.run", side_effect=[evaluation(gap=g) for g in gaps]), \
         patch("src.agents.experiment_agent.run",
               side_effect=[asks("Evaluation Agent", [f"re-read metric {i}"]) for i in range(10)]):
        loop = master_agent._rerun_loop(ctx)
    assert len(loop["reruns"]) == 3 and "limit" in loop["end_reason"]


def test_reruns_are_numbered_after_the_first_attempts(tmp_path):
    ctx = make_ctx(tmp_path)
    with patch("src.agents.replication_agent.run", return_value=dict(ctx["replication_result"])), \
         patch("src.agents.evaluation_agent.run", return_value=evaluation(gap=1.0)), \
         patch("src.agents.experiment_agent.run", side_effect=[asks("Replication Agent"), stops()]):
        master_agent._rerun_loop(ctx)
    assert (master_agent.REPORT_DIR / f"replication_agent_attempt{master_agent.MAX_ATTEMPTS + 1}.txt").is_file()
    assert (master_agent.REPORT_DIR / f"experiment_agent_attempt{master_agent.MAX_ATTEMPTS + 1}.txt").is_file()


# --- Master Agent's own check of a request ---

def check(req, history=(), source="reconstructed", rows=1000):
    ctx = {"replication_result": {"source": source}, "cleaning_result": {"rows_total": rows}}
    return master_agent._review_rerun_request(req, list(history), ctx)


def rerun(n, target, improved=False, instructions=("x",), goal=None, **extra):
    return {"rerun": n, "target_agent": target, "instructions": list(instructions), "goal": goal or {},
            "improved": improved, "gap_after": 4.0, "outcome": "no change", **extra}


def test_valid_requests_are_approved():
    assert check(asks("Replication Agent")["requests"][0]) == ""
    assert check(eval_request("rerun_data_agent", goal={"target_entries": 5000})) == ""


def test_agents_that_would_replace_the_parent_paper_are_not_rerun():
    for target in ("Research Agent", "Validation Agent", "Intake Agent"):
        assert "replace the Parent Paper" in check(asks(target)["requests"][0])
    assert "not an agent" in check(asks("Modeling Agent")["requests"][0])


def test_a_request_needs_new_knowledge_and_evidence():
    assert "no instructions or goal" in check(asks("Replication Agent", instructions=())["requests"][0])
    assert "no verified evidence" in check(asks("Replication Agent", evidence=False)["requests"][0])


def test_cloned_code_is_not_revised():
    assert "cloned" in check(asks("Replication Agent")["requests"][0], source="cloned_repository")


def test_a_repeated_request_is_declined():
    req = asks("Replication Agent", ["fix"])["requests"][0]
    assert "already run" in check(req, [rerun(1, "Replication Agent", instructions=["fix"])])


def test_a_streak_with_progress_is_not_redundant():
    history = [rerun(i, "Data Agent", instructions=[f"s{i}"]) for i in (1, 2)] + \
              [rerun(3, "Data Agent", improved=True, instructions=["s3"])]
    assert check(asks("Data Agent", ["s4"])["requests"][0], history) == ""
    stalled = [rerun(i, "Data Agent", instructions=[f"s{i}"]) for i in (1, 2, 3)]
    assert check(asks("Data Agent", ["s4"])["requests"][0], stalled).startswith("redundant")


def test_data_goals_must_be_above_the_rows_used_and_not_already_failed():
    assert "not above" in check(eval_request("rerun_data_agent", goal={"target_entries": 900}))
    failed = [rerun(1, "Data Agent", goal={"target_entries": 9000}, found_no_data=True)]
    assert "found no usable new data" in check(eval_request("rerun_data_agent", goal={"target_entries": 8000}),
                                               failed)
    assert check(eval_request("rerun_data_agent", goal={"target_entries": 20000}), failed) == ""
