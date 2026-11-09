"""The Experiment Agent as an expert data scientist: it reads the gap and
the Evaluation Agent's findings and returns ONE checked rerun request (or
"stop") for Master Agent. It never runs code itself."""
import json
from pathlib import Path
from unittest.mock import patch

from src.agents import experiment_agent

PAPER = {"title": "Fraud Detection With XGBoost", "official_abstract": "XGBoost reached an accuracy of 99.2%.",
         "results_summary": "accuracy 99.2%"}


def metric(published=99.2, reproduced=95.0, diff=4.2, classification="APPROXIMATE", name="accuracy"):
    return {"metric": name, "published_value": published, "reproduced_value": reproduced,
            "difference_points": diff, "classification": classification, "notes": ""}


def evaluation(metrics=None, findings=None):
    return {"status": "APPROXIMATE", "metrics": [metric()] if metrics is None else metrics,
            "review": {"assessment": "The script trains on 10% of the rows.", "static_findings": [],
                       "findings": findings or [], "data_facts": {"master_rows": 28000}},
            "decision": {"action": "run_experiment", "reason": "valid baseline"}}


def replication(tmp_path: Path, source="reconstructed"):
    slug = tmp_path / "replication" / "paper"
    (slug / "reconstructed").mkdir(parents=True)
    (slug / "reconstructed" / "train.py").write_text("model = XGBClassifier(n_estimators=50)\n")
    (slug / "run_stdout.txt").write_text("METRIC accuracy=0.95\n")
    return {"status": "OK", "source": source, "replication_dir": str(slug),
            "stdout_path": str(slug / "run_stdout.txt"), "inferred_assumptions": ["n_estimators=50 was guessed"]}


def answer(action="rerun_agent", target="Replication Agent", instructions=("Use n_estimators=500 as the paper "
           "states.",), evidence=("n_estimators=50",), target_entries=None):
    return json.dumps({
        "analysis": "The model is undertrained.",
        "hypotheses": [{"cause": "Too few trees", "evidence": "n_estimators=50", "fix": "Replication Agent"}],
        "decision": {"action": action, "target_agent": target, "instructions": list(instructions),
                     "target_entries": target_entries, "reason": "Match the paper's model size.",
                     "expected_effect": "accuracy closer to 99.2", "evidence": list(evidence)},
    })


def run(tmp_path, content, evaluation_result=None, **kwargs):
    with patch("src.llm.chat", return_value=content) as chat:
        result = experiment_agent.run(PAPER, replication(tmp_path, kwargs.pop("source", "reconstructed")),
                                      evaluation_result or evaluation(), **kwargs)
    return result, chat


def test_a_supported_request_goes_to_master_agent(tmp_path):
    result, chat = run(tmp_path, answer())
    assert result["status"] == "REQUEST"
    assert result["gap_points"] == 4.2
    request = result["requests"][0]
    assert request["sender"] == "experiment_agent" and request["recipient"] == "master_agent"
    assert request["target_agent"] == "Replication Agent"
    assert request["instructions"] == ["Use n_estimators=500 as the paper states."]
    assert request["evidence"] == [{"issue": "quoted by the Experiment Agent", "evidence": "n_estimators=50"}]
    assert result["hypotheses"][0]["verified"] is True
    assert chat.call_args.kwargs["agent"] == "experiment_agent"


def test_the_prompt_has_the_gap_findings_history_and_master_feedback(tmp_path):
    history = [{"rerun": 1, "requested_by": "Experiment Agent", "target_agent": "Data Agent",
                "instructions": ["collect more"], "goal": {"target_entries": 50000},
                "outcome": "gap 4.5 -> 4.2 points"}]
    feedback = ["Master Agent declined the request to rerun the Data Agent: redundant"]
    findings = [{"severity": "major", "category": "hyperparameters", "issue": "Too few trees",
                 "evidence": "n_estimators=50", "verified": True}]
    _result, chat = run(tmp_path, answer(), evaluation(findings=findings), history=history, feedback=feedback)
    prompt = chat.call_args.args[0]
    assert "mean gap: 4.2" in prompt and "Too few trees" in prompt
    assert "gap 4.5 -> 4.2 points" in prompt
    assert "redundant" in prompt
    assert "Closest, not highest" in prompt and "leakage" in prompt


def test_invented_evidence_is_marked_unverified_and_not_sent(tmp_path):
    result, _chat = run(tmp_path, answer(evidence=("the paper used 1,000 trees",)))
    assert result["evidence"] == [{"quote": "the paper used 1,000 trees", "verified": False}]
    assert result["requests"][0]["evidence"] == []  # Master Agent declines a request with no verified evidence


def test_agent_names_are_normalized():
    assert experiment_agent.canonical_agent("data_agent") == "Data Agent"
    assert experiment_agent.canonical_agent("code discovery") == "Code Discovery Agent"
    assert experiment_agent.canonical_agent("Research Agent") == "Research Agent"
    assert experiment_agent.canonical_agent("Modeling Agent") == ""


def test_an_unknown_target_becomes_stop(tmp_path):
    result, _chat = run(tmp_path, answer(target="Modeling Agent"))
    assert result["status"] == "STOP" and result["requests"] == []
    assert any("Unknown target" in n for n in result["decision_notes"])


def test_a_closed_gap_stops(tmp_path):
    result, _chat = run(tmp_path, answer(), evaluation([metric(reproduced=98.8, diff=0.4, classification="EXACT")]))
    assert result["status"] == "STOP"
    assert any("already within" in n for n in result["decision_notes"])


def test_no_published_number_means_no_gap_to_close(tmp_path):
    no_published = [metric(published=None, diff=None, classification="INCONCLUSIVE")]
    result, _chat = run(tmp_path, answer(), evaluation(no_published))
    assert result["status"] == "STOP" and result["gap_points"] is None


def test_a_missing_reproduced_metric_can_still_be_requested(tmp_path):
    not_printed = [metric(reproduced=None, diff=None, classification="INCONCLUSIVE")]
    result, _chat = run(tmp_path, answer(instructions=("Print METRIC accuracy=<value>.",)), evaluation(not_printed))
    assert result["status"] == "REQUEST"


def test_a_data_goal_is_capped_and_only_kept_for_the_data_agent(tmp_path):
    result, _chat = run(tmp_path, answer(target="data agent", target_entries=10_000_000))
    assert result["requests"][0]["goal"] == {"target_entries": experiment_agent.MAX_MASTER_ROWS}
    result, _chat = run(Path(tmp_path) / "b", answer(target_entries=90000))
    assert result["requests"][0]["goal"] == {}


def test_a_failed_analysis_call_stops_without_raising(tmp_path):
    with patch("src.llm.chat", side_effect=RuntimeError("router down")):
        result = experiment_agent.run(PAPER, replication(tmp_path), evaluation())
    assert result["status"] == "STOP"
    assert any("router down" in n for n in result["decision_notes"])


def test_score_gap_is_the_mean_over_metrics_on_both_sides():
    assert experiment_agent.score_gap({"metrics": [metric(diff=4.0), metric(diff=2.0, name="f1"),
                                                   metric(published=None, diff=None, name="auc")]}) == 3.0
    assert experiment_agent.score_gap({"metrics": []}) is None
