"""Insight memories, step 3 (docs/CLAUDE_BACKEND_PLAN.md, 18.3.3): each
agent reflects on a finished run about its own job. A lesson is stored only
when its quotes and numbers are found in the run's records, at least one
quote is a FACT (a proposal or opinion never is), and a "measured" lesson
quotes a measured result. Each case below comes from a live failure."""
import json
from unittest.mock import patch

from src.agents import master_agent
from src.memory import reflection, retain
from src.orchestration import run_store
from tests.test_memory_insights import XGB, _rerun, _stage
from tests.test_memory_retain import INTAKE, _memories

FINDING = {"category": "preprocessing", "severity": "major", "verified": True,
           "issue": "RFE feature selection is fitted on unbalanced training data without class weights",
           "evidence": "rfe.fit(X_train, y_train)", "fix": "Try 50 or 100 features instead of 30 in RFE"}
RERUN_METRICS = ("f1=0.0583, recall=0.0324", "f1=0.0976, recall=0.0541")


def _run(tmp_path, monkeypatch, reruns=None):
    run_id = run_store.start_run("insurance fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    _stage("Intake Agent", INTAKE, 0)
    _stage("Data Agent", {"mode": "parent_only", "parent_dataset_found": True,
                          "selected_datasets": [{"display_name": "Vehicle Claims", "entry_count": 15420}]}, 0)
    for n, script in enumerate([XGB.format(trees=100, depth=6), XGB.format(trees=500, depth=6)]):
        folder = tmp_path / f"replication_{n}"
        folder.mkdir(parents=True)
        (folder / "train.py").write_text(script, encoding="utf-8")
        _stage("Replication Agent", {"status": "OK", "source": "reconstructed", "entry_point": "train.py",
                                     "replication_dir": str(folder), "exit_code": 0}, n * master_agent.MAX_ATTEMPTS)
        _stage("Evaluation Agent", {"status": "FAILED", "metrics": [{"metric": "f1", "published_value": 0.9926,
                                                                     "reproduced_value": [0.0583, 0.0976][n]}],
                                    "review": {"static_findings": [], "findings": [FINDING] if n == 0 else [],
                                               "assessment": "The order of the steps is the most likely cause."},
                                    "decision": {"action": "revise_replication",
                                                 "instructions": ["Use class_weight='balanced' in RFE"]}},
               n * master_agent.MAX_ATTEMPTS)
    reruns = reruns if reruns is not None else [_rerun(1, *RERUN_METRICS, by="Evaluation Agent")]
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", parent_paper={"title": "A fraud paper"},
                         result={"phase": "EXPERIMENTATION", "structured": INTAKE, "reruns": reruns,
                                 "evaluation_result": {"status": "FAILED", "metrics": [
                                     {"metric": "f1", "published_value": 0.9926, "reproduced_value": 0.0976,
                                      "difference_points": 89.5, "classification": "FAILED"}]},
                                 "cleaning_result": {"rows_total": 15420, "columns_total": 34}})
    return run_id


MEASURED = {"claim": "Fitting RFE with class weights on the unbalanced data raised recall from 0.0324 to 0.0541.",
            "basis": "measured", "applies_when": "random forest RFE on imbalanced fraud data",
            "action": "Fit RFE with class_weight='balanced'.", "polarity": "positive",
            "evidence": ["RFE feature selection is fitted on unbalanced training data without class weights",
                         f"metrics {RERUN_METRICS[0]} -> {RERUN_METRICS[1]}"]}
OBSERVED = {"claim": "RFE was fitted on the unbalanced training data without class weights in the first baseline.",
            "basis": "observed", "polarity": "negative",
            "evidence": ["RFE feature selection is fitted on unbalanced training data without class weights"]}
FROM_A_PROPOSAL = {"claim": "Using 50 or 100 features instead of 30 improved recall on this fraud data.",
                   "basis": "measured", "polarity": "positive", "evidence": ["Try 50 or 100 features instead of 30 in RFE"]}
FROM_AN_OPINION = {"claim": "The order of the steps caused the low recall on this fraud data.", "basis": "observed",
                   "polarity": "negative", "evidence": ["The order of the steps is the most likely cause"]}


def test_numbers_must_be_in_the_facts_rounded_or_rescaled():
    facts = "metrics f1=0.0976, recall=0.0541; gap 27.076 -> 28.645"
    assert reflection.numbers_grounded("F1 rose to 0.10 (9.76%), the gap grew to 28.6 points", facts) == []
    assert reflection.numbers_grounded("recall reached 0.78 with 500 trees", facts) == [0.78, 500.0]
    assert reflection.numbers_grounded("F1 and rerun ids are not numbers", facts) == []


def test_records_separate_measured_results_facts_and_proposals(tmp_path, monkeypatch):
    blocks = reflection.records(retain.load_run(_run(tmp_path, monkeypatch)), "Evaluation Agent", ["an insight"])
    assert f"metrics {RERUN_METRICS[0]} -> {RERUN_METRICS[1]}" in blocks["measured"]
    assert blocks["measured"] in blocks["facts"]
    assert "RFE feature selection is fitted on unbalanced" in blocks["facts"]  # a verified finding is a fact
    for suggestion in ("Try 50 or 100 features", "Use class_weight", "most likely cause"):
        assert suggestion in blocks["proposals"] and suggestion not in blocks["facts"]
    assert "insurance fraud detection" in blocks["context"] and "an insight" in blocks["context"]
    assert "insurance fraud detection" not in blocks["facts"]  # the context is shown, never quotable


def test_only_lessons_resting_on_facts_are_accepted(tmp_path, monkeypatch):
    blocks = reflection.records(retain.load_run(_run(tmp_path, monkeypatch)), "Evaluation Agent", [])
    assert reflection.check(MEASURED, blocks) == ""
    assert reflection.check(OBSERVED, blocks) == ""
    assert reflection.check(FROM_A_PROPOSAL, blocks) == "every quote is from a proposal; a lesson needs a recorded fact"
    assert reflection.check(FROM_AN_OPINION, blocks) == "every quote is from a proposal; a lesson needs a recorded fact"
    assert reflection.check(dict(OBSERVED, basis="measured"), blocks) == "a measured lesson must quote a measured result"


def test_short_labels_are_not_evidence(tmp_path, monkeypatch):
    blocks = reflection.records(retain.load_run(_run(tmp_path, monkeypatch)), "Evaluation Agent", [])
    label = dict(OBSERVED, evidence=["random_forest"])  # live: the Data Agent quoted a task-scope label
    assert reflection.check(label, blocks).startswith("evidence too short to support a claim")
    wrapped = dict(OBSERVED, evidence=['"RFE feature selection is fitted on unbalanced training data..."'])
    assert reflection.check(wrapped, blocks) == ""  # quote marks a model adds are not part of the quote


def test_every_other_check(tmp_path, monkeypatch):
    blocks = reflection.records(retain.load_run(_run(tmp_path, monkeypatch)), "Evaluation Agent", [])
    assert "too short" in reflection.check(dict(OBSERVED, claim="Use RFE."), blocks)
    assert reflection.check(dict(OBSERVED, polarity="maybe"), blocks) == "unknown polarity"
    assert reflection.check(dict(OBSERVED, basis="certain"), blocks).startswith("unknown basis")
    assert reflection.check(dict(OBSERVED, evidence=[]), blocks) == "no evidence quote"
    assert "not found" in reflection.check(dict(OBSERVED, evidence=["a quote that was never written anywhere"]), blocks)
    assert "0.5" in reflection.check(dict(OBSERVED, claim=OBSERVED["claim"] + " Precision reached 0.5."), blocks)
    assert reflection.check("not a lesson", blocks) == "not an object"


def test_the_data_agent_reflects_only_on_its_own_job(tmp_path, monkeypatch):
    reruns = [_rerun(1, *RERUN_METRICS, by="Evaluation Agent"),
              dict(_rerun(2, "f1=0.0976", "f1=0.11", by="Evaluation Agent"), target_agent="Data Agent")]
    run = retain.load_run(_run(tmp_path, monkeypatch, reruns=reruns))
    blocks = reflection.records(run, "Data Agent", [])
    assert "rerun 2" in blocks["measured"] and "rerun 1" not in blocks["measured"]
    assert "Earlier evaluations' verified findings" not in blocks["facts"]
    assert "Final baseline script" not in blocks["facts"]
    assert "finding, choosing and downloading the dataset" in reflection.prompt("Data Agent", blocks)


def test_each_agent_that_ran_reflects_once_and_good_lessons_are_stored(tmp_path, monkeypatch):
    run = retain.load_run(_run(tmp_path, monkeypatch))
    answer = json.dumps({"lessons": [MEASURED, FROM_A_PROPOSAL]})
    with patch("src.llm.chat", return_value=answer) as chat:
        report = reflection.reflect_run(run, [])
    asked = [c.args[0].split(" of a multi-agent")[0].removeprefix("You are the ") for c in chat.call_args_list]
    assert asked == ["Data Agent", "Replication Agent", "Evaluation Agent"]  # the Experiment Agent did not run
    stored = [m for m in _memories(run["run_id"]) if (m.get("evidence") or {}).get("rule") == "reflection"]
    # The Data Agent's records hold no RFE finding, so MEASURED is rejected there too.
    assert {m["agent"] for m in stored} == {"Replication Agent", "Evaluation Agent"}
    assert len(report["lessons"]) == 2 and report["answered"] == {a: 2 for a in asked}
    lesson = next(m for m in stored if m["agent"] == "Evaluation Agent")
    assert lesson["kind"] == "lesson" and lesson["polarity"] == "positive"
    assert lesson["state"] == "candidate"  # not recalled until approved or confirmed by another run
    assert lesson["title"].startswith("Lesson (measured): Fitting RFE with class weights")
    assert lesson["summary"].startswith("Measured: ") and "Seen once" in lesson["summary"]
    assert lesson["quality"] == reflection.QUALITY["measured"]


def test_a_hypothesis_is_stored_with_its_label_and_lower_quality(tmp_path, monkeypatch):
    run = retain.load_run(_run(tmp_path, monkeypatch))
    hypothesis = dict(OBSERVED, basis="hypothesis",
                      claim="Fitting RFE without class weights may drop fraud features; it was not tested alone.")
    with patch("src.llm.chat", return_value=json.dumps({"lessons": [hypothesis]})):
        reflection.reflect_run(run, [])
    [lesson] = [m for m in _memories(run["run_id"], agent="Evaluation Agent")
                if (m.get("evidence") or {}).get("rule") == "reflection"]
    assert lesson["summary"].startswith("Hypothesis (not tested): ")
    assert lesson["quality"] == reflection.QUALITY["hypothesis"] < reflection.QUALITY["observed"]


def test_failures_and_unreadable_answers_are_reported_not_silent(tmp_path, monkeypatch):
    run = retain.load_run(_run(tmp_path, monkeypatch))
    with patch("src.llm.chat", side_effect=RuntimeError("HTTP 502")):
        report = reflection.reflect_run(run, [])
    assert report["lessons"] == [] and len(report["errors"]) == 3
    with patch("src.llm.chat", return_value="I have no lessons to share."):
        report = reflection.reflect_run(run, [])
    assert all(e.endswith("the answer had no lessons list") for e in report["errors"]) and len(report["errors"]) == 3
    with patch("src.llm.chat", return_value='{"lessons": []}'):
        report = reflection.reflect_run(run, [])
    assert report["errors"] == [] and set(report["answered"].values()) == {0}


def test_retain_runs_reflection_instead_of_the_curator(tmp_path, monkeypatch):
    with patch("src.memory.reflection.reflect_run", return_value={"lessons": []}) as reflect, \
            patch("src.memory.curator.curate_run", return_value={}) as curate:
        _run(tmp_path, monkeypatch)
    reflect.assert_called_once()
    curate.assert_not_called()
    monkeypatch.setattr("src.config.MEMORY_REFLECTION", False)
    with patch("src.memory.reflection.reflect_run") as reflect, \
            patch("src.memory.curator.curate_run", return_value={}) as curate:
        _run(tmp_path / "second", monkeypatch)
    reflect.assert_not_called()
    curate.assert_called_once()
