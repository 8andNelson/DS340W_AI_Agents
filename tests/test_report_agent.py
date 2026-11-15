"""Milestone 13: the Report Agent writes the final report only from the run's records."""
import json

import pytest

from src import llm
from src.agents import master_agent, report_agent


PARENT = {"title": "Gradient boosting for credit card fraud", "authors": ["A. Author", "B. Author"], "year": 2023,
          "venue": "Expert Systems with Applications", "doi": "10.1016/j.eswa.2023.1", "paper_url": "https://x/p",
          "validation_status": "VERIFIED", "abstract": "XGBoost reaches an AUC of 0.97 on the ULB dataset.",
          "methodology": "XGBoost with SMOTE on the training split.", "dataset": "ULB Credit Card Fraud"}
OTHER = {"title": "Graph networks for fraud", "authors": [], "year": 2024, "venue": "KDD", "doi": "",
         "paper_url": "https://y", "validation_status": "UNVERIFIED"}


def project(source="reconstructed", status="APPROXIMATE", reruns=None):
    return {
        "topic": "machine learning for detecting credit card fraud",
        "structured": {"domain": "financial fraud detection", "ml_task": "classification",
                       "keywords": ["credit card fraud"], "target_terms": ["fraud"]},
        "target_terms": ["fraud"],
        "validation_result": {"all": [PARENT, OTHER, dict(OTHER, title="Old", validation_status="REJECTED")],
                              "justification": "Clear method and a public dataset.", "candidate_pool_size": 2,
                              "comparison_table": [{"title": PARENT["title"], "publication_year": 2023,
                                                    "composite_score": 8.5, "reproducibility_difficulty": "low",
                                                    "dataset_availability": "public", "code_availability": False}]},
        "parent_paper": PARENT,
        "dataset_result": {"mode": "parent_only", "parent_dataset_found": True, "total_entries": 250000,
                           "data_plan": {"paper_dataset": "ULB", "paper_dataset_size": 284807},
                           "selected_datasets": [{"display_name": "ULB Credit Card Fraud", "name": "https://k/ulb",
                                                  "source": "Kaggle", "license": "DbCL", "entry_count": 250000,
                                                  "original_entry_count": 284807, "sampled": True}]},
        "cleaning_result": {"rows_total": 249000, "columns_total": 31, "datasets_merged": 1, "datasets_attempted": 1,
                            "label_column": "class", "split_rows": {"train": 174300, "test": 49800,
                                                                    "validation": 24900}, "split_seed": 42},
        "code_discovery_result": {"status": "NOT_FOUND"},
        "replication_result": {"status": "OK", "source": source, "repo_url": "https://github.com/a/b",
                               "commit_sha": "abc123", "entry_point": "baseline.py", "exit_code": 0,
                               "duration_seconds": 41.2, "inferred_assumptions": ["learning rate 0.1 (not stated)"],
                               "generated_by": ["qwen3.8-27b"], "paper_text_status": "FULL_TEXT"},
        "evaluation_result": {"status": status, "likely_causes": ["The data was subsampled."],
                              "metrics": [{"metric": "auc", "published_value": 0.97, "reproduced_value": 0.9412,
                                           "difference_points": 2.88, "classification": status}]},
        "reruns": reruns or [],
        "rerun_end_reason": "the Experiment Agent requested no further rerun",
        "experiment_results": [],
    }


def prose(**overrides):
    text = {k: f"The {k.replace('_', ' ')} is grounded in [P1]." for k in report_agent.PROSE_SECTIONS}
    text.update(overrides)
    return json.dumps(text)


@pytest.fixture
def answer(monkeypatch):
    holder = {"text": prose()}
    monkeypatch.setattr(report_agent.llm, "chat", lambda *a, **k: holder["text"])
    return holder


# --- The grounding check ---

def _facts():
    return report_agent.gather_facts(project())


def test_a_sentence_with_a_number_not_in_the_records_is_removed():
    facts = _facts()
    text, removed = report_agent._ground("The AUC was 0.9412. It reached 99.9% recall.", facts,
                                         report_agent.fact_numbers(facts))
    assert text == "The AUC was 0.9412."
    assert removed == [{"sentence": "It reached 99.9% recall.", "reason": "states 99.9, not in the run's records"}]


def test_rounding_and_percentages_of_recorded_numbers_are_allowed():
    facts = _facts()
    known = report_agent.fact_numbers(facts)
    text, removed = report_agent._ground("The reproduced AUC was 94.1%, or 0.94, against 0.97 in [P1]. "
                                         "We used 249,000 rows.", facts, known)
    assert removed == [] and "249,000 rows" in text


def test_an_unknown_citation_is_removed():
    facts = _facts()
    _, removed = report_agent._ground("As shown in [P9], fraud is rare.", facts, report_agent.fact_numbers(facts))
    assert removed[0]["reason"] == "cites P9, not in the literature list"


def test_claims_beyond_the_evaluation_are_removed():
    facts = _facts()  # APPROXIMATE, reconstructed
    known = report_agent.fact_numbers(facts)
    text, removed = report_agent._ground(
        "We exactly reproduced the paper. We ran the authors' original code. "
        "The authors' code was not available. The result is approximate.", facts, known)
    assert text == "The authors' code was not available. The result is approximate."
    assert [r["reason"].split(",")[0] for r in removed] == [
        "claims an exact reproduction", "presents the reconstructed baseline as the authors' code"]
    failed = report_agent.gather_facts(project(status="FAILED"))
    _, removed = report_agent._ground("We successfully reproduced it.", failed, known)
    assert removed and "successful" in removed[0]["reason"]


# --- The report ---

def test_the_report_has_every_section_and_its_tables_come_from_the_records(answer):
    result = report_agent.run(project(), run_id="run-1")
    text = open(result["report_path"], encoding="utf-8").read()
    assert result["status"] == "OK" and result["removed_claims"] == []
    for section in report_agent.SECTIONS + report_agent.APPENDICES:
        assert f" {section}" in text, section
    assert "| auc | 0.97 | 0.9412 | 2.88 | APPROXIMATE | n/a |" in text
    assert "It is not the authors' code." in text and "learning rate 0.1 (not stated)" in text
    assert "(UNVERIFIED)" in text and "Old" not in text  # the rejected paper is not in the literature
    assert "https://doi.org/10.1016/j.eswa.2023.1" in text
    assert "The data was subsampled to the local row budget." in text
    facts = json.load(open(result["facts_path"], encoding="utf-8"))
    assert facts["evaluation"]["metrics"][0]["reproduced_value"] == 0.9412
    assert "metrics.svg" in text and open(result["metrics_figure_path"], encoding="utf-8").read().startswith("<svg")
    assert "gap_figure_path" not in result  # no rerun, nothing to plot over time


def test_a_cloned_baseline_names_the_repository_and_commit(answer):
    text = open(report_agent.run(project(source="cloned_repository"))["report_path"], encoding="utf-8").read()
    assert "The baseline is the authors' published code: https://github.com/a/b at commit abc123." in text


def test_the_gap_figure_shows_kept_and_reverted_reruns(answer):
    reruns = [{"rerun": 1, "requested_by": "Experiment Agent", "target_agent": "Data Agent", "instructions": ["more"],
               "gap_before": 4.0, "gap_after": 2.88, "applied": True, "improved": True, "outcome": "kept"},
              {"rerun": 2, "requested_by": "Experiment Agent", "target_agent": "Replication Agent",
               "instructions": ["tune"], "gap_before": 2.88, "gap_after": 6.1, "applied": False, "outcome": "reverted"}]
    result = report_agent.run(project(reruns=reruns))
    svg = open(result["gap_figure_path"], encoding="utf-8").read()
    assert "Baseline: 4.0 points" in svg and "Rerun 2 (Replication Agent): 6.1 points (not kept)" in svg
    text = open(result["report_path"], encoding="utf-8").read()
    assert "(baseline before reruns: 4.0)" in text and "| 2 | Experiment Agent | Replication Agent | tune |" in text


def test_invented_numbers_in_the_prose_are_removed_and_listed(answer):
    answer["text"] = prose(abstract="We reproduced [P1]. The model reached an F1 of 0.991.")
    result = report_agent.run(project())
    text = open(result["report_path"], encoding="utf-8").read()
    assert "0.991" not in text.split("## Appendix B")[0]
    assert result["removed_claims"][0]["section"] == "abstract"
    assert "Sentences removed by the check: 1." in text


def test_without_the_llm_the_report_is_still_written_and_degraded(monkeypatch):
    def down(*a, **k):
        raise RuntimeError("router down")
    monkeypatch.setattr(report_agent.llm, "chat", down)
    result = report_agent.run(project())
    text = open(result["report_path"], encoding="utf-8").read()
    assert result["status"] == "DEGRADED" and "router down" in result["warnings"][0]
    assert "*This section was not written: RuntimeError: router down*" in text
    assert "| auc | 0.97 | 0.9412 |" in text
    assert master_agent._check_report(result) == []  # degraded prose does not block the run


def test_master_agent_checks_the_file_not_the_summary(answer, tmp_path):
    result = report_agent.run(project())
    assert master_agent._check_report(result) == []
    path = tmp_path / "short.md"
    path.write_text("## 1. Abstract\n", encoding="utf-8")
    issue = master_agent._check_report(dict(result, report_path=str(path)))[0]["issue"]
    assert "lacks the section(s)" in issue and "References" in issue
    assert "was not written" in master_agent._check_report({"report_path": str(tmp_path / "none.md")})[0]["issue"]


def test_a_stale_figure_from_an_earlier_run_is_removed(answer):
    first = report_agent.run(project())
    no_metrics = project()
    no_metrics["evaluation_result"]["metrics"] = []
    second = report_agent.run(no_metrics)
    assert "metrics_figure_path" not in second
    assert not __import__("os").path.exists(first["metrics_figure_path"])


def test_the_models_used_come_from_this_runs_call_log():
    llm.CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(llm.CALL_LOG, "w", encoding="utf-8") as f:
        for entry in ({"run_id": "r1", "served_by": "m-a"}, {"run_id": "r1", "served_by": "m-a"},
                      {"run_id": "r1", "served_by": "m-b", "error": "502"}, {"run_id": "r2", "served_by": "m-c"}):
            f.write(json.dumps(entry) + "\n")
    assert report_agent._models_used("r1") == {"m-a": 2}


def test_an_approved_report_is_never_turned_into_a_memory():
    from src.memory import retain
    from src.orchestration import run_store
    run_id = run_store.start_run("fraud")
    run_store.record_stage(run_id, "Report Agent", 1, "Approved", {"status": "OK", "report_path": "r.md"})
    memories, _ = retain.build_memories(retain.load_run(run_id))
    assert not [m for m in memories if m["agent"] == "Report Agent"]


def test_a_sandboxed_repository_run_is_described():
    line = report_agent._environment_line({"source": "cloned_repository", "sandbox": {
        "image": "python:3.12-slim", "digest": "python@sha256:abc", "run_network": "none",
        "run_user": "65534:65534", "capabilities": "all dropped, no-new-privileges", "memory": "6g", "cpus": "8"}})
    assert "Docker sandbox: python:3.12-slim (python@sha256:abc), network none" in line


def test_a_fallback_says_why_the_repository_did_not_run():
    line = report_agent._environment_line({"source": "reconstructed", "repository_fallback": {
        "repo_url": "https://github.com/a/b", "reason": "Docker is not available."}})
    assert "virtual environment" in line
    assert "https://github.com/a/b did not run (Docker is not available.)" in line
