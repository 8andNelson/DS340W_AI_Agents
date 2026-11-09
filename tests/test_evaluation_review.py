"""The Evaluation Agent's senior data analyst review: deterministic leakage
checks, evidence-checked LLM findings, and rule-checked decisions that
become requests to Master Agent."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.agents import evaluation_agent

LEAKY_SCRIPT = """import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
df = pd.concat([pd.read_csv(os.environ["TRAIN_CSV"]), pd.read_csv(os.environ["TEST_CSV"])])
X, y = df.drop(columns=["is_fraud"]), df["is_fraud"]
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
model.fit(X_train, y_train)
print("METRIC accuracy=0.999")
"""
CLEAN_SCRIPT = """import os
import pandas as pd
train = pd.read_csv(os.environ["TRAIN_CSV"]); test = pd.read_csv(os.environ["TEST_CSV"])
X_res, y_res = SMOTE(random_state=42).fit_resample(train.drop(columns=["is_fraud"]), train["is_fraud"])
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)
print("METRIC accuracy=0.95")
"""


def make_replication(tmp_path: Path, script: str, source: str = "reconstructed") -> dict:
    slug = tmp_path / "replication" / "paper"
    (slug / "reconstructed").mkdir(parents=True)
    (slug / "reconstructed" / "train.py").write_text(script)
    (slug / "run_stdout.txt").write_text("METRIC accuracy=0.95\n")
    (slug / "run_stderr.txt").write_text("")
    return {"status": "OK", "source": source, "replication_dir": str(slug), "entry_point": "train.py",
            "stdout_path": str(slug / "run_stdout.txt"), "stderr_path": str(slug / "run_stderr.txt"),
            "inferred_assumptions": [], "modifications": []}


def make_data(tmp_path: Path, rows: int = 1000, positives: int = 20):
    train = tmp_path / "train.csv"
    pd.DataFrame({"amount": range(rows), "is_fraud": [1] * positives + [0] * (rows - positives)}).to_csv(train,
                                                                                                    index=False)
    cleaning = {"rows_total": rows, "split_rows": {"train": rows, "test": 200, "validation": 100},
                "train_csv_path": str(train), "label_column": "is_fraud", "sampled": False, "datasets_merged": 1}
    dataset = {"mode": "parent_only", "parent_dataset_found": True,
               "data_plan": {"paper_dataset_size": 284807, "target_entries": 284807, "reasoning": "match the paper"}}
    return cleaning, dataset


def llm_answer(findings=None, action="accept", **decision) -> str:
    return json.dumps({"assessment": "Reviewed.", "findings": findings or [],
                       "decision": {"action": action, "reason": "because", **decision}})


PAPER = {"title": "Fraud", "official_abstract": "We use 284,807 transactions and report accuracy of 99.9%."}


def run_eval(tmp_path, script, answer, source="reconstructed", rows=1000):
    replication = make_replication(tmp_path, script, source)
    cleaning, dataset = make_data(tmp_path, rows)
    with patch.object(evaluation_agent, "_extract_metrics", return_value={"accuracy": 99.9}), \
         patch.object(evaluation_agent.llm, "chat", return_value=answer) as chat:
        result = evaluation_agent.run(PAPER, replication, cleaning, dataset)
    return result, chat


def test_smote_before_the_split_forces_a_fix_request_even_if_the_llm_accepts(tmp_path):
    result, chat = run_eval(tmp_path, LEAKY_SCRIPT, llm_answer(action="accept"))

    checks = [f["check"] for f in result["review"]["static_findings"]]
    assert "resample_before_split" in checks
    assert result["decision"]["action"] == "revise_replication"
    assert result["decision"]["target_agent"] == "Replication Agent"
    assert any("Split first" in i for i in result["decision"]["instructions"])
    assert any("leakage must be fixed" in n for n in result["review"]["decision_notes"])
    request = result["requests"][0]
    assert request["recipient"] == "master_agent" and request["action"] == "revise_replication"
    assert request["evidence"]
    prompt = chat.call_args.args[0]
    assert "senior data analyst" in prompt and "fit_resample(X, y)" in prompt


def test_leakage_in_cloned_code_is_reported_but_not_revised(tmp_path):
    replication = make_replication(tmp_path, "", source="cloned_repository")
    repo = Path(replication["replication_dir"]) / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(LEAKY_SCRIPT)
    cleaning, dataset = make_data(tmp_path)
    with patch.object(evaluation_agent, "_extract_metrics", return_value={}), \
         patch.object(evaluation_agent.llm, "chat", return_value=llm_answer()):
        result = evaluation_agent.run(PAPER, replication, cleaning, dataset)
    assert result["review"]["static_findings"]
    assert result["decision"]["action"] == "accept"
    assert result["requests"] == []


def test_verified_insufficient_data_finding_becomes_a_data_request(tmp_path):
    finding = {"category": "insufficient_data", "severity": "major",
               "issue": "Only 1000 rows were used; the paper used 284,807.", "evidence": '"master_rows": 1000',
               "fix": "Collect more data."}
    result, _ = run_eval(tmp_path, CLEAN_SCRIPT, llm_answer([finding], "rerun_data_agent", target_entries=50000))

    assert result["review"]["findings"][0]["verified"] is True
    assert result["decision"]["action"] == "rerun_data_agent"
    assert result["requests"][0]["target_agent"] == "Data Agent"
    assert result["requests"][0]["goal"] == {"target_entries": 50000}


def test_data_request_without_verified_evidence_is_not_sent(tmp_path):
    finding = {"category": "insufficient_data", "severity": "major", "issue": "Too little data.",
               "evidence": "the paper used ten million rows", "fix": ""}
    result, _ = run_eval(tmp_path, CLEAN_SCRIPT, llm_answer([finding], "rerun_data_agent", target_entries=50000))

    assert result["review"]["findings"][0]["verified"] is False
    assert result["decision"]["action"] == "accept"
    assert result["requests"] == []


def test_data_goal_must_be_meaningfully_above_the_rows_used(tmp_path):
    finding = {"category": "insufficient_data", "severity": "major", "issue": "x", "evidence": '"master_rows": 1000'}
    result, _ = run_eval(tmp_path, CLEAN_SCRIPT, llm_answer([finding], "rerun_data_agent", target_entries=1100))
    assert result["decision"]["action"] == "accept"


def test_data_goal_is_capped_at_the_row_budget(tmp_path):
    finding = {"category": "insufficient_data", "severity": "major", "issue": "x", "evidence": '"master_rows": 1000'}
    result, _ = run_eval(tmp_path, CLEAN_SCRIPT, llm_answer([finding], "rerun_data_agent", target_entries=10**9))
    assert result["decision"]["target_entries"] == evaluation_agent.MAX_MASTER_ROWS


def test_revision_needs_a_verified_serious_finding(tmp_path):
    weak = {"category": "hyperparameters", "severity": "minor", "issue": "n_estimators may differ",
            "evidence": "n_estimators=100", "fix": "use 500"}
    result, _ = run_eval(tmp_path, CLEAN_SCRIPT, llm_answer([weak], "revise_replication", instructions=["use 500"]))
    assert result["decision"]["action"] == "accept"

    strong = dict(weak, severity="major", issue="The paper states 500 trees; the script uses 100.")
    result, _ = run_eval(tmp_path / "b", CLEAN_SCRIPT,
                         llm_answer([strong], "revise_replication", instructions=["use 500 trees"]))
    assert result["decision"]["action"] == "revise_replication"
    assert any("500" in i for i in result["decision"]["instructions"])


def test_experiment_recommendation_keeps_a_supported_category(tmp_path):
    result, _ = run_eval(tmp_path, CLEAN_SCRIPT, llm_answer(action="run_experiment",
                                                             experiment_category="alternative_sampling_technique",
                                                             instructions=["try ADASYN"]))
    assert result["decision"]["action"] == "run_experiment"
    assert result["requests"][0]["goal"] == {"experiment_category": "alternative_sampling_technique"}

    result, _ = run_eval(tmp_path / "b", CLEAN_SCRIPT, llm_answer(action="run_experiment",
                                                                   experiment_category="quantum_annealing"))
    assert result["decision"]["experiment_category"] == ""


def test_review_failure_still_returns_a_result(tmp_path):
    replication = make_replication(tmp_path, CLEAN_SCRIPT)
    with patch.object(evaluation_agent, "_extract_metrics", return_value={}), \
         patch.object(evaluation_agent.llm, "chat", side_effect=RuntimeError("router down")):
        result = evaluation_agent.run(PAPER, replication)
    assert result["decision"]["action"] == "accept"
    assert "router down" in result["review"]["decision_notes"][0]


def test_data_facts_count_the_minority_class(tmp_path):
    cleaning, dataset = make_data(tmp_path, rows=500, positives=7)
    facts = evaluation_agent.data_facts(cleaning, dataset)
    assert facts["minority_class_train_rows"] == 7
    assert facts["paper_dataset_size"] == 284807
    assert facts["train_rows"] == 500


@pytest.mark.parametrize("evidence,expected", [("fit_resample(X, y)", True), ("FIT_RESAMPLE(x,   y)", True),
                                               ("not in the script at all", False), ("", False)])
def test_evidence_must_occur_in_the_corpus(evidence, expected):
    findings = evaluation_agent._verify([{"category": "data_leakage", "severity": "critical", "issue": "i",
                                          "evidence": evidence}], LEAKY_SCRIPT)
    assert findings[0]["verified"] is expected
