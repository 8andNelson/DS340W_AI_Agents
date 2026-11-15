"""The Parent Paper's published values: a paper that reports many scenarios
(models x preprocessing x datasets) in results tables.

In run 20261115T171901Z-86734b16 the paper's Table 5 gave accuracy, F1,
recall, precision and AUC for every scenario, but the Evaluation Agent
reported "Not stated in the paper's own text" for every metric: the model
answered with a list of AUC values per metric, which the parser dropped,
and the excerpt it read had no tables at all."""
import json
from unittest.mock import patch

from src.agents import evaluation_agent
from src.search import paper_fulltext

# A paper shaped like the live one, as pypdf extracts it.
FILLER = "\n".join(f"Related work sentence {i} about fraud detection methods in insurance." for i in range(400))
PAPER = f"""Enhancing Insurance Fraud Detection Accuracy
Abstract
The proposed KBD + RFE + Oversampling + RF framework achieved the highest AUC and F1-score.
{FILLER}
3.1 Datasets
The insurance fraud dataset contains 15,419 claims. The insurance claims dataset contains 10,302 car insurance claims.
{FILLER}
Experiment 6 This final experiment assessed the combined influence of discretization, feature selection and
resampling. The (KBD + RFE + Over + RF) combination achieved the highest AUC scores of 99.26% on the first
dataset and 89.29% on the second.
{FILLER}
Scenarios Dataset)1( Dataset (2)
AC F1 RC PR AUC AC F1 RC PR AUC
RF 0.9361 0.0837 0.0452 0.5625 0.5214 0.7598 0.4343 0.3220 0.6667 0.6287
Over + RF 0.9550 0.9550 1.0000 0.9401 0.9550 0.8824 0.8874 0.8720 0.8753 0.8817
KBD + RFE + Over + RF 0.9926 0.9926 1.0000 0.9953 0.9926 0.8928 0.8986 0.9349 0.8650 0.8921
Table 5 The performance of the classification models on different scenarios
{FILLER}
"""

LIVE_REPLY = ('{"auc": [{"value": 68.15, "unit": "percent"}, {"value": 99.26, "unit": "percent"}, '
              '{"value": 89.29, "unit": "percent"}]}')


def test_results_text_keeps_the_results_tables_and_the_dataset_sizes():
    assert len(PAPER) > evaluation_agent.PUBLISHED_TEXT_CHARS  # the whole paper does not fit
    text = paper_fulltext.results_text(PAPER, max_chars=evaluation_agent.PUBLISHED_TEXT_CHARS)
    assert len(text) <= evaluation_agent.PUBLISHED_TEXT_CHARS + 200
    assert "KBD + RFE + Over + RF 0.9926 0.9926 1.0000 0.9953 0.9926" in text
    assert "AC F1 RC PR AUC" in text and "Table 5" in text
    assert "15,419" in text and "99.26%" in text


def test_table_blocks_ignore_prose_and_keep_whole_tables():
    tables = paper_fulltext.table_blocks(PAPER)
    assert tables.count("RF 0.") == 3
    # Only the lines just around the table come with it, never the prose pages.
    assert tables.count("Related work sentence") <= 2 * paper_fulltext.TABLE_CONTEXT_LINES


def test_live_reply_with_several_values_is_ambiguous_not_unstated():
    details = {}
    with patch.object(evaluation_agent.llm, "chat", return_value=LIVE_REPLY):
        metrics = evaluation_agent._extract_metrics(PAPER, "a paper", max_chars=len(PAPER), verify=True,
                                                    details=details)
    assert metrics == {}  # several values and no scenario chosen: none is used, never a guess
    assert details["status"] == "AMBIGUOUS"
    assert any("several values" in n for n in details["notes"]) and details["raw_reply"]
    assert any(n.startswith("auc: 68.15") for n in details["notes"])  # not in this paper: UNVERIFIED


def test_a_list_whose_values_agree_is_used():
    reply = '{"auc": [{"value": 99.26, "unit": "percent"}, {"value": 0.9926, "unit": "fraction"}]}'
    with patch.object(evaluation_agent.llm, "chat", return_value=reply):
        assert evaluation_agent._extract_metrics(PAPER, "a paper", max_chars=len(PAPER)) == {"auc": 0.9926}


def test_the_matching_scenario_is_read_and_each_value_is_checked_against_the_text():
    reply = json.dumps({"scenario": "KBD + RFE + Over + RF", "dataset": "Dataset (1)", "reason": "matches",
                        "metrics": {"accuracy": {"value": 0.9926, "unit": "fraction"},
                                    "recall": {"value": 1.0, "unit": "fraction"},
                                    "precision": {"value": 0.9953, "unit": "fraction"},
                                    "auc": {"value": 99.26, "unit": "percent"},
                                    "mcc": {"value": 0.97, "unit": "fraction"}}})  # 0.97 is not in the paper
    details = {}
    with patch.object(evaluation_agent.llm, "chat", return_value=reply) as chat:
        metrics = evaluation_agent._extract_metrics(PAPER, "a paper", max_chars=len(PAPER),
                                                    context="Method details: oversampling, RFE, random forest.",
                                                    verify=True, details=details)
    assert metrics == {"accuracy": 0.9926, "recall": 1.0, "precision": 0.9953, "auc": 0.9926}
    assert details["status"] == "FOUND" and details["scenario"] == "KBD + RFE + Over + RF"
    assert any(n.startswith("mcc:") and "UNVERIFIED" in n for n in details["notes"])
    prompt = chat.call_args.args[0]
    assert "WHAT WAS REPRODUCED" in prompt and "oversampling, RFE" in prompt


def test_an_unreadable_answer_is_a_failed_extraction():
    details = {}
    with patch.object(evaluation_agent.llm, "chat", return_value="I could not find it."):
        assert evaluation_agent._extract_metrics(PAPER, "a paper", details=details) == {}
    assert details["status"] == "FAILED"
    details = {}
    with patch.object(evaluation_agent.llm, "chat", side_effect=RuntimeError("HTTP 502")):
        evaluation_agent._extract_metrics(PAPER, "a paper", details=details)
    assert details["status"] == "FAILED" and "HTTP 502" in details["notes"][0]


def _run(reply: str, tmp_path):
    stdout = tmp_path / "run_stdout.txt"
    stdout.write_text("METRIC accuracy=0.94\nMETRIC f1=0.0976\nMETRIC roc_auc=0.7734\n", encoding="utf-8")
    replication = {"status": "OK", "source": "reconstructed", "stdout_path": str(stdout),
                   "inferred_assumptions": ["RandomOverSampler is applied before RFE.", "A random forest classifier."]}
    paper = {"status": "FULL_TEXT", "text": PAPER}
    with patch.object(evaluation_agent.paper_fulltext, "get_parent_paper_text", return_value=paper), \
            patch.object(evaluation_agent.llm, "chat", return_value=reply) as chat, \
            patch.object(evaluation_agent, "review", return_value={
                "decision": {"action": "accept", "target_agent": ""}, "static_findings": [], "decision_notes": [],
                "review_status": "OK"}), \
            patch.object(evaluation_agent, "_request_for", return_value=[]):
        result = evaluation_agent.run({"title": "Enhancing Insurance Fraud Detection Accuracy"}, replication,
                                      cleaning_result={"rows_total": 15420},
                                      dataset_result={"selected_datasets": [{"display_name": "Vehicle Claim Fraud"}]})
    return result, chat.call_args_list[0].args[0]


def test_run_compares_against_the_papers_table_values(tmp_path):
    reply = json.dumps({"scenario": "KBD + RFE + Over + RF", "dataset": "Dataset (1)",
                        "metrics": {"accuracy": {"value": 0.9926, "unit": "fraction"},
                                    "f1": {"value": 0.9926, "unit": "fraction"},
                                    "auc": {"value": 0.9926, "unit": "fraction"}}})
    result, prompt = _run(reply, tmp_path)
    assert "0.9926 0.9926 1.0000 0.9953" in prompt  # the table reached the model
    assert "15420 rows" in prompt and "RandomOverSampler" in prompt
    by_name = {m["metric"]: m for m in result["metrics"]}
    assert by_name["accuracy"]["published_value"] == 0.9926
    assert by_name["f1"]["classification"] == "FAILED"
    assert by_name["auc"]["reproduced_value"] == 0.7734  # the run printed roc_auc
    assert "KBD + RFE + Over + RF" in by_name["f1"]["notes"]
    assert result["status"] == "FAILED"
    assert result["published_extraction"]["status"] == "FOUND"


def test_a_papers_auc_matches_a_runs_roc_auc():
    for paper_name, run_name in (("AUC", "roc_auc"), ("AUC-ROC", "roc_auc"), ("F1-score", "f1"),
                                 ("Sensitivity", "recall"), ("ACC", "accuracy")):
        assert evaluation_agent._normalize_metric_name(paper_name) == evaluation_agent._normalize_metric_name(run_name)
    assert evaluation_agent._normalize_metric_name("precision") != evaluation_agent._normalize_metric_name("recall")


def test_run_never_says_not_stated_when_the_values_could_not_be_read(tmp_path):
    result, _prompt = _run(LIVE_REPLY, tmp_path)
    assert result["status"] == "INCONCLUSIVE"
    assert result["published_extraction"]["status"] == "AMBIGUOUS"
    for metric in result["metrics"]:
        assert "Not stated" not in metric["notes"]
        assert "could not be read reliably" in metric["notes"]
    assert any("does not mean the paper states no result" in c for c in result["likely_causes"])
