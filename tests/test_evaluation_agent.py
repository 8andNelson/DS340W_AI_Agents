import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents import evaluation_agent


def make_paper(**overrides) -> dict:
    paper = {
        "title": "Big Data-Driven Distributed ML for Fraud Detection",
        "abstract": "A distributed machine learning framework for credit card fraud detection.",
        "results_summary": "Random forest produces a maximum accuracy of 96% (AUC 98.9%).",
    }
    paper.update(overrides)
    return paper


def make_replication_result(**overrides) -> dict:
    result = {
        "status": "OK", "source": "cloned_repository", "stdout_path": "", "inferred_assumptions": [],
        "modifications": [],
    }
    result.update(overrides)
    return result


def llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    response.raise_for_status = lambda: None
    return response


class TestExtractMetrics(unittest.TestCase):
    def test_empty_text_returns_empty_without_llm_call(self):
        with patch("requests.post") as post_mock:
            result = evaluation_agent._extract_metrics("", "test")
        post_mock.assert_not_called()
        self.assertEqual(result, {})

    def test_parses_valid_json_response(self):
        response = llm_response('{"accuracy": 94.2, "f1": 0.91}')
        with patch("requests.post", return_value=response):
            result = evaluation_agent._extract_metrics("some text with accuracy 94.2%", "test")
        self.assertEqual(result, {"accuracy": 94.2, "f1": 0.91})

    def test_malformed_json_falls_back_to_regex_extraction(self):
        response = llm_response('Here you go:\n{"accuracy": 94.2}\nHope that helps!')
        with patch("requests.post", return_value=response):
            result = evaluation_agent._extract_metrics("text", "test")
        self.assertEqual(result, {"accuracy": 94.2})

    def test_non_numeric_values_are_dropped(self):
        response = llm_response('{"accuracy": 94.2, "notes": "high confidence"}')
        with patch("requests.post", return_value=response):
            result = evaluation_agent._extract_metrics("text", "test")
        self.assertEqual(result, {"accuracy": 94.2})

    def test_empty_object_response_returns_empty(self):
        response = llm_response('{}')
        with patch("requests.post", return_value=response):
            result = evaluation_agent._extract_metrics("vague text with no numbers", "test")
        self.assertEqual(result, {})

    def test_request_failure_returns_empty_without_raising(self):
        with patch("requests.post", side_effect=RuntimeError("network down")):
            result = evaluation_agent._extract_metrics("some text", "test")
        self.assertEqual(result, {})

    def test_non_dict_response_returns_empty(self):
        response = llm_response('[1, 2, 3]')
        with patch("requests.post", return_value=response):
            result = evaluation_agent._extract_metrics("text", "test")
        self.assertEqual(result, {})


class TestClassify(unittest.TestCase):
    def test_exact_within_tight_tolerance(self):
        classification, diff = evaluation_agent._classify(94.2, 94.5)
        self.assertEqual(classification, "EXACT")
        self.assertAlmostEqual(diff, 0.3)

    def test_approximate_within_looser_tolerance(self):
        classification, diff = evaluation_agent._classify(94.0, 97.0)
        self.assertEqual(classification, "APPROXIMATE")

    def test_failed_outside_tolerance(self):
        classification, diff = evaluation_agent._classify(94.0, 70.0)
        self.assertEqual(classification, "FAILED")

    def test_fraction_scale_is_normalized_to_percentage_points(self):
        # 0.94 vs 0.945 -- both <= 1.0, treated as fractions, scaled to
        # percentage points (94.0 vs 94.5) before comparing thresholds.
        classification, diff = evaluation_agent._classify(0.94, 0.945)
        self.assertEqual(classification, "EXACT")
        self.assertAlmostEqual(diff, 0.5)

    def test_mixed_scale_percentage_vs_fraction(self):
        # published as a percentage (94.0), reproduced as a fraction that
        # happens to be <= 1.0 is NOT rescaled unless BOTH are <= 1.0.
        classification, diff = evaluation_agent._classify(94.0, 0.94)
        self.assertEqual(classification, "FAILED")


class TestReadStdout(unittest.TestCase):
    def test_empty_path_returns_empty_string(self):
        self.assertEqual(evaluation_agent._read_stdout(""), "")

    def test_reads_file_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stdout.txt"
            path.write_text("accuracy: 0.9")
            self.assertEqual(evaluation_agent._read_stdout(str(path)), "accuracy: 0.9")

    def test_missing_file_returns_empty_without_raising(self):
        self.assertEqual(evaluation_agent._read_stdout("/nonexistent/path.txt"), "")


class TestLikelyCauses(unittest.TestCase):
    def test_reconstructed_source_adds_a_cause(self):
        causes = evaluation_agent._likely_causes(make_paper(), make_replication_result(source="reconstructed"), [])
        self.assertTrue(any("reconstructed" in c for c in causes))

    def test_inferred_assumptions_add_a_cause(self):
        causes = evaluation_agent._likely_causes(
            make_paper(), make_replication_result(inferred_assumptions=["Assumed default LR."]), [],
        )
        self.assertTrue(any("1 detail" in c for c in causes))

    def test_modifications_add_a_cause(self):
        causes = evaluation_agent._likely_causes(
            make_paper(), make_replication_result(modifications=["Copied train.csv -> repo/data/train.csv"]), [],
        )
        self.assertTrue(any("dataset wiring" in c for c in causes))

    def test_no_metrics_adds_a_cause(self):
        causes = evaluation_agent._likely_causes(make_paper(), make_replication_result(), [])
        self.assertTrue(any("No comparable metric" in c for c in causes))

    def test_clean_cloned_run_with_exact_metrics_has_no_extra_causes(self):
        metrics = [evaluation_agent._default_metric(metric="accuracy", classification="EXACT")]
        causes = evaluation_agent._likely_causes(make_paper(), make_replication_result(), metrics)
        self.assertEqual(causes, [])


class TestRun(unittest.TestCase):
    def test_matched_metric_is_classified(self):
        with patch.object(evaluation_agent, "_extract_metrics", side_effect=[{"accuracy": 94.0}, {"accuracy": 94.5}]):
            result = evaluation_agent.run(make_paper(), make_replication_result(stdout_path="x"))

        self.assertEqual(result["status"], "EXACT")
        self.assertEqual(len(result["metrics"]), 1)
        self.assertEqual(result["metrics"][0]["classification"], "EXACT")

    def test_metric_name_matching_is_case_and_punctuation_insensitive(self):
        with patch.object(evaluation_agent, "_extract_metrics",
                           side_effect=[{"Accuracy (%)": 94.0}, {"accuracy": 94.5}]):
            result = evaluation_agent.run(make_paper(), make_replication_result(stdout_path="x"))

        self.assertEqual(len(result["metrics"]), 1)
        self.assertEqual(result["metrics"][0]["published_value"], 94.0)
        self.assertEqual(result["metrics"][0]["reproduced_value"], 94.5)

    def test_published_metric_with_no_reproduced_counterpart_is_inconclusive(self):
        with patch.object(evaluation_agent, "_extract_metrics", side_effect=[{"accuracy": 94.0}, {}]):
            result = evaluation_agent.run(make_paper(), make_replication_result(stdout_path="x"))

        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["metrics"][0]["classification"], "INCONCLUSIVE")
        self.assertIsNone(result["metrics"][0]["reproduced_value"])

    def test_reproduced_metric_with_no_published_counterpart_is_inconclusive(self):
        with patch.object(evaluation_agent, "_extract_metrics", side_effect=[{}, {"accuracy": 94.0}]):
            result = evaluation_agent.run(make_paper(), make_replication_result(stdout_path="x"))

        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertIsNone(result["metrics"][0]["published_value"])

    def test_no_metrics_at_all_is_inconclusive(self):
        with patch.object(evaluation_agent, "_extract_metrics", return_value={}):
            result = evaluation_agent.run(make_paper(), make_replication_result(stdout_path="x"))

        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["metrics"], [])

    def test_overall_status_is_worst_case_among_multiple_metrics(self):
        with patch.object(evaluation_agent, "_extract_metrics",
                           side_effect=[{"accuracy": 94.0, "f1": 90.0}, {"accuracy": 94.1, "f1": 60.0}]):
            result = evaluation_agent.run(make_paper(), make_replication_result(stdout_path="x"))

        classifications = {m["metric"]: m["classification"] for m in result["metrics"]}
        self.assertEqual(classifications["accuracy"], "EXACT")
        self.assertEqual(classifications["f1"], "FAILED")
        self.assertEqual(result["status"], "FAILED")

    def test_never_raises_on_none_inputs(self):
        with patch.object(evaluation_agent, "_extract_metrics", return_value={}):
            result = evaluation_agent.run(None, None)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_published_text_combines_results_summary_and_abstract(self):
        paper = make_paper(results_summary="Accuracy 94%.", abstract="Some abstract text.")
        with patch.object(evaluation_agent, "_extract_metrics", return_value={}) as extract_mock:
            evaluation_agent.run(paper, make_replication_result())

        published_call_text = extract_mock.call_args_list[0].args[0]
        self.assertIn("Accuracy 94%.", published_call_text)
        self.assertIn("Some abstract text.", published_call_text)


if __name__ == "__main__":
    unittest.main()
