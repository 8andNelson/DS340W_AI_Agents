import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from src.agents import experiment_agent


def make_paper(**overrides) -> dict:
    paper = {
        "title": "Big Data-Driven Distributed ML for Fraud Detection",
        "methodology": "Uses Logistic Regression, Decision Trees, Random Forests, XGBoost, and CatBoost.",
    }
    paper.update(overrides)
    return paper


def make_replication_result(**overrides) -> dict:
    result = {
        "status": "OK", "source": "cloned_repository", "entry_point": "train.py",
        "command": ["C:/venv/Scripts/python.exe", "train.py"], "inferred_assumptions": [],
    }
    result.update(overrides)
    return result


def make_evaluation_result(**overrides) -> dict:
    result = {
        "metrics": [
            {"metric": "accuracy", "published_value": 94.2, "reproduced_value": 92.0, "classification": "APPROXIMATE"},
        ],
    }
    result.update(overrides)
    return result


def make_cleaning_result(**overrides) -> dict:
    result = {
        "master_csv_path": "data/processed/master_dataset.csv",
        "train_csv_path": "data/processed/train.csv",
        "test_csv_path": "data/processed/test.csv",
        "validation_csv_path": "data/processed/validation.csv",
    }
    result.update(overrides)
    return result


def run_result(exit_code=0, stdout="", stderr="", timed_out=False, duration_seconds=1.0) -> dict:
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
            "duration_seconds": duration_seconds, "timed_out": timed_out}


def llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    response.raise_for_status = lambda: None
    return response


class ExperimentAgentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._replication_dir_patch = patch.object(experiment_agent, "REPLICATION_DIR", Path(self._tmpdir.name) / "replication")
        self._experiments_dir_patch = patch.object(experiment_agent, "EXPERIMENTS_DIR", Path(self._tmpdir.name) / "experiments")
        self._replication_dir_patch.start()
        self._experiments_dir_patch.start()

    def tearDown(self):
        self._replication_dir_patch.stop()
        self._experiments_dir_patch.stop()
        self._tmpdir.cleanup()


class TestNextCategory(unittest.TestCase):
    def test_reconstructed_picks_first_untried_code_level_category(self):
        category = experiment_agent._next_category("reconstructed", [])
        self.assertEqual(category, experiment_agent.CODE_LEVEL_CATEGORIES[0])

    def test_reconstructed_skips_already_tried_categories_in_order(self):
        previous = [{"category": experiment_agent.CODE_LEVEL_CATEGORIES[0]}]
        category = experiment_agent._next_category("reconstructed", previous)
        self.assertEqual(category, experiment_agent.CODE_LEVEL_CATEGORIES[1])

    def test_reconstructed_falls_back_to_dataset_split_after_all_code_level_tried(self):
        previous = [{"category": c} for c in experiment_agent.CODE_LEVEL_CATEGORIES]
        category = experiment_agent._next_category("reconstructed", previous)
        self.assertEqual(category, experiment_agent.DATASET_SPLIT_CATEGORY)

    def test_reconstructed_returns_empty_when_everything_exhausted(self):
        previous = [{"category": c} for c in experiment_agent.CODE_LEVEL_CATEGORIES]
        previous.append({"category": experiment_agent.DATASET_SPLIT_CATEGORY})
        category = experiment_agent._next_category("reconstructed", previous)
        self.assertEqual(category, "")

    def test_cloned_repository_only_ever_offers_dataset_split(self):
        category = experiment_agent._next_category("cloned_repository", [])
        self.assertEqual(category, experiment_agent.DATASET_SPLIT_CATEGORY)

    def test_cloned_repository_returns_empty_once_dataset_split_tried(self):
        previous = [{"category": experiment_agent.DATASET_SPLIT_CATEGORY}]
        category = experiment_agent._next_category("cloned_repository", previous)
        self.assertEqual(category, "")


class TestBaselineMetricsFromEvaluation(unittest.TestCase):
    def test_extracts_reproduced_values_only(self):
        evaluation_result = {
            "metrics": [
                {"metric": "accuracy", "reproduced_value": 92.0},
                {"metric": "f1", "reproduced_value": None, "notes": "not printed"},
            ],
        }
        self.assertEqual(experiment_agent._baseline_metrics_from_evaluation(evaluation_result), {"accuracy": 92.0})

    def test_none_input_returns_empty(self):
        self.assertEqual(experiment_agent._baseline_metrics_from_evaluation(None), {})


class TestClassifyChange(unittest.TestCase):
    def test_improved_above_threshold(self):
        classification, diff = experiment_agent._classify_change(90.0, 93.0)
        self.assertEqual(classification, "IMPROVED")

    def test_regressed_below_threshold(self):
        classification, diff = experiment_agent._classify_change(90.0, 85.0)
        self.assertEqual(classification, "REGRESSED")

    def test_neutral_within_threshold(self):
        classification, diff = experiment_agent._classify_change(90.0, 90.5)
        self.assertEqual(classification, "NEUTRAL")

    def test_fraction_scale_is_normalized(self):
        classification, diff = experiment_agent._classify_change(0.90, 0.95)
        self.assertEqual(classification, "IMPROVED")
        self.assertAlmostEqual(diff, 5.0)


class TestCompareToBaseline(unittest.TestCase):
    def test_matched_metric_classified(self):
        comparison = experiment_agent._compare_to_baseline({"accuracy": 90.0}, {"accuracy": 93.0})
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison[0]["classification"], "IMPROVED")

    def test_baseline_only_metric_is_inconclusive(self):
        comparison = experiment_agent._compare_to_baseline({"accuracy": 90.0}, {})
        self.assertEqual(comparison[0]["classification"], "INCONCLUSIVE")
        self.assertIsNone(comparison[0]["experiment_value"])

    def test_experiment_only_metric_is_inconclusive(self):
        comparison = experiment_agent._compare_to_baseline({}, {"accuracy": 90.0})
        self.assertEqual(comparison[0]["classification"], "INCONCLUSIVE")
        self.assertIsNone(comparison[0]["baseline_value"])


class TestOverallFromComparison(unittest.TestCase):
    def test_empty_comparison_is_inconclusive(self):
        self.assertEqual(experiment_agent._overall_from_comparison([]), "INCONCLUSIVE")

    def test_regressed_wins_over_improved(self):
        comparison = [{"classification": "IMPROVED"}, {"classification": "REGRESSED"}]
        self.assertEqual(experiment_agent._overall_from_comparison(comparison), "REGRESSED")

    def test_neutral_wins_over_improved_alone(self):
        comparison = [{"classification": "IMPROVED"}, {"classification": "NEUTRAL"}]
        self.assertEqual(experiment_agent._overall_from_comparison(comparison), "NEUTRAL")


class TestResplitDataset(ExperimentAgentTestCase):
    def test_writes_new_split_csvs(self):
        with tempfile.TemporaryDirectory() as data_tmp:
            master_path = Path(data_tmp) / "master.csv"
            pd.DataFrame({"id": range(100), "label": [0, 1] * 50}).to_csv(master_path, index=False)
            experiment_dir = Path(self._tmpdir.name) / "exp_001"
            experiment_dir.mkdir(parents=True)

            new_split = experiment_agent._resplit_dataset({"master_csv_path": str(master_path)}, experiment_dir)

        self.assertTrue(Path(new_split["train_csv_path"]).exists())
        self.assertTrue(Path(new_split["test_csv_path"]).exists())
        self.assertTrue(Path(new_split["validation_csv_path"]).exists())


class TestRewriteHardcodedCsvPaths(unittest.TestCase):
    def test_replaces_original_paths_with_new_ones(self):
        script = 'df = pd.read_csv("data/processed/train.csv")\ntest = pd.read_csv("data/processed/test.csv")'
        cleaning_result = {"train_csv_path": "data/processed/train.csv", "test_csv_path": "data/processed/test.csv"}
        new_split = {"train_csv_path": "experiments/x/data/train.csv", "test_csv_path": "experiments/x/data/test.csv",
                     "validation_csv_path": ""}

        result = experiment_agent._rewrite_hardcoded_csv_paths(script, cleaning_result, new_split)

        self.assertIn("experiments/x/data/train.csv", result)
        self.assertIn("experiments/x/data/test.csv", result)
        self.assertNotIn("data/processed/train.csv", result)


class TestProposeCodeChange(unittest.TestCase):
    def test_parses_valid_json_response(self):
        response = llm_response(
            '{"hypothesis": "XGBoost should outperform RF.", "what_changed": "Swapped model.", '
            '"code": "print(1)"}'
        )
        with patch("requests.post", return_value=response):
            result = experiment_agent._propose_code_change("alternative_model", make_paper(), "print(0)", [])
        self.assertEqual(result["what_changed"], "Swapped model.")

    def test_malformed_json_falls_back_to_regex_extraction(self):
        response = llm_response('Sure:\n{"hypothesis": "h", "what_changed": "c", "code": "print(1)"}\nDone.')
        with patch("requests.post", return_value=response):
            result = experiment_agent._propose_code_change("alternative_model", make_paper(), "print(0)", [])
        self.assertEqual(result["code"], "print(1)")

    def test_request_failure_returns_empty(self):
        with patch("requests.post", side_effect=RuntimeError("down")):
            result = experiment_agent._propose_code_change("alternative_model", make_paper(), "print(0)", [])
        self.assertEqual(result, {})

    def test_missing_code_field_returns_empty(self):
        response = llm_response('{"hypothesis": "h", "what_changed": "c"}')
        with patch("requests.post", return_value=response):
            result = experiment_agent._propose_code_change("alternative_model", make_paper(), "print(0)", [])
        self.assertEqual(result, {})


class TestRun(ExperimentAgentTestCase):
    def test_no_category_available_returns_skipped(self):
        previous = [{"category": c} for c in experiment_agent.CODE_LEVEL_CATEGORIES]
        previous.append({"category": experiment_agent.DATASET_SPLIT_CATEGORY})

        result = experiment_agent.run(
            make_paper(), make_replication_result(source="reconstructed"), make_evaluation_result(),
            make_cleaning_result(), previous,
        )

        self.assertEqual(result["status"], "SKIPPED")

    def test_cloned_repo_dataset_split_runs_against_a_fresh_copy(self):
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "repo").mkdir(parents=True)
        (baseline_dir / "repo" / "train.py").write_text("print('baseline')")
        (baseline_dir / "venv").mkdir(parents=True)

        with tempfile.TemporaryDirectory() as data_tmp:
            master_path = Path(data_tmp) / "master.csv"
            pd.DataFrame({"id": range(100), "label": [0, 1] * 50}).to_csv(master_path, index=False)
            cleaning_result = make_cleaning_result(master_csv_path=str(master_path))

            with patch.object(experiment_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="accuracy: 93.0")):
                result = experiment_agent.run(
                    make_paper(), make_replication_result(source="cloned_repository", command=["python", "train.py"]),
                    make_evaluation_result(), cleaning_result, [],
                )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["category"], experiment_agent.DATASET_SPLIT_CATEGORY)
        # The baseline repo directory must be untouched -- a fresh copy was made.
        self.assertTrue((baseline_dir / "repo" / "train.py").is_file())
        self.assertTrue(Path(result["experiment_dir"], "repo", "train.py").is_file())

    def test_reconstructed_dataset_split_rewrites_hardcoded_paths(self):
        """previous_experiments must already exhaust every code-level
        category so _next_category falls through to dataset_split --
        otherwise run() takes the LLM-proposal branch instead, which this
        test doesn't mock."""
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "reconstructed").mkdir(parents=True)
        (baseline_dir / "venv").mkdir(parents=True)
        previous = [{"category": c} for c in experiment_agent.CODE_LEVEL_CATEGORIES]

        with tempfile.TemporaryDirectory() as data_tmp:
            master_path = Path(data_tmp) / "master.csv"
            train_path = Path(data_tmp) / "train.csv"
            pd.DataFrame({"id": range(100), "label": [0, 1] * 50}).to_csv(master_path, index=False)
            cleaning_result = make_cleaning_result(master_csv_path=str(master_path), train_csv_path=str(train_path))
            (baseline_dir / "reconstructed" / "train.py").write_text(f'pd.read_csv("{train_path}")')

            with patch.object(experiment_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="accuracy: 93.0")):
                result = experiment_agent.run(
                    make_paper(), make_replication_result(source="reconstructed"),
                    make_evaluation_result(), cleaning_result, previous,
                )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["category"], experiment_agent.DATASET_SPLIT_CATEGORY)
        experiment_script = Path(result["experiment_dir"]) / "train.py"
        self.assertNotIn(str(train_path), experiment_script.read_text())

    def test_code_level_experiment_generation_failure_returns_blocked(self):
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "reconstructed").mkdir(parents=True)
        (baseline_dir / "reconstructed" / "train.py").write_text("print('baseline')")
        (baseline_dir / "venv").mkdir(parents=True)

        with patch.object(experiment_agent, "_propose_code_change", return_value={}):
            result = experiment_agent.run(
                make_paper(), make_replication_result(source="reconstructed"),
                make_evaluation_result(), make_cleaning_result(), [],
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["category"], experiment_agent.CODE_LEVEL_CATEGORIES[0])

    def test_code_level_experiment_success_computes_comparison(self):
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "reconstructed").mkdir(parents=True)
        (baseline_dir / "reconstructed" / "train.py").write_text("print('baseline')")
        (baseline_dir / "venv").mkdir(parents=True)

        proposal = {"hypothesis": "XGBoost should do better.", "what_changed": "Swapped RF for XGBoost.", "code": "print(1)"}

        with patch.object(experiment_agent, "_propose_code_change", return_value=proposal), \
             patch.object(experiment_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="accuracy: 95.0")), \
             patch.object(experiment_agent, "_extract_metrics", return_value={"accuracy": 95.0}):
            result = experiment_agent.run(
                make_paper(), make_replication_result(source="reconstructed"),
                make_evaluation_result(), make_cleaning_result(), [],
            )

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["overall"], "IMPROVED")
        self.assertEqual(result["baseline_metrics"], {"accuracy": 92.0})
        self.assertEqual(result["experiment_metrics"], {"accuracy": 95.0})

    def test_experiment_script_failure_is_blocked_without_metric_extraction(self):
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "reconstructed").mkdir(parents=True)
        (baseline_dir / "reconstructed" / "train.py").write_text("print('baseline')")
        (baseline_dir / "venv").mkdir(parents=True)

        proposal = {"hypothesis": "h", "what_changed": "c", "code": "raise ValueError()"}

        with patch.object(experiment_agent, "_propose_code_change", return_value=proposal), \
             patch.object(experiment_agent, "_run_bounded", return_value=run_result(exit_code=1, stderr="ValueError")), \
             patch.object(experiment_agent, "_extract_metrics") as extract_mock:
            result = experiment_agent.run(
                make_paper(), make_replication_result(source="reconstructed"),
                make_evaluation_result(), make_cleaning_result(), [],
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["overall"], "INCONCLUSIVE")
        extract_mock.assert_not_called()

    def test_timeout_is_reported(self):
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "reconstructed").mkdir(parents=True)
        (baseline_dir / "reconstructed" / "train.py").write_text("print('baseline')")
        (baseline_dir / "venv").mkdir(parents=True)

        proposal = {"hypothesis": "h", "what_changed": "c", "code": "import time; time.sleep(999)"}

        with patch.object(experiment_agent, "_propose_code_change", return_value=proposal), \
             patch.object(experiment_agent, "_run_bounded", return_value=run_result(timed_out=True)):
            result = experiment_agent.run(
                make_paper(), make_replication_result(source="reconstructed"),
                make_evaluation_result(), make_cleaning_result(), [],
            )

        self.assertEqual(result["status"], "TIMEOUT")

    def test_experiment_numbering_follows_previous_experiments_count(self):
        slug = experiment_agent._slugify(make_paper()["title"])
        baseline_dir = experiment_agent.REPLICATION_DIR / slug
        (baseline_dir / "reconstructed").mkdir(parents=True)
        (baseline_dir / "reconstructed" / "train.py").write_text("print('baseline')")
        (baseline_dir / "venv").mkdir(parents=True)

        previous = [{"category": experiment_agent.CODE_LEVEL_CATEGORIES[0]}]
        proposal = {"hypothesis": "h", "what_changed": "c", "code": "print(1)"}

        with patch.object(experiment_agent, "_propose_code_change", return_value=proposal), \
             patch.object(experiment_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="ok")):
            result = experiment_agent.run(
                make_paper(), make_replication_result(source="reconstructed"),
                make_evaluation_result(), make_cleaning_result(), previous,
            )

        self.assertIn("exp_002", result["experiment_dir"])


if __name__ == "__main__":
    unittest.main()
