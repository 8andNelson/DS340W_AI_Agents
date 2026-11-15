"""The published-score leakage check: a paper whose near-perfect score is
reached only when the data is oversampled before the train/test split.
The Evaluation Agent reports "the published result may come from leakage"
and the Experiment Agent stops chasing that score."""
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agents import evaluation_agent, experiment_agent
from src.orchestration import leakage_probe
from tests.test_experiment_agent import answer, evaluation, metric, run as run_experiment

# Run 20261115T171901Z-86734b16, in points: the paper's Table 5 row, and the probe on the run's data.
PUBLISHED = {"accuracy": 99.26, "precision": 99.53, "recall": 100.0, "f1": 99.26, "auc": 99.26}
REPRODUCED = {"accuracy": 94.0, "precision": 50.0, "recall": 5.41, "f1": 9.76, "auc": 77.34}
PROBE = {"rows": 15420, "positive_rate": 0.06, "features": 171,
         "honest": {"accuracy": 0.9414, "precision": 0.75, "recall": 0.0325, "f1": 0.0623, "auc": 0.8171},
         "leaky": {"accuracy": 0.9975, "precision": 0.995, "recall": 1.0, "f1": 0.9975, "auc": 1.0}}
PAPER_TEXT = "Three data resampling strategies (Undersampling, Oversampling and SMOTE) address class imbalance."


def test_scores_reached_only_with_leakage_are_consistent_with_leakage():
    result = leakage_probe.compare(PUBLISHED, PROBE)
    assert result["status"] == "CONSISTENT_WITH_LEAKAGE"
    explained = {r["metric"]: r["explained_by_leakage"] for r in result["metrics"]}
    # accuracy is high either way on 6% fraud, so it is not evidence of leakage
    assert explained == {"accuracy": False, "precision": True, "recall": True, "f1": True, "auc": True}


def test_scores_reachable_with_a_correct_split_are_not_leakage():
    published = {"accuracy": 94.5, "precision": 70.0, "recall": 6.0, "f1": 8.0, "auc": 82.0}
    assert leakage_probe.compare(published, PROBE)["status"] == "NOT_CONSISTENT"


def test_a_paper_far_from_both_probes_is_not_leakage():
    published = {"f1": 60.0, "recall": 55.0}  # far above honest, far below leaky
    assert leakage_probe.compare(published, PROBE)["status"] == "NOT_CONSISTENT"


def test_the_probe_runs_only_when_it_can_matter(tmp_path):
    replication, cleaning = {"replication_dir": str(tmp_path)}, {"master_csv_path": "m.csv", "label_column": "y"}
    small_gap = {k: v - 2 for k, v in PUBLISHED.items()}
    with patch.object(leakage_probe, "run_probe") as probe:
        assert leakage_probe.check(PUBLISHED, small_gap, PAPER_TEXT, replication, cleaning)["status"] == "SKIPPED"
        assert leakage_probe.check(PUBLISHED, REPRODUCED, "A random forest was trained.", replication,
                                   cleaning)["status"] == "SKIPPED"
        unavailable = leakage_probe.check(PUBLISHED, REPRODUCED, PAPER_TEXT, replication, cleaning)
    probe.assert_not_called()
    assert unavailable["status"] == "UNAVAILABLE" and "replication environment" in unavailable["reason"]


def _venv(tmp_path):
    python = tmp_path / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
    return {"replication_dir": str(tmp_path)}


def test_check_runs_the_probe_in_the_replication_venv_without_secrets(tmp_path, monkeypatch):
    csv = tmp_path / "master.csv"
    csv.write_text("a,y\n1,0\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    leakage_probe._CACHE.clear()
    done = SimpleNamespace(returncode=0, stdout=json.dumps(PROBE) + "\n", stderr="")
    with patch.object(leakage_probe.subprocess, "run", return_value=done) as call:
        result = leakage_probe.check(PUBLISHED, REPRODUCED, PAPER_TEXT, _venv(tmp_path),
                                     {"master_csv_path": str(csv), "label_column": "y"})
        leakage_probe.check(PUBLISHED, REPRODUCED, PAPER_TEXT, _venv_again(tmp_path),
                            {"master_csv_path": str(csv), "label_column": "y"})
    assert result["status"] == "CONSISTENT_WITH_LEAKAGE" and result["probe"]["rows"] == 15420
    assert "resampling" in result["resampling_quote"].lower()
    argv = call.call_args.args[0]
    assert argv[1] == "-I" and argv[2].endswith("leakage_probe_script.py") and argv[3:] == [str(csv), "y", "50000"]
    assert "OPENROUTER_API_KEY" not in call.call_args.kwargs["env"]
    assert call.call_count == 1  # the second Evaluation Agent run uses the cached probe


def _venv_again(tmp_path):
    return {"replication_dir": str(tmp_path)}


def test_a_failed_probe_is_unavailable_not_a_verdict(tmp_path):
    csv = tmp_path / "master.csv"
    csv.write_text("a,y\n1,0\n")
    leakage_probe._CACHE.clear()
    failed = SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'sklearn'\n")
    with patch.object(leakage_probe.subprocess, "run", return_value=failed):
        result = leakage_probe.check(PUBLISHED, REPRODUCED, PAPER_TEXT, _venv(tmp_path),
                                     {"master_csv_path": str(csv), "label_column": "y"})
    assert result["status"] == "UNAVAILABLE" and "sklearn" in result["reason"]
    with patch.object(leakage_probe.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 600)):
        result = leakage_probe.check(PUBLISHED, REPRODUCED, PAPER_TEXT, _venv_again(tmp_path),
                                     {"master_csv_path": str(csv), "label_column": "y"})
    assert result["status"] == "UNAVAILABLE" and "did not finish" in result["reason"]


def test_the_probe_script_separates_honest_and_leaky_scores(tmp_path):
    pytest.importorskip("sklearn")  # the project's .venv has no scikit-learn; a replication venv does
    import numpy as np
    rng = np.random.default_rng(0)
    rows = ["noise,kind,label"] + [f"{rng.normal():.4f},{rng.choice(['a', 'b'])},{int(rng.random() < 0.06)}"
                                   for _ in range(1500)]
    csv = tmp_path / "master.csv"
    csv.write_text("\n".join(rows))
    out = subprocess.run([sys.executable, "-I", str(leakage_probe.PROBE_SCRIPT), str(csv), "label", "50000"],
                         capture_output=True, text=True, timeout=300)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["leaky"]["recall"] > 0.9 > result["honest"]["recall"]  # pure noise: only the leak finds fraud


def test_the_evaluation_agent_names_the_leakage_as_a_likely_cause():
    verdict = {**leakage_probe.compare(PUBLISHED, PROBE), "probe": {"rows": 15420}}
    with patch.object(evaluation_agent, "_extract_metrics", side_effect=[
            {k: v / 100 for k, v in PUBLISHED.items()}, {k: v / 100 for k, v in REPRODUCED.items()}]), \
            patch.object(evaluation_agent.leakage_probe, "check", return_value=verdict) as check:
        result = evaluation_agent.run({"title": "T", "official_abstract": "AUC 99.26%"}, {"stdout_path": "x"})
    assert result["published_leakage_check"]["status"] == "CONSISTENT_WITH_LEAKAGE"
    assert any(c.startswith("The published result may come from leakage") for c in result["likely_causes"])
    published_points = check.call_args.args[0]
    assert published_points["f1"] == pytest.approx(99.26)


def test_the_evaluation_agent_never_probes_values_from_a_search_snippet():
    with patch.object(evaluation_agent, "_extract_metrics", side_effect=[{"f1": 0.99}, {"f1": 0.1}]), \
            patch.object(evaluation_agent.leakage_probe, "check") as check:
        result = evaluation_agent.run({"title": "T", "results_summary": "F1 0.99"}, {"stdout_path": "x"})
    check.assert_not_called()
    assert result["published_leakage_check"]["status"] == "SKIPPED"


def _leaky_evaluation():
    result = evaluation([metric(published=0.9926, reproduced=0.0976, diff=89.5, classification="FAILED", name="f1")])
    result["published_leakage_check"] = leakage_probe.compare(PUBLISHED, PROBE)
    return result


def test_the_experiment_agent_stops_chasing_a_leaky_published_score(tmp_path):
    result, chat = run_experiment(tmp_path, answer(), _leaky_evaluation())
    assert result["requests"] == []
    assert result["decision"]["action"] == "stop"
    assert any("may come from leakage" in n for n in result["decision_notes"])
    assert "WARNING -- the published result may come from leakage" in chat.call_args.args[0]


def test_the_experiment_agent_still_works_on_a_clean_gap(tmp_path):
    clean = evaluation()
    clean["published_leakage_check"] = {"status": "NOT_CONSISTENT", "reason": "", "metrics": []}
    result, chat = run_experiment(tmp_path, answer(), clean)
    assert result["decision"]["action"] == "rerun_agent"
    assert "WARNING" not in chat.call_args.args[0]
