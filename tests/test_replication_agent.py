import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agents import replication_agent


def make_paper(**overrides) -> dict:
    paper = {
        "title": "Big Data-Driven Distributed ML for Fraud Detection",
        "abstract": "A distributed machine learning framework for credit card fraud detection.",
        "methodology": "Uses Logistic Regression, Decision Trees, Random Forests, XGBoost, and CatBoost.",
        "results_summary": "Focuses on scalability using distributed machine learning.",
        "dataset": "Large-scale transaction datasets",
    }
    paper.update(overrides)
    return paper


def make_repository(**overrides) -> dict:
    repo = {
        "repo_url": "https://github.com/example/fraud-detection",
        "source": "github_search",
        "dependency_files": ["requirements.txt"],
    }
    repo.update(overrides)
    return repo


def make_code_discovery_result(**overrides) -> dict:
    result = {"status": "FOUND", "repository": make_repository(), "source_stage": "github_search"}
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


class ReplicationAgentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._replication_dir_patch = patch.object(replication_agent, "REPLICATION_DIR", Path(self._tmpdir.name))
        self._replication_dir_patch.start()

    def tearDown(self):
        self._replication_dir_patch.stop()
        self._tmpdir.cleanup()


class TestRunBounded(unittest.TestCase):
    def test_success_returns_exit_code_and_output(self):
        proc = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=proc):
            result = replication_agent._run_bounded(["echo", "hi"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "ok")
        self.assertFalse(result["timed_out"])

    def test_timeout_is_reported_without_raising(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            result = replication_agent._run_bounded(["sleep", "999"], timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])

    def test_launch_failure_is_reported_without_raising(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = replication_agent._run_bounded(["git", "clone"])
        self.assertFalse(result["timed_out"])
        self.assertIsNone(result["exit_code"])
        self.assertIn("git not found", result["stderr"])

    def test_always_uses_an_argv_list_never_a_shell_string(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")) as run_mock:
            replication_agent._run_bounded(["python", "train.py"])
        args, kwargs = run_mock.call_args
        self.assertIsInstance(args[0], list)
        self.assertNotIn("shell", kwargs)


class TestCloneRepo(unittest.TestCase):
    def test_clone_failure_returns_not_ok(self):
        with patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=1, stderr="fatal")):
            result = replication_agent._clone_repo("https://github.com/a/b", Path("/tmp/x"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "fatal")

    def test_clone_success_captures_commit_sha(self):
        def fake_run(cmd, cwd=None, timeout=None):
            if cmd[:2] == ["git", "clone"]:
                return run_result(exit_code=0)
            return run_result(exit_code=0, stdout="abc123\n")

        with patch.object(replication_agent, "_run_bounded", side_effect=fake_run):
            result = replication_agent._clone_repo("https://github.com/a/b", Path("/tmp/x"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["commit_sha"], "abc123")


class TestSelectDependencyInstall(unittest.TestCase):
    def test_prefers_requirements_txt(self):
        args, reason = replication_agent._select_dependency_install(["requirements.txt", "setup.py"])
        self.assertEqual(args, ["install", "-r", "requirements.txt"])
        self.assertEqual(reason, "")

    def test_falls_back_to_pip_install_dot_for_pyproject(self):
        args, reason = replication_agent._select_dependency_install(["pyproject.toml"])
        self.assertEqual(args, ["install", "."])

    def test_conda_only_is_blocked(self):
        args, reason = replication_agent._select_dependency_install(["environment.yml"])
        self.assertIsNone(args)
        self.assertIn("conda", reason)

    def test_no_recognized_file_is_blocked(self):
        args, reason = replication_agent._select_dependency_install([])
        self.assertIsNone(args)
        self.assertIn("No recognized dependency file", reason)


class TestEntryPointDetection(ReplicationAgentTestCase):
    def test_convention_filename_is_detected(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        (repo_dir / "train.py").write_text("print('hi')")
        self.assertEqual(replication_agent._detect_entry_point_by_convention(repo_dir), "train.py")

    def test_no_convention_filename_returns_empty(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        self.assertEqual(replication_agent._detect_entry_point_by_convention(repo_dir), "")

    def test_readme_llm_extracts_documented_command(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("Run with: python src/train.py --epochs 10")
        response = llm_response('{"command": ["python", "src/train.py", "--epochs", "10"]}')

        with patch("requests.post", return_value=response):
            command = replication_agent._extract_entry_point_from_readme(repo_dir, "Some Paper")

        self.assertEqual(command, ["python", "src/train.py", "--epochs", "10"])

    def test_readme_llm_returns_empty_when_nothing_documented(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("A repo with no run instructions.")
        response = llm_response('{"command": null}')

        with patch("requests.post", return_value=response):
            command = replication_agent._extract_entry_point_from_readme(repo_dir, "Some Paper")

        self.assertEqual(command, [])

    def test_no_readme_returns_empty_without_llm_call(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        with patch("requests.post") as post_mock:
            command = replication_agent._extract_entry_point_from_readme(repo_dir, "Some Paper")
        post_mock.assert_not_called()
        self.assertEqual(command, [])

    def test_malformed_llm_json_falls_back_to_regex_extraction(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("Run with: python train.py")
        response = llm_response('Sure thing:\n{"command": ["python", "train.py"]}\nLet me know if that helps.')

        with patch("requests.post", return_value=response):
            command = replication_agent._extract_entry_point_from_readme(repo_dir, "Some Paper")

        self.assertEqual(command, ["python", "train.py"])


class TestWireDatasetIntoRepo(ReplicationAgentTestCase):
    def test_copies_csvs_when_data_dir_exists(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        (repo_dir / "data").mkdir(parents=True)
        train_csv = Path(self._tmpdir.name) / "train.csv"
        train_csv.write_text("a,b\n1,2\n")

        modifications = replication_agent._wire_dataset_into_repo(repo_dir, {"train_csv_path": str(train_csv)})

        self.assertTrue((repo_dir / "data" / "train.csv").exists())
        self.assertEqual(len(modifications), 1)

    def test_no_op_when_no_data_dir(self):
        repo_dir = Path(self._tmpdir.name) / "repo"
        repo_dir.mkdir()
        modifications = replication_agent._wire_dataset_into_repo(
            repo_dir, {"train_csv_path": str(Path(self._tmpdir.name) / "train.csv")}
        )
        self.assertEqual(modifications, [])


class TestReplicateFromRepository(ReplicationAgentTestCase):
    def test_clone_failure_returns_blocked(self):
        with patch.object(replication_agent, "_clone_repo", return_value={"ok": False, "error": "fatal", "commit_sha": ""}):
            result = replication_agent._replicate_from_repository(make_paper(), make_repository(), make_cleaning_result())

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["source"], "cloned_repository")
        self.assertIn("Clone failed", result["warnings"][0])

    def test_venv_failure_returns_blocked(self):
        with patch.object(replication_agent, "_clone_repo",
                           return_value={"ok": True, "error": "", "commit_sha": "abc"}), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=1, stderr="no venv")):
            result = replication_agent._replicate_from_repository(make_paper(), make_repository(), make_cleaning_result())

        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("Virtualenv creation failed", result["warnings"][0])

    def test_conda_only_dependency_file_blocks_without_execution(self):
        with patch.object(replication_agent, "_clone_repo",
                           return_value={"ok": True, "error": "", "commit_sha": "abc"}), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_run_bounded") as run_mock:
            result = replication_agent._replicate_from_repository(
                make_paper(), make_repository(dependency_files=["environment.yml"]), make_cleaning_result(),
            )

        self.assertEqual(result["status"], "BLOCKED")
        run_mock.assert_not_called()

    def test_successful_run_with_convention_entry_point(self):
        repo_dir = replication_agent.REPLICATION_DIR / replication_agent._slugify(make_paper()["title"]) / "repo"

        def fake_run_bounded(cmd, cwd=None, timeout=None):
            # _create_venv and _pip_install are mocked separately below, so
            # the only real _run_bounded calls left are git clone/rev-parse
            # and the final execution of the cloned repo's entry point.
            if cmd[:2] == ["git", "clone"]:
                repo_dir.mkdir(parents=True, exist_ok=True)
                (repo_dir / "train.py").write_text("print('training')")
                return run_result(exit_code=0)
            if cmd[:2] == ["git", "rev-parse"]:
                return run_result(exit_code=0, stdout="abc123\n")
            return run_result(exit_code=0, stdout="accuracy: 0.9")

        with patch.object(replication_agent, "_run_bounded", side_effect=fake_run_bounded), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value="pandas==2.0.0"):
            result = replication_agent._replicate_from_repository(make_paper(), make_repository(), make_cleaning_result())

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["entry_point"], "train.py")
        self.assertEqual(result["entry_point_source"], "convention")
        self.assertEqual(result["commit_sha"], "abc123")

    def test_no_entry_point_found_blocks_without_executing(self):
        repo_dir = replication_agent.REPLICATION_DIR / replication_agent._slugify(make_paper()["title"]) / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(replication_agent, "_clone_repo",
                           return_value={"ok": True, "error": "", "commit_sha": "abc"}), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_extract_entry_point_from_readme", return_value=[]), \
             patch.object(replication_agent, "_run_bounded") as run_mock:
            result = replication_agent._replicate_from_repository(make_paper(), make_repository(), make_cleaning_result())

        self.assertEqual(result["status"], "BLOCKED")
        run_mock.assert_not_called()

    def test_timeout_during_execution_is_reported(self):
        with patch.object(replication_agent, "_clone_repo",
                           return_value={"ok": True, "error": "", "commit_sha": "abc"}), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_detect_entry_point_by_convention", return_value="train.py"), \
             patch.object(replication_agent, "_wire_dataset_into_repo", return_value=[]), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(timed_out=True)):
            result = replication_agent._replicate_from_repository(make_paper(), make_repository(), make_cleaning_result())

        self.assertEqual(result["status"], "TIMEOUT")


class TestReplicateByReconstruction(ReplicationAgentTestCase):
    def test_generation_failure_returns_blocked(self):
        with patch.object(replication_agent, "_generate_baseline_script", return_value={}):
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["source"], "reconstructed")

    def test_non_allowlisted_libraries_are_dropped(self):
        generated = {
            "code": "print('train')",
            "inferred_assumptions": ["Used default hyperparameters."],
            "libraries_required": ["pandas", "some-sketchy-package"],
        }
        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)) as pip_mock, \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="done")):
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        pip_mock.assert_called_once_with(unittest.mock.ANY, unittest.mock.ANY, ["install", "pandas"])
        self.assertTrue(any("some-sketchy-package" in w for w in result["warnings"]))
        self.assertEqual(result["status"], "OK")

    def test_repair_loop_succeeds_on_second_attempt(self):
        generated = {"code": "broken code", "inferred_assumptions": [], "libraries_required": []}
        attempts = [run_result(exit_code=1, stderr="SyntaxError"), run_result(exit_code=0, stdout="accuracy: 0.8")]

        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", side_effect=attempts), \
             patch.object(replication_agent, "_repair_generated_script", return_value="fixed code"):
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        self.assertEqual(result["status"], "OK")
        self.assertIn("Repair attempts: 2", result["notes"])

    def test_gives_up_after_max_repair_attempts(self):
        generated = {"code": "broken code", "inferred_assumptions": [], "libraries_required": []}

        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=1, stderr="still broken")), \
             patch.object(replication_agent, "_repair_generated_script", return_value="still broken code") as repair_mock:
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(repair_mock.call_count, replication_agent.MAX_REPAIR_ATTEMPTS - 1)

    def test_inferred_assumptions_are_carried_through(self):
        generated = {
            "code": "print('ok')", "inferred_assumptions": ["Assumed a 70/20/10 split.", "Assumed default LR."],
            "libraries_required": [],
        }
        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="ok")):
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        self.assertEqual(result["inferred_assumptions"], generated["inferred_assumptions"])


class TestRunDispatch(unittest.TestCase):
    def test_dispatches_to_repository_path_when_found(self):
        with patch.object(replication_agent, "_replicate_from_repository", return_value={"status": "OK"}) as repo_fn, \
             patch.object(replication_agent, "_replicate_by_reconstruction") as recon_fn:
            replication_agent.run(make_paper(), make_code_discovery_result(), make_cleaning_result())

        repo_fn.assert_called_once()
        recon_fn.assert_not_called()

    def test_dispatches_to_reconstruction_path_when_not_found(self):
        with patch.object(replication_agent, "_replicate_from_repository") as repo_fn, \
             patch.object(replication_agent, "_replicate_by_reconstruction", return_value={"status": "OK"}) as recon_fn:
            replication_agent.run(make_paper(), make_code_discovery_result(status="NOT_FOUND", repository=None), make_cleaning_result())

        recon_fn.assert_called_once()
        repo_fn.assert_not_called()

    def test_dispatches_to_reconstruction_path_when_code_discovery_result_is_none(self):
        with patch.object(replication_agent, "_replicate_from_repository") as repo_fn, \
             patch.object(replication_agent, "_replicate_by_reconstruction", return_value={"status": "OK"}) as recon_fn:
            replication_agent.run(make_paper(), None, make_cleaning_result())

        recon_fn.assert_called_once()
        repo_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
