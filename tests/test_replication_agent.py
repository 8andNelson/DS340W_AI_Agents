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
             patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="METRIC accuracy=0.9")):
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        pip_mock.assert_called_once_with(unittest.mock.ANY, unittest.mock.ANY,
                                         ["install", "numpy", "pandas", "scikit-learn"])
        self.assertTrue(any("some-sketchy-package" in w for w in result["warnings"]))
        self.assertEqual(result["status"], "OK")

    def test_repair_loop_succeeds_on_second_attempt(self):
        generated = {"code": "broken code", "inferred_assumptions": [], "libraries_required": []}
        attempts = [run_result(exit_code=1, stderr="SyntaxError"), run_result(exit_code=0, stdout="METRIC accuracy=0.8")]

        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)),              patch.object(replication_agent, "_run_bounded", side_effect=attempts), \
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
             patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="METRIC f1=0.7")):
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


class TestReconstructionLiveRunFixes(ReplicationAgentTestCase):
    def test_exit_zero_without_metric_lines_is_repaired_then_blocked(self):
        generated = {"code": "print('trained')", "inferred_assumptions": [], "libraries_required": []}
        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(exit_code=0, stdout="trained")), \
             patch.object(replication_agent, "_repair_generated_script", return_value="print('again')") as repair_mock:
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("METRIC", repair_mock.call_args.args[1])
        self.assertTrue(any("printed no" in w for w in result["warnings"]))

    def test_script_gets_data_paths_through_env(self):
        generated = {"code": "print(1)", "inferred_assumptions": [], "libraries_required": []}
        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded",
                          return_value=run_result(exit_code=0, stdout="METRIC accuracy=0.9")) as run_mock:
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())

        self.assertEqual(result["status"], "OK")
        self.assertEqual(run_mock.call_args.kwargs["env"], {
            "TRAIN_CSV": "data/processed/train.csv", "TEST_CSV": "data/processed/test.csv",
            "VALIDATION_CSV": "data/processed/validation.csv",
        })
        attempts_dir = Path(result["replication_dir"]) / "reconstructed" / "attempts"
        self.assertTrue((attempts_dir / "attempt_1.py").is_file())

    def test_install_falls_back_to_one_library_at_a_time(self):
        def pip(venv, cwd, args):
            return run_result(exit_code=1 if "tensorflow" in args else 0)

        with patch.object(replication_agent, "_pip_install", side_effect=pip):
            installed, failed = replication_agent._install_libraries(Path("v"), Path("c"), ["pandas", "tensorflow"])
        self.assertEqual((installed, failed), (["pandas"], ["tensorflow"]))

    def test_generation_parses_json_and_python_blocks(self):
        content = ('```json\n{"inferred_assumptions": ["Used 100 trees."], "libraries_required": ["xgboost"]}\n```\n'
                   '```python\nimport os\nprint(os.environ["TRAIN_CSV"])\n```')
        with patch("requests.post", return_value=llm_response(content)):
            generated = replication_agent._generate_baseline_script(make_paper(), make_cleaning_result(), "profile")
        self.assertEqual(generated["inferred_assumptions"], ["Used 100 trees."])
        self.assertEqual(generated["libraries_required"], ["xgboost"])
        self.assertIn('os.environ["TRAIN_CSV"]', generated["code"])

    def test_dataset_profile_describes_real_columns(self):
        train = Path(self._tmpdir.name) / "train.csv"
        train.write_text("amt,category,is_fraud,_source_dataset\n1.5,food,0,x\n9.0,travel,1,x\n")
        profile = replication_agent._dataset_profile(make_cleaning_result(
            train_csv_path=str(train), split_rows={"train": 2, "test": 1, "validation": 1}))
        self.assertIn("is_fraud", profile)
        self.assertIn("train=2", profile)
        self.assertIn("candidate labels", profile)
        self.assertIn("_source_dataset", profile)


class TestReadsTheParentPaper(ReplicationAgentTestCase):
    """The reconstruction reads the Parent Paper itself, not only the
    search record's one-line summary."""

    def test_generation_prompt_contains_the_papers_method_text(self):
        paper_text = {"status": "FULL_TEXT", "text": "We split 80/20, then applied SMOTE to the training set only."}
        with patch.object(replication_agent.llm, "chat", return_value="") as chat:
            replication_agent._generate_baseline_script(make_paper(), make_cleaning_result(), "profile", paper_text)
        prompt = chat.call_args.args[0]
        self.assertIn("applied SMOTE to the training set only", prompt)
        self.assertIn("Excerpts from the paper's own full text", prompt)

    def test_without_full_text_the_prompt_says_so(self):
        with patch.object(replication_agent.llm, "chat", return_value="") as chat:
            replication_agent._generate_baseline_script(make_paper(), make_cleaning_result(), "profile")
        prompt = chat.call_args.args[0]
        self.assertIn("full text was not available", prompt)
        self.assertIn("Random Forests", prompt)  # the stored methodology is still given

    def test_result_records_what_could_be_read(self):
        generated = {"code": "print(1)", "inferred_assumptions": [], "libraries_required": []}
        with patch.object(replication_agent, "_generate_baseline_script", return_value=generated), \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(stdout="METRIC accuracy=0.9")):
            result = replication_agent._replicate_by_reconstruction(make_paper(), make_cleaning_result())
        self.assertEqual(result["paper_text_status"], "ABSTRACT_ONLY")


class TestRevision(ReplicationAgentTestCase):
    """Master Agent relays an Evaluation Agent request: fix stated defects,
    or re-run the same script on more data. The previous baseline is
    archived first, and a cloned repository is never revised."""

    def _baseline(self, code="X_res = SMOTE().fit_resample(X, y)\nsplit()\n"):
        recon = Path(self._tmpdir.name) / replication_agent._slugify(make_paper()["title"]) / "reconstructed"
        recon.mkdir(parents=True)
        (recon / "train.py").write_text(code)
        (recon.parent / "run_stdout.txt").write_text("METRIC accuracy=0.999")
        return recon

    def _run(self, revision, revised=None):
        revised = revised if revised is not None else {"code": "fixed()", "inferred_assumptions": ["new"],
                                                       "libraries_required": []}
        with patch.object(replication_agent, "_revise_baseline_script", return_value=revised) as revise, \
             patch.object(replication_agent, "_generate_baseline_script") as generate, \
             patch.object(replication_agent, "_create_venv", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_install", return_value=run_result(exit_code=0)), \
             patch.object(replication_agent, "_pip_freeze", return_value=""), \
             patch.object(replication_agent, "_run_bounded", return_value=run_result(stdout="METRIC accuracy=0.91")):
            result = replication_agent.run(make_paper(), {"status": "NOT_FOUND"}, make_cleaning_result(), revision)
        return result, revise, generate

    def test_revision_archives_the_old_baseline_and_fixes_the_defects(self):
        recon = self._baseline()
        result, revise, generate = self._run({"round": 1, "instructions": ["Split before SMOTE."],
                                              "previous_inferred_assumptions": ["old"]})
        generate.assert_not_called()
        self.assertIn("SMOTE().fit_resample(X, y)", revise.call_args.args[0])
        self.assertEqual(revise.call_args.args[1], ["Split before SMOTE."])
        archive = recon.parent / "revisions" / "round_1"
        self.assertIn("fit_resample", (archive / "train.py").read_text())
        self.assertEqual((archive / "run_stdout.txt").read_text(), "METRIC accuracy=0.999")
        self.assertEqual((recon / "train.py").read_text(), "fixed()")
        self.assertEqual(result["revision_round"], 1)
        self.assertEqual(result["inferred_assumptions"], ["old", "new"])
        self.assertEqual(result["status"], "OK")

    def test_a_retry_of_the_same_round_keeps_the_original_archive(self):
        recon = self._baseline("original()")
        self._run({"round": 1, "instructions": ["fix"]})
        (recon / "train.py").write_text("failed attempt()")
        self._run({"round": 1, "instructions": ["fix"]})
        self.assertEqual((recon.parent / "revisions" / "round_1" / "train.py").read_text(), "original()")

    def test_more_data_reruns_the_same_script(self):
        recon = self._baseline("same_script()")
        result, revise, generate = self._run({"round": 1, "reuse_script": True,
                                              "previous_inferred_assumptions": ["kept"]})
        revise.assert_not_called()
        generate.assert_not_called()
        self.assertEqual((recon / "train.py").read_text(), "same_script()")
        self.assertEqual(result["inferred_assumptions"], ["kept"])

    def test_cloned_repository_is_never_revised(self):
        with patch.object(replication_agent, "_replicate_from_repository",
                          return_value=replication_agent._default_result(status="OK", source="cloned_repository")), \
             patch.object(replication_agent, "_revise_baseline_script") as revise:
            result = replication_agent.run(make_paper(), make_code_discovery_result(), make_cleaning_result(),
                                           {"round": 1, "instructions": ["fix leakage"]})
        revise.assert_not_called()
        self.assertTrue(any("Revision not applied" in w for w in result["warnings"]))

    def test_revise_prompt_states_the_defects_and_the_leakage_rule(self):
        with patch.object(replication_agent.llm, "chat", return_value="") as chat:
            replication_agent._revise_baseline_script("code()", ["SMOTE is applied before the split."], make_paper(),
                                                      {"status": "ABSTRACT_ONLY", "text": "t"}, "profile")
        prompt = chat.call_args.args[0]
        self.assertIn("SMOTE is applied before the split.", prompt)
        self.assertIn("fitted on the\nTRAINING data only", prompt)


def approved_review(clone_dir="", commit_sha="abc") -> dict:
    return {"verdict": "APPROVED", "commit_sha": commit_sha, "clone_dir": str(clone_dir), "summary": "APPROVED."}


class FakeSandbox:
    """Stands in for sandbox.Sandbox: records every pip call and run."""

    def __init__(self, pip=None, runs=(), broken_versions=()):
        self._pip = pip or (lambda args: run_result(exit_code=0))
        self._runs = iter(runs)
        self.broken_versions = set(broken_versions)
        self.volume, self.version, self.image, self.digest = "", "", "", "python@sha256:test"
        self.versions, self.pips, self.commands, self.closed = [], [], [], False

    def use_python(self, version):
        self.versions.append(version)
        if version in self.broken_versions:
            return run_result(exit_code=1, stderr="pull failed")
        self.volume, self.version, self.image = "vol", version, f"python:{version}-slim"
        return run_result(exit_code=0)

    def pip(self, args, timeout):
        self.pips.append(list(args))
        return self._pip(args)

    def run(self, command, env, timeout):
        self.commands.append((list(command), dict(env)))
        return next(self._runs)

    def freeze(self):
        return "pandas==2.2.3\n"

    def data_env(self, variables):
        return {var: f"/data/{split}.csv" for var, split in variables.items()}

    def describe(self):
        return {"image": self.image, "digest": self.digest, "python": self.version, "run_network": "none",
                "run_user": "65534:65534", "memory": "6g", "cpus": "8", "pids": "1024",
                "capabilities": "all dropped, no-new-privileges", "host_mounts": ["repository copy (read-write)"]}

    def close(self):
        self.closed = True


class RepositoryTestCase(ReplicationAgentTestCase):
    """The repository path, with the reviewed clone replaced by files the
    test writes and the Docker sandbox by FakeSandbox (no git, no Docker)."""
    files = {"train.py": "print('training')\n"}

    def setUp(self):
        super().setUp()
        self.slug_dir = replication_agent.REPLICATION_DIR / replication_agent._slugify(make_paper()["title"])
        self.repo_dir = self.slug_dir / "repo"
        replication_agent._FAILED_REPOSITORIES.clear()

    def tearDown(self):
        replication_agent._FAILED_REPOSITORIES.clear()
        super().tearDown()

    def _code(self, files=None):
        def fake(repository, dest):
            dest.mkdir(parents=True, exist_ok=True)
            for name, text in (self.files if files is None else files).items():
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                (dest / name).write_text(text)
            return {"ok": True, "error": "", "commit_sha": "abc"}
        return patch.object(replication_agent, "_get_repository_code", side_effect=fake)

    def _env(self, **overrides):
        env = {"ok": True, "error": "", "python": "3.12 (python@sha256:test)", "install": "install -r requirements.txt",
               "modifications": [], "warnings": []}
        env.update(overrides)
        return patch.object(replication_agent, "_setup_repository_environment", return_value=env)

    def _docker(self, box, available=True):
        return (patch.object(replication_agent.sandbox, "engine_available",
                             return_value=(available, "" if available else "The Docker engine is not running.")),
                patch.object(replication_agent.sandbox, "Sandbox", return_value=box))

    def _repository(self):
        return make_repository(review=approved_review())


class TestGetRepositoryCode(ReplicationAgentTestCase):
    def _git_repo(self) -> tuple:
        source = Path(self._tmpdir.name) / "reviewed"
        source.mkdir()
        (source / "train.py").write_text("print(1)\n")
        git = ["git", "-c", "user.email=t@example.com", "-c", "user.name=t"]
        for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
            subprocess.run(git + args, cwd=source, check=True, capture_output=True)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True).stdout.strip()
        return source, sha

    def test_copies_the_reviewed_clone_at_the_reviewed_commit(self):
        source, sha = self._git_repo()
        dest = Path(self._tmpdir.name) / "repo"
        result = replication_agent._get_repository_code({"review": approved_review(source, sha)}, dest)

        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(result["commit_sha"], sha)
        self.assertTrue((dest / "train.py").is_file())

    def test_a_different_commit_is_refused(self):
        source, _sha = self._git_repo()
        result = replication_agent._get_repository_code({"review": approved_review(source, "0" * 40)},
                                                         Path(self._tmpdir.name) / "repo")
        self.assertFalse(result["ok"])
        self.assertIn("not the reviewed", result["error"])

    def test_a_repository_master_agent_did_not_approve_is_never_copied(self):
        dest = Path(self._tmpdir.name) / "repo"
        for review in (None, {"verdict": "REJECTED", "summary": "REJECTED: obfuscated_exec"}):
            result = replication_agent._get_repository_code({"review": review}, dest)
            self.assertFalse(result["ok"])
            self.assertIn("has not approved", result["error"])
        self.assertFalse(dest.exists())


class TestRepositoryEnvironment(ReplicationAgentTestCase):
    """Live run 20261114T203359Z: pandas==2.2.3 did not install on Python 3.14."""

    def setUp(self):
        super().setUp()
        self.repo_dir = Path(self._tmpdir.name) / "repo"
        self.repo_dir.mkdir()

    def _setup(self, box):
        return replication_agent._setup_repository_environment(self.repo_dir, box)

    def test_wheels_are_tried_on_each_sandbox_python_before_a_source_build(self):
        (self.repo_dir / "requirements.txt").write_text("pandas==2.2.3\n")
        attempts = iter([1, 0])
        box = FakeSandbox(pip=lambda args: run_result(exit_code=next(attempts)))
        env = self._setup(box)

        self.assertTrue(env["ok"])
        self.assertEqual(box.versions, list(replication_agent.sandbox.SANDBOX_PYTHON_VERSIONS[:2]))
        self.assertEqual(box.pips, [["install", "--only-binary=:all:", "-r", "requirements.txt"]] * 2)
        self.assertIn(f"Python {replication_agent.sandbox.SANDBOX_PYTHON_VERSIONS[1]}", env["modifications"][0])
        self.assertIn("python@sha256:test", env["python"])

    def test_a_source_build_is_tried_only_after_every_wheel_install(self):
        (self.repo_dir / "requirements.txt").write_text("oldpkg==0.1\n")
        versions = replication_agent.sandbox.SANDBOX_PYTHON_VERSIONS
        box = FakeSandbox(pip=lambda args: run_result(exit_code=1 if "--only-binary=:all:" in args else 0))
        env = self._setup(box)

        self.assertTrue(env["ok"])
        self.assertEqual(box.pips[-1], ["install", "-r", "requirements.txt"])
        self.assertEqual(box.versions, [*versions, versions[0]])
        self.assertEqual(env["modifications"], [])  # the first Python, the declared versions

    def test_a_python_whose_image_cannot_be_made_is_skipped(self):
        (self.repo_dir / "requirements.txt").write_text("numpy\n")
        first, second = replication_agent.sandbox.SANDBOX_PYTHON_VERSIONS[:2]
        box = FakeSandbox(broken_versions={first})
        env = self._setup(box)

        self.assertTrue(env["ok"])
        self.assertEqual(box.version, second)
        self.assertIn("could not be made", " ".join(env["warnings"]))

    def test_a_pin_that_cannot_install_is_relaxed_and_recorded(self):
        (self.repo_dir / "requirements.txt").write_text("pandas==2.2.3\nnumpy==2.2.6\n")
        box = FakeSandbox(pip=lambda args: run_result(
            exit_code=1 if "-r" in args or args == ["install", "pandas==2.2.3"] else 0))
        env = self._setup(box)

        self.assertTrue(env["ok"])
        self.assertIn(["install", "pandas"], box.pips)
        self.assertIn("without the pin 'pandas==2.2.3'", " ".join(env["modifications"]))
        self.assertNotIn("numpy", " ".join(env["modifications"]))

    def test_a_requirement_that_never_installs_fails_the_setup(self):
        (self.repo_dir / "requirements.txt").write_text("pandas==2.2.3\n")
        env = self._setup(FakeSandbox(pip=lambda args: run_result(exit_code=1, stderr="boom")))

        self.assertFalse(env["ok"])
        self.assertIn("pandas==2.2.3", env["error"])

    def test_packages_the_code_imports_but_does_not_declare_are_installed(self):
        (self.repo_dir / "requirements.txt").write_text("numpy\n")
        (self.repo_dir / "train.py").write_text("import numpy\nimport sklearn.ensemble\nimport helpers\n"
                                                "import os\nimport strange_lib\n")
        (self.repo_dir / "helpers.py").write_text("x = 1\n")
        box = FakeSandbox()
        env = self._setup(box)

        self.assertIn(["install", "scikit-learn"], box.pips)
        self.assertNotIn(["install", "numpy"], box.pips)  # declared: installed by -r
        self.assertNotIn(["install", "helpers"], box.pips)  # the repository's own module
        self.assertIn("strange_lib", " ".join(env["warnings"]))  # unknown: logged, never installed
        self.assertFalse(any("strange_lib" in " ".join(a) for a in box.pips))

    def test_no_dependency_file_is_not_a_failure(self):
        (self.repo_dir / "train.py").write_text("import pandas\n")
        box = FakeSandbox()
        env = self._setup(box)

        self.assertTrue(env["ok"])
        self.assertEqual(box.pips, [["install", "pandas"]])


class TestRepositoryRun(RepositoryTestCase):
    def _run(self, run_results, fix=None):
        box = FakeSandbox(runs=run_results)
        fix = fix or {"diagnosis": "", "actions": []}
        engine, sandbox_class = self._docker(box)
        with self._code(), self._env(), engine, sandbox_class, \
             patch.object(replication_agent, "_wire_dataset_into_repo", return_value=[]), \
             patch.object(replication_agent, "_extract_entry_point_from_readme", return_value=[]), \
             patch.object(replication_agent, "_propose_repository_fix", return_value=fix) as propose:
            result = replication_agent._replicate_from_repository(make_paper(), self._repository(),
                                                                  make_cleaning_result())
        return result, box, propose

    def test_successful_run_in_the_sandbox(self):
        result, box, propose = self._run([run_result(exit_code=0, stdout="accuracy: 0.9")])

        self.assertEqual(result["status"], "OK")
        self.assertFalse(result["repository_failed"])
        self.assertEqual(result["entry_point"], "train.py")
        self.assertEqual(result["repository_review"]["verdict"], "APPROVED")
        self.assertEqual(result["sandbox"]["run_network"], "none")
        command, env = box.commands[0]
        self.assertEqual(command, ["python", "train.py"])
        self.assertEqual(env["TRAIN_CSV"], "/data/train.csv")  # the container path, not the host one
        self.assertTrue(box.closed)  # the sandbox volume is always removed
        self.assertIn("Sandbox: Docker", (self.slug_dir / "ENVIRONMENT.md").read_text())
        propose.assert_not_called()

    def test_without_docker_the_repository_is_never_run(self):
        box = FakeSandbox()
        engine, sandbox_class = self._docker(box, available=False)
        with self._code(), engine, sandbox_class as made, \
             patch.object(replication_agent, "_setup_repository_environment") as setup:
            result = replication_agent._replicate_from_repository(make_paper(), self._repository(),
                                                                  make_cleaning_result())

        made.assert_not_called()
        setup.assert_not_called()
        self.assertTrue(result["repository_failed"])
        self.assertIn("Docker is not available", result["notes"])

    def test_a_missing_module_is_installed_and_the_run_repeated(self):
        result, box, propose = self._run([
            run_result(exit_code=1, stderr="ModuleNotFoundError: No module named 'sklearn'"),
            run_result(exit_code=0)])

        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(box.commands), 2)
        self.assertEqual(box.pips, [["install", "scikit-learn"]])
        self.assertIn("ModuleNotFoundError", " ".join(result["modifications"]))
        propose.assert_not_called()

    def test_an_unknown_missing_module_is_never_installed(self):
        result, box, _ = self._run([
            run_result(exit_code=1, stderr="ModuleNotFoundError: No module named 'evil_pkg'"),
            run_result(exit_code=1, stderr="ModuleNotFoundError: No module named 'evil_pkg'")])

        self.assertEqual(box.pips, [])
        self.assertTrue(result["repository_failed"])

    def test_a_failing_run_gets_one_checked_repair_and_runs_again(self):
        self.files = {"train.py": "x\n", "src/main.py": "x\n"}
        fix = {"diagnosis": "train.py is a stub; src/main.py is the experiment.",
               "actions": [{"type": "command", "command": ["python", "src/main.py", "--seed", "1"]}]}
        result, box, _ = self._run([run_result(exit_code=1, stderr="error"), run_result(exit_code=0)], fix)

        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["entry_point_source"], "repair_llm")
        self.assertEqual(box.commands[-1][0], ["python", "src/main.py", "--seed", "1"])
        self.assertIn("Repair: ran python src/main.py --seed 1 instead.", result["modifications"])

    def test_still_failing_after_the_repair_marks_the_repository_failed(self):
        fix = {"diagnosis": "d", "actions": [{"type": "install", "packages": ["xgboost"]}]}
        result, box, _ = self._run([run_result(exit_code=1, stderr="e1"), run_result(exit_code=1, stderr="e2")], fix)

        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(result["repository_failed"])
        self.assertEqual(len(box.commands), 2)  # one run, one run after the repair -- no more
        self.assertIn(["install", "xgboost"], box.pips)

    def test_a_repair_with_no_valid_action_does_not_run_again(self):
        fix = {"diagnosis": "d", "actions": [{"type": "command", "command": ["bash", "-c", "rm -rf /"]}]}
        result, box, _ = self._run([run_result(exit_code=1, stderr="e1")], fix)

        self.assertTrue(result["repository_failed"])
        self.assertEqual(len(box.commands), 1)
        self.assertIn("Repair action rejected", " ".join(result["warnings"]))

    def test_no_entry_point_asks_for_a_repair(self):
        self.files = {"experiment.py": "x\n"}
        fix = {"diagnosis": "d", "actions": [{"type": "command", "command": ["python", "experiment.py"]}]}
        result, box, propose = self._run([run_result(exit_code=0)], fix)

        propose.assert_called_once()
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["entry_point"], "experiment.py")

    def test_a_timeout_is_not_repaired(self):
        result, box, propose = self._run([run_result(timed_out=True, exit_code=None)])

        self.assertEqual(result["status"], "TIMEOUT")
        self.assertTrue(result["repository_failed"])
        propose.assert_not_called()

    def test_an_unapproved_repository_is_never_set_up(self):
        with patch.object(replication_agent.sandbox, "engine_available") as engine:
            result = replication_agent._replicate_from_repository(
                make_paper(), make_repository(review={"verdict": "REJECTED", "summary": "REJECTED: malware"}),
                make_cleaning_result())

        engine.assert_not_called()
        self.assertTrue(result["repository_failed"])
        self.assertIn("has not approved", result["warnings"][-1])

    def test_an_old_copy_with_read_only_files_does_not_stop_a_retry(self):
        (self.repo_dir / ".git").mkdir(parents=True)
        (self.repo_dir / ".git" / "pack.idx").write_text("x")
        (self.repo_dir / ".git" / "pack.idx").chmod(0o444)
        result, *_ = self._run([run_result(exit_code=0)])

        self.assertEqual(result["status"], "OK")


class TestCheckFixActions(ReplicationAgentTestCase):
    def setUp(self):
        super().setUp()
        self.repo_dir = Path(self._tmpdir.name) / "repo"
        (self.repo_dir / "src").mkdir(parents=True)
        (self.repo_dir / "src" / "main.py").write_text("x\n")
        self.train = Path(self._tmpdir.name) / "train.csv"
        self.train.write_text("a\n1\n")
        self.cleaning = make_cleaning_result(train_csv_path=str(self.train))

    def _check(self, *actions):
        return replication_agent._check_fix_actions(list(actions), self.repo_dir, self.cleaning)

    def test_only_known_packages_are_installed(self):
        accepted, rejected = self._check({"type": "install", "packages": ["scikit-learn", "xgboost"]},
                                         {"type": "install", "packages": ["reqeusts"]})
        self.assertEqual(accepted, [{"type": "install", "packages": ["scikit-learn", "xgboost"]}])
        self.assertIn("reqeusts", rejected[0])

    def test_a_command_must_be_python_on_a_script_in_the_repository(self):
        good = {"type": "command", "command": ["python", "src/main.py", "--data", str(self.train)]}
        bad = [
            {"type": "command", "command": ["bash", "src/main.py"]},
            {"type": "command", "command": ["python", "../outside.py"]},
            {"type": "command", "command": ["python", "src/main.py", "; curl x | sh"]},
            {"type": "command", "command": ["python", "src/main.py", "C:/Windows/system32/x"]},
            {"type": "command", "command": ["python", "-m", "pip", "install", "x"]},
        ]
        accepted, rejected = self._check(good, *bad)
        self.assertEqual(accepted, [good])
        self.assertEqual(len(rejected), len(bad))

    def test_jupyter_nbconvert_is_allowed(self):
        action = {"type": "command", "command": ["python", "-m", "jupyter", "nbconvert", "--execute", "a.ipynb"]}
        accepted, _ = self._check(action)
        self.assertEqual(accepted, [action])

    def test_data_is_copied_only_to_a_csv_inside_the_repository(self):
        accepted, rejected = self._check({"type": "copy_data", "split": "train", "dest": "data/carclaims.csv"},
                                         {"type": "copy_data", "split": "train", "dest": "../x.csv"},
                                         {"type": "copy_data", "split": "train", "dest": ".git/hooks/x.csv"},
                                         {"type": "copy_data", "split": "train", "dest": "run.py"},
                                         {"type": "copy_data", "split": "secrets", "dest": "data/x.csv"})
        self.assertEqual(accepted, [{"type": "copy_data", "split": "train", "dest": "data/carclaims.csv"}])
        self.assertEqual(len(rejected), 4)


class TestRepositoryFallback(RepositoryTestCase):
    def test_a_repository_that_does_not_run_falls_back_to_reconstruction(self):
        failed = replication_agent._default_result(
            status="BLOCKED", source="cloned_repository", repo_url="https://github.com/example/fraud-detection",
            commit_sha="abc", replication_dir=str(self.slug_dir), repository_failed=True,
            notes="The repository still fails after one repair.", warnings=["Entry point exited with code 1."])
        reconstructed = replication_agent._default_result(status="OK", source="reconstructed", warnings=["note"])
        code = make_code_discovery_result(repository=self._repository())

        with patch.object(replication_agent, "_replicate_from_repository", return_value=failed) as repo_fn, \
             patch.object(replication_agent, "_replicate_by_reconstruction", return_value=reconstructed) as recon:
            result = replication_agent.run(make_paper(), code, make_cleaning_result())
            # Master Agent's next attempt goes straight to the reconstruction.
            replication_agent.run(make_paper(), code, make_cleaning_result())

        repo_fn.assert_called_once()
        self.assertEqual(recon.call_count, 2)
        self.assertEqual(result["source"], "reconstructed")
        self.assertEqual(result["repository_fallback"]["repo_url"], "https://github.com/example/fraud-detection")
        self.assertIn("did not run", result["warnings"][0])
        self.assertIn("Entry point exited with code 1.", result["warnings"])
        self.assertIn("note", result["warnings"])

    def test_a_repository_that_runs_does_not_fall_back(self):
        ok = replication_agent._default_result(status="OK", source="cloned_repository")
        with patch.object(replication_agent, "_replicate_from_repository", return_value=ok), \
             patch.object(replication_agent, "_replicate_by_reconstruction") as recon:
            result = replication_agent.run(make_paper(), make_code_discovery_result(), make_cleaning_result())

        recon.assert_not_called()
        self.assertEqual(result["source"], "cloned_repository")


class TestChildEnvironment(unittest.TestCase):
    def test_a_child_process_never_gets_the_projects_secrets(self):
        secrets = {"OPENROUTER_API_KEY": "sk-1", "SLACK_BOT_TOKEN": "xoxb", "KAGGLE_KEY": "k",
                   "MEMORY_DATABASE_URL": "postgres://u:p@h/db", "FREELLMAPI_API_KEY": "f"}
        with patch.dict(replication_agent.os.environ, {**secrets, "PATH_EXTRA_OK": "1"}), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")) as run:
            replication_agent._run_bounded(["python", "train.py"], env={"TRAIN_CSV": "t.csv"})

        child = run.call_args.kwargs["env"]
        self.assertFalse(set(secrets) & set(child))
        self.assertEqual((child["PATH_EXTRA_OK"], child["TRAIN_CSV"], child["PYTHONUTF8"]), ("1", "t.csv", "1"))


class TestRemoveTree(ReplicationAgentTestCase):
    def test_remove_tree_deletes_read_only_files(self):
        target = Path(self._tmpdir.name) / "repo" / ".git" / "objects" / "pack"
        target.mkdir(parents=True)
        pack = target / "pack-1.idx"
        pack.write_text("x")
        pack.chmod(0o444)

        replication_agent._remove_tree(Path(self._tmpdir.name) / "repo")

        self.assertFalse((Path(self._tmpdir.name) / "repo").exists())
