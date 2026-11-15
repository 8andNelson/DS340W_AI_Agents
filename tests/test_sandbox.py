"""The Docker sandbox the Replication Agent runs a cloned repository in.
The docker CLI is mocked: these tests check the exact commands."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.orchestration import sandbox


def ok(stdout: str = "") -> dict:
    return {"exit_code": 0, "stdout": stdout, "stderr": "", "duration_seconds": 0.1, "timed_out": False}


class SandboxTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        self.train = root / "train.csv"
        self.train.write_text("a\n1\n")
        self.calls = []

        def fake_docker(args, timeout, name=""):
            self.calls.append(list(args))
            if args[:2] == ["image", "inspect"] and "--format" in args:
                return ok("python@sha256:abc\n")
            return ok()

        self._patch = patch.object(sandbox, "_docker", side_effect=fake_docker)
        self._patch.start()
        self.box = sandbox.Sandbox(self.repo, {"train": str(self.train), "test": str(root / "missing.csv")})

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def runs(self) -> list:
        return [c for c in self.calls if c[0] == "run"]


class TestEnvironment(SandboxTestCase):
    def test_use_python_makes_a_volume_and_a_venv_and_records_the_digest(self):
        self.box.use_python("3.12")

        self.assertEqual(self.box.image, "python:3.12-slim")
        self.assertEqual(self.box.digest, "python@sha256:abc")
        self.assertTrue(self.box.volume.startswith(sandbox.CONTAINER_PREFIX))
        self.assertIn(["volume", "create", self.box.volume], self.calls)
        self.assertEqual(self.runs()[-1][-4:], ["python", "-m", "venv", "/deps/venv"])

    def test_a_missing_image_is_pulled(self):
        def fake_docker(args, timeout, name=""):
            self.calls.append(list(args))
            return {**ok(), "exit_code": 1} if args == ["image", "inspect", "python:3.11-slim"] else ok()

        with patch.object(sandbox, "_docker", side_effect=fake_docker):
            self.box.use_python("3.11")
        self.assertIn(["pull", "-q", "python:3.11-slim"], self.calls)

    def test_a_new_python_removes_the_old_volume(self):
        self.box.use_python("3.12")
        first = self.box.volume
        self.box.use_python("3.11")
        self.assertIn(["volume", "rm", "-f", first], self.calls)
        self.box.close()
        self.assertIn(["volume", "rm", "-f", f"{sandbox.CONTAINER_PREFIX}{self.box.id}-py311"], self.calls)
        self.assertEqual(self.box.volume, "")

    def test_nothing_runs_without_an_environment(self):
        result = self.box.run(["python", "train.py"], env={}, timeout=10)
        self.assertIsNone(result["exit_code"])
        self.assertEqual(self.runs(), [])


class TestContainerCommands(SandboxTestCase):
    def setUp(self):
        super().setUp()
        self.box.use_python("3.12")
        self.calls.clear()

    def _flag(self, args: list, flag: str) -> list:
        return [args[i + 1] for i, a in enumerate(args) if a == flag]

    def test_the_run_has_no_network_runs_as_nobody_and_is_limited(self):
        self.box.run(["python", "train.py", "--epochs", "3"], env={"TRAIN_CSV": "/data/train.csv"}, timeout=60)
        args = self.runs()[0]

        self.assertEqual(self._flag(args, "--network"), ["none"])
        self.assertEqual(self._flag(args, "--user"), [sandbox.RUN_USER])
        self.assertEqual(self._flag(args, "--cap-drop"), ["ALL"])
        self.assertEqual(self._flag(args, "--security-opt"), ["no-new-privileges"])
        self.assertEqual(self._flag(args, "--memory"), [sandbox.MEMORY_LIMIT])
        self.assertEqual(self._flag(args, "--pids-limit"), [sandbox.PIDS_LIMIT])
        self.assertIn("--rm", args)
        self.assertEqual(args[-4:], [sandbox.VENV_PYTHON, "train.py", "--epochs", "3"])

    def test_only_the_repository_and_the_existing_csvs_are_mounted(self):
        self.box.run(["python", "train.py"], env={}, timeout=60)
        mounts = self._flag(self.runs()[0], "--mount")

        self.assertEqual(len(mounts), 3)  # the venv volume, the repository, train.csv (test.csv does not exist)
        self.assertIn(f"type=bind,source={self.repo.resolve()},target=/repo", mounts)
        self.assertIn(f"type=bind,source={self.train.resolve()},target=/data/train.csv,readonly", mounts)
        self.assertIn(f"type=volume,source={self.box.volume},target=/deps", mounts)

    def test_no_host_environment_variable_is_passed(self):
        with patch.dict(sandbox.os.environ, {"OPENROUTER_API_KEY": "sk-secret"}):
            self.box.run(["python", "train.py"], env={"TRAIN_CSV": "/data/train.csv"}, timeout=60)
        variables = self._flag(self.runs()[0], "-e")

        self.assertIn("TRAIN_CSV=/data/train.csv", variables)
        self.assertFalse(any("sk-secret" in v or "OPENROUTER" in v for v in variables))
        self.assertNotIn("--env-file", self.runs()[0])

    def test_pip_has_network_but_no_privileges(self):
        self.box.pip(["install", "-r", "requirements.txt"], timeout=60)
        args = self.runs()[0]

        self.assertEqual(self._flag(args, "--network"), ["bridge"])
        self.assertEqual(self._flag(args, "--cap-drop"), ["ALL"])
        self.assertEqual(args[args.index(sandbox.VENV_PYTHON):],
                         [sandbox.VENV_PYTHON, "-m", "pip", "--disable-pip-version-check", "--no-input",
                          "install", "-r", "requirements.txt"])

    def test_a_host_data_path_becomes_its_container_path(self):
        self.box.run(["python", "train.py", "--data", str(self.train)], env={}, timeout=60)
        self.assertEqual(self.runs()[0][-2:], ["--data", "/data/train.csv"])

    def test_data_env_names_only_mounted_splits(self):
        self.assertEqual(self.box.data_env({"TRAIN_CSV": "train", "TEST_CSV": "test"}),
                         {"TRAIN_CSV": "/data/train.csv"})

    def test_a_path_with_a_comma_is_refused(self):
        box = sandbox.Sandbox(Path(self._tmp.name) / "a,b", {})
        box.volume, box.image = "v", "python:3.12-slim"
        with self.assertRaises(ValueError):
            box.run(["python", "x.py"], env={}, timeout=10)


class TestDockerCli(unittest.TestCase):
    def _run(self, fail_with=None):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[1] == "run" and fail_with:
                raise fail_with
            return MagicMock(returncode=0, stdout="29.9.2", stderr="")

        with patch.object(sandbox, "find_docker", return_value="docker"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch.dict(sandbox.os.environ, {"OPENROUTER_API_KEY": "sk-secret"}):
            result = sandbox._docker(["run", "x"], timeout=5, name="ds340w-replication-x")
        return result, calls

    def test_a_timeout_removes_the_container(self):
        result, calls = self._run(subprocess.TimeoutExpired(["docker", "run"], 5))

        self.assertTrue(result["timed_out"])
        self.assertEqual(calls[-1][0], ["docker", "rm", "-f", "ds340w-replication-x"])

    def test_the_docker_cli_gets_no_secrets(self):
        _result, calls = self._run()
        self.assertNotIn("OPENROUTER_API_KEY", calls[0][1]["env"])

    def test_without_the_docker_cli_nothing_runs(self):
        with patch("subprocess.run") as run:
            result = sandbox._docker(["run", "x"], timeout=5)  # conftest: no docker CLI
        run.assert_not_called()
        self.assertIn("not found", result["stderr"])
        self.assertEqual(sandbox.engine_available(), (False, "The docker command was not found. Install Docker "
                                                             "Desktop to run repository code."))

    def test_a_running_engine_is_available(self):
        with patch.object(sandbox, "find_docker", return_value="docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="29.9.2", stderr="")):
            self.assertEqual(sandbox.engine_available(), (True, ""))

    def test_a_stopped_engine_without_docker_desktop_is_not_available(self):
        with patch.object(sandbox, "find_docker", return_value="docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="down")):
            available, reason = sandbox.engine_available()
        self.assertFalse(available)
        self.assertIn("not running", reason)


if __name__ == "__main__":
    unittest.main()
