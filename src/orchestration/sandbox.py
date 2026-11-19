"""
Docker sandbox for a cloned repository's code (Replication Agent).

The authors' code, its pip install (pip runs setup.py while it installs)
and every repair run inside a throwaway Linux container, never on this
machine:

* Image python:<version>-slim (SANDBOX_PYTHON_VERSIONS, in order). Its
  digest is recorded, so the run can be repeated on the same image.
* The venv lives in a Docker volume made for this attempt (close() removes
  it), not in a host folder.
* The only host folder mounted is the repository copy (read-write, so the
  code can write its outputs). Each prepared CSV is mounted read-only at
  /data/<split>.csv.
* Install steps have network (pip must download) and run as root with every
  Linux capability dropped and no-new-privileges. The run itself has no
  network (--network none) and runs as nobody (uid 65534), with memory,
  CPU and process limits.
* No host environment variable reaches a container: docker run passes only
  the -e values given here, so no secret can. The docker CLI itself gets
  repo_review.child_environment().

When the engine is not running, Docker Desktop is started (Windows) and
waited for. When Docker still is not available, the repository is not run
at all; the Replication Agent reconstructs the baseline from the paper.
"""
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from src.orchestration import repo_review

SANDBOX_PYTHON_VERSIONS = ("3.12", "3.11", "3.13", "3.10")  # widest wheel coverage for research pins first
IMAGE = "python:{version}-slim"
MEMORY_LIMIT = "6g"     # Docker Desktop's VM here has 7.9 GB
CPU_LIMIT = "8"
PIDS_LIMIT = "1024"
RUN_USER = "65534:65534"  # nobody
PULL_TIMEOUT_SECONDS = 900
ENGINE_CHECK_TIMEOUT_SECONDS = 20
ENGINE_START_TIMEOUT_SECONDS = 180
VENV_PYTHON = "/deps/venv/bin/python"
CONTAINER_PREFIX = "ds340w-replication-"

HARDENING = ["--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", PIDS_LIMIT,
             "--memory", MEMORY_LIMIT, "--cpus", CPU_LIMIT]


def _no_window() -> dict:
    # docker is a console program; without this every call flashes a window.
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def find_docker() -> str:
    found = shutil.which("docker")
    return found if found and Path(found).suffix.lower() not in (".cmd", ".bat") else ""


def _find_docker_desktop() -> Path | None:
    if os.name != "nt":
        return None
    candidates = []
    if find_docker():
        # <install>\resources\bin\docker.exe -> <install>\Docker Desktop.exe
        candidates.append(Path(find_docker()).resolve().parent.parent.parent / "Docker Desktop.exe")
    for var, sub in (("LOCALAPPDATA", ("Programs", "DockerDesktop")), ("ProgramFiles", ("Docker", "Docker"))):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var], *sub, "Docker Desktop.exe"))
    return next((c for c in candidates if c.is_file()), None)


def _docker(args: list, timeout: int | None, name: str = "") -> dict:
    """Run the docker CLI. Never raises; the same result shape as the
    Replication Agent's _run_bounded. On a timeout the named container is
    removed too -- killing the CLI alone would leave it running."""
    start = time.monotonic()
    docker = find_docker()
    if not docker:
        return {"exit_code": None, "stdout": "", "stderr": "The docker command was not found.",
                "duration_seconds": 0.0, "timed_out": False}
    if timeout is None:
        # No time limit (a repository's run): stopped only if it hangs.
        from src.orchestration import watchdog
        return watchdog.run([docker, *args], env=repo_review.child_environment(), label="Sandbox",
                            progress=watchdog.docker_progress(docker, name) if name else None,
                            on_kill=(lambda: subprocess.run([docker, "rm", "-f", name], capture_output=True,
                                                            timeout=60, **_no_window())) if name else None,
                            popen_kwargs=_no_window())
    try:
        proc = subprocess.run([docker, *args], capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, env=repo_review.child_environment(),
                              **_no_window())
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "duration_seconds": time.monotonic() - start, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        if name:
            subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=60, **_no_window())
        return {"exit_code": None, "stdout": _text(e.stdout), "stderr": _text(e.stderr) + f"\nTimed out after {timeout}s.",
                "duration_seconds": time.monotonic() - start, "timed_out": True}
    except Exception as e:
        return {"exit_code": None, "stdout": "", "stderr": str(e), "duration_seconds": time.monotonic() - start,
                "timed_out": False}


def _text(output) -> str:
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else (output or "")


def _engine_up() -> bool:
    return _docker(["info", "--format", "{{.ServerVersion}}"], timeout=ENGINE_CHECK_TIMEOUT_SECONDS)["exit_code"] == 0


def engine_available(start: bool = True) -> tuple:
    """(available, reason). Starts Docker Desktop when the engine is down."""
    if not find_docker():
        return False, "The docker command was not found. Install Docker Desktop to run repository code."
    if _engine_up():
        return True, ""
    desktop = _find_docker_desktop() if start else None
    if not desktop:
        return False, "The Docker engine is not running."
    print("[Sandbox] Starting Docker Desktop...")
    subprocess.Popen([str(desktop)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    deadline = time.time() + ENGINE_START_TIMEOUT_SECONDS
    while time.time() < deadline:
        if _engine_up():
            return True, ""
        time.sleep(3)
    return False, f"The Docker engine did not start in {ENGINE_START_TIMEOUT_SECONDS} seconds."


class Sandbox:
    """One attempt's container setup for one repository copy."""

    def __init__(self, repo_dir: Path, data_files: dict):
        self.repo_dir = Path(repo_dir).resolve()
        # split -> host CSV; only files that exist are mounted.
        self.data_files = {split: str(Path(path).resolve()) for split, path in (data_files or {}).items()
                           if path and Path(path).is_file()}
        self.id = uuid.uuid4().hex[:10]
        self.volume = ""
        self.version = ""
        self.image = ""
        self.digest = ""

    # --- paths ---

    @staticmethod
    def data_path(split: str) -> str:
        return f"/data/{split}.csv"

    def data_env(self, variables: dict) -> dict:
        """{"TRAIN_CSV": "train", ...} -> container paths of the mounted splits."""
        return {var: self.data_path(split) for var, split in variables.items() if split in self.data_files}

    def translate(self, args: list) -> list:
        """A host CSV path in a command becomes its path in the container."""
        by_host = {os.path.normcase(host): self.data_path(split) for split, host in self.data_files.items()}
        return [by_host.get(os.path.normcase(str(Path(a).resolve())), a) if Path(a).is_absolute() else a
                for a in map(str, args)]

    def _mounts(self) -> list:
        if "," in str(self.repo_dir) or any("," in p for p in self.data_files.values()):
            raise ValueError("A path with a comma cannot be mounted into the sandbox.")
        mounts = ["--mount", f"type=volume,source={self.volume},target=/deps",
                  "--mount", f"type=bind,source={self.repo_dir},target=/repo"]
        for split, host in self.data_files.items():
            mounts += ["--mount", f"type=bind,source={host},target={self.data_path(split)},readonly"]
        return mounts

    # --- environment ---

    def use_python(self, version: str) -> dict:
        """A new venv on python:<version>-slim (pulled if missing), in a new volume."""
        self.close()
        image = IMAGE.format(version=version)
        if _docker(["image", "inspect", image], timeout=60)["exit_code"] != 0:
            print(f"[Sandbox] Pulling {image}...")
            pulled = _docker(["pull", "-q", image], timeout=PULL_TIMEOUT_SECONDS)
            if pulled["exit_code"] != 0:
                return pulled
        self.volume = f"{CONTAINER_PREFIX}{self.id}-py{version.replace('.', '')}"
        made = _docker(["volume", "create", self.volume], timeout=60)
        if made["exit_code"] != 0:
            self.volume = ""
            return made
        self.version, self.image = version, image
        digest = _docker(["image", "inspect", image, "--format", "{{index .RepoDigests 0}}"], timeout=60)
        self.digest = digest["stdout"].strip() if digest["exit_code"] == 0 else ""
        return self._container(["python", "-m", "venv", "/deps/venv"], network=True, timeout=300)

    def pip(self, args: list, timeout: int) -> dict:
        """pip <args> in the venv, with network, in the repository folder."""
        return self._container([VENV_PYTHON, "-m", "pip", "--disable-pip-version-check", "--no-input",
                                *self.translate(args)], network=True, timeout=timeout)

    def freeze(self) -> str:
        result = self._container([VENV_PYTHON, "-m", "pip", "freeze"], network=False, timeout=120)
        return result["stdout"] if result["exit_code"] == 0 else ""

    def run(self, command: list, env: dict, timeout: int) -> dict:
        """The repository's command, with no network, as nobody. A leading
        "python"/"python3" is the venv's Python."""
        command = [str(c) for c in command]
        if command and command[0] in ("python", "python3"):
            command = [VENV_PYTHON, *command[1:]]
        return self._container(self.translate(command), network=False, timeout=timeout, env=env, user=RUN_USER)

    def _container(self, command: list, network: bool, timeout: int, env: dict = None, user: str = "") -> dict:
        if not self.volume:
            return {"exit_code": None, "stdout": "", "stderr": "No sandbox environment was made.",
                    "duration_seconds": 0.0, "timed_out": False}
        name = f"{CONTAINER_PREFIX}{self.id}-{uuid.uuid4().hex[:6]}"
        variables = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "HOME": "/tmp", "MPLBACKEND": "Agg",
                     **(env or {})}
        args = ["run", "--rm", "--name", name, *HARDENING, "--network", "bridge" if network else "none",
                *(["--user", user] if user else []), *self._mounts(), "-w", "/repo"]
        for key, value in variables.items():
            args += ["-e", f"{key}={value}"]
        return _docker([*args, self.image, *command], timeout=timeout, name=name)

    def describe(self) -> dict:
        return {"image": self.image, "digest": self.digest, "python": self.version, "run_network": "none",
                "run_user": RUN_USER, "memory": MEMORY_LIMIT, "cpus": CPU_LIMIT, "pids": PIDS_LIMIT,
                "capabilities": "all dropped, no-new-privileges", "host_mounts": ["repository copy (read-write)"]
                + [f"{split} CSV (read-only)" for split in self.data_files]}

    def close(self) -> None:
        if self.volume:
            _docker(["volume", "rm", "-f", self.volume], timeout=120)
            self.volume = ""
