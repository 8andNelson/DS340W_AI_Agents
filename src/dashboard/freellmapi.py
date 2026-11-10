"""
Starts the FreeLLMAPI router (Docker) for the dashboard.

When LLM_PROVIDER=freellmapi, every agent's LLM call goes to a FreeLLMAPI
router on localhost:3001, which runs as a Docker Compose project. The
manual steps were: open Docker Desktop, then `docker compose up -d` in the
FreeLLMAPI folder. The dashboard does the same steps:

  1. If the router already answers /api/ping, there is nothing to do.
  2. If the Docker engine is not running, start Docker Desktop and wait.
  3. Run `docker compose up -d` in FREELLMAPI_COMPOSE_DIR (.env).
  4. Wait until /api/ping answers.

This is the same as typing the commands; the pipeline does not change.
"Stop" runs `docker compose down`. When the dashboard closes, it also runs
`docker compose down` (stop_now), unless a pipeline run is still active: a
run can outlive the dashboard and still needs the router.
"""
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DOCKER_START_TIMEOUT = 180   # Docker Desktop's engine can take minutes to come up
COMPOSE_TIMEOUT = 600        # the first `up` may pull the image
HEALTH_TIMEOUT = 90
PING_CACHE_SECONDS = 5
MAX_LOG_LINES = 40


def _settings() -> dict:
    """Read at call time from the environment (.env is loaded by src.dashboard.__main__)."""
    base = os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1").rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    return {
        "enabled": os.environ.get("LLM_PROVIDER", "openrouter").strip().lower() == "freellmapi",
        "compose_dir": os.environ.get("FREELLMAPI_COMPOSE_DIR", "").strip(),
        "root_url": root,
    }


def _no_window() -> dict:
    # Docker is a console program; without this every call flashes a window.
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def find_docker() -> str | None:
    found = shutil.which("docker")
    if not found or Path(found).suffix.lower() in (".cmd", ".bat"):
        return None
    return found


def find_docker_desktop() -> Path | None:
    if os.name != "nt":
        return None
    candidates = []
    docker = find_docker()
    if docker:
        # <install>\resources\bin\docker.exe -> <install>\Docker Desktop.exe
        candidates.append(Path(docker).resolve().parent.parent.parent / "Docker Desktop.exe")
    for var, sub in (("LOCALAPPDATA", ("Programs", "DockerDesktop")), ("ProgramFiles", ("Docker", "Docker"))):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var], *sub, "Docker Desktop.exe"))
    return next((c for c in candidates if c.is_file()), None)


class FreeLLMAPIManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self.state = "idle"     # idle | starting | running | stopping | stopped | error
        self.message = ""
        self.log: list[str] = []
        self._ping_cache = (0.0, False)

    # --- Status ---

    def healthy(self, use_cache: bool = True) -> bool:
        checked_at, ok = self._ping_cache
        if use_cache and time.time() - checked_at < PING_CACHE_SECONDS:
            return ok
        try:
            ok = requests.get(f"{_settings()['root_url']}/api/ping", timeout=2).ok
        except requests.RequestException:
            ok = False
        self._ping_cache = (time.time(), ok)
        return ok

    def status(self) -> dict:
        settings = _settings()
        busy = self._thread is not None and self._thread.is_alive()
        healthy = self.healthy() if settings["enabled"] else False
        state = self.state
        if not busy:
            # The router can be started or stopped outside the dashboard too.
            if healthy:
                state = "running"
            elif state == "running":
                state = "stopped"
        return {
            "enabled": settings["enabled"], "configured": bool(settings["compose_dir"]),
            "compose_dir": settings["compose_dir"], "url": settings["root_url"],
            "state": state, "healthy": healthy, "busy": busy, "message": self.message, "log": self.log[-12:],
        }

    def _note(self, text: str, state: str | None = None) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            if state:
                self.state = state
            self.message = text
            self.log.append(f"{stamp}  {text}")
            del self.log[:-MAX_LOG_LINES]
        print(f"[Dashboard] FreeLLMAPI: {text}")

    # --- Start / stop (each runs in a background thread) ---

    def start_async(self) -> bool:
        return self._spawn(self._start)

    def stop_async(self) -> bool:
        return self._spawn(self._stop)

    def stop_now(self) -> None:
        """Run `docker compose down` in this thread (used when the dashboard closes)."""
        self._guard(self._stop)()

    def _spawn(self, target) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._guard(target), daemon=True)
            self._thread.start()
        return True

    def _guard(self, target):
        def run():
            try:
                target()
            except Exception as e:  # report, never crash the dashboard
                self._note(f"Unexpected error: {e}", "error")
        return run

    def _docker(self, args: list[str], timeout: int, cwd: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([find_docker(), *args], cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL, **_no_window())

    def _engine_up(self) -> bool:
        try:
            return self._docker(["info", "--format", "{{.ServerVersion}}"], timeout=20).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _compose_dir(self) -> str | None:
        compose_dir = _settings()["compose_dir"]
        if not compose_dir:
            self._note("Set FREELLMAPI_COMPOSE_DIR in .env to the FreeLLMAPI folder (the one with "
                       "docker-compose.yml).", "error")
            return None
        if not (Path(compose_dir) / "docker-compose.yml").is_file():
            self._note(f"No docker-compose.yml in {compose_dir}. Check FREELLMAPI_COMPOSE_DIR in .env.", "error")
            return None
        return compose_dir

    def _start(self) -> None:
        with self._lock:
            self.state, self.message = "starting", "Checking whether the router and Docker are running..."
        if self.healthy(use_cache=False):
            self._note("The router is already running.", "running")
            return
        compose_dir = self._compose_dir()
        if not compose_dir:
            return
        if not find_docker():
            self._note("The docker command was not found. Install Docker Desktop.", "error")
            return

        if not self._engine_up():
            desktop = find_docker_desktop()
            if not desktop:
                self._note("The Docker engine is not running and Docker Desktop was not found. "
                           "Start Docker Desktop, then select Start router.", "error")
                return
            self._note("Starting Docker Desktop...", "starting")
            kwargs = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
            subprocess.Popen([str(desktop)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, **kwargs)
            deadline = time.time() + DOCKER_START_TIMEOUT
            while not self._engine_up():
                if time.time() > deadline:
                    self._note(f"The Docker engine did not start in {DOCKER_START_TIMEOUT} seconds. "
                               "Check Docker Desktop, then select Start router.", "error")
                    return
                time.sleep(3)
            self._note("The Docker engine is running.", "starting")

        self._note("Running docker compose up -d...", "starting")
        try:
            result = self._docker(["compose", "up", "-d"], timeout=COMPOSE_TIMEOUT, cwd=compose_dir)
        except subprocess.TimeoutExpired:
            self._note(f"docker compose up did not finish in {COMPOSE_TIMEOUT} seconds.", "error")
            return
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["no output"]
            self._note(f"docker compose up failed: {detail[0]}", "error")
            return

        deadline = time.time() + HEALTH_TIMEOUT
        while not self.healthy(use_cache=False):
            if time.time() > deadline:
                self._note(f"The containers started, but the router did not answer /api/ping in "
                           f"{HEALTH_TIMEOUT} seconds. Check docker compose logs.", "error")
                return
            time.sleep(2)
        self._note("The router is running.", "running")

    def _stop(self) -> None:
        compose_dir = self._compose_dir()
        if not compose_dir:
            return
        self._note("Running docker compose down...", "stopping")
        try:
            result = self._docker(["compose", "down"], timeout=120, cwd=compose_dir)
        except (OSError, subprocess.TimeoutExpired) as e:
            self._note(f"docker compose down did not finish: {e}", "error")
            return
        self._ping_cache = (0.0, False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["no output"]
            self._note(f"docker compose down failed: {detail[0]}", "error")
            return
        self._note("The router is stopped.", "stopped")
