"""
Starts and stops one pipeline run for the dashboard.

A run is the same `python run.py "<topic>"` a user would type, started as a
child process so the dashboard stays responsive and a crash in the pipeline
cannot take the dashboard down with it. Its combined stdout/stderr goes to
its own file in dashboard_files/consoles/, which the dashboard streams to
the browser and keeps, so a past run's console can be read again later.

The run is detached from the dashboard: it gets its own process group (and,
on Windows, its own hidden console), so Ctrl+C in the dashboard's terminal,
closing that terminal, or a dashboard crash does not stop the pipeline. The
run keeps writing to its console log after the dashboard exits. Stop is the
only way the dashboard ends a run, and only when the user asks.

Only one run at a time: every agent writes to the same logs/, data/ and
replication/ paths, so two concurrent runs would overwrite each other.
"""
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.dashboard.readers import open_shared, read_text

REPO_ROOT = Path(__file__).parent.parent.parent
CONSOLE_DIR = REPO_ROOT / "dashboard_files" / "consoles"

MAX_TOPIC_CHARS = 500


def _detached_popen_kwargs() -> list[dict]:
    """Popen options that separate the run from the dashboard, most separate first."""
    if os.name != "nt":
        return [{"start_new_session": True}]
    base = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    # Breaking away from the dashboard's job object (some terminals put
    # their processes in one and kill it on close) is refused by jobs that
    # do not allow it, so fall back to the plain detached launch.
    return [{"creationflags": base | subprocess.CREATE_BREAKAWAY_FROM_JOB}, {"creationflags": base}]


def _complete_utf8(data: bytes) -> bytes:
    """Drop a multi-byte UTF-8 character cut off at the end of a read; it is
    read again, whole, on the next poll."""
    trailing = 0
    while trailing < 3 and trailing < len(data) and (data[-1 - trailing] & 0xC0) == 0x80:
        trailing += 1
    lead_pos = len(data) - 1 - trailing
    if lead_pos >= 0 and data[lead_pos] >= 0xC0:
        needed = 2 if data[lead_pos] < 0xE0 else 3 if data[lead_pos] < 0xF0 else 4
        if trailing + 1 < needed:
            return data[:lead_pos]
    return data


class RunError(Exception):
    """A run could not be started or stopped; the message is shown to the user."""


def validate_topic(topic: str) -> str:
    if not isinstance(topic, str):
        raise RunError("Enter a research topic as text.")
    topic = " ".join((topic or "").split())
    if not topic:
        raise RunError("Enter a research topic first.")
    if len(topic) > MAX_TOPIC_CHARS:
        raise RunError(f"The topic is too long (maximum {MAX_TOPIC_CHARS} characters).")
    if topic.startswith("-"):
        # src/main.py treats any argument that starts with "--" as a flag.
        raise RunError("The topic must not start with '-'.")
    return topic


class PipelineRunner:
    def __init__(self, console_log: Path | None = None, command_prefix: list | None = None,
                 console_dir: Path = CONSOLE_DIR):
        # Each run writes a new console file in console_dir, unless a fixed
        # console_log is given (tests).
        self._fixed_console_log = console_log
        self.console_dir = console_dir
        self.console_log = console_log
        # The command a run executes, before its own arguments. Tests replace
        # it with a short stand-in script.
        self.command_prefix = command_prefix or [sys.executable, "-u", str(REPO_ROOT / "run.py")]
        self._lock = threading.Lock()
        self._process = None
        self._log_handle = None
        self.topic = ""
        self.started_at = None
        self.finished_at = None
        self.stopped_by_user = False
        self.run_id = None  # the run database id, once the pipeline has opened its record

    def is_running(self) -> bool:
        with self._lock:
            return self._poll_locked()

    def _poll_locked(self) -> bool:
        if self._process is None:
            return False
        if self._process.poll() is None:
            return True
        if self.finished_at is None:
            self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
        return False

    def start(self, topic: str, reset: bool = False, parent_paper_input: Path | None = None) -> None:
        topic = validate_topic(topic)
        with self._lock:
            if self._poll_locked():
                raise RunError("A pipeline run is already in progress. Stop it or wait for it to finish.")

            args = list(self.command_prefix)
            if reset:
                args.append("--reset")
            if parent_paper_input is not None:
                from src.paper_input import load_input, PaperInputError
                try:
                    load_input(parent_paper_input, topic)
                except PaperInputError as error:
                    raise RunError(str(error)) from error
                args.extend(["--parent-paper-input", str(Path(parent_paper_input).resolve())])
            args.append(topic)

            env = dict(os.environ)
            # Agents print non-ASCII characters; a redirected Windows console
            # would otherwise encode with cp1252 and crash on them.
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"

            if self._fixed_console_log is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                self.console_log = self.console_dir / f"{stamp}.log"
            self.console_log.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.console_log, "w", encoding="utf-8")
            error = None
            for kwargs in _detached_popen_kwargs():
                try:
                    self._process = subprocess.Popen(
                        args, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=self._log_handle, stderr=subprocess.STDOUT, **kwargs,
                    )
                    break
                except OSError as e:
                    error = e
            else:
                self._log_handle.close()
                self._log_handle = None
                raise RunError(f"Could not start the pipeline: {error}") from error

            self.topic = topic
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.finished_at = None
            self.stopped_by_user = False
            self.run_id = None

    def stop(self) -> None:
        with self._lock:
            if not self._poll_locked():
                raise RunError("No pipeline run is in progress.")
            self.stopped_by_user = True
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._poll_locked()

    def info(self) -> dict:
        with self._lock:
            running = self._poll_locked()
            exit_code = None if self._process is None or running else self._process.returncode
            return {
                "running": running,
                "topic": self.topic,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "exit_code": exit_code,
                "stopped_by_user": self.stopped_by_user,
            }

    def read_console(self, offset: int = 0, max_bytes: int = 200_000, path: Path | None = None) -> dict:
        """Console output from byte `offset` on, so the browser only fetches new
        text. `path` reads an older run's console instead of this run's."""
        path = path or self.console_log
        try:
            size = path.stat().st_size
        except (OSError, AttributeError):
            return {"text": "", "offset": 0, "reset": offset > 0}
        reset = offset > size  # a different console file than the browser had
        if reset:
            offset = 0
        with open_shared(path) as f:
            f.seek(offset)
            data = f.read(max_bytes)
        data = _complete_utf8(data)
        return {"text": data.decode("utf-8", errors="replace"), "offset": offset + len(data), "reset": reset}

    def console_text(self) -> str:
        """The whole console log (only used to find the attempt now running)."""
        try:
            return read_text(self.console_log)
        except (OSError, TypeError):
            return ""
