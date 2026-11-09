"""
Read-only access to the pipeline's outputs.

The dashboard and the pipeline are separate programs. The pipeline does not
know the dashboard exists; the dashboard only reads files the pipeline
already writes. Three rules keep the dashboard from getting in its way:

1. Files are opened with full sharing (open_shared). On Windows a normal
   open() blocks other processes from deleting or renaming the file, so a
   dataset download in the browser could make the Replication Agent's
   shutil.rmtree fail. A shared handle lets the pipeline delete or replace
   the file at any time.

2. The run database is opened read-only (mode=ro), never through
   run_store._connect, which creates tables and so takes a write lock. A
   read-only connection cannot block the pipeline's writes for more than
   the few milliseconds a SELECT takes.

3. A JSON file read halfway through a pipeline write (the pipeline writes
   in place, not atomically) is not an error: the last good copy is used.
"""
import hashlib
import io
import json
import os
import re
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                      wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_ALL = 0x1 | 0x2 | 0x4  # read | write | delete
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _INVALID_HANDLE = wintypes.HANDLE(-1).value


def open_shared(path: Path):
    """Open a file for binary reading without stopping other processes from
    writing, renaming or deleting it."""
    if os.name != "nt":
        return open(path, "rb")  # POSIX opens never block unlink/rename
    handle = _kernel32.CreateFileW(str(path), _GENERIC_READ, _FILE_SHARE_ALL, None,
                                   _OPEN_EXISTING, _FILE_ATTRIBUTE_NORMAL, None)
    if handle == _INVALID_HANDLE or handle is None:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(fd, "rb")


def read_text(path: Path) -> str:
    with open_shared(path) as f:
        return f.read().decode("utf-8", errors="replace")


def open_text(path: Path):
    return io.TextIOWrapper(open_shared(path), encoding="utf-8", errors="replace", newline="")


_last_good_json: dict = {}


def read_json(path: Path, default):
    """The file's JSON, or its last good copy if it is being rewritten now."""
    key = str(path)
    try:
        value = json.loads(read_text(path))
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError):
        return _last_good_json.get(key, default)
    _last_good_json[key] = value
    return value


# --- Run database (src/orchestration/run_store.py writes it) ---

def _connect_read_only(db_file: Path) -> sqlite3.Connection | None:
    if not db_file.is_file():
        return None
    conn = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def list_runs(db_file: Path, limit: int = 50) -> list[dict]:
    conn = _connect_read_only(db_file)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT run_id, topic, started_at, finished_at, status, final_phase, parent_paper_title "
            "FROM runs ORDER BY started_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def latest_run(db_file: Path) -> dict | None:
    """The newest run and its recorded agent attempts, oldest attempt first."""
    conn = _connect_read_only(db_file)
    if conn is None:
        return None
    try:
        run = conn.execute(
            "SELECT run_id, topic, started_at, finished_at, status, final_phase FROM runs "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if not run:
            return None
        stages = conn.execute(
            "SELECT agent, attempt, decision, report_text, recorded_at FROM stages "
            "WHERE run_id = ? ORDER BY id", (run["run_id"],),
        ).fetchall()
        return {**dict(run), "stages": [dict(s) for s in stages]}
    finally:
        conn.close()


# --- Which agent is running now ---

# src/agents/master_agent.py prints this line as each attempt starts.
_RUNNING_LINE = re.compile(r"\[Master Agent\] Running (.+?) \(attempt (\d+)/(\d+)\)")
_STATUS_BY_DECISION = {"Approved": "approved", "Retrying": "retrying", "Rejected": "rejected"}
SUMMARY_MAX_CHARS = 400


def _summary(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= SUMMARY_MAX_CHARS else text[:SUMMARY_MAX_CHARS].rstrip() + " ..."


def _parse_time(iso: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(iso) if iso else None
    except ValueError:
        return None


def last_activity(run: dict) -> datetime | None:
    times = [_parse_time(s["recorded_at"]) for s in run.get("stages", [])] + [_parse_time(run.get("started_at"))]
    times = [t for t in times if t]
    return max(times) if times else None


def hours_since(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds() / 3600


def agent_progress(run: dict | None, pipeline_order: list[str], console_text: str = "",
                   run_active: bool = False) -> dict:
    """
    Per-agent status for one run, built only from what the pipeline already
    records: each finished attempt is a row in the run database, and the
    attempt now running is the last "[Master Agent] Running ..." console
    line (when the dashboard has the console), else the next agent in
    pipeline order after the last recorded attempt.
    """
    if not run:
        return {}
    progress = {}
    previous_time = run.get("started_at")
    for stage in run.get("stages", []):
        progress[stage["agent"]] = {
            "status": _STATUS_BY_DECISION.get(stage["decision"], stage["decision"].lower()),
            "attempt": stage["attempt"], "max_attempts": None,
            "started_at": previous_time, "finished_at": stage["recorded_at"],
            "summary": _summary(stage["report_text"]),
        }
        previous_time = stage["recorded_at"]

    if not run_active:
        return progress

    recorded = {(s["agent"], s["attempt"]) for s in run.get("stages", [])}
    running = None
    matches = _RUNNING_LINE.findall(console_text or "")
    if matches:
        name, attempt, max_attempts = matches[-1]
        if (name, int(attempt)) not in recorded:
            running = (name, int(attempt), int(max_attempts))
    elif run.get("stages"):
        last = run["stages"][-1]
        if last["decision"] == "Retrying":
            running = (last["agent"], last["attempt"] + 1, None)
        elif last["decision"] == "Approved" and last["agent"] in pipeline_order:
            i = pipeline_order.index(last["agent"]) + 1
            if i < len(pipeline_order):
                running = (pipeline_order[i], 1, None)
    elif pipeline_order:
        running = (pipeline_order[0], 1, None)

    if running:
        name, attempt, max_attempts = running
        progress[name] = {"status": "running", "attempt": attempt, "max_attempts": max_attempts,
                          "started_at": previous_time, "finished_at": None, "summary": ""}
    return progress


# --- Files each agent produced (run_store's artifacts table) ---

def get_run(db_file: Path, run_id: str) -> dict | None:
    conn = _connect_read_only(db_file)
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT run_id, topic, started_at, finished_at, status, final_phase, parent_paper_title "
                           "FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_artifacts(db_file: Path, run_id: str) -> list[dict]:
    conn = _connect_read_only(db_file)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT id, run_id, agent, kind, original_path, sha256, size_bytes, content_stored, recorded_at "
            "FROM artifacts WHERE run_id = ? ORDER BY id", (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def stage_outputs(db_file: Path, run_id: str) -> dict:
    """Each agent's last recorded structured output in one run."""
    conn = _connect_read_only(db_file)
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT agent, output_json FROM stages WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    finally:
        conn.close()
    outputs = {}
    for row in rows:
        if row["output_json"]:
            try:
                outputs[row["agent"]] = json.loads(row["output_json"])
            except json.JSONDecodeError:
                continue
    return outputs


def file_matches(path: Path, sha256_hex: str, max_bytes: int = 20 * 1024 * 1024) -> str:
    """Whether the file on disk is still the version a run recorded:
    "same", "changed", "missing", or "unknown" (too large to check here)."""
    try:
        if path.stat().st_size > max_bytes:
            return "unknown"
        digest = hashlib.sha256()
        with open_shared(path) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unknown"
    return "same" if digest.hexdigest() == sha256_hex else "changed"


def stage_attempts(db_file: Path, run_id: str) -> list[dict]:
    """Every agent attempt in one run, in order, with its report and decoded output."""
    conn = _connect_read_only(db_file)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT agent, attempt, decision, output_json, report_text, recorded_at FROM stages "
            "WHERE run_id = ? ORDER BY id", (run_id,),
        ).fetchall()
    finally:
        conn.close()
    attempts = []
    for row in rows:
        try:
            output = json.loads(row["output_json"]) if row["output_json"] else None
        except json.JSONDecodeError:
            output = None
        attempts.append({"agent": row["agent"], "attempt": row["attempt"], "decision": row["decision"],
                         "report_text": row["report_text"] or "", "recorded_at": row["recorded_at"],
                         "output": output})
    return attempts


class _BlobReader(io.RawIOBase):
    """Reads a zlib-compressed blob a piece at a time, so a large recorded
    dataset can be previewed or downloaded without decompressing all of it."""

    def __init__(self, compressed: bytes):
        self._source = io.BytesIO(compressed)
        self._inflate = zlib.decompressobj()
        self._pending = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending:
            chunk = self._source.read(64 * 1024)
            if not chunk:
                self._pending = self._inflate.flush()
                if not self._pending:
                    return 0
                break
            self._pending = self._inflate.decompress(chunk)
        n = min(len(buffer), len(self._pending))
        buffer[:n] = self._pending[:n]
        self._pending = self._pending[n:]
        return n


def open_blob(db_file: Path, sha256_hex: str):
    """A binary file object over one stored blob, or None if it is not stored."""
    conn = _connect_read_only(db_file)
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT content FROM blobs WHERE sha256 = ?", (sha256_hex,)).fetchone()
    finally:
        conn.close()
    return io.BufferedReader(_BlobReader(row["content"])) if row else None


def get_artifact_record(db_file: Path, artifact_id: int) -> dict | None:
    """One artifact's record, without its contents."""
    conn = _connect_read_only(db_file)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT id, run_id, agent, kind, original_path, sha256, size_bytes, content_stored "
            "FROM artifacts WHERE id = ?", (artifact_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
