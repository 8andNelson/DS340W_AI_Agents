"""
Run database: a permanent record of every pipeline run.

logs/project_state.json only holds the *current* run, and the files under
data/, replication/ and experiments/ are overwritten by the next run. This
module keeps a copy of each run in one SQLite file (database/project_runs.db,
gitignored), so any past run can be reviewed or restored later.

Evidence tables:

  runs       -- one row per `python run.py` invocation: topic, start/finish
                time, final status and phase, LLM provider, the project's own
                git commit, and the approved Parent Paper.
  stages     -- one row per agent attempt (pass or fail): the agent's full
                structured output as JSON, the Master Agent's decision, and
                the human-readable report text.
  stage_timings -- one durable start/finish and monotonic duration per
                invocation, independent of approval and completion order.
  artifacts  -- one row per file a stage produced (master CSV, splits, each
                replication/experiment script, stdout/stderr, ENVIRONMENT.md,
                and later the report and slides).
  blobs      -- file contents, zlib-compressed and keyed by SHA-256, so a file
                that is identical across runs is stored only once.

Two more tables hold each finished run's measurements, written by
src/orchestration/run_metrics.py when the run finishes:

  run_conditions -- one row per run: the conditions it ran under (topic,
                    Parent Paper, data, code version, models, memory).
  run_metrics    -- one row per metric value (result quality, reliability,
                    cost and time, memory use), so runs can be compared.

A file larger than MAX_BLOB_BYTES is recorded by path, size and hash only
(content_stored = 0), so one huge raw download cannot inflate the database.

Every public function fails soft, like Slack posting: a database error is
printed and swallowed, so it never changes pipeline behavior.
"""
import json
import platform
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from src.orchestration import run_context

REPO_ROOT = Path(__file__).parent.parent.parent
DB_FILE = REPO_ROOT / "database" / "project_runs.db"
_write_lock = threading.RLock()  # timers must not time out behind another agent's artifact transaction

MAX_BLOB_BYTES = 200 * 1024 * 1024  # larger files are recorded by hash only

# Files kept from a replication/experiment directory: source code, configs,
# notebooks and run logs. venv/ and .git/ are skipped, and so are data copies
# (the master CSV and splits are stored once, from the Cleaning Agent stage).
SOURCE_SUFFIXES = {
    ".py", ".ipynb", ".r", ".sh", ".md", ".txt", ".cfg", ".toml", ".yml", ".yaml", ".json", ".ini",
}
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", "data", "node_modules"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    topic               TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,  -- running | success | halted | crashed
    final_phase         TEXT,
    llm_provider        TEXT,
    project_git_commit  TEXT,
    parent_paper_title  TEXT,
    parent_paper_doi    TEXT,
    parent_paper_json   TEXT,
    result_json         TEXT
);
CREATE TABLE IF NOT EXISTS stages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    agent        TEXT NOT NULL,
    attempt      INTEGER NOT NULL,
    decision     TEXT NOT NULL,     -- Approved | Retrying | Rejected
    output_json  TEXT,
    report_text  TEXT,
    recorded_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (
    sha256      TEXT PRIMARY KEY,
    size_bytes  INTEGER NOT NULL,
    content     BLOB NOT NULL       -- zlib-compressed
);
CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    stage_id        INTEGER REFERENCES stages(id),
    agent           TEXT NOT NULL,
    kind            TEXT NOT NULL,  -- the output key it came from, e.g. master_csv_path, script
    original_path   TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    content_stored  INTEGER NOT NULL,
    recorded_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stages_run ON stages(run_id);
CREATE TABLE IF NOT EXISTS stage_timings (
    invocation_id    TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    agent            TEXT NOT NULL,
    attempt          INTEGER NOT NULL,
    max_attempts     INTEGER NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    duration_seconds REAL CHECK (duration_seconds >= 0),
    UNIQUE (run_id, agent, attempt)
);
CREATE INDEX IF NOT EXISTS idx_stage_timings_run ON stage_timings(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE TABLE IF NOT EXISTS run_conditions (
    run_id               TEXT NOT NULL REFERENCES runs(run_id),
    metrics_version      INTEGER NOT NULL,
    status               TEXT,     -- success | halted | crashed (a stale "running" run is crashed)
    contributor          TEXT,
    started_at           TEXT,
    finished_at          TEXT,
    topic                TEXT,
    topic_key            TEXT,     -- normalized topic, groups runs on the same task
    parent_paper_doi     TEXT,
    parent_paper_title   TEXT,
    dataset_identities   TEXT,     -- JSON list of resolved dataset links
    rows_used            INTEGER,
    sampled              INTEGER,
    code_source          TEXT,     -- cloned_repo | reconstructed
    model_family         TEXT,     -- JSON list
    git_commit           TEXT,
    git_dirty            INTEGER,
    llm_provider         TEXT,
    served_models        TEXT,     -- JSON {model: successful calls}
    memory_mode          TEXT,
    memory_backend       TEXT,
    memory_reuse         INTEGER,
    memory_tuning        INTEGER,
    memory_curator       INTEGER,
    memory_exposure      TEXT,     -- none | partial | full
    memory_bank_size     TEXT,     -- JSON {agent: visible memories}
    memory_bank_snapshot TEXT,
    comparison_group     TEXT,
    seeds                TEXT,     -- JSON
    python_version       TEXT,
    os                   TEXT,
    log_matching         TEXT,     -- JSON {log: run_id | time_span}
    timing_method        TEXT,
    notes                TEXT,     -- JSON list of caveats about this run's measurements
    computed_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, metrics_version)
);
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    metrics_version  INTEGER NOT NULL,
    scope            TEXT NOT NULL,  -- "run" or an agent name
    metric           TEXT NOT NULL,
    value_num        REAL,
    value_text       TEXT,
    unit             TEXT,
    missing_reason   TEXT,           -- set when the value is null; a missing value is never 0
    computed_at      TEXT NOT NULL,
    PRIMARY KEY (run_id, metrics_version, scope, metric)
);
CREATE INDEX IF NOT EXISTS idx_run_metrics_metric ON run_metrics(metric, scope);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def _warn(action: str, error: Exception) -> None:
    print(f"[Run Store] WARNING: could not {action}: {error}")


def _project_git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _project_git_dirty() -> bool | None:
    """True when tracked files differ from the commit; None when unknown."""
    try:
        proc = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10)
        return bool(proc.stdout.strip()) if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _to_json(value) -> str:
    return json.dumps(value, default=str)


# --- Writing ---

def start_run(topic: str, llm_provider: str = "") -> str | None:
    """Open a new run row. Returns its run_id, or None if the database is unavailable."""
    try:  # an earlier run killed before its `finally` block: resolve its lineage now
        from src.memory import events
        events.recover_unfinished()
    except Exception as e:
        _warn("recover unfinished runs in the memory store", e)
    _sync_memory()  # writes left over from an earlier run that could not reach the hosted store
    run_id =datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, topic, started_at, status, llm_provider, project_git_commit) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (run_id, topic, _now(), llm_provider, _project_git_commit()),
            )
        run_context.set_run(run_id, git_dirty=_project_git_dirty(), python_version=platform.python_version(),
                            os=platform.platform())
        return run_id
    except sqlite3.Error as e:
        _warn("start a run record", e)
        return None


def record_stage(run_id: str | None, agent: str, attempt: int, decision: str,
                 output, report_text: str = "") -> int | None:
    """Store one agent attempt and every file its output points to. Returns the stage id."""
    if not run_id:
        return None
    try:
        with _write_lock, _connect() as conn:
            cur = conn.execute(
                "INSERT INTO stages (run_id, agent, attempt, decision, output_json, report_text, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, agent, attempt, decision, _to_json(output), report_text, _now()),
            )
            stage_id = cur.lastrowid
            for kind, path in collect_artifact_paths(output):
                _store_file(conn, run_id, stage_id, agent, kind, path)
        return stage_id
    except (sqlite3.Error, OSError) as e:
        _warn(f"record the {agent} stage", e)
        return None


def last_stage_attempt(run_id: str | None, agent: str) -> int:
    """Keep invocation numbers unique across ordinary, rerun and Slack requests."""
    if not run_id:
        return 0
    try:
        with _connect() as conn:
            row = conn.execute("SELECT MAX(attempt) FROM (SELECT attempt FROM stages WHERE run_id=? AND agent=? "
                               "UNION ALL SELECT attempt FROM stage_timings WHERE run_id=? AND agent=?)",
                               (run_id, agent, run_id, agent)).fetchone()
        return row[0] or 0
    except (sqlite3.Error, OSError) as error:
        _warn(f"read the {agent} invocation number", error)
        return 0


def start_stage_attempt(run_id: str | None, agent: str, attempt: int, max_attempts: int) -> str | None:
    """Persist a start before invoking an agent, including attempts that crash.

    Completion/approval remains in stages; this record measures execution only.
    """
    if not run_id:
        return None
    invocation_id = str(uuid.uuid4())
    try:
        with _write_lock, _connect() as conn:
            conn.execute("INSERT INTO stage_timings (invocation_id, run_id, agent, attempt, max_attempts, started_at) "
                         "VALUES (?, ?, ?, ?, ?, ?)",
                         (invocation_id, run_id, agent, attempt, max_attempts,
                          datetime.now(timezone.utc).isoformat(timespec="microseconds")))
        return invocation_id
    except (sqlite3.Error, OSError) as error:
        _warn(f"start the {agent} attempt timer", error)
        return None


def finish_stage_attempt(invocation_id: str | None, duration_seconds: float) -> None:
    """Close an execution timer once; a missing finish stays unknown."""
    if not invocation_id:
        return
    finished_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    try:
        with _write_lock, _connect() as conn:
            conn.execute("UPDATE stage_timings SET finished_at = ?, duration_seconds = ? "
                         "WHERE invocation_id = ? AND finished_at IS NULL",
                         (finished_at, max(0.0, duration_seconds), invocation_id))
    except (sqlite3.Error, OSError) as error:
        _warn("finish an agent attempt timer", error)


@contextmanager
def stage_timer(run_id: str | None, agent: str, attempt: int, max_attempts: int):
    """Time the invocation body, excluding reporting, approval and diagnosis.

    UTC timestamps support live displays; a monotonic clock measures elapsed
    seconds even if the system clock changes. Persisting the finish is outside
    the measured interval. A killed process leaves an unfinished start record.
    """
    invocation_id = start_stage_attempt(run_id, agent, attempt, max_attempts)
    started = time.monotonic()
    try:
        yield invocation_id
    finally:
        elapsed = time.monotonic() - started
        finish_stage_attempt(invocation_id, elapsed)


def record_artifact(run_id: str | None, agent: str, kind: str, path) -> None:
    """Store one file directly. For agents whose files are not named in their
    output dict (for example a future Report or Presentation Agent)."""
    if not run_id:
        return
    try:
        with _write_lock, _connect() as conn:
            _store_file(conn, run_id, None, agent, kind, Path(path))
    except (sqlite3.Error, OSError) as e:
        _warn(f"record artifact {path}", e)


def finish_run(run_id: str | None, status: str, final_phase: str = "",
               parent_paper: dict | None = None, result: dict | None = None) -> None:
    if not run_id:
        return
    parent_paper = parent_paper or {}
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, status = ?, final_phase = ?, parent_paper_title = ?, "
                "parent_paper_doi = ?, parent_paper_json = ?, result_json = ? WHERE run_id = ?",
                (_now(), status, final_phase, parent_paper.get("title"), parent_paper.get("doi"),
                 _to_json(parent_paper) if parent_paper else None,
                 _to_json(result) if result is not None else None, run_id),
            )
    except sqlite3.Error as e:
        _warn("finish the run record", e)
    # Record the run's end and each agent attempt's lineage (kept, superseded,
    # reverted, rejected, unknown) in the memory store; fail-soft.
    try:
        from src.memory import events
        events.run_finished(run_id, status)
    except Exception as e:
        _warn(f"record the end of run {run_id} in the memory store", e)
    # Measure the run (success, halted or crashed) from what was just recorded.
    # Fail-soft: `python -m src.orchestration.run_metrics compute <run_id>`
    # can make the same measurements later.
    try:
        from src.orchestration import run_metrics
        run_metrics.compute(run_id)
    except Exception as e:
        _warn(f"compute the metrics of run {run_id}", e)
    run_context.clear(run_id)
    _sync_memory()


def _sync_memory() -> None:
    """Send the memory outbox and run metrics to the team's hosted store
    (only when MEMORY_BACKEND=postgres; fail-soft, see src/memory)."""
    try:
        from src import memory
        if memory.backend_name() == "postgres":
            report = memory.sync()
            if report.get("error"):
                print(f"[Memory] Sync incomplete: {report['error']}")
    except Exception as e:
        _warn("sync the memory store", e)


def collect_artifact_paths(output) -> list[tuple[str, Path]]:
    """
    Find the files an agent's output dict points to, by key name:
      - any key ending in "_path" (master_csv_path, stdout_path, ...),
        including inside lists of dicts (Data Agent's selected_datasets);
      - "replication_dir" / "experiment_dir", walked for source files
        (scripts, configs, ENVIRONMENT.md), skipping SKIP_DIRS.
    Paths that do not exist on disk are skipped. A path is listed once.
    """
    found: dict[Path, str] = {}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item:
                    if key.endswith("_path"):
                        path = Path(item)
                        if path.is_file():
                            found.setdefault(path.resolve(), key)
                    elif key in ("replication_dir", "experiment_dir"):
                        for path in _walk_source_files(Path(item)):
                            found.setdefault(path.resolve(), "script" if path.suffix == ".py" else "source_file")
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(output)
    return [(kind, path) for path, kind in found.items()]


def _walk_source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
            files.append(path)
    return files


def _store_file(conn: sqlite3.Connection, run_id: str, stage_id: int | None,
                agent: str, kind: str, path: Path) -> None:
    size = path.stat().st_size
    digest = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    file_hash = digest.hexdigest()

    stored = size <= MAX_BLOB_BYTES
    if stored:
        exists = conn.execute("SELECT 1 FROM blobs WHERE sha256 = ?", (file_hash,)).fetchone()
        if not exists:
            conn.execute("INSERT INTO blobs (sha256, size_bytes, content) VALUES (?, ?, ?)",
                         (file_hash, size, zlib.compress(path.read_bytes(), 6)))

    conn.execute(
        "INSERT INTO artifacts (run_id, stage_id, agent, kind, original_path, sha256, size_bytes, "
        "content_stored, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, stage_id, agent, kind, str(path), file_hash, size, int(stored), _now()),
    )


# --- Reading ---

def list_runs(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT run_id, topic, started_at, finished_at, status, final_phase, parent_paper_title "
            "FROM runs ORDER BY started_at DESC LIMIT ?", (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not run:
            return None
        stages = conn.execute(
            "SELECT id, agent, attempt, decision, recorded_at FROM stages WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        artifacts = conn.execute(
            "SELECT id, agent, kind, original_path, size_bytes, content_stored FROM artifacts "
            "WHERE run_id = ? ORDER BY id", (run_id,),
        ).fetchall()
        timings = conn.execute("SELECT * FROM stage_timings WHERE run_id = ? ORDER BY started_at, invocation_id",
                               (run_id,)).fetchall()
    return {**dict(run), "stages": [dict(s) for s in stages], "artifacts": [dict(a) for a in artifacts],
            "stage_timings": [dict(t) for t in timings]}


def get_stage_output(run_id: str, agent: str) -> object:
    """The last recorded output of one agent in one run, decoded from JSON."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT output_json FROM stages WHERE run_id = ? AND agent = ? ORDER BY id DESC LIMIT 1",
            (run_id, agent),
        ).fetchone()
    return json.loads(row["output_json"]) if row and row["output_json"] else None


def read_artifact(artifact_id: int) -> bytes | None:
    """The stored bytes of one artifact, or None if only its hash was recorded."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT b.content FROM artifacts a JOIN blobs b ON a.sha256 = b.sha256 WHERE a.id = ?",
            (artifact_id,),
        ).fetchone()
    return zlib.decompress(row["content"]) if row else None


# --- CLI: python -m src.orchestration.run_store list | show <run_id> | export <artifact_id> <dest> ---

def _main(argv: list[str]) -> int:
    if not argv or argv[0] == "list":
        for r in list_runs():
            print(f"{r['run_id']}  {r['status']:<8} {r['final_phase'] or '':<22} {r['topic'][:50]}")
            if r["parent_paper_title"]:
                print(f"    Parent Paper: {r['parent_paper_title'][:90]}")
        return 0
    if argv[0] == "show" and len(argv) == 2:
        run = get_run(argv[1])
        if not run:
            print(f"No run with id {argv[1]}")
            return 1
        print(f"Run {run['run_id']} -- {run['status']} at {run['final_phase']}")
        print(f"  Topic: {run['topic']}")
        print(f"  Started/finished: {run['started_at']} / {run['finished_at']}")
        print(f"  Project commit: {run['project_git_commit'] or 'unknown'}  LLM provider: {run['llm_provider']}")
        print(f"  Parent Paper: {run['parent_paper_title'] or 'none approved'}")
        print("  Stages:")
        for s in run["stages"]:
            print(f"    {s['agent']} attempt {s['attempt']}: {s['decision']}")
        print("  Artifacts (id, agent, kind, size, path):")
        for a in run["artifacts"]:
            note = "" if a["content_stored"] else "  [hash only, not stored]"
            print(f"    {a['id']:>5}  {a['agent']:<20} {a['kind']:<20} {a['size_bytes']:>10}  {a['original_path']}{note}")
        return 0
    if argv[0] == "export" and len(argv) == 3:
        content = read_artifact(int(argv[1]))
        if content is None:
            print(f"Artifact {argv[1]} has no stored content.")
            return 1
        Path(argv[2]).write_bytes(content)
        print(f"Wrote {len(content)} bytes to {argv[2]}")
        return 0
    print("Usage: python -m src.orchestration.run_store [list | show <run_id> | export <artifact_id> <dest>]")
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
