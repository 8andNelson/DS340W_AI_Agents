"""
Shared agent memory: the store and its fail-soft API (milestone M2 of
docs/CLAUDE_BACKEND_PLAN.md; sections 4-6 and 11.2).

This module is the only code that touches the memory store. Agents call
it; it never calls an agent. Every public function fails soft, like Slack
posting: an error is printed and swallowed, and the pipeline runs as if
there were no memory.

MEMORY_BACKEND (.env):
  sqlite    (default) the whole store is the local file
            database/memory_local.db.
  postgres  the team's hosted store (MEMORY_DATABASE_URL). Every write is
            applied to the local file and put in its outbox in one
            transaction; sync() sends the outbox and the run metrics, and
            refreshes the local copy. A read goes to the hosted store and
            caches the result; while the store is unreachable, reads use
            the local copy if it synced in the last CACHE_MAX_AGE, else
            return nothing.
  off       no memory; every call returns its empty value.

Memory records are append-only: no function deletes or overwrites one.
Only counters change after insert (and, from milestone M4, the state). A
secret scrubber (scrub.py) runs on every row before it is written.

Public API:
  put_memory(memory) -> id | None        get_memory(id) -> dict | None
  list_memories(agent, kind, limit)      record_usage(usage) -> id | None
  increment(memory_id, **counters)       sync() -> report     status() -> dict
"""
import hashlib
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import MEMORY_BACKEND, MEMORY_CONTRIBUTOR, MEMORY_DATABASE_URL
from src.memory import schema
from src.memory.backends.base import RemoteUnavailable
from src.memory.backends.sqlite import SqliteStore
from src.memory.scrub import scrub as _scrub

REPO_ROOT = Path(__file__).parent.parent.parent
LOCAL_DB = REPO_ROOT / "database" / "memory_local.db"

BACKENDS = ("off", "sqlite", "postgres")
CACHE_MAX_AGE = timedelta(days=7)       # an older local copy is not used for reads (quarantines may be missing)
MAX_SEND_ATTEMPTS = 5                   # an op rejected this often is set aside as "failed", for review
REMOTE_RETRY_SECONDS = 300              # after a connection failure, reads skip the hosted store this long

_client = None
_client_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _warn(action: str, error: Exception) -> None:
    print(f"[Memory] WARNING: could not {action}: {error}")


class MemoryClient:
    def __init__(self, backend: str, local_path: Path = None, remote=None):
        if backend not in BACKENDS:
            raise ValueError(f"MEMORY_BACKEND must be one of {', '.join(BACKENDS)}, not '{backend}'")
        self.backend = backend
        self.local = SqliteStore(local_path or LOCAL_DB) if backend != "off" else None
        if remote is None and backend == "postgres":
            from src.memory.backends.postgres import PostgresStore
            remote = PostgresStore(MEMORY_DATABASE_URL)
        self.remote = remote
        self._remote_down_until = 0.0
        self.last_remote_error = ""

    # --- Remote availability ---

    def _remote_usable(self) -> bool:
        return self.remote is not None and time.monotonic() >= self._remote_down_until

    def _mark_down(self, error: Exception) -> None:
        self.last_remote_error = str(error)
        self._remote_down_until = time.monotonic() + REMOTE_RETRY_SECONDS
        _warn("reach the hosted memory store (using the local copy)", error)

    def cache_usable(self) -> bool:
        last = self.local.meta_get("last_sync") if self.local else None
        if not last:
            return False
        return datetime.now(timezone.utc) - datetime.fromisoformat(last) <= CACHE_MAX_AGE

    # --- Writes and reads ---

    def write(self, op: dict) -> None:
        if self.backend == "sqlite":
            self.local.apply(op)
        elif self.backend == "postgres":
            self.local.apply_and_enqueue(op)

    def fetch(self, **filters) -> list[dict]:
        if self.backend == "off":
            return []
        if self.backend == "sqlite":
            return self.local.fetch_memories(**filters)
        if self._remote_usable():
            try:
                rows = self.remote.fetch_memories(**filters)
                self.local.cache_memories(rows)
                return rows
            except RemoteUnavailable as e:
                self._mark_down(e)
        return self.local.fetch_memories(**filters) if self.cache_usable() else []

    # --- Sync ---

    def _queue_run_metrics(self) -> int:
        """Put the run database's measured runs that were not sent yet in the outbox."""
        from src.orchestration import run_store

        if not Path(run_store.DB_FILE).exists():
            return 0
        queued = self.local.queued_runs()
        count = 0
        with run_store._connect() as conn:
            conditions = conn.execute("SELECT * FROM run_conditions ORDER BY started_at").fetchall()
            for cond in conditions:
                key = (cond["run_id"], cond["metrics_version"])
                if key in queued:
                    continue
                metrics = conn.execute("SELECT * FROM run_metrics WHERE run_id = ? AND metrics_version = ?",
                                       key).fetchall()
                columns = schema.TABLES["run_conditions"]
                ops = [{"op_id": str(uuid.uuid4()), "type": "insert", "table": "run_conditions",
                        "rows": [_scrub({k: cond[k] for k in columns if k in cond.keys()})]}]
                if metrics:
                    ops.append({"op_id": str(uuid.uuid4()), "type": "insert", "table": "run_metrics",
                                "rows": [_scrub({k: m[k] for k in schema.TABLES["run_metrics"]}) for m in metrics]})
                if self.local.enqueue_run(*key, ops):
                    count += 1
        return count

    def sync(self) -> dict:
        report = {"backend": self.backend, "runs_queued": 0, "sent": 0, "already_applied": 0, "rejected": 0,
                  "set_aside": 0, "refreshed": 0, "error": ""}
        if self.backend != "postgres":
            report["error"] = f"nothing to sync (MEMORY_BACKEND={self.backend})"
            return report
        report["runs_queued"] = self._queue_run_metrics()
        try:
            with self.remote.session() as conn:
                version = self.remote.schema_version(conn)
                if version != schema.SCHEMA_VERSION:
                    report["error"] = (
                        "the hosted store is not set up (run `python -m src.memory migrate`)" if version is None else
                        f"the hosted store has schema version {version}, this code has {schema.SCHEMA_VERSION}: "
                        + ("pull the latest code" if version > schema.SCHEMA_VERSION else "run the migrations"))
                    return report
                for op in self.local.pending_ops(limit=1_000_000):
                    try:
                        applied = self.remote.apply(op, conn)
                    except RemoteUnavailable:
                        raise
                    except Exception as e:  # rejected by the store: keep it, count the attempt
                        status = self.local.mark_error(op["op_id"], f"{type(e).__name__}: {e}", MAX_SEND_ATTEMPTS)
                        report["rejected"] += 1
                        report["set_aside"] += status == "failed"
                        continue
                    self.local.mark_sent(op["op_id"])
                    report["sent" if applied else "already_applied"] += 1
                ids = self.local.memory_ids()
                for start in range(0, len(ids), 500):
                    chunk = ids[start:start + 500]
                    rows = self.remote.fetch_memories(states=None, ids=chunk, limit=len(chunk), conn=conn)
                    self.local.cache_memories(rows)
                    report["refreshed"] += len(rows)
            self.local.meta_set("last_sync", _now())
            self._remote_down_until = 0.0
        except RemoteUnavailable as e:
            self._mark_down(e)
            report["error"] = str(e)
        report["outbox"] = self.local.outbox_counts()
        return report

    def status(self, check_remote: bool = False) -> dict:
        info = {"backend": self.backend}
        if self.local:
            info.update(local_db=str(self.local.path), local_schema_version=self.local.schema_version(),
                        outbox=self.local.outbox_counts(), last_sync=self.local.meta_get("last_sync"),
                        memories_local=self.local.count("memories"))
        if self.backend == "postgres":
            info["cache_usable"] = self.cache_usable()
            if check_remote:
                try:
                    info["remote_schema_version"] = self.remote.schema_version()
                    info["remote"] = "reachable"
                except Exception as e:
                    info["remote"] = f"unreachable: {e}"
        return info


def client() -> MemoryClient | None:
    """The process's memory client (made on first use), or None if it could not be made."""
    global _client
    with _client_lock:
        if _client is None:
            try:
                _client = MemoryClient(MEMORY_BACKEND)
            except Exception as e:
                _warn("open the memory store; memory is off for this process", e)
                _client = MemoryClient("off")
        return _client


def reset_client(new_client: MemoryClient = None) -> None:
    """Replace the process's client (tests; a changed configuration)."""
    global _client
    with _client_lock:
        _client = new_client


def backend_name() -> str:
    c = client()
    return c.backend if c else "off"


# --- Building records ---

_code_version_cache = []


def _code_version() -> str | None:
    if not _code_version_cache:
        from src.orchestration import run_store
        _code_version_cache.append(run_store._project_git_commit() or None)
    return _code_version_cache[0]


def _cut(text: str, limit: int) -> str:
    if not isinstance(text, str) or len(text) <= limit:
        return text
    note = f"\n[cut: {len(text) - limit} characters removed]"
    return text[:limit - len(note)] + note


def build_memory(memory: dict) -> dict:
    """A complete, scrubbed, checked memories row from the fields an agent gives
    (kind, agent, polarity, title, summary are required)."""
    missing = [k for k in ("kind", "agent", "polarity", "title", "summary") if not memory.get(k)]
    if missing:
        raise ValueError(f"a memory needs {', '.join(missing)}")
    row = _scrub({k: v for k, v in memory.items() if k in schema.TABLES["memories"]})
    row["content"] = _cut(row.get("content"), schema.MAX_CONTENT_CHARS)
    row["summary"] = _cut(row["summary"], schema.MAX_SUMMARY_CHARS)
    now = _now()
    row.setdefault("id", str(uuid.uuid4()))
    for key, default in (("state", "active"), ("lineage", "unknown"), ("trust", "normal"), ("seen_count", 1),
                         ("times_recalled", 0), ("times_used", 0), ("times_helped", 0), ("times_hurt", 0)):
        row.setdefault(key, default)
    row.update(schema_version=schema.SCHEMA_VERSION, created_at=row.get("created_at") or now, updated_at=now)
    row.setdefault("code_version", _code_version())
    row.setdefault("contributor", MEMORY_CONTRIBUTOR or None)
    content = row.get("content")
    row["content_sha"] = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None
    row["search_text"] = " ".join(filter(None, [row["title"], row["summary"], " ".join(row.get("tags") or [])]))
    schema.check_row("memories", row)
    return row


# --- Public API (fail-soft) ---

def put_memory(memory: dict) -> str | None:
    """Store one memory. Returns its id, or None (memory off, or an error)."""
    try:
        c = client()
        if c.backend == "off":
            return None
        row = build_memory(memory)
        c.write({"op_id": str(uuid.uuid4()), "type": "insert", "table": "memories", "rows": [row]})
        return row["id"]
    except Exception as e:
        _warn("store a memory", e)
        return None


def get_memory(memory_id: str) -> dict | None:
    """One memory by id, in any state."""
    try:
        rows = client().fetch(ids=[memory_id], states=None, limit=1)
        return rows[0] if rows else None
    except Exception as e:
        _warn(f"read memory {memory_id}", e)
        return None


def list_memories(agent: str = None, kind: str = None, limit: int = 50) -> list[dict]:
    """Active memories, newest first. (Ranked recall is milestone M5.)"""
    try:
        return client().fetch(agent=agent, kind=kind, limit=limit)
    except Exception as e:
        _warn("list memories", e)
        return []


def record_usage(usage: dict) -> str | None:
    """Store one memory_usage row (a memory shown to an agent, and whether it was used)."""
    try:
        c = client()
        if c.backend == "off":
            return None
        row = _scrub({k: v for k, v in usage.items() if k in schema.TABLES["memory_usage"]})
        row.setdefault("id", str(uuid.uuid4()))
        row.setdefault("created_at", _now())
        schema.check_row("memory_usage", row)
        c.write({"op_id": str(uuid.uuid4()), "type": "insert", "table": "memory_usage", "rows": [row]})
        return row["id"]
    except Exception as e:
        _warn("record memory usage", e)
        return None


def increment(memory_id: str, **counters) -> bool:
    """Add to a memory's counters (seen_count, times_recalled, times_used, times_helped, times_hurt)."""
    try:
        c = client()
        if c.backend == "off":
            return False
        c.write({"op_id": str(uuid.uuid4()), "type": "increment", "table": "memories",
                 "key": {"id": memory_id}, "counters": counters})
        return True
    except Exception as e:
        _warn(f"update the counters of memory {memory_id}", e)
        return False


def sync() -> dict:
    """Send the outbox and the run metrics to the hosted store, and refresh
    the local copy. Does nothing unless MEMORY_BACKEND=postgres."""
    try:
        return client().sync()
    except Exception as e:
        _warn("sync the memory store", e)
        return {"error": str(e)}


def status(check_remote: bool = False) -> dict:
    try:
        return client().status(check_remote)
    except Exception as e:
        _warn("read the memory store's status", e)
        return {"error": str(e)}
