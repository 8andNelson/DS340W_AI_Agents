"""
SQLite memory store (database/memory_local.db, gitignored).

Two roles, same file and schema:
  * MEMORY_BACKEND=sqlite: the whole store (offline use, tests).
  * MEMORY_BACKEND=postgres: the local mirror -- every write is applied
    here and put in the outbox in one transaction, so a write is never
    lost when the network fails; reads that reached the hosted store are
    cached here for use while it is unreachable.

Migrations in migrations/sqlite are applied automatically on open.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.memory import schema
from src.memory.backends.base import MigrationError, check_op, migration_files


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode(table: str, row: dict) -> dict:
    types = schema.TABLES[table]
    out = {}
    for column, value in row.items():
        kind = types[column]
        if value is None:
            out[column] = None
        elif kind in ("json", "list"):
            out[column] = json.dumps(value, default=str)
        elif kind == "bool":
            out[column] = int(bool(value))
        else:
            out[column] = value
    return out


def decode(table: str, row: sqlite3.Row) -> dict:
    types = schema.TABLES[table]
    out = {}
    for column in row.keys():
        value = row[column]
        kind = types.get(column)
        if value is not None and kind in ("json", "list"):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        elif value is not None and kind == "bool":
            value = bool(value)
        out[column] = value
    return out


class SqliteStore:
    dialect = "sqlite"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Schema ---

    def migrate(self) -> list[str]:
        """Apply the migrations not yet applied. Returns their file names."""
        applied = []
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                         "checksum TEXT NOT NULL, applied_at TEXT NOT NULL)")
            done = {r["version"]: r["checksum"] for r in conn.execute("SELECT version, checksum FROM schema_migrations")}
        for version, name, sql, checksum in migration_files("sqlite"):
            if version in done:
                if done[version] != checksum:
                    raise MigrationError(f"{name} changed after it was applied to {self.path}")
                continue
            conn = self._connect()
            try:
                conn.executescript("BEGIN;\n" + sql + "\nCOMMIT;")
                with conn:
                    conn.execute("INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
                                 (version, name, checksum, _now()))
            except sqlite3.Error:
                conn.rollback()
                raise
            finally:
                conn.close()
            applied.append(name)
        # The schema version is the code's data-schema version, not the
        # number of migration files (see the Postgres runner).
        current = self.schema_version()
        if current is not None and current > schema.SCHEMA_VERSION:
            raise MigrationError(f"{self.path} has schema version {current}, newer than this code's "
                                 f"{schema.SCHEMA_VERSION}: pull the latest code")
        self.meta_set("schema_version", str(schema.SCHEMA_VERSION))
        return applied

    @contextmanager
    def session(self):
        """Same interface as PostgresStore.session (each call opens its own connection)."""
        yield None

    def schema_version(self, conn=None) -> int | None:
        value = self.meta_get("schema_version")
        return int(value) if value else None

    # --- Writes ---

    def _apply(self, conn: sqlite3.Connection, op: dict) -> bool:
        check_op(op)
        if conn.execute("INSERT OR IGNORE INTO applied_ops (op_id, applied_at) VALUES (?, ?)",
                        (op["op_id"], _now())).rowcount == 0:
            return False
        table = op["table"]
        if op["type"] == "insert":
            for row in op["rows"]:
                encoded = encode(table, row)
                columns = list(encoded)
                conn.execute(f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "
                             f"VALUES ({', '.join('?' * len(columns))})", [encoded[c] for c in columns])
        elif op["type"] == "set_state":
            conn.execute("UPDATE memories SET state = ?, superseded_by = COALESCE(?, superseded_by), updated_at = ? "
                         "WHERE id = ?", (op["state"], op.get("superseded_by"), _now(), op["key"]["id"]))
        else:
            sets = ", ".join(f"{c} = {c} + ?" for c in op["counters"])
            where = " AND ".join(f"{k} = ?" for k in op["key"])
            conn.execute(f"UPDATE {table} SET {sets}, updated_at = ? WHERE {where}",
                         [*op["counters"].values(), _now(), *op["key"].values()])
        return True

    def apply(self, op: dict, conn=None) -> bool:
        with self._connect() as conn:
            return self._apply(conn, op)

    def apply_and_enqueue(self, op: dict) -> None:
        """Apply an op locally and put it in the outbox, in one transaction."""
        with self._connect() as conn:
            self._apply(conn, op)
            conn.execute("INSERT OR IGNORE INTO outbox (op_id, op, created_at) VALUES (?, ?, ?)",
                         (op["op_id"], json.dumps(op, default=str), _now()))

    def enqueue_run(self, run_id: str, metrics_version: int, ops: list) -> bool:
        """Put a run's metric uploads in the outbox and mark the run queued, in
        one transaction. False when the run was already queued."""
        with self._connect() as conn:
            if conn.execute("INSERT OR IGNORE INTO synced_runs (run_id, metrics_version, queued_at) VALUES (?, ?, ?)",
                            (run_id, metrics_version, _now())).rowcount == 0:
                return False
            for op in ops:
                check_op(op)
                conn.execute("INSERT OR IGNORE INTO outbox (op_id, op, created_at) VALUES (?, ?, ?)",
                             (op["op_id"], json.dumps(op, default=str), _now()))
        return True

    def queued_runs(self) -> set:
        with self._connect() as conn:
            return {(r["run_id"], r["metrics_version"]) for r in conn.execute("SELECT * FROM synced_runs")}

    # --- Outbox ---

    def pending_ops(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT op FROM outbox WHERE status = 'pending' ORDER BY seq LIMIT ?", (limit,))
            return [json.loads(r["op"]) for r in rows]

    def mark_sent(self, op_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE outbox SET status = 'sent', last_error = NULL WHERE op_id = ?", (op_id,))

    def mark_error(self, op_id: str, error: str, max_attempts: int) -> str:
        """Count a failed send; after max_attempts the op is set aside as
        'failed' (kept for review, never deleted). Returns the new status."""
        with self._connect() as conn:
            conn.execute("UPDATE outbox SET attempts = attempts + 1, last_error = ?, "
                         "status = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END WHERE op_id = ?",
                         (error[:1000], max_attempts, op_id))
            return conn.execute("SELECT status FROM outbox WHERE op_id = ?", (op_id,)).fetchone()["status"]

    def outbox_counts(self) -> dict:
        with self._connect() as conn:
            counts = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM outbox GROUP BY status")}
        return {status: counts.get(status, 0) for status in ("pending", "sent", "failed")}

    # --- Meta and cache ---

    def meta_get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM memory_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO memory_meta (key, value) VALUES (?, ?)", (key, value))

    def cache_memories(self, rows: list[dict]) -> None:
        """Store memories read from the hosted store (their newest state and
        counters replace the local copy)."""
        with self._connect() as conn:
            for row in rows:
                encoded = encode("memories", {k: v for k, v in row.items() if k in schema.TABLES["memories"]})
                columns = list(encoded)
                conn.execute(f"INSERT OR REPLACE INTO memories ({', '.join(columns)}) "
                             f"VALUES ({', '.join('?' * len(columns))})", [encoded[c] for c in columns])

    def memory_ids(self) -> list[str]:
        with self._connect() as conn:
            return [r["id"] for r in conn.execute("SELECT id FROM memories")]

    # --- Reads ---

    def fetch_memories(self, agent: str = None, kind: str = None, states: tuple = ("active",),
                       ids: list = None, limit: int = 50, conn=None, content_shas: list = None) -> list[dict]:
        where, params = [], []
        if content_shas is not None:
            if not content_shas:
                return []
            where.append(f"content_sha IN ({', '.join('?' * len(content_shas))})")
            params.extend(content_shas)
        if agent:
            where.append("agent = ?")
            params.append(agent)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if states:
            where.append(f"state IN ({', '.join('?' * len(states))})")
            params.extend(states)
        if ids is not None:
            if not ids:
                return []
            where.append(f"id IN ({', '.join('?' * len(ids))})")
            params.extend(ids)
        sql = "SELECT * FROM memories" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY created_at DESC LIMIT ?"
        with self._connect() as conn:
            return [decode("memories", r) for r in conn.execute(sql, [*params, limit])]

    def rows(self, table: str, order_by: str = None, **equals) -> list[dict]:
        """Rows of one table whose columns equal the given values (decoded)."""
        columns = schema.TABLES.get(table, {})
        if not columns or set(equals) - set(columns) or (order_by and order_by not in columns):
            raise ValueError(f"unknown table or column: {table} {sorted(equals)} {order_by or ''}")
        sql = f"SELECT * FROM {table}"
        if equals:
            sql += " WHERE " + " AND ".join(f"{c} = ?" for c in equals)
        if order_by:
            sql += f" ORDER BY {order_by}"
        with self._connect() as conn:
            return [decode(table, r) for r in conn.execute(sql, list(equals.values()))]

    def count(self, table: str) -> int:
        if table not in schema.TABLES:
            raise ValueError(f"unknown memory table: {table}")
        with self._connect() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
