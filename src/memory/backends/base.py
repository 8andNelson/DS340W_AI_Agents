"""
What every memory store backend does, and the parts they share.

A backend applies write "ops" and answers reads. An op is a JSON-safe dict:

  {"op_id": <uuid>, "type": "insert", "table": <table>, "rows": [<row>, ...]}
      Insert rows; a row whose primary key exists is skipped (never
      overwritten -- memory records are append-only).
  {"op_id": <uuid>, "type": "increment", "table": "memories",
   "key": {"id": <id>}, "counters": {<counter column>: <n>, ...}}
      Add to counter columns (schema.COUNTER_COLUMNS) only.

apply(op) applies an op exactly once: its op_id is recorded in
applied_ops in the same transaction, so an op resent after a lost
acknowledgement is skipped. This is what makes the outbox safe to retry.

Backends (sqlite.SqliteStore, postgres.PostgresStore) provide:
  apply(op) -> bool                   True if applied, False if already applied
  fetch_memories(agent, kind, states, ids, limit) -> list[dict]
  schema_version() -> int | None
"""
import hashlib
from pathlib import Path

from src.memory import schema

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


def check_op(op: dict) -> None:
    """Raise ValueError when an op is malformed or names an unknown table or column."""
    if not isinstance(op, dict) or not op.get("op_id"):
        raise ValueError("an op needs an op_id")
    if op.get("type") == "insert":
        rows = op.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("an insert op needs rows")
        for row in rows:
            schema.check_row(op.get("table"), row)
    elif op.get("type") == "increment":
        table = op.get("table")
        allowed = schema.COUNTER_COLUMNS.get(table, ())
        counters = op.get("counters") or {}
        if not counters or set(counters) - set(allowed):
            raise ValueError(f"an increment op may change only {', '.join(allowed) or 'nothing'} in {table}")
        if not all(isinstance(n, int) for n in counters.values()):
            raise ValueError("counter increments must be integers")
        key = op.get("key") or {}
        if set(key) != set(schema.PRIMARY_KEYS[table]) or not all(key.values()):
            raise ValueError(f"an increment op needs the key {schema.PRIMARY_KEYS[table]}")
    else:
        raise ValueError(f"unknown op type: {op.get('type')}")


def migration_files(dialect: str) -> list[tuple[int, str, str, str]]:
    """[(version, name, sql, checksum)] for one dialect, in order."""
    files = []
    for path in sorted((MIGRATIONS_DIR / dialect).glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        version = int(path.name.split("_", 1)[0])
        files.append((version, path.name, sql, hashlib.sha256(sql.encode("utf-8")).hexdigest()))
    return files


class RemoteUnavailable(RuntimeError):
    """The hosted store could not be reached (network, timeout, paused
    project). Transient: the outbox keeps the write for the next sync."""


class MigrationError(RuntimeError):
    """An applied migration file was changed afterwards: the store and the
    code no longer agree on what the schema is."""
