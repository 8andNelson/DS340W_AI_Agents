"""
What every memory store backend does, and the parts they share.

A backend applies write "ops" and answers reads. An op is a JSON-safe dict:

  {"op_id": <uuid>, "type": "insert", "table": <table>, "rows": [<row>, ...]}
      Insert rows; a row whose primary key exists is skipped (never
      overwritten -- memory records are append-only).
  {"op_id": <uuid>, "type": "increment", "table": "memories",
   "key": {"id": <id>}, "counters": {<counter column>: <n>, ...}}
      Add to counter columns (schema.COUNTER_COLUMNS) only.
  {"op_id": <uuid>, "type": "set_state", "table": "memories", "key": {"id": <id>},
   "state": <state>, "superseded_by": <id or None>}
      Change a memory's state (plan 8.4); the change itself is recorded by a
      separate insert into memory_state_changes.
  {"op_id": <uuid>, "type": "usage_outcome", "table": "memory_usage", "key": {"id": <usage id>},
   "memory_id": <id>, "helped": true|false}
      Record what followed a memory's use (plan 9.3), once: the usage row's
      helped is set only while it is still empty, and only then does the
      memory's times_helped or times_hurt go up -- in one transaction.
  {"op_id": <uuid>, "type": "set_quality", "table": "memories", "key": {"id": <id>},
   "quality": <0..1>, "quality_reasons": {...}}
      Change a memory's score and its reasons (a contradiction, plan 8.4).

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
    elif op.get("type") == "set_state":
        if op.get("table") != "memories" or set(op.get("key") or {}) != {"id"} or not op["key"]["id"]:
            raise ValueError("a set_state op changes one memory, by id")
        if op.get("state") not in schema.STATES:
            raise ValueError(f"state must be one of {', '.join(schema.STATES)}")
    elif op.get("type") == "usage_outcome":
        if op.get("table") != "memory_usage" or not (op.get("key") or {}).get("id") or not op.get("memory_id"):
            raise ValueError("a usage_outcome op needs the usage id and its memory id")
        if not isinstance(op.get("helped"), bool):
            raise ValueError("helped must be true or false")
    elif op.get("type") == "set_quality":
        if op.get("table") != "memories" or not (op.get("key") or {}).get("id"):
            raise ValueError("a set_quality op changes one memory, by id")
        if not isinstance(op.get("quality"), (int, float)) or not 0 <= op["quality"] <= 1:
            raise ValueError("quality must be between 0 and 1")
        if not isinstance(op.get("quality_reasons"), dict):
            raise ValueError("a set_quality op needs its quality_reasons")
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
