"""
The memory store's tables, columns and allowed values -- one definition
used by both backends (docs/CLAUDE_BACKEND_PLAN.md, sections 6 and 11.5).

The SQL migrations (migrations/sqlite, migrations/postgres) create these
tables; tests/test_memory_store.py checks that the SQLite migration has
exactly these columns. Table and column names are only ever taken from
this module, never from input, so building SQL from them is safe.

Column types: "text", "int", "real", "bool", "json" (dict or list),
"list" (list of strings), "ts" (ISO-8601 UTC text). SQLite stores json and
list as JSON text; Postgres stores them as jsonb and text[].
"""
SCHEMA_VERSION = 2  # 2: run events and invocation lineage (milestone M3)

KINDS = ("solution", "failure", "lesson", "decision")
POLARITIES = ("positive", "negative", "neutral")
STATES = ("candidate", "active", "quarantined", "superseded", "retired")
# kept: part of the run's final baseline; superseded: accepted, then replaced
# by a later kept attempt of the same agent; reverted: approved inside a
# rerun that was undone; rejected: failed its checks; unknown: the record is
# not complete (crash). See src/memory/events.py.
LINEAGES = ("kept", "superseded", "reverted", "rejected", "unknown")
EVENT_TYPES = ("attempt", "rerun_opened", "rerun_closed", "request_declined", "run_finished")
TRUST = ("normal", "low_trust", "seed")

MAX_CONTENT_CHARS = 64 * 1024
MAX_SUMMARY_CHARS = 1000

TABLES = {
    "memories": {
        "id": "text", "kind": "text", "agent": "text", "run_id": "text", "stage_attempt": "int",
        "invocation_id": "text", "polarity": "text", "title": "text", "summary": "text", "content": "text",
        "content_sha": "text", "task": "json", "tags": "list", "outcome": "json", "evidence": "json",
        "quality": "real", "quality_reasons": "json", "trust": "text", "state": "text", "superseded_by": "text",
        "lineage": "text", "schema_version": "int", "code_version": "text", "contributor": "text",
        "seen_count": "int", "times_recalled": "int", "times_used": "int", "times_helped": "int",
        "times_hurt": "int", "search_text": "text", "created_at": "ts", "updated_at": "ts",
    },
    "memory_usage": {
        "id": "text", "run_id": "text", "agent": "text", "attempt": "int", "invocation_id": "text",
        "memory_id": "text", "rank_shown": "int", "trigger": "text", "query": "json", "used": "bool",
        "supplied_digest": "text", "stage_decision": "text", "gap_delta": "real", "helped": "bool",
        "created_at": "ts",
    },
    "memory_ratings": {
        "memory_id": "text", "contributor": "text", "rating": "int", "note": "text", "created_at": "ts",
    },
    "memory_state_changes": {
        "id": "text", "memory_id": "text", "old_state": "text", "new_state": "text", "reason": "text",
        "actor": "text", "created_at": "ts",
    },
    "memory_runs": {
        "run_id": "text", "contributor": "text", "topic": "text", "status": "text", "final_phase": "text",
        "git_commit": "text", "memory_mode": "text", "memories_recalled": "int", "memories_used": "int",
        "trust": "text", "started_at": "ts", "finished_at": "ts",
    },
    # What happened in a run, written as it happens (milestone M3).
    "memory_events": {
        "id": "text", "run_id": "text", "seq": "int", "event_type": "text", "invocation_id": "text",
        "agent": "text", "attempt": "int", "rerun": "int", "decision": "text", "payload": "json",
        "contributor": "text", "created_at": "ts",
    },
    # Each agent attempt's place in the run's final result, set at run end.
    "invocation_lineage": {
        "invocation_id": "text", "run_id": "text", "agent": "text", "attempt": "int", "lineage": "text",
        "reason": "text", "contributor": "text", "resolved_at": "ts",
    },
    # Copies of the run database's measurement tables (src/orchestration/run_store.py),
    # so the team's runs can be compared in one place.
    "run_conditions": {
        "run_id": "text", "metrics_version": "int", "status": "text", "contributor": "text",
        "started_at": "text", "finished_at": "text", "topic": "text", "topic_key": "text",
        "parent_paper_doi": "text", "parent_paper_title": "text", "dataset_identities": "text",
        "rows_used": "int", "sampled": "int", "code_source": "text", "model_family": "text",
        "git_commit": "text", "git_dirty": "int", "llm_provider": "text", "served_models": "text",
        "memory_mode": "text", "memory_backend": "text", "memory_reuse": "int", "memory_tuning": "int",
        "memory_curator": "int", "memory_exposure": "text", "memory_bank_size": "text",
        "memory_bank_snapshot": "text", "comparison_group": "text", "seeds": "text", "python_version": "text",
        "os": "text", "log_matching": "text", "timing_method": "text", "notes": "text", "computed_at": "text",
    },
    "run_metrics": {
        "run_id": "text", "metrics_version": "int", "scope": "text", "metric": "text", "value_num": "real",
        "value_text": "text", "unit": "text", "missing_reason": "text", "computed_at": "text",
    },
}

PRIMARY_KEYS = {
    "memories": ("id",),
    "memory_usage": ("id",),
    "memory_ratings": ("memory_id", "contributor"),
    "memory_state_changes": ("id",),
    "memory_runs": ("run_id",),
    "memory_events": ("id",),
    "invocation_lineage": ("invocation_id",),
    "run_conditions": ("run_id", "metrics_version"),
    "run_metrics": ("run_id", "metrics_version", "scope", "metric"),
}

# The only columns a client may change after insert (plan 4.2: the
# pipeline's database role has UPDATE on these and no DELETE at all).
COUNTER_COLUMNS = {
    "memories": ("seen_count", "times_recalled", "times_used", "times_helped", "times_hurt"),
}


def check_row(table: str, row: dict) -> None:
    """Raise ValueError for an unknown table or column, or a disallowed value."""
    if table not in TABLES:
        raise ValueError(f"unknown memory table: {table}")
    unknown = set(row) - set(TABLES[table])
    if unknown:
        raise ValueError(f"unknown column(s) in {table}: {', '.join(sorted(unknown))}")
    missing = [k for k in PRIMARY_KEYS[table] if row.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{table} row has no {', '.join(missing)}")
    allowed_values = {
        "memories": (("kind", KINDS), ("polarity", POLARITIES), ("state", STATES), ("lineage", LINEAGES),
                     ("trust", TRUST)),
        "memory_events": (("event_type", EVENT_TYPES),),
        "invocation_lineage": (("lineage", LINEAGES),),
    }
    for column, allowed in allowed_values.get(table, ()):
        if row.get(column) is not None and row[column] not in allowed:
            raise ValueError(f"{table}.{column} must be one of {', '.join(allowed)}")
