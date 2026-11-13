-- Run recording (milestone M3): events written as they happen, and each
-- agent attempt's lineage, resolved at run end. Columns match
-- src/memory/schema.py.

CREATE TABLE memory_events (
    id             TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    event_type     TEXT NOT NULL,
    invocation_id  TEXT,
    agent          TEXT,
    attempt        INTEGER,
    rerun          INTEGER,
    decision       TEXT,
    payload        TEXT,
    contributor    TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_memory_events_run ON memory_events(run_id, seq);

CREATE TABLE invocation_lineage (
    invocation_id  TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    agent          TEXT NOT NULL,
    attempt        INTEGER,
    lineage        TEXT NOT NULL,
    reason         TEXT,
    contributor    TEXT,
    resolved_at    TEXT NOT NULL
);
CREATE INDEX idx_invocation_lineage_run ON invocation_lineage(run_id);
