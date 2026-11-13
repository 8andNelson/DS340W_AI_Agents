-- Memory store, SQLite version: the local mirror (read cache + outbox) and
-- the whole store when MEMORY_BACKEND=sqlite. Columns match
-- src/memory/schema.py. JSON and list values are stored as JSON text.

CREATE TABLE memories (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    agent            TEXT NOT NULL,
    run_id           TEXT,
    stage_attempt    INTEGER,
    invocation_id    TEXT,
    polarity         TEXT NOT NULL,
    title            TEXT NOT NULL,
    summary          TEXT NOT NULL,
    content          TEXT,
    content_sha      TEXT,
    task             TEXT,
    tags             TEXT,
    outcome          TEXT,
    evidence         TEXT,
    quality          REAL,
    quality_reasons  TEXT,
    trust            TEXT NOT NULL DEFAULT 'normal',
    state            TEXT NOT NULL DEFAULT 'active',
    superseded_by    TEXT,
    lineage          TEXT NOT NULL DEFAULT 'unknown',
    schema_version   INTEGER NOT NULL,
    code_version     TEXT,
    contributor      TEXT,
    seen_count       INTEGER NOT NULL DEFAULT 1,
    times_recalled   INTEGER NOT NULL DEFAULT 0,
    times_used       INTEGER NOT NULL DEFAULT 0,
    times_helped     INTEGER NOT NULL DEFAULT 0,
    times_hurt       INTEGER NOT NULL DEFAULT 0,
    search_text      TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX idx_memories_agent_kind ON memories(agent, kind);
CREATE INDEX idx_memories_sha ON memories(content_sha, agent);
CREATE INDEX idx_memories_state ON memories(state);

CREATE TABLE memory_usage (
    id               TEXT PRIMARY KEY,
    run_id           TEXT,
    agent            TEXT,
    attempt          INTEGER,
    invocation_id    TEXT,
    memory_id        TEXT NOT NULL,
    rank_shown       INTEGER,
    trigger          TEXT,
    query            TEXT,
    used             INTEGER,
    supplied_digest  TEXT,
    stage_decision   TEXT,
    gap_delta        REAL,
    helped           INTEGER,
    created_at       TEXT NOT NULL
);
CREATE INDEX idx_memory_usage_memory ON memory_usage(memory_id);

CREATE TABLE memory_ratings (
    memory_id    TEXT NOT NULL,
    contributor  TEXT NOT NULL,
    rating       INTEGER NOT NULL,
    note         TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (memory_id, contributor)
);

CREATE TABLE memory_state_changes (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    old_state   TEXT,
    new_state   TEXT NOT NULL,
    reason      TEXT,
    actor       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE memory_runs (
    run_id             TEXT PRIMARY KEY,
    contributor        TEXT,
    topic              TEXT,
    status             TEXT,
    final_phase        TEXT,
    git_commit         TEXT,
    memory_mode        TEXT,
    memories_recalled  INTEGER,
    memories_used      INTEGER,
    trust              TEXT,
    started_at         TEXT,
    finished_at        TEXT
);

CREATE TABLE run_conditions (
    run_id               TEXT NOT NULL,
    metrics_version      INTEGER NOT NULL,
    status               TEXT,
    contributor          TEXT,
    started_at           TEXT,
    finished_at          TEXT,
    topic                TEXT,
    topic_key            TEXT,
    parent_paper_doi     TEXT,
    parent_paper_title   TEXT,
    dataset_identities   TEXT,
    rows_used            INTEGER,
    sampled              INTEGER,
    code_source          TEXT,
    model_family         TEXT,
    git_commit           TEXT,
    git_dirty            INTEGER,
    llm_provider         TEXT,
    served_models        TEXT,
    memory_mode          TEXT,
    memory_backend       TEXT,
    memory_reuse         INTEGER,
    memory_tuning        INTEGER,
    memory_curator       INTEGER,
    memory_exposure      TEXT,
    memory_bank_size     TEXT,
    memory_bank_snapshot TEXT,
    comparison_group     TEXT,
    seeds                TEXT,
    python_version       TEXT,
    os                   TEXT,
    log_matching         TEXT,
    timing_method        TEXT,
    notes                TEXT,
    computed_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, metrics_version)
);

CREATE TABLE run_metrics (
    run_id           TEXT NOT NULL,
    metrics_version  INTEGER NOT NULL,
    scope            TEXT NOT NULL,
    metric           TEXT NOT NULL,
    value_num        REAL,
    value_text       TEXT,
    unit             TEXT,
    missing_reason   TEXT,
    computed_at      TEXT NOT NULL,
    PRIMARY KEY (run_id, metrics_version, scope, metric)
);

-- Store bookkeeping: key/value settings (schema_version, last_sync, ...).
CREATE TABLE memory_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

-- Ops already applied when this file is the store a remote client writes
-- to (tests use a second SQLite file as the "hosted" store), so a resent
-- op is applied once.
CREATE TABLE applied_ops (
    op_id       TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

-- Local only: writes waiting to be sent to the hosted store (MEMORY_BACKEND=postgres).
CREATE TABLE outbox (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    op_id       TEXT NOT NULL UNIQUE,
    op          TEXT NOT NULL,      -- JSON: {"type", "table", ...}
    created_at  TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    status      TEXT NOT NULL DEFAULT 'pending'  -- pending | sent | failed
);

-- Local only: runs whose metrics were put in the outbox for upload.
CREATE TABLE synced_runs (
    run_id           TEXT NOT NULL,
    metrics_version  INTEGER NOT NULL,
    queued_at        TEXT NOT NULL,
    PRIMARY KEY (run_id, metrics_version)
);
