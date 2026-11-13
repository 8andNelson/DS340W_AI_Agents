-- Memory store, hosted Postgres version (shared by the team). Columns
-- match src/memory/schema.py. Applied by `python -m src.memory migrate`
-- with the admin connection (MEMORY_ADMIN_DATABASE_URL), never by a
-- pipeline run.
--
-- The tables are in their own schema, not "public": on Supabase, "public"
-- is exposed through its web API, and this schema must not be.

CREATE SCHEMA IF NOT EXISTS agent_memory;

CREATE TABLE agent_memory.memories (
    id               text PRIMARY KEY,
    kind             text NOT NULL CHECK (kind IN ('solution', 'failure', 'lesson', 'decision')),
    agent            text NOT NULL,
    run_id           text,
    stage_attempt    integer,
    invocation_id    text,
    polarity         text NOT NULL CHECK (polarity IN ('positive', 'negative', 'neutral')),
    title            text NOT NULL,
    summary          text NOT NULL,
    content          text,
    content_sha      text,
    task             jsonb,
    tags             text[],
    outcome          jsonb,
    evidence         jsonb,
    quality          real,
    quality_reasons  jsonb,
    trust            text NOT NULL DEFAULT 'normal' CHECK (trust IN ('normal', 'low_trust', 'seed')),
    state            text NOT NULL DEFAULT 'active'
                     CHECK (state IN ('candidate', 'active', 'quarantined', 'superseded', 'retired')),
    superseded_by    text,
    lineage          text NOT NULL DEFAULT 'unknown' CHECK (lineage IN ('kept', 'reverted', 'rejected', 'unknown')),
    schema_version   integer NOT NULL,
    code_version     text,
    contributor      text,
    seen_count       integer NOT NULL DEFAULT 1,
    times_recalled   integer NOT NULL DEFAULT 0,
    times_used       integer NOT NULL DEFAULT 0,
    times_helped     integer NOT NULL DEFAULT 0,
    times_hurt       integer NOT NULL DEFAULT 0,
    search_text      text,
    -- Server-side only: full-text search over title, summary and tags.
    fts              tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, coalesce(search_text, ''))) STORED,
    created_at       timestamptz NOT NULL,
    updated_at       timestamptz NOT NULL
);
CREATE INDEX idx_memories_agent_kind ON agent_memory.memories (agent, kind);
CREATE INDEX idx_memories_tags ON agent_memory.memories USING gin (tags);
CREATE INDEX idx_memories_fts ON agent_memory.memories USING gin (fts);
CREATE INDEX idx_memories_sha ON agent_memory.memories (content_sha, agent);
CREATE INDEX idx_memories_state ON agent_memory.memories (state);

CREATE TABLE agent_memory.memory_usage (
    id               text PRIMARY KEY,
    run_id           text,
    agent            text,
    attempt          integer,
    invocation_id    text,
    memory_id        text NOT NULL REFERENCES agent_memory.memories (id),
    rank_shown       integer,
    trigger          text,
    query            jsonb,
    used             boolean,
    supplied_digest  text,
    stage_decision   text,
    gap_delta        real,
    helped           boolean,
    created_at       timestamptz NOT NULL
);
CREATE INDEX idx_memory_usage_memory ON agent_memory.memory_usage (memory_id);

CREATE TABLE agent_memory.memory_ratings (
    memory_id    text NOT NULL REFERENCES agent_memory.memories (id),
    contributor  text NOT NULL,
    rating       integer NOT NULL CHECK (rating IN (-1, 1)),
    note         text,
    created_at   timestamptz NOT NULL,
    PRIMARY KEY (memory_id, contributor)
);

CREATE TABLE agent_memory.memory_state_changes (
    id          text PRIMARY KEY,
    memory_id   text NOT NULL REFERENCES agent_memory.memories (id),
    old_state   text,
    new_state   text NOT NULL,
    reason      text,
    actor       text,
    created_at  timestamptz NOT NULL
);

CREATE TABLE agent_memory.memory_runs (
    run_id             text PRIMARY KEY,
    contributor        text,
    topic              text,
    status             text,
    final_phase        text,
    git_commit         text,
    memory_mode        text,
    memories_recalled  integer,
    memories_used      integer,
    trust              text,
    started_at         timestamptz,
    finished_at        timestamptz
);

CREATE TABLE agent_memory.run_conditions (
    run_id               text NOT NULL,
    metrics_version      integer NOT NULL,
    status               text,
    contributor          text,
    started_at           text,
    finished_at          text,
    topic                text,
    topic_key            text,
    parent_paper_doi     text,
    parent_paper_title   text,
    dataset_identities   text,
    rows_used            integer,
    sampled              integer,
    code_source          text,
    model_family         text,
    git_commit           text,
    git_dirty            integer,
    llm_provider         text,
    served_models        text,
    memory_mode          text,
    memory_backend       text,
    memory_reuse         integer,
    memory_tuning        integer,
    memory_curator       integer,
    memory_exposure      text,
    memory_bank_size     text,
    memory_bank_snapshot text,
    comparison_group     text,
    seeds                text,
    python_version       text,
    os                   text,
    log_matching         text,
    timing_method        text,
    notes                text,
    computed_at          text NOT NULL,
    PRIMARY KEY (run_id, metrics_version)
);

CREATE TABLE agent_memory.run_metrics (
    run_id           text NOT NULL,
    metrics_version  integer NOT NULL,
    scope            text NOT NULL,
    metric           text NOT NULL,
    value_num        double precision,
    value_text       text,
    unit             text,
    missing_reason   text,
    computed_at      text NOT NULL,
    PRIMARY KEY (run_id, metrics_version, scope, metric)
);
CREATE INDEX idx_run_metrics_metric ON agent_memory.run_metrics (metric, scope);

-- Store settings; the migration runner records schema_version here.
CREATE TABLE agent_memory.memory_meta (
    key    text PRIMARY KEY,
    value  text
);

-- Each client op is applied once: a resent op (after a lost
-- acknowledgement) finds its id here and is skipped.
CREATE TABLE agent_memory.applied_ops (
    op_id       text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
