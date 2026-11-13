-- Run recording (milestone M3): events written as they happen, and each
-- agent attempt's lineage, resolved at run end. Columns match
-- src/memory/schema.py.

CREATE TABLE agent_memory.memory_events (
    id             text PRIMARY KEY,
    run_id         text NOT NULL,
    seq            integer NOT NULL,
    event_type     text NOT NULL CHECK (event_type IN ('attempt', 'rerun_opened', 'rerun_closed',
                                                       'request_declined', 'run_finished')),
    invocation_id  text,
    agent          text,
    attempt        integer,
    rerun          integer,
    decision       text,
    payload        jsonb,
    contributor    text,
    created_at     timestamptz NOT NULL
);
CREATE INDEX idx_memory_events_run ON agent_memory.memory_events (run_id, seq);

CREATE TABLE agent_memory.invocation_lineage (
    invocation_id  text PRIMARY KEY,
    run_id         text NOT NULL,
    agent          text NOT NULL,
    attempt        integer,
    lineage        text NOT NULL CHECK (lineage IN ('kept', 'superseded', 'reverted', 'rejected', 'unknown')),
    reason         text,
    contributor    text,
    resolved_at    timestamptz NOT NULL
);
CREATE INDEX idx_invocation_lineage_run ON agent_memory.invocation_lineage (run_id);

-- "superseded": accepted, then replaced by a later kept attempt of the same agent.
ALTER TABLE agent_memory.memories DROP CONSTRAINT IF EXISTS memories_lineage_check;
ALTER TABLE agent_memory.memories ADD CONSTRAINT memories_lineage_check
    CHECK (lineage IN ('kept', 'superseded', 'reverted', 'rejected', 'unknown'));

-- The pipeline's role: read and add only, like the other memory tables.
GRANT SELECT, INSERT ON agent_memory.memory_events, agent_memory.invocation_lineage TO memory_agent;
REVOKE ALL ON agent_memory.memory_events, agent_memory.invocation_lineage FROM PUBLIC;
DO $$
DECLARE api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format('REVOKE ALL ON agent_memory.memory_events, agent_memory.invocation_lineage FROM %I',
                           api_role);
        END IF;
    END LOOP;
END
$$;
