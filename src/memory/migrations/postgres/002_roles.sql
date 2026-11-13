-- The pipeline's database role (docs/CLAUDE_BACKEND_PLAN.md, section 4.2):
-- SELECT and INSERT on the memory tables, UPDATE only on the counter
-- and state columns, and no DELETE or TRUNCATE at all -- so a defect or a
-- bad model output cannot erase the team's memory.
--
-- The role is created without a password. The admin sets one by hand
-- (never in this repository):
--     ALTER ROLE memory_agent WITH LOGIN PASSWORD '<a long random password>';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'memory_agent') THEN
        CREATE ROLE memory_agent NOLOGIN;
    END IF;
END
$$;

REVOKE ALL ON SCHEMA agent_memory FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA agent_memory FROM PUBLIC;
GRANT USAGE ON SCHEMA agent_memory TO memory_agent;

GRANT SELECT, INSERT ON agent_memory.memories, agent_memory.memory_usage, agent_memory.memory_ratings,
    agent_memory.memory_state_changes, agent_memory.memory_runs, agent_memory.run_conditions,
    agent_memory.run_metrics, agent_memory.applied_ops TO memory_agent;
GRANT SELECT ON agent_memory.memory_meta TO memory_agent;
GRANT UPDATE (state, superseded_by, quality, quality_reasons, seen_count, times_recalled, times_used,
              times_helped, times_hurt, updated_at) ON agent_memory.memories TO memory_agent;
GRANT UPDATE (helped, gap_delta) ON agent_memory.memory_usage TO memory_agent;

-- Supabase's web API roles must not reach this schema.
DO $$
DECLARE api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format('REVOKE ALL ON SCHEMA agent_memory FROM %I', api_role);
            EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA agent_memory FROM %I', api_role);
        END IF;
    END LOOP;
END
$$;
