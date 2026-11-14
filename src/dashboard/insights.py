"""
The dashboard's Compare runs and Memory views (milestone M9) -- read-only.

Compare runs: the run database's measurements (run_conditions, run_metrics),
opened mode=ro like every other dashboard read, compared with the same code
the pipeline's `run_metrics compare` uses (src/analysis.py), so the two
always show the same numbers.

Memory: the team's memory store. With MEMORY_BACKEND=postgres and a
MEMORY_DATABASE_URL (.env, loaded by `python -m src.dashboard`) it reads the
hosted store over a connection whose every transaction is READ ONLY;
otherwise the local mirror (database/memory_local.db) opened mode=ro. It
never writes: the dashboard writes only to dashboard_files/. A memory's
score is its effective quality (src/analysis.py, the same formula recall
ranks by).
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from src import analysis

MEMORY_COLUMNS = ("id", "kind", "agent", "run_id", "stage_attempt", "invocation_id", "polarity", "title", "summary",
                  "content_sha", "task", "tags", "outcome", "evidence", "quality", "quality_reasons", "trust", "state",
                  "superseded_by", "lineage", "schema_version", "code_version", "contributor", "seen_count",
                  "times_recalled", "times_used", "times_helped", "times_hurt", "created_at", "updated_at")
JSON_COLUMNS = {"task", "tags", "outcome", "evidence", "quality_reasons"}
CACHE_SECONDS = 60
BEST_WORST = 5

_cache = {}
_cache_lock = threading.Lock()


def _read_only(db_file: Path) -> sqlite3.Connection | None:
    if not Path(db_file).is_file():
        return None
    conn = sqlite3.connect(f"{Path(db_file).resolve().as_uri()}?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


# --- Compare runs ---

def load_runs(db_file: Path) -> list:
    """Every measured run (newest metrics version), as run_metrics.load_runs
    returns them -- read-only."""
    conn = _read_only(db_file)
    if conn is None:
        return []
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "run_conditions" not in tables:
            return []
        conditions = conn.execute(
            "SELECT c.* FROM run_conditions c JOIN (SELECT run_id, MAX(metrics_version) v FROM run_conditions "
            "GROUP BY run_id) latest ON c.run_id = latest.run_id AND c.metrics_version = latest.v "
            "ORDER BY c.started_at").fetchall()
        runs = []
        for row in conditions:
            cond = dict(row)
            rows = conn.execute("SELECT * FROM run_metrics WHERE run_id = ? AND metrics_version = ?",
                                (cond["run_id"], cond["metrics_version"])).fetchall()
            runs.append({"conditions": cond, "metrics": analysis.metrics_from_rows(rows, cond["metrics_version"])})
        return runs
    finally:
        conn.close()


def compare(db_file: Path, group_by: str = "memory_mode", topic: str = "", group: str = "",
            all_tasks: bool = False, metrics: list = None) -> dict:
    """The comparison and a per-run table, ready for JSON."""
    runs = load_runs(db_file)
    if topic:
        words = set(topic.lower().split())
        runs = [r for r in runs if words <= set((r["conditions"].get("topic_key") or "").split())]
    if group:
        runs = [r for r in runs if r["conditions"].get("comparison_group") == group]
    chosen = tuple(metrics) if metrics else analysis.DEFAULT_COMPARE_METRICS
    result = analysis.compare_runs(runs, group_by, chosen, all_tasks)
    shown = ("final_phase_reached", "final_gap_points", "attempts", "llm_calls", "run_seconds", "memories_used")
    result["runs"] = [{
        "run_id": r["conditions"]["run_id"], "started_at": r["conditions"].get("started_at"),
        "status": r["conditions"].get("status"), "topic": r["conditions"].get("topic"),
        "group": r["conditions"].get(group_by), "memory_exposure": r["conditions"].get("memory_exposure"),
        "metrics_version": r["conditions"].get("metrics_version"), "task": analysis.task_of(r["conditions"]),
        "values": {m: r["metrics"].get(("run", m), {}).get("value") for m in shown},
    } for r in runs]
    result["group_fields"] = list(analysis.GROUP_FIELDS)
    result["groups_available"] = sorted({r["conditions"].get("comparison_group") for r in load_runs(db_file)
                                         if (r["conditions"].get("comparison_group") or "").startswith("test:")})
    return result


# --- Memory ---

def _decode_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if key in JSON_COLUMNS and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if isinstance(value, datetime):
            value = value.isoformat(timespec="seconds")
        out[key] = value
    out["effective_quality"] = analysis.effective_quality(out)
    return out


def _postgres_url() -> str:
    if os.environ.get("MEMORY_BACKEND", "sqlite").strip().lower() != "postgres":
        return ""
    return os.environ.get("MEMORY_DATABASE_URL", "").strip()


def _query_postgres(url: str, sql: str, params: tuple = ()) -> list:
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(url, connect_timeout=5, row_factory=dict_row, prepare_threshold=None) as conn:
        conn.read_only = True  # every transaction: BEGIN READ ONLY
        rows = conn.execute(sql.replace("?", "%s").replace("{schema}", "agent_memory."), params).fetchall()
        conn.rollback()
    return rows


def _query(local_db: Path, sql: str, params: tuple = ()) -> tuple[list, str]:
    """(rows, the store they came from). The hosted store when configured,
    else the local mirror; never written."""
    url = _postgres_url()
    if url:
        try:
            return [dict(r) for r in _query_postgres(url, sql, params)], "the team's hosted store (read-only)"
        except Exception as e:
            fallback = f"the local copy (the hosted store was not reachable: {type(e).__name__})"
    else:
        fallback = "the local memory store (read-only)"
    conn = _read_only(local_db)
    if conn is None:
        return [], "no memory store found"
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "memories" not in tables:
            return [], "no memory store found"
        return [dict(r) for r in conn.execute(sql.replace("{schema}", ""), params)], fallback
    finally:
        conn.close()


def _all_memories(local_db: Path) -> tuple[list, str]:
    key = ("memories", str(local_db), _postgres_url())
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
            return hit[1], hit[2]
    rows, source = _query(local_db, f"SELECT {', '.join(MEMORY_COLUMNS)} FROM {{schema}}memories")
    rows = [_decode_row(r) for r in rows]
    with _cache_lock:
        _cache[key] = (time.monotonic(), rows, source)
    return rows, source


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def list_memories(local_db: Path, agent: str = "", state: str = "active", search: str = "") -> dict:
    """Memories with their effective score, and the best and worst per agent."""
    rows, source = _all_memories(local_db)
    words = [w for w in search.lower().split() if w]
    chosen = [m for m in rows if (not agent or m["agent"] == agent) and (not state or m["state"] == state)
              and all(w in f"{m['title']} {m['summary']} {' '.join(m.get('tags') or [])}".lower() for w in words)]
    chosen.sort(key=lambda m: -m["effective_quality"])
    by_agent = {}
    for m in chosen:
        by_agent.setdefault(m["agent"], []).append(m)
    summary = {a: {"count": len(ms), "best": [m["id"] for m in ms[:BEST_WORST]],
                   "worst": [m["id"] for m in ms[-BEST_WORST:]] if len(ms) > BEST_WORST else []}
               for a, ms in sorted(by_agent.items())}
    states = {}
    for m in rows:
        states[m["state"]] = states.get(m["state"], 0) + 1
    return {"source": source, "total": len(rows), "states": states, "agents": sorted({m["agent"] for m in rows}),
            "memories": chosen, "by_agent": summary}


def memory_detail(local_db: Path, memory_id: str) -> dict | None:
    """One memory with its content, evidence, score reasons, state history and use."""
    rows, source = _query(local_db, f"SELECT {', '.join(MEMORY_COLUMNS)}, content FROM {{schema}}memories WHERE id = ?",
                          (memory_id,))
    if not rows:
        return None
    memory = _decode_row(rows[0])
    changes, _ = _query(local_db, "SELECT old_state, new_state, reason, actor, created_at FROM "
                                  "{schema}memory_state_changes WHERE memory_id = ? ORDER BY created_at", (memory_id,))
    usage, _ = _query(local_db, "SELECT run_id, agent, used, helped, created_at FROM {schema}memory_usage "
                                "WHERE memory_id = ? ORDER BY created_at", (memory_id,))
    memory["state_changes"] = [{k: (v.isoformat(timespec="seconds") if isinstance(v, datetime) else v)
                                for k, v in c.items()} for c in changes]
    memory["use"] = {"shown": len(usage), "used": sum(bool(u["used"]) for u in usage),
                     "helped": sum(u["helped"] in (True, 1) for u in usage),
                     "hurt": sum(u["helped"] in (False, 0) and u["helped"] is not None for u in usage),
                     "runs": sorted({u["run_id"] for u in usage if u.get("run_id")})}
    memory["source"] = source
    return memory
