import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src import memory
from src.memory import schema, scrub
from src.memory.backends.base import RemoteUnavailable, check_op
from src.memory.backends.sqlite import SqliteStore
from src.orchestration import run_store


class FakeRemote(SqliteStore):
    """A second SQLite file standing in for the hosted store, which can be
    switched offline (every call then raises RemoteUnavailable)."""

    def __init__(self, path):
        self.online = True
        self.reject = set()  # op ids the store refuses (a permanent error)
        super().__init__(path)

    def _check(self):
        if not self.online:
            raise RemoteUnavailable("offline")

    def session(self):
        self._check()
        return super().session()

    def apply(self, op, conn=None):
        self._check()
        if op["op_id"] in self.reject:
            raise ValueError("rejected by the store")
        return super().apply(op)

    def fetch_memories(self, *args, **kwargs):
        self._check()
        return super().fetch_memories(*args, **kwargs)

    def schema_version(self, conn=None):
        self._check()
        return super().schema_version()


@pytest.fixture
def remote(tmp_path):
    return FakeRemote(tmp_path / "remote.db")


@pytest.fixture
def hosted(tmp_path, remote):
    """The process's client in postgres mode, with the fake hosted store."""
    client = memory.MemoryClient("postgres", tmp_path / "local.db", remote=remote)
    memory.reset_client(client)
    return client


def _memory(**overrides):
    base = {"kind": "solution", "agent": "Replication Agent", "polarity": "positive",
            "title": "XGBoost baseline", "summary": "Stratified split, no leakage.", "tags": ["xgboost"],
            "content": "print('hi')\n"}
    base.update(overrides)
    return base


# --- Schema ---

def _sql_columns(sql: str) -> dict:
    tables = {}
    for name, body in re.findall(r"CREATE TABLE (?:agent_memory\.)?(\w+) \((.*?)\n\);", sql, re.S):
        cols = [line.strip().split()[0] for line in body.splitlines()
                if line.strip() and not line.strip().startswith(("PRIMARY", "--", "CHECK"))
                and line.startswith("    ") and not line.startswith("     ")]
        tables[name] = [c for c in cols if c != "fts"]
    return tables


@pytest.mark.parametrize("dialect", ["sqlite", "postgres"])
def test_migrations_match_the_schema_module(dialect):
    from src.memory.backends.base import migration_files
    sql = "\n".join(f[2] for f in migration_files(dialect))
    tables = _sql_columns(sql)
    for table, columns in schema.TABLES.items():
        assert tables[table] == list(columns), f"{dialect} {table}"


def test_local_store_migrates_once_and_detects_a_changed_migration(tmp_path, monkeypatch):
    store = SqliteStore(tmp_path / "m.db")
    assert store.schema_version() == schema.SCHEMA_VERSION
    assert store.migrate() == []
    from src.memory.backends import sqlite as sqlite_backend
    original = sqlite_backend.migration_files
    monkeypatch.setattr(sqlite_backend, "migration_files",
                        lambda d: [(v, n, s + "\n-- edited", "other") for v, n, s, _ in original(d)])
    with pytest.raises(Exception, match="changed after it was applied"):
        store.migrate()


def test_schema_version_is_the_codes_not_the_number_of_migration_files(tmp_path, monkeypatch):
    from src.memory.backends import sqlite as sqlite_backend
    original = sqlite_backend.migration_files
    monkeypatch.setattr(sqlite_backend, "migration_files",
                        lambda d: original(d) + [(99, "099_extra.sql", "CREATE TABLE extra (x TEXT);", "c2")])
    store = SqliteStore(tmp_path / "m.db")
    assert store.schema_version() == schema.SCHEMA_VERSION
    store.meta_set("schema_version", str(schema.SCHEMA_VERSION + 1))  # migrated by newer code
    with pytest.raises(Exception, match="pull the latest code"):
        store.migrate()


# --- Local store (MEMORY_BACKEND=sqlite) ---

def test_put_get_and_list_with_defaults(monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_CONTRIBUTOR", "aiden")
    memory_id = memory.put_memory(_memory())
    stored = memory.get_memory(memory_id)
    assert stored["state"] == "active" and stored["lineage"] == "unknown" and stored["trust"] == "normal"
    assert stored["tags"] == ["xgboost"] and stored["contributor"] == "aiden"
    assert stored["code_version"] == "test-commit" and stored["schema_version"] == schema.SCHEMA_VERSION
    assert stored["content_sha"] and stored["search_text"] == "XGBoost baseline Stratified split, no leakage. xgboost"
    assert [m["id"] for m in memory.list_memories(agent="Replication Agent")] == [memory_id]
    assert memory.list_memories(agent="Data Agent") == []


def test_content_and_summary_are_cut_to_their_limits():
    memory_id = memory.put_memory(_memory(content="x" * (schema.MAX_CONTENT_CHARS + 10), summary="s" * 5000))
    stored = memory.get_memory(memory_id)
    assert len(stored["content"]) == schema.MAX_CONTENT_CHARS and "[cut: " in stored["content"]
    assert len(stored["summary"]) == schema.MAX_SUMMARY_CHARS


def test_memories_are_append_only():
    memory_id = memory.put_memory(_memory())
    assert memory.put_memory(_memory(id=memory_id, title="replaced")) == memory_id
    assert memory.get_memory(memory_id)["title"] == "XGBoost baseline"


def test_counters_and_usage(capsys):
    memory_id = memory.put_memory(_memory())
    assert memory.increment(memory_id, times_used=1, times_helped=1)
    stored = memory.get_memory(memory_id)
    assert stored["times_used"] == 1 and stored["times_helped"] == 1
    assert memory.increment(memory_id, title=1) is False  # only counters may change
    assert "could not update the counters" in capsys.readouterr().out
    usage_id = memory.record_usage({"memory_id": memory_id, "run_id": "r1", "agent": "Replication Agent",
                                    "used": True, "query": {"model_family": "xgboost"}, "not_a_column": 1})
    assert usage_id and memory.client().local.count("memory_usage") == 1


def test_bad_records_are_refused_fail_soft(capsys):
    assert memory.put_memory(_memory(kind="opinion")) is None
    assert memory.put_memory({"kind": "solution"}) is None
    assert "could not store a memory" in capsys.readouterr().out
    with pytest.raises(ValueError):
        check_op({"op_id": "1", "type": "insert", "table": "runs", "rows": [{"run_id": "x"}]})


def test_memory_off_does_nothing():
    memory.reset_client(memory.MemoryClient("off"))
    assert memory.put_memory(_memory()) is None
    assert memory.list_memories() == [] and memory.get_memory("x") is None
    assert memory.sync()["error"].startswith("nothing to sync")


def test_a_broken_store_never_raises(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SqliteStore, "apply", broken)
    monkeypatch.setattr(SqliteStore, "fetch_memories", broken)
    assert memory.put_memory(_memory()) is None
    assert memory.list_memories() == []
    assert "disk I/O error" in capsys.readouterr().out


def test_unknown_backend_turns_memory_off(monkeypatch, capsys):
    monkeypatch.setattr(memory, "MEMORY_BACKEND", "mongo")
    memory.reset_client()
    assert memory.backend_name() == "off"
    assert "memory is off for this process" in capsys.readouterr().out


# --- Scrubber ---

def test_scrubber_removes_secrets_and_keeps_settings(tmp_path):
    scrub.ENV_FILE.write_text("OPENROUTER_API_KEY=or-secret-value-123\nLLM_PROVIDER=freellmapi\n"
                              "MEMORY_DATABASE_URL=postgresql://memory_agent:pw123456@db.example.com/postgres\n",
                              encoding="utf-8")
    scrub._env_values.cache_clear()
    text = ("key or-secret-value-123 provider freellmapi\n"
            "Authorization: Bearer abcdef1234567890\n"
            "url postgresql://memory_agent:pw123456@db.example.com/postgres\n"
            "SLACK_BOT_TOKEN=xoxb-1234567890-abcdef\n"
            "token sk-ant-api03-abcdefghijklmnop\n"
            "mail someone@example.com\n"
            "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----\n"
            "max_tokens = 64000\nsha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n")
    out = scrub.scrub_text(text)
    for secret in ("or-secret-value-123", "abcdef1234567890", "pw123456", "xoxb-1234567890", "sk-ant-api03",
                   "someone@example.com", "MIIabc"):
        assert secret not in out
    assert "freellmapi" in out and "max_tokens = 64000" in out and "9f86d081884c7d65" in out
    assert scrub.scrub({"a": ["or-secret-value-123"]}) == {"a": ["[REDACTED:env_value]"]}


def test_put_memory_scrubs_before_writing():
    memory_id = memory.put_memory(_memory(content="api_key = 'sk-abcdefghijklmnopqrstu'"))
    assert "sk-abcdefghij" not in memory.get_memory(memory_id)["content"]


# --- Hosted store (MEMORY_BACKEND=postgres), outbox and sync ---

def test_offline_writes_are_kept_and_sent_once(hosted, remote):
    remote.online = False
    memory_id = memory.put_memory(_memory())
    assert memory_id and hosted.local.outbox_counts()["pending"] == 1
    report = memory.sync()
    assert report["error"] == "offline" and hosted.local.outbox_counts()["pending"] == 1

    remote.online = True
    report = memory.sync()
    assert report["sent"] == 1 and report["error"] == ""
    assert remote.count("memories") == 1 and hosted.local.outbox_counts() == {"pending": 0, "sent": 1, "failed": 0}
    assert memory.sync()["sent"] == 0
    assert remote.count("memories") == 1


def test_a_resent_op_is_applied_once(hosted, remote):
    memory_id = memory.put_memory(_memory())
    memory.increment(memory_id, times_used=1)
    for op in hosted.local.pending_ops():
        remote.apply(op)  # the store applied them, but the acknowledgement was lost
    report = memory.sync()
    assert report["already_applied"] == 2 and report["sent"] == 0
    assert remote.fetch_memories(ids=[memory_id])[0]["times_used"] == 1


def test_a_rejected_op_is_retried_then_set_aside_never_lost(hosted, remote):
    memory.put_memory(_memory())
    op_id = hosted.local.pending_ops()[0]["op_id"]
    remote.reject.add(op_id)
    for _ in range(memory.MAX_SEND_ATTEMPTS):
        memory.sync()
    assert hosted.local.outbox_counts()["failed"] == 1
    with sqlite3.connect(hosted.local.path) as conn:
        error = conn.execute("SELECT last_error FROM outbox WHERE op_id = ?", (op_id,)).fetchone()[0]
    assert "rejected by the store" in error


def test_sync_refuses_a_store_with_another_schema_version(hosted, remote):
    memory.put_memory(_memory())
    remote.meta_set("schema_version", str(schema.SCHEMA_VERSION + 1))
    report = memory.sync()
    assert "pull the latest code" in report["error"]
    assert remote.count("memories") == 0 and hosted.local.outbox_counts()["pending"] == 1


def test_reads_use_the_hosted_store_and_a_recent_local_copy_when_offline(hosted, remote):
    memory.put_memory(_memory())
    memory.sync()
    assert len(memory.list_memories()) == 1  # from the hosted store, cached locally
    remote.online = False
    hosted._remote_down_until = 0
    assert len(memory.list_memories()) == 1  # local copy, synced just now
    old = (datetime.now(timezone.utc) - memory.CACHE_MAX_AGE - timedelta(hours=1)).isoformat(timespec="seconds")
    hosted.local.meta_set("last_sync", old)
    assert memory.list_memories() == []  # too old: a quarantine may be missing from it


def test_sync_refreshes_the_state_of_local_copies(hosted, remote):
    memory_id = memory.put_memory(_memory())
    memory.sync()
    with sqlite3.connect(remote.path) as conn:  # an admin quarantines it on the hosted store
        conn.execute("UPDATE memories SET state = 'quarantined' WHERE id = ?", (memory_id,))
    memory.sync()
    assert hosted.local.fetch_memories(ids=[memory_id], states=None)[0]["state"] == "quarantined"
    remote.online = False
    hosted._remote_down_until = 0
    assert memory.list_memories() == []


def test_run_metrics_are_uploaded_once(hosted, remote):
    run_id = run_store.start_run("fraud detection")
    run_store.finish_run(run_id, "halted", "TOPIC_ANALYSIS", result={"phase": "TOPIC_ANALYSIS"})
    # finish_run computed the metrics and synced them
    assert remote.count("run_conditions") == 1 and remote.count("run_metrics") > 10
    report = memory.sync()
    assert report["runs_queued"] == 0 and report["sent"] == 0
    assert remote.count("run_conditions") == 1


def test_runs_measured_before_a_backend_was_configured_are_sent_later(tmp_path, remote):
    run_id = run_store.start_run("fraud detection")  # MEMORY_BACKEND=sqlite: nothing is sent
    run_store.finish_run(run_id, "halted", "TOPIC_ANALYSIS", result={})
    assert remote.count("run_conditions") == 0
    memory.reset_client(memory.MemoryClient("postgres", tmp_path / "local.db", remote=remote))
    assert memory.sync()["runs_queued"] == 1
    assert remote.count("run_conditions") == 1


def test_status_and_cli(hosted, capsys):
    from src.memory.__main__ import _main
    memory.put_memory(_memory())
    info = memory.status()
    assert info["backend"] == "postgres" and info["outbox"]["pending"] == 1 and info["cache_usable"] is False
    assert _main(["sync"]) == 0
    assert _main(["list"]) == 0
    assert "XGBoost baseline" in capsys.readouterr().out


def test_postgres_encoding():
    from psycopg.types.json import Jsonb
    from src.memory.backends import postgres
    row = postgres.encode("memories", {"task": {"a": 1}, "tags": ["x", 2], "created_at": "2026-11-13T10:00:00+00:00",
                                       "title": "t", "seen_count": 1})
    assert isinstance(row["task"], Jsonb) and row["tags"] == ["x", "2"]
    assert row["created_at"] == datetime(2026, 11, 13, 10, tzinfo=timezone.utc)
    decoded = postgres.decode("memories", {"created_at": row["created_at"], "fts": "ignored", "title": "t"})
    assert decoded == {"created_at": "2026-11-13T10:00:00+00:00", "title": "t"}
