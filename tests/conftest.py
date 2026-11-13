"""
Redirect every file the agents persist to a per-test temp directory.

Without this, pipeline tests wrote their fixture data into the real logs/
caches -- e.g. a fake "Parent Paper" in validated_papers_cache.json and a
fake Kaggle repo in code_discovery_cache.json -- which a later live run
could then read back as if it were real.
"""
import pytest
import requests

from src import llm, memory
from src.memory import scrub as memory_scrub
from src.slack import slack_client

from src.agents import (
    cleaning_agent, data_agent, master_agent, replication_agent,
)
from src.orchestration import (
    code_discovery_cache, column_mapping, dataset_cache, run_context, run_store, state_manager,
)
from src.search import camoufox_fetch, page_fetch, paper_fulltext, serpapi_search, web_search


@pytest.fixture(autouse=True)
def isolate_persisted_files(tmp_path, monkeypatch):
    # Tests must not post status messages or read a real team's channel.
    # Slack-specific tests explicitly install their own token/mock client.
    monkeypatch.setattr(slack_client, "SLACK_BOT_TOKEN", "")
    monkeypatch.setattr(slack_client, "_client", None)
    logs = tmp_path / "logs"
    monkeypatch.setattr(master_agent, "LOG_FILE", logs / "master_agent_log.json")
    monkeypatch.setattr(master_agent, "REPORT_DIR", logs / "agent_reports")
    monkeypatch.setattr(state_manager, "STATE_FILE", logs / "project_state.json")
    monkeypatch.setattr(dataset_cache, "VALIDATED_PAPERS_CACHE", logs / "validated_papers_cache.json")
    monkeypatch.setattr(dataset_cache, "DATASET_SEARCH_CACHE", logs / "dataset_search_cache.json")
    monkeypatch.setattr(code_discovery_cache, "CODE_DISCOVERY_CACHE", logs / "code_discovery_cache.json")
    monkeypatch.setattr(cleaning_agent, "CONFLICTS_LOG", logs / "cleaning_conflicts.json")
    monkeypatch.setattr(cleaning_agent, "PROCESSED_DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(data_agent, "RAW_DATA_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(replication_agent, "REPLICATION_DIR", tmp_path / "replication")
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "test-key")

    # A test that forgets to mock an LLM call must fail loudly, never reach
    # the real API. Tests that patch requests.post themselves override this.
    def _network_disabled(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests (requests.post/get was not mocked).")

    monkeypatch.setattr(requests, "post", _network_disabled)
    monkeypatch.setattr(requests, "get", _network_disabled)
    # Existing LLM tests mock requests.post (the OpenRouter path); the
    # Claude path is tested separately with a mocked SDK client.
    monkeypatch.setattr(llm, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(cleaning_agent, "MAPPINGS_LOG", logs / "column_mappings.json")
    monkeypatch.setattr(column_mapping, "MAPPING_CACHE", logs / "column_mapping_cache.json")
    monkeypatch.setattr(llm, "CALL_LOG", logs / "llm_calls.jsonl")
    monkeypatch.setattr(run_store, "DB_FILE", tmp_path / "database" / "project_runs.db")
    monkeypatch.setattr(run_store, "_project_git_commit", lambda: "test-commit")
    monkeypatch.setattr(run_store, "_project_git_dirty", lambda: False)
    # A run a test opened and never finished must not tag later tests' log lines.
    monkeypatch.setattr(run_context, "_run", {})
    monkeypatch.setattr(master_agent, "ROUND_BACKUP_DIR", logs / "round_backups")
    # The Parent Paper's full text: no download (no open-access links) and
    # no reading of a real papers/parent/ cache, so agents fall back to the
    # paper's stored metadata. tests/test_paper_fulltext.py tests the real
    # lookup with mocked HTTP.
    monkeypatch.setattr(paper_fulltext, "PARENT_PAPER_DIR", tmp_path / "papers" / "parent")
    monkeypatch.setattr(paper_fulltext, "candidate_pdf_urls", lambda parent_paper: [])
    # Search API calls are mocked; isolate provider counters and fetch logs.
    monkeypatch.setattr(serpapi_search, "SERPENT_API_KEY", "")
    # Browser tests opt in explicitly with a mocked launcher; no live browser in unit tests.
    monkeypatch.setattr(camoufox_fetch, "CAMOUFOX_ENABLED", False)
    monkeypatch.setattr(web_search, "_failures", {name: 0 for name, _m in web_search.PROVIDERS})
    monkeypatch.setattr(web_search, "SEARCH_LOG", logs / "search_calls.jsonl")
    monkeypatch.setattr(page_fetch, "FETCH_LOG", logs / "page_fetch_calls.jsonl")
    # Memory store: a temp SQLite file, never the real database/memory_local.db
    # or a hosted store, whatever .env says; the scrubber does not read the real .env.
    monkeypatch.setattr(memory, "MEMORY_BACKEND", "sqlite")
    monkeypatch.setattr(memory, "LOCAL_DB", tmp_path / "database" / "memory_local.db")
    monkeypatch.setattr(memory_scrub, "ENV_FILE", tmp_path / ".env")
    memory_scrub._env_values.cache_clear()
    memory.reset_client()
    yield
    memory.reset_client()
    memory_scrub._env_values.cache_clear()
