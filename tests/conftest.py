"""
Redirect every file the agents persist to a per-test temp directory.

Without this, pipeline tests wrote their fixture data into the real logs/
caches -- e.g. a fake "Parent Paper" in validated_papers_cache.json and a
fake Kaggle repo in code_discovery_cache.json -- which a later live run
could then read back as if it were real.
"""
import pytest
import requests

from src import llm

from src.agents import (
    cleaning_agent, data_agent, experiment_agent, master_agent, replication_agent,
)
from src.orchestration import code_discovery_cache, column_mapping, dataset_cache, run_store, state_manager


@pytest.fixture(autouse=True)
def isolate_persisted_files(tmp_path, monkeypatch):
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
    monkeypatch.setattr(experiment_agent, "REPLICATION_DIR", tmp_path / "replication")
    monkeypatch.setattr(experiment_agent, "EXPERIMENTS_DIR", tmp_path / "experiments")
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
