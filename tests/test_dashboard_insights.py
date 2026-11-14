"""Milestone M9: the dashboard's Compare runs and Memory views (read-only,
the same numbers as the pipeline's CLI) and the Report Agent's memory section."""
import json
import os

import pytest

from src import memory
from src.dashboard import insights, server
from src.memory import learning, ranking
from src.memory import report as memory_report
from src.orchestration import run_metrics, run_store
from tests.test_comparison import fake_pipeline  # noqa: F401  (fixture)
from tests.test_dashboard import repo  # noqa: F401  (fixture)


def _finished_run(provider, gap, doi="10.1/x"):
    run_id = run_store.start_run("credit card fraud detection", llm_provider=provider)
    evaluation = {"metrics": [{"metric": "accuracy", "published_value": 0.95, "reproduced_value": 0.9,
                               "difference_points": gap, "classification": "APPROXIMATE"}]}
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", parent_paper={"title": "P", "doi": doi},
                         result={"evaluation_result": evaluation, "reruns": []})
    return run_id


@pytest.fixture
def client(repo, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "MEMORY_DB", memory.LOCAL_DB)  # conftest's temp memory store
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    insights.clear_cache()
    return server.create_app().test_client()


# --- Compare runs ---

def test_the_dashboard_shows_the_same_comparison_as_the_cli(client):
    for gap in (1.0, 2.0, 3.0):
        _finished_run("openrouter", gap)
    for gap in (6.0, 7.0, 8.0):
        _finished_run("freellmapi", gap)
    cli = run_metrics.compare_runs(run_metrics.load_runs(), group_by="llm_provider")
    response = client.get("/api/compare?group_by=llm_provider")
    assert response.status_code == 200
    data = response.get_json()
    assert data["blocks"] == json.loads(json.dumps(cli["blocks"]))  # identical numbers, tests and confounders
    assert data["kind"] == cli["kind"] == "descriptive" and len(data["runs"]) == 6
    assert data["blocks"][0]["tests"]["final_gap_points"]["p_value"] == pytest.approx(0.1)


def test_compare_filters_and_refuses_an_unknown_grouping(client):
    _finished_run("openrouter", 1.0)
    assert client.get("/api/compare?topic=lung").get_json()["runs"] == []
    assert len(client.get("/api/compare?topic=fraud").get_json()["runs"]) == 1
    assert client.get("/api/compare?group_by=colour").status_code == 400


def test_a_controlled_test_is_selectable_and_reported_as_controlled(client, fake_pipeline):  # noqa: F811
    from src.orchestration import comparison
    comparison.make_plan("t1", ["machine learning for detecting credit card fraud"], ["off", "read"], 1)
    while not comparison.run_one("t1").get("done"):
        pass
    data = client.get("/api/compare?group=test:t1&group_by=memory_mode").get_json()
    assert data["groups_available"] == ["test:t1"] and data["kind"] == "controlled" and len(data["runs"]) == 2


def test_no_run_database_is_an_empty_comparison_not_an_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "RUN_DB", tmp_path / "missing.db")
    assert client.get("/api/compare").get_json()["runs"] == []


# --- Memory ---

def _put(**m):
    base = {"kind": "solution", "agent": "Replication Agent", "polarity": "positive", "title": "a case",
            "summary": "a summary", "quality": 0.6, "lineage": "kept", "task": {"model_family": ["xgboost"]},
            "content": "print(1)"}
    base.update(m)
    return memory.put_memory(base)


def test_the_memory_view_ranks_by_the_same_effective_quality_as_recall(client):
    good = _put(title="xgboost script that helped", quality=0.6)
    bad = _put(title="script that hurt", quality=0.7)
    for _ in range(3):
        learning.record_outcome(memory.record_usage({"memory_id": good, "used": True}), good, True)
        learning.record_outcome(memory.record_usage({"memory_id": bad, "used": True}), bad, False)
    _put(title="quarantined one", state="quarantined")
    data = client.get("/api/memories").get_json()
    assert [m["id"] for m in data["memories"]] == [good, bad]  # active only, best first
    for m in data["memories"]:
        assert m["effective_quality"] == ranking.effective_quality(memory.get_memory(m["id"]))
    assert data["source"] == "the local memory store (read-only)" and data["states"] == {"active": 2, "quarantined": 1}
    assert data["by_agent"]["Replication Agent"]["best"][0] == good
    assert [m["title"] for m in client.get("/api/memories?state=&q=quarantined").get_json()["memories"]] == \
        ["quarantined one"]


def test_one_memory_shows_its_content_evidence_reasons_state_history_and_use(client):
    mid = _put(title="the case", quality_reasons={"outcome": {"value": 0.9, "reason": "gap 2 points"}})
    learning.record_outcome(memory.record_usage({"memory_id": mid, "used": True, "run_id": "r1"}), mid, True)
    memory.set_state(mid, "quarantined", "copied a metric", actor="tester")
    detail = client.get(f"/api/memories/{mid}").get_json()
    assert detail["content"] == "print(1)" and detail["quality_reasons"]["outcome"]["reason"] == "gap 2 points"
    assert detail["state_changes"][0]["reason"] == "copied a metric" and detail["state_changes"][0]["actor"] == "tester"
    assert detail["use"] == {"shown": 1, "used": 1, "helped": 1, "hurt": 0, "runs": ["r1"]}
    assert client.get("/api/memories/no-such-id").status_code == 404


def test_the_memory_view_never_writes_to_the_store(client):
    mid = _put()
    before = os.stat(memory.LOCAL_DB).st_mtime_ns, memory.client().local.count("memories")
    client.get("/api/memories")
    client.get(f"/api/memories/{mid}")
    assert (os.stat(memory.LOCAL_DB).st_mtime_ns, memory.client().local.count("memories")) == before


def test_an_unreachable_hosted_store_falls_back_to_the_local_copy(client, monkeypatch):
    _put(title="local copy")
    monkeypatch.setenv("MEMORY_BACKEND", "postgres")
    monkeypatch.setenv("MEMORY_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/db")
    insights.clear_cache()
    data = client.get("/api/memories").get_json()
    assert data["source"].startswith("the local copy (the hosted store was not reachable")
    assert [m["title"] for m in data["memories"]] == ["local copy"]


def test_the_memory_view_without_a_store(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MEMORY_DB", tmp_path / "missing.db")
    insights.clear_cache()
    data = client.get("/api/memories").get_json()
    assert data["memories"] == [] and data["source"] == "no memory store found"


# --- The Report Agent's memory section ---

def test_the_memory_used_section_names_fast_mode_as_adapted_and_gives_outcomes(monkeypatch):
    source = _put(title="kept xgboost baseline", run_id="run-a")
    unused = _put(title="shown but not used", run_id="run-b")
    run_id = run_store.start_run("fraud")
    run_store.record_stage(run_id, "Replication Agent", 1, "Approved", {"fast_mode_from": source})
    used_row = memory.record_usage({"memory_id": source, "run_id": run_id, "agent": "Replication Agent", "used": True})
    memory.record_usage({"memory_id": unused, "run_id": run_id, "agent": "Replication Agent", "used": False})
    learning.record_outcome(used_row, source, False)
    rows = memory_report.memory_used(run_id)
    assert [(r["title"], r["used"], r["fast_mode"], r["outcome"]) for r in rows] == [
        ("kept xgboost baseline", True, True, "followed by a bad outcome"),
        ("shown but not used", False, False, "not used")]
    text = memory_report.render_markdown(run_id=run_id)
    assert "adapted from the script of run run-a (fast mode), not written anew" in text
    assert "is the agents' own report" in text


def test_every_number_in_the_test_section_cites_its_run_metrics_rows(fake_pipeline):  # noqa: F811
    from src.orchestration import comparison
    comparison.make_plan("t1", ["machine learning for detecting credit card fraud"], ["off", "read"], 3)
    while not comparison.run_one("t1").get("done"):
        pass
    data = memory_report.comparison("t1")
    gap_rows = [r for r in data["rows"] if r["metric"] == "final_gap_points"]
    assert {r["condition"] for r in gap_rows} == {"off", "read"}
    for r in gap_rows:
        assert r["n"] == 3 and len(r["sources"]) == 3
        for s in r["sources"]:  # each cited value is the stored run_metrics value
            run = next(x for x in run_metrics.load_runs() if x["conditions"]["run_id"] == s["run_id"])
            assert run["metrics"][("run", "final_gap_points")]["value"] == s["value"]
        assert r["median"] == sorted(s["value"] for s in r["sources"])[1]
    text = memory_report.render_markdown(test_name="t1")
    assert "| final_gap_points | read | 3/3 |" in text
    assert gap_rows[0]["sources"][0]["run_id"] in text and "Permutation tests" in text
