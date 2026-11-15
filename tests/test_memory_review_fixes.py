"""The memory-bank findings of PROJECT_REVIEW_2026-11-14.md (1-14), one
group of tests per finding, each written as the review's failure scenario."""
import json
import os
import uuid

import pytest

from src import config, llm, memory
from src.agents import master_agent, replication_agent
from src.memory import learning, recall, retain
from src.memory.backends.sqlite import SqliteStore
from src.orchestration import comparison, run_context, run_store
from tests.test_memory_learning import GOOD_EVALUATION, _run_that_uses_a_memory
from tests.test_memory_retain import SCRIPT, _memories, make_run
from tests.test_memory_store import FakeRemote


def _put(**m):
    base = {"kind": "solution", "agent": "Replication Agent", "polarity": "positive", "title": "a case",
            "summary": "a summary", "quality": 0.8, "lineage": "kept", "task": {"model_family": ["xgboost"]},
            "created_at": "2026-01-01T00:00:00+00:00"}
    base.update(m)
    return memory.put_memory(base)


def _scores(info):
    run_context.set_run("comparison-run", **info)
    recall.clear_cache()
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    return [(i["memory"]["id"], i["score"]) for i in ctx.items]


# --- 1. The frozen comparison bank does not learn ---

def test_a_comparison_ranks_by_the_bank_as_frozen_in_its_plan(monkeypatch):
    a = _put(title="xgboost case A")
    b = _put(title="xgboost case B", quality=0.7)
    plan = comparison.make_plan("frozen", ["credit card fraud"])
    info = {"memory_snapshot": plan["memory_snapshot"], "memory_bank_file": plan["memory_bank_file"]}
    before = _scores(info)

    # The live bank changes after the plan: A is hurt three times, B is quarantined.
    run_context.clear()
    for _ in range(3):
        learning.record_outcome(memory.record_usage({"memory_id": a, "used": True}), a, False)
    memory.set_state(b, "quarantined", "a teammate's review")
    assert memory.get_memory(a)["times_hurt"] == 3

    assert _scores(info) == before  # the test's runs see the same ranking
    assert {mid for mid, _ in before} == {a, b}
    assert comparison.bank_intact(plan)


def test_read_mode_records_use_but_never_changes_the_bank(monkeypatch):
    mid = _put()
    monkeypatch.setattr(config, "MEMORY_MODE", "read")
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: '{"ok": true, "memory_used": ["M1"]}')
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    llm.chat("write it", memory=ctx)
    stored = memory.get_memory(mid)
    assert (stored["times_recalled"], stored["times_used"]) == (0, 0)  # the bank's counters did not move
    assert len(memory.client().local.rows("memory_usage", memory_id=mid)) == 1  # exposure is still measured
    usage_id = memory.client().local.rows("memory_usage", memory_id=mid)[0]["id"]
    assert learning.record_outcome(usage_id, mid, False) is False
    assert memory.get_memory(mid)["times_hurt"] == 0
    assert learning.learn_run("any-run")["skipped"]


# --- 2 and 3. The hosted store and the local mirror ---

@pytest.fixture
def hosted(tmp_path):
    remote = FakeRemote(tmp_path / "remote.db")
    client = memory.MemoryClient("postgres", tmp_path / "local.db", remote=remote)
    memory.reset_client(client)
    return client, remote


def test_a_hosted_read_keeps_this_machines_unsent_changes(hosted):
    client, remote = hosted
    mid = _put()
    memory.sync()  # the hosted store has it, active, with zero counters
    memory.set_state(mid, "quarantined", "found leaky here")
    memory.increment(mid, times_recalled=1)

    assert memory.find_memories(agent="Replication Agent") == []  # the hosted row is active, the local change wins
    local = client.local.rows("memories", id=mid)[0]
    assert (local["state"], local["times_recalled"]) == ("quarantined", 1)

    remote.online = False
    client._remote_down_until = 0.0
    assert memory.find_memories(agent="Replication Agent") == []  # offline: still quarantined
    remote.online = True
    memory.sync()
    assert remote.fetch_memories(ids=[mid], states=None)[0]["state"] == "quarantined"


def test_a_memory_made_here_and_not_sent_is_in_a_hosted_read(hosted):
    mid = _put(title="made here")
    assert [m["id"] for m in memory.find_memories(agent="Replication Agent")] == [mid]


def test_a_memory_quarantined_on_the_hosted_store_does_not_come_back_offline(hosted):
    client, remote = hosted
    mid = _put()
    memory.sync()
    assert [m["id"] for m in memory.find_memories(agent="Replication Agent")] == [mid]

    # A teammate quarantines it on the hosted store.
    remote.apply({"op_id": str(uuid.uuid4()), "type": "set_state", "table": "memories", "key": {"id": mid},
                  "state": "quarantined"})
    assert memory.find_memories(agent="Replication Agent") == []  # the hosted answer omits it...
    assert client.local.rows("memories", id=mid)[0]["state"] == "quarantined"  # ...and the copy is reconciled

    remote.online = False
    client._remote_down_until = 0.0
    assert memory.find_memories(agent="Replication Agent") == []  # no resurrection during the outage


# --- 4. Same content, different outcome: a new observation ---

def test_a_success_later_found_leaky_keeps_the_warning_and_supersedes_the_success(tmp_path, monkeypatch):
    first = make_run(tmp_path, monkeypatch, gap=1.0, name="first")
    success = next(m for m in _memories(first, agent="Replication Agent") if m["polarity"] == "positive")
    leak = {"check": "smote_before_split", "issue": "SMOTE is fitted before the split", "category": "data_leakage",
            "severity": "critical", "evidence": "SMOTE().fit_resample(X, y)"}
    second = make_run(tmp_path, monkeypatch, gap=1.0, leaks=[leak], name="second")  # the same SCRIPT

    warning = next(m for m in _memories(second, agent="Replication Agent") if m["title"].startswith("Leaky"))
    assert warning["content_sha"] == success["content_sha"] and warning["polarity"] == "negative"
    old = memory.get_memory(success["id"])
    assert old["state"] == "superseded" and old["superseded_by"] == warning["id"]


def test_a_success_after_a_warning_of_the_same_content_is_kept_and_linked(tmp_path, monkeypatch):
    leak = {"check": "smote_before_split", "issue": "SMOTE is fitted before the split", "category": "data_leakage",
            "severity": "critical", "evidence": "SMOTE().fit_resample(X, y)"}
    first = make_run(tmp_path, monkeypatch, gap=1.0, leaks=[leak], name="first")
    warning = next(m for m in _memories(first, agent="Replication Agent") if m["title"].startswith("Leaky"))
    second = make_run(tmp_path, monkeypatch, gap=1.0, name="second")

    success = next(m for m in _memories(second, agent="Replication Agent") if m["polarity"] == "positive")
    assert success["content_sha"] == warning["content_sha"]
    assert warning["id"] in success["quality_reasons"]["contradicted_by"]
    assert success["id"] in memory.get_memory(warning["id"])["quality_reasons"]["contradicted_by"]


# --- 5. A failed write leaves the run to be retained again ---

def test_a_failed_write_does_not_mark_the_run_retained_and_a_retry_repairs_it(tmp_path, monkeypatch):
    real_put = memory.put_memory
    failing = {"on": True}
    monkeypatch.setattr(memory, "put_memory",
                        lambda row: None if failing["on"] and row["agent"] == "Replication Agent" else real_put(row))
    monkeypatch.setattr(config, "MEMORY_MODE", "read_write")
    run_id = make_run(tmp_path, monkeypatch, gap=1.0)

    assert memory.client().local.rows("memory_runs", run_id=run_id) == []
    assert not _memories(run_id, agent="Replication Agent")
    failing["on"] = False
    report = retain.retain_run(run_id)
    assert report["skipped"] is None and not report["failed"]
    assert memory.client().local.rows("memory_runs", run_id=run_id)
    assert _memories(run_id, agent="Replication Agent")


def test_a_retried_duplicate_counts_once():
    mid = _put()
    op = retain._op_id("run-1", "seen", mid, "row")
    assert memory.increment(mid, op_id=op, seen_count=1) and memory.increment(mid, op_id=op, seen_count=1)
    assert memory.get_memory(mid)["seen_count"] == 2  # 1 at insert + 1 once


# --- 6. Research Agent output is retained ---

def test_an_approved_research_agent_list_is_retained(monkeypatch):
    run_id = run_store.start_run("credit card fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    papers = [{"title": "A fraud paper", "year": 2024, "venue": "Expert Syst. Appl.", "doi": "10.1/x"}]
    master_agent._supervise("Research Agent", lambda: papers, lambda out: [], lambda out: "report", max_attempts=1)
    run_store.finish_run(run_id, "halted", "LITERATURE_SEARCH", result={"reruns": []})
    research = _memories(run_id, agent="Research Agent")
    assert research and "A fraud paper" in research[0]["content"]


# --- 7. Every recall describes the run's task ---

def test_recall_uses_the_runs_task_record():
    run_context.set_run("task-run")
    run_context.update_task(topic="credit card fraud detection", intake={"ml_task": "binary classification"},
                            dataset={"selected_datasets": [{"name": "https://kaggle.com/d/carclaims"}]})
    ctx = recall.for_agent("Replication Agent", parent_paper={"title": "P", "methodology": "We train XGBoost."})
    assert ctx.query["task_type"] == "binary"
    assert ctx.query["model_family"] == ["xgboost"]
    assert ctx.query["topic"] == "credit card fraud detection"
    assert ctx.query["dataset_identity"] == ["https://kaggle.com/d/carclaims"]


def test_the_replication_agents_recall_reads_the_current_script():
    run_context.set_run("task-run")
    run_context.update_task(intake={"ml_task": "classification"})
    ctx = replication_agent._memory_context({"title": "P"}, script="from lightgbm import LGBMClassifier\n")
    assert ctx.query["model_family"] == ["lightgbm"] and ctx.query["task_type"] == "binary"


def test_fast_mode_needs_a_known_task_type():
    _put(title="kept xgboost", content=SCRIPT, quality=0.95,
         task={"model_family": ["xgboost"], "task_type": "binary", "dataset_traits": {"rows": "1k-10k"}})
    ctx = recall.context_for("Replication Agent", {"model_family": ["xgboost"]})  # task type unknown
    assert ctx.fast_mode_case() is None


# --- 9. Paging past the newest candidates ---

def test_an_old_relevant_case_behind_many_newer_ones_is_found(monkeypatch):
    monkeypatch.setattr(recall, "CANDIDATE_PAGE", 5)
    old = _put(title="the one that matters", task={"model_family": ["catboost"]},
               created_at="2025-01-01T00:00:00+00:00")
    for n in range(12):
        _put(title=f"newer {n}", task={"model_family": ["svm"]}, created_at=f"2026-02-{n + 1:02d}T00:00:00+00:00")
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["catboost"]}, ["Replication Agent"])
    assert [i["memory"]["id"] for i in ctx.items] == [old]


def test_a_candidate_cut_is_counted(monkeypatch):
    monkeypatch.setattr(recall, "CANDIDATE_PAGE", 2)
    monkeypatch.setattr(recall, "MAX_CANDIDATES", 4)
    for n in range(6):
        _put(title=f"case {n}", created_at=f"2026-02-{n + 1:02d}T00:00:00+00:00")
    with run_context.invocation("inv-1", "Replication Agent", 1) as inv:
        recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert inv["memory"]["candidates_truncated"] == 1


# --- 10. The whole answer's memory text has a budget ---

def test_a_model_request_cannot_grow_the_memory_text_without_limit():
    for n in range(8):
        _put(title=f"xgboost case {n}", content="x = 1\n" * 200)
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"],
                               pack_chars=3000)
    first = ctx.block()
    more = ctx.request({"query": "more xgboost"})
    assert len(first) <= 3000
    assert ctx.prompt_chars <= recall.AGGREGATE_PACKS * 3000
    assert more.startswith("MORE PAST CASES")


# --- 11. Rescoring leaves curator lessons alone ---

def test_rescoring_does_not_retire_a_curator_lesson(tmp_path, monkeypatch):
    run_id = make_run(tmp_path, monkeypatch, gap=1.0)
    lesson = memory.put_memory({"kind": "lesson", "agent": "Replication Agent", "polarity": "positive",
                                "run_id": run_id, "title": "Split before resampling", "summary": "rule",
                                "state": "candidate", "quality": 0.5, "evidence": {"rule": "curator"}})
    report = retain.rescore_run(run_id)
    assert lesson not in [m["id"] for m in report["retired"]]
    assert memory.get_memory(lesson)["state"] == "candidate"


# --- 12. Database files are released at once ---

def test_the_memory_store_releases_its_file(tmp_path):
    store = SqliteStore(tmp_path / "m.db")
    store.meta_set("k", "v")
    assert store.meta_get("k") == "v"
    os.remove(tmp_path / "m.db")  # Windows refuses this while a connection is open


def test_the_run_database_releases_its_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "DB_FILE", tmp_path / "runs.db")
    with run_store._connect() as conn:
        conn.execute("SELECT 1")
    os.remove(tmp_path / "runs.db")


# --- 13 and 14. What learning credits, and what it reports ---

def _leaky_evaluation():
    return {**GOOD_EVALUATION, "review": {"review_status": "COMPLETE", "findings": [], "static_findings": [
        {"check": "smote_before_split", "issue": "SMOTE before split", "category": "data_leakage",
         "severity": "critical", "evidence": "SMOTE().fit_resample(X, y)"}]}}


@pytest.mark.parametrize("evaluation,expected", [
    (GOOD_EVALUATION, (1, 0)),
    (None, (0, 0)),  # validity unknown: not judged
    ({**GOOD_EVALUATION, "metrics": [{"metric": "accuracy", "difference_points": 15.0}]}, (0, 0)),  # far: no credit
    ({**GOOD_EVALUATION, "review": {"review_status": "UNAVAILABLE", "static_findings": [], "findings": []}}, (0, 0)),
])
def test_an_approved_replication_earns_credit_only_with_a_valid_close_baseline(monkeypatch, evaluation, expected):
    mid = _put(task={"model_family": ["xgboost"]})
    _run_that_uses_a_memory(monkeypatch, "xgboost", passes=True, finish_status="success", evaluation=evaluation)
    stored = memory.get_memory(mid)
    assert (stored["times_helped"], stored["times_hurt"]) == expected


def test_a_leaky_final_baseline_counts_against_the_memories_it_used(monkeypatch):
    mid = _put(task={"model_family": ["xgboost"]})
    _run_that_uses_a_memory(monkeypatch, "xgboost", passes=True, finish_status="success",
                            evaluation=_leaky_evaluation())
    assert memory.get_memory(mid)["times_hurt"] == 1


def test_several_calls_of_one_attempt_are_one_outcome(monkeypatch):
    mid = _put(task={"model_family": ["xgboost"]})
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: json.dumps({"ok": True, "memory_used": ["M1"]}))
    run_id = run_store.start_run("credit card fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)

    def work():
        for _ in range(3):
            llm.chat("call", memory=recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]},
                                                         ["Replication Agent"]))
        return {"ok": True}

    master_agent._supervise("Replication Agent", work, lambda out: [], lambda out: "r", max_attempts=1)
    run_store.finish_run(run_id, "success", "BASELINE_EVALUATION",
                         result={"reruns": [], "evaluation_result": GOOD_EVALUATION})
    assert memory.get_memory(mid)["times_helped"] == 1  # not 3


def test_a_failed_outcome_write_is_reported_not_counted(monkeypatch):
    mid = _put(task={"model_family": ["xgboost"]})
    monkeypatch.setattr(learning, "record_outcome", lambda *a: False)
    reports = []
    real = learning.learn_run
    monkeypatch.setattr(learning, "learn_run", lambda run_id: reports.append(real(run_id)) or reports[-1])
    _run_that_uses_a_memory(monkeypatch, "xgboost", passes=True, finish_status="success")
    assert reports[0]["helped"] == 0 and len(reports[0]["failed"]) == 1
    assert memory.get_memory(mid)["times_helped"] == 0


# --- The curator's evidence counts once per run (review: "Curator support is not an idempotent observation") ---

def test_a_repeated_or_duplicated_support_counts_once(monkeypatch):
    from src.memory import curator
    lesson_id = _put(kind="lesson", run_id="run-1", state="candidate", title="Stratify the split",
                     summary="Use a stratified split for rare fraud labels.", quality=0.5)
    case = memory.get_memory(_put(run_id="run-2", title="random forest baseline",
                                  summary="The stratified split kept 0.17% fraud in both sets."))
    support = {"lesson": "L1", "case": "N1", "quote": "stratified split kept 0.17% fraud"}
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: json.dumps({"lessons": [], "supports": [support, support]}))
    curator.curate_run("run-2", [case])
    curator.curate_run("run-2", [case])  # the same run curated again
    assert memory.get_memory(lesson_id)["seen_count"] == 2  # 1 at insert + 1 support, not 4
