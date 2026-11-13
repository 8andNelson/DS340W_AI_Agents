"""Milestone M5: recall -- ranking, the pack shown to the model, the
model's memory requests, usage records, and the Replication Agent's and
Master Agent's recall points."""
import json

import pytest

from src import config, llm, memory
from src.agents import master_agent, replication_agent
from src.memory import recall
from src.orchestration import run_context, run_metrics, run_store

XGB_SCRIPT = "from xgboost import XGBClassifier\nmodel = XGBClassifier(random_state=42)\n"


def _put(**m):
    base = {"kind": "solution", "agent": "Replication Agent", "polarity": "positive", "title": "a case",
            "summary": "a summary", "quality": 0.8, "task": {}, "created_at": "2026-01-01T00:00:00+00:00"}
    base.update(m)
    return memory.put_memory(base)


@pytest.fixture
def bank():
    return {
        "xgb": _put(title="XGBoost baseline script", content=XGB_SCRIPT, quality=0.9,
                    task={"model_family": ["xgboost"], "task_type": "binary"}),
        "rf": _put(title="random_forest baseline script", content="RandomForestClassifier()", quality=0.95,
                   task={"model_family": ["random_forest"], "task_type": "binary"}),
        "leak": _put(kind="failure", polarity="negative", title="Leaky baseline: SMOTE before split", quality=0.7,
                     task={"model_family": ["xgboost"]}),
        "keyerr": _put(kind="failure", polarity="negative", title="Replication Agent attempt failed: keyerror",
                       quality=0.9, task={"failure_signature": "keyerror: <name>"}),
        "neutral": _put(polarity="neutral", title="unknown-lineage xgboost script", task={"model_family": ["xgboost"]}),
        "diag": _put(kind="decision", agent="Master Agent", title="Diagnosis: the label column was renamed",
                     task={"failure_signature": "keyerror: <name>"}, quality=0.85),
    }


def _labels(ctx):
    return {i["label"]: i["memory"]["id"] for i in ctx.items}


# --- Ranking ---

def test_recall_ranks_by_relevance_and_quality_and_keeps_good_and_warnings_apart(bank):
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"], "task_type": "binary"},
                               ["Replication Agent"])
    ids = [(i["group"], i["memory"]["id"]) for i in ctx.items]
    assert ids[0] == ("good", bank["xgb"])  # same model family beats a higher-quality other family
    assert ("warning", bank["leak"]) in ids
    assert all(i["memory"]["id"] != bank["neutral"] for i in ctx.items)  # neutral memories are not shown


def test_a_failure_signature_finds_the_matching_warning_and_diagnosis(bank):
    ctx = recall.context_for("Master Agent", {}, text="KeyError", search_agents=["Replication Agent", "Master Agent"],
                             kinds=("failure", "decision"), failure="Traceback\nKeyError: 'Class'")
    found = {i["memory"]["id"] for i in ctx.items}
    assert {bank["keyerr"], bank["diag"]} <= found


def test_the_current_runs_own_memories_and_later_ones_are_never_recalled(bank):
    run_id = run_store.start_run("fraud")
    own = _put(run_id=run_id, title="this run's xgboost script", task={"model_family": ["xgboost"]}, quality=1.0)
    later = _put(title="made after the run started", task={"model_family": ["xgboost"]}, quality=1.0,
                 created_at="2999-01-01T00:00:00+00:00")
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    found = {i["memory"]["id"] for i in ctx.items}
    assert own not in found and later not in found and bank["xgb"] in found


def test_quarantined_memories_are_not_recalled(bank):
    memory.set_state(bank["xgb"], "quarantined", "test")
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert bank["xgb"] not in {i["memory"]["id"] for i in ctx.items}


def test_unrelated_memories_are_not_shown(bank):
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["lstm"], "task_type": "regression"},
                               ["Replication Agent"])
    assert ctx.items == [] and ctx.block() == ""


# --- The pack ---

def test_the_pack_has_the_rules_labels_scores_and_content(bank):
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    text = ctx.block()
    assert text.startswith("PAST CASES FROM EARLIER RUNS")
    assert "Never report a number from a case as a result" in text
    assert "[M1] GOOD CASE" in text and "WARNING" in text and XGB_SCRIPT.strip() in text
    assert '"memory_request"' in text  # requests are offered while some remain
    assert all(i.get("digest") for i in ctx.items)


def test_the_pack_budget_drops_content_before_cases(bank):
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"],
                               pack_chars=1200)
    text = ctx.block()
    assert "[M1]" in text and XGB_SCRIPT.strip() not in text


def test_used_labels_are_parsed_and_unknown_ones_are_rejected(bank):
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert ctx.used_labels('```json\n{"memory_used": ["M1", "M9"]}\n```') == (["M1"], ["M9"])
    assert ctx.used_labels("# MEMORY_USED: M2, M1\nimport os") == (["M2", "M1"], [])
    assert ctx.used_labels("no labels here") == ([], [])


# --- llm.chat(memory=...) ---

def test_chat_adds_the_pack_and_records_what_was_shown_and_used(bank, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or
                        '```json\n{"memory_used": ["M1"]}\n```\n```python\nprint(1)\n```')
    run_id = run_store.start_run("fraud")
    with run_context.invocation("inv-1", "Replication Agent", 1) as inv:
        ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
        llm.chat("Write the script.", memory=ctx)
    assert prompts[0].startswith("Write the script.") and "[M1] GOOD CASE" in prompts[0]
    rows = memory.client().local.rows("memory_usage", run_id=run_id)
    assert len(rows) == len(ctx.items) and {r["invocation_id"] for r in rows} == {"inv-1"}
    used = [r for r in rows if r["used"]]
    assert [r["memory_id"] for r in used] == [_labels(ctx)["M1"]] and used[0]["supplied_digest"]
    assert memory.get_memory(_labels(ctx)["M1"])["times_used"] == 1
    assert memory.get_memory(_labels(ctx)["M1"])["times_recalled"] == 1
    assert inv["memory"]["used"] == [_labels(ctx)["M1"]] and inv["memory"]["recalls_with_cases"] == 1


def test_the_model_may_request_more_cases_at_most_twice(bank, monkeypatch):
    answers = iter(['{"memory_request": {"query": "random forest", "filters": {"model_family": "random_forest"}}}',
                    '{"memory_request": {"query": "anything else"}}',
                    '{"memory_request": {"query": "again"}}'])
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or next(answers))
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    content = llm.chat("Write the script.", memory=ctx)
    assert len(prompts) == 3 and ctx.requests == 2
    assert "MORE PAST CASES" in prompts[1] and "random_forest baseline script" in prompts[1]
    assert recall.ANSWER_NOW in prompts[2]
    assert json.loads(content) == {"memory_request": {"query": "again"}}  # a third request is a bad answer
    assert any(i["trigger"] == "model_request" for i in ctx.items)


def test_an_answer_with_code_is_not_a_memory_request():
    assert recall.memory_request('{"memory_request": {"query": "x"}}') == {"query": "x"}
    assert recall.memory_request('```python\nx = {"memory_request": {}}\n```') is None
    assert recall.memory_request('{"memory_request": {"query": "x"}, "cause": "y"}') is None


def test_chat_without_memory_is_unchanged(monkeypatch):
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompt)
    assert llm.chat("plain prompt") == "plain prompt"
    assert llm.chat("plain prompt", memory=recall.MemoryContext("X", {"model_family": ["none"]}, ["X"])) == \
        "plain prompt"  # an empty pack adds nothing


def test_memory_mode_off_gives_no_context(bank, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_MODE", "off")
    assert recall.context_for("Replication Agent", {"model_family": ["xgboost"]}) is None


def test_a_broken_store_gives_an_empty_pack_not_an_error(monkeypatch):
    monkeypatch.setattr(memory.MemoryClient, "fetch", lambda *a, **k: 1 / 0)
    with run_context.invocation("inv", "Replication Agent", 1) as inv:
        ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert ctx.items == [] and inv["memory"]["recall_errors"] == 1


# --- The recall points ---

def test_the_replication_agent_sees_good_and_bad_past_scripts_when_it_generates(bank, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or
                        '```json\n{"inferred_assumptions": [], "libraries_required": [], "memory_used": ["M1"]}\n```\n'
                        "```python\nprint(1)\n```")
    paper = {"title": "Fraud with XGBoost", "methodology": "We train an XGBClassifier on the transactions."}
    result = replication_agent._generate_baseline_script(paper, {}, dataset_profile="cols")
    assert result["code"] == "print(1)\n"
    assert "XGBoost baseline script" in prompts[0] and XGB_SCRIPT.strip() in prompts[0]
    assert "WARNING" in prompts[0] and "Leaky baseline" in prompts[0]


def test_the_replication_repair_sees_warnings_and_diagnoses_for_the_same_failure(bank, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or "```python\nprint(2)\n```")
    replication_agent._repair_generated_script("x", "Traceback\nKeyError: 'Class'", {"title": "p"})
    assert "Replication Agent attempt failed: keyerror" in prompts[0]
    assert "Diagnosis: the label column was renamed" in prompts[0]


def test_master_agents_diagnosis_sees_past_diagnoses_of_the_same_failure(bank, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or
                        '{"cause": "renamed label", "guidance": ["use isFraud"], "memory_used": ["M1"]}')
    diagnosis = master_agent._diagnose("Replication Agent", 1, "Traceback\nKeyError: 'Class'", "report", [])
    assert diagnosis["cause"] == "renamed label"
    assert "Diagnosis: the label column was renamed" in prompts[0]


def test_master_agent_logs_named_cases_that_were_not_shown_and_records_memory_use(bank, monkeypatch):
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: '{"x": 1, "memory_used": ["M1", "M7"]}')
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)

    def agent_work():
        ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
        llm.chat("go", memory=ctx)
        return {"ok": True}

    master_agent._supervise("Replication Agent", agent_work, lambda out: [], lambda out: "report", max_attempts=1)
    log = json.loads(master_agent.LOG_FILE.read_text())
    assert any("M7" in e["issue"] and e["decision"] == "Ignored" for e in log)
    event = memory.client().local.rows("memory_events", run_id=run_id, event_type="attempt")[0]
    assert event["payload"]["memory"]["invalid_refs"] == ["M7"] and len(event["payload"]["memory"]["used"]) == 1
    assert run_context.current_invocation() is None

    run_store.finish_run(run_id, "halted", "BASELINE_REPLICATION", result={})
    run = next(r for r in run_metrics.load_runs() if r["conditions"]["run_id"] == run_id)
    assert run["conditions"]["memory_exposure"] == "full"
    assert run["metrics"][("run", "memories_used")]["value"] == 1
    assert run["metrics"][("run", "memories_recalled")]["value"] == len(event["payload"]["memory"]["shown"])
    assert run["metrics"][("Replication Agent", "memories_used")]["value"] == 1


def test_a_read_only_run_recalls_but_does_not_change_the_bank(bank, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_MODE", "read")
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    master_agent._supervise("Intake Agent", lambda: {"ok": True}, lambda out: [], lambda out: "r", max_attempts=1)
    run_store.finish_run(run_id, "halted", "TOPIC_ANALYSIS", result={})
    assert memory.client().local.rows("memories", run_id=run_id) == []
    run = next(r for r in run_metrics.load_runs() if r["conditions"]["run_id"] == run_id)
    assert run["conditions"]["memory_mode"] == "read"


# --- The retrieval benchmark ---

def test_the_benchmark_scores_hits_harmful_memories_critical_warnings_and_no_match_queries(bank, tmp_path):
    from src.memory import benchmark
    cases = {"status": "test", "labelled_by": "test", "cases": [
        {"id": "hit", "agent": "Replication Agent", "search_agents": ["Replication Agent"],
         "query": {"model_family": ["xgboost"]}, "acceptable": [bank["xgb"]], "critical_warnings": [bank["leak"]]},
        {"id": "harm", "agent": "Replication Agent", "search_agents": ["Replication Agent"],
         "query": {"model_family": ["xgboost"]}, "acceptable": [bank["rf"]], "harmful": [bank["xgb"]]},
        {"id": "none", "agent": "Replication Agent", "search_agents": ["Replication Agent"],
         "query": {"model_family": ["lstm"]}, "expect_none": True},
        {"id": "sig", "agent": "Master Agent", "search_agents": ["Replication Agent", "Master Agent"],
         "kinds": ["failure", "decision"], "failure": "KeyError: 'Class'", "acceptable": [bank["diag"]]},
    ]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    report = benchmark.run(path)
    by_id = {r["id"]: r for r in report["results"]}
    assert by_id["hit"]["passed"] and by_id["none"]["passed"] and by_id["sig"]["passed"]
    assert by_id["harm"]["harmful_shown"] == [bank["xgb"]] and not by_id["harm"]["passed"]
    assert report["hit_rate"] == 2 / 3 and report["harmful_shown"] == 1 and report["gate_passed"] is False
    assert memory.get_memory(bank["xgb"])["times_recalled"] == 0  # read-only
    assert "harmful shown" in benchmark.format_report(report)
