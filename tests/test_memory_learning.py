"""Milestone M6: the learning loop -- outcomes of used memories, effective
quality, fast mode, curator lessons, contradictions, and ReviseRank."""
import json
from unittest.mock import patch

import pytest

from src import config, llm, memory
from src.agents import master_agent, replication_agent
from src.memory import curator, learning, ranking, recall
from src.orchestration import run_context, run_metrics, run_store

XGB_SCRIPT = "import os\nfrom xgboost import XGBClassifier\nprint('METRIC accuracy=0.9')\n"


def _put(**m):
    base = {"kind": "solution", "agent": "Replication Agent", "polarity": "positive", "title": "a case",
            "summary": "a summary", "quality": 0.8, "lineage": "kept", "task": {},
            "created_at": "2026-01-01T00:00:00+00:00"}
    base.update(m)
    return memory.put_memory(base)


def _answer(used=("M1",), ok=True):
    return lambda prompt, *a: json.dumps({"ok": ok, "memory_used": list(used)})


def _run_that_uses_a_memory(monkeypatch, memory_id_family, passes: bool, finish_status="halted"):
    """A run whose Replication Agent attempt uses the best xgboost case, then passes or fails its checks."""
    monkeypatch.setattr(llm, "_chat_once", _answer())
    run_id = run_store.start_run("credit card fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)

    def work():
        ctx = recall.MemoryContext("Replication Agent", {"model_family": [memory_id_family]}, ["Replication Agent"])
        return {"ok": passes, "used": [i["memory"]["id"] for i in ctx.items if i["label"] == "M1"],
                "reply": llm.chat("write it", memory=ctx)}

    check = lambda out: [] if out["ok"] else [master_agent._log_entry("Correction", "Replication Agent",  # noqa: E731
                                                                     "script failed", "Rejected", "")]
    output, _ = master_agent._supervise("Replication Agent", work, check, lambda out: "report", max_attempts=1)
    run_store.finish_run(run_id, finish_status, "BASELINE_REPLICATION", result={"reruns": []})
    return run_id, output["used"][0]


# --- Effective quality ---

def test_effective_quality_moves_toward_the_recorded_outcomes():
    assert ranking.effective_quality({"quality": 0.6}) == 0.6
    up = ranking.effective_quality({"quality": 0.6, "times_helped": 3})
    down = ranking.effective_quality({"quality": 0.6, "times_hurt": 3})
    assert down < 0.6 < up
    # w = 1 / (1 + 3/3) = 0.5; use = (3 + 1) / (3 + 2) = 0.8
    assert up == round(0.5 * 0.6 + 0.5 * 0.8, 4)
    many = ranking.effective_quality({"quality": 0.9, "times_hurt": 30})
    assert many < 0.2  # many bad outcomes outweigh a high base score


def test_a_usage_outcome_is_applied_once(tmp_path):
    mid = _put()
    usage_id = memory.record_usage({"memory_id": mid, "used": True})
    assert learning.record_outcome(usage_id, mid, True)
    assert learning.record_outcome(usage_id, mid, False)  # a second outcome for the same use is ignored
    stored = memory.get_memory(mid)
    assert (stored["times_helped"], stored["times_hurt"]) == (1, 0)
    assert memory.client().local.rows("memory_usage", id=usage_id)[0]["helped"] is True


def test_usage_outcomes_reach_the_hosted_store(tmp_path):
    from tests.test_memory_store import FakeRemote
    remote = FakeRemote(tmp_path / "remote.db")
    memory.reset_client(memory.MemoryClient("postgres", tmp_path / "local.db", remote=remote))
    mid = _put()
    usage_id = memory.record_usage({"memory_id": mid, "used": True})
    learning.record_outcome(usage_id, mid, False)
    memory.sync()
    assert remote.fetch_memories(ids=[mid], states=None)[0]["times_hurt"] == 1


# --- Learning from two runs on the same topic (the M6 check) ---

def test_the_memory_followed_by_a_good_outcome_ranks_above_the_one_followed_by_a_bad_one(monkeypatch):
    good = _put(title="xgboost script A", task={"model_family": ["xgboost"]}, quality=0.8)
    bad = _put(title="random forest script B", task={"model_family": ["random_forest"]}, quality=0.8)
    _, used_good = _run_that_uses_a_memory(monkeypatch, "xgboost", passes=True, finish_status="success")
    _, used_bad = _run_that_uses_a_memory(monkeypatch, "random_forest", passes=False)
    assert (used_good, used_bad) == (good, bad)
    g, b = memory.get_memory(good), memory.get_memory(bad)
    assert (g["times_helped"], g["times_hurt"]) == (1, 0) and (b["times_helped"], b["times_hurt"]) == (0, 1)
    assert ranking.effective_quality(g) > ranking.effective_quality(b)
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost", "random_forest"]},
                               ["Replication Agent"])
    assert ctx.items[0]["memory"]["id"] == good  # same relevance and base score: the outcome decides


def test_a_crashed_run_judges_nothing(monkeypatch):
    mid = _put(task={"model_family": ["xgboost"]})
    _run_that_uses_a_memory(monkeypatch, "xgboost", passes=True, finish_status="crashed")
    stored = memory.get_memory(mid)
    assert (stored["times_helped"], stored["times_hurt"]) == (0, 0)  # lineage unknown: not judged


def test_a_diagnosis_is_judged_by_the_failed_agents_next_attempt(monkeypatch):
    diag = _put(kind="decision", agent="Master Agent", title="Diagnosis: rename the label",
                task={"failure_signature": "keyerror: <name>"})
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: '{"cause": "c", "guidance": ["g"], "memory_used": ["M1"]}')
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    outputs = iter([{"ok": False}, {"ok": True}])
    check = lambda out: [] if out["ok"] else [master_agent._log_entry("Correction", "Replication Agent",  # noqa: E731
                                                                     "KeyError: 'Class'", "Retrying", "")]
    master_agent._supervise("Replication Agent", lambda: next(outputs), check, lambda out: "r", max_attempts=2)
    run_store.finish_run(run_id, "halted", "BASELINE_REPLICATION", result={"reruns": []})
    assert memory.get_memory(diag)["times_helped"] == 1  # the retry after the diagnosis passed
    run = next(r for r in run_metrics.load_runs() if r["conditions"]["run_id"] == run_id)
    assert run["metrics"][("run", "memories_helped")]["value"] == 1
    assert run["conditions"]["metrics_version"] == run_metrics.METRICS_VERSION


# --- Fast mode ---

def _fast_mode_bank(quality=0.9, family=("xgboost",)):
    return _put(title="kept xgboost baseline", content=XGB_SCRIPT, quality=quality,
                task={"model_family": list(family), "task_type": "binary"})


def test_fast_mode_adapts_a_good_kept_script_of_the_same_family(monkeypatch):
    mid = _fast_mode_bank()
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or
                        '```json\n{"inferred_assumptions": [], "memory_used": ["M1"]}\n```\n```python\nprint(1)\n```')
    paper = {"title": "XGBoost fraud", "methodology": "An XGBClassifier is trained."}
    result = replication_agent._generate_baseline_script(paper, {}, dataset_profile="cols")
    assert "FAST MODE: past case [M1]" in prompts[0] and XGB_SCRIPT.strip() in prompts[0]
    assert result["fast_mode_from"] == mid and result["fast_mode_usage_id"]


@pytest.mark.parametrize("quality,family,reuse", [(0.7, ("xgboost",), True), (0.9, ("xgboost", "svm"), True),
                                                  (0.9, ("xgboost",), False)])
def test_fast_mode_needs_quality_the_same_family_and_reuse_on(monkeypatch, quality, family, reuse):
    _fast_mode_bank(quality=quality, family=family)
    monkeypatch.setattr(config, "MEMORY_REUSE", reuse)
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or "```python\nprint(1)\n```")
    result = replication_agent._generate_baseline_script({"title": "t", "methodology": "XGBClassifier"}, {})
    assert "FAST MODE" not in prompts[0] and "fast_mode_from" not in result


def test_an_adapted_script_that_fails_its_first_run_ranks_lower_even_after_repair(monkeypatch, tmp_path):
    mid = _fast_mode_bank()
    generation = '```json\n{"inferred_assumptions": [], "memory_used": ["M1"]}\n```\n```python\nraise SystemExit(1)\n```'
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: generation if "FAST MODE" in prompt or
                        "reconstructing" in prompt else "```python\nprint('METRIC accuracy=0.9')\n```")
    runs = iter([{"exit_code": 1, "stdout": "", "stderr": "boom", "timed_out": False, "duration_seconds": 1},
                 {"exit_code": 0, "stdout": "METRIC accuracy=0.9", "stderr": "", "timed_out": False,
                  "duration_seconds": 1}])
    ok = {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False, "duration_seconds": 0}
    with patch.object(replication_agent, "_run_bounded", side_effect=lambda *a, **k: next(runs)), \
         patch.object(replication_agent, "_create_venv", return_value=ok), \
         patch.object(replication_agent, "_install_libraries", return_value=([], [])), \
         patch.object(replication_agent, "_pip_freeze", return_value=""):
        result = replication_agent._replicate_by_reconstruction(
            {"title": "XGBoost fraud", "methodology": "An XGBClassifier is trained."}, {})
    assert result["status"] == "OK" and result["fast_mode_from"] == mid  # the repair made it pass
    stored = memory.get_memory(mid)
    assert stored["times_hurt"] == 1 and ranking.effective_quality(stored) < 0.9


# --- Curator ---

def _new_case(**m):
    mid = _put(**m)
    return memory.get_memory(mid)


def _curator_answer(monkeypatch, answer: dict):
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: json.dumps(answer))


def test_curator_stores_only_lessons_whose_quotes_are_verified_as_candidates(monkeypatch):
    case = _new_case(run_id="run-1", title="Leaky baseline: SMOTE before split",
                     summary="SMOTE was fitted on the full data before train_test_split.",
                     task={"model_family": ["xgboost"]})
    _curator_answer(monkeypatch, {"lessons": [
        {"agent": "Replication Agent", "polarity": "negative", "title": "Split before resampling",
         "rule": "Fit SMOTE on the training split only.",
         "evidence": [{"case": "N1", "quote": "SMOTE was fitted on the full data"}]},
        {"agent": "Replication Agent", "polarity": "positive", "title": "Invented",
         "rule": "Always use 500 trees.", "evidence": [{"case": "N1", "quote": "500 trees gave the best AUC"}]},
        {"agent": "Data Agent", "polarity": "positive", "title": "Wrong agent", "rule": "x",
         "evidence": [{"case": "N1", "quote": "SMOTE was fitted on the full data"}]},
    ]})
    report = curator.curate_run("run-1", [case])
    assert len(report["lessons"]) == 1 and len(report["rejected"]) == 2
    lesson = memory.get_memory(report["lessons"][0])
    assert lesson["kind"] == "lesson" and lesson["state"] == "candidate" and lesson["quality"] == 0.5
    assert lesson["task"]["model_family"] == ["xgboost"]
    assert lesson["evidence"]["cited"] == [{"memory_id": case["id"], "quote": "SMOTE was fitted on the full data"}]
    # A candidate is never given to agents.
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"], "text": "SMOTE split"},
                               ["Replication Agent"])
    assert lesson["id"] not in {i["memory"]["id"] for i in ctx.items}


def test_a_lesson_becomes_active_when_an_independent_run_supports_it(monkeypatch):
    lesson_id = _put(kind="lesson", run_id="run-1", state="candidate", title="Stratify the split",
                     summary="Use a stratified split for rare fraud labels.", quality=0.5)
    case = _new_case(run_id="run-2", title="random forest baseline",
                     summary="The stratified split kept 0.17% fraud in both sets.")
    _curator_answer(monkeypatch, {"lessons": [], "supports": [
        {"lesson": "L1", "case": "N1", "quote": "stratified split kept 0.17% fraud"}]})
    report = curator.curate_run("run-2", [case])
    assert report["activated"] == [lesson_id]
    stored = memory.get_memory(lesson_id)
    assert stored["state"] == "active" and stored["seen_count"] == 2
    change = memory.client().local.rows("memory_state_changes", memory_id=lesson_id)[0]
    assert change["actor"] == "curator" and "independent run" in change["reason"]


def test_a_run_cannot_confirm_its_own_lesson_and_unverified_support_is_ignored(monkeypatch):
    own = _put(kind="lesson", run_id="run-1", state="candidate", title="own lesson", summary="own")
    case = _new_case(run_id="run-1", title="a case", summary="something true about the split")
    _curator_answer(monkeypatch, {"supports": [{"lesson": "L1", "case": "N1", "quote": "something true about the split"}]})
    assert curator.curate_run("run-1", [case])["supported"] == []  # its own lesson is not even offered
    other = _put(kind="lesson", run_id="run-0", state="candidate", title="other lesson", summary="other")
    _curator_answer(monkeypatch, {"supports": [{"lesson": "L1", "case": "N1", "quote": "a quote that is not there"}]})
    assert curator.curate_run("run-1", [case])["supported"] == []
    assert memory.get_memory(other)["state"] == "candidate" and memory.get_memory(own)["state"] == "candidate"


def test_a_verified_contradiction_links_both_and_lowers_their_scores(monkeypatch):
    lesson_id = _put(kind="lesson", run_id="run-1", state="active", title="Scaling always helps XGBoost",
                     summary="Standardize every feature before XGBoost.", quality=0.6,
                     task={"model_family": ["xgboost"]})
    case = _new_case(run_id="run-2", title="Experiment: z-score scaling", task={"model_family": ["xgboost"]},
                     summary="Standardization changed no metric for the XGBoost model.", quality=0.7)
    _curator_answer(monkeypatch, {"contradicts": [
        {"lesson": "L1", "case": "N1", "quote": "Standardization changed no metric"}]})
    report = curator.curate_run("run-2", [case])
    assert report["contradicted"] == [[lesson_id, case["id"]]]
    lesson, new = memory.get_memory(lesson_id), memory.get_memory(case["id"])
    assert lesson["quality"] == pytest.approx(0.4) and new["quality"] == pytest.approx(0.5)
    assert lesson["quality_reasons"]["contradicted_by"] == [case["id"]]
    ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert "CONFLICTING CASES" in ctx.block()


def test_the_curator_is_off_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "MEMORY_CURATOR", False)
    monkeypatch.setattr(llm, "_chat_once", lambda *a: pytest.fail("no curator call"))
    assert curator.curate_run("run", [_new_case()])["lessons"] == []


# --- ReviseRank ---

def _two_cases():
    first = _put(title="first case", task={"model_family": ["xgboost"]}, quality=0.9)
    second = _put(title="second case", task={"model_family": ["xgboost"]}, quality=0.7)
    return first, second


def test_revise_rank_reorders_a_retrys_pack_for_its_feedback(monkeypatch):
    first, second = _two_cases()
    calls = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: calls.append(prompt) or "[2] > [1]")
    with llm.supervisor_notes(["Attempt 1 failed: the label column was wrong."]):
        ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert ctx.reranked and [(i["label"], i["memory"]["id"]) for i in ctx.items] == [("M1", second), ("M2", first)]
    assert "the label column was wrong" in calls[0]


@pytest.mark.parametrize("answer", ["[1] > [1]", "[3] > [1]", "no idea"])
def test_revise_rank_keeps_the_order_on_a_wrong_answer(monkeypatch, answer):
    first, second = _two_cases()
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: answer)
    with llm.supervisor_notes(["feedback"]):
        ctx = recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"])
    assert not ctx.reranked and [i["memory"]["id"] for i in ctx.items] == [first, second]


def test_revise_rank_is_skipped_on_a_first_attempt(monkeypatch):
    _two_cases()
    monkeypatch.setattr(llm, "_chat_once", lambda *a: pytest.fail("no rerank call without feedback"))
    assert not recall.MemoryContext("Replication Agent", {"model_family": ["xgboost"]}, ["Replication Agent"]).reranked
