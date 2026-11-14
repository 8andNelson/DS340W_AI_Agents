"""Milestone M7: every agent uses memory at its own LLM call -- its prompt
gets a pack when matching memories exist, its own parser still sees the
format it asked for, and MEMORY_MODE=off changes nothing."""
import json

import pandas as pd
import pytest

from src import config, llm, memory
from src.agents import (code_discovery_agent, data_agent, evaluation_agent, experiment_agent, intake_agent,
                        master_agent, research_agent, validation_agent)
from src.memory import recall
from src.orchestration import column_mapping, run_store

PAPER = {"title": "Handling imbalanced data for credit card fraudulent detection", "doi": "10.1/handling",
         "methodology": "A RandomForestClassifier is trained with SMOTE.", "dataset": "European credit card fraud"}


def _put(agent, title, **m):
    base = {"kind": "solution", "agent": agent, "polarity": "positive", "title": title,
            "summary": f"{title} -- a past case", "quality": 0.8, "lineage": "kept",
            "created_at": "2026-01-01T00:00:00+00:00",
            "task": {"topic": "credit card fraud", "parent_paper": {"doi": PAPER["doi"], "title": PAPER["title"].lower()},
                     "model_family": ["random_forest"],
                     "dataset_identity": ["https://kaggle.com/fraud"]}}
    base.update(m)
    return memory.put_memory(base)


def _capture(monkeypatch, answer: str):
    prompts = []
    monkeypatch.setattr(llm, "_chat_once", lambda prompt, *a: prompts.append(prompt) or answer + "\nMEMORY_USED: M1")
    return prompts


def _calls():
    """Each agent's hooked call, its answer, and a check that its parser still worked."""
    results = [{"title": "A fraud paper", "url": "https://doi.org/10.1/x", "description": "credit card fraud"}]
    return {
        "Intake Agent": (lambda: intake_agent.run("machine learning for detecting credit card fraud"),
                         json.dumps({"domain": "fraud", "ml_task": "classification", "keywords": ["fraud"],
                                     "search_queries": ["q"], "target_terms": ["fraud"]}),
                         lambda out: out["target_terms"] == ["fraud"]),
        "Research Agent": (lambda: research_agent._extract_papers_from_results(results, "credit card fraud", "fraud ml"),
                           "[]", lambda out: out == []),
        "Validation Agent": (lambda: validation_agent._llm_validate_batch([{"title": PAPER["title"], "year": 2024}]),
                             "[]", lambda out: isinstance(out, list)),
        "Data Agent": (lambda: data_agent.plan_data_needs(PAPER, ["fraud"], {"status": "ABSTRACT_ONLY",
                                                                            "text": "the paper text"}),
                       json.dumps({"dataset_name": "European credit card fraud", "dataset_source": "", "dataset_urls": [],
                                   "paper_model": "", "paper_dataset_size": None, "target_entries": 1000,
                                   "reasoning": "r"}),
                       lambda out: isinstance(out, dict)),
        "Code Discovery Agent": (lambda: code_discovery_agent._extract_repo_candidates_from_results(
            results, PAPER["title"], "credit card fraud"), "[]", lambda out: out == []),
        "Cleaning Agent": (lambda: column_mapping.propose_mappings(
            pd.DataFrame({"amount": [1.0], "isfraud": [0]}), "https://kaggle.com/fraud",
            pd.DataFrame({"amt": [2.0], "is_fraud": [1]}), "https://kaggle.com/other", set()),
            "[]", lambda out: out == []),
        "Evaluation Agent": (lambda: evaluation_agent.review(PAPER, {"status": "", "text": ""}, {}, [], "", None, None, []),
                             json.dumps({"assessment": "fine", "findings": [], "decision": {"action": "accept"}}),
                             lambda out: out["assessment"] == "fine"),
        "Experiment Agent": (lambda: experiment_agent.run(PAPER, {}, {"metrics": []}),
                             json.dumps({"analysis": "a", "hypotheses": [], "decision": {"action": "stop"}}),
                             lambda out: isinstance(out, dict)),
    }


TITLES = {"Intake Agent": "Intake: credit card fraud framing", "Research Agent": "credit card fraud search queries",
          "Validation Agent": "Parent Paper: Handling imbalanced data for credit card fraudulent detection",
          "Data Agent": "Dataset for the handling imbalanced data paper",
          "Code Discovery Agent": "Repository for the handling imbalanced data paper",
          "Cleaning Agent": "Cleaning: kaggle fraud merged with its label",
          "Evaluation Agent": "Evaluation of the random forest fraud baseline",
          "Experiment Agent": "Experiment analysis for the handling imbalanced data paper"}


@pytest.mark.parametrize("agent", list(TITLES))
def test_each_agents_prompt_gets_a_pack_and_its_parser_still_works(agent, monkeypatch):
    memory_id = _put(agent, TITLES[agent])
    call, answer, parsed_ok = _calls()[agent]
    prompts = _capture(monkeypatch, answer)
    out = call()
    shown = [p for p in prompts if "PAST CASES FROM EARLIER RUNS" in p]
    assert shown and TITLES[agent] in shown[0], f"{agent} got no pack"
    assert parsed_ok(out), f"{agent}'s parser broke on the MEMORY_USED line"
    used = [u for u in memory.client().local.rows("memory_usage") if u["memory_id"] == memory_id]
    assert used and used[0]["agent"] == agent and used[0]["used"] is True


@pytest.mark.parametrize("agent", list(TITLES))
def test_memory_mode_off_changes_no_prompt(agent, monkeypatch):
    _put(agent, TITLES[agent])
    monkeypatch.setattr(config, "MEMORY_MODE", "off")
    call, answer, parsed_ok = _calls()[agent]
    prompts = _capture(monkeypatch, answer)
    call()
    assert prompts and not any("PAST CASES" in p for p in prompts)


def test_an_agent_with_no_matching_memory_gets_no_pack(monkeypatch):
    _put("Data Agent", "Dataset for an unrelated lung CT paper",
         task={"parent_paper": {"doi": "10.9/lung", "title": "lung ct"}, "topic": "lung cancer"})
    call, answer, _ = _calls()["Code Discovery Agent"]
    prompts = _capture(monkeypatch, answer)
    call()
    assert not any("PAST CASES" in p for p in prompts)


def test_the_used_line_is_removed_before_the_agent_parses_the_answer():
    assert recall.strip_used_lines('[{"a": 1}]\nMEMORY_USED: M1, M2') == '[{"a": 1}]'
    assert recall.strip_used_lines("# MEMORY_USED: M1\nimport os\n") == "import os"
    assert recall.strip_used_lines('{"a": "memory_used stays"}') == '{"a": "memory_used stays"}'


def test_the_topic_feature_and_generic_words():
    query = {"topic": "machine learning for detecting lung cancer"}
    lung = {"task": {"topic": "machine learning for detecting lung cancer"}}
    fraud = {"task": {"topic": "machine learning for detecting credit card fraud"}}
    assert recall.relevance(query, lung)[0] == 1.0
    assert recall.relevance(query, fraud)[0] == 0.0  # "machine learning for detecting" is not a match


def test_parallel_recalls_share_one_fetch_and_a_write_refreshes_it(monkeypatch):
    _put("Research Agent", "credit card fraud search queries")
    calls = []
    original = memory.MemoryClient.fetch
    monkeypatch.setattr(memory.MemoryClient, "fetch", lambda self, **k: calls.append(k) or original(self, **k))
    for _ in range(3):
        recall.MemoryContext("Research Agent", {"text": "credit card fraud"}, ["Research Agent"])
    assert len(calls) == 1
    _put("Research Agent", "another fraud search")  # this process wrote: the cache is cleared
    ctx = recall.MemoryContext("Research Agent", {"text": "credit card fraud search"}, ["Research Agent"])
    assert len(calls) == 2 and len(ctx.items) == 2


def test_master_agent_posts_one_slack_line_when_an_attempt_used_past_cases(monkeypatch):
    _put("Data Agent", TITLES["Data Agent"])
    posts = []
    monkeypatch.setattr(master_agent.slack_client, "post_message", lambda text: posts.append(text))
    run_id = run_store.start_run("fraud")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    call, answer, _ = _calls()["Data Agent"]
    _capture(monkeypatch, answer)
    master_agent._supervise("Data Agent", lambda: call() or {"ok": True}, lambda out: [], lambda out: "r",
                            max_attempts=1)
    memory_posts = [p for p in posts if "MEMORY: past cases" in p]
    assert len(memory_posts) == 1
    assert "SHOWN: 1" in memory_posts[0] and TITLES["Data Agent"] in memory_posts[0]
