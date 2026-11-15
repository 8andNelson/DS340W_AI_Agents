"""Insight memories, step 2 (docs/CLAUDE_BACKEND_PLAN.md, 18.4): recall by
scope. A lesson is matched on the model family, task type and data traits
(row count, minority-class share, anonymized columns), not only on the same
paper; records of a stage's output (snapshots) are not shown as lessons."""
from unittest.mock import patch

import pandas as pd

from src.agents import cleaning_agent
from src.memory import features, recall

FRAUD_TRAITS = {"rows": "10k-100k", "minority_share": "5-20%", "anonymized_columns": False}


def memory(mid, rule="setting_change", kind="lesson", polarity="positive", families=("xgboost",),
           traits=None, paper="paper a", quality=0.8, title="model: XGBClassifier n_estimators 100 -> 500"):
    return {"id": mid, "state": "active", "kind": kind, "polarity": polarity, "agent": "Experiment Agent",
            "run_id": f"run-{mid}", "title": title, "summary": "", "tags": [], "quality": quality,
            "created_at": "2026-11-15T00:00:00+00:00", "evidence": {"rule": rule},
            "task": {"model_family": list(families), "task_type": "binary", "parent_paper": {"title": paper},
                     "dataset_traits": traits if traits is not None else FRAUD_TRAITS}}


QUERY = {"model_family": ["xgboost"], "task_type": "binary", "parent_paper": {"title": "paper b"},
         "dataset_traits": FRAUD_TRAITS}


def test_data_traits_match_by_bucket():
    assert recall.trait_match(FRAUD_TRAITS, FRAUD_TRAITS) == 1.0
    near = {"rows": "100k-1M", "minority_share": "1-5%", "anonymized_columns": False}
    assert recall.trait_match(FRAUD_TRAITS, near) == (0.5 + 0.5 + 1.0) / 3
    far = {"rows": ">=1M", "minority_share": ">=40%", "anonymized_columns": True}
    assert recall.trait_match(FRAUD_TRAITS, far) == 0.0
    assert recall.trait_match(FRAUD_TRAITS, {}) == 0.0  # unknown is not a match


def test_the_minority_share_comes_from_recorded_label_counts():
    cleaning = {"rows_total": 15420, "columns_total": 34, "label_counts": {"0": 14497, "1": 923}}
    assert round(features.minority_share(cleaning), 3) == 0.06
    assert features.dataset_traits(cleaning, None)["minority_share"] == "5-20%"
    assert "minority_share" not in features.dataset_traits({"rows_total": 15420}, None)  # older runs: unknown


def test_the_cleaning_agent_records_the_label_counts(tmp_path):
    csv = tmp_path / "claims.csv"
    pd.DataFrame({"amount": range(20), "fraud": [1] * 3 + [0] * 17}).to_csv(csv, index=False)
    with patch.object(cleaning_agent, "PROCESSED_DATA_DIR", tmp_path / "processed"), \
            patch.object(cleaning_agent, "CONFLICTS_LOG", tmp_path / "conflicts.json"):
        result = cleaning_agent.run([{"name": "https://x/claims", "display_name": "Claims",
                                      "local_csv_path": str(csv)}], target_terms=["fraud"])
    assert result["label_counts"] == {"0": 17, "1": 3}


def test_a_lesson_from_another_paper_with_the_same_scope_is_recalled():
    same_scope = memory("a", paper="paper a")
    other_data = memory("b", paper="paper a", traits={"rows": ">=1M", "minority_share": ">=40%"})
    ranked = recall.rank([other_data, same_scope], QUERY)
    assert [m["id"] for _g, m, _s, _p in ranked][0] == "a"
    parts = ranked[0][3]
    assert parts["data_traits"] == 1.0 and parts["model_family"] == 1.0 and parts["parent_paper"] == 0.0


def test_snapshots_are_not_shown_as_lessons():
    snapshot = memory("s", rule="approved", kind="solution", title="xgboost baseline script (reconstructed)")
    rerun = memory("r", rule="rerun", kind="decision", title="Rerun of the Replication Agent: kept")
    warning = memory("w", rule="approved", kind="failure", polarity="negative",
                     title="Cleaning: the merged data has no label column")
    lesson = memory("l")
    shown = {m["id"] for _g, m, _s, _p in recall.rank([snapshot, rerun, warning, lesson], QUERY)}
    assert shown == {"l", "w"}
    # fast mode may still adapt a past kept script
    with_snapshots = recall.rank([snapshot, lesson], QUERY, include_snapshots=True)
    assert {m["id"] for _g, m, _s, _p in with_snapshots} == {"s", "l"}


def test_only_the_replication_agent_with_fast_mode_gets_snapshots(monkeypatch):
    seen = {}

    def fake_rank(candidates, query, **kwargs):
        seen[query.get("agent_name")] = kwargs["include_snapshots"]
        return []

    monkeypatch.setattr(recall, "rank", fake_rank)
    monkeypatch.setattr(recall.MemoryContext, "_candidates", lambda self, agents, info=None: [])
    monkeypatch.setattr("src.config.MEMORY_REUSE", True)
    for agent in ("Replication Agent", "Experiment Agent"):
        recall.MemoryContext(agent, {"agent_name": agent}, [agent])
    assert seen == {"Replication Agent": True, "Experiment Agent": False}
    monkeypatch.setattr("src.config.MEMORY_REUSE", False)
    recall.MemoryContext("Replication Agent", {"agent_name": "Replication Agent"}, ["Replication Agent"])
    assert seen["Replication Agent"] is False


def test_each_case_shows_its_scope():
    block = recall._case_block("M1", "good", memory("a"), 0.9, with_content=False)
    assert 'Scope (the task this case comes from):' in block
    assert '"minority_share": "5-20%"' in block and '"model_family": ["xgboost"]' in block


def test_the_query_carries_the_current_tasks_data_traits(monkeypatch):
    monkeypatch.setattr("src.config.MEMORY_MODE", "read")
    monkeypatch.setattr(recall.MemoryContext, "__init__", lambda self, agent, query, agents, kinds=None:
                        setattr(self, "query", query))
    with patch("src.memory.backend_name", return_value="sqlite"):
        ctx = recall.context_for("Experiment Agent", {"model_family": ["xgboost"],
                                                      "dataset_traits": FRAUD_TRAITS})
    assert ctx.query["dataset_traits"] == FRAUD_TRAITS
