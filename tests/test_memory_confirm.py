"""Insight memories, step 4 (docs/CLAUDE_BACKEND_PLAN.md, 18.5): confidence
from repeats. A later run's insight or lesson about the same thing, in a
compatible scope, agrees or disagrees; agreement raises quality and
activates a candidate lesson, disagreement lowers it and links the two."""
import json
from unittest.mock import patch

from src import memory
from src.memory import confirm, recall
from tests.test_memory_insights import XGB, _insights, _record_run, _rerun


def change(call, param, before, after, area="model"):
    return {"call": call, "area": area, "param": param, "before": before, "after": after}


def test_a_change_key_is_what_changed_and_which_way_not_the_values():
    a = confirm.change_keys([change("XGBClassifier", "n_estimators", 100, 500)])
    b = confirm.change_keys([change("XGBClassifier#2", "n_estimators", 300, 3010)])
    assert a == b == {("model", "XGBClassifier", "n_estimators", "increase")}
    assert confirm.change_keys([change("XGBClassifier", "n_estimators", 500, 100)]) == \
        {("model", "XGBClassifier", "n_estimators", "decrease")}
    assert confirm.change_keys([change("StratifiedKFold", None, "absent", {"n_splits": 5}, "validation")]) == \
        {("validation", "StratifiedKFold", "added")}
    assert confirm.change_keys([change("RandomForestClassifier", "class_weight", "absent", "balanced")]) == \
        {("model", "RandomForestClassifier", "class_weight", "set to balanced")}
    order = {"call": "order", "area": "pipeline_order", "param": None,
             "before": ["RFE", "RandomOverSampler"], "after": ["RandomOverSampler", "RFE"]}
    assert confirm.change_keys([order]) == {("pipeline_order", "RandomOverSampler before RFE")}


def scoped(task_type="binary", families=("xgboost",), minority="5-20%"):
    return {"task": {"task_type": task_type, "model_family": list(families),
                     "dataset_traits": {"minority_share": minority} if minority else {}}}


def test_scopes_must_be_compatible_where_both_are_known():
    assert confirm.compatible(scoped(), scoped(minority="1-5%"))          # the next bucket
    assert confirm.compatible(scoped(), scoped(minority=None))            # unknown is no objection
    assert not confirm.compatible(scoped(), scoped(task_type="regression"))
    assert not confirm.compatible(scoped(), scoped(families=("random_forest",)))
    assert not confirm.compatible(scoped(), scoped(minority=">=40%"))     # 6% fraud vs balanced data


def insight(run, direction, changes, valid=True, agent="Replication Agent"):
    return {"id": f"{run}-{direction}", "agent": agent, "run_id": run, "evidence": {"rule": "setting_change"},
            "content": json.dumps({"changes": changes}), "outcome": {"direction": direction, "valid": valid},
            **scoped()}


TREES_UP = [change("XGBClassifier", "n_estimators", 100, 500)]


def test_setting_change_insights_agree_disagree_or_are_unrelated():
    assert confirm.relation(insight("b", "better", TREES_UP), insight("a", "better", TREES_UP)) == "agree"
    assert confirm.relation(insight("b", "worse", TREES_UP), insight("a", "better", TREES_UP)) == "disagree"
    depth = [change("XGBClassifier", "max_depth", 6, 3)]
    assert confirm.relation(insight("b", "better", depth), insight("a", "better", TREES_UP)) is None
    assert confirm.relation(insight("b", "better", TREES_UP, valid=False), insight("a", "better", TREES_UP)) is None
    assert confirm.relation(insight("a", "better", TREES_UP), insight("a", "better", TREES_UP)) is None  # same run
    assert confirm.relation(insight("b", "better", TREES_UP, agent="Experiment Agent"),
                            insight("a", "better", TREES_UP)) is None


def lesson(run, claim, polarity="negative"):
    return {"id": f"{run}-lesson", "agent": "Evaluation Agent", "run_id": run, "polarity": polarity,
            "evidence": {"rule": "reflection"}, "content": json.dumps({"claim": claim}), **scoped()}


def test_reflection_lessons_agree_when_they_say_the_same_thing():
    a = lesson("a", "Fitting RFE without class weights on imbalanced fraud data dropped minority recall.")
    b = lesson("b", "RFE fitted without class weights on imbalanced fraud data dropped the minority recall.")
    assert confirm.relation(b, a) == "agree"
    assert confirm.relation(dict(b, polarity="positive"), a) == "disagree"
    other = lesson("b", "Kaggle downloads need credentials before the dataset search starts.")
    assert confirm.relation(other, a) is None


def _trees_run(tmp_path, monkeypatch, name, before, after, metrics):
    return _record_run(tmp_path / name, monkeypatch,
                       [XGB.format(trees=before, depth=6), XGB.format(trees=after, depth=6)],
                       [_rerun(1, *metrics)])


def test_a_repeated_insight_gains_confidence_and_a_contradiction_is_linked(tmp_path, monkeypatch):
    a = _trees_run(tmp_path, monkeypatch, "a", 100, 500, ("f1=0.10", "f1=0.20"))
    [first] = _insights(a)
    assert "confidence" not in (first.get("quality_reasons") or {})          # seen once
    base = first["quality"]

    b = _trees_run(tmp_path, monkeypatch, "b", 300, 1000, ("f1=0.12", "f1=0.25"))
    [first] = _insights(a)
    conf = first["quality_reasons"]["confidence"]
    assert conf["agree"] == sorted([a, b]) and conf["disagree"] == [] and conf["value"] == 1.0
    assert first["quality"] == round(min(1.0, base + confirm.AGREE_BONUS), 4)
    assert first["seen_count"] == 2

    c = _trees_run(tmp_path, monkeypatch, "c", 100, 800, ("f1=0.30", "f1=0.18"))
    [first] = _insights(a)
    [opposite] = _insights(c)
    conf = first["quality_reasons"]["confidence"]
    assert conf["disagree"] == [c] and conf["value"] == round(2 / 3, 3)
    assert opposite["id"] in first["quality_reasons"]["contradicted_by"]   # recall shows them together
    assert first["state"] == "active"                                        # never hidden for disagreeing
    block = recall._case_block("M1", "good", first, 0.9, with_content=False)
    assert "Confidence: seen in 3 run(s): 2 agree, 1 disagree -- other runs found the opposite" in block


def test_a_retried_retain_does_not_count_a_run_twice(tmp_path, monkeypatch):
    a = _trees_run(tmp_path, monkeypatch, "a", 100, 500, ("f1=0.10", "f1=0.20"))
    b = _trees_run(tmp_path, monkeypatch, "b", 300, 1000, ("f1=0.12", "f1=0.25"))
    [first] = _insights(a)
    confirm.confirm_run(b, _insights(b))                                     # the same comparison again
    [again] = _insights(a)
    assert again["quality"] == first["quality"] and again["seen_count"] == first["seen_count"] == 2
    assert again["quality_reasons"]["confidence"] == first["quality_reasons"]["confidence"]


CLAIM = "After the trees change on this imbalanced data, f1 rose from 0.10 to 0.20 for the xgboost baseline."


def test_a_candidate_lesson_confirmed_by_another_run_becomes_active(tmp_path, monkeypatch):
    answer = json.dumps({"lessons": [{"claim": CLAIM, "basis": "measured", "polarity": "positive",
                                      "applies_when": "xgboost on imbalanced data", "action": "keep more trees",
                                      "evidence": ["metrics f1=0.10 -> f1=0.20; gap None -> None"]}]})

    def reflections(run_id):
        return [m for m in memory.find_memories(agent="Evaluation Agent", kind="lesson", states=None)
                if m["run_id"] == run_id and (m.get("evidence") or {}).get("rule") == "reflection"]

    with patch("src.llm.chat", return_value=answer):
        a = _trees_run(tmp_path, monkeypatch, "a", 100, 500, ("f1=0.10", "f1=0.20"))
        assert [m["state"] for m in reflections(a)] == ["candidate"]         # seen once: not recalled
        b = _trees_run(tmp_path, monkeypatch, "b", 100, 500, ("f1=0.10", "f1=0.20"))
    assert [m["state"] for m in reflections(a)] == ["active"]
    assert [m["state"] for m in reflections(b)] == ["active"]
    assert reflections(a)[0]["quality_reasons"]["confidence"]["agree"] == sorted([a, b])
