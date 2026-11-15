"""Insight memories, step 1 (docs/CLAUDE_BACKEND_PLAN.md, 18.3.1): what a
rerun changed in the baseline script's settings, joined with what it did
to the metrics and the gap."""
from src.agents import master_agent
from src.memory import insights, retain
from src.orchestration import run_store
from tests.test_memory_retain import INTAKE, _memories

XGB = """import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
N_TREES = {trees}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, y_tr = RandomOverSampler(random_state=42).fit_resample(X_tr, y_tr)
model = XGBClassifier(n_estimators=N_TREES, max_depth={depth}, learning_rate=lr)
"""


def test_settings_are_read_with_their_literal_values():
    settings = insights.model_settings(XGB.format(trees=500, depth=4))
    assert [(s["call"], s["area"]) for s in settings] == [("train_test_split", "validation"),
                                                           ("RandomOverSampler", "resampling"), ("XGBClassifier", "model")]
    xgb = settings[-1]["params"]
    assert xgb["n_estimators"] == 500           # through the variable N_TREES
    assert xgb["max_depth"] == 4
    assert xgb["learning_rate"] == "expr:lr"    # unknown: kept as source text, never guessed
    assert insights.model_settings("def broken(:") is None


def test_the_diff_lists_changed_added_and_removed_settings():
    before = insights.model_settings(XGB.format(trees=100, depth=6))
    after = insights.model_settings(XGB.format(trees=500, depth=6).replace(
        "RandomOverSampler(random_state=42).fit_resample(X_tr, y_tr)", "X_tr, y_tr") +
        "from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(class_weight='balanced')\n")
    described = [insights.describe_change(c) for c in insights.settings_diff(before, after)]
    assert "model: XGBClassifier n_estimators 100 -> 500" in described
    assert "resampling: RandomOverSampler removed (random_state=42)" in described
    assert "model: RandomForestClassifier added (class_weight=balanced)" in described
    assert not any("max_depth" in d for d in described)


def test_a_rename_is_not_a_change_but_a_reorder_is():
    a = "s = RandomOverSampler()\nr = RFE(estimator=est)\nm = RandomForestClassifier()\n"
    b = "r = RFE(estimator=rfe_est)\ns = RandomOverSampler()\nm = RandomForestClassifier()\n"
    changes = insights.settings_diff(insights.model_settings(a), insights.model_settings(b))
    assert [insights.describe_change(c) for c in changes] == ["steps reordered: RFE moved before RandomOverSampler"]


def test_two_calls_of_one_model_are_paired_in_order():
    a = "f = RandomForestClassifier(n_estimators=50)\nm = RandomForestClassifier(n_estimators=100)\n"
    b = "f = RandomForestClassifier(n_estimators=50, class_weight='balanced')\nm = RandomForestClassifier(n_estimators=100)\n"
    changes = insights.settings_diff(insights.model_settings(a), insights.model_settings(b))
    assert [insights.describe_change(c) for c in changes] == ["model: RandomForestClassifier#1 class_weight default -> balanced"]


def test_metric_summaries_are_parsed():
    assert insights.parse_metrics("accuracy=0.94, f1=0.0976, roc_auc=0.7734") == \
        {"accuracy": 0.94, "f1": 0.0976, "roc_auc": 0.7734}


# --- The retain rule ---

LEAK = {"check": "smote_before_split", "issue": "RandomOverSampler is fitted before the split",
        "category": "data_leakage", "severity": "critical", "evidence": "fit_resample(X, y)"}


def _stage(agent, output, offset):
    return master_agent._supervise(agent, lambda: output, lambda out: [], lambda out: f"{agent} report",
                                   max_attempts=1, attempt_offset=offset)


def _record_run(tmp_path, monkeypatch, scripts, reruns, leaks=None):
    """A run with one baseline and one evaluated rerun per later script."""
    run_id = run_store.start_run("insurance fraud detection")
    monkeypatch.setattr(master_agent, "_current_run_id", run_id)
    _stage("Intake Agent", INTAKE, 0)
    leaks = leaks or {}
    for n, script in enumerate(scripts):
        folder = tmp_path / f"replication_{n}"
        folder.mkdir(parents=True)
        (folder / "train.py").write_text(script, encoding="utf-8")
        replication = {"status": "OK", "source": "reconstructed", "entry_point": "train.py",
                       "replication_dir": str(folder), "exit_code": 0}
        _stage("Replication Agent", replication, n * master_agent.MAX_ATTEMPTS)
        _stage("Evaluation Agent", {"status": "INCONCLUSIVE", "metrics": [],
                                    "review": {"static_findings": leaks.get(n, []), "findings": []},
                                    "decision": {"action": "accept"}}, n * master_agent.MAX_ATTEMPTS)
    run_store.finish_run(run_id, "success", "EXPERIMENTATION", parent_paper={"title": "A fraud paper"},
                         result={"phase": "EXPERIMENTATION", "structured": INTAKE, "reruns": reruns,
                                 "cleaning_result": {"rows_total": 15420, "columns_total": 34}})
    return run_id


def _rerun(n, before, after, applied=True, gap=(None, None), by="Experiment Agent"):
    return {"rerun": n, "requested_by": by, "target_agent": "Replication Agent", "instructions": ["change trees"],
            "metrics_before": before, "metrics_after": after, "gap_before": gap[0], "gap_after": gap[1],
            "applied": applied, "improved": False, "outcome": ""}


def _insights(run_id, agent="Replication Agent"):
    return [m for m in _memories(run_id, agent=agent) if m["evidence"]["rule"] == "setting_change"]


def test_a_setting_change_that_helped_is_a_positive_insight(tmp_path, monkeypatch):
    run_id = _record_run(tmp_path, monkeypatch, [XGB.format(trees=100, depth=6), XGB.format(trees=500, depth=6)],
                         [_rerun(1, "f1=0.10, recall=0.05", "f1=0.23, recall=0.78")])
    [m] = _insights(run_id)
    assert m["kind"] == "lesson" and m["polarity"] == "positive" and m["lineage"] == "kept"
    assert m["title"].startswith("model: XGBClassifier n_estimators 100 -> 500")
    assert "f1 +13.0" in m["summary"] and "recall +73.0" in m["summary"]
    assert m["task"]["model_family"] == ["xgboost"]
    assert len(_insights(run_id, "Experiment Agent")) == 1  # the agent that chooses the next change


def test_a_reverted_rerun_teaches_a_warning_and_leaves_the_baseline(tmp_path, monkeypatch):
    scripts = [XGB.format(trees=100, depth=6), XGB.format(trees=3000, depth=6), XGB.format(trees=100, depth=3)]
    run_id = _record_run(tmp_path, monkeypatch, scripts,
                         [_rerun(1, "f1=0.20", "f1=0.12", applied=False, gap=(20.0, 28.0)),
                          _rerun(2, "f1=0.20", "f1=0.30", gap=(20.0, 10.0))])
    by_rerun = {m["outcome"]["gap_after"]: m for m in _insights(run_id)}
    worse, better = by_rerun[28.0], by_rerun[10.0]
    assert worse["polarity"] == "negative" and worse["lineage"] == "reverted"
    assert "n_estimators 100 -> 3000" in worse["title"]
    # rerun 2 is compared with the baseline that was kept (100 trees), not the reverted one
    assert better["polarity"] == "positive" and "max_depth 6 -> 3" in better["title"]
    assert "n_estimators" not in better["title"]


def test_a_gain_with_verified_leakage_is_not_a_success(tmp_path, monkeypatch):
    run_id = _record_run(tmp_path, monkeypatch, [XGB.format(trees=100, depth=6), XGB.format(trees=500, depth=6)],
                         [_rerun(1, "f1=0.10", "f1=0.95")], leaks={1: [LEAK]})
    [m] = _insights(run_id)
    assert m["polarity"] == "negative" and "verified leakage" in m["summary"]
    assert m["outcome"]["valid"] is False


def test_no_insight_without_a_setting_change_or_a_measured_effect(tmp_path, monkeypatch):
    same = XGB.format(trees=100, depth=6)
    run_id = _record_run(tmp_path, monkeypatch, [same, same], [_rerun(1, "f1=0.10", "f1=0.30")])
    assert _insights(run_id) == []
    run_id = _record_run(tmp_path / "b", monkeypatch, [same, XGB.format(trees=500, depth=6)],
                         [_rerun(1, "f1=0.100", "f1=0.104")])  # +0.4 points: noise
    assert _insights(run_id) == []


def test_a_lower_is_better_score_does_not_count_as_a_gain(tmp_path, monkeypatch):
    run_id = _record_run(tmp_path, monkeypatch, [XGB.format(trees=100, depth=6), XGB.format(trees=500, depth=6)],
                         [_rerun(1, "f1=0.20, brier=0.05", "f1=0.20, brier=0.30")])
    assert _insights(run_id) == []  # brier rose 25 points: worse, not a +12.5 mean gain


RICH = """
import numpy as np, optuna, re
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
PATTERN = re.compile("x")
np.random.seed({seed})
cv = StratifiedKFold({folds}, shuffle=True, random_state=42)
grid = {{"max_depth": {depths}}}
search = GridSearchCV(XGBClassifier(), param_grid=grid, cv=cv, scoring="{scoring}")
def objective(trial):
    d = trial.suggest_int("max_depth", 2, {top})
    return cross_val_score(XGBClassifier(max_depth=d), X, y, cv=cv).mean()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials={trials})
model = CalibratedClassifierCV(search.best_estimator_, method="{method}", cv=3)
model.fit(X, y, sample_weight=w)
pred = (model.predict_proba(X_te)[:, 1] >= {threshold}).astype(int)
"""
BASE = dict(seed=7, folds=5, depths=[3, 5], scoring="f1", top=10, trials=50, method="sigmoid", threshold=0.5)


def test_every_data_science_area_is_read():
    areas = {s["call"]: s["area"] for s in insights.model_settings(RICH.format(**BASE))}
    assert areas == {"seed": "seed", "StratifiedKFold": "validation", "GridSearchCV": "tuning", "XGBClassifier": "model",
                     "search:max_depth": "tuning", "cross_val_score": "validation", "create_study": "tuning",
                     "optimize": "tuning", "CalibratedClassifierCV": "calibration", "fit": "training",
                     "decision_threshold": "threshold"}  # re.compile is not a Keras compile


def test_changes_outside_the_model_are_found_and_named_by_area():
    changed = dict(BASE, folds=10, depths=[3, 5, 7], top=15, trials=200, method="isotonic", threshold=0.35, seed=1)
    changes = insights.settings_diff(insights.model_settings(RICH.format(**BASE)),
                                     insights.model_settings(RICH.format(**changed)))
    described = {insights.describe_change(c) for c in changes}
    assert described == {
        "validation: StratifiedKFold arg0 5 -> 10",
        "tuning: GridSearchCV param_grid {'max_depth': [3, 5]} -> {'max_depth': [3, 5, 7]}",
        "tuning: search:max_depth range [2, 10] -> [2, 15]",
        "tuning: optimize n_trials 50 -> 200",
        "calibration: CalibratedClassifierCV method sigmoid -> isotonic",
        "threshold: decision_threshold threshold 0.5 -> 0.35",
        "seed: seed seed 7 -> 1",
    }
    assert insights.changed_areas(changes) == ["validation", "tuning", "calibration", "seed", "threshold"]


def test_training_settings_of_a_neural_network_are_read():
    keras = ("model = Sequential([Dense(64), Dropout({drop}), Dense(1)])\n"
             "model.compile(optimizer=Adam(learning_rate={lr}), loss='binary_crossentropy')\n"
             "model.fit(X, y, epochs={epochs}, batch_size=256, callbacks=[EarlyStopping(patience={patience})])\n")
    before = insights.model_settings(keras.format(drop=0.2, lr=0.001, epochs=20, patience=3))
    after = insights.model_settings(keras.format(drop=0.5, lr=0.0003, epochs=50, patience=10))
    described = {insights.describe_change(c) for c in insights.settings_diff(before, after)}
    assert described == {"neural_network: Dropout arg0 0.2 -> 0.5", "optimizer: Adam learning_rate 0.001 -> 0.0003",
                         "training: fit epochs 20 -> 50", "training: EarlyStopping patience 3 -> 10"}


def test_an_insight_names_the_areas_it_changed(tmp_path, monkeypatch):
    run_id = _record_run(tmp_path, monkeypatch,
                         [RICH.format(**BASE), RICH.format(**dict(BASE, folds=10, threshold=0.35))],
                         [_rerun(1, "f1=0.10, recall=0.05", "f1=0.21, recall=0.40")])
    [m] = _insights(run_id)
    assert m["title"].startswith("validation: StratifiedKFold arg0 5 -> 10")  # areas other than the model come first
    assert "Areas changed: validation, threshold." in m["summary"]
    assert '"areas": [' in m["content"] and '"threshold"' in m["content"]
