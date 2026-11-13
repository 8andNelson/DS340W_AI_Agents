"""
Task features of a memory (docs/CLAUDE_BACKEND_PLAN.md, section 5.2):
what task a memory belongs to, so later recall can match it to a new one.

Made deterministically from recorded data -- the Intake Agent's request,
the baseline script (Python ast, never executed), the Cleaning and Data
Agents' results, the evaluation -- never by an LLM. Values come from fixed
vocabularies, so they mean the same thing on every machine; a value
outside a vocabulary is kept as "other:<word>". An unknown value is left
out, not guessed.
"""
import ast
import re

from src.orchestration.run_metrics import metric_key, model_families

TASK_TYPES = ("binary", "multiclass", "regression", "anomaly")

# Identifiers, call names and keyword arguments in a script -> technique.
TECHNIQUES = {
    "smote": ("SMOTE", "BorderlineSMOTE", "SVMSMOTE", "SMOTENC", "SMOTETomek", "SMOTEENN"),
    "adasyn": ("ADASYN",),
    "oversampling": ("RandomOverSampler",),
    "undersampling": ("RandomUnderSampler", "NearMiss", "TomekLinks", "EditedNearestNeighbours"),
    "class_weight": ("class_weight", "scale_pos_weight", "compute_class_weight", "auto_class_weights"),
    "standard_scaler": ("StandardScaler",),
    "minmax_scaler": ("MinMaxScaler",),
    "robust_scaler": ("RobustScaler",),
    "pca": ("PCA",),
    "one_hot": ("OneHotEncoder", "get_dummies"),
    "grid_search": ("GridSearchCV",),
    "random_search": ("RandomizedSearchCV",),
    "bayesian_search": ("hyperopt", "optuna", "BayesSearchCV", "fmin"),
    "cross_validation": ("cross_val_score", "cross_validate", "cross_val_predict", "KFold", "StratifiedKFold"),
    "early_stopping": ("early_stopping_rounds", "EarlyStopping", "early_stopping"),
    "threshold_tuning": ("precision_recall_curve", "decision_threshold"),
}

_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/][^\s'\"]+")
_ID = re.compile(r"\b[0-9a-f]{8,}(?:-[0-9a-f]{4,})*\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
# A quoted name, but not an apostrophe inside a word ("paper's").
_QUOTED = re.compile(r"(?<!\w)'[^']*'(?!\w)|\"[^\"]*\"")


def _words(text: str) -> set:
    return set(re.findall(r"\w+", text or ""))


def script_names(script: str) -> set:
    """Identifiers, attributes, keyword argument names and imported names of a script."""
    try:
        names = set()
        for node in ast.walk(ast.parse(script or "")):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
            elif isinstance(node, ast.alias):
                names.update(node.name.split("."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
        return names
    except SyntaxError:
        return _words(script)


def techniques(script: str) -> list:
    names = script_names(script)
    found = {t for t, markers in TECHNIQUES.items() if names & set(markers)}
    if "train_test_split" in names and "stratify" in names:
        found.add("stratified_split")
    return sorted(found)


def task_type(ml_task) -> str | None:
    """The Intake Agent's ml_task -> binary / multiclass / regression / anomaly.
    'classification' and 'detection' are taken as binary: the pipeline's
    target is one outcome (target_terms), predicted or not. A model may give
    ml_task as a list ("classification", "anomaly detection")."""
    if isinstance(ml_task, (list, tuple)):
        ml_task = " ".join(str(t) for t in ml_task)
    text = str(ml_task or "").lower()
    if not text:
        return None
    if "anomal" in text or "outlier" in text:
        return "anomaly"
    if "regress" in text or "forecast" in text:
        return "regression"
    if "multi" in text:
        return "multiclass"
    if "classif" in text or "detect" in text:
        return "binary"
    return "other:" + metric_key(text)[:30]


def _bucket(value, edges: tuple, labels: tuple) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def dataset_traits(cleaning: dict | None, dataset: dict | None) -> dict:
    cleaning, dataset = cleaning or {}, dataset or {}
    traits = {
        "rows": _bucket(cleaning.get("rows_total"), (1_000, 10_000, 100_000, 1_000_000),
                        ("<1k", "1k-10k", "10k-100k", "100k-1M", ">=1M")),
        "columns": _bucket(cleaning.get("columns_total"), (10, 30, 100), ("<10", "10-29", "30-99", ">=100")),
        "sampled": bool(cleaning.get("sampled")) if cleaning else None,
    }
    notes = " ".join(str(d.get("column_notes") or "") for d in dataset.get("selected_datasets") or [])
    if dataset.get("selected_datasets"):
        traits["anonymized_columns"] = bool(re.search(r"anonym|pca|undescribed|no descriptive", notes, re.I))
    return {k: v for k, v in traits.items() if v is not None}


def failure_signature(text: str) -> str | None:
    """A failure in a form that matches across runs: exception type plus its
    message with paths, ids, numbers and quoted names removed."""
    if not text:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    # A traceback ends with "Type: message"; otherwise use the first line.
    error_lines = [line for line in lines if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exceeded|Interrupt)\b", line)]
    line = error_lines[-1] if error_lines else lines[0] if lines else ""
    line = _PATH.sub("<path>", line)
    line = _QUOTED.sub("<name>", line)
    line = _ID.sub("<id>", line)
    line = _NUMBER.sub("<n>", line)
    return re.sub(r"\s+", " ", line).strip().lower()[:160] or None


def task_features(run: dict, agent: str, script: str = None, failure: str = None) -> dict:
    """The features of one memory: the run's task plus what is known of
    this agent's part (plan 5.2). run is retain.load_run's record."""
    intake = run.get("intake") or {}
    parent = run.get("parent_paper") or {}
    replication = run.get("replication") or {}
    evaluation = run.get("final_evaluation") or {}
    features = {
        "agent": agent,
        "topic": run.get("topic"),
        "target_terms": intake.get("target_terms") or None,
        "task_type": task_type(intake.get("ml_task")),
        "domain_tags": sorted({w for k in intake.get("keywords") or [] for w in re.findall(r"[a-z]{3,}", str(k).lower())})[:12] or None,
        "parent_paper": ({"doi": parent.get("doi") or None, "title": (parent.get("title") or "").strip().lower() or None}
                         if parent else None),
        "dataset_identity": [d.get("name") for d in (run.get("dataset") or {}).get("selected_datasets") or []
                             if d.get("name")] or None,
        "dataset_traits": dataset_traits(run.get("cleaning"), run.get("dataset")) or None,
        "metrics": sorted({metric_key(m.get("metric")) for m in evaluation.get("metrics") or []}) or None,
        "source": replication.get("source") or None,
    }
    if script:
        features["model_family"] = model_families(script) or None
        features["techniques"] = techniques(script) or None
    elif parent.get("methodology"):
        features["model_family"] = model_families(parent["methodology"]) or None
    if failure:
        features["failure_signature"] = failure_signature(failure)
    return {k: v for k, v in features.items() if v not in (None, [], {})}


def tags(features: dict) -> list:
    """A flat list of feature values for fast filters ("model:xgboost")."""
    out = []
    for key, prefix in (("agent", "agent"), ("task_type", "task"), ("source", "source"),
                        ("failure_signature", "failure")):
        if features.get(key):
            out.append(f"{prefix}:{features[key]}")
    for key, prefix in (("model_family", "model"), ("techniques", "tech"), ("metrics", "metric"),
                        ("target_terms", "target")):
        out.extend(f"{prefix}:{v}" for v in features.get(key) or [])
    paper = features.get("parent_paper") or {}
    if paper.get("doi"):
        out.append(f"paper:{paper['doi'].lower()}")
    return out
