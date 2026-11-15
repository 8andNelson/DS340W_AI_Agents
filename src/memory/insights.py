"""
Insight memories, step 1 (docs/CLAUDE_BACKEND_PLAN.md, section 18.3.1):
what changed in a baseline script between two reruns, found by code.

model_settings(script) reads every data-science step of a script with
Python's ast (the code is never run): the model, and also the split and
cross-validation, hyperparameter search, resampling, feature selection,
preprocessing, training settings (epochs, early stopping, class weights),
calibration, the decision threshold and the random seed. Each step has an
area (AREAS) and its literal arguments. settings_diff(before, after) lists
the differences: a setting changed, added or removed, a step added or
removed, or steps reordered. retain.py joins that with the rerun's measured
effect (metrics and gap before/after) to make the insight.

Pure functions: no project imports, never raise.
"""
import ast
import re

# Each area: (exact call names, a name pattern). A call matching an area is
# a step whose settings can change a result. Checked in this order.
AREAS = (
    ("validation", {"train_test_split", "KFold", "StratifiedKFold", "RepeatedKFold", "RepeatedStratifiedKFold",
                    "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit", "ShuffleSplit", "StratifiedShuffleSplit",
                    "LeaveOneOut", "LeavePOut", "cross_val_score", "cross_validate", "cross_val_predict"}, None),
    ("tuning", {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "BayesSearchCV",
                "create_study", "optimize", "fmin", "TPESampler", "suggest_int", "suggest_float", "suggest_categorical",
                "suggest_loguniform", "suggest_uniform"}, None),
    ("calibration", {"CalibratedClassifierCV", "TunedThresholdClassifierCV", "FixedThresholdClassifier"}, None),
    ("resampling", {"SMOTETomek", "SMOTEENN"},
     re.compile(r"(Sampler|SMOTE\w*|ADASYN|NearMiss|TomekLinks|EditedNearestNeighbours|CondensedNearestNeighbour)$")),
    ("class_weights", {"compute_class_weight", "compute_sample_weight"}, None),
    ("feature_selection", {"SelectKBest", "SelectPercentile", "SelectFromModel", "VarianceThreshold",
                           "SequentialFeatureSelector", "mutual_info_classif", "chi2", "f_classif"}, re.compile(r"^RFE\w*$")),
    ("preprocessing", {"get_dummies", "PolynomialFeatures", "Normalizer", "Binarizer", "PCA", "TruncatedSVD",
                       "ColumnTransformer", "FunctionTransformer", "QuantileTransformer", "PowerTransformer"},
     re.compile(r"(Scaler|Encoder|Imputer|Discretizer)$")),
    ("neural_network", {"Dense", "Dropout", "LSTM", "GRU", "Conv1D", "Conv2D", "BatchNormalization", "Embedding",
                        "Linear", "Sequential", "MultiheadAttention", "TransformerEncoderLayer"}, None),
    ("optimizer", {"Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "compile", "StepLR", "ReduceLROnPlateau",
                   "CosineAnnealingLR"}, None),
    ("training", {"EarlyStopping", "ModelCheckpoint", "DataLoader"}, None),
    ("model", {"VotingClassifier", "StackingClassifier"},
     re.compile(r"(Classifier|Regressor|Regression|SVC|SVR|NB|IsolationForest|LocalOutlierFactor|OneClassSVM|"
                r"KMeans|DBSCAN|Booster|^train)$")),
)
NOT_SETTINGS = {"Pipeline", "make_pipeline"}  # containers: their steps are read as steps of their own
# .fit(...) is a step only with training settings, not for every fitted object.
FIT_SETTINGS = {"epochs", "batch_size", "early_stopping_rounds", "eval_set", "sample_weight", "class_weight",
                "validation_split", "callbacks", "num_boost_round", "verbose_eval", "n_trials"}
SEED_CALLS = {"seed", "manual_seed", "set_seed", "set_random_seed"}
MAX_VALUE_CHARS = 80
_UNKNOWN = object()


def _name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def area_of(name: str) -> str | None:
    """The data-science area of a call name, or None when it is not a step."""
    if name in NOT_SETTINGS:
        return None
    for area, names, pattern in AREAS:
        if name in names or (pattern is not None and pattern.search(name)):
            return area
    return None


def _constants(tree: ast.AST) -> dict:
    """Variables the script sets to one literal value (N_TREES = 500). A name
    set more than once, or to anything else, is left out."""
    seen: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                value = _UNKNOWN
            seen[name] = value if name not in seen else _UNKNOWN
    return {k: v for k, v in seen.items() if v is not _UNKNOWN}


def _literal(node: ast.AST, constants: dict):
    """A literal (also through a variable set to one literal), else _UNKNOWN."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal(node.operand, constants)
        return -value if isinstance(value, (int, float)) else _UNKNOWN
    return _UNKNOWN


def _value(node: ast.AST, constants: dict = None):
    """A literal as itself; a nested call as "<Name(...)>"; anything else as
    its source text (a variable, an expression)."""
    value = _literal(node, constants or {})
    if value is not _UNKNOWN:
        return value
    if isinstance(node, ast.Call):
        return f"<{_name(node.func)}(...)>"
    try:
        text = ast.unparse(node)
    except Exception:
        text = type(node).__name__
    return f"expr:{text[:MAX_VALUE_CHARS]}"


def _call_step(node: ast.Call, constants: dict) -> dict | None:
    name = _name(node.func)
    keywords = {k.arg: _value(k.value, constants) for k in node.keywords if k.arg}
    if name == "fit":
        training = {k: v for k, v in keywords.items() if k in FIT_SETTINGS}
        return {"call": "fit", "area": "training", "params": training} if training else None
    if name in SEED_CALLS:
        seed = _literal(node.args[0], constants) if node.args else _UNKNOWN
        return {"call": name, "area": "seed", "params": {"seed": seed}} if seed is not _UNKNOWN else None
    area = area_of(name)
    if area is None:
        return None
    if name.startswith("suggest_") and node.args:
        # optuna: trial.suggest_int("n_estimators", 100, 1000) -> the searched range of one hyperparameter
        searched = _literal(node.args[0], constants)
        bounds = [_value(a, constants) for a in node.args[1:]]
        return {"call": f"search:{searched}", "area": "tuning",
                "params": {"range": bounds, **keywords} if bounds else keywords}
    # Literal positional arguments count too (StratifiedKFold(5), PolynomialFeatures(2));
    # data arguments (X, y) are variables, not settings, and are left out.
    positional = {f"arg{i}": v for i, v in enumerate(_literal(a, constants) for a in node.args) if v is not _UNKNOWN}
    if name == "fmin":
        keywords = {k: v for k, v in keywords.items() if k in ("max_evals", "algo")}
    if name == "compile":
        keywords = {k: v for k, v in keywords.items() if k in ("optimizer", "loss", "metrics")}
        if not keywords:
            return None  # re.compile(...) and the like, not a Keras model
    if name == "optimize":
        keywords = {k: v for k, v in keywords.items() if k in ("n_trials", "timeout")}
        if not keywords:
            return None  # some other .optimize(...), not an optuna study
    return {"call": name, "area": area, "params": {**positional, **keywords}}


def _threshold_step(node: ast.Compare, constants: dict) -> dict | None:
    """A decision threshold: predict_proba(...) compared with a number
    (proba >= 0.38), instead of the default 0.5 of predict()."""
    sides = [node.left, *node.comparators]
    try:
        text = " ".join(ast.unparse(s) for s in sides)
    except Exception:
        return None
    if "proba" not in text and "prob" not in text.lower():
        return None
    for side in sides:
        value = _literal(side, constants)
        if isinstance(value, float) and 0.0 < value < 1.0:
            return {"call": "decision_threshold", "area": "threshold", "params": {"threshold": value}}
    return None


def model_settings(script: str) -> list[dict] | None:
    """Every data-science step of a script, in source order:
    [{"call": "StratifiedKFold", "area": "validation", "params": {"n_splits": 5}}, ...]
    or None when the script does not parse."""
    try:
        tree = ast.parse(script or "")
    except (SyntaxError, ValueError):
        return None
    constants = _constants(tree)
    steps = []
    for node in ast.walk(tree):
        step = _call_step(node, constants) if isinstance(node, ast.Call) else \
            _threshold_step(node, constants) if isinstance(node, ast.Compare) else None
        if step:
            steps.append((node.lineno, node.col_offset, step))
    return [s for _line, _col, s in sorted(steps, key=lambda s: (s[0], s[1]))]


def _grouped(settings: list) -> dict:
    groups: dict = {}
    for item in settings:
        groups.setdefault(item["call"], []).append(item)
    return groups


def settings_diff(before: list, after: list) -> list[dict]:
    """The differences between two scripts' steps. Steps of the same name
    are paired in source order. Each change:
    {"call", "area", "param" (None for a whole step), "before", "after"};
    a value that is absent is "absent" (a default, not None)."""
    changes = []
    old, new = _grouped(before or []), _grouped(after or [])
    for call in sorted(set(old) | set(new), key=lambda c: (c not in old, c)):
        olds, news = old.get(call, []), new.get(call, [])
        for i in range(max(len(olds), len(news))):
            label = call if max(len(olds), len(news)) == 1 else f"{call}#{i + 1}"
            area = (olds[i] if i < len(olds) else news[i])["area"]
            if i >= len(olds):
                changes.append({"call": label, "area": area, "param": None, "before": "absent",
                                "after": news[i]["params"]})
            elif i >= len(news):
                changes.append({"call": label, "area": area, "param": None, "before": olds[i]["params"],
                                "after": "absent"})
            else:
                for param in sorted(set(olds[i]["params"]) | set(news[i]["params"])):
                    b, a = olds[i]["params"].get(param, "absent"), news[i]["params"].get(param, "absent")
                    if b != a and not (_is_expr(b) and _is_expr(a)):  # two unknown expressions: a rename, not a fact
                        changes.append({"call": label, "area": area, "param": param, "before": b, "after": a})
    order_before = [c["call"] for c in before or []]
    order_after = [c["call"] for c in after or []]
    if order_before != order_after and sorted(order_before) == sorted(order_after):
        # The same steps in a new order (e.g. oversampling moved before feature selection).
        changes.append({"call": "order", "area": "pipeline_order", "param": None, "before": order_before,
                        "after": order_after})
    return changes


def changed_areas(changes: list) -> list:
    """The areas a rerun changed, in a fixed order (model last: the other
    areas are what is easy to miss)."""
    order = [a for a, _n, _p in AREAS if a != "model"] + ["seed", "threshold", "pipeline_order", "model"]
    found = {c.get("area") for c in changes}
    return [a for a in order if a in found]


def _is_expr(value) -> bool:
    return isinstance(value, str) and (value.startswith("expr:") or value.startswith("<"))


def describe_change(change: dict, brief: bool = False) -> str:
    """One change in words: "validation: StratifiedKFold n_splits 5 -> 10".
    brief: an added or removed step without its settings (for a title)."""
    prefix = f"{change['area']}: " if change.get("area") and change["area"] != "pipeline_order" else ""
    if change["call"] == "order":
        before, after = change["before"], change["after"]
        i = next(i for i, (b, a) in enumerate(zip(before, after)) if b != a)
        moved = after[i]
        jumped = before[i:before.index(moved, i)] if moved in before[i:] else [before[i]]
        return f"steps reordered: {moved} moved before {', '.join(jumped)}"
    if change["param"] is None:
        verb = "added" if change["before"] == "absent" else "removed"
        params = change["after"] if verb == "added" else change["before"]
        shown = ", ".join(f"{k}={_show(v)}" for k, v in params.items()) if isinstance(params, dict) and not brief             else ""
        return f"{prefix}{change['call']} {verb}" + (f" ({shown})" if shown else "")
    return f"{prefix}{change['call']} {change['param']} {_show(change['before'])} -> {_show(change['after'])}"


def _show(value) -> str:
    if value == "absent":
        return "default"
    return str(value)[:MAX_VALUE_CHARS]


_METRIC = re.compile(r"([A-Za-z][\w\-]*)\s*=\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_metrics(summary: str) -> dict:
    """Master Agent's rerun metric summary ("accuracy=0.94, f1=0.0976") as
    {name: value}."""
    return {name: float(value) for name, value in _METRIC.findall(summary or "")}
