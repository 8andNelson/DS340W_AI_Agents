"""
Deterministic checks for data leakage in a training script.

The Evaluation Agent runs these before its LLM review. They read the
script's syntax tree (ast), never run it, and report only what the code
literally does, each with the exact line as evidence:

  - resampling (SMOTE, RandomOverSampler, ... -- any .fit_resample) before
    the train/test split, on test data, or on data combined from the train
    and test sets: synthetic or duplicated test rows end up in training,
    so the test score is inflated;
  - a preprocessing step (scaler, encoder, imputer, feature selector, PCA,
    ...) fitted before the split, on test data, or on combined data;
  - a model fitted, tuned or early-stopped with the test set
    (.fit(X_test...), eval_set=[(X_test, ...)]);
  - cross-validation run on already-resampled data, so every validation
    fold contains synthetic copies of its own training rows.

Order is by line number, which matches execution order for a top-level
script (what the Replication Agent generates). The checks are heuristics
over names: a finding is strong evidence, and the Evaluation Agent's
review still weighs it, but every finding quotes real code.
"""
import ast
import re

_TEST_NAME = re.compile(r"(?i)\b(x_?test|y_?test|test_?df|df_?test|test_?data|test_?set|x_?te|y_?te|test)\b")
_TRAIN_NAME = re.compile(r"(?i)\b(x_?train|y_?train|train_?df|df_?train|train_?data|train_?set|x_?tr|y_?tr|train)\b")
_PREPROCESSOR = re.compile(r"(?i)(scal|encod|imput|select|pca|normaliz|vectoriz|discretiz|kbest|variance|"
                           r"transformer|binariz|polynomial|power|quantile|robust|minmax|standard)")
_CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict"}
_SEARCH_CLASSES = re.compile(r"(?i)(gridsearch|randomizedsearch|halving)")


def _src(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _args_src(call: ast.Call) -> str:
    return ", ".join([_src(a) for a in call.args] + [_src(k) for k in call.keywords])


def _assigned_names(node: ast.AST) -> set:
    names = set()
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    for target in targets:
        for sub in ast.walk(target) if target is not None else []:
            if isinstance(sub, ast.Name):
                names.add(sub.id)
    return names


def _uses(text: str, names: set) -> bool:
    return any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)


def find_leakage(script: str) -> list:
    """Leakage findings for a Python script: a list of
    {"check", "category", "severity", "issue", "evidence", "line", "fix"}.
    [] when nothing is found or the script does not parse."""
    try:
        tree = ast.parse(script or "")
    except SyntaxError:
        return []
    lines = (script or "").splitlines()

    def evidence(lineno: int) -> str:
        return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""

    split_lines = sorted(n.lineno for n in ast.walk(tree)
                         if isinstance(n, ast.Call) and _call_name(n) == "train_test_split")
    first_split = split_lines[0] if split_lines else None

    combined, resampled = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(getattr(node, "value", None), ast.Call):
            call = node.value
            args = _args_src(call)
            if _call_name(call) in ("concat", "vstack", "hstack", "append") and _TEST_NAME.search(args) \
                    and _TRAIN_NAME.search(args):
                combined |= _assigned_names(node)
            if _call_name(call) in ("fit_resample", "fit_sample"):
                resampled |= _assigned_names(node)

    findings, seen = [], set()

    def add(check: str, severity: str, issue: str, lineno: int, fix: str):
        key = (check, lineno)
        if key in seen:
            return
        seen.add(key)
        findings.append({"check": check, "category": "data_leakage", "severity": severity, "issue": issue,
                         "evidence": evidence(lineno), "line": lineno, "fix": fix})

    for node in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: n.lineno):
        name = _call_name(node)
        args = _args_src(node)
        receiver = _src(node.func.value) if isinstance(node.func, ast.Attribute) else ""

        if name in ("fit_resample", "fit_sample"):
            if first_split is not None and node.lineno < first_split:
                add("resample_before_split", "critical",
                    "Resampling (e.g. SMOTE) is applied before the train/test split, so synthetic samples built "
                    "from test rows leak into training and the test score is inflated.", node.lineno,
                    "Split first; call fit_resample on the training data only.")
            elif _TEST_NAME.search(args):
                add("resample_on_test", "critical",
                    "Resampling is applied to test data; the test set must keep its real class distribution.",
                    node.lineno, "Resample only the training data; evaluate on the untouched test set.")
            elif combined and _uses(args, combined):
                add("resample_on_combined", "critical",
                    "Resampling is applied to data combined from the train and test sets.", node.lineno,
                    "Resample only the training data.")

        elif name in ("fit", "fit_transform"):
            is_preprocessor = bool(_PREPROCESSOR.search(receiver))
            if _TEST_NAME.search(args):
                if is_preprocessor:
                    add("preprocessor_fit_on_test", "critical",
                        f"A preprocessing step ({receiver}) is fitted on test data.", node.lineno,
                        "Fit preprocessing on the training data; only transform the test data.")
                elif "eval_set" in args or "validation_data" in args:
                    add("test_used_for_early_stopping", "major",
                        "The test set is used as the evaluation set during training (early stopping / model "
                        "selection), so it is no longer an unseen test set.", node.lineno,
                        "Use the validation split (VALIDATION_CSV) for early stopping; keep the test set unseen.")
                else:
                    add("model_fit_on_test", "critical",
                        f"{receiver or 'A model'} is fitted on test data.", node.lineno,
                        "Fit only on training data.")
            elif is_preprocessor and first_split is not None and node.lineno < first_split:
                add("preprocessor_fit_before_split", "major",
                    f"A preprocessing step ({receiver}) is fitted before the train/test split, so test-set "
                    "statistics leak into training.", node.lineno,
                    "Split first; fit preprocessing on the training data only.")
            elif is_preprocessor and combined and _uses(args, combined):
                add("preprocessor_fit_on_combined", "major",
                    f"A preprocessing step ({receiver}) is fitted on data combined from the train and test sets.",
                    node.lineno, "Fit preprocessing on the training data only.")
            elif resampled and _SEARCH_CLASSES.search(receiver) and _uses(args, resampled):
                add("cv_on_resampled_data", "major",
                    "Hyperparameter search runs cross-validation on already-resampled data, so validation folds "
                    "contain synthetic copies of their own training rows.", node.lineno,
                    "Put the resampler inside an imblearn Pipeline so it runs within each training fold.")

        elif name in _CV_FUNCS and resampled and _uses(args, resampled):
            add("cv_on_resampled_data", "major",
                "Cross-validation runs on already-resampled data, so validation folds contain synthetic copies "
                "of their own training rows.", node.lineno,
                "Put the resampler inside an imblearn Pipeline so it runs within each training fold.")

    return findings
