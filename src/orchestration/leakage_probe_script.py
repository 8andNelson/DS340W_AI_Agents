"""
The probe leakage_probe.py runs in the replication venv (the project's own
.venv has no scikit-learn). Not imported by the pipeline; run as a script:

    python -I leakage_probe_script.py <master.csv> <label column> <max rows>

It trains the same standard random forest twice on the run's own data and
prints one JSON line with the test scores of each:

  * honest -- stratified 70/30 split, then the minority class is
    oversampled (copied) in the training rows only;
  * leaky  -- the whole dataset is oversampled first, then split, so copies
    of the same minority rows are in both train and test.

Fixed seed, deterministic. Never prints anything but that JSON line.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.3
MAX_CATEGORIES = 50  # a text column with more values is an identifier or free text, not a feature


def oversample(X: np.ndarray, y: np.ndarray, rng) -> tuple:
    """Copy random minority rows until both classes have the same count."""
    minority = int(np.argmin(np.bincount(y)))
    extra = rng.choice(np.flatnonzero(y == minority), size=abs(int((y == 0).sum() - (y == 1).sum())), replace=True)
    order = rng.permutation(len(y) + len(extra))
    return np.vstack([X, X[extra]])[order], np.concatenate([y, y[extra]])[order]


def scores(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, prob)}


def fit(X, y):
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X, y)


def main(path: str, label: str, max_rows: int) -> dict:
    df = pd.read_csv(path, low_memory=False)
    if label not in df.columns:
        return {"error": f"label column '{label}' is not in the data"}
    df = df.dropna(subset=[label])
    values = list(df[label].unique())
    if len(values) != 2:
        return {"error": f"the label has {len(values)} values; the probe needs a binary label"}
    positive = 1 if 1 in values else df[label].value_counts().idxmin()
    y = (df[label] == positive).astype(int).to_numpy()
    X = df.drop(columns=[label] + [c for c in df.columns if c.startswith("_source")])
    for column in list(X.columns):
        unique = X[column].nunique(dropna=True)
        numeric = pd.api.types.is_numeric_dtype(X[column])
        if (not numeric and unique > MAX_CATEGORIES) or (numeric and unique == len(X)) or unique <= 1:
            X = X.drop(columns=column)  # identifiers and constant columns
    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    X = pd.get_dummies(X, dummy_na=True).astype(float).to_numpy()
    if len(y) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=SEED)

    rng = np.random.default_rng(SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    honest = scores(fit(*oversample(X_train, y_train, rng)), X_test, y_test)

    X_all, y_all = oversample(X, y, np.random.default_rng(SEED))
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                                                        random_state=SEED)
    leaky = scores(fit(X_train, y_train), X_test, y_test)
    return {"rows": int(len(y)), "positive_rate": float(y.mean()), "features": int(X.shape[1]),
            "honest": {k: round(float(v), 4) for k, v in honest.items()},
            "leaky": {k: round(float(v), 4) for k, v in leaky.items()}}


if __name__ == "__main__":
    try:
        result = main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    except Exception as e:  # reported to the caller, never raised
        result = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result))
