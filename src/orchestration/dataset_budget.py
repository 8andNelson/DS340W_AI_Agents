"""
Row budget and label detection shared by the Data and Cleaning Agents.

Local runs must not stall on very large datasets (a 24.4M-row, 2.3 GB
Kaggle file was accepted in a live run). Every dataset is therefore capped
at MAX_ROWS_PER_DATASET rows as it is read, and the merged master CSV at
MAX_MASTER_ROWS, by a deterministic sample stratified on the label column
so the class ratio -- critical for imbalanced tasks like fraud detection --
is preserved. Every cap is recorded, never silent: callers report the
original and sampled row counts.

The sample is single-pass and memory-bounded (read_csv_capped): each chunk
gets a seeded random key per row, and only the smallest-key rows per class
are kept, so a file far larger than memory is never fully loaded.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

MAX_ROWS_PER_DATASET = 250_000
MAX_MASTER_ROWS = 500_000
SAMPLE_SEED = 42
CHUNK_ROWS = 200_000
LABEL_SCAN_ROWS = 5_000

_KEY = "__sample_key"
_ROW = "__row_order"

_POSITIVE_LABELS = {"1", "1.0", "yes", "y", "true", "t", "fraud", "positive"}
_NEGATIVE_LABELS = {"0", "0.0", "no", "n", "false", "f", "legit", "legitimate", "negative", "normal"}


def _compact(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def find_label_column(df: pd.DataFrame, target_terms: list) -> str | None:
    """The column that holds the project's prediction target: its name
    contains a target term (e.g. "fraud" in "is_fraud" / "Is Fraud?") and
    it has exactly two distinct non-null values. None when no column
    qualifies or no target terms are known."""
    terms = [_compact(t) for t in (target_terms or []) if _compact(t)]
    if not terms:
        return None
    for column in df.columns:
        if any(term in _compact(column) for term in terms) and df[column].dropna().nunique() == 2:
            return column
    return None


def binary_label_map(values) -> dict | None:
    """Map a two-valued label's values onto 0/1, so labels written as 0/1,
    "Yes"/"No" or True/False merge into one column. None when the values
    are not recognizably positive/negative -- the caller must then not
    merge on that label rather than guess which value means fraud."""
    mapping = {}
    for value in values:
        key = str(value).strip().lower()
        if key in _POSITIVE_LABELS:
            mapping[value] = 1
        elif key in _NEGATIVE_LABELS:
            mapping[value] = 0
        else:
            return None
    return mapping if sorted(set(mapping.values())) == [0, 1] else None


def check_target_fit(local_csv_path: str, target_terms: list, dataset_text: str = "") -> tuple[bool, str]:
    """Does a downloaded dataset actually predict the project's target?
    Deterministic: a column whose name contains a target term, or (for
    datasets with a generic label such as "Class") the dataset's own name/
    description containing one. Catches a dataset that only shares the
    topic's vocabulary -- e.g. "Default of Credit Card Clients" (loan
    default) for a credit-card *fraud* project. Not checked (passes, with a
    note) when no target terms are known or the file can't be re-read."""
    terms = [t for t in (target_terms or []) if _compact(t)]
    if not terms:
        return True, "not checked -- no target terms known"
    try:
        sample = pd.read_csv(local_csv_path, nrows=LABEL_SCAN_ROWS, low_memory=False)
    except Exception as e:
        return True, f"not checked -- could not re-read downloaded file ({e})"

    label = find_label_column(sample, terms)
    if label:
        return True, f"label column '{label}' matches target {terms}"
    if any(_compact(t) in _compact(dataset_text) for t in terms):
        return True, f"dataset name/description matches target {terms}"
    return False, (f"no column or dataset description matches the prediction target {terms} "
                   f"(columns: {', '.join(map(str, list(sample.columns)[:12]))}) -- the dataset "
                   "shares the topic's vocabulary but predicts something else")


def stratified_cap(df: pd.DataFrame, max_rows: int, label_column: str | None = None,
                   seed: int = SAMPLE_SEED) -> tuple[pd.DataFrame, dict]:
    """In-memory version of the cap for a frame already loaded. Returns
    (frame, info) -- info records whether and how it was sampled."""
    info = {"original_rows": len(df), "sampled_rows": len(df), "sampled": False,
            "label_column": label_column, "seed": seed}
    if len(df) <= max_rows:
        return df, info
    keyed = df.assign(**{_KEY: np.random.default_rng(seed).random(len(df)), _ROW: np.arange(len(df))})
    sampled = _proportional_take(keyed, max_rows, label_column, Counter(_class_counts(keyed, label_column)))
    info.update(sampled_rows=len(sampled), sampled=True)
    return sampled, info


def read_csv_capped(source, max_rows: int = MAX_ROWS_PER_DATASET, target_terms: list = None,
                    seed: int = SAMPLE_SEED, **read_kwargs) -> tuple[pd.DataFrame, dict]:
    """Read a CSV (path or file-like) keeping at most max_rows rows,
    stratified on the label column when one is found. Single pass; memory
    holds at most one chunk plus max_rows rows per class. Returns
    (frame, info); rows keep their original file order."""
    rng = np.random.default_rng(seed)
    kept = None
    counts = Counter()
    total = 0
    label_column = None

    # Per class: how many rows are kept, and -- once max_rows are kept -- the
    # largest key among them. A later row with a key at or above that
    # threshold can never enter the class's final smallest-max_rows set, so
    # it is dropped with one vectorized comparison instead of being
    # concatenated and re-sorted (a 24M-row file then costs little more
    # than parsing it).
    kept_sizes = Counter()
    thresholds = {}

    if read_kwargs.get("engine") != "python":  # the python engine rejects low_memory
        read_kwargs.setdefault("low_memory", False)
    for index, chunk in enumerate(pd.read_csv(source, chunksize=CHUNK_ROWS, **read_kwargs)):
        if index == 0:
            label_column = find_label_column(chunk.head(LABEL_SCAN_ROWS), target_terms)
        chunk = chunk.assign(**{_KEY: rng.random(len(chunk)), _ROW: np.arange(total, total + len(chunk))})
        total += len(chunk)
        classes = _class_labels(chunk, label_column)
        counts.update(classes.value_counts().to_dict())
        if thresholds:
            limit = classes.map(thresholds).fillna(np.inf).to_numpy()
            mask = chunk[_KEY].to_numpy() < limit
            chunk, classes = chunk[mask], classes[mask]
        kept_sizes.update(classes.value_counts().to_dict())
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        # Trim only when a class holds over twice its share, so the trim
        # (a per-class partial sort) runs rarely.
        if any(size > 2 * max_rows for size in kept_sizes.values()):
            kept = _smallest_keys_per_class(kept, max_rows, label_column)
            kept_sizes, thresholds = Counter(), {}
            for name, group in _class_groups(kept, label_column):
                kept_sizes[name] = len(group)
                if len(group) >= max_rows:
                    thresholds[name] = group[_KEY].max()

    if kept is None:
        return pd.DataFrame(), {"original_rows": 0, "sampled_rows": 0, "sampled": False,
                                "label_column": None, "seed": seed}

    info = {"original_rows": total, "sampled_rows": min(total, max_rows), "sampled": total > max_rows,
            "label_column": label_column, "seed": seed}
    if total <= max_rows:
        return kept.sort_values(_ROW).drop(columns=[_KEY, _ROW]).reset_index(drop=True), info
    sampled = _proportional_take(kept, max_rows, label_column, counts)
    info["sampled_rows"] = len(sampled)
    return sampled, info


def _class_labels(df: pd.DataFrame, label_column: str | None) -> pd.Series:
    """Each row's class as a string (one shared class without a label)."""
    if not label_column:
        return pd.Series("__all__", index=df.index)
    return df[label_column].astype(str)


def _class_counts(df: pd.DataFrame, label_column: str | None) -> dict:
    return _class_labels(df, label_column).value_counts().to_dict()


def _class_groups(df: pd.DataFrame, label_column: str | None):
    if not label_column:
        return [("__all__", df)]
    return list(df.groupby(df[label_column].astype(str), dropna=False, sort=True))


def _smallest_keys_per_class(df: pd.DataFrame, n: int, label_column: str | None) -> pd.DataFrame:
    return pd.concat([group.nsmallest(n, _KEY) for _, group in _class_groups(df, label_column)],
                     ignore_index=True)


def _proportional_take(df: pd.DataFrame, max_rows: int, label_column: str | None, counts: Counter) -> pd.DataFrame:
    """Take max_rows rows split across classes in proportion to their full
    counts (at least one row per class that exists), smallest keys first."""
    total = sum(counts.values())
    groups = _class_groups(df, label_column)
    shares = {name: max(1, round(max_rows * counts.get(name, 0) / total)) if total else 0 for name, _ in groups}
    # Rounding can overshoot max_rows; trim the largest class, never the
    # rare one (in fraud data the rare class is the one that matters).
    while sum(shares.values()) > max_rows:
        largest = max(shares, key=shares.get)
        shares[largest] -= 1
    sampled = pd.concat([group.nsmallest(shares[name], _KEY) for name, group in groups], ignore_index=True)
    return sampled.sort_values(_ROW).drop(columns=[_KEY, _ROW]).reset_index(drop=True)
