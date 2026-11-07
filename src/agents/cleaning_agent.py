"""
Cleaning Agent.

Takes the datasets Data Agent already downloaded and verified
(`dataset_result["selected_datasets"]`, each guaranteed to have a real
local CSV at `local_csv_path` -- see data_agent.py) and merges them into a
single cleaned master CSV for the Parent Paper's replication.

Does no downloading of its own -- if a candidate has no local CSV, that is
itself logged as a conflict rather than assumed away.

Per dataset: read the CSV under the row budget (dataset_budget: at most
MAX_ROWS_PER_DATASET rows, a seeded sample stratified on the label), normalize
column names (whitespace/case/separators), drop fully-empty rows/columns and
exact-duplicate rows, and tag every row with its source dataset for
provenance.

Merging is label-aware when the project's target terms are known (the
Intake Agent's target_terms, e.g. ["fraud"]):

  - The anchor is the first dataset whose label column matches the target
    -- not simply the first dataset in the list, which in a live run let a
    loan-default dataset push out two real fraud datasets.
  - A dataset with no label column for the target is excluded and logged.
  - Labels are normalized to 0/1 (0/1, Yes/No and True/False all merge).
  - Other columns are aligned onto the anchor by verified semantic mapping
    (src/orchestration/column_mapping.py): the AI proposes, the data must
    confirm it, and every accepted or rejected mapping is logged to
    logs/column_mappings.json. Semantic renaming is therefore allowed, but
    only verified and logged -- never silent.

Without target terms, the earlier rule applies: datasets merge on shared
(normalized) column names, and one sharing no column is excluded.

Accepted datasets are combined with a union-of-columns concat (missing
values become blank), deduplicated once more across the whole merged set,
and capped at MAX_MASTER_ROWS (stratified). Every reason a dataset couldn't be merged -- whether it had no CSV,
an unreadable file, or an incompatible schema -- is written to
logs/cleaning_conflicts.json and printed as it's found, per CLAUDE.md's
"never silently drop, always log why" policy.

The project's >=10,000-entry target is enforced here, on the merged
master CSV's actual row count -- not on any individual dataset. Data
Agent no longer gates individual candidates on entry count (removed so
Kaggle's own dataset-search API, which doesn't expose row counts up
front, can be used for discovery); this is the one place total volume
actually gets checked, since it's the only point where the true, final,
deduplicated row count is known.

Once the master CSV is written, it's also split into train (70%), test
(20%), and validation (10%) CSVs for the future Modeling Agent to consume.
The split is a single fixed-seed shuffle-then-slice (SPLIT_SEED) -- not
stratified, since Cleaning Agent has no notion of what the eventual
modeling target column will be -- so it's reproducible across runs
(CLAUDE.md's Environment Reproducibility: record random seeds) without
assuming anything about the data's shape.
"""
import json
import re
from pathlib import Path

import pandas as pd

from src.orchestration import column_mapping
from src.orchestration.dataset_budget import (
    MAX_MASTER_ROWS, MAX_ROWS_PER_DATASET, binary_label_map, find_label_column, read_csv_capped, stratified_cap,
)

REPO_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
CONFLICTS_LOG = REPO_ROOT / "logs" / "cleaning_conflicts.json"
MAPPINGS_LOG = REPO_ROOT / "logs" / "column_mappings.json"
MASTER_CSV_NAME = "master_dataset.csv"
TRAIN_CSV_NAME = "train.csv"
TEST_CSV_NAME = "test.csv"
VALIDATION_CSV_NAME = "validation.csv"

SOURCE_COLUMN = "_source_dataset"
TARGET_ROWS = 10000

TRAIN_FRAC = 0.7
TEST_FRAC = 0.2
VALIDATION_FRAC = 0.1  # implicit: whatever's left after train/test
SPLIT_SEED = 42


def _normalize_column_name(col) -> str:
    """Deterministic normalization only (whitespace/case/separators) -- no
    semantic/fuzzy matching, so overlap detection never guesses that two
    differently-named columns mean the same thing."""
    name = str(col).strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "column"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize every column name, de-duplicating collisions that the
    normalization itself creates (e.g. 'Amount' and 'amount ' both
    normalizing to 'amount')."""
    seen = {}
    new_cols = []
    for c in df.columns:
        norm = _normalize_column_name(c)
        if norm in seen:
            seen[norm] += 1
            norm = f"{norm}_{seen[norm]}"
        else:
            seen[norm] = 1
        new_cols.append(norm)
    df = df.copy()
    df.columns = new_cols
    return df


def _clean_dataframe(df: pd.DataFrame, source_link: str) -> pd.DataFrame:
    df = _normalize_columns(df)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.drop_duplicates()
    df[SOURCE_COLUMN] = source_link
    return df


def _log_conflict(conflicts: list, label: str, link: str, reason: str) -> None:
    conflicts.append({"dataset": label, "link": link, "reason": reason})
    print(f"[Cleaning Agent] Skipped '{label}': {reason}")


def _write_conflicts_log(conflicts: list) -> None:
    CONFLICTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    CONFLICTS_LOG.write_text(json.dumps(conflicts, indent=2))


def _split_train_test_validation(master: pd.DataFrame, seed: int = SPLIT_SEED) -> tuple:
    """
    Deterministic 70/20/10 train/test/validation split: shuffle once with
    a fixed seed, then slice by row count. Not stratified -- Cleaning
    Agent doesn't know what the Modeling Agent's eventual target column
    will be, so a plain random split is the only one that doesn't assume
    anything about the data's shape or task.

    Slicing (rather than three independent random draws) guarantees every
    row ends up in exactly one split and the three split sizes always sum
    to the master's total row count, regardless of rounding.
    """
    shuffled = master.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = round(n * TRAIN_FRAC)
    test_end = train_end + round(n * TEST_FRAC)
    train_df = shuffled.iloc[:train_end]
    test_df = shuffled.iloc[train_end:test_end]
    validation_df = shuffled.iloc[test_end:]
    return train_df, test_df, validation_df


def _write_split_csvs(master: pd.DataFrame) -> dict:
    """Split the merged master CSV and write train/test/validation CSVs
    alongside it. Returns their paths and row counts for the pipeline
    result/state, so the future Modeling Agent (and this run's report) can
    find them without recomputing the split."""
    train_df, test_df, validation_df = _split_train_test_validation(master)

    train_path = PROCESSED_DATA_DIR / TRAIN_CSV_NAME
    test_path = PROCESSED_DATA_DIR / TEST_CSV_NAME
    validation_path = PROCESSED_DATA_DIR / VALIDATION_CSV_NAME
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    print(f"[Cleaning Agent] Split into train/test/validation: "
          f"{len(train_df)}/{len(test_df)}/{len(validation_df)} rows "
          f"(seed={SPLIT_SEED}) -> {train_path}, {test_path}, {validation_path}.")

    return {
        "train_csv_path": str(train_path),
        "test_csv_path": str(test_path),
        "validation_csv_path": str(validation_path),
        "split_rows": {"train": len(train_df), "test": len(test_df), "validation": len(validation_df)},
        "split_seed": SPLIT_SEED,
    }


def _write_mappings_log(mappings: list) -> None:
    MAPPINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    MAPPINGS_LOG.write_text(json.dumps(mappings, indent=2, default=str))


def _load_datasets(selected_datasets: list, target_terms: list, conflicts: list) -> list:
    """Read, budget-cap and clean every dataset. Returns one dict per
    readable dataset: label, link, frame, label column, sampling info."""
    loaded = []
    for candidate in selected_datasets:
        label = candidate.get("display_name") or candidate.get("name", "")
        link = candidate.get("name", "")
        path = candidate.get("local_csv_path", "")

        if not path:
            _log_conflict(conflicts, label, link,
                          "no local CSV path recorded (Data Agent did not download this dataset)")
            continue

        try:
            df, sampling = read_csv_capped(path, MAX_ROWS_PER_DATASET, target_terms)
        except Exception as e:
            _log_conflict(conflicts, label, link, f"could not read CSV: {e}")
            continue

        df = _clean_dataframe(df, link)
        if df.empty or df.shape[1] <= 1:  # only the source column left
            _log_conflict(conflicts, label, link,
                          "file contained no usable rows/columns after cleaning")
            continue
        if sampling["sampled"]:
            print(f"[Cleaning Agent] '{label}': sampled {sampling['sampled_rows']} of {sampling['original_rows']} "
                  f"rows (row budget {MAX_ROWS_PER_DATASET}, stratified on "
                  f"'{sampling['label_column'] or 'none'}', seed {sampling['seed']}).")

        loaded.append({"label": label, "link": link, "df": df, "sampling": sampling,
                       "label_column": find_label_column(df, target_terms)})
    return loaded


def _merge_by_name(loaded: list, conflicts: list) -> list:
    """Without target terms: merge on shared (normalized) column names; a
    dataset sharing no column with those already accepted is excluded."""
    accepted, accepted_columns = [], set()
    for item in loaded:
        df_columns = set(item["df"].columns) - {SOURCE_COLUMN}
        if accepted and not (df_columns & accepted_columns):
            _log_conflict(conflicts, item["label"], item["link"],
                          "schema incompatible -- shares no columns with the datasets already "
                          f"merged (its columns: {sorted(df_columns)})")
            continue
        accepted.append(item)
        accepted_columns |= df_columns
        print(f"[Cleaning Agent] Merged '{item['label']}' ({len(item['df'])} rows, {len(df_columns)} columns) "
              f"from {item['link']}.")
    return accepted


def _merge_by_label(loaded: list, conflicts: list, mappings: list) -> tuple[list, str]:
    """Label-aware merge (see module docstring). Returns (accepted items
    with aligned frames, the anchor's label column name)."""
    for item in loaded:
        values = item["df"][item["label_column"]].dropna().unique() if item["label_column"] else []
        item["label_map"] = binary_label_map(values) if item["label_column"] else None

    anchor = next((i for i in loaded if i["label_map"]), None)
    if anchor is None:
        for item in loaded:
            reason = ("no column holds the prediction target" if not item["label_column"] else
                      f"label column '{item['label_column']}' has values that are not recognizably "
                      "positive/negative")
            _log_conflict(conflicts, item["label"], item["link"], reason)
        return [], ""

    label_column = anchor["label_column"]
    anchor["df"][label_column] = anchor["df"][label_column].map(anchor["label_map"])
    accepted = [anchor]
    print(f"[Cleaning Agent] Anchor: '{anchor['label']}' (label column '{label_column}', "
          f"{len(anchor['df'])} rows).")

    for index, item in enumerate(loaded):
        if item is anchor:
            continue
        if not item["label_column"]:
            _log_conflict(conflicts, item["label"], item["link"],
                          f"no column holds the prediction target (anchor label: '{label_column}')")
            continue
        if not item["label_map"]:
            _log_conflict(conflicts, item["label"], item["link"],
                          f"label column '{item['label_column']}' has values that are not recognizably "
                          "positive/negative -- not merged rather than guessing which value is positive")
            continue

        df = item["df"].copy()
        df[item["label_column"]] = df[item["label_column"]].map(item["label_map"])
        df = df.rename(columns={item["label_column"]: label_column})
        aligned, records = column_mapping.align(
            anchor["df"], anchor["link"], df, item["link"],
            skip={label_column, SOURCE_COLUMN}, dataset_tag=f"ds{index + 1}",
        )
        accepted_records = [r for r in records if r["accepted"]]
        mappings.append({"dataset": item["label"], "link": item["link"], "anchor": anchor["link"],
                         "label": {"source": item["label_column"], "target": label_column,
                                   "value_map": {str(k): v for k, v in item["label_map"].items()}},
                         "columns": records})
        mapped = ", ".join(f"{r['source']}->{r['target']}" for r in accepted_records) or "none"
        print(f"[Cleaning Agent] Merged '{item['label']}' ({len(aligned)} rows): label "
              f"'{item['label_column']}' -> '{label_column}', {len(accepted_records)} verified column "
              f"mapping(s) ({mapped}), {len(records) - len(accepted_records)} rejected.")
        item["df"] = aligned
        accepted.append(item)
    return accepted, label_column


def run(selected_datasets: list, target_terms: list = None) -> dict:
    """
    Merge every dataset in selected_datasets (Data Agent's output, each
    already downloaded to a local CSV) into one cleaned master CSV. Never
    raises -- any dataset that can't be merged is logged as a conflict and
    excluded, not silently dropped and not fatal to the others.

    target_terms (the Intake Agent's prediction-target keywords) enables
    the label-aware, semantically mapped merge; without them, datasets
    merge on shared column names.
    """
    print(f"\n[Cleaning Agent] Cleaning and merging {len(selected_datasets)} dataset(s)...")

    conflicts = []
    mappings = []
    loaded = _load_datasets(selected_datasets, target_terms, conflicts)

    label_column = ""
    if target_terms and any(i["label_column"] for i in loaded):
        accepted, label_column = _merge_by_label(loaded, conflicts, mappings)
    else:
        if target_terms and loaded:
            print(f"[Cleaning Agent] No dataset has a column matching the target {target_terms}; "
                  "merging on shared column names instead.")
        accepted = _merge_by_name(loaded, conflicts)

    _write_conflicts_log(conflicts)
    _write_mappings_log(mappings)
    sampling = [{"dataset": i["label"], "link": i["link"], **i["sampling"]} for i in loaded]

    if not accepted:
        print("[Cleaning Agent] Done. Status=NOT_FOUND, no dataset could be merged.")
        return {
            "status": "NOT_FOUND",
            "master_csv_path": "",
            "datasets_attempted": len(selected_datasets),
            "datasets_merged": 0,
            "rows_total": 0,
            "columns_total": 0,
            "target_rows": TARGET_ROWS,
            "target_met": False,
            "conflicts": conflicts,
            "label_column": "",
            "column_mappings": mappings,
            "sampling": sampling,
            "sampled": any(i["sampled"] for i in sampling),
            "train_csv_path": "",
            "test_csv_path": "",
            "validation_csv_path": "",
            "split_rows": {"train": 0, "test": 0, "validation": 0},
            "split_seed": SPLIT_SEED,
            "notes": "No dataset could be merged into a master CSV.",
        }

    master = pd.concat([i["df"] for i in accepted], axis=0, join="outer", ignore_index=True, sort=False)
    before_dedupe = len(master)
    master = master.drop_duplicates()
    deduped = before_dedupe - len(master)

    master, master_cap = stratified_cap(master, MAX_MASTER_ROWS, label_column or None)
    if master_cap["sampled"]:
        print(f"[Cleaning Agent] Master CSV sampled to {master_cap['sampled_rows']} of "
              f"{master_cap['original_rows']} rows (row budget {MAX_MASTER_ROWS}).")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PROCESSED_DATA_DIR / MASTER_CSV_NAME
    master.to_csv(master_path, index=False)

    target_met = len(master) >= TARGET_ROWS
    status = "OK" if (not conflicts and target_met) else "DEGRADED"
    sampled = master_cap["sampled"] or any(i["sampled"] for i in sampling)

    notes_parts = []
    if deduped:
        notes_parts.append(f"Removed {deduped} duplicate row(s) across the merged set.")
    if sampled:
        notes_parts.append(f"Row budget applied (at most {MAX_ROWS_PER_DATASET} rows per dataset, "
                           f"{MAX_MASTER_ROWS} in total; stratified, seed fixed) -- see 'sampling'.")
    if not target_met:
        notes_parts.append(f"Merged total is {len(master)} rows, below the {TARGET_ROWS}-row target.")
    notes = " ".join(notes_parts)

    print(f"[Cleaning Agent] Done. Status={status}, merged={len(accepted)}/{len(selected_datasets)}, "
          f"rows={len(master)}/{TARGET_ROWS} (target met: {target_met}), "
          f"columns={master.shape[1]}, label='{label_column or 'unknown'}', wrote {master_path}.")

    split_info = _write_split_csvs(master)

    return {
        "status": status,
        "master_csv_path": str(master_path),
        "datasets_attempted": len(selected_datasets),
        "datasets_merged": len(accepted),
        "rows_total": len(master),
        "columns_total": master.shape[1],
        "target_rows": TARGET_ROWS,
        "target_met": target_met,
        "conflicts": conflicts,
        "label_column": label_column,
        "column_mappings": mappings,
        "sampling": sampling,
        "master_sampling": master_cap,
        "sampled": sampled,
        "notes": notes,
        **split_info,
    }
