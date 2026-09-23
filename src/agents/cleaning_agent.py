"""
Cleaning Agent.

Takes the datasets Data Agent already downloaded and verified
(`dataset_result["selected_datasets"]`, each guaranteed to have a real
local CSV at `local_csv_path` -- see data_agent.py) and merges them into a
single cleaned master CSV for the Parent Paper's replication.

Does no downloading of its own -- if a candidate has no local CSV, that is
itself logged as a conflict rather than assumed away.

Per dataset: read the CSV, normalize column names (whitespace/case/
separator only -- never fuzzy or semantic renaming, so two genuinely
different columns are never silently treated as the same one), drop
fully-empty rows/columns and exact-duplicate rows, and tag every row with
its source dataset for provenance. A dataset is only excluded -- logged as
a conflict, not silently dropped -- when it shares literally zero columns
with the datasets already accepted, since there would be no basis at all
to relate its rows to the rest of the pool.

Accepted datasets are combined with a union-of-columns concat (missing
values become blank), then deduplicated once more across the whole merged
set. Every reason a dataset couldn't be merged -- whether it had no CSV,
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
"""
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
CONFLICTS_LOG = REPO_ROOT / "logs" / "cleaning_conflicts.json"
MASTER_CSV_NAME = "master_dataset.csv"

SOURCE_COLUMN = "_source_dataset"
TARGET_ROWS = 10000


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


def run(selected_datasets: list) -> dict:
    """
    Merge every dataset in selected_datasets (Data Agent's output, each
    already downloaded to a local CSV) into one cleaned master CSV. Never
    raises -- any dataset that can't be merged is logged as a conflict and
    excluded, not silently dropped and not fatal to the others.
    """
    print(f"\n[Cleaning Agent] Cleaning and merging {len(selected_datasets)} dataset(s)...")

    conflicts = []
    accepted_frames = []
    accepted_columns = set()

    for candidate in selected_datasets:
        label = candidate.get("display_name") or candidate.get("name", "")
        link = candidate.get("name", "")
        path = candidate.get("local_csv_path", "")

        if not path:
            _log_conflict(conflicts, label, link,
                          "no local CSV path recorded (Data Agent did not download this dataset)")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            _log_conflict(conflicts, label, link, f"could not read CSV: {e}")
            continue

        df = _clean_dataframe(df, link)

        if df.empty or df.shape[1] <= 1:  # only the source column left
            _log_conflict(conflicts, label, link,
                          "file contained no usable rows/columns after cleaning")
            continue

        df_columns = set(df.columns) - {SOURCE_COLUMN}
        if accepted_frames and not (df_columns & accepted_columns):
            _log_conflict(conflicts, label, link,
                          "schema incompatible -- shares no columns with the datasets already "
                          f"merged (its columns: {sorted(df_columns)})")
            continue

        accepted_frames.append(df)
        accepted_columns |= df_columns
        print(f"[Cleaning Agent] Merged '{label}' ({len(df)} rows, {len(df_columns)} columns) from {link}.")

    _write_conflicts_log(conflicts)

    if not accepted_frames:
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
            "notes": "No dataset could be merged into a master CSV.",
        }

    master = pd.concat(accepted_frames, axis=0, join="outer", ignore_index=True, sort=False)
    before_dedupe = len(master)
    master = master.drop_duplicates()
    deduped = before_dedupe - len(master)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PROCESSED_DATA_DIR / MASTER_CSV_NAME
    master.to_csv(master_path, index=False)

    target_met = len(master) >= TARGET_ROWS
    status = "OK" if (not conflicts and target_met) else "DEGRADED"

    notes_parts = []
    if deduped:
        notes_parts.append(f"Removed {deduped} duplicate row(s) across the merged set.")
    if not target_met:
        notes_parts.append(f"Merged total is {len(master)} rows, below the {TARGET_ROWS}-row target.")
    notes = " ".join(notes_parts)

    print(f"[Cleaning Agent] Done. Status={status}, merged={len(accepted_frames)}/{len(selected_datasets)}, "
          f"rows={len(master)}/{TARGET_ROWS} (target met: {target_met}), "
          f"columns={master.shape[1]}, wrote {master_path}.")

    return {
        "status": status,
        "master_csv_path": str(master_path),
        "datasets_attempted": len(selected_datasets),
        "datasets_merged": len(accepted_frames),
        "rows_total": len(master),
        "columns_total": master.shape[1],
        "target_rows": TARGET_ROWS,
        "target_met": target_met,
        "conflicts": conflicts,
        "notes": notes,
    }
