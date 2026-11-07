"""
Semantic column mapping for the Cleaning Agent: the AI proposes, the code
verifies.

Two datasets often name the same feature differently ("amt" vs "amount",
"income" vs "monthly_income"). Matching by name alone excluded real,
mergeable datasets in a live run. But a name-similar pair can also differ
in unit -- monthly vs annual income -- and merging those without a
conversion would silently corrupt the column. So:

  1. One LLM call proposes, for each column of the incoming dataset, a
     target column in the anchor dataset and a relation: "same",
     "same_concept_different_unit" (with an explicit multiply factor), or
     nothing.
  2. Every proposal -- and every pair that already shares a name -- is
     verified against the actual data, deterministically: compatible
     kinds, numeric medians within MEDIAN_RATIO_LIMIT after any
     conversion, categorical value sets overlapping. A failed check keeps
     the columns separate. Uncertainty never merges.
  3. The label is mapped deterministically (both sides identified by the
     project's target terms) and its values normalized to 0/1.

Proposals are cached per dataset pair (logs/column_mapping_cache.json) so
reruns reuse the same AI decision; verification always reruns. Every
accepted and rejected mapping, with its evidence, is returned for the
Cleaning Agent to log (logs/column_mappings.json).
"""
import json
import re
from pathlib import Path

import pandas as pd

from src import llm

REPO_ROOT = Path(__file__).parent.parent.parent
MAPPING_CACHE = REPO_ROOT / "logs" / "column_mapping_cache.json"

MEDIAN_RATIO_LIMIT = 3.0
CATEGORY_OVERLAP_MIN = 0.2
NUMERIC_PARSE_MIN = 0.95
TOP_CATEGORIES = 50
PROFILE_EXAMPLES = 4

RELATION_SAME = "same"
RELATION_UNIT = "same_concept_different_unit"

_NUMERIC_JUNK = re.compile(r"[\s$€£¥,%]")


def to_numeric(series: pd.Series) -> pd.Series | None:
    """The series as numbers, parsing currency/percent/thousands formatting
    ("$1,234.50" -> 1234.5), or None when it isn't numeric (fewer than
    NUMERIC_PARSE_MIN of its non-null values parse)."""
    if pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series
    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = pd.to_numeric(non_null.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True), errors="coerce")
    if parsed.notna().mean() < NUMERIC_PARSE_MIN:
        return None
    return pd.to_numeric(series.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True), errors="coerce")


def _profile_column(series: pd.Series) -> str:
    numeric = to_numeric(series)
    if numeric is not None and numeric.notna().any():
        return (f"numeric, min={numeric.min():.4g}, median={numeric.median():.4g}, max={numeric.max():.4g}, "
                f"examples={list(series.dropna().astype(str).unique()[:PROFILE_EXAMPLES])}")
    values = series.dropna().astype(str)
    return (f"text/categorical, {values.nunique()} distinct, "
            f"examples={list(values.value_counts().index[:PROFILE_EXAMPLES])}")


def profile(df: pd.DataFrame, skip: set) -> str:
    return "\n".join(f"  - {c}: {_profile_column(df[c])}" for c in df.columns if c not in skip)


def _load_cache() -> dict:
    try:
        return json.loads(MAPPING_CACHE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MAPPING_CACHE.write_text(json.dumps(cache, indent=2))


def propose_mappings(anchor_df: pd.DataFrame, anchor_link: str, df: pd.DataFrame, df_link: str,
                     skip: set) -> list:
    """The AI's proposed mappings from df's columns onto anchor_df's.
    Cached per (anchor, dataset) pair; [] when the call fails (the
    dataset can still merge on its label and same-named columns)."""
    cache_key = f"{anchor_link}||{df_link}"
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"""You are aligning the columns of two tabular datasets so they can be merged for one machine learning task.

ANCHOR dataset (the schema to map onto):
{profile(anchor_df, skip)}

INCOMING dataset:
{profile(df, skip)}

For each INCOMING column that records the same real-world quantity or attribute as an ANCHOR column, propose a mapping. Examples: "amt" -> "amount" (same); "monthly_income" -> "income" where income is annual (same concept, different unit, factor 12).

Rules:
- Only map columns that genuinely mean the same thing. When unsure, leave the column out.
- "relation" is "{RELATION_SAME}" when values are directly comparable, or "{RELATION_UNIT}" when the incoming values must be multiplied by a constant to match the anchor's unit.
- "factor" is that multiplier (a positive number) for "{RELATION_UNIT}", else null.
- Each anchor column may be the target of at most one mapping.

Return ONLY a JSON array:
[{{"source": "<incoming column>", "target": "<anchor column>", "relation": "{RELATION_SAME}" | "{RELATION_UNIT}", "factor": <number or null>, "reason": "<one short sentence>"}}]
Return [] if no column maps."""

    try:
        content = llm.chat(prompt, timeout=300, agent="cleaning_agent")
    except Exception as e:
        print(f"[Cleaning Agent] Column-mapping proposal call failed: {e}")
        return []
    proposals = llm.parse_json(content, default=[])
    proposals = [p for p in proposals if isinstance(p, dict)] if isinstance(proposals, list) else []
    cache[cache_key] = proposals
    _save_cache(cache)
    return proposals


def verify(source: pd.Series, target: pd.Series, relation: str, factor) -> tuple[bool, str, pd.Series | None]:
    """Deterministic check of one proposed mapping against the data.
    Returns (accepted, evidence, converted source values or None)."""
    source_num, target_num = to_numeric(source), to_numeric(target)

    if source_num is not None and target_num is not None:
        if relation == RELATION_UNIT:
            try:
                factor = float(factor)
            except (TypeError, ValueError):
                return False, "unit conversion proposed without a numeric factor", None
            if factor <= 0:
                return False, f"invalid factor {factor}", None
            converted = source_num * factor
        elif relation == RELATION_SAME:
            converted = source_num
        else:
            return False, f"unknown relation '{relation}'", None
        m_s, m_t = converted.median(), target_num.median()
        if pd.isna(m_s) or pd.isna(m_t):
            return False, "no numeric values to compare", None
        if abs(m_s) < 1e-12 and abs(m_t) < 1e-12:
            return True, "both medians are 0", converted
        if m_s * m_t <= 0:
            return False, f"medians differ in sign or one is 0 ({m_s:.4g} vs {m_t:.4g})", None
        ratio = max(abs(m_s), abs(m_t)) / min(abs(m_s), abs(m_t))
        evidence = f"medians {m_s:.4g} vs {m_t:.4g} (ratio {ratio:.2f}, limit {MEDIAN_RATIO_LIMIT})"
        return ratio <= MEDIAN_RATIO_LIMIT, evidence, converted if ratio <= MEDIAN_RATIO_LIMIT else None

    if source_num is not None or target_num is not None:
        return False, "one column is numeric and the other is not", None
    if relation != RELATION_SAME:
        return False, "a unit conversion applies only to numeric columns", None

    a = set(source.dropna().astype(str).str.strip().str.lower().value_counts().index[:TOP_CATEGORIES])
    b = set(target.dropna().astype(str).str.strip().str.lower().value_counts().index[:TOP_CATEGORIES])
    if not a or not b:
        return False, "no values to compare", None
    overlap = len(a & b) / min(len(a), len(b))
    evidence = f"top-value overlap {overlap:.2f} (minimum {CATEGORY_OVERLAP_MIN})"
    return overlap >= CATEGORY_OVERLAP_MIN, evidence, source if overlap >= CATEGORY_OVERLAP_MIN else None


def align(anchor_df: pd.DataFrame, anchor_link: str, df: pd.DataFrame, df_link: str,
          skip: set, dataset_tag: str) -> tuple[pd.DataFrame, list]:
    """Rename/convert df's columns onto anchor_df's schema using verified
    mappings only. Columns in skip (label, provenance) are handled by the
    caller. Returns (aligned frame, mapping records). An unverified column
    that shares a name with an anchor column is renamed
    "<name>__<dataset_tag>" so it is not merged by accident."""
    proposals = propose_mappings(anchor_df, anchor_link, df, df_link, skip)
    by_source = {}
    for p in proposals:
        source, target = p.get("source"), p.get("target")
        if source in df.columns and target in anchor_df.columns and source not in skip and target not in skip:
            by_source.setdefault(source, {**p, "origin": "ai_proposal"})
    for column in df.columns:
        if column in anchor_df.columns and column not in skip and column not in by_source:
            by_source[column] = {"source": column, "target": column, "relation": RELATION_SAME,
                                 "factor": None, "reason": "same column name", "origin": "same_name"}

    records, renamed, used_targets = [], {}, set()
    aligned = df.copy()
    for source, proposal in by_source.items():
        target = proposal["target"]
        if target in used_targets:
            ok, evidence, converted = False, f"anchor column '{target}' already mapped", None
        else:
            ok, evidence, converted = verify(df[source], anchor_df[target], proposal.get("relation"),
                                             proposal.get("factor"))
        records.append({"source": source, "target": target, "relation": proposal.get("relation"),
                        "factor": proposal.get("factor"), "origin": proposal["origin"],
                        "reason": proposal.get("reason", ""), "accepted": ok, "evidence": evidence})
        if ok:
            used_targets.add(target)
            aligned[source] = converted
            renamed[source] = target

    # One rename over the original names: an unmapped column that shares a
    # name with an anchor column (including a target another column was
    # just mapped onto) gets a dataset suffix instead of merging.
    collisions = {c: f"{c}__{dataset_tag}" for c in df.columns
                  if c not in renamed and c not in skip and (c in anchor_df.columns or c in used_targets)}
    return aligned.rename(columns={**collisions, **renamed}), records
