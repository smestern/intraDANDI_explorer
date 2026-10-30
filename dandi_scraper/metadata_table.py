"""Per-dandiset metadata table (`dandiset_metadata.csv`) and normalisation of
brain-region / species values.

The CSV is the source of truth for `brain_region` and `brain_subregion`. Rows
are hand-reviewed (`reviewed=True`); dandisets missing from it get an
LLM-suggested row appended with `reviewed=False` for later review.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Tuple

import pandas as pd

from . import config

logger = logging.getLogger("dandi_scraper.metadata_table")

TABLE_PATH = os.path.join(os.path.dirname(__file__), "dandiset_metadata.csv")
COLUMNS = [
    "dandiset_id", "title", "brain_region", "brain_subregion",
    "source", "confidence", "reviewed", "notes",
]
_REGION_LOOKUP = {r.lower(): r for r in config.BRAIN_REGIONS}


def _as_int_id(dandiset_id) -> int:
    return int(float(dandiset_id))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_region(raw) -> Tuple[str, str]:
    """Map a free-text region onto ``(brain_region, brain_subregion)``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "Unknown", ""
    key = str(raw).strip().lower()
    if key in _REGION_LOOKUP:
        return _REGION_LOOKUP[key], ""
    if key in config.BRAIN_REGION_SYNONYMS:
        return config.BRAIN_REGION_SYNONYMS[key]
    logger.warning("unmapped brain region %r; add it to BRAIN_REGION_SYNONYMS", raw)
    return "Unknown", str(raw).strip()


def normalize_species(name, identifier: Optional[str] = None):
    """Display name for a species, keyed on NCBITaxon id when available."""
    if identifier:
        taxon = str(identifier).rsplit("_", 1)[-1]
        if taxon in config.SPECIES_BY_TAXON:
            return config.SPECIES_BY_TAXON[taxon]
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    return config.SPECIES_REPLACEMENT.get(name, name)


# ---------------------------------------------------------------------------
# Table I/O
# ---------------------------------------------------------------------------

def load_table(path: str = TABLE_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS).set_index("dandiset_id")
    text_cols = ["title", "brain_subregion", "source", "confidence", "notes"]
    df = pd.read_csv(path, dtype={c: str for c in text_cols})
    df["dandiset_id"] = df["dandiset_id"].apply(_as_int_id)
    df["reviewed"] = df["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])
    df[text_cols] = df[text_cols].fillna("")
    bad = ~df["brain_region"].isin(config.BRAIN_REGIONS)
    if bad.any():
        raise ValueError(
            f"{path}: brain_region not in config.BRAIN_REGIONS for dandisets "
            f"{df.loc[bad, 'dandiset_id'].tolist()}: {df.loc[bad, 'brain_region'].tolist()}"
        )
    return df.set_index("dandiset_id")


def save_table(df: pd.DataFrame, path: str = TABLE_PATH) -> None:
    df.sort_index().reset_index()[COLUMNS].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def lookup_or_add(
    dandiset_id,
    title: str,
    suggest: Callable[[], Optional[dict]],
    path: str = TABLE_PATH,
) -> Tuple[str, str]:
    """Return ``(brain_region, brain_subregion)`` for a dandiset.

    Uses the table row if present. Otherwise (or if the row is an unreviewed
    placeholder from a run without the LLM) calls ``suggest()`` and stores its
    result with ``reviewed=False``. ``suggest`` returns a dict with
    brain_region, brain_subregion, confidence and notes, or None on failure.
    """
    did = _as_int_id(dandiset_id)
    table = load_table(path)
    row = table.loc[did] if did in table.index else None
    if row is not None and (row["reviewed"] or row["source"] != "default"):
        if not row["reviewed"]:
            logger.info("dandiset %s: using unreviewed %s row from %s", did, row["source"], path)
        return row["brain_region"], row["brain_subregion"]

    suggestion = suggest()
    if suggestion is not None:
        new_row = dict(source="llm", **suggestion)
    elif row is not None:
        return row["brain_region"], row["brain_subregion"]
    else:
        new_row = dict(source="default", brain_region="Unknown", brain_subregion="",
                       confidence="", notes="no LLM suggestion available")
    table.loc[did] = dict(title=title, reviewed=False, **new_row)
    save_table(table, path)
    logger.warning(
        "dandiset %s: added %s row (%s / %s) to %s - please review",
        did, new_row["source"], new_row["brain_region"], new_row["brain_subregion"], path,
    )
    return new_row["brain_region"], new_row["brain_subregion"]


def apply_table(
    meta_df: pd.DataFrame,
    id_col: str = "dandiset_id",
    species_cols=("species", "species_meta"),
    path: str = TABLE_PATH,
) -> pd.DataFrame:
    """Overwrite brain_region/brain_subregion from the table and normalise
    species, so cached metadata picks up table edits without refetching."""
    meta_df = meta_df.copy()
    table = load_table(path)
    if "brain_region" not in meta_df:
        meta_df["brain_region"] = None
    ids = meta_df[id_col].apply(lambda x: _as_int_id(x) if pd.notna(x) else None)
    in_table = ids.isin(table.index)

    legacy = meta_df.loc[~in_table, "brain_region"].apply(normalize_region)
    for did in sorted(ids[~in_table].dropna().unique()):
        logger.warning("dandiset %d not in %s; normalising its cached brain_region", did, path)
    meta_df["brain_subregion"] = ""
    meta_df.loc[~in_table, "brain_region"] = legacy.str[0]
    meta_df.loc[~in_table, "brain_subregion"] = legacy.str[1]
    meta_df.loc[in_table, "brain_region"] = ids[in_table].map(table["brain_region"])
    meta_df.loc[in_table, "brain_subregion"] = ids[in_table].map(table["brain_subregion"])

    for col in species_cols:
        if col in meta_df:
            meta_df[col] = meta_df[col].apply(normalize_species)
    return meta_df
