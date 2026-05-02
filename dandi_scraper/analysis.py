"""Raw analysis primitives: dandiset listing, per-dandiset feature extraction,
metadata fetch, and feature-space utilities (QC, scaling).

All functions read defaults from `config.CONFIG`/module-level constants but
accept overrides as keyword arguments.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dandi.dandiapi import DandiAPIClient

from pyAPisolation.database.build_database import run_analysis

from . import config
from ._metadata_parser import dandi_meta_parser

logger = logging.getLogger("dandi_scraper.analysis")


# ---------------------------------------------------------------------------
# Dandiset listing & metadata
# ---------------------------------------------------------------------------

def build_dandiset_df() -> pd.DataFrame:
    """Pull the full list of dandisets from DANDI and shape it into a DataFrame."""
    client = DandiAPIClient()
    dandisets = list(client.get_dandisets())

    def is_nwb(metadata):
        return any(
            x["identifier"] == "RRID:SCR_015242"
            for x in metadata["assetsSummary"].get("dataStandard", {})
        )

    data: Dict[str, list] = defaultdict(list)
    for dandiset in dandisets:
        identifier = dandiset.identifier
        metadata = dandiset.get_raw_metadata()
        if not is_nwb(metadata) or not dandiset.draft_version.size:
            continue
        data["identifier"].append(identifier)
        data["created"].append(dandiset.created)
        data["size"].append(dandiset.draft_version.size)
        if "species" in metadata["assetsSummary"] and len(metadata["assetsSummary"]["species"]):
            data["species"].append(metadata["assetsSummary"]["species"][0]["name"])
        else:
            data["species"].append(np.nan)

        for modality, ndtypes in config.NEURODATA_TYPE_MAP.items():
            data[modality].append(
                any(x in ndtypes for x in metadata["assetsSummary"]["variableMeasured"])
            )

        if "numberOfSubjects" in metadata["assetsSummary"]:
            data["numberOfSubjects"].append(metadata["assetsSummary"]["numberOfSubjects"])
        else:
            data["numberOfSubjects"].append(np.nan)

        data["keywords"].append([x.lower() for x in metadata.get("keywords", [])])

    df = pd.DataFrame.from_dict(data)
    for key, val in config.SPECIES_REPLACEMENT.items():
        df["species"] = df["species"].replace(key, val)
    return df


def filter_dandiset_df(row, species=None, modality=None, keywords=None, method="or"):
    keywords = keywords or []
    flags = []
    if species is not None:
        flags.append(row["species"] == species)
    if modality is not None:
        flags.append(row[modality] == True)  # noqa: E712
    if len(keywords) > 0:
        flags.append(np.any(np.ravel([[x in j for j in row["keywords"]] for x in keywords])))
    if method == "or":
        return any(flags)
    if method == "and":
        return all(flags)
    raise ValueError("method must be 'or' or 'and'")


def get_dandi_metadata(code: str):
    """Fetch raw + LLM-parsed metadata for a single dandiset."""
    client = DandiAPIClient()
    client.get_dandiset(code)  # validate the code resolves
    metadata_parser = dandi_meta_parser(code)
    return metadata_parser.asset_data


# ---------------------------------------------------------------------------
# Per-dandiset analysis
# ---------------------------------------------------------------------------

def analyze_dandiset(code: str, cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Run the per-cell pyAPisolation pipeline for one dandiset folder."""
    cache_dir = cache_dir or config.CONFIG.cache_dir
    df_dandiset = run_analysis(
        os.path.join(cache_dir, code),
        outfile=os.path.join(cache_dir, f"{code}.csv"),
    )
    if "error_stage" in df_dandiset.columns:
        fail_mask = df_dandiset["error_stage"].notna() & (df_dandiset["error_stage"] != "")
        n_fail = int(fail_mask.sum())
        n_total = len(df_dandiset)
        if n_fail:
            by_stage = df_dandiset.loc[fail_mask, "error_stage"].value_counts().to_dict()
            by_class = (
                df_dandiset.loc[fail_mask, "error_class"].value_counts().to_dict()
                if "error_class" in df_dandiset.columns else {}
            )
            logger.warning(
                "dandiset %s: %d/%d cells failed; stages=%s classes=%s",
                code, n_fail, n_total, by_stage, by_class,
            )
        else:
            logger.info("dandiset %s: all %d cells succeeded", code, n_total)
    return df_dandiset


# ---------------------------------------------------------------------------
# Feature-space utilities (QC, scaling)
# ---------------------------------------------------------------------------

def quick_qc(
    df: pd.DataFrame,
    qc_features: Optional[Dict[str, Tuple[float, float]]] = None,
    return_drops: bool = False,
):
    """Apply feature-range QC to ``df``. Logs and optionally returns the
    per-cell drop reasons so upstream callers can persist them to a per-cell
    status csv."""
    if qc_features is None:
        qc_features = config.QC_FEATURE_BOUNDS
    drop_rows: List[dict] = []
    for feature, (min_val, max_val) in qc_features.items():
        if feature in df.columns:
            fail_mask = (df[feature] < min_val) | (df[feature] > max_val)
            _failing = df[fail_mask]
            num_failing = len(_failing)
            logger.warning(
                "quick_qc: %d cells failed %s (bounds [%g, %g]); examples=%s",
                num_failing, feature, min_val, max_val,
                _failing[feature].head().to_dict(),
            )
            if num_failing and return_drops:
                for idx, val in _failing[feature].items():
                    drop_rows.append({
                        "specimen_id": idx,
                        "feature": feature,
                        "value": val,
                        "reason": f"out_of_range[{min_val},{max_val}]",
                    })
            df = df[~fail_mask]
    if return_drops:
        return df, pd.DataFrame(drop_rows)
    return df


def scale_features(df: pd.DataFrame, features: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if features is None:
        features = config.SCALE_FEATURES
    for feature, method in features.items():
        if feature in df.columns:
            if method == "log":
                df[feature] = np.log10(df[feature])
            elif method == "log-1000":
                df[feature] = np.log10(df[feature] * 1000)
            elif method == "zscore":
                df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
            else:
                raise ValueError("method must be 'log', 'log-1000', or 'zscore'")
    return df
