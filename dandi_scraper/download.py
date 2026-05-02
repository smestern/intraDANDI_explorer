"""Dandiset download + cache management + trace plotting orchestration."""
from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import Iterable, Optional

import pandas as pd
from dandi.dandiapi import DandiAPIClient
from dandi.download import download as dandi_download
import dandi.download as dandi_download_utils

from pyAPisolation.database.build_database import build_dataset_traces

from . import config
from .analysis import analyze_dandiset, build_dandiset_df, filter_dandiset_df
from .logging_utils import configure_logging

logger = logging.getLogger("dandi_scraper.download")


def download_dandiset(code: str, save_dir: Optional[str] = None, overwrite: bool = False) -> None:
    """Download a single dandiset into ``save_dir`` (defaults to cache dir)."""
    client = DandiAPIClient()
    dandiset = client.get_dandiset(code)
    if save_dir is None:
        save_dir = config.CONFIG.cache_dir
    if os.path.exists(os.path.join(save_dir, code)) and not overwrite:
        return
    dandi_download(
        dandiset.api_url,
        save_dir,
        existing=dandi_download_utils.DownloadExisting.OVERWRITE_DIFFERENT,
    )


# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------

def run_analyze_dandiset(
    cache_dir: Optional[str] = None,
    skip: Optional[Iterable[str]] = None,
    include: Optional[Iterable[str]] = None,
) -> None:
    """Discover, download and analyze every relevant icephys dandiset.

    Skips any dandisets already present as ``<cache_dir>/<code>.csv`` and
    augments the discovered list with explicit `include` codes.
    """
    configure_logging()
    cache_dir = cache_dir or config.CONFIG.cache_dir
    skip = list(skip) if skip is not None else config.DANDISETS_TO_SKIP
    include = list(include) if include is not None else config.DANDISETS_TO_INCLUDE

    dandi_df = build_dandiset_df()
    filtered_df = dandi_df[dandi_df.apply(
        lambda x: filter_dandiset_df(
            x, modality="icephys", keywords=["intracellular", "patch"], method="or",
        ),
        axis=1,
    )]
    logger.info("found %d dandisets to analyze", len(filtered_df))

    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))
    csv_codes = [os.path.splitext(os.path.basename(x))[0] for x in csv_files]
    filtered_df = filtered_df[~filtered_df["identifier"].isin(csv_codes)]
    for code in include:
        if code not in filtered_df["identifier"].values:
            filtered_df = pd.concat([filtered_df, dandi_df[dandi_df["identifier"] == code]])

    for row in list(filtered_df.iterrows())[::-1]:
        code = row[1]["identifier"]
        logger.info("downloading dandiset %s", code)
        if code in skip:
            logger.info("skipping dandiset %s (on skip list)", code)
            continue
        try:
            download_dandiset(code, save_dir=cache_dir, overwrite=True)
        except Exception:
            logger.exception("dandiset %s: download failed", code)
            continue
        try:
            df_dandiset = analyze_dandiset(code, cache_dir=cache_dir)
        except Exception:
            logger.exception("dandiset %s: analyze_dandiset raised", code)
            continue
        df_dandiset["dandiset"] = code
        df_dandiset["created"] = row[1]["created"]
        df_dandiset["species"] = row[1]["species"]
        df_dandiset.to_csv(os.path.join(cache_dir, f"{code}.csv"))


def run_plot_dandiset(
    cache_dir: Optional[str] = None,
    threshold: Optional[float] = None,
    skip: Optional[Iterable[str]] = None,
) -> None:
    """Render trace SVGs for every dandiset whose code is above ``threshold``."""
    configure_logging()
    cache_dir = cache_dir or config.CONFIG.cache_dir
    threshold = config.CODES_TO_PLOT_THRESHOLD if threshold is None else threshold
    skip = list(skip) if skip is not None else config.DANDISETS_TO_SKIP

    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))
    csv_codes = [os.path.splitext(os.path.basename(x))[0] for x in csv_files]

    for code in csv_codes:
        if code == "all":
            print(f"Skipping {code}")
            continue
        try:
            if int(code) < threshold:
                continue
        except ValueError:
            continue
        df = pd.read_csv(os.path.join(cache_dir, f"{code}.csv"), index_col=0)
        ids = [x.split("/dandi/")[-1] for x in df.index.values]
        print(f"Processing {code}")
        if code in skip:
            print(f"Skipping {code}")
            continue
        folder = os.path.join(cache_dir, code)
        build_dataset_traces(folder, ids, parallel=True)


def sort_plot_dandiset(
    cache_dir: Optional[str] = None,
    traces_dir: Optional[str] = None,
) -> None:
    """Move generated SVGs from the dandi cache into the local traces dir."""
    cache_dir = cache_dir or config.CONFIG.cache_dir
    traces_dir = traces_dir or config.CONFIG.traces_dir
    svg_files = glob.glob(os.path.join(cache_dir, "**", "*.svg"), recursive=True)
    for svg_file in svg_files:
        print(f"Processing {svg_file}")
        # Strip everything up to and including the cache-dir prefix so the
        # output path mirrors the dandiset/subject layout.
        rel = os.path.relpath(svg_file, cache_dir)
        local_path = os.path.join(traces_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.move(svg_file, local_path)
