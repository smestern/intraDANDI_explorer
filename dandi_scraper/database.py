"""Merge per-dandiset analysis CSVs into the final database (`all_new.csv`).

Includes imputation, scaling, UMAP/PCA embedding, GMM clustering, and
per-cell status tracking written to `all_new.errors.csv`.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from dandi.dandiapi import DandiAPIClient
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from . import config
from .analysis import get_dandi_metadata, quick_qc, scale_features
from .logging_utils import configure_logging

logger = logging.getLogger("dandi_scraper.database")


def run_merge_dandiset(
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_cached_metadata: bool = True,
) -> None:
    """Merge every per-dandiset CSV in ``cache_dir`` into the final database
    written under ``output_dir``."""
    configure_logging()
    cache_dir = cache_dir or config.CONFIG.cache_dir
    output_dir = output_dir or config.CONFIG.output_dir
    os.makedirs(output_dir, exist_ok=True)

    merged_csv = os.path.join(output_dir, "all_new.csv")
    merged_errors_csv = os.path.join(output_dir, "all_new.errors.csv")
    dataset_numeric_pkl = os.path.join(output_dir, "dataset_numeric.pkl")
    dataset_numeric_norm_pkl = os.path.join(output_dir, "dataset_numeric_norm.pkl")

    # cell_status tracks each id_full through the pipeline; terminal status is
    # written to all_new.errors.csv at the end so every missing cell can be
    # traced back to a specific drop site.
    cell_status: dict = {}

    def _mark(ids, status, detail=""):
        for _id in ids:
            cell_status[_id] = (status, detail)

    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))
    csv_codes = [os.path.splitext(os.path.basename(x))[0] for x in csv_files]
    dfs = []
    for code in csv_codes:
        if code == "all":
            continue
        temp_df = pd.read_csv(os.path.join(cache_dir, f"{code}.csv"), index_col=0)
        temp_df.rename(columns={"dandiset": "dandiset label", "species label": "species"}, inplace=True)
        logger.info("loaded dandiset csv %s: %d rows", code, len(temp_df))
        dfs.append(temp_df)

    df_old = pd.read_csv(merged_csv, index_col=0) if os.path.exists(merged_csv) else None

    dfs = pd.concat(dfs)
    dfs["dandiset label"] = dfs["dandiset label"].apply(
        lambda x: "000000"[: 6 - len(str(x))] + str(x)
    )

    # Remap the indexes; original code used a custom split on "dandi//".
    dfs.index = dfs.index.map(lambda x: "".join(x.split("dandi//")[1]))
    logger.debug("first indexes after remap: %s", list(dfs.index[:5]))
    dfs["specimen_id"] = dfs.index
    dfs["id_full"] = dfs["dandiset label"] + "/" + dfs["specimen_id"]

    # Carry pyAPisolation analysis-stage failures into the merge log.
    if "error_stage" in dfs.columns:
        fail_mask = dfs["error_stage"].notna() & (dfs["error_stage"] != "")
        for _id, stage, msg in zip(
            dfs.loc[fail_mask, "id_full"],
            dfs.loc[fail_mask, "error_stage"],
            dfs.get("error_message", pd.Series(index=dfs.index, dtype=object)).loc[fail_mask],
        ):
            cell_status[_id] = (
                f"analysis_failed:{stage}",
                str(msg) if pd.notna(msg) else "",
            )
        logger.info("analysis-stage failures carried into merge: %d", int(fail_mask.sum()))
    _mark([i for i in dfs["id_full"] if i not in cell_status], "loaded")

    pre_qc_ids = set(dfs["id_full"])
    dfs, qc_drops = quick_qc(dfs, return_drops=True)
    post_qc_ids = set(dfs["id_full"])
    dropped_by_qc = pre_qc_ids - post_qc_ids
    if not qc_drops.empty:
        qc_drops["id_full"] = qc_drops["specimen_id"].map(
            lambda sid: dfs["id_full"].get(sid) if sid in dfs.index else sid
        )
        reason_by_id = qc_drops.groupby("id_full")["feature"].apply(
            lambda s: ",".join(sorted(set(s)))
        ).to_dict()
    else:
        reason_by_id = {}
    for _id in dropped_by_qc:
        cell_status[_id] = ("dropped_quick_qc", reason_by_id.get(_id, "unknown feature"))
    logger.info("quick_qc dropped %d cells", len(dropped_by_qc))

    before_cols = set(dfs.columns)
    dfs = dfs.dropna(axis=1, thresh=int(len(dfs) * 0.9))
    dropped_cols = before_cols - set(dfs.columns)
    if dropped_cols:
        logger.warning(
            "dropped %d columns failing <90%% density: %s",
            len(dropped_cols), sorted(dropped_cols),
        )

    missing_cols_per_dandiset: dict = {}
    idxs = []
    meta_data = []
    dataset_numeric = []
    for code in dfs["dandiset label"].unique():
        temp_df = dfs.loc[dfs["dandiset label"] == code]
        logger.info("merging dandiset %s (%d rows)", code, len(temp_df))
        missing_here = [c for c in config.COLS_TO_KEEP if c not in temp_df.columns]
        if missing_here:
            missing_cols_per_dandiset[code] = missing_here
            logger.warning(
                "dandiset %s missing %d cols_to_keep: %s",
                code, len(missing_here), missing_here,
            )
        if use_cached_metadata and df_old is not None:
            meta_ = df_old.loc[
                df_old["dandiset_id"] == int(code),
                ["dandiset_id", "age", "subject_id", "cell_id",
                 "brain_region", "species", "filepath", "contributor"],
            ]
        else:
            meta_ = get_dandi_metadata(code)
        meta_data.append(meta_)

        data_num = temp_df.select_dtypes(include=np.number).dropna(axis=1, how="all")
        assert len(data_num) == len(temp_df)
        temp_data_num = data_num.copy()
        if data_num.empty or len(data_num.columns) < 3:
            for _id in temp_df["id_full"]:
                cell_status[_id] = ("dropped_empty_numeric", f"numeric cols={len(data_num.columns)}")
            logger.warning(
                "dandiset %s: skipping (numeric cols=%d)", code, len(data_num.columns),
            )
            continue

        idxs.append(data_num.index.values)
        logger.info("dandiset %s: %d cells entering imputation", code, len(data_num))
        data_num = np.nan_to_num(data_num, nan=np.nan, posinf=np.nan, neginf=np.nan)
        impute = KNNImputer(keep_empty_features=True)
        data_num = impute.fit_transform(data_num)
        logger.debug(
            "dandiset %s: imputed shape=(%d, %d)", code, len(data_num), len(data_num[0])
        )
        for i in range(data_num.shape[1]):
            col = data_num[:, i]
            lower_bound = np.nanpercentile(col, 0.5)
            upper_bound = np.nanpercentile(col, 99.5)
            data_num[:, i] = np.clip(col, lower_bound, upper_bound)

        data_num = pd.DataFrame(data_num, columns=temp_data_num.columns, index=temp_data_num.index)
        assert len(data_num) == len(temp_df)
        dataset_numeric.append(data_num)

    meta_data = pd.concat(meta_data, axis=0)
    logger.info("meta data shape: %s", meta_data.shape)

    dfs = dfs.join(meta_data, how="left", rsuffix="_meta")
    logger.info("dfs shape after joining meta data: %s", dfs.shape)

    dfs = dfs.loc[np.hstack(idxs)]
    logger.info("dfs shape after index filter: %s", dfs.shape)

    concat_num = pd.concat(dataset_numeric, axis=0)
    missing_final = [c for c in config.COLS_TO_KEEP if c not in concat_num.columns]
    if missing_final:
        logger.error(
            "cols_to_keep missing from concatenated numeric frame: %s — "
            "cells in dandisets that lacked these will be dropped by final "
            "dropna(any). Per-dandiset missing map: %s",
            missing_final, missing_cols_per_dandiset,
        )
    dataset_numeric = concat_num[config.COLS_TO_KEEP]
    logger.info("dataset_numeric shape after concatenation: %s", dataset_numeric.shape)
    dataset_numeric = scale_features(dataset_numeric)

    # Final KNN impute on the numeric feature matrix.
    impute = KNNImputer(keep_empty_features=True)
    dataset_numeric.loc[:, :] = impute.fit_transform(dataset_numeric.values)

    dataset_numeric = dataset_numeric.dropna(axis=1, how="any")
    logger.info("dataset_numeric shape after dropping columns: %s", dataset_numeric.shape)

    pre_align_dfs_ids = set(dfs["id_full"])
    dfs = dfs.loc[dataset_numeric.index]
    post_align_ids = set(dfs["id_full"])
    for _id in (pre_align_dfs_ids - post_align_ids):
        if _id not in cell_status or cell_status[_id][0] == "loaded":
            cell_status[_id] = ("dropped_final_align", "row missing after final dropna/align")

    joblib.dump(dataset_numeric, dataset_numeric_pkl)
    scaler = RobustScaler()
    dataset_numeric.loc[:, :] = scaler.fit_transform(dataset_numeric.values)

    reducer = umap.UMAP(densmap=False, min_dist=0.3, spread=10, metric="cosine",
                        n_neighbors=500, verbose=True)
    reducer.fit_transform(dataset_numeric)
    reducer2 = umap.UMAP(n_neighbors=150, min_dist=0.1, spread=2,
                         repulsion_strength=5, metric="cosine")
    reducer2.fit(dataset_numeric)

    dfs["umap X"] = reducer2.embedding_[:, 0]
    dfs["umap Y"] = reducer2.embedding_[:, 1]
    plt.scatter(dfs["umap X"], dfs["umap Y"], s=0.1)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(dataset_numeric.fillna(0))
    dfs["pca X"] = embedding[:, 0]
    dfs["pca Y"] = embedding[:, 1]

    gmm = GaussianMixture(n_components=20)
    gmm.fit(dataset_numeric)
    dfs["GMM cluster label"] = gmm.predict(dataset_numeric)

    reducer3 = umap.UMAP(densmap=False, target_weight=0.1, n_neighbors=50, verbose=True)
    reducer3.fit(dataset_numeric, y=dfs["GMM cluster label"]) + reducer2
    dfs["supervised umap X"] = reducer3.embedding_[:, 0]
    dfs["supervised umap Y"] = reducer3.embedding_[:, 1]

    # Per-dandiset min-max normalization, then UMAP again.
    dataset_numeric_norm = dataset_numeric.copy()
    for code in dfs["dandiset label"].unique():
        temp_df = dataset_numeric.loc[dfs["dandiset label"] == code]
        if len(temp_df) < 10:
            continue
        scaler = MinMaxScaler()
        dataset_numeric_norm.loc[temp_df.index, :] = scaler.fit_transform(temp_df.values)
    reducer4 = umap.UMAP(densmap=False, min_dist=0.3, spread=10, metric="cosine",
                         n_neighbors=500, verbose=True)
    embedding = reducer4.fit_transform(dataset_numeric_norm)
    dfs["norm umap X"] = embedding[:, 0]
    dfs["norm umap Y"] = embedding[:, 1]

    joblib.dump(dataset_numeric_norm, dataset_numeric_norm_pkl)

    dfs["dandiset_link"] = dfs["dandiset label"].apply(
        lambda x: f"https://dandiarchive.org/dandiset/{str(int(x)).zfill(6)}"
    )
    file_link = []
    meta_data_link = []
    with DandiAPIClient() as client:
        for dandiset_id, specimen_id in zip(dfs["dandiset label"], dfs["specimen_id"]):
            if use_cached_metadata and df_old is not None:
                asset = df_old.loc[df_old["id_full"] == (dandiset_id + "/" + specimen_id)]
                if asset.empty:
                    asset = client.get_dandiset(
                        str(int(dandiset_id)).zfill(6), "draft"
                    ).get_asset_by_path("/".join(specimen_id.split("/")[1:]))
                    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                    meta_data_link.append(asset.api_url)
                    file_link.append(s3_url)
                    continue
                s3_url = asset["file_link"].values[0]
                meta_data_link.append(asset["meta_data_link"].values[0])
            else:
                asset = client.get_dandiset(
                    str(int(dandiset_id)).zfill(6), "draft"
                ).get_asset_by_path("/".join(specimen_id.split("/")[1:]))
                s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                meta_data_link.append(asset.api_url)
            file_link.append(s3_url)
    dfs["file_link"] = file_link
    dfs["meta_data_link"] = meta_data_link

    dfs.to_csv(merged_csv)

    for _id in dfs["id_full"]:
        cell_status[_id] = ("kept", "")
    status_rows = [
        {"id_full": k, "status": v[0], "detail": v[1]}
        for k, v in cell_status.items()
    ]
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(merged_errors_csv, index=False)
    terminal_counts = status_df["status"].value_counts().to_dict()
    logger.info(
        "merge complete: wrote %s (%d kept) and %s (%d rows); status breakdown=%s",
        merged_csv,
        int((status_df["status"] == "kept").sum()),
        merged_errors_csv,
        len(status_df),
        terminal_counts,
    )
