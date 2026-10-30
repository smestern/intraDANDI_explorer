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
from .metadata_table import apply_table

logger = logging.getLogger("dandi_scraper.database")


def _drop_duplicate_cells(dfs: pd.DataFrame, feature_cols, cell_status: dict) -> pd.DataFrame:
    """Drop cells whose features exactly match another cell's.

    The same recording is often published in several dandisets (e.g. Allen's
    000636 re-releases 000142/000209/000570), or saved twice within one dandiset
    under different session names. Identical feature vectors only come from the
    same file analysed twice, so these are exact duplicates. Keep the copy from
    the highest dandiset id (the newest release) and record the other copies'
    dandisets in ``also_in``.
    """
    key = pd.util.hash_pandas_object(dfs[feature_cols], index=False)
    order = pd.DataFrame({"key": key, "label": dfs["dandiset label"]}).sort_values(
        "label", ascending=False, kind="stable"
    )

    is_dup = order.duplicated("key", keep="first")
    kept_by_key = pd.Series(order.index[~is_dup], index=order.loc[~is_dup, "key"])
    dup_idx = order.index[is_dup]
    kept_idx = kept_by_key.loc[order.loc[is_dup, "key"]].values

    dfs = dfs.copy()
    dfs["also_in"] = ""
    for d, k in zip(dup_idx, kept_idx):
        cell_status[dfs.at[d, "id_full"]] = ("dropped_duplicate", f"same features as {dfs.at[k, 'id_full']}")
        dfs.at[k, "also_in"] = ";".join(filter(None, [dfs.at[k, "also_in"], dfs.at[d, "dandiset label"]]))
    logger.info(
        "de-duplication dropped %d cells (%d kept cells had copies)",
        len(dup_idx), len(set(kept_idx)),
    )
    return dfs.drop(index=dup_idx)


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
        if ".errors" in code: #debugging: skip error logs from previous runs
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
    # dfs["dandiset_id"]=dfs["dandiset label"]

    # Remap the indexes to "<6-digit dandiset>/<asset path>", independent of
    # where the cache lives. (Splitting on "00" broke when the cache path
    # itself contained "00", collapsing thousands of cells onto one key and
    # blowing up the metadata join below.)
    dfs.index = dfs.index.str.extract(r"/(\d{6}/.+)$", expand=False)
    if dfs.index.isna().any() or dfs.index.has_duplicates:
        raise ValueError(
            f"index remap produced {dfs.index.isna().sum()} unparsed and "
            f"{dfs.index.duplicated().sum()} duplicate ids"
        )
    logger.debug("first indexes after remap: %s", list(dfs.index[:5]))
    dfs["specimen_id"] = dfs.index
    dfs["id_full"] = dfs["dandiset label"].astype(str) + "/" + dfs["specimen_id"].astype(str)

    # Carry gigaseal analysis-stage failures into the merge log.
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
        reason_by_id = qc_drops.groupby("id_full")["feature"].apply(
            lambda s: ",".join(sorted(set(s)))
        ).to_dict()
    else:
        reason_by_id = {}
    for _id in dropped_by_qc:
        cell_status[_id] = ("dropped_quick_qc", reason_by_id.get(_id, "unknown feature"))
    logger.info("quick_qc dropped %d cells", len(dropped_by_qc))

    # Drop rows missing >50% of the COLS_TO_KEEP features, since imputation is
    # less likely to be successful and these may be low-quality cells. Density
    # is measured on COLS_TO_KEEP only: counting every column penalized cells
    # that lack the higher-amplitude *_N_long_square sweeps (e.g. datasets with
    # coarse current steps), even though those columns are never used.
    present_keep = [c for c in config.COLS_TO_KEEP if c in dfs.columns]
    sparse_mask = dfs[present_keep].notna().sum(axis=1) < int(len(present_keep) * 0.5)
    for _id in dfs.loc[sparse_mask, "id_full"]:
        if cell_status.get(_id, ("loaded",))[0] == "loaded":
            cell_status[_id] = ("dropped_sparse_row", "<50% of COLS_TO_KEEP present")
    logger.info("sparse-row filter dropped %d cells", int(sparse_mask.sum()))
    dfs = dfs.loc[~sparse_mask]

    before_cols = set(dfs.columns)
    dfs = dfs.dropna(axis=1, thresh=int(len(dfs) * 0.5))
    dropped_cols = before_cols - set(dfs.columns)
    if dropped_cols:
        logger.warning(
            "dropped %d columns failing <50%% density: %s",
            len(dropped_cols), sorted(dropped_cols),
        )

    dfs = _drop_duplicate_cells(dfs, [c for c in present_keep if c in dfs.columns], cell_status)

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
        meta_ = None
        if use_cached_metadata and df_old is not None:
            meta_ = df_old.loc[
                df_old["dandiset_id"] == int(float(code)),
                ["dandiset_id", "age", "subject_id", "cell_id",
                 "brain_region", "species", "filepath", "contributor"],
            ]
            if meta_.empty:
                logger.info("dandiset %s not in cached metadata, fetching from DANDI", code)
                meta_ = None
        if meta_ is None:
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
    # Region/species come from dandiset_metadata.csv keyed on the dandiset label,
    # so cells missing from the cached metadata still get them.
    dfs = apply_table(dfs, id_col="dandiset label")
    logger.info("dfs shape after joining meta data: %s", dfs.shape)

    #dfs = dfs.loc[np.hstack(idxs)]
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
    dataset_numeric = scale_features(dataset_numeric) #not a standardizing scaler but a scale for individual features eg log scale ISIs

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

    logger.info("Dumping dataset_numeric to %s", dataset_numeric_pkl)
    #joblib.dump(dataset_numeric, dataset_numeric_pkl)

    #logger info about scaling
    logger.info("Scaling dataset_numeric with RobustScaler")
    scaler = RobustScaler()
    dataset_numeric.loc[:, :] = np.clip(scaler.fit_transform(dataset_numeric.values),
                                        -config.ROBUST_CLIP, config.ROBUST_CLIP)

    logger.info("creating umap from dataset_numeric")
    reducer = umap.UMAP(densmap=False, min_dist=0.3, spread=10, metric="cosine",
                        n_neighbors=500, verbose=True)
    reducer.fit_transform(dataset_numeric)
    reducer2 = umap.UMAP(n_neighbors=150, min_dist=0.1, spread=2,
                         repulsion_strength=5, metric="cosine")
    reducer2.fit(dataset_numeric)

    dfs["umap X"] = reducer2.embedding_[:, 0]
    dfs["umap Y"] = reducer2.embedding_[:, 1]
    plt.scatter(dfs["umap X"], dfs["umap Y"], s=0.1)
    plt.savefig("umap_scatter.png")
    plt.close()
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

    plt.scatter(dfs["norm umap X"], dfs["norm umap Y"], s=0.1)
    plt.savefig("norm_umap_scatter.png")
    plt.close()
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
                    logger.info(
                        "asset not found in cached metadata for %s/%s, fetching from DANDI",
                        dandiset_id, specimen_id
                    )
                    try:
                        asset = client.get_dandiset(
                            str(int(dandiset_id)).zfill(6), 
                        ).get_asset_by_path("/".join(specimen_id.split("/")[1:]))
                    except Exception as e:
                        try:
                            asset = client.get_dandiset(
                            str(int(dandiset_id)).zfill(6), 
                        ).get_asset_by_path("/".join(specimen_id.split("/")[2:])) #no idea man... Sometimes this fails, Maybe due to rate limiting? 
                        except Exception as e:
                            logger.warning(
                                "failed to retrieve asset for %s/%s: %s",
                                dandiset_id, specimen_id, str(e),
                            )
                            meta_data_link.append("")
                            file_link.append("")
                            continue
                    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
                    meta_data_link.append(asset.api_url)
                    file_link.append(s3_url)
                    continue
                logger.info(
                    "using cached metadata for %s/%s",
                    dandiset_id, specimen_id
                )
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
