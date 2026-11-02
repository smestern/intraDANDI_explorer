"""UMAP workup: how much of the embedding's structure is dataset batch vs. biology.

Rebuilds the feature matrix the merge feeds to UMAP, scores batch mixing and biology
retention in every space (features, site UMAP, norm UMAP, PCA), decomposes per-feature
variance into species/region vs. dandiset, checks the usual technical suspects (voltage
offsets, step protocol, imputation, redundant features), audits UMAP islands, and refits
a handful of light-touch pipeline variants to compare them on the same scores.

    python tools/umap_workup.py --out figures/umap_workup [--refresh] [--skip-variants]

UMAP fits are cached in ``<out>/cache`` (keyed on variant name); ``--refresh`` refits.
Nothing in the merge pipeline is changed; this only reads ``all_new.csv``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from umap import UMAP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dandi_scraper import config  # noqa: E402
from dandi_scraper.analysis import scale_features  # noqa: E402
from dandi_scraper.database import _drop_duplicate_cells  # noqa: E402
from blog_figures import (CONTEXT, INK, INK_2, MUTED, SLOTS, SURFACE, NHP,  # noqa: E402
                          SPECIES_GROUP_COLORS, umap_axes)

SEED = 0
K = 15  # neighbourhood size for purity / self-fraction scores
K_STRATUM = 50  # wider neighbourhood for the within-stratum score, since only same-stratum neighbours count

# Lab families: same lab usually means same rig, solutions, protocol and analysis, so
# datasets inside a family that still separate are either sampling or real batch.
LAB_FAMILY = {
    **{d: "Allen (AIBS)" for d in [
        "000020", "000023", "000043", "000109", "000142", "000209", "000228", "000288", "000337",
        "000569", "000570", "000630", "000635", "000636", "000768", "000769", "000933", "000934",
        "001065", "001348", "001351", "001359", "001455", "001464", "001465", "001534", "001625",
        "001626", "001748", "001753", "001855", "001856", "001858", "001938"]},
    "000008": "Tolias (BCM)", "000035": "Tolias (BCM)",
    **{d: "Zhang (spinal cord)" for d in ["000117", "000148", "000245", "000295"]},
}
LAB_ORDER = ["Allen (AIBS)", "Tolias (BCM)", "Zhang (spinal cord)", "Other labs"]
LAB_COLORS = dict(zip(LAB_ORDER, [SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[3]]))

# Datasets that should mix if there were no batch effect: same lab, rig, sampling rate, protocol,
# species and broad cell-class sampling. 000630 (L1) and 000636 (interneurons) are left out on
# purpose, and so is 000020 vs 001455: 000020 looks GABAergic-focused (25% fast-spiking) while 001455
# has no fast-spiking cells, so their separation is confounded with cell class. It is still
# scored, as REPORTED_PAIRS, but not averaged into the control score.
CONTROL_PAIRS = [("000142", "000109"), ("000209", "000142"),
                 ("001359", "001464"), ("000769", "001359"), ("001348", "001465")]
REPORTED_PAIRS = [("000020", "001455"), ("000008", "000035")]

VOLTAGE_FEATURES = [
    "v_baseline", "ap_1_threshold_v_0_long_square", "ap_1_peak_v_0_long_square",
    "ap_1_fast_trough_v_0_long_square", "ap_mean_threshold_v_0_long_square",
    "ap_mean_peak_v_0_long_square", "ap_mean_fast_trough_v_0_long_square",
]
FEATURE_GROUPS = {
    "absolute voltages": VOLTAGE_FEATURES,
    "passive (Rin, tau, sag)": ["input_resistance", "tau", "sag_nearest_minus_100"],
    "AP kinetics (upstroke, width)": [
        "ap_1_upstroke_0_long_square", "ap_1_width_0_long_square",
        "ap_mean_upstroke_0_long_square", "ap_mean_width_0_long_square"],
    "rheobase firing (rate, latency)": ["avg_rate_0_long_square", "latency_0_long_square"],
}
SHORT = {
    "input_resistance": "Rin", "tau": "tau", "v_baseline": "V baseline", "sag_nearest_minus_100": "sag",
    "ap_1_threshold_v_0_long_square": "AP1 threshold", "ap_1_peak_v_0_long_square": "AP1 peak",
    "ap_1_upstroke_0_long_square": "AP1 upstroke", "ap_1_width_0_long_square": "AP1 width",
    "ap_1_fast_trough_v_0_long_square": "AP1 fast trough",
    "ap_mean_threshold_v_0_long_square": "APmean threshold", "ap_mean_peak_v_0_long_square": "APmean peak",
    "ap_mean_upstroke_0_long_square": "APmean upstroke", "ap_mean_width_0_long_square": "APmean width",
    "ap_mean_fast_trough_v_0_long_square": "APmean fast trough",
    "avg_rate_0_long_square": "rheobase rate", "latency_0_long_square": "rheobase latency",
    "fi_linear_fit_slope": "f-I slope",
    "ap_1_thr_rel": "AP1 thr - Vb", "ap_1_amp": "AP1 peak - thr", "ap_1_ahp": "AP1 trough - thr",
    "ap_mean_thr_rel": "APmean thr - Vb", "ap_mean_amp": "APmean peak - thr",
    "ap_mean_ahp": "APmean trough - thr",
}

DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a12b2b"])
SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    "blues", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})


def save(fig, out: str, name: str) -> None:
    path = os.path.join(out, f"{name}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------

def load_cells(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["umap X"]).reset_index(drop=True)
    df["dandiset label"] = df["dandiset label"].astype(str).str.zfill(6)
    n = len(df)
    df = _drop_duplicate_cells(df, list(config.COLS_TO_KEEP), {}).reset_index(drop=True)
    print(f"{n - len(df)} duplicate cells dropped, {len(df)} cells")
    ds = df["dandiset label"]
    df["ds"] = ds
    df["species"] = df["species"].fillna("Unknown species")
    df["species group"] = np.where(df["species"].isin(NHP), "Non-human primate", df["species"])
    df["region"] = df["brain_region"].fillna("Unknown")
    df["lab"] = ds.map(LAB_FAMILY).fillna("Other labs")
    # Species x region is the biology we want the map to keep. The "Unknown" region is the Allen
    # HMBA/BICAN uploads, so species|Unknown is still a coherent same-lab stratum.
    df["stratum"] = df["species"] + " | " + df["region"]

    w = df["analyzed_window"].str.split("-", expand=True).astype(float)
    df["step duration"] = (w[1] - w[0]).round(1)
    df["step gap"] = [step_gap(a, r) for a, r in zip(df["analyzed_sweeps"], df["rheobase_i"])]
    df["overshoot"] = df["step gap"] / df["rheobase_i"]

    width = df["ap_1_width_0_long_square"] * 1000
    ratio = df["ap_1_upstroke_downstroke_ratio_0_long_square"]
    df["fast spiking"] = (width < 0.5) & (ratio < 2.0)
    return df


def step_gap(analyzed, rheobase) -> float:
    """Rheobase minus the largest smaller positive step: the step-size resolution at rheobase."""
    if not isinstance(analyzed, str) or not np.isfinite(rheobase):
        return np.nan
    amps = [float(x.split(":")[1]) for x in analyzed.split(";")]
    below = [a for a in amps if 0 < a < rheobase - 1e-6]
    return rheobase - max(below) if below else np.nan


def add_relative_voltages(df: pd.DataFrame) -> pd.DataFrame:
    """Voltage differences, which a liquid-junction or offset error cancels out of."""
    for p in ("ap_1", "ap_mean"):
        thr = df[f"{p}_threshold_v_0_long_square"]
        df[f"{p}_thr_rel"] = thr - df["v_baseline"]
        df[f"{p}_amp"] = df[f"{p}_peak_v_0_long_square"] - thr
        df[f"{p}_ahp"] = df[f"{p}_fast_trough_v_0_long_square"] - thr
    return df


def pipeline_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Mirror run_merge_dandiset's preprocessing (dandi_scraper/database.py):
    per-dandiset KNN impute + 0.5-99.5 percentile clip, SCALE_FEATURES log transforms,
    global KNN impute, RobustScaler clipped to +/- ROBUST_CLIP.

    Approximation: the merge imputes each dandiset using all of its numeric columns as
    neighbours; here only ``cols`` are used.
    """
    parts = []
    for _, grp in df[cols].groupby(df["ds"]):
        grp = grp.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        vals = KNNImputer(keep_empty_features=True).fit_transform(grp.values)
        lo, hi = np.nanpercentile(vals, [0.5, 99.5], axis=0)
        parts.append(pd.DataFrame(np.clip(vals, lo, hi), index=grp.index, columns=grp.columns))
    X = pd.concat(parts).reindex(index=df.index, columns=cols)
    X = scale_features(X).replace([np.inf, -np.inf], np.nan)
    X.loc[:, :] = KNNImputer(keep_empty_features=True).fit_transform(X.values)
    X.loc[:, :] = np.clip(RobustScaler().fit_transform(X.values), -config.ROBUST_CLIP, config.ROBUST_CLIP)
    return X


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def knn(E: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1).fit(E)
    return nn.kneighbors(E, return_distance=False)[:, 1:]


def same_frac(idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (labels[idx] == labels[:, None]).mean(axis=1)


def expected_same(labels: np.ndarray) -> np.ndarray:
    s = pd.Series(labels)
    return ((s.map(s.value_counts()) - 1) / (len(s) - 1)).to_numpy()


def inverse_simpson(idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
    codes = pd.factorize(labels)[0]
    out = np.empty(len(idx))
    for i, row in enumerate(codes[idx]):
        p = np.bincount(row) / len(row)
        out[i] = 1.0 / np.sum(p ** 2)
    return out


def within_stratum_self(idx: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: among neighbours from the same species|region stratum, the fraction from the
    same dandiset, against what random mixing inside the stratum would give. Only strata with
    >= 2 dandisets of >= 20 cells count, so biology between strata can't inflate it."""
    ds, st = df["ds"].to_numpy(), df["stratum"].to_numpy()
    n_ds = df.groupby("ds")["ds"].transform("size").to_numpy()
    big = df.loc[n_ds >= 20]
    n_big = big.groupby("stratum")["ds"].nunique()
    ok_strata = set(n_big.index[n_big >= 2])
    n_st = df.groupby("stratum")["ds"].transform("size").to_numpy()
    rows = []
    for i in np.flatnonzero(np.isin(st, list(ok_strata)) & (n_ds >= 20)):
        nb = idx[i][st[idx[i]] == st[i]]
        if len(nb) < 5:
            continue
        rows.append((i, (ds[nb] == ds[i]).mean(), (n_ds[i] - 1) / (n_st[i] - 1)))
    return pd.DataFrame(rows, columns=["i", "obs", "exp"]).set_index("i")


def pair_mixing(idx: np.ndarray, df: pd.DataFrame, a: str, b: str) -> float:
    """For cells of datasets a and b, the share of their a-or-b neighbours from the other
    dataset, divided by what random mixing gives (1 = fully mixed, 0 = fully separated)."""
    ds = df["ds"].to_numpy()
    n = {a: (ds == a).sum(), b: (ds == b).sum()}
    obs, exp = [], []
    for i in np.flatnonzero(np.isin(ds, [a, b])):
        nb = ds[idx[i]]
        nb = nb[np.isin(nb, [a, b])]
        if len(nb) == 0:
            continue
        other = b if ds[i] == a else a
        obs.append((nb == other).mean())
        exp.append(n[other] / (n[a] + n[b] - 1))
    return float(np.mean(obs) / np.mean(exp)) if obs else np.nan


def score_space(E: np.ndarray, df: pd.DataFrame, X_in: np.ndarray | None = None,
                metric: str = "euclidean") -> dict:
    """Batch-mixing and biology-retention scores for one embedding or feature space."""
    idx_wide = knn(E, K_STRATUM, metric)
    idx = idx_wide[:, :K]
    ds = df["ds"].to_numpy()
    s = {}
    s["dandiset self frac"] = same_frac(idx, ds).mean()
    s["dandiset self frac (chance)"] = expected_same(ds).mean()
    s["dandiset iLISI"] = inverse_simpson(idx, ds).mean()
    ws = within_stratum_self(idx_wide, df)
    s["within-stratum self frac"] = ws["obs"].mean()
    s["within-stratum self frac (chance)"] = ws["exp"].mean()
    s["within-stratum enrichment"] = ws["obs"].mean() / ws["exp"].mean()
    for col, name in [("species", "species"), ("species group", "species group")]:
        lab = df[col].to_numpy()
        s[f"{name} purity enrichment"] = same_frac(idx, lab).mean() / expected_same(lab).mean()
    known = (df["region"] != "Unknown").to_numpy()
    reg = df["region"].to_numpy()
    nb_known = known[idx] & known[:, None]
    obs = ((reg[idx] == reg[:, None]) & nb_known)[known].sum() / nb_known[known].sum()
    exp = expected_same(reg[known]).mean()
    s["region purity enrichment"] = obs / exp
    fs = df["fast spiking"].to_numpy()
    s["FS purity enrichment"] = fs[idx[fs]].mean() / fs.mean()
    # Do fast-spiking cells from different datasets meet? iLISI over FS cells' FS neighbours.
    sub = np.flatnonzero(fs)
    idx_fs = knn(E[sub], K, metric)
    s["FS dandiset iLISI"] = inverse_simpson(idx_fs, ds[sub]).mean()
    for a, b in CONTROL_PAIRS + REPORTED_PAIRS:
        s[f"mix {a}-{b}"] = pair_mixing(idx, df, a, b)
    s["control-pair mixing (mean)"] = np.nanmean([s[f"mix {a}-{b}"] for a, b in CONTROL_PAIRS])
    if X_in is not None:
        rng = np.random.default_rng(SEED)
        sub = rng.choice(len(E), 3000, replace=False)
        s["trustworthiness"] = trustworthiness(X_in[sub], E[sub], n_neighbors=K)
    return s


def silhouettes(E: np.ndarray, df: pd.DataFrame, metric="euclidean") -> dict:
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(E), 5000, replace=False)
    return {f"silhouette by {c}": silhouette_score(E[sub], df[c].to_numpy()[sub], metric=metric)
            for c in ["ds", "species", "stratum", "lab"]}


# ---------------------------------------------------------------------------
# Section 0/1: rebuild, validate, score every space
# ---------------------------------------------------------------------------

# Mirrors reducer2 in database.run_merge_dandiset.
SITE_PARAMS = dict(n_neighbors=150, min_dist=0.1, spread=2, repulsion_strength=1, metric="cosine")


def fit_umap(name: str, X: np.ndarray, cache: str, refresh: bool, **params) -> np.ndarray:
    path = os.path.join(cache, f"emb_{name}.npz")
    key = json.dumps(params, sort_keys=True) + f"|{X.shape}|{float(np.round(X.sum(), 3))}"
    if os.path.exists(path) and not refresh:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            return z["emb"]
    print(f"  fitting UMAP {name} {params}")
    emb = UMAP(**{"random_state": SEED, **params}).fit_transform(X)
    np.savez(path, emb=emb, key=key)
    return emb


def section_spaces(df, X, cache, refresh, out) -> pd.DataFrame:
    print("section 0/1: spaces")
    Xv = X.to_numpy()
    spaces = {
        "features (cosine)": (Xv, "cosine", None),
        "features (euclidean)": (Xv, "euclidean", None),
        "site UMAP (stored)": (df[["umap X", "umap Y"]].to_numpy(), "euclidean", Xv),
        "site UMAP (refit here)": (fit_umap("site_refit", Xv, cache, refresh, **SITE_PARAMS),
                                   "euclidean", Xv),
        "norm UMAP (stored)": (df[["norm umap X", "norm umap Y"]].to_numpy(), "euclidean", Xv),
        "supervised UMAP (stored)": (df[["supervised umap X", "supervised umap Y"]].to_numpy(),
                                     "euclidean", Xv),
        "PCA 2-D (stored)": (df[["pca X", "pca Y"]].to_numpy(), "euclidean", Xv),
    }
    rows = {}
    for name, (E, metric, X_in) in spaces.items():
        print(f"  scoring {name}")
        rows[name] = {**score_space(E, df, X_in, metric), **silhouettes(E, df, metric)}
    res = pd.DataFrame(rows).T
    res.to_csv(os.path.join(out, "scores_spaces.csv"))

    # per-dandiset self-enrichment in the stored site UMAP and in feature space
    per_ds = []
    for name in ["features (cosine)", "site UMAP (stored)"]:
        E, metric, _ = spaces[name]
        idx = knn(E, K, metric)
        sf = pd.Series(same_frac(idx, df["ds"].to_numpy()), index=df.index)
        exp = pd.Series(expected_same(df["ds"].to_numpy()), index=df.index)
        per_ds.append(sf.groupby(df["ds"]).mean().rename(name))
    per_ds.append(exp.groupby(df["ds"]).mean().rename("chance"))
    per_ds = pd.concat(per_ds, axis=1)
    meta = df.groupby("ds").agg(n=("ds", "size"), species=("species", "first"),
                                region=("region", "first"), lab=("lab", "first"),
                                group=("species group", "first"))
    per_ds = meta.join(per_ds).sort_values("site UMAP (stored)", ascending=False)
    per_ds.to_csv(os.path.join(out, "self_enrichment_per_dandiset.csv"))
    fig_self_enrichment(per_ds, out)
    fig_overview(df, out)
    return res


def fig_overview(df, out):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, col, colors, title in [
        (axs[0], "species group", SPECIES_GROUP_COLORS, "Species"),
        (axs[1], "lab", LAB_COLORS, "Lab family"),
    ]:
        c = df[col].map(colors).fillna(CONTEXT).to_numpy()
        ax.scatter(df["umap X"].to_numpy()[order], df["umap Y"].to_numpy()[order], c=c[order],
                   s=1.5, linewidths=0, rasterized=True)
        umap_axes(ax)
        counts = df[col].value_counts()
        handles = [plt.Line2D([], [], ls="", marker="o", ms=5, mfc=v, mec="none",
                              label=f"{k} (n={counts.get(k, 0):,})") for k, v in colors.items()]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.07), fontsize=7, ncol=2)
        ax.set_title(f"Site UMAP by {title.lower()}", loc="left", color=INK)
    fig.tight_layout()
    save(fig, out, "01_site_umap_species_lab")


def fig_self_enrichment(per_ds, out):
    d = per_ds[per_ds["n"] >= 20].sort_values("site UMAP (stored)")
    fig, ax = plt.subplots(figsize=(6.4, 0.17 * len(d) + 0.8))
    y = np.arange(len(d))
    ax.hlines(y, d["features (cosine)"], d["site UMAP (stored)"], color=CONTEXT, lw=2, zorder=1)
    ax.scatter(d["chance"], y, marker="|", s=40, color=MUTED, zorder=2, label="chance (dataset size)")
    ax.scatter(d["features (cosine)"], y, s=18, facecolors="white", edgecolors=INK_2, lw=1,
               zorder=3, label="16-D feature space (cosine)")
    ax.scatter(d["site UMAP (stored)"], y, s=22, c=d["group"].map(SPECIES_GROUP_COLORS).fillna(MUTED),
               zorder=4, edgecolors=SURFACE, lw=0.6, label="site UMAP (colour = species)")
    ax.set_yticks(y, [f"{i}  {r.region[:14]}  (n={r.n})  [{r.lab.split(' ')[0]}]"
                      for i, r in d.iterrows()], fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Share of {K} nearest neighbours from the same dataset")
    ax.set_title("How strongly each dataset keeps to itself (datasets with >= 20 cells)",
                 loc="left", color=INK)
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, out, "02_self_enrichment_per_dandiset")


# ---------------------------------------------------------------------------
# Section 2: biology vs batch
# ---------------------------------------------------------------------------

def r2_by(X: pd.DataFrame, groups: pd.Series) -> pd.Series:
    means = X.groupby(groups.to_numpy()).transform("mean")
    return 1 - ((X - means) ** 2).sum() / ((X - X.mean()) ** 2).sum()


def section_variance(df, X, out) -> pd.DataFrame:
    print("section 2: variance partition")
    r2_sp = r2_by(X, df["species"])
    r2_st = r2_by(X, df["stratum"])
    r2_ds = r2_by(X, df["ds"])
    res = pd.DataFrame({"species": r2_sp, "species x region": r2_st, "dandiset": r2_ds})
    res["dandiset beyond stratum"] = res["dandiset"] - res["species x region"]
    res = res.sort_values("dandiset", ascending=True)
    res.to_csv(os.path.join(out, "variance_partition.csv"))

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    y = np.arange(len(res))
    a = res["species"].to_numpy()
    b = res["species x region"].to_numpy() - a
    c = res["dandiset beyond stratum"].to_numpy()
    for left, w, col, lab in [(0, a, SLOTS[0], "species"),
                              (a, b, "#86b6ef", "+ region (within species)"),
                              (a + b, c, SLOTS[1], "+ dandiset (within species x region)")]:
        ax.barh(y, w, left=left, height=0.7, color=col, edgecolor=SURFACE, lw=1, label=lab)
    ax.set_yticks(y, [SHORT.get(f, f) for f in res.index])
    ax.set_xlabel("Share of feature variance explained (R²)")
    ax.set_xlim(0, max(0.8, res["dandiset"].max() + 0.05))
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Per-feature variance: species, region, then dataset", loc="left", color=INK)
    save(fig, out, "03_variance_partition")
    return res


def pair_auc(Xv: np.ndarray, ds: np.ndarray, a: str, b: str, rng, cap=500) -> float:
    ia, ib = np.flatnonzero(ds == a), np.flatnonzero(ds == b)
    ia = rng.choice(ia, min(cap, len(ia)), replace=False)
    ib = rng.choice(ib, min(cap, len(ib)), replace=False)
    Xp = np.vstack([Xv[ia], Xv[ib]])
    yp = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(LogisticRegression(max_iter=500, class_weight="balanced"),
                          StandardScaler().fit_transform(Xp), yp, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(yp, p)


def section_pair_auc(df, X, out, cache, refresh) -> pd.DataFrame:
    """Cross-validated AUC for telling every pair of datasets apart from their features.
    0.5 = indistinguishable. Linear model, so it is a lower bound on separability."""
    print("section 2: pairwise dataset separability")
    path = os.path.join(cache, "pair_auc.csv")
    meta = df.groupby("ds").agg(n=("ds", "size"), group=("species group", "first"),
                                species=("species", "first"), lab=("lab", "first"),
                                region=("region", "first"))
    meta = meta[meta["n"] >= 30]
    meta["lab_rank"] = meta["lab"].map({l: i for i, l in enumerate(LAB_ORDER)})
    meta = meta.sort_values(["group", "lab_rank", "region", "species"])
    order = list(meta.index)
    if os.path.exists(path) and not refresh:
        auc = pd.read_csv(path, index_col=0, dtype={0: str})
        auc.index = auc.index.astype(str).str.zfill(6)
        auc.columns = auc.columns.astype(str).str.zfill(6)
        if set(auc.index) == set(order):
            auc = auc.loc[order, order]
        else:
            auc = None
    else:
        auc = None
    if auc is None:
        rng = np.random.default_rng(SEED)
        Xv, ds = X.to_numpy(), df["ds"].to_numpy()
        auc = pd.DataFrame(np.nan, index=order, columns=order)
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                auc.loc[a, b] = auc.loc[b, a] = pair_auc(Xv, ds, a, b, rng)
        auc.to_csv(path)

    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    im = ax.imshow(auc.to_numpy(dtype=float), cmap=SEQUENTIAL, vmin=0.5, vmax=1.0)
    labels = [f"{d} {meta.loc[d, 'species'][:12]} {meta.loc[d, 'region'][:10]}" for d in order]
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(order)), labels, fontsize=5.5)
    for t, d in zip(ax.get_yticklabels(), order):
        t.set_color(LAB_COLORS[meta.loc[d, "lab"]])
    # outline same-species blocks
    groups = meta["group"].to_numpy()
    edges = np.flatnonzero(groups[1:] != groups[:-1]) + 0.5
    for e in edges:
        ax.axhline(e, color=INK, lw=0.8)
        ax.axvline(e, color=INK, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label("Cross-validated AUC (0.5 = can't tell apart)")
    ax.set_title("How well features alone tell two datasets apart  "
                 "(blocks = mouse / NHP / human; label colour = lab family)", loc="left", color=INK)
    save(fig, out, "04_pairwise_dataset_auc")
    return auc


def summarize_pair_auc(auc: pd.DataFrame, df: pd.DataFrame, out: str) -> pd.DataFrame:
    meta = df.groupby("ds").agg(group=("species group", "first"), species=("species", "first"),
                                lab=("lab", "first"), region=("region", "first"))
    rows = []
    for i, a in enumerate(auc.index):
        for b in auc.index[i + 1:]:
            ma, mb = meta.loc[a], meta.loc[b]
            same_sp = ma.species == mb.species
            same_lab = ma.lab == mb.lab and ma.lab != "Other labs"
            same_reg = ma.region == mb.region
            kind = ("same species, lab, region" if same_sp and same_lab and same_reg else
                    "same species & region, different lab" if same_sp and same_reg else
                    "same species, different region" if same_sp else "different species")
            rows.append((a, b, kind, auc.loc[a, b]))
    pairs = pd.DataFrame(rows, columns=["a", "b", "kind", "auc"])
    pairs.to_csv(os.path.join(out, "pair_auc_long.csv"), index=False)
    return pairs


def fig_pair_kinds(pairs: pd.DataFrame, out: str) -> None:
    kinds = ["same species, lab, region", "same species & region, different lab",
             "same species, different region", "different species"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    rng = np.random.default_rng(SEED)
    for i, k in enumerate(kinds):
        v = pairs.loc[pairs["kind"] == k, "auc"].to_numpy()
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=8, color=SLOTS[0], alpha=0.55, lw=0)
        ax.vlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
        ax.text(1.005, i, f"median {np.median(v):.2f}  (n={len(v)})", va="center", fontsize=7,
                color=INK_2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(kinds)), kinds)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Pairwise dataset AUC (0.5 = indistinguishable)")
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.set_title("Same-lab, same-region datasets are still easy to tell apart", loc="left", color=INK)
    save(fig, out, "05_pair_auc_by_kind")


# ---------------------------------------------------------------------------
# Section 3: feature-level drivers
# ---------------------------------------------------------------------------

def section_feature_drivers(df, X, out) -> dict:
    print("section 3: feature drivers")
    res = {}
    # dataset x feature median heatmap
    big = df.groupby("ds").size()
    big = big[big >= 30].index
    meta = df.groupby("ds").agg(group=("species group", "first"), lab=("lab", "first"),
                                region=("region", "first"), species=("species", "first"))
    meta = meta.loc[big]
    meta["lab_rank"] = meta["lab"].map({l: i for i, l in enumerate(LAB_ORDER)})
    meta = meta.sort_values(["group", "lab_rank", "region"])
    med = X.groupby(df["ds"].to_numpy()).median().loc[meta.index]
    med.to_csv(os.path.join(out, "dandiset_feature_medians.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 0.16 * len(med) + 1.6))
    im = ax.imshow(med.to_numpy(), cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(med.shape[1]), [SHORT.get(c, c) for c in med.columns], rotation=60, ha="right")
    ax.set_yticks(range(len(med)), [f"{d} {meta.loc[d, 'species'][:12]} {meta.loc[d, 'region'][:10]}"
                                    for d in med.index], fontsize=6)
    for t, d in zip(ax.get_yticklabels(), med.index):
        t.set_color(LAB_COLORS[meta.loc[d, "lab"]])
    for s in ax.spines.values():
        s.set_visible(False)
    groups = meta["group"].to_numpy()
    for e in np.flatnonzero(groups[1:] != groups[:-1]) + 0.5:
        ax.axhline(e, color=INK, lw=0.8)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("Dataset median (robust-scaled IQR units)")
    ax.set_title("Dataset medians per feature (label colour = lab family)", loc="left", color=INK)
    save(fig, out, "06_dandiset_feature_medians")

    # voltage offsets: do a dataset's four voltages move together (offset-like) or not?
    raw = df[VOLTAGE_FEATURES[:4]].copy()
    sp_med = raw.groupby(df["species group"].to_numpy()).transform("median")
    off = (raw - sp_med).groupby(df["ds"].to_numpy()).median().loc[meta.index]
    off.columns = ["V baseline", "AP1 threshold", "AP1 peak", "AP1 fast trough"]
    off["common shift"] = off[["V baseline", "AP1 threshold", "AP1 fast trough"]].median(axis=1)
    off["spread"] = off[["V baseline", "AP1 threshold", "AP1 fast trough"]].std(axis=1)
    off.to_csv(os.path.join(out, "voltage_offsets_per_dandiset.csv"))
    res["voltage offsets"] = off
    fig, ax = plt.subplots(figsize=(6.4, 0.16 * len(off) + 1.0))
    y = np.arange(len(off))
    for col, color, m in [("V baseline", SLOTS[0], "o"), ("AP1 threshold", SLOTS[1], "s"),
                          ("AP1 fast trough", SLOTS[2], "D"), ("AP1 peak", SLOTS[4], "^")]:
        ax.scatter(off[col], y, s=14, color=color, marker=m, label=col, lw=0, zorder=3)
    lo = off[["V baseline", "AP1 threshold", "AP1 fast trough", "AP1 peak"]].min(axis=1)
    hi = off[["V baseline", "AP1 threshold", "AP1 fast trough", "AP1 peak"]].max(axis=1)
    ax.hlines(y, lo, hi, color=CONTEXT, lw=2, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(y, [f"{d} {meta.loc[d, 'species'][:12]} {meta.loc[d, 'region'][:10]}" for d in off.index],
                  fontsize=6)
    for t, d in zip(ax.get_yticklabels(), off.index):
        t.set_color(LAB_COLORS[meta.loc[d, "lab"]])
    ax.invert_yaxis()
    ax.set_xlabel("Dataset median minus species-group median (mV)")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Voltage offsets: markers moving together look like an LJP/offset shift",
                 loc="left", color=INK)
    save(fig, out, "07_voltage_offsets")

    # redundancy of the ap_1 / ap_mean pairs
    pairs = [(f"ap_1_{f}_0_long_square", f"ap_mean_{f}_0_long_square")
             for f in ["threshold_v", "peak_v", "upstroke", "width", "fast_trough_v"]]
    res["ap1 vs apmean r"] = {SHORT[a].replace("AP1 ", ""): float(np.corrcoef(X[a], X[b])[0, 1])
                             for a, b in pairs if a in X and b in X}
    corr = X.corr()
    corr.to_csv(os.path.join(out, "feature_correlation.csv"))

    # protocol: step duration and step resolution vs rheobase firing
    prot = df.groupby("ds").agg(step_duration=("step duration", "median"),
                                step_gap_pA=("step gap", "median"),
                                overshoot=("overshoot", "median"),
                                rheobase_rate_Hz=("avg_rate_0_long_square", "median"),
                                rheobase_latency_s=("latency_0_long_square", "median"),
                                n=("ds", "size"))
    prot.to_csv(os.path.join(out, "protocol_per_dandiset.csv"))
    res["protocol"] = prot
    spikes = df["avg_rate_0_long_square"] * df["step duration"]
    res["single-spike rheobase share"] = float((spikes.round() <= 1).mean())
    res["rate vs 1/duration"] = df.groupby("step duration")["avg_rate_0_long_square"].median().to_dict()
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    ok = df["step duration"].isin([0.5, 0.6, 0.8, 1.0])
    durs = [0.5, 0.6, 0.8, 1.0]
    for ax, col, lab, log in [(axs[0], "avg_rate_0_long_square", "Rheobase-sweep rate (Hz)", True),
                              (axs[1], "latency_0_long_square", "Rheobase latency (s)", True)]:
        data = [df.loc[ok & (df["step duration"] == d), col].dropna().to_numpy() for d in durs]
        ax.boxplot(data, positions=range(len(durs)), widths=0.55, showfliers=False,
                   medianprops=dict(color=INK, lw=1.5), boxprops=dict(color=INK_2),
                   whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2))
        if col == "avg_rate_0_long_square":
            ax.scatter(range(len(durs)), [1 / d for d in durs], marker="_", s=200, color=SLOTS[1],
                       zorder=4, label="1 spike / step")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks(range(len(durs)), [f"{d:g} s\n(n={len(x):,})" for d, x in zip(durs, data)])
        ax.set_xlabel("Long-square step duration")
        ax.set_ylabel(lab)
        if log:
            ax.set_yscale("log")
    g = df.dropna(subset=["overshoot"])
    g = g[(g["overshoot"] > 0) & (g["overshoot"] < 3)]
    bins = pd.qcut(g["overshoot"], 6)
    m = g.groupby(bins, observed=True)["avg_rate_0_long_square"].median()
    axs[2].plot(range(len(m)), m.to_numpy(), marker="o", color=SLOTS[0], lw=2, ms=6)
    axs[2].set_xticks(range(len(m)), [f"{iv.left:.2f}-\n{iv.right:.2f}" for iv in m.index], fontsize=6.5)
    axs[2].set_xlabel("Step gap / rheobase (sextiles)")
    axs[2].set_ylabel("Median rheobase-sweep rate (Hz)")
    for ax, t in zip(axs, ["Rate at rheobase tracks 1/step duration", "Latency by step duration",
                           "Coarser steps overshoot rheobase"]):
        ax.set_title(t, loc="left", color=INK, fontsize=8)
    fig.tight_layout()
    save(fig, out, "08_protocol_effects")

    # imputation: missing share per dataset x feature
    nan = df[list(config.COLS_TO_KEEP)].replace([np.inf, -np.inf], np.nan).isna()
    miss = nan.groupby(df["ds"].to_numpy()).mean().loc[meta.index]
    miss.to_csv(os.path.join(out, "missing_share_per_dandiset.csv"))
    res["imputed cells"] = int((nan.sum(axis=1) > 0).sum())
    res["datasets fully missing a feature"] = {
        d: [SHORT[c] for c in miss.columns[miss.loc[d] >= 0.99]] for d in miss.index
        if (miss.loc[d] >= 0.99).any()}
    df["n imputed"] = nan.sum(axis=1).to_numpy()
    return res


def read_acquisition(path: str) -> dict | None:
    """Sampling rate and amplifier settings of a file's current-clamp sweeps."""
    import h5py
    try:
        with h5py.File(path, "r") as h:
            rates, bridge, capcomp = [], [], []
            for g in h["acquisition"].values():
                if g.attrs.get("neurodata_type") not in ("CurrentClampSeries", b"CurrentClampSeries"):
                    continue
                st = g.get("starting_time")
                if st is not None and "rate" in st.attrs:
                    rates.append(float(st.attrs["rate"]))
                for key, lst in (("bridge_balance", bridge), ("capacitance_compensation", capcomp)):
                    if key in g:
                        lst.append(float(np.ravel(g[key][()])[0]))
            if not rates:
                return None
            return {"rate_kHz": float(np.median(rates)) / 1000,
                    "bridge_MOhm": float(np.nanmedian(bridge)) / 1e6 if bridge else np.nan,
                    "capcomp_pF": float(np.nanmedian(capcomp)) * 1e12 if capcomp else np.nan}
    except Exception as e:  # unreadable or missing file: skip it
        print(f"    could not read {path}: {e}")
        return None


def section_acquisition(df, out, cache, refresh, per_ds=6) -> pd.DataFrame:
    """Sample a few raw NWB files per dataset for sampling rate / bridge / cap. compensation,
    which shape AP peak, upstroke and width but are not in the feature CSV."""
    print("section 3: acquisition settings from raw NWB")
    path = os.path.join(cache, "acquisition.csv")
    if os.path.exists(path) and not refresh:
        acq = pd.read_csv(path, index_col=0)
        acq.index = acq.index.astype(str).str.zfill(6)
    else:
        rng = np.random.default_rng(SEED)
        rows = []
        for d, g in df.groupby("ds"):
            for sid in rng.choice(g["specimen_id"].to_numpy(), min(per_ds, len(g)), replace=False):
                r = read_acquisition(os.path.join(config.CONFIG.cache_dir, sid))
                if r:
                    rows.append({"ds": d, **r})
        acq = pd.DataFrame(rows).groupby("ds").median()
        acq.to_csv(path)
    feats = X_medians_raw(df)
    acq = acq.join(feats)
    acq.to_csv(os.path.join(out, "acquisition_per_dandiset.csv"))
    return acq


def X_medians_raw(df):
    cols = {"ap_1_upstroke_0_long_square": "AP1 upstroke (mV/ms)", "ap_1_width_0_long_square": "AP1 width (ms)",
            "ap_1_peak_v_0_long_square": "AP1 peak (mV)", "lab": "lab", "species": "species"}
    med = df.groupby("ds").agg({k: ("first" if k in ("lab", "species") else "median") for k in cols})
    med["ap_1_width_0_long_square"] *= 1000
    return med.rename(columns=cols)


def section_ablation(df, X, Xrel, X_fi, out) -> pd.DataFrame:
    """Within-stratum batch score in 16-D feature space with each feature group removed."""
    print("section 3: feature-group ablation")
    rows = {}

    def run(name, M):
        idx = knn(M, K_STRATUM, "cosine")
        ws = within_stratum_self(idx, df)
        sp = same_frac(idx[:, :K], df["species"].to_numpy()).mean() / expected_same(df["species"].to_numpy()).mean()
        rows[name] = {"within-stratum enrichment": ws["obs"].mean() / ws["exp"].mean(),
                      "species purity enrichment": sp, "n features": M.shape[1]}

    run(f"all {X.shape[1]} features", X.to_numpy())
    for g, cols in FEATURE_GROUPS.items():
        run(f"without {g}", X.drop(columns=[c for c in cols if c in X]).to_numpy())
    run("relative voltages (keep V baseline)", Xrel.to_numpy())
    run("relative voltages, no V baseline", Xrel.drop(columns=["v_baseline"]).to_numpy())
    if any(c.startswith("ap_mean") for c in X.columns):
        run("only ap_1 (drop ap_mean)", X.drop(columns=[c for c in X.columns if c.startswith("ap_mean")]).to_numpy())
        run("only ap_mean (drop ap_1)", X.drop(columns=[c for c in X.columns if c.startswith("ap_1")]).to_numpy())
    if "avg_rate_0_long_square" in X.columns:
        run("f-I slope instead of rheobase rate", X_fi.to_numpy())
    res = pd.DataFrame(rows).T
    res.to_csv(os.path.join(out, "feature_ablation.csv"))

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(res))[::-1]
    base = res.iloc[0]["within-stratum enrichment"]
    ax.barh(y, res["within-stratum enrichment"], height=0.6,
            color=[INK_2] + [SLOTS[0]] * (len(res) - 1), edgecolor=SURFACE)
    for yi, (n, r) in zip(y, res.iterrows()):
        ax.text(r["within-stratum enrichment"] + 0.03, yi,
                f"{r['within-stratum enrichment']:.2f}  (species {r['species purity enrichment']:.2f})",
                va="center", fontsize=6.5, color=INK_2)
    ax.axvline(base, color=MUTED, lw=0.8, ls=":")
    ax.axvline(1, color=MUTED, lw=0.8)
    ax.set_yticks(y, res.index)
    ax.set_xlabel("Within-stratum same-dataset neighbour enrichment, 16-D space (1 = mixed)")
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.set_xlim(0, res["within-stratum enrichment"].max() * 1.35)
    ax.set_title("Which features carry the batch signal (no UMAP involved)", loc="left", color=INK)
    save(fig, out, "09_feature_ablation")
    return res


def section_temperature(df, out) -> pd.DataFrame:
    """000008 (room temperature) vs 000035 (temperature-controlled): same lab, region, protocol."""
    a, b = df[df["ds"] == "000008"], df[df["ds"] == "000035"]
    cols = list(config.COLS_TO_KEEP)
    res = pd.DataFrame({"000008 median": a[cols].median(), "000035 median": b[cols].median()})
    res["ratio 35/8"] = res["000035 median"] / res["000008 median"]
    res.index = [SHORT[c] for c in res.index]
    res.to_csv(os.path.join(out, "temperature_000008_vs_000035.csv"))
    return res


# ---------------------------------------------------------------------------
# Section 4: islands and clusters
# ---------------------------------------------------------------------------

def section_islands(df, X, out) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("section 4: islands")
    E = df[["umap X", "umap Y"]].to_numpy()
    lab = HDBSCAN(min_cluster_size=60, min_samples=20).fit_predict(E)
    df["island"] = lab
    rows = []
    zall = X
    for c in sorted(set(lab) - {-1}):
        m = lab == c
        g = df.loc[m]
        comp = g["ds"].value_counts(normalize=True)
        shift = (zall[m].median() - zall[~m].median()).sort_values(key=np.abs, ascending=False)
        rows.append({
            "island": c, "n": int(m.sum()),
            "x": float(np.median(E[m, 0])), "y": float(np.median(E[m, 1])),
            "top datasets": "; ".join(f"{d} {p:.0%}" for d, p in comp.head(3).items()),
            "top dataset share": float(comp.iloc[0]),
            "n datasets >=5%": int((comp >= 0.05).sum()),
            "species": "; ".join(f"{s} {p:.0%}" for s, p in g["species"].value_counts(normalize=True).head(2).items()),
            "regions": "; ".join(f"{s} {p:.0%}" for s, p in g["region"].value_counts(normalize=True).head(2).items()),
            "FS share": float(g["fast spiking"].mean()),
            "mean imputed features": float(g["n imputed"].mean()),
            "distinguishing features": "; ".join(f"{SHORT[f]} {v:+.1f}" for f, v in shift.head(4).items()),
        })
    isl = pd.DataFrame(rows).sort_values("top dataset share", ascending=False)
    isl.to_csv(os.path.join(out, "islands.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(E[:, 0], E[:, 1], c=CONTEXT, s=1.2, lw=0, rasterized=True)
    for _, r in isl.iterrows():
        m = lab == r["island"]
        mono = r["top dataset share"] >= 0.8
        ax.scatter(E[m, 0], E[m, 1], s=1.5, lw=0, rasterized=True,
                   c=SLOTS[7] if mono else SLOTS[0])
        ax.text(r["x"], r["y"], str(int(r["island"])), fontsize=7, fontweight="bold", color=INK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, mfc=SLOTS[7], mec="none",
                          label=">=80% one dataset"),
               plt.Line2D([], [], ls="", marker="o", ms=5, mfc=SLOTS[0], mec="none",
                          label="mixed datasets")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.06), fontsize=7, ncol=2)
    umap_axes(ax)
    ax.set_title("Dense regions of the site UMAP (HDBSCAN), numbered as in the report", loc="left", color=INK)
    save(fig, out, "10_islands")

    cl = {}
    for col in ["coarse cluster", "fine cluster", "GMM cluster label"]:
        lbl = df[col].astype(str)
        cl[col] = {"AMI vs dandiset": adjusted_mutual_info_score(df["ds"], lbl),
                   "AMI vs species": adjusted_mutual_info_score(df["species"], lbl),
                   "AMI vs stratum": adjusted_mutual_info_score(df["stratum"], lbl),
                   "clusters >80% one dataset": int(
                       (pd.crosstab(lbl, df["ds"], normalize="index").max(axis=1) > 0.8).sum()),
                   "n clusters": int(lbl.nunique())}
    cl = pd.DataFrame(cl).T
    cl.to_csv(os.path.join(out, "clusters_vs_dandiset.csv"))
    return isl, cl


# ---------------------------------------------------------------------------
# Section 5: light-touch variants
# ---------------------------------------------------------------------------

def center_by_dataset(X: pd.DataFrame, df: pd.DataFrame, cols) -> pd.DataFrame:
    X = X.copy()
    n = df.groupby("ds")["ds"].transform("size").to_numpy()
    med = X[cols].groupby(df["ds"].to_numpy()).transform("median")
    X.loc[n >= 10, cols] = (X[cols] - med).loc[n >= 10]
    return X


def section_variants(df, X, Xrel, X_fi, Xrel_fi, cache, refresh, out) -> pd.DataFrame:
    print("section 5: variants")
    no_mean = [c for c in X.columns if not c.startswith("ap_mean")]
    rel_no_mean = [c for c in Xrel.columns if not c.startswith("ap_mean")]
    # Upstroke and width track sampling rate across datasets; halve their weight.
    Xrel_half = Xrel[rel_no_mean].copy()
    kin = [c for c in FEATURE_GROUPS["AP kinetics (upstroke, width)"] if c in Xrel_half.columns]
    Xrel_half[kin] *= 0.5
    variants = {
        "A. current site params": (X, SITE_PARAMS),
        "A2. current site params, seed 1": (X, {**SITE_PARAMS, "random_state": 1}),
        "A3. current site params, seed 2": (X, {**SITE_PARAMS, "random_state": 2}),
        "B. euclidean metric": (X, {**SITE_PARAMS, "metric": "euclidean"}),
        "C. n_neighbors 30": (X, {**SITE_PARAMS, "n_neighbors": 30}),
        "D. repulsion 5 (previous site setting)": (X, {**SITE_PARAMS, "repulsion_strength": 5}),
        "E. UMAP defaults (pre-change)": (X, {}),
        "F. relative voltages": (Xrel, SITE_PARAMS),
        "G. relative voltages, no V baseline": (Xrel.drop(columns=["v_baseline"]), SITE_PARAMS),
        "H. drop ap_mean set": (X[no_mean], SITE_PARAMS),
        "I. relative voltages + drop ap_mean": (Xrel[rel_no_mean], SITE_PARAMS),
        "J. I + euclidean, repulsion 1": (Xrel[rel_no_mean], {**SITE_PARAMS, "metric": "euclidean",
                                                               "repulsion_strength": 1}),
        "K. f-I slope instead of rheobase rate": (X_fi, SITE_PARAMS),
        "L. I + f-I slope instead of rheobase rate": (Xrel_fi[[c for c in Xrel_fi.columns
                                                            if not c.startswith("ap_mean")]], SITE_PARAMS),
        "M. I + AP kinetics at half weight": (Xrel_half, SITE_PARAMS),
        "Z1. centre voltages per dataset (bound)": (
            center_by_dataset(X, df, [c for c in VOLTAGE_FEATURES if c in X]), SITE_PARAMS),
        "Z2. centre all features per dataset (bound)": (center_by_dataset(X, df, list(X.columns)), SITE_PARAMS),
    }
    rows, embs, seen = {}, {}, set()
    for name, (Xi, params) in variants.items():
        # Variants that the current config already applies would just repeat an earlier fit.
        sig = (Xi.to_numpy().tobytes(), json.dumps(params, sort_keys=True))
        if sig in seen:
            print(f"  skipping {name}: same input and params as an earlier variant")
            continue
        seen.add(sig)
        key = name.split(".")[0]
        E = fit_umap(f"variant_{key}", Xi.to_numpy(), cache, refresh, **params)
        embs[name] = E
        print(f"  scoring {name}")
        # trustworthiness against the current pipeline's feature space, so feature swaps can't game it
        rows[name] = score_space(E, df, X.to_numpy())
    res = pd.DataFrame(rows).T
    res.to_csv(os.path.join(out, "scores_variants.csv"))
    fig_variant_tradeoff(res, out)
    fig_variant_grid(df, embs, res, out)
    return res


def fig_variant_tradeoff(res, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    bio = res[["species purity enrichment", "region purity enrichment", "FS purity enrichment"]]
    bio = bio / bio.loc["A. current site params"]
    res = res.assign(biology=bio.mean(axis=1))
    for ax, ycol, ylab in [(axs[0], "biology", "Biology kept, relative to current\n"
                                               "(mean of species, region, FS purity)"),
                           (axs[1], "control-pair mixing (mean)", "Same-lab control pairs mixing\n"
                                                                    "(1 = as mixed as chance)")]:
        for name, r in res.iterrows():
            bound = name.startswith("Z")
            cur = name.startswith("A")
            ax.scatter(r["within-stratum enrichment"], r[ycol], s=60 if cur else 40,
                       c=INK if cur else (MUTED if bound else SLOTS[0]), edgecolors=SURFACE, lw=1.5,
                       zorder=3, marker="D" if bound else "o")
            ax.annotate(name.split(".")[0], (r["within-stratum enrichment"], r[ycol]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xlabel("Within-stratum same-dataset enrichment (lower = better mixing)")
        ax.set_ylabel(ylab)
        ax.invert_xaxis()
    axs[0].set_title("Batch mixing vs biology kept (upper right = better)", loc="left", color=INK)
    axs[1].set_title("Positive controls: same-lab datasets that should mix", loc="left", color=INK)
    fig.text(0.01, -0.06, "  ".join(res.index), fontsize=6, color=INK_2, wrap=True)
    fig.tight_layout()
    save(fig, out, "11_variant_tradeoff")


def fig_variant_grid(df, embs, res, out):
    n = len(embs)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.3))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    c = df["species group"].map(SPECIES_GROUP_COLORS).fillna(CONTEXT).to_numpy()[order]
    for ax, (name, E) in zip(axs.flat, embs.items()):
        ax.scatter(E[order, 0], E[order, 1], c=c, s=0.6, lw=0, rasterized=True)
        umap_axes(ax)
        r = res.loc[name]
        ax.set_title(f"{name}\nbatch {r['within-stratum enrichment']:.2f} · species "
                     f"{r['species purity enrichment']:.2f} · ctrl mix {r['control-pair mixing (mean)']:.2f}",
                     loc="left", fontsize=6.5, color=INK)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, mfc=v, mec="none", label=k)
               for k, v in SPECIES_GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    save(fig, out, "12_variant_grid")


def fig_variant_by_dataset(df, E, name, out, fname):
    """Small multiples: each control dataset highlighted on an embedding."""
    sets = ["000020", "001455", "000008", "000035", "001938", "001348", "000636", "000630"]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, d in zip(axs.flat, sets):
        m = (df["ds"] == d).to_numpy()
        ax.scatter(E[~m, 0], E[~m, 1], c=CONTEXT, s=0.6, lw=0, rasterized=True)
        ax.scatter(E[m, 0], E[m, 1], c=SLOTS[0], s=1.4, lw=0, rasterized=True)
        r = df.loc[m].iloc[0]
        ax.set_title(f"{d} {r['species']} {r['region']}\n{r['lab']} (n={m.sum()})", loc="left",
                     fontsize=7, color=INK)
        umap_axes(ax)
    fig.suptitle(name, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, out, fname)


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=config.CONFIG.merged_csv)
    p.add_argument("--out", default="figures/umap_workup")
    p.add_argument("--refresh", action="store_true", help="Refit every UMAP and recompute pair AUCs.")
    p.add_argument("--skip-variants", action="store_true")
    args = p.parse_args(argv)
    out, cache = args.out, os.path.join(args.out, "cache")
    os.makedirs(cache, exist_ok=True)

    df = load_cells(args.csv)
    cols = list(config.COLS_TO_KEEP)
    X = pipeline_features(df, cols)
    df = add_relative_voltages(df)
    prefixes = [p for p in ("ap_1", "ap_mean") if f"{p}_threshold_v_0_long_square" in cols]
    rel_cols = [c for c in cols if c not in VOLTAGE_FEATURES or c == "v_baseline"] + [
        f"{p}_{s}" for p in prefixes for s in ("thr_rel", "amp", "ahp")]
    Xrel = pipeline_features(df, rel_cols)
    # f-I slope (Hz/pA) is duration-normalised; the rheobase-sweep rate is mostly 1 spike / step length.
    df["fi_slope_log"] = np.log10(df["fi_linear_fit_slope"].clip(lower=0.005))
    SHORT["fi_slope_log"] = "f-I slope (log)"
    swap = lambda cs: [("fi_slope_log" if c == "avg_rate_0_long_square" else c) for c in cs]  # noqa: E731
    X_fi = pipeline_features(df, swap(cols))
    Xrel_fi = pipeline_features(df, swap(rel_cols))

    summary = {"n cells": len(df), "n dandisets": int(df["ds"].nunique()),
               "FS share": float(df["fast spiking"].mean())}
    spaces = section_spaces(df, X, cache, args.refresh, out)
    var = section_variance(df, X, out)
    auc = section_pair_auc(df, X, out, cache, args.refresh)
    pairs = summarize_pair_auc(auc, df, out)
    fig_pair_kinds(pairs, out)
    summary["pair AUC median by kind"] = pairs.groupby("kind")["auc"].median().to_dict()
    drivers = section_feature_drivers(df, X, out)
    abl = section_ablation(df, X, Xrel, X_fi, out)
    temp = section_temperature(df, out)
    acq = section_acquisition(df, out, cache, args.refresh)
    isl, cl = section_islands(df, X, out)
    fig_variant_by_dataset(df, df[["umap X", "umap Y"]].to_numpy(), "Stored site UMAP", out,
                           "13_datasets_on_site_umap")
    summary.update({k: v for k, v in drivers.items() if not isinstance(v, pd.DataFrame)})
    if not args.skip_variants:
        var_res = section_variants(df, X, Xrel, X_fi, Xrel_fi, cache, args.refresh, out)
        best = "I. relative voltages + drop ap_mean"
        E = fit_umap("variant_I", Xrel[[c for c in Xrel.columns if not c.startswith("ap_mean")]].to_numpy(),
                     cache, False, **SITE_PARAMS)
        fig_variant_by_dataset(df, E, best, out, "14_datasets_on_variant_I")

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n== spaces ==\n", spaces.round(3).T)
    print("\n== variance ==\n", var.round(3))
    print("\n== ablation ==\n", abl.round(3))
    print("\n== temperature ==\n", temp.round(3))
    print("\n== acquisition ==\n", acq.round(2).to_string())
    print("\n== islands ==\n", isl.to_string())
    print("\n== clusters ==\n", cl.round(3))
    if not args.skip_variants:
        print("\n== variants ==\n", var_res.round(3).T)
    print("\n== summary ==\n", json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
