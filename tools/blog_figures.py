"""Blog figures: species, clusters, features and brain regions on the UMAP, feature
summaries per cluster, and example cells per cluster.

Every panel is drawn as its own figure and saved as a standalone SVG (text kept as
text) so panels can be combined in Illustrator. Grids of related panels are saved
too, as SVG and PNG, for a quick look at the whole set.

    python tools/blog_figures.py --out figures/blog [--no-refresh] [--n-examples 3]
    python tools/blog_figures.py --only features,within,regions

Example traces are re-read from the raw NWB files under ``CONFIG.cache_dir`` and
cached in ``<out>/cache``; ``--no-refresh`` reuses that cache so restyling reruns
don't touch the disk again. Sections that don't need traces never read it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dandi_scraper import config  # noqa: E402
from dandi_scraper.analysis import scale_features  # noqa: E402

logger = logging.getLogger("blog_figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
CONTEXT = "#e1e0d9"  # background cells in highlight panels
SURFACE = "#fcfcfb"

# Reference categorical palette, light mode (dataviz skill palette.md), in slot order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# No 6-colour subset of the palette clears the all-pairs scatter floors; this one is
# the most separable (normal-vision dE 13.2, CVD dE 6.1), so cluster maps always carry
# direct number labels and have per-cluster small multiples alongside.
CLUSTER_COLORS = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#e87ba4", 5: "#008300", 6: "#e34948"}

SPECIES_ORDER = ["House mouse", "Human", "Pigtail macaque", "Cynomolgus macaque",
                 "Common marmoset", "Rhesus macaque"]
SPECIES_COLORS = dict(zip(SPECIES_ORDER, SLOTS))  # adjacent pairs only (stacked bars)
NHP = {"Pigtail macaque", "Cynomolgus macaque", "Common marmoset", "Rhesus macaque"}
SPECIES_GROUP_COLORS = {"House mouse": SLOTS[0], "Human": SLOTS[1], "Non-human primate": SLOTS[2]}

# The 8 small regions get palette slots, which can't all pass the scatter colour check
# (worst pair dE 7.1), so the overview labels them directly and each has its own panel.
REGION_COLORS = {
    "Amygdala": SLOTS[0], "Basal forebrain": SLOTS[1], "Brainstem": SLOTS[2], "Spinal cord": SLOTS[3],
    "Midbrain": SLOTS[4], "Striatum": SLOTS[5], "Hippocampus": SLOTS[6], "Pallidum": SLOTS[7],
}
NEOCORTEX_COLOR = "#c3c2b7"
REGION_HIGHLIGHT = {**REGION_COLORS, "Neocortex": INK_2, "Unknown": MUTED}
SUBREGION_LABELS = {
    "visual cortex": "Visual cortex", "motor cortex": "Motor cortex", "anterior insula": "Anterior insula",
    "temporal cortex": "Temporal cortex", "L1": "Layer 1", "premotor cortex": "Premotor cortex",
}

# One-hue palette ramp (steps 150-700) for feature values. 32 levels keeps matplotlib's
# colourbar as vector rectangles; it rasterizes from 50 levels up.
SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    "blue_ramp", ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                  "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"], N=32)
DIVERGING = mcolors.LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"], N=32)
HEATMAP_LIMIT = 2.0  # IQR units from the population median

# col: (label, units, factor to display units). Log features in config.SCALE_FEATURES
# already land in display units, so their factor only applies to linear features.
FEATURE_LABELS = {
    "input_resistance": ("Input resistance", "M\u03a9", 1),
    "tau": ("Membrane tau", "ms", 1),
    "v_baseline": ("Baseline voltage", "mV", 1),
    "sag_nearest_minus_100": ("Sag ratio", "", 1),
    "ap_1_threshold_v_0_long_square": ("AP threshold", "mV", 1),
    "ap_1_peak_v_0_long_square": ("AP peak", "mV", 1),
    "ap_1_upstroke_0_long_square": ("AP upstroke", "V/s", 1),
    "ap_1_width_0_long_square": ("AP width", "ms", 1),
    "ap_1_fast_trough_v_0_long_square": ("AP fast trough", "mV", 1),
    "fi_linear_fit_slope": ("F-I slope", "Hz/pA", 1),
    "latency_0_long_square": ("First-spike latency", "ms", 1000),
}

UMAP_SIZE = (3.5, 3.5)
MAP_SIZE = (3.5, 3.1)  # fixed canvas for panels that must stack exactly in Illustrator
MAP_RECT = (0.06, 0.07, 0.90, 0.83)
TRACE_SIZE = (2.2, 1.6)
DOT = 2.0

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.dpi": 300,  # only affects artists marked rasterized in the SVGs
})


def shade(hex_color: str, amount: float) -> str:
    """Mix ``hex_color`` toward black (``amount`` < 0) or white (> 0); |amount| <= 1."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    target = 1.0 if amount > 0 else 0.0
    return mcolors.to_hex(rgb + (target - rgb) * abs(amount))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save(fig, out_dir: str, name: str, png: bool = False, tight: bool = True) -> None:
    path = os.path.join(out_dir, f"{name}.svg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    else:  # keep the full canvas so panels stack exactly
        fig.savefig(path)
    if png:
        fig.savefig(os.path.join(out_dir, f"{name}_preview.png"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_cells(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["umap X", "umap Y", "coarse cluster"])
    df["coarse cluster"] = df["coarse cluster"].astype(int)
    df["fine cluster"] = df["fine cluster"].astype(str)
    df["species group"] = np.where(df["species"].isin(NHP), "Non-human primate", df["species"])
    return df.reset_index(drop=True)


def scaled_features(df: pd.DataFrame) -> np.ndarray:
    """The feature matrix the clustering ran on, rebuilt as in database.run_merge_dandiset."""
    X = scale_features(df[config.COLS_TO_KEEP].copy())
    X.loc[:, :] = KNNImputer(keep_empty_features=True).fit_transform(X.values)
    X = X.dropna(axis=1, how="any")
    return np.clip(RobustScaler().fit_transform(X.values), -config.ROBUST_CLIP, config.ROBUST_CLIP)


# ---------------------------------------------------------------------------
# UMAP panels
# ---------------------------------------------------------------------------

def umap_axes(ax) -> None:
    """Strip the axes and draw a small UMAP 1 / UMAP 2 corner arrow."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    kw = dict(xycoords="axes fraction", textcoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", lw=0.6, color=MUTED, shrinkA=0, shrinkB=0,
                              mutation_scale=5))
    ax.annotate("", xy=(0.13, 0.0), xytext=(0.0, 0.0), **kw)
    ax.annotate("", xy=(0.0, 0.13), xytext=(0.0, 0.0), **kw)
    ax.text(0.015, -0.035, "UMAP 1", transform=ax.transAxes, color=MUTED, fontsize=6, va="top")
    ax.text(-0.035, 0.015, "UMAP 2", transform=ax.transAxes, color=MUTED, fontsize=6,
            rotation=90, ha="right")


def scatter_categories(ax, df: pd.DataFrame, colors: pd.Series, rng) -> None:
    order = rng.permutation(len(df))
    ax.scatter(df["umap X"].to_numpy()[order], df["umap Y"].to_numpy()[order],
               c=colors.to_numpy()[order], s=DOT, linewidths=0, rasterized=False)
    umap_axes(ax)


def scatter_highlight(ax, df: pd.DataFrame, mask: np.ndarray, color: str, title: str) -> None:
    ax.scatter(df.loc[~mask, "umap X"], df.loc[~mask, "umap Y"], c=CONTEXT, s=DOT, linewidths=0)
    ax.scatter(df.loc[mask, "umap X"], df.loc[mask, "umap Y"], c=color, s=DOT, linewidths=0)
    ax.set_title(title, loc="left", color=INK)
    umap_axes(ax)


def legend(ax, labels_colors, title=None, loc="upper left", anchor=(1.0, 1.0)) -> None:
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, mfc=c, mec="none", label=l)
               for l, c in labels_colors]
    lg = ax.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=anchor,
                   handletextpad=0.3, labelspacing=0.4, borderaxespad=0, fontsize=7)
    if title:
        lg.get_title().set_fontsize(7)
        lg.get_title().set_color(INK_2)


def label_clusters(ax, df: pd.DataFrame, bins: int = 40) -> None:
    """Number each cluster on its main mass. Clusters are fit in feature space and can have
    several lobes on the UMAP, so a median can fall outside all of them."""
    xe = np.linspace(df["umap X"].min(), df["umap X"].max(), bins + 1)
    ye = np.linspace(df["umap Y"].min(), df["umap Y"].max(), bins + 1)
    for c, g in df.groupby("coarse cluster"):
        x, y = densest_point(g, xe, ye)
        ax.text(x, y, str(c), fontsize=9, fontweight="bold", color=INK, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=CLUSTER_COLORS[c], lw=1.2))


def fig_species(df: pd.DataFrame, out: str, rng) -> None:
    print("species panels")
    fig, ax = plt.subplots(figsize=UMAP_SIZE)
    scatter_categories(ax, df, df["species group"].map(SPECIES_GROUP_COLORS).fillna(CONTEXT), rng)  # 22 cells lack species
    counts = df["species group"].value_counts()
    legend(ax, [(f"{g} (n={counts.get(g, 0):,})", c) for g, c in SPECIES_GROUP_COLORS.items()],
           loc="upper left", anchor=(0.0, 1.08))
    save(fig, out, "umap_species_overview")

    counts = df["species"].value_counts()
    for sp in SPECIES_ORDER:
        fig, ax = plt.subplots(figsize=UMAP_SIZE)
        scatter_highlight(ax, df, (df["species"] == sp).to_numpy(), SLOTS[0], f"{sp}  (n={counts[sp]:,})")
        save(fig, out, f"species/umap_species_{slug(sp)}")

    fig, axs = plt.subplots(2, 3, figsize=(UMAP_SIZE[0] * 3, UMAP_SIZE[1] * 2))
    for ax, sp in zip(axs.flat, SPECIES_ORDER):
        scatter_highlight(ax, df, (df["species"] == sp).to_numpy(), SLOTS[0], f"{sp}  (n={counts[sp]:,})")
    fig.tight_layout()
    save(fig, out, "umap_species_grid", png=True)


def fig_species_by_cluster(df: pd.DataFrame, out: str) -> None:
    print("species x cluster bars")
    tab = pd.crosstab(df["coarse cluster"], df["species"]).reindex(columns=SPECIES_ORDER, fill_value=0)
    frac = tab.div(tab.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    clusters = list(tab.index)[::-1]  # cluster 1 on top
    y = np.arange(len(clusters))
    left = np.zeros(len(clusters))
    for sp in SPECIES_ORDER:
        w = frac.loc[clusters, sp].to_numpy()
        ax.barh(y, w, left=left, height=0.72, color=SPECIES_COLORS[sp], edgecolor=SURFACE,
                linewidth=1.0, label=sp)
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= 0.10:
                txt = INK if SPECIES_COLORS[sp] in ("#eda100",) else "white"
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=6, color=txt)
        left += w
    ax.set_yticks(y, [f"Cluster {c}  (n={tab.loc[c].sum():,})" for c in clusters])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of cells in cluster")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5, handlelength=1.0,
              handleheight=1.0, borderaxespad=0)
    save(fig, out, "species_by_cluster", png=True)


def fig_clusters(df: pd.DataFrame, exemplars: pd.DataFrame | None, out: str, rng) -> None:
    print("cluster panels")
    counts = df["coarse cluster"].value_counts()
    items = [(f"Cluster {c} (n={counts[c]:,})", CLUSTER_COLORS[c]) for c in sorted(CLUSTER_COLORS)]

    def coarse_map(with_exemplars: bool):
        fig, ax = plt.subplots(figsize=UMAP_SIZE)
        scatter_categories(ax, df, df["coarse cluster"].map(CLUSTER_COLORS), rng)
        label_clusters(ax, df)
        if with_exemplars and exemplars is not None:
            ax.scatter(exemplars["umap X"], exemplars["umap Y"], s=28, facecolors="none",
                       edgecolors=INK, linewidths=0.8)
            for _, r in exemplars.iterrows():
                ax.annotate(r["tag"], (r["umap X"], r["umap Y"]), xytext=(3, 3),
                            textcoords="offset points", fontsize=6, color=INK)
        legend(ax, items)
        return fig

    save(coarse_map(False), out, "umap_coarse_clusters", png=True)
    if exemplars is not None:
        save(coarse_map(True), out, "umap_coarse_clusters_exemplars", png=True)

    for c in sorted(CLUSTER_COLORS):
        fig, ax = plt.subplots(figsize=UMAP_SIZE)
        scatter_highlight(ax, df, (df["coarse cluster"] == c).to_numpy(), CLUSTER_COLORS[c],
                          f"Cluster {c}  (n={counts[c]:,})")
        save(fig, out, f"clusters/umap_cluster_{c}")
    fig, axs = plt.subplots(2, 3, figsize=(UMAP_SIZE[0] * 3, UMAP_SIZE[1] * 2))
    for ax, c in zip(axs.flat, sorted(CLUSTER_COLORS)):
        scatter_highlight(ax, df, (df["coarse cluster"] == c).to_numpy(), CLUSTER_COLORS[c],
                          f"Cluster {c}  (n={counts[c]:,})")
    fig.tight_layout()
    save(fig, out, "umap_cluster_grid", png=True)

    # Fine clusters: lightness steps of their coarse cluster's hue.
    fine_colors = {}
    for c in sorted(CLUSTER_COLORS):
        subs = sorted(df.loc[df["coarse cluster"] == c, "fine cluster"].unique(),
                      key=lambda s: int(s.split(".")[1]))
        for k, s in enumerate(subs):
            fine_colors[s] = shade(CLUSTER_COLORS[c], np.linspace(-0.45, 0.45, len(subs))[k]
                                   if len(subs) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=UMAP_SIZE)
    scatter_categories(ax, df, df["fine cluster"].map(fine_colors), rng)
    label_clusters(ax, df)
    n_fine = df["fine cluster"].nunique()
    ax.set_title(f"{n_fine} fine clusters, shaded within their coarse cluster", loc="left", color=INK)
    save(fig, out, "umap_fine_clusters", png=True)


# ---------------------------------------------------------------------------
# Example cells
# ---------------------------------------------------------------------------

def parse_sweeps(analyzed: str) -> dict[int, float]:
    if not isinstance(analyzed, str) or not analyzed:
        return {}
    pairs = (x.split(":") for x in analyzed.split(";"))
    return {int(s): float(a) for s, a in pairs}


def pick_sweeps(sweeps: dict[int, float], rheobase: float) -> dict[str, tuple[int, float]] | None:
    neg = {s: a for s, a in sweeps.items() if a < 0}
    pos = {s: a for s, a in sweeps.items() if a > 0}
    if not neg or not pos or not np.isfinite(rheobase):
        return None
    closest = lambda d, x: min(d.items(), key=lambda kv: abs(kv[1] - x))  # noqa: E731
    rheo = closest(pos, rheobase)
    strong = closest(pos, max(2 * rheobase, rheobase + 100))
    if strong[1] <= rheo[1]:
        strong = max(pos.items(), key=lambda kv: kv[1])
    if strong[1] <= rheo[1]:
        return None
    return {"hyper": closest(neg, -100), "rheo": rheo, "strong": strong}


def minmax_decimate(t, v, n_bins=3000):
    """Keep the min and max of each bin so spikes survive downsampling."""
    if len(t) <= 2 * n_bins:
        return t, v
    edges = np.linspace(0, len(t), n_bins + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = v[a:b]
        lo, hi = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        idx.extend(sorted((lo, hi)))
    idx = np.array(idx)
    return t[idx], v[idx]


def extract_traces(row: pd.Series, picks: dict) -> dict | None:
    from gigaseal.database.protocol_detect import _iter_current_clamp_traces, get_stim_characteristics
    from gigaseal.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES

    GLOBAL_STIM_NAMES.stim_inc = [""]
    GLOBAL_STIM_NAMES.stim_exc = []
    path = os.path.join(config.CONFIG.cache_dir, row["specimen_id"])
    try:
        _, _, _, ds = loadNWB(path, return_obj=True)
    except Exception as e:  # noqa: BLE001 - a bad file just means try the next cell
        logger.warning("could not load %s: %s", path, e)
        return None
    if ds is None:
        return None
    want = {sw: name for name, (sw, _) in picks.items()}
    traces = {}
    for sw, t, v, i, skip in _iter_current_clamp_traces(ds):
        if skip is not None or sw not in want:
            continue
        start, dur, amp, _, _ = get_stim_characteristics(i, t)
        if start is None:
            continue
        keep = (t >= start - 0.15 * dur) & (t <= start + 1.3 * dur)
        tt, vv = minmax_decimate(t[keep] - start, v[keep])
        traces[want[sw]] = dict(t=tt, v=vv, amp=amp, dur=dur)
    return traces if len(traces) == 3 else None


def choose_exemplars(df: pd.DataFrame, X: np.ndarray, n_examples: int, n_candidates: int = 60) -> pd.DataFrame:
    """Per cluster, walk cells from nearest to the cluster mean and keep ``n_examples``
    whose traces load, preferring new species, then new dandisets."""
    chosen = []
    for c in sorted(df["coarse cluster"].unique()):
        members = np.flatnonzero(df["coarse cluster"].to_numpy() == c)
        d = np.linalg.norm(X[members] - X[members].mean(axis=0), axis=1)
        ranked = members[np.argsort(d)][:n_candidates]
        picked, species, dandisets, tried = [], set(), set(), set()
        for pass_ in ("species", "dandiset", "any"):
            for idx in ranked:
                if len(picked) == n_examples:
                    break
                r = df.iloc[idx]
                if idx in tried:
                    continue
                if pass_ == "species" and r["species"] in species:
                    continue
                if pass_ == "dandiset" and r["dandiset label"] in dandisets:
                    continue
                tried.add(idx)
                picks = pick_sweeps(parse_sweeps(r["analyzed_sweeps"]), r["rheobase_i"])
                if picks is None:
                    continue
                traces = extract_traces(r, picks)
                if traces is None:
                    continue
                rank = int(np.flatnonzero(ranked == idx)[0])
                picked.append(dict(row=idx, rank=rank, traces=traces))
                species.add(r["species"])
                dandisets.add(r["dandiset label"])
                print(f"  cluster {c}: {r['specimen_id']} ({r['species']}, rank {rank})")
        if len(picked) < n_examples:
            logger.warning("cluster %d: only %d usable exemplars", c, len(picked))
        for k, p in enumerate(picked):
            p["tag"] = f"{c}{'abcdefghij'[k]}"
            chosen.append(p)
    return chosen


def save_cache(chosen, df, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    arrays, meta = {}, []
    for p in chosen:
        r = df.iloc[p["row"]]
        m = dict(tag=p["tag"], specimen_id=r["specimen_id"], cluster=int(r["coarse cluster"]),
                 fine_cluster=r["fine cluster"], species=r["species"],
                 dandiset=str(int(r["dandiset label"])).zfill(6), rank_to_centre=p["rank"],
                 rheobase_i=r["rheobase_i"], **{"umap X": r["umap X"], "umap Y": r["umap Y"]})
        for name, tr in p["traces"].items():
            arrays[f"{p['tag']}_{name}_t"] = tr["t"]
            arrays[f"{p['tag']}_{name}_v"] = tr["v"]
            m[f"{name}_pA"] = round(tr["amp"])
            m[f"{name}_dur_s"] = tr["dur"]
        meta.append(m)
    np.savez_compressed(os.path.join(cache_dir, "exemplar_traces.npz"), **arrays)
    pd.DataFrame(meta).to_csv(os.path.join(cache_dir, "exemplars.csv"), index=False)


def load_cache(cache_dir):
    meta = pd.read_csv(os.path.join(cache_dir, "exemplars.csv"), dtype={"dandiset": str})
    z = np.load(os.path.join(cache_dir, "exemplar_traces.npz"))
    traces = {r["tag"]: {n: dict(t=z[f"{r['tag']}_{n}_t"], v=z[f"{r['tag']}_{n}_v"], amp=r[f"{n}_pA"])
                         for n in ("hyper", "rheo", "strong")}
              for _, r in meta.iterrows()}
    return meta, traces


def draw_example(ax, meta_row, traces, ylim, xlim, scalebar=True) -> None:
    c = int(meta_row["cluster"])
    color = CLUSTER_COLORS[c]
    styles = {"hyper": (MUTED, 0.7), "strong": (color, 0.6), "rheo": (shade(color, -0.55), 0.7)}
    for name in ("hyper", "strong", "rheo"):  # rheobase drawn last so its first spike reads
        col, lw = styles[name]
        ax.plot(traces[name]["t"], traces[name]["v"], color=col, lw=lw, solid_joinstyle="round",
                label=name)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.axis("off")
    amps = "/".join(f"{traces[n]['amp']:+d}".replace("-", "−") for n in ("hyper", "rheo", "strong"))
    ax.set_title(f"{meta_row['tag']}", loc="left", fontsize=8, fontweight="bold", color=INK, pad=2)
    ax.text(0.0, 0.0, f"{meta_row['species']} · DANDI {meta_row['dandiset']}\n{amps} pA",
            transform=ax.transAxes, fontsize=5.5, color=INK_2, va="top", linespacing=1.3)
    if scalebar:  # top right is empty: spikes stop at step offset, the tail sits near rest
        x0, y0 = xlim[1] - 0.2, ylim[1] - 22
        ax.plot([x0, x0 + 0.2], [y0, y0], color=INK, lw=0.8, solid_capstyle="butt", gid="scalebar_t")
        ax.plot([x0 + 0.2, x0 + 0.2], [y0, y0 + 20], color=INK, lw=0.8, solid_capstyle="butt",
                gid="scalebar_v")
        ax.text(x0 + 0.1, y0 - 1.5, "200 ms", fontsize=5.5, color=INK_2, ha="center", va="top")
        ax.text(x0 + 0.215, y0 + 10, "20 mV", fontsize=5.5, color=INK_2, ha="left", va="center")


def fig_examples(meta: pd.DataFrame, traces: dict, out: str) -> None:
    print("example traces")
    allv = np.concatenate([tr["v"] for cell in traces.values() for tr in cell.values()])
    ylim = (np.floor(allv.min() / 10) * 10, np.ceil(allv.max() / 10) * 10 + 2)
    longest = max(float(meta[["hyper_dur_s", "rheo_dur_s", "strong_dur_s"]].max().max()), 0.5)
    xlim = (-0.15 * longest, 1.3 * longest)
    for _, r in meta.iterrows():
        fig, ax = plt.subplots(figsize=TRACE_SIZE)
        draw_example(ax, r, traces[r["tag"]], ylim, xlim)
        save(fig, out, f"examples/cluster{r['tag']}")

    clusters = sorted(meta["cluster"].unique())
    n_cols = int(meta.groupby("cluster").size().max())
    fig, axs = plt.subplots(len(clusters), n_cols, squeeze=False,
                            figsize=(TRACE_SIZE[0] * n_cols * 1.05, TRACE_SIZE[1] * len(clusters) * 1.15))
    for i, c in enumerate(clusters):
        rows = meta[meta["cluster"] == c].reset_index(drop=True)
        for j in range(n_cols):
            ax = axs[i, j]
            if j < len(rows):
                draw_example(ax, rows.iloc[j], traces[rows.iloc[j]["tag"]], ylim, xlim,
                             scalebar=(i == len(clusters) - 1 and j == n_cols - 1))
            else:
                ax.axis("off")
    fig.tight_layout(h_pad=1.2)
    save(fig, out, "cluster_examples_grid", png=True)

    # Shared key for the three sweep colours.
    fig, ax = plt.subplots(figsize=(2.2, 0.5))
    ax.axis("off")
    key = [("Hyperpolarizing step", MUTED), ("Rheobase step", shade(CLUSTER_COLORS[1], -0.55)),
           ("Strong step", CLUSTER_COLORS[1])]
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=1.2, label=l) for l, c in key],
              loc="center left", ncol=1, fontsize=6, handlelength=1.5)
    save(fig, out, "examples/trace_key")


# ---------------------------------------------------------------------------
# Feature maps and within-cluster summaries
# ---------------------------------------------------------------------------

@dataclass
class FeatureScale:
    """One feature in colour space: log10 of display units for log features, else display
    units. Limits are the population's 1st-99th percentile, shared by every panel."""
    col: str
    label: str
    units: str
    values: np.ndarray
    is_log: bool
    norm: mcolors.Normalize
    ticks: np.ndarray
    ticklabels: list

    @property
    def title(self) -> str:
        return f"{self.label} ({self.units})" if self.units else self.label

    @property
    def slug(self) -> str:
        return slug(self.label)

    def real(self, values):
        return 10 ** values if self.is_log else values


def number_label(v: float) -> str:
    return f"{v:.3g}".replace("-", "−")


def cell_label(v: float) -> str:
    """Heatmap number: two decimals under 1 so a column reads evenly, else three figures."""
    if abs(v) >= 1:
        return number_label(v)
    return "0" if round(v, 2) == 0 else f"{v:.2f}".replace("-", "\u2212")


def nice_log_ticks(vmin: float, vmax: float) -> np.ndarray:
    """1-2-5 values (in log10 space) that fall inside the range."""
    decades = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
    candidates = np.log10([m * 10 ** d for d in decades for m in (1, 2, 5)])
    return candidates[(candidates >= vmin) & (candidates <= vmax)]


def feature_scales(df: pd.DataFrame) -> list[FeatureScale]:
    transformed = scale_features(df[config.COLS_TO_KEEP].copy())
    scales = []
    for col in config.COLS_TO_KEEP:
        label, units, factor = FEATURE_LABELS[col]
        is_log = col in config.SCALE_FEATURES
        values = transformed[col].to_numpy(dtype=float) * (1 if is_log else factor)
        vmin, vmax = np.nanpercentile(values, [1, 99])
        if is_log:
            ticks = nice_log_ticks(vmin, vmax)
            ticklabels = [number_label(10 ** t) for t in ticks]
        else:
            ticks = MaxNLocator(nbins=4).tick_values(vmin, vmax)
            ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
            ticklabels = [number_label(t) for t in ticks]
        scales.append(FeatureScale(col, label, units, values, is_log, mcolors.Normalize(vmin, vmax),
                                   ticks, ticklabels))
    return scales


def map_frame(df: pd.DataFrame, pad: float = 0.03) -> tuple[float, float, float, float]:
    x0, x1 = df["umap X"].min(), df["umap X"].max()
    y0, y1 = df["umap Y"].min(), df["umap Y"].max()
    p = pad * max(x1 - x0, y1 - y0)
    return x0 - p, x1 + p, y0 - p, y1 + p


def fix_frame(ax, frame) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(frame[0], frame[1])
    ax.set_ylim(frame[2], frame[3])


def map_figure(extra_width: float = 0.0):
    """Fixed canvas. ``extra_width`` (inches) adds room on the right and leaves the axes
    the same size in inches, so the map still stacks with the other panels."""
    w, h = MAP_SIZE
    x, y, ax_w, ax_h = MAP_RECT
    fig = plt.figure(figsize=(w + extra_width, h))
    k = w / (w + extra_width)
    return fig, fig.add_axes((x * k, y, ax_w * k, ax_h))


def scale_colorbar(cax, scale: FeatureScale) -> None:
    cb = cax.figure.colorbar(mcm.ScalarMappable(norm=scale.norm, cmap=SEQUENTIAL), cax=cax,
                             orientation="horizontal")
    cb.set_ticks(scale.ticks, labels=scale.ticklabels)
    cb.minorticks_off()
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=5.5, length=2, width=0.4, pad=1, color=MUTED, labelcolor=INK_2)


def draw_feature_map(ax, df, scale: FeatureScale, frame, rng, cluster: int | None = None, dot: float = DOT,
                     raster_context: bool = False, raster_dots: bool = False, colorbar: bool = True,
                     title: str | None = None, arrows: bool = True) -> None:
    """Colour cells by a feature. With ``cluster``, only that cluster's cells; cells outside
    it and cells with no measured value (never the imputed one) are grey."""
    x, y = df["umap X"].to_numpy(), df["umap Y"].to_numpy()
    show = np.isfinite(scale.values)
    if cluster is not None:
        show &= df["coarse cluster"].to_numpy() == cluster
    ax.scatter(x[~show], y[~show], c=CONTEXT, s=dot, linewidths=0, rasterized=raster_context)
    order = rng.permutation(np.flatnonzero(show))  # not sorted by value, so extremes don't hide the rest
    ax.scatter(x[order], y[order], c=scale.values[order], cmap=SEQUENTIAL, norm=scale.norm, s=dot,
               linewidths=0, rasterized=raster_dots)
    if arrows:
        umap_axes(ax)
    else:
        ax.axis("off")
    fix_frame(ax, frame)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=3)
    if colorbar:  # the map's top left corner is empty
        scale_colorbar(ax.inset_axes([0.02, 0.92, 0.28, 0.03]), scale)


def fig_feature_maps(df, scales, frame, out: str, rng) -> None:
    print("feature maps")
    for s in scales:
        fig, ax = map_figure()
        draw_feature_map(ax, df, s, frame, rng, title=s.title)
        save(fig, out, f"features/umap_{s.slug}", tight=False)
    n_cols = 4
    n_rows = int(np.ceil(len(scales) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(MAP_SIZE[0] * n_cols, MAP_SIZE[1] * n_rows))
    for ax in axs.flat[len(scales):]:
        ax.axis("off")
    for ax, s in zip(axs.flat, scales):
        draw_feature_map(ax, df, s, frame, rng, raster_context=True, raster_dots=True, title=s.title)
    fig.tight_layout()
    save(fig, out, "umap_features_grid", png=True)


def fig_within_clusters(df, scales, frame, out: str, rng) -> None:
    """Each cluster's cells coloured by each feature, on one colour scale per feature."""
    print("per-cluster feature maps")
    clusters = sorted(df["coarse cluster"].unique())
    for s in scales:
        for c in clusters:
            fig, ax = map_figure()
            draw_feature_map(ax, df, s, frame, rng, cluster=c, raster_context=True, colorbar=False,
                             title=f"{s.label}: cluster {c}")
            save(fig, out, f"within_clusters/{s.slug}/cluster{c}", tight=False)
        fig = plt.figure(figsize=(1.4, 0.45))
        cax = fig.add_axes((0.08, 0.55, 0.84, 0.2))
        scale_colorbar(cax, s)
        cax.set_title(s.title, fontsize=6, color=INK_2, pad=2)
        save(fig, out, f"within_clusters/colorbar_{s.slug}")

    # Grids are for reading the set at a glance, so dots are rasterized to keep them small.
    n_cols = 4
    n_rows = int(np.ceil(len(scales) / n_cols))
    for c in clusters:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(MAP_SIZE[0] * n_cols, MAP_SIZE[1] * n_rows))
        for ax in axs.flat[len(scales):]:
            ax.axis("off")
        for ax, s in zip(axs.flat, scales):
            draw_feature_map(ax, df, s, frame, rng, cluster=c, raster_context=True, raster_dots=True,
                             title=s.title)
        fig.suptitle(f"Cluster {c}", x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout()
        save(fig, out, f"within_clusters/cluster{c}_features_grid")

    fig, axs = plt.subplots(len(clusters), len(scales), squeeze=False,
                            figsize=(1.6 * len(scales), 1.4 * len(clusters)))
    for i, c in enumerate(clusters):
        for j, s in enumerate(scales):
            draw_feature_map(axs[i, j], df, s, frame, rng, cluster=c, dot=0.5, colorbar=False, arrows=False,
                             title=s.label if i == 0 else None)
        axs[i, 0].text(-0.05, 0.5, f"Cluster {c}", transform=axs[i, 0].transAxes, rotation=90,
                       ha="right", va="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out, "within_clusters_matrix_preview.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def heatmap_colorbar(fig, ax, label: str) -> None:
    norm = mcolors.Normalize(-HEATMAP_LIMIT, HEATMAP_LIMIT)
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=DIVERGING), ax=ax, fraction=0.025, pad=0.02,
                      extend="both")
    cb.set_label(label, fontsize=6, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2, width=0.4, color=MUTED, labelcolor=INK_2)


def feature_xticks(ax, scales) -> None:
    ax.set_xticks(np.arange(len(scales)) + 0.5, [s.title for s in scales], rotation=40, ha="right",
                  rotation_mode="anchor", color=INK_2)
    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_cluster_heatmaps(df, scales, X: np.ndarray, out: str) -> None:
    print("cluster x feature heatmaps")
    norm = mcolors.Normalize(-HEATMAP_LIMIT, HEATMAP_LIMIT)
    colorbar_label = "Median, in IQRs from the population median"
    clusters = sorted(df["coarse cluster"].unique())
    coarse = df["coarse cluster"].to_numpy()
    counts = df["coarse cluster"].value_counts()

    # Coarse: colour is the scaled value the clustering saw, the number is in real units.
    medians = np.array([np.median(X[coarse == c], axis=0) for c in clusters])
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.pcolormesh(np.arange(len(scales) + 1), np.arange(len(clusters) + 1), medians, cmap=DIVERGING,
                  norm=norm, edgecolors=SURFACE, linewidth=1.0)
    for i, c in enumerate(clusters):
        for j, s in enumerate(scales):
            real = np.nanmedian(s.real(s.values[coarse == c]))
            ink = "white" if abs(medians[i, j]) > 0.65 * HEATMAP_LIMIT else INK
            ax.text(j + 0.5, i + 0.5, cell_label(real), ha="center", va="center", fontsize=6, color=ink)
    ax.set_yticks(np.arange(len(clusters)) + 0.5, [f"Cluster {c}  (n={counts[c]:,})" for c in clusters],
                  color=INK_2)
    ax.invert_yaxis()
    feature_xticks(ax, scales)
    heatmap_colorbar(fig, ax, colorbar_label)
    save(fig, out, "cluster_feature_heatmap", png=True)

    # Fine: grouped by coarse cluster, with a colour strip for the parent and gaps between groups.
    fine = df["fine cluster"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 6.0))
    y, gap = 0.0, 0.5
    ticks, ticklabels = [], []
    for c in clusters:
        subs = sorted(np.unique(fine[coarse == c]), key=lambda f: int(f.split(".")[1]))
        block = np.array([np.median(X[fine == f], axis=0) for f in subs])
        ax.pcolormesh(np.arange(len(scales) + 1), y + np.arange(len(subs) + 1), block, cmap=DIVERGING,
                      norm=norm, edgecolors=SURFACE, linewidth=0.6)
        ax.add_patch(plt.Rectangle((-0.22, y), 0.14, len(subs), color=CLUSTER_COLORS[c], clip_on=False, lw=0))
        for k, f in enumerate(subs):
            ticks.append(y + k + 0.5)
            ticklabels.append(f"{f}  (n={(fine == f).sum():,})")
        y += len(subs) + gap
    ax.set_yticks(ticks, ticklabels, color=INK_2)
    ax.tick_params(axis="y", pad=9)
    ax.set_xlim(0, len(scales))
    ax.set_ylim(y - gap, 0)
    feature_xticks(ax, scales)
    ax.tick_params(axis="y", pad=9)
    heatmap_colorbar(fig, ax, colorbar_label)
    save(fig, out, "fine_cluster_feature_heatmap", png=True)


def draw_violins(ax, df, scale: FeatureScale) -> None:
    clusters = sorted(df["coarse cluster"].unique())
    coarse = df["coarse cluster"].to_numpy()
    finite = np.isfinite(scale.values)
    data = [scale.values[finite & (coarse == c)] for c in clusters]
    parts = ax.violinplot(data, positions=clusters, widths=0.82, showextrema=False, points=300)
    for body, c in zip(parts["bodies"], clusters):
        body.set_facecolor(CLUSTER_COLORS[c])
        body.set_edgecolor("none")
        body.set_alpha(1.0)
    for c, d in zip(clusters, data):
        ax.plot([c - 0.28, c + 0.28], [np.median(d)] * 2, color=INK, lw=0.9, solid_capstyle="butt")
    # Axis limits only: outliers stay in the density estimate but off the visible range.
    lo, hi = np.percentile(scale.values[finite], [0.5, 99.5])
    pad = 0.03 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_yticks(scale.ticks, scale.ticklabels)
    ax.set_xticks(clusters)
    ax.set_xlabel("Cluster")
    ax.set_title(scale.title, loc="left", color=INK)
    ax.tick_params(length=2, width=0.4, labelcolor=INK_2)


def fig_distributions(df, scales, out: str) -> None:
    print("feature distributions")
    for s in scales:
        fig, ax = plt.subplots(figsize=(2.6, 1.9))
        draw_violins(ax, df, s)
        save(fig, out, f"distributions/{s.slug}")
    n_cols = 4
    n_rows = int(np.ceil(len(scales) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows))
    for ax in axs.flat[len(scales):]:
        ax.axis("off")
    for ax, s in zip(axs.flat, scales):
        draw_violins(ax, df, s)
    fig.tight_layout()
    save(fig, out, "distributions_grid", png=True)


# ---------------------------------------------------------------------------
# Brain regions
# ---------------------------------------------------------------------------

def dandiset_note(cells: pd.DataFrame) -> str:
    k = cells["dandiset label"].nunique()
    return f"n={len(cells):,}, {k} dandiset{'' if k == 1 else 's'}"


def densest_point(g: pd.DataFrame, xe, ye, sigma: float = 3.0) -> tuple[float, float]:
    """Peak of the smoothed 2D density, so a tight little clump can't outrank the main body."""
    h, _, _ = np.histogram2d(g["umap X"], g["umap Y"], bins=[xe, ye])
    ix, iy = np.unravel_index(np.argmax(gaussian_filter(h, sigma)), h.shape)
    return (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2


def place_labels(ax, anchors: dict[str, tuple[float, float]]) -> None:
    """Text labels with leader lines, each at the first nearby offset that overlaps no
    earlier label. A plain background box keeps them as editable text in the SVG."""
    offsets = [(a * r, b * r) for r in (10, 18, 28, 40) for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for name, xy in anchors.items():
        for off in offsets:
            t = ax.annotate(name, xy, xytext=off, textcoords="offset points", fontsize=6, color=INK,
                            ha="left" if off[0] > 0 else "right", va="center",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                            arrowprops=dict(arrowstyle="-", lw=0.4, color=INK_2, shrinkA=0, shrinkB=0))
            t.update_positions(renderer)
            # Text's own extent: Annotation's also counts the leader line, which over-rejects.
            box = mtext.Text.get_window_extent(t, renderer).expanded(1.05, 1.3)
            if not any(box.overlaps(b) for b in placed) or off == offsets[-1]:
                placed.append(box)
                break
            t.remove()


def fig_regions(df, frame, out: str) -> None:
    print("brain region panels")
    region = df["brain_region"]
    small = [r for r in region.value_counts().index if r in REGION_COLORS]

    fig, ax = map_figure(extra_width=1.15)  # legend sits right of the map, clear of its left lobe
    for name, color in (("Unknown", CONTEXT), ("Neocortex", NEOCORTEX_COLOR)):
        cells = df[region == name]
        ax.scatter(cells["umap X"], cells["umap Y"], c=color, s=DOT, linewidths=0)
    xe = np.linspace(frame[0], frame[1], 41)
    ye = np.linspace(frame[2], frame[3], 41)
    anchors = {}
    for r in small:
        cells = df[region == r]
        ax.scatter(cells["umap X"], cells["umap Y"], c=REGION_COLORS[r], s=7, linewidths=0.2,
                   edgecolors=SURFACE)
        anchors[r] = densest_point(cells, xe, ye)
    umap_axes(ax)
    fix_frame(ax, frame)
    place_labels(ax, anchors)
    counts = region.value_counts()
    legend(ax, [(f"{r} ({counts[r]:,})", NEOCORTEX_COLOR if r == "Neocortex" else CONTEXT)
                for r in ("Neocortex", "Unknown")] + [(f"{r} ({counts[r]:,})", REGION_COLORS[r]) for r in small],
           loc="upper left", anchor=(1.01, 1.0))
    save(fig, out, "regions/umap_regions_overview", png=True, tight=False)

    order = ["Neocortex"] + small + ["Unknown"]
    for r in order:
        fig, ax = map_figure()
        scatter_highlight(ax, df, (region == r).to_numpy(), REGION_HIGHLIGHT[r], f"{r}  ({dandiset_note(df[region == r])})")
        fix_frame(ax, frame)
        save(fig, out, f"regions/umap_region_{slug(r)}", tight=False)
    fig, axs = plt.subplots(2, 5, figsize=(MAP_SIZE[0] * 5, MAP_SIZE[1] * 2))
    for ax, r in zip(axs.flat, order):
        scatter_highlight(ax, df, (region == r).to_numpy(), REGION_HIGHLIGHT[r], f"{r}  ({dandiset_note(df[region == r])})")
        fix_frame(ax, frame)
    fig.tight_layout()
    save(fig, out, "umap_regions_grid", png=True)

    sub = df["brain_subregion"]
    for s, label in SUBREGION_LABELS.items():
        fig, ax = map_figure()
        scatter_highlight(ax, df, (sub == s).to_numpy(), SLOTS[0], f"{label}  ({dandiset_note(df[sub == s])})")
        fix_frame(ax, frame)
        save(fig, out, f"regions/umap_subregion_{slug(label)}", tight=False)
    fig, axs = plt.subplots(2, 3, figsize=(MAP_SIZE[0] * 3, MAP_SIZE[1] * 2))
    for ax, (s, label) in zip(axs.flat, SUBREGION_LABELS.items()):
        scatter_highlight(ax, df, (sub == s).to_numpy(), SLOTS[0], f"{label}  ({dandiset_note(df[sub == s])})")
        fix_frame(ax, frame)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    n_unassigned = int(((region == "Neocortex") & sub.isna()).sum())
    fig.text(0.01, 0.01, f"{n_unassigned:,} neocortical cells have no subregion and stay grey in every panel.",
             fontsize=7, color=INK_2)
    save(fig, out, "umap_subregions_grid", png=True)


def fig_region_by_cluster(df, out: str) -> None:
    print("region x cluster bars")
    counts = df["brain_region"].value_counts()
    regions = ["Neocortex"] + [r for r in counts.index if r in REGION_COLORS] + ["Unknown"]
    clusters = sorted(df["coarse cluster"].unique())
    frac = pd.crosstab(df["brain_region"], df["coarse cluster"], normalize="index")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    rows = regions[::-1]  # first region on top
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for c in clusters:
        w = frac.loc[rows, c].to_numpy()
        ax.barh(y, w, left=left, height=0.72, color=CLUSTER_COLORS[c], edgecolor=SURFACE, linewidth=1.0,
                label=f"Cluster {c}")
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= 0.12:
                ink = INK if CLUSTER_COLORS[c] == "#eda100" else "white"
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=6, color=ink)
        left += w
    ax.set_yticks(y, [f"{r}  ({dandiset_note(df[df['brain_region'] == r])})" for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of the region's cells")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5, handlelength=1.0,
              handleheight=1.0, borderaxespad=0)
    save(fig, out, "region_by_cluster", png=True)


# ---------------------------------------------------------------------------

SECTIONS = ("species", "clusters", "examples", "features", "within", "regions")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=config.CONFIG.merged_csv)
    p.add_argument("--out", default="figures/blog")
    p.add_argument("--refresh", default=True, action=argparse.BooleanOptionalAction,
                   help="Re-pick exemplars and re-read their NWB files.")
    p.add_argument("--only", default=",".join(SECTIONS),
                   help=f"Comma-separated sections to draw, from: {','.join(SECTIONS)}.")
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    sections = set(args.only.split(","))
    if sections - set(SECTIONS):
        p.error(f"unknown sections: {sorted(sections - set(SECTIONS))}")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    df = load_cells(args.csv)
    print(f"{len(df):,} cells, {df['coarse cluster'].nunique()} coarse / {df['fine cluster'].nunique()} fine clusters")
    if sections & {"clusters", "examples"}:
        cache_dir = os.path.join(args.out, "cache")
        have_cache = os.path.exists(os.path.join(cache_dir, "exemplars.csv"))
        if args.refresh or not have_cache:
            print("choosing exemplars (reading NWB files)")
            chosen = choose_exemplars(df, scaled_features(df), args.n_examples)
            save_cache(chosen, df, cache_dir)
        meta, traces = load_cache(cache_dir)

    rng = np.random.default_rng(args.seed)
    if "species" in sections:
        fig_species(df, args.out, rng)
        fig_species_by_cluster(df, args.out)
    if "clusters" in sections:
        fig_clusters(df, meta, args.out, rng)
    if "examples" in sections:
        fig_examples(meta, traces, args.out)
    frame = map_frame(df)
    if sections & {"features", "within"}:
        scales = feature_scales(df)
    if "features" in sections:
        fig_feature_maps(df, scales, frame, args.out, rng)
    if "within" in sections:
        fig_within_clusters(df, scales, frame, args.out, rng)
        fig_cluster_heatmaps(df, scales, scaled_features(df), args.out)
        fig_distributions(df, scales, args.out)
    if "regions" in sections:
        fig_regions(df, frame, args.out)
        fig_region_by_cluster(df, args.out)
    with open(os.path.join(args.out, "palette.json"), "w") as f:
        json.dump({"clusters": CLUSTER_COLORS, "species": SPECIES_COLORS,
                   "species_groups": SPECIES_GROUP_COLORS, "regions": REGION_HIGHLIGHT}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
