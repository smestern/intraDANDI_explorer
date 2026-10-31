"""Blog figures: species and GMM clusters on the UMAP, plus example cells per cluster.

Every panel is drawn as its own figure and saved as a standalone SVG (text kept as
text) so panels can be combined in Illustrator. Grids of related panels are saved
too, as SVG and PNG, for a quick look at the whole set.

    python tools/blog_figures.py --out figures/blog [--refresh] [--n-examples 3]

Example traces are re-read from the raw NWB files under ``CONFIG.cache_dir`` and
cached in ``<out>/cache`` so restyling reruns don't touch the disk again.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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

UMAP_SIZE = (3.5, 3.5)
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
})


def shade(hex_color: str, amount: float) -> str:
    """Mix ``hex_color`` toward black (``amount`` < 0) or white (> 0); |amount| <= 1."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    target = 1.0 if amount > 0 else 0.0
    return mcolors.to_hex(rgb + (target - rgb) * abs(amount))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save(fig, out_dir: str, name: str, png: bool = False) -> None:
    path = os.path.join(out_dir, f"{name}.svg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
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
    """Number each cluster at its densest spot. GMM clusters are fit in feature space and
    are not contiguous on the UMAP, so their median often falls in another cluster."""
    xe = np.linspace(df["umap X"].min(), df["umap X"].max(), bins + 1)
    ye = np.linspace(df["umap Y"].min(), df["umap Y"].max(), bins + 1)
    for c, g in df.groupby("coarse cluster"):
        h, _, _ = np.histogram2d(g["umap X"], g["umap Y"], bins=[xe, ye])
        ix, iy = np.unravel_index(np.argmax(h), h.shape)
        x, y = (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2
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

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=config.CONFIG.merged_csv)
    p.add_argument("--out", default="figures/blog")
    p.add_argument("--refresh", action="store_true", help="Re-pick exemplars and re-read their NWB files.")
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    df = load_cells(args.csv)
    print(f"{len(df):,} cells, {df['coarse cluster'].nunique()} coarse / {df['fine cluster'].nunique()} fine clusters")
    cache_dir = os.path.join(args.out, "cache")
    have_cache = os.path.exists(os.path.join(cache_dir, "exemplars.csv"))
    if args.refresh or not have_cache:
        print("choosing exemplars (reading NWB files)")
        chosen = choose_exemplars(df, scaled_features(df), args.n_examples)
        save_cache(chosen, df, cache_dir)
    meta, traces = load_cache(cache_dir)

    rng = np.random.default_rng(args.seed)
    fig_species(df, args.out, rng)
    fig_species_by_cluster(df, args.out)
    fig_clusters(df, meta, args.out, rng)
    fig_examples(meta, traces, args.out)
    with open(os.path.join(args.out, "palette.json"), "w") as f:
        json.dump({"clusters": CLUSTER_COLORS, "species": SPECIES_COLORS,
                   "species_groups": SPECIES_GROUP_COLORS}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
