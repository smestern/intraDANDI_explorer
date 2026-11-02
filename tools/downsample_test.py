"""Does recording at 10 kHz change the measured features? A direct test on 50 kHz cells.

No dandiset has both 10 and 50 kHz cells, so across datasets the sampling rate can't be told
apart from lab and region. Here the same 50 kHz sweeps are degraded to look like lower-rate
recordings and pushed through the same feature extraction the merge uses, so every difference
comes from the degradation alone:

    original    50 kHz, analysis dv/dt filter at 10 kHz (what the database holds)
    filter 4k   50 kHz, analysis filter at 4 kHz: the cutoff the pipeline uses at 10 kHz, no resampling
    25 kHz      every 2nd sample
    10 kHz      every 5th sample, no anti-alias filter
    10k+5k      causal 4-pole Bessel at 5 kHz, then 10 kHz, like an amplifier filter at Nyquist
    10k+2k      causal 4-pole Bessel at 2 kHz, then 10 kHz, like a heavy amplifier filter

    python tools/downsample_test.py --out figures/downsampling_test [--per-group 25] [--n-jobs 8]

Results are cached per cell in ``<out>/cache`` so a rerun only redoes the figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dandi_scraper import config  # noqa: E402
from dandi_scraper.analysis import scale_features  # noqa: E402
from blog_figures import INK, INK_2, MUTED, SLOTS, SURFACE, CONTEXT  # noqa: E402

SEED = 0
BASE_RATE = 50.0  # kHz

# name -> (decimation factor, analog Bessel cutoff in Hz or None, analysis filter cap in Hz)
CONDITIONS = {
    "original": (1, None, 10000),
    "filter 4k": (1, None, 4000),
    "25 kHz": (2, None, 10000),
    "10 kHz": (5, None, 10000),
    "10k+5k": (5, 5000, 10000),
    "10k+2k": (5, 2000, 10000),
}
ORDER = list(CONDITIONS)

# Cells are drawn per dataset and class, so fast-spiking and broad cells are both represented.
DATASETS = ["000020", "000636", "001938", "001455", "001348", "001359"]

FEATURES = {
    "ap_1_upstroke_0_long_square": "upstroke (mV/ms)",
    "ap_1_downstroke_0_long_square": "downstroke (mV/ms)",
    "ap_1_upstroke_downstroke_ratio_0_long_square": "up/down ratio",
    "ap_1_width_0_long_square": "width (ms)",
    "ap_1_peak_v_0_long_square": "peak (mV)",
    "ap_1_threshold_v_0_long_square": "threshold (mV)",
    "ap_1_fast_trough_v_0_long_square": "fast trough (mV)",
    "input_resistance": "Rin (MOhm)",
    "tau": "tau (ms)",
    "v_baseline": "V baseline (mV)",
    "sag_nearest_minus_100": "sag",
    "latency_0_long_square": "latency (s)",
    "fi_linear_fit_slope": "f-I slope (Hz/pA)",
}
MS_FEATURES = {"ap_1_width_0_long_square", "tau"}  # stored in seconds


def degrade(t, v, i, factor, cutoff):
    """Apply an amplifier-like causal filter, then keep every ``factor``-th sample."""
    if cutoff is not None:
        fs = 1.0 / (t[1] - t[0])
        sos = signal.bessel(4, cutoff, fs=fs, norm="mag", output="sos")
        zi = signal.sosfilt_zi(sos) * v[0]
        v, _ = signal.sosfilt(sos, v, zi=zi)
    return t[::factor], v[::factor], i[::factor]


def measure_cell(specimen_id: str, analyzed_sweeps: str, cache_dir: str, cell_cache: str) -> list[dict]:
    """Feature rows for one cell, one per condition."""
    path = os.path.join(cell_cache, hashlib.md5(specimen_id.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    # Imported here so each worker process sets up gigaseal on its own.
    from gigaseal.database.long_square import long_square_features, select_long_square_run
    from gigaseal.loadFile.loadNWB import GLOBAL_STIM_NAMES, loadNWB
    from ipfx.sweep import Sweep, SweepSet
    GLOBAL_STIM_NAMES.stim_inc = [""]
    GLOBAL_STIM_NAMES.stim_exc = []
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")

    rows = []
    try:
        sweep_ids = {int(x.split(":")[0]) for x in analyzed_sweeps.split(";")}
        _, _, _, data_set = loadNWB(os.path.join(cache_dir, specimen_id), return_obj=True)
        run = select_long_square_run(data_set, sweep_ids=sweep_ids)
    except Exception as e:
        rows = [{"specimen_id": specimen_id, "condition": c, "error": f"load: {e}"} for c in ORDER]
        run = None
    if run is not None:
        for cond, (factor, cutoff, max_filter) in CONDITIONS.items():
            row = {"specimen_id": specimen_id, "condition": cond}
            try:
                swps = []
                for s in run.sweeps.sweeps:
                    t, v, i = degrade(np.asarray(s.t), np.asarray(s.v), np.asarray(s.i), factor, cutoff)
                    swps.append(Sweep(t, v, i, clamp_mode="CurrentClamp",
                                      sampling_rate=int(round(1 / (t[1] - t[0]))), sweep_number=s.sweep_number))
                features, _ = long_square_features(SweepSet(swps), run.start, run.end, amp_interval=20,
                                                   max_filter_hz=max_filter)
                row.update({k: features.get(k, np.nan) for k in FEATURES})
                row["n_spikes_ok"] = True
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, default=float)
    return rows


def pick_cells(df: pd.DataFrame, rates: pd.Series, per_group: int) -> pd.DataFrame:
    """``per_group`` fast-spiking and ``per_group`` other cells per dataset, all at exactly 50 kHz."""
    d = df[df["specimen_id"].map(rates).round(1).eq(BASE_RATE) & df["ds"].isin(DATASETS)]
    d = d[d["analyzed_sweeps"].notna()]
    picks = []
    for (_, _), g in d.groupby(["ds", "fast spiking"]):
        picks.append(g.sample(min(per_group, len(g)), random_state=SEED))
    return pd.concat(picks)


def load_cells(csv_path: str) -> pd.DataFrame:
    cols = ["specimen_id", "dandiset label", "analyzed_sweeps", "ap_1_upstroke_downstroke_ratio_0_long_square",
            "ap_1_width_0_long_square", "sampling rate (kHz)", *FEATURES, *config.COLS_TO_KEEP]
    df = pd.read_csv(csv_path, low_memory=False, usecols=lambda c: c in set(cols))
    df["ds"] = df["dandiset label"].astype(str).str.zfill(6)
    width = df["ap_1_width_0_long_square"] * 1000
    df["fast spiking"] = (width < 0.5) & (df["ap_1_upstroke_downstroke_ratio_0_long_square"] < 2.0)
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def to_display_units(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    for f in MS_FEATURES:
        res[f] = res[f] * 1000
    return res


def effect_table(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Median per-cell change from the original, in each condition, for every feature."""
    base = wide["original"]
    rows = {}
    for cond in ORDER[1:]:
        for f, name in FEATURES.items():
            diff = (wide[cond][f] - base[f]).dropna()
            pct = (100 * diff / base[f].abs()).dropna()
            rows[(cond, name)] = {"median change": diff.median(), "median % change": pct.median(),
                                  "n cells": len(diff)}
    return pd.DataFrame(rows).T.rename_axis(["condition", "feature"])


def observed_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset medians of real recordings: how far apart 10 and 50 kHz sets actually are."""
    d = df.copy()
    d["rate"] = d["sampling rate (kHz)"].round(1)
    med = d[d["rate"].isin([10.0, 50.0])].groupby(["rate", "ds"])[list(FEATURES)].median()
    n = d[d["rate"].isin([10.0, 50.0])].groupby(["rate", "ds"]).size()
    med = med[n >= 20]
    return to_display_units(med).groupby("rate").median().T.rename(index=FEATURES)


def map_shift(df: pd.DataFrame, wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Move each test cell in the pipeline's feature space: how far does degrading it carry it,
    against its distance to its own neighbours, and does it land nearer the real 10 kHz sets?"""
    cols = list(config.COLS_TO_KEEP)
    ref = df[df[cols].notna().all(axis=1)]
    ref_x = scale_features(ref[cols].copy()).replace([np.inf, -np.inf], np.nan)
    keep = ref_x.notna().all(axis=1)
    ref, ref_x = ref[keep], ref_x[keep]
    scaler = RobustScaler().fit(ref_x.values)
    clip = config.ROBUST_CLIP
    ref_z = np.clip(scaler.transform(ref_x.values), -clip, clip)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(ref_z)
    is_10k = ref["sampling rate (kHz)"].round(1).eq(10.0).to_numpy()
    ids = ref["specimen_id"].to_numpy()

    def project(table):
        t = table.reindex(columns=cols).dropna()
        z = scale_features(t.copy()).replace([np.inf, -np.inf], np.nan).dropna()
        return pd.DataFrame(np.clip(scaler.transform(z.values), -clip, clip), index=z.index, columns=cols)

    base = project(wide["original"])
    rows = {}
    for cond in ORDER:
        z = project(wide[cond]).reindex(base.index).dropna()
        dist, idx = nn.kneighbors(z.values)
        share, nb_dist = [], []
        for cell, d_row, i_row in zip(z.index, dist, idx):
            keep_nb = ids[i_row] != cell  # a cell's own database entry doesn't count as a neighbour
            share.append(is_10k[i_row[keep_nb]][:15].mean())
            nb_dist.append(d_row[keep_nb][:15].mean())
        b = base.loc[z.index].values
        cos = 1 - np.sum(b * z.values, axis=1) / (np.linalg.norm(b, axis=1) * np.linalg.norm(z.values, axis=1))
        rows[cond] = {"share of neighbours from 10 kHz sets": np.mean(share),
                      "median cosine distance moved": np.median(cos),
                      "median distance to neighbours": np.median(nb_dist),
                      "n cells": len(z)}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save(fig, out, name):
    path = os.path.join(out, f"{name}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_paired(wide, cells, out):
    panels = [("ap_1_upstroke_0_long_square", "AP upstroke (mV/ms)"),
              ("ap_1_width_0_long_square", "AP width (ms)"),
              ("ap_1_peak_v_0_long_square", "AP peak (mV)")]
    fs = cells.set_index("specimen_id")["fast spiking"]
    conds = ["10 kHz", "10k+5k", "10k+2k"]
    fig, axs = plt.subplots(len(conds), len(panels), figsize=(9.2, 8.4))
    for r, cond in enumerate(conds):
        for c, (f, lab) in enumerate(panels):
            ax = axs[r, c]
            x, y = wide["original"][f], wide[cond][f]
            ok = x.notna() & y.notna()
            is_fs = fs.reindex(x.index).fillna(False).astype(bool)
            for mask, color, name in [(~is_fs, SLOTS[0], "other"), (is_fs, SLOTS[1], "fast-spiking")]:
                ax.scatter(x[ok & mask], y[ok & mask], s=9, color=color, lw=0, alpha=0.7, label=name)
            lims = [min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())]
            ax.plot(lims, lims, color=MUTED, lw=0.8, ls=":", zorder=0)
            ax.set_xlabel(f"{lab}, original 50 kHz" if r == len(conds) - 1 else "")
            if c == 0:
                ax.set_ylabel(cond, color=INK)
            if r == 0:
                ax.set_title(lab, loc="left", color=INK)
            if r == 0 and c == 2:
                ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, out, "01_same_cell_original_vs_degraded")


def fig_effects(wide, gap, out):
    """Median change per condition, with the real gap between 10 and 50 kHz datasets for scale."""
    feats = [("ap_1_upstroke_0_long_square", "upstroke (mV/ms)"), ("ap_1_width_0_long_square", "width (ms)"),
             ("ap_1_peak_v_0_long_square", "peak (mV)"), ("ap_1_fast_trough_v_0_long_square", "fast trough (mV)"),
             ("ap_1_threshold_v_0_long_square", "threshold (mV)")]
    fig, axs = plt.subplots(1, len(feats), figsize=(11, 2.9), sharey=True)
    conds = ORDER[1:]
    y = np.arange(len(conds))
    for ax, (f, name) in zip(axs, feats):
        base = wide["original"][f]
        med = [(wide[c][f] - base).median() for c in conds]
        ax.barh(y, med, height=0.62, color=SLOTS[0], edgecolor=SURFACE)
        real = gap.loc[name, 10.0] - gap.loc[name, 50.0]
        ax.axvline(real, color=SLOTS[1], lw=2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{name}\nreal gap {real:+.3g}", loc="left", color=INK, fontsize=8)
        ax.margins(x=0.25)
        ax.set_xlabel("median change")
    axs[0].set_yticks(y, conds)
    axs[0].invert_yaxis()
    axs[0].tick_params(axis="y", length=0, labelcolor=INK_2)
    fig.suptitle("Effect of degrading the same 50 kHz cells (bars) vs. the real gap between 10 and 50 kHz "
                 "datasets (orange line)", x=0.01, ha="left", color=INK, fontsize=9, y=1.08)
    fig.tight_layout()
    save(fig, out, "02_effect_vs_real_gap")


def fig_trace(wide, cells, cache_dir, out):
    """One fast-spiking and one broad spike, as measured under each degradation."""
    from gigaseal.database.long_square import select_long_square_run
    from gigaseal.loadFile.loadNWB import GLOBAL_STIM_NAMES, loadNWB
    GLOBAL_STIM_NAMES.stim_inc = [""]
    GLOBAL_STIM_NAMES.stim_exc = []

    picks = []
    for is_fs in (True, False):
        g = cells[cells["fast spiking"] == is_fs]
        up = wide["original"]["ap_1_upstroke_0_long_square"].reindex(g["specimen_id"]).dropna()
        if len(up):
            picks.append((is_fs, up.sort_values().index[len(up) // 2]))

    fig, axs = plt.subplots(1, len(picks), figsize=(8, 3.1))
    styles = {"original": (INK, 1.6, "-"), "10 kHz": (SLOTS[0], 1.1, "-"), "10k+5k": (SLOTS[2], 1.1, "-"),
              "10k+2k": (SLOTS[1], 1.1, "-")}
    for ax, (is_fs, sid) in zip(np.atleast_1d(axs), picks):
        row = cells[cells["specimen_id"] == sid].iloc[0]
        _, _, _, ds = loadNWB(os.path.join(cache_dir, sid), return_obj=True)
        ids = {int(x.split(":")[0]) for x in row["analyzed_sweeps"].split(";")}
        run = select_long_square_run(ds, sweep_ids=ids)
        # The weakest step whose response spikes inside the step window holds a clean first spike.
        # Depolarizing only: a stimulus artifact outside the window can cross the same voltage.
        best, best_amp = None, np.inf
        for s in run.sweeps.sweeps:
            t, v = np.asarray(s.t), np.asarray(s.v)
            in_step = (t >= run.start) & (t <= run.end)
            amp = np.median(np.asarray(s.i)[in_step])
            if v[in_step].max() > -10 and 0 < amp < best_amp:
                best, best_amp = s, amp
        t, v, i = np.asarray(best.t), np.asarray(best.v), np.asarray(best.i)
        in_step = (t >= run.start) & (t <= run.end)
        peak = t[in_step][np.argmax(v[in_step] > -10)]
        peak = t[(t >= peak) & (t <= peak + 0.003)][np.argmax(v[(t >= peak) & (t <= peak + 0.003)])]
        for cond, (color, lw, ls) in styles.items():
            factor, cutoff, _ = CONDITIONS[cond]
            td, vd, _ = degrade(t, v, i, factor, cutoff)
            m = (td > peak - 0.0015) & (td < peak + 0.003)
            ax.plot((td[m] - peak) * 1000, vd[m], color=color, lw=lw, ls=ls, marker="." if factor > 1 else None,
                    ms=3, label=cond)
        ax.set_xlabel("Time from peak (ms)")
        ax.set_ylabel("mV")
        ax.set_title(f"{'Fast-spiking' if is_fs else 'Broad'} cell, {sid.split('/')[0]}", loc="left", color=INK)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, out, "03_example_spikes")


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=config.CONFIG.merged_csv)
    p.add_argument("--out", default="figures/downsampling_test")
    p.add_argument("--per-group", type=int, default=25, help="Cells per dataset per class.")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--limit", type=int, help="Only run this many cells, for a quick check.")
    args = p.parse_args(argv)
    cell_cache = os.path.join(args.out, "cache")
    os.makedirs(cell_cache, exist_ok=True)

    df = load_cells(args.csv)
    rates = pd.read_csv(os.path.join(config.CONFIG.output_dir, "sampling_rates.csv"),
                        index_col=0)["sampling_rate_khz"]
    cells = pick_cells(df, rates, args.per_group)
    if args.limit:
        cells = cells.sample(args.limit, random_state=SEED)
    print(f"{len(cells)} cells: {cells.groupby(['ds', 'fast spiking']).size().to_dict()}")

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=5)(
        joblib.delayed(measure_cell)(sid, sw, config.CONFIG.cache_dir, cell_cache)
        for sid, sw in zip(cells["specimen_id"], cells["analyzed_sweeps"]))
    res = pd.DataFrame([r for rows in results for r in rows])
    res.to_csv(os.path.join(args.out, "cell_results.csv"), index=False)

    if "error" in res:
        print("failures by condition:\n", res[res["error"].notna()].groupby("condition").size().to_string())
    disp = to_display_units(res)
    wide = {c: g.set_index("specimen_id") for c, g in disp.groupby("condition")}

    # the "original" condition must reproduce what the database holds
    stored = to_display_units(cells.set_index("specimen_id"))
    check = {}
    for f, name in FEATURES.items():
        a, b = wide["original"][f], stored[f].reindex(wide["original"].index)
        ok = a.notna() & b.notna()
        check[name] = {"correlation": np.corrcoef(a[ok], b[ok])[0, 1], "median abs diff": (a[ok] - b[ok]).abs().median(),
                       "n": int(ok.sum())}
    check = pd.DataFrame(check).T
    check.to_csv(os.path.join(args.out, "reproduces_database.csv"))

    effects = effect_table(wide)
    effects.to_csv(os.path.join(args.out, "effects.csv"))
    gap = observed_gap(df)
    gap.to_csv(os.path.join(args.out, "real_gap_10_vs_50_khz.csv"))
    fs_ids = set(cells.loc[cells["fast spiking"], "specimen_id"])
    by_class = {}
    for cond in ORDER[1:]:
        for label, ids in [("fast-spiking", fs_ids), ("other", set(cells["specimen_id"]) - fs_ids)]:
            base, w = wide["original"], wide[cond]
            idx = base.index.intersection(list(ids))
            for f in ["ap_1_upstroke_0_long_square", "ap_1_width_0_long_square", "ap_1_peak_v_0_long_square"]:
                by_class[(cond, label, FEATURES[f])] = {
                    "original median": base.loc[idx, f].median(), "degraded median": w.loc[idx, f].median(),
                    "median % change": (100 * (w.loc[idx, f] - base.loc[idx, f]) / base.loc[idx, f].abs()).median()}
    by_class = pd.DataFrame(by_class).T
    by_class.to_csv(os.path.join(args.out, "effects_by_class.csv"))

    # how much of the real gap between 10 and 50 kHz datasets each degradation accounts for
    share = {}
    for cond in ORDER[1:]:
        for f, name in FEATURES.items():
            real = gap.loc[name, 10.0] - gap.loc[name, 50.0]
            change = (wide[cond][f] - wide["original"][f]).median()
            share[(cond, name)] = {"median change": change, "real gap": real, "share of gap": change / real}
    pd.DataFrame(share).T.rename_axis(["condition", "feature"]).to_csv(os.path.join(args.out, "share_of_real_gap.csv"))

    # slower spikes lose less, and the real 10 kHz sets are slow
    up = "ap_1_upstroke_0_long_square"
    bins = pd.cut(wide["original"][up], [0, 100, 150, 200, 250, 300, 400, 1000])
    speed = {}
    for cond in ORDER[1:]:
        pct = 100 * (wide[cond][up] - wide["original"][up]) / wide["original"][up]
        speed[cond] = pct.groupby(bins, observed=True).median()
    speed = pd.DataFrame(speed)
    speed["n cells"] = wide["original"][up].groupby(bins, observed=True).size()
    speed.to_csv(os.path.join(args.out, "upstroke_change_by_spike_speed.csv"))

    # map_shift works in the units the merge uses, so it takes the raw (seconds) values
    raw = {c: g.set_index("specimen_id") for c, g in res.groupby("condition")}
    shift = map_shift(df, raw)
    shift.to_csv(os.path.join(args.out, "map_shift.csv"))

    fig_paired(wide, cells, args.out)
    fig_effects(wide, gap, args.out)
    fig_trace(wide, cells, config.CONFIG.cache_dir, args.out)

    pd.set_option("display.width", 220)
    print("\n== reproduces database ==\n", check.round(3).to_string())
    print("\n== effects (median % change) ==\n",
          effects["median % change"].unstack(0)[ORDER[1:]].round(1).to_string())
    print("\n== by class ==\n", by_class.round(2).to_string())
    print("\n== real gap ==\n", gap.round(2).to_string())
    print("\n== map shift ==\n", shift.round(3).to_string())
    print("\n== upstroke % change by original upstroke ==\n", speed.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
