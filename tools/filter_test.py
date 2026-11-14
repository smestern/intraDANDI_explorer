"""Which harmonisation makes cells recorded at different sampling rates comparable?

The pipeline filters dv/dt at min(10 kHz, 0.8 x Nyquist), so a 10 kHz cell is filtered at 4 kHz and a
50 kHz cell at 10 kHz. Two ways to put every cell on the same footing are tested on real recordings:

    native        what the database holds (rerun, to check it reproduces)
    filter 4k     keep each cell's own rate, filter dv/dt at 4 kHz for everyone
    resample 10k  resample every cell above 10 kHz down to 10 kHz (zero-phase FIR anti-alias), then
                  run the pipeline as usual, which filters at 4 kHz. Cells at <= 10 kHz are untouched

Cells at <= 10 kHz come out the same under all three, so their database values are used as they are.

Two tests:
  1. Within dataset. Several datasets mix slow (< 25 kHz) and fast (>= 25 kHz) cells from the same lab
     and rig. A harmonisation that works shrinks the slow-vs-fast gap inside them.
  2. Whole database. A sample from every other dataset, scored on whether it mixes better with the
     real 10 kHz sets and whether fast-spiking cells still stand apart.

    python tools/filter_test.py --out figures/filter_test [--per-dataset 40] [--n-jobs 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from fractions import Fraction

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dandi_scraper import config  # noqa: E402
from dandi_scraper.analysis import scale_features  # noqa: E402
from downsample_test import FEATURES, load_cells, to_display_units  # noqa: E402

SEED = 0
TARGET_HZ = 10000.0
SLOW_KHZ = 25.0  # below this a cell's native dv/dt filter is under 10 kHz
CONDITIONS = ["native", "filter 4k", "resample 10k"]
KEY = ["ap_1_upstroke_0_long_square", "ap_1_downstroke_0_long_square", "ap_1_width_0_long_square",
       "ap_1_peak_v_0_long_square", "ap_1_threshold_v_0_long_square"]


def resample(t, v, i, fs):
    """Down to TARGET_HZ. Voltage goes through resample_poly's anti-alias FIR; the stimulus is a
    step, which an FIR would ring on, so it is just read off at the new sample times."""
    ratio = Fraction(TARGET_HZ / fs).limit_denominator(1000)
    v2 = signal.resample_poly(v, ratio.numerator, ratio.denominator)
    t2 = t[0] + np.arange(len(v2)) / TARGET_HZ
    # some files (e.g. the Tolias 25 kHz sets) store a stimulus of a different length than t
    return t2, v2, np.interp(t2, t[0] + np.arange(len(i)) / fs, i)


def measure_cell(specimen_id: str, analyzed_sweeps: str, cache_dir: str, cell_cache: str) -> list[dict]:
    path = os.path.join(cell_cache, hashlib.md5(specimen_id.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    from gigaseal.database.long_square import long_square_features, select_long_square_run
    from gigaseal.loadFile.loadNWB import GLOBAL_STIM_NAMES, loadNWB
    from ipfx.sweep import Sweep, SweepSet
    GLOBAL_STIM_NAMES.stim_inc = [""]
    GLOBAL_STIM_NAMES.stim_exc = []
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")

    try:
        sweep_ids = {int(x.split(":")[0]) for x in analyzed_sweeps.split(";")}
        _, _, _, data_set = loadNWB(os.path.join(cache_dir, specimen_id), return_obj=True)
        run = select_long_square_run(data_set, sweep_ids=sweep_ids)
    except Exception as e:
        rows = [{"specimen_id": specimen_id, "condition": c, "error": f"load: {e}"} for c in CONDITIONS]
        run = None
    if run is not None:
        rows = []
        for cond in CONDITIONS:
            row = {"specimen_id": specimen_id, "condition": cond}
            try:
                swps = []
                for s in run.sweeps.sweeps:
                    t, v, i = np.asarray(s.t), np.asarray(s.v), np.asarray(s.i)
                    fs = 1.0 / (t[1] - t[0])
                    if cond == "resample 10k" and fs > 1.05 * TARGET_HZ:
                        t, v, i = resample(t, v, i, fs)
                    swps.append(Sweep(t, v, i, clamp_mode="CurrentClamp",
                                      sampling_rate=int(round(1 / (t[1] - t[0]))), sweep_number=s.sweep_number))
                features, _ = long_square_features(SweepSet(swps), run.start, run.end, amp_interval=20,
                                                   max_filter_hz=4000 if cond == "filter 4k" else 10000)
                row.update({k: features.get(k, np.nan) for k in set(FEATURES) | set(config.COLS_TO_KEEP)})
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, default=float)
    return rows


def pick_cells(df: pd.DataFrame, per_dataset: int) -> pd.DataFrame:
    """Every slow cell in a dataset that also has fast cells, with up to ``per_dataset`` of its fast
    cells; plus up to ``per_dataset`` cells from every other dataset recorded above 10 kHz."""
    d = df[df["analyzed_sweeps"].notna() & (df["rate"] > 10.5)]
    picks = []
    for ds, g in d.groupby("ds"):
        slow, fast = g[g["rate"] < SLOW_KHZ], g[g["rate"] >= SLOW_KHZ]
        if len(slow) and len(fast):
            picks += [slow, fast.sample(min(max(per_dataset, len(slow)), len(fast)), random_state=SEED)]
        else:
            picks.append(g.sample(min(per_dataset, len(g)), random_state=SEED))
    return pd.concat(picks)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def feature_space(table: pd.DataFrame, scaler=None):
    """The merge's feature transform: log scales, then a RobustScaler fit on the reference set."""
    cols = list(config.COLS_TO_KEEP)
    x = scale_features(table[cols].copy()).replace([np.inf, -np.inf], np.nan)
    x = x[x.notna().all(axis=1)]
    if scaler is None:
        scaler = RobustScaler().fit(x.values)
    return pd.DataFrame(np.clip(scaler.transform(x.values), -config.ROBUST_CLIP, config.ROBUST_CLIP),
                        index=x.index, columns=cols), scaler


def auc(X: np.ndarray, y: np.ndarray) -> float:
    if min(np.bincount(y.astype(int), minlength=2)) < 5:
        return np.nan
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(LogisticRegression(max_iter=500, class_weight="balanced"),
                          StandardScaler().fit_transform(X), y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def within_dataset(cells: pd.DataFrame, wide: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slow minus fast median, per mixed dataset and condition, and how well features tell them apart."""
    mixed = cells.groupby("ds")["rate"].agg(lambda r: (r < SLOW_KHZ).any() and (r >= SLOW_KHZ).any())
    mixed = mixed[mixed].index
    gaps, aucs = [], []
    for cond in CONDITIONS:
        disp = to_display_units(wide[cond].reset_index()).set_index("specimen_id")
        z, _ = feature_space(wide[cond])
        for ds in mixed:
            c = cells[cells["ds"] == ds].set_index("specimen_id")
            slow = c.index[c["rate"] < SLOW_KHZ].intersection(disp.index)
            fast = c.index[c["rate"] >= SLOW_KHZ].intersection(disp.index)
            row = {"condition": cond, "ds": ds, "n slow": len(slow), "n fast": len(fast)}
            for f in KEY:
                row[FEATURES[f]] = disp.loc[slow, f].median() - disp.loc[fast, f].median()
            gaps.append(row)
            idx = z.index.intersection(slow.append(fast))
            aucs.append({"condition": cond, "ds": ds, "n slow": len(slow),
                         "AUC slow vs fast": auc(z.loc[idx].values, np.isin(idx, slow).astype(int))})
    return pd.DataFrame(gaps), pd.DataFrame(aucs)


def whole_database(df: pd.DataFrame, cells: pd.DataFrame, wide: dict) -> pd.DataFrame:
    """Test cells under each condition, scored together with the real 10 kHz cells (whose values
    don't change under any condition)."""
    ten = df[df["rate"] <= 10.5].set_index("specimen_id")
    fs = df.set_index("specimen_id")["fast spiking"]
    ds_of = df.set_index("specimen_id")["ds"]
    rows = {}
    for cond in CONDITIONS:
        table = pd.concat([ten[list(config.COLS_TO_KEEP)], wide[cond][list(config.COLS_TO_KEEP)]])
        table = table[~table.index.duplicated()]
        z, _ = feature_space(table)
        is_ten = z.index.isin(ten.index)
        is_fs = fs.reindex(z.index).fillna(False).to_numpy(bool)
        ds = ds_of.reindex(z.index).to_numpy()
        nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(z.values)
        _, idx = nn.kneighbors(z.values)
        idx = idx[:, 1:]
        test = ~is_ten
        # pairwise AUC between each real 10 kHz set and each large tested set
        pair = []
        big_ten = [d for d, n in pd.Series(ds[is_ten]).value_counts().items() if n >= 30]
        big_test = [d for d, n in pd.Series(ds[test]).value_counts().items() if n >= 30]
        for a in big_ten:
            for b in big_test:
                m = (ds == a) | (ds == b)
                pair.append(auc(z.values[m], (ds[m] == a).astype(int)))
        rows[cond] = {
            "test cells: share of neighbours from 10 kHz sets": is_ten[idx[test]].mean(),
            "10 kHz cells: share of neighbours from test cells": (~is_ten[idx[is_ten]]).mean(),
            "test cells: same-dataset share": (ds[idx[test]] == ds[test, None]).mean(),
            "FS cells: share of FS neighbours": is_fs[idx[is_fs]].mean(),
            "FS vs other AUC (test cells)": auc(z.values[test], is_fs[test].astype(int)),
            "median AUC, 10 kHz set vs tested set": np.nanmedian(pair),
            "n test cells": int(test.sum()), "n 10 kHz cells": int(is_ten.sum()),
        }
    return pd.DataFrame(rows).T


def effects(cells: pd.DataFrame, wide: dict) -> pd.DataFrame:
    disp = {c: to_display_units(w.reset_index()).set_index("specimen_id") for c, w in wide.items()}
    fs = cells.set_index("specimen_id")["fast spiking"]
    rows = {}
    for cond in CONDITIONS[1:]:
        for label, mask in [("fast-spiking", fs), ("other", ~fs)]:
            ids = fs.index[mask].intersection(disp[cond].index).intersection(disp["native"].index)
            for f in KEY:
                a, b = disp["native"].loc[ids, f], disp[cond].loc[ids, f]
                rows[(cond, label, FEATURES[f])] = {"native median": a.median(), "harmonised median": b.median(),
                                                    "median % change": (100 * (b - a) / a.abs()).median()}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=config.CONFIG.merged_csv)
    p.add_argument("--out", default="figures/filter_test")
    p.add_argument("--per-dataset", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--limit", type=int, help="Only run this many cells, for a quick check.")
    args = p.parse_args(argv)
    cell_cache = os.path.join(args.out, "cache")
    os.makedirs(cell_cache, exist_ok=True)

    df = load_cells(args.csv)
    rates = pd.read_csv(os.path.join(config.CONFIG.output_dir, "sampling_rates.csv"),
                        index_col=0)["sampling_rate_khz"]
    df["rate"] = df["specimen_id"].map(rates)
    cells = pick_cells(df, args.per_dataset)
    if args.limit:
        cells = cells.sample(args.limit, random_state=SEED)
    print(f"{len(cells)} cells from {cells['ds'].nunique()} datasets, "
          f"{int((cells['rate'] < SLOW_KHZ).sum())} of them slow")

    results = joblib.Parallel(n_jobs=args.n_jobs, verbose=5)(
        joblib.delayed(measure_cell)(sid, sw, config.CONFIG.cache_dir, cell_cache)
        for sid, sw in zip(cells["specimen_id"], cells["analyzed_sweeps"]))
    res = pd.DataFrame([r for rows in results for r in rows])
    res.to_csv(os.path.join(args.out, "cell_results.csv"), index=False)
    if "error" in res:
        print("failures by condition:\n", res[res["error"].notna()].groupby("condition").size().to_string())
    ok = res[res.get("error", pd.Series(np.nan, index=res.index)).isna()]
    wide = {c: g.set_index("specimen_id") for c, g in ok.groupby("condition")}

    stored = cells.set_index("specimen_id")
    check = {}
    for f, name in FEATURES.items():
        a, b = wide["native"][f], stored[f].reindex(wide["native"].index)
        m = a.notna() & b.notna()
        check[name] = {"correlation": np.corrcoef(a[m], b[m])[0, 1], "n": int(m.sum()),
                       "share within 1%": ((a[m] - b[m]).abs() <= 0.01 * b[m].abs() + 1e-9).mean()}
    check = pd.DataFrame(check).T
    check.to_csv(os.path.join(args.out, "reproduces_database.csv"))

    gaps, aucs = within_dataset(cells, wide)
    gaps.to_csv(os.path.join(args.out, "within_dataset_gaps.csv"), index=False)
    aucs.to_csv(os.path.join(args.out, "within_dataset_auc.csv"), index=False)
    whole = whole_database(df, cells, wide)
    whole.to_csv(os.path.join(args.out, "whole_database.csv"))
    eff = effects(cells, wide)
    eff.to_csv(os.path.join(args.out, "effects_by_class.csv"))

    pd.set_option("display.width", 250)
    print("\n== native reproduces database ==\n", check.round(3).to_string())
    print("\n== slow minus fast, median over mixed datasets ==\n",
          gaps.groupby("condition")[[FEATURES[f] for f in KEY]].median().loc[CONDITIONS].round(3).to_string())
    print("\n== per dataset ==\n", gaps.round(3).to_string(index=False))
    print("\n== slow vs fast AUC ==\n", aucs.pivot(index="ds", columns="condition", values="AUC slow vs fast")
          [CONDITIONS].round(3).to_string())
    print("\n== whole database ==\n", whole.T.round(3).to_string())
    print("\n== effects by class ==\n", eff.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
