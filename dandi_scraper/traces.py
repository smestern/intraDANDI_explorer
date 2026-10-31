"""Per-cell trace and FI curve SVGs for the explorer's plots.

Moved from gigaseal.database.build_database. plot_data reuses the sweeps that
extract.data_for_specimen_id analyzed (via its CSV row) when available, and otherwise
picks the run with the same ranking.
"""
import os
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt

from ipfx.sweep import SweepSet

from gigaseal.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from gigaseal.featureExtractor import analyze_spike_times
from gigaseal.database.build_database import glob_files
from gigaseal.database.protocol_detect import find_time_index
from gigaseal.database.long_square import select_long_square_run

from .extract import _rank_long_square_group




def _sweeps_from_row(row):
    """Get the raw sweep ids of the long square run data_for_specimen_id analyzed, from its output CSV row.
    Uses analyzed_sweeps if present, otherwise falls back to older CSVs: sweeps_used holds positions within the
    analyzed run (ipfx resets the index), so pick the protocol group they fit in, preferring the step closest to 1s
    (same first key as _rank_long_square_group). Returns a list of sweep ids, or None if the row has no usable info."""
    if row is None:
        return None
    analyzed = row.get("analyzed_sweeps")
    if isinstance(analyzed, str) and analyzed:
        return [int(x.split(":")[0]) for x in analyzed.split(";")]
    sweeps_used, groups = row.get("sweeps_used"), row.get("protocol_groups")
    if not isinstance(sweeps_used, str) or not isinstance(groups, str):
        return None
    positions = [int(x) for x in sweeps_used.strip("[]").replace(";", " ").split()]
    parsed = []
    for g in groups.split("|"):
        duration, ids = g.split(":")
        parsed.append((float(duration.rstrip("s")), [int(x) for x in ids.split(",")]))
    fits = [(d, ids) for d, ids in parsed if not positions or max(positions) < len(ids)]
    if not fits:
        return None
    return min(fits, key=lambda x: round(abs(x[0] - 1.0), 2))[1]


def build_dataset_traces(folder, ids =None, ext="nwb", parallel=True, sweep_info=None, overwrite_older_than=None):
    # this function will take a run_analysis function and return a dataset of traces
    # sweep_info is an optional {id: csv row} of data_for_specimen_id output, keyed like ids, so plot_data can reuse the analyzed sweeps
    # overwrite_older_than is passed through to plot_data to regenerate svgs that predate the latest analysis run
    files = glob_files(folder)[::-1]
    import joblib
    if ids is not None:
        #ids are cache-relative (code/subject/file.nwb); match the same relpath, not a hardcoded /dandi prefix
        cache_root = os.path.dirname(os.path.normpath(folder))
        ids = set(ids)
        files_check = [os.path.relpath(x, cache_root) in ids for x in files]
        files = np.array(files)[files_check]
    sweep_info = sweep_info or {}
    rows = [next((r for k, r in sweep_info.items() if f.endswith(k)), None) for f in files]

    file_idx = np.arange(len(files))
    GLOBAL_STIM_NAMES.stim_inc = ['']
    #GLOBAL_STIM_NAMES.stim_type = ['CurrentClamp', 'CC', 'IC', 'Current Clamp'] #drops non-CC series in the loader, shifting sweep ids vs analyzed_sweeps
    if parallel == True:
        #Run in parallel
        parallel = 2 # nwb loads into memory, so keep this consservative
    else:
        parallel = 1
    #results = joblib.Parallel(n_jobs=parallel)(joblib.delayed(plot_data)(specimen_id, files, sweep_info=rows[specimen_id], overwrite_older_than=overwrite_older_than) for specimen_id in file_idx)
    #don't keep plot_data's return: it holds figures and views of full sweep arrays, so accumulating it grows memory per file until OOM
    #workers return None and the generator is drained without storing, so nothing piles up in the parent either
    jobs = (joblib.delayed(_plot_data_discard)(specimen_id, files, sweep_info=rows[specimen_id], overwrite_older_than=overwrite_older_than) for specimen_id in file_idx)
    for _ in joblib.Parallel(n_jobs=parallel, return_as="generator_unordered")(jobs):
        pass



def _plot_data_discard(*args, **kwargs):
    #module-level so joblib can pickle it; Agg since workers only save svgs and never show
    plt.switch_backend("Agg")
    try:
        plot_data(*args, **kwargs)
    except Exception:
        #one bad file shouldn't kill the whole dataset run
        logging.exception(f"plot_data failed for {args[1][args[0]] if len(args) > 1 else args}")
    finally:
        plt.close('all')
    return None




def _as_timestamp(value):
    """Normalize value to epoch seconds for mtime comparison. Accepts None, a datetime,
    an epoch number, or a path to an existing file (its modification time is used)."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and os.path.exists(value):
        return os.path.getmtime(value)
    return None


def plot_data(specimen_id, file_list=None, target_amps=[-100, -20, 20, 100, 150, 250, 500, 1000], overwrite=False, save=True, stim_override=None, sweep_info=None, overwrite_older_than=None) -> dict:
    #sweep_info is this file's row of data_for_specimen_id output (dict or Series); if given, plot the sweeps it analyzed
    #rather than re-deriving the long square run
    #overwrite_older_than: optional datetime, epoch seconds, or path; if the existing svg predates it, regenerate it even when overwrite is False
    sweep_ids = _sweeps_from_row(sweep_info)
    result = {}
    svg_path = f"{file_list[specimen_id]}.svg"
    if os.path.exists(svg_path) and not overwrite:
        threshold = _as_timestamp(overwrite_older_than)
        if threshold is not None and os.path.getmtime(svg_path) < threshold:
            print(f"regenerating {file_list[specimen_id]} because it predates the latest analysis run")
            logging.debug(f"regenerating {file_list[specimen_id]} because it predates the latest analysis run")
        else:
            print(f"skipping {file_list[specimen_id]} because it already exists")
            logging.debug(f"skipping {file_list[specimen_id]} because it already exists")
            return result

    if stim_override is not None:
        GLOBAL_STIM_NAMES.stim_inc = [stim_override]
    else:
        GLOBAL_STIM_NAMES.stim_inc = ['']
        GLOBAL_STIM_NAMES.stim_exc = []

    result["specimen_id"] = file_list[specimen_id]
    _, _, _,  data_set = loadNWB(file_list[specimen_id], return_obj=True)
    if data_set is None or len(data_set.dataY)<1:
        print(f"data_set is None for {file_list[specimen_id]}")
        return result
    #with sweep_ids, replot exactly the run data_for_specimen_id analyzed; otherwise pick it the same way
    run = select_long_square_run(data_set, rank=_rank_long_square_group, sweep_ids=sweep_ids)
    if len(run.sweeps.sweeps) < 1:
        return result
    sweeps = list(run.sweeps.sweeps)
    start_time, end_time = run.start, run.end
    sweep_amp = run.amps_used
    idx_pass = np.arange(len(sweeps))

    sweep_amp_filter = np.array(sweep_amp)[idx_pass]
    if len(sweep_amp_filter) < 1:
        return result
    #the sweeps closest to the target amp
    idx_targets = [np.argmin(np.abs(sweep_amp_filter - x)) for x in target_amps]

    idx_pass = idx_pass[idx_targets]

    #sweeps = np.array(sweeps, dtype=object)[idx_pass]

    sweeps = SweepSet(sweeps)
    #plot the sweeps
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    fi_s = []
    fi_i = []
    plotted = []
    sweep_ys = []
    sweep_xs = []
    for i, sweep in enumerate(sweeps.sweeps):
        idx_start = find_time_index(sweep.t, np.clip(start_time*0.8, 0.0, np.inf))
        idx_end = find_time_index(sweep.t, np.clip(end_time*1.4, end_time, sweep.t[-1]))
        if i in idx_pass and i not in plotted:
            plotted.append(i)
            ax.plot(sweep.t[idx_start:idx_end], sweep.v[idx_start:idx_end], label=f"sweep {i}", c='k', alpha=0.5)
            sweep_ys.append(sweep.v[idx_start:idx_end])
            sweep_xs.append(sweep.t[idx_start:idx_end])

        spikes = analyze_spike_times(sweep.t, sweep.v, sweep.i)
        fi_s.append(len(spikes) / (end_time - start_time))
        fi_i.append(sweep_amp[i])
        #ax.plot(sweep.t[idx_start:idx_end], sweep.v[idx_start:idx_end])



    #turn off the upper and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #save the figure as a svg

    plt.savefig(f"{file_list[specimen_id]}.svg", bbox_inches='tight') if save==True else None

    fig2,ax = plt.subplots(figsize=(3,3))

    #if there is the same sweep_amp, average it
    fi_s = np.array(fi_s)
    fi_i = np.array(fi_i)

    #get duplicates of fi_i
    unique, counts = np.unique(fi_i, return_counts=True)
    duplicates = unique[counts > 1]
    for dup in duplicates:
        idx_dup = np.where(fi_i == dup)[0]
        fi_s[idx_dup] = np.mean(fi_s[idx_dup])
        fi_i[idx_dup] = np.mean(fi_i[idx_dup])
    #sort the fi_i and fi_s
    idx_sort = np.argsort(fi_i)
    fi_i = fi_i[idx_sort]
    fi_s = fi_s[idx_sort]
    #plot the fi curve


    plt.plot(fi_i, fi_s, c='k', marker='o')
    plt.xticks(np.arange(target_amps[0], target_amps[-1]+250, 250))
    plt.xlim( target_amps[0]-100, target_amps[-1]+100)
    plt.xlabel("Current (pA)")
    plt.ylabel("Firing Rate (Hz)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f"{file_list[specimen_id]}_FI.svg", bbox_inches='tight') if save==True else None
    plt.close('all') if save==True else None



    print(f"saved {file_list[specimen_id]}.svg", end="\r") if save==True else None
    return {'fig_trace': fig, 'fi_i': fi_i, 'fi_s': fi_s, 'sweep_ys': sweep_ys, 'sweep_xs': sweep_xs, 'fi_fig':  fig2}

