"""Per-cell long square feature extraction for DANDI current clamp NWBs.

Run selection and feature extraction are gigaseal.database.long_square; this module
supplies the DANDI policy (which run to analyze), the CSV row format, and a QC debug
plot next to the dandiset cache folder.
"""
import os
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gigaseal.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from gigaseal.database.build_database import glob_files
from gigaseal.database.long_square import select_long_square_run, long_square_features


def run_analysis(folder, outfile='out.csv', ext="nwb", parallel=False, n_jobs=4):
    files = glob_files(folder, ext=ext)[::-1]
    file_idx = np.arange(len(files))
    import joblib
    GLOBAL_STIM_NAMES.stim_inc = ['']
    GLOBAL_STIM_NAMES.stim_exc = []
    get_data_partial = partial(data_for_specimen_id,
                            passed_only=False,
                            data_source='filesystem',
                            ontology=None,
                            file_list=files)
    if parallel == True:
        if n_jobs is None:
            n_jobs = joblib.cpu_count()
        results = joblib.Parallel(n_jobs=n_jobs, backend='multiprocessing')(joblib.delayed(get_data_partial)(specimen_id) for specimen_id in file_idx)
    else:
        results = [get_data_partial(specimen_id) for specimen_id in file_idx]

    results = pd.DataFrame().from_dict(results).set_index('specimen_id')
    return results


def _rank_long_square_group(g, start_times, end_times, amps, volts, target_duration=1.0):
    """Sort key for picking the run to analyze, lower is better: step duration closest to
    target_duration, has both hyperpolarizing and depolarizing steps, earliest in the recording
    (most likely baseline), then most sweeps."""
    duration = end_times[g[0]] - start_times[g[0]]
    a = np.asarray(amps)[g]
    v = np.asarray(volts)[g]
    coverage = int((a < 0).any()) + int((a > 0).any())
    vm_coverage = int((v < -60).any()) + int((v > -10).any())
    return (round(abs(duration - target_duration), 2), -coverage, -vm_coverage, g[0]//10, -len(g))


def data_for_specimen_id(specimen_id, passed_only, data_source, ontology, file_list=None, amp_interval=20, max_above_rheo=None, debug=True):
    #Long square features for one file, as one CSV row. Failures and rejections are recorded in
    #error_stage/error_class/error_message, which analysis.py and database.py summarize.
    result = {}
    result["specimen_id"] = file_list[specimen_id]
    stage = "load"
    try:
        #ipfx does not play nice with many NWBs on dandi, so we load with gigaseal and use the lower level ipfx functions
        _, _, _, data_set = loadNWB(file_list[specimen_id], return_obj=True)
        if data_set is None or len(data_set.dataY)<1:
            result.update(error_stage=stage, error_class="EmptyFile", error_message="no sweeps loaded")
            return result

        stage = "select"
        run = select_long_square_run(data_set, rank=_rank_long_square_group)
        sweeps, start_time, end_time = run.sweeps, run.start, run.end
        debug_log, traces = run.log, run.traces
        if run.primary is not None:
            cands, starts, ends = run.candidates, run.start_times, run.end_times
            result["n_protocol_groups"] = len(run.groups)
            result["protocol_groups"] = "|".join(
                f"{ends[g[0]] - starts[g[0]]:.3f}s:" + ",".join(str(cands[k].sweep_number) for k in g)
                for g in run.groups)
            #record the raw sweep ids (and stim amps) of the analyzed run, so plot_data can reuse them without re-deriving
            result["analyzed_sweeps"] = ";".join(f"{cands[k].sweep_number}:{run.amps[k]:.1f}" for k in run.primary)
            result["analyzed_window"] = f"{start_time:.4f}-{end_time:.4f}"

        stage = "debug_plot"
        # ========== Debugging and QC Visualization ==========
        if debug:
            fig, ax = plt.subplots(2, 1) #multiple subplots for QC visualization, one for passed sweeps and one for failed sweeps
            labelled = (set(), set()) #categories already in each subplot's legend, so we get one legend entry per category rather than per sweep
            other_colors = {} #distinct color per non-"failed QC" failure category, so the legend entries can be told apart
            color_cycle = (c for c in plt.cm.tab10.colors[1:] if c != plt.cm.tab10.colors[3]) #skip blue/red, used for start time and "failed QC"
            def _label(ax_idx, category):
                if category in labelled[ax_idx]:
                    return "_nolegend_"
                labelled[ax_idx].add(category)
                return category
            for sweep in debug_log.keys():
                category = debug_log[sweep]
                if category.startswith("passed QC"):
                    print(f"sweep {sweep} {category}")
                    #plot analyzed sweeps in green, passed sweeps from other protocol runs in orange
                    ax[1].plot(*traces[sweep], label=_label(1, category),
                               c='tab:orange' if "not used" in category else 'g')
                    continue
                print(f"sweep {sweep} failed QC because it was {category}")
                if category == "failed QC":
                    ax[0].plot(*(traces[sweep] if sweep in traces else (data_set.dataX[sweep], data_set.dataY[sweep])), label=_label(0, category), c='r') #plot failed sweeps in red
                else:
                    if category not in other_colors:
                        other_colors[category] = next(color_cycle, 'k') #fall back to black if we run out of colors
                    ax[0].plot(*(traces[sweep] if sweep in traces else (data_set.dataX[sweep], data_set.dataY[sweep])), label=_label(0, category), c=other_colors[category]) #plot other failed sweeps by category
                    #continue
            ax[1].axvline(x=start_time, color='b', linestyle='--', label='start time') if start_time is not None else None #mark the most common start time
            ax[1].axvline(x=end_time, color='m', linestyle='--', label='end time') if end_time is not None else None #mark the most common end time
            ax[0].legend(loc='right', bbox_to_anchor=(1.2, 0.5))
            ax[1].legend(loc='right', bbox_to_anchor=(1.2, 0.5))

            dandi_set_no = os.path.dirname(file_list[specimen_id])
            dir_base = os.path.dirname(file_list[specimen_id])
            dir_of_the_dir = os.path.dirname(dir_base)
            os.makedirs(f"{dir_of_the_dir}/debug/{os.path.basename(dir_base)}", exist_ok=True)
            fig.savefig(f"{dir_of_the_dir}/debug/{os.path.basename(dir_base)}/{os.path.basename(file_list[specimen_id])}.png")
            plt.close()
            #plt.pause(0.2)

        if len(sweeps.v) < 1:
            result.update(error_stage="select", error_class="NoLongSquareRun", error_message="no long square sweeps passed QC")
            return result

        stage = "features"
        features, sweep_indexes = long_square_features(sweeps, start_time, end_time,
                                                       amp_interval=amp_interval, max_above_rheo=max_above_rheo)
        result.update(features)

        #we should record the name of the stimuli used and the sweeps used, for either plotting or debugging
        result["stimulus_name"] = data_set.sweepMetadata[0]['stim_dict']['description']
        result["sweeps_used"] = np.array2string(sweep_indexes, separator=";", threshold=np.inf, max_line_width=np.inf)

    except Exception as e:
        print("error with specimen_id: ", specimen_id)
        print(e)
        result.update(error_stage=stage, error_class=type(e).__name__, error_message=str(e))
        plt.close()
        return result
    plt.close()
    return result
