"""Per-cell long square feature extraction for DANDI current clamp NWBs.

Moved from gigaseal.database.build_database: picks the long square run to analyze
from each file, runs ipfx's LongSquareAnalysis on it, and writes a QC debug plot
next to the dandiset cache folder. The stimulus and sweep primitives stay in
gigaseal.database.protocol_detect.
"""
import os
import logging
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ipfx.bin import run_feature_collection
from ipfx.sweep import SweepSet, Sweep
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.feature_vectors as fv

from gigaseal.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from gigaseal.database.build_database import glob_files
from gigaseal.database.protocol_detect import (
    _iter_current_clamp_traces, _group_long_square_runs, match_protocol,
    get_stim_characteristics, QC_voltage_data,
)


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
    
    result = {}
    result["specimen_id"] = file_list[specimen_id]
    #if True:
    try:
        #this is a clone of the function in ipfx/bin/run_feature_collection.py,
        # here we are gonna try to use it to handle data that may not be in an NWB format IPFX can handle
        _, _, _, data_set = loadNWB(file_list[specimen_id], return_obj=True)
        if data_set is None or len(data_set.dataY)<1:
            return result
        #here we are going to perform long square analysis on the data, 
        #ipfx does not play nice with many NWBs on dandi, so we are going to link into the lower level functions
        #and do the analysis ourselves
        #hopefully this will be fixed in the future and we can use ipfx for this
        sweeps = []
        start_times = []
        end_times = []
        stim_amps = []
        debug_log = {}
        volts_range = []
        traces = {} #the (t, v) actually analyzed per sweep id, for the debug plot
        for sweep, t, v, i, skip_reason in _iter_current_clamp_traces(data_set):
            if skip_reason is not None:
                logging.debug(f"sweep {sweep} skipped: {skip_reason}")
                debug_log[sweep] = skip_reason
                continue
            traces[sweep] = (t, v)
            dt = t[1] - t[0]

            #try to figure out if this is a long square
            if match_protocol(i, t) != "Long Square":
                logging.debug(f"skipping sweep {sweep} because it is not a long square")
                debug_log[sweep] = "likely not a long square"
                continue

            start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t)

            if QC_voltage_data(t, v, i) == 0:
                logging.debug(f"skipping sweep {sweep} because it failed QC")
                debug_log[sweep] = "failed QC"
                continue
            #construct a sweep obj

            #again floating point errors can cause issues here, so we will round to the nearest ms, pA, and uV
            #t = np.round(t, decimals=3)
            i = np.round(i, decimals=1) #since we are doing long square, round to nearest 0.1 pA, 
            #if this was ramp or noise or something this is bad.
            #v = np.round(v, decimals=3)
            debug_log[sweep] = f"passed QC {start_time:.3f}-{start_time+duration:.3f}s"
            start_times.append(start_time)
            end_times.append(start_time+duration)
            sweep_item = Sweep(t, v, i, clamp_mode="CurrentClamp", sampling_rate=int(1/dt), sweep_number=sweep)
            sweeps.append(sweep_item)
            stim_amps.append(amplitude)
            volts_range.append([v.min(), v.max()])
        #split the passing sweeps into runs of the same protocol, then analyze only the best run
       
        groups = _group_long_square_runs(start_times, end_times)
        if groups:
            primary = min(groups, key=lambda g: _rank_long_square_group(g, start_times, end_times, stim_amps, volts_range))
            #onsets can differ within a group, so shift each sweep's time axis to line up on the latest onset
            start_time = max(start_times[k] for k in primary)
            end_time = start_time + float(np.median([end_times[k] - start_times[k] for k in primary]))
            for n, g in enumerate(groups):
                if g is not primary:
                    for k in g:
                        debug_log[sweeps[k].sweep_number] += f", not used (group {n})"
            result["n_protocol_groups"] = len(groups)
            result["protocol_groups"] = "|".join(
                f"{end_times[g[0]] - start_times[g[0]]:.3f}s:" + ",".join(str(sweeps[k].sweep_number) for k in g)
                for g in groups)
            #record the raw sweep ids (and stim amps) of the analyzed run, so plot_data can reuse them without re-deriving
            result["analyzed_sweeps"] = ";".join(f"{sweeps[k].sweep_number}:{stim_amps[k]:.1f}" for k in primary)
            result["analyzed_window"] = f"{start_time:.4f}-{end_time:.4f}"
            sweeps = SweepSet([Sweep(sweeps[k].t + (start_time - start_times[k]), sweeps[k].v, sweeps[k].i,
                                     clamp_mode="CurrentClamp", sampling_rate=int(round(1 / (sweeps[k].t[1] - sweeps[k].t[0]))),
                                     sweep_number=sweeps[k].sweep_number) for k in primary])
        else:
            logging.error("No long square sweeps passed QC")
            sweeps = SweepSet([])
            start_time, end_time = None, None

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


        if len(sweeps.v) < 1: #no valid sweeps to analyze, should log this??
            return result


        # ====== DATA EXTRACTION ======
        
        lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
            sweeps,
            start=start_time , #if the start times are not the same, this will fail
            end=end_time, #if the end times are not the same, this will fail
            min_peak=-25,
        )

        #ipfx defaults to a filter @10kHz, this breaks for signals sampled at or below 20kHz, so cap it below Nyquist.
        #Do NOT put the cutoff right at Nyquist (we used to use 0.99999*Nyquist): the 4-pole bessel then has a pole
        #just outside the unit circle, filtfilt grows ~1.0001^N, and long sweeps give dv/dt up to ~1e23 mV/ms
        min_sample_rate = min(sweeps.sampling_rate)
        filter_cutoff = min(10000, 0.8 * min_sample_rate / 2)
        if filter_cutoff < 10000:
            print(f"Warning: Minimum sample rate is {min_sample_rate} Hz, lowering the dv/dt filter cutoff to {filter_cutoff:.0f} Hz.")

        lsq_spx.filter = filter_cutoff / 1000 #in kHz

        lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx,
            subthresh_min_amp=-200.0)
        if start_time < 0.01:
           lsq_an.sptx.baseline_interval = start_time*0.1
           lsq_an.sptx.sag_baseline_interval = start_time*0.1
        lsq_features = lsq_an.analyze(sweeps)

        result.update({
                "input_resistance": lsq_features["input_resistance"],
                "tau": lsq_features["tau"],
                "v_baseline": lsq_features["v_baseline"],
                "sag_nearest_minus_100": lsq_features["sag"],
                "sag_measured_at": lsq_features["vm_for_sag"],
                "rheobase_i": int(lsq_features["rheobase_i"]),
                "fi_linear_fit_slope": lsq_features["fi_fit_slope"],
            })
        # Identify suprathreshold set for analysis
        sweep_table = lsq_features["spiking_sweeps"]
        mask_supra = sweep_table["stim_amp"] >= lsq_features["rheobase_i"]
        sweep_indexes = fv._consolidated_long_square_indexes(sweep_table.loc[mask_supra, :])
        amps = np.rint(sweep_table.loc[sweep_indexes, "stim_amp"].values - lsq_features["rheobase_i"])
        spike_data = lsq_features["spikes_set"]

        # Assign each sweep to the nearest amp_interval step above rheobase, keeping the
        # closest sweep per step. Requiring an exact multiple of amp_interval left cells
        # recorded with other step sizes (e.g. 10, 25, 100 pA) with only the rheobase sweep.
        best_by_label = {}
        for amp, swp_ind in zip(amps, sweep_indexes):
            amp_label = int(np.floor(amp / amp_interval + 0.5))
            if (amp < 0) or (max_above_rheo is not None and amp_label * amp_interval > max_above_rheo):
                continue
            offset = abs(amp - amp_label * amp_interval)
            if amp_label not in best_by_label or offset < best_by_label[amp_label][0]:
                best_by_label[amp_label] = (offset, amp, swp_ind)

        for amp_label, (_, amp, swp_ind) in sorted(best_by_label.items()):

            first_spike_lsq_sweep_features = run_feature_collection.first_spike_lsq(spike_data[swp_ind])
            result.update({"ap_1_{:s}_{:d}_long_square".format(f, amp_label): v
                                for f, v in first_spike_lsq_sweep_features.items()})

            mean_spike_lsq_sweep_features = run_feature_collection.mean_spike_lsq(spike_data[swp_ind])
            result.update({"ap_mean_{:s}_{:d}_long_square".format(f, amp_label): v
                                for f, v in mean_spike_lsq_sweep_features.items()})

            sweep_feature_list = [
                "first_isi",
                "avg_rate",
                "isi_cv",
                "latency",
                "median_isi",
                "adapt",
            ]

            result.update({"{:s}_{:d}_long_square".format(f, amp_label): sweep_table.at[swp_ind, f]
                                for f in sweep_feature_list})
            result["stimulus_amplitude_{:d}_long_square".format(amp_label)] = int(amp + lsq_features["rheobase_i"])

        rates = sweep_table.loc[sweep_indexes, "avg_rate"].values
        result.update(run_feature_collection.fi_curve_fit(amps, rates))

        #we should record the name of the stimuli used and the sweeps used, for either plotting or debugging
        result["stimulus_name"] = data_set.sweepMetadata[0]['stim_dict']['description']
        result["sweeps_used"] = np.array2string(sweep_indexes, separator=";", threshold=np.inf, max_line_width=np.inf)


    except Exception as e:
        print("error with specimen_id: ", specimen_id)
        print(e)
        plt.close()
        return result
    plt.close()
    return result

