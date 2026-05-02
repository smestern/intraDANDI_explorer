"""Web visualization server entry point (`build_server`)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

import pyAPisolation.webViz.run_web_viz as wbz
import pyAPisolation.webViz.webVizConfig as wvc
from pyAPisolation.loadFile.loadNWB import GLOBAL_STIM_NAMES

from . import config


def build_server(
    database_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Configure and launch the static web visualization."""
    output_dir = output_dir or config.CONFIG.output_dir
    database_csv = database_csv or os.path.join(output_dir, "all_new.csv")
    shuffled_csv = os.path.join(output_dir, "all2_2.csv")

    GLOBAL_STIM_NAMES.stim_inc = [""]
    GLOBAL_VARS = wvc.webVizConfig()
    GLOBAL_VARS.file_index = "specimen_id"
    GLOBAL_VARS.file_path = "specimen_id"
    GLOBAL_VARS.table_vars_rq = [
        "specimen_id", "ap_1_width_0_long_square", "input_resistance", "tau",
        "v_baseline", "sag_nearest_minus_100", "ap_1_threshold_v_0_long_square",
        "ap_1_peak_v_0_long_square", "file_link", "dandiset_link", "meta_data_link",
    ]
    GLOBAL_VARS.table_vars = [
        "input_resistance", "tau", "v_baseline", "sag_nearest_minus_100",
        "species", "brain_region",
    ]
    GLOBAL_VARS.para_vars = [
        "ap_1_width_0_long_square", "input_resistance", "tau", "v_baseline",
        "sag_nearest_minus_100", "species", "brain_region",
    ]
    GLOBAL_VARS.para_var_colors = "ap_1_width_0_long_square"
    GLOBAL_VARS.umap_labels = [
        "dandiset label", "species", "brain_region", "contributor",
        "GMM cluster label",
        {"Ephys Feat:": config.COLS_TO_KEEP},
    ]
    GLOBAL_VARS.plots_path = "."
    GLOBAL_VARS.umap_cols = ["umap X", "umap Y"]
    GLOBAL_VARS.hidden_table = True
    GLOBAL_VARS.hidden_table_vars = ["dandiset label", "species"]
    GLOBAL_VARS.db_title = "Icephys Dandiset Visualization"
    GLOBAL_VARS.db_description = (
        " This is a visualization of some of the intracellular electrophysiology "
        "(icephys) data found across the open \n    neuroscience initiative DANDI. "
        "The data is visualized using a UMAP and a parallel coordinates plot. "
        "The data is also visualized in a table format. \n    This is currently a "
        "work in progress and is not yet complete. Please cite the original authors "
        "of the data when using this data. "
    )
    GLOBAL_VARS.db_subtitle = ""
    GLOBAL_VARS.db_links = {
        "about this project": "https://www.smestern.com/intraDANDI_explorer/dandi_scraper/notes_on_intra_ephys.html",
        "Dandi": "https://dandiarchive.org/",
        "smestern on X": "https://twitter.com/smestern",
    }
    GLOBAL_VARS.db_para_title = "Paracoords"
    GLOBAL_VARS.db_embed_title = "UMAP"

    GLOBAL_VARS.col_rename = {
        "ap_1_width_0_long_square": "Rheo-AP width Log[(ms)]",
        "sag_nearest_minus_100": "Sag",
        "input_resistance": "Input resistance Log[(MOhm)]",
        "tau": "Tau Log[(ms)]",
        "ap_1_threshold_v_0_long_square": "Rheo-AP Threshold (mV)",
        "ap_1_peak_v_0_long_square": "Rheo-AP Peak (mV)",
        "ap_1_upstroke_0_long_square": "Rheo-AP Upstroke (mV/ms)",
        "ap_1_fast_trough_v_0_long_square": "Rheo-AP Fast Trough (mV)",
        "ap_mean_threshold_v_0_long_square": "Mean AP Threshold (mV)",
        "ap_mean_peak_v_0_long_square": "Mean AP Peak (mV)",
        "ap_mean_upstroke_0_long_square": "Mean AP Upstroke (mV/ms)",
        "ap_mean_width_0_long_square": "Mean AP Width Log[(ms)]",
        "ap_mean_fast_trough_v_0_long_square": "Mean AP Fast Trough (mV)",
        "avg_rate_0_long_square": "Avg Firing Rate (Hz)",
        "latency_0_long_square": "Latency (s)",
        "v_baseline": "Baseline voltage (mV)",
        "dandiset_link": "View Dandiset",
        "meta_data_link": "View File Metadata",
        "file_link": "File Download",
    }
    GLOBAL_VARS.table_spec = {
        "View Dandiset": "links",
        "View File Metadata": "links",
        "File Download": "links",
    }

    file = pd.read_csv(database_csv)

    file["ap_1_width_0_long_square"] = np.log10(file["ap_1_width_0_long_square"] * 1000)
    file["tau"] = np.log10(file["tau"] * 1000)
    file["input_resistance"] = np.log10(file["input_resistance"])
    file["GMM cluster label"] = file["GMM cluster label"].apply(lambda x: f"Cluster {x}")

    file = file.sample(frac=1)
    file.to_csv(shuffled_csv)

    wbz.run_web_viz(database_file=shuffled_csv, config=GLOBAL_VARS, backend="static")
