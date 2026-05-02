"""Backwards-compatible shim.

The original implementation of this module was a single ~720-line file. It has
been split into focused submodules; this file simply re-exports the public
runner functions so legacy imports such as
``dandi_scraper.dandi_scraper.sort_plot_dandiset`` keep resolving.
"""
from .analysis import (
    analyze_dandiset,
    build_dandiset_df,
    filter_dandiset_df,
    get_dandi_metadata,
    quick_qc,
    scale_features,
)
from .database import run_merge_dandiset
from .download import (
    download_dandiset,
    run_analyze_dandiset,
    run_plot_dandiset,
    sort_plot_dandiset,
)
from .logging_utils import configure_logging as _configure_logging
from .server import build_server

__all__ = [
    "analyze_dandiset",
    "build_dandiset_df",
    "build_server",
    "download_dandiset",
    "filter_dandiset_df",
    "get_dandi_metadata",
    "quick_qc",
    "run_analyze_dandiset",
    "run_merge_dandiset",
    "run_plot_dandiset",
    "scale_features",
    "sort_plot_dandiset",
    "_configure_logging",
]
