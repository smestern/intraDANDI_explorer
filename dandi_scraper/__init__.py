"""Public API for dandi_scraper.

Re-exports the runner entry points that previously lived in the monofile so
existing scripts (``run_analyze.py``, ``run_app.py``) keep working unchanged.
"""
import os as _os
import sys as _sys

#The pipeline only saves figures, and the default TkAgg backend breaks under joblib: Tk objects
#garbage-collected off the main thread raise "main thread is not in main loop". Use Agg unless
#the caller already chose a backend (MPLBACKEND, or pyplot imported first, e.g. _analyze_umap.py).
if "matplotlib.pyplot" not in _sys.modules and not _os.environ.get("MPLBACKEND"):
    import matplotlib as _mpl
    _mpl.use("Agg")
from . import dandi_scraper  # back-compat: ``dandi_scraper.dandi_scraper.X``
from .database import run_merge_dandiset
from .download import (
    download_dandiset,
    run_analyze_dandiset,
    run_plot_dandiset,
    sort_plot_dandiset,
)
from .server import build_server

__all__ = [
    "build_server",
    "dandi_scraper",
    "download_dandiset",
    "run_analyze_dandiset",
    "run_merge_dandiset",
    "run_plot_dandiset",
    "sort_plot_dandiset",
]
