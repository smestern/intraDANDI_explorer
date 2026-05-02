"""Public API for dandi_scraper.

Re-exports the runner entry points that previously lived in the monofile so
existing scripts (``run_analyze.py``, ``run_app.py``) keep working unchanged.
"""
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
