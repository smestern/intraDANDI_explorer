"""argparse CLI for `dandi_scraper`.

Usage::

    python -m dandi_scraper analyze [--cache-dir DIR] [--debug]
    python -m dandi_scraper merge   [--cache-dir DIR] [--output-dir DIR]
    python -m dandi_scraper plot    [--cache-dir DIR] [--threshold N]
    python -m dandi_scraper sort-plots [--cache-dir DIR] [--traces-dir DIR]
    python -m dandi_scraper serve   [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from . import config


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", help="Directory holding raw dandi downloads + per-dandiset CSVs.")
    p.add_argument("--output-dir", help="Directory where merged CSV / pickles / shuffled CSV are written.")
    p.add_argument("--traces-dir", help="Directory where SVG traces are placed by sort-plots.")
    p.add_argument("--log-path", help="File path for the log handler.")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging (sets DANDI_SCRAPER_DEBUG=1).")
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved CONFIG and exit (no work performed).")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dandi_scraper")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Download + analyze every relevant dandiset.")
    _add_common_flags(p_analyze)

    p_merge = sub.add_parser("merge", help="Merge per-dandiset CSVs into the final database.")
    _add_common_flags(p_merge)
    p_merge.add_argument("--no-cached-metadata", action="store_true",
                         help="Re-fetch dandiset metadata via the LLM parser instead of reusing all_new.csv.")

    p_plot = sub.add_parser("plot", help="Render trace SVGs for new dandisets.")
    _add_common_flags(p_plot)
    p_plot.add_argument("--threshold", type=float,
                        help="Only process dandisets whose code (parsed as int) >= threshold.")

    p_sort = sub.add_parser("sort-plots", help="Move generated SVGs into the local traces dir.")
    _add_common_flags(p_sort)

    p_serve = sub.add_parser("serve", help="Build and launch the static web visualization.")
    _add_common_flags(p_serve)
    p_serve.add_argument("--database-csv", help="Path to merged CSV (default: <output-dir>/all_new.csv).")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        os.environ["DANDI_SCRAPER_DEBUG"] = "1"

    config.apply_overrides(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        traces_dir=args.traces_dir,
        log_path=args.log_path,
    )

    if args.print_config:
        cfg = config.CONFIG
        print(f"cache_dir   = {cfg.cache_dir}")
        print(f"output_dir  = {cfg.output_dir}")
        print(f"traces_dir  = {cfg.traces_dir}")
        print(f"log_path    = {cfg.log_path}")
        print(f"merged_csv  = {cfg.merged_csv}")
        return 0

    if args.command == "analyze":
        from .download import run_analyze_dandiset
        run_analyze_dandiset()
    elif args.command == "merge":
        from .database import run_merge_dandiset
        run_merge_dandiset(use_cached_metadata=not args.no_cached_metadata)
    elif args.command == "plot":
        from .download import run_plot_dandiset
        run_plot_dandiset(threshold=args.threshold)
    elif args.command == "sort-plots":
        from .download import sort_plot_dandiset
        sort_plot_dandiset()
    elif args.command == "serve":
        from .server import build_server
        build_server(database_csv=args.database_csv)
    else:  # pragma: no cover - argparse `required=True` makes this unreachable
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
