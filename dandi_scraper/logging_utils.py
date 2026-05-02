"""Shared logging configuration for dandi_scraper modules."""
from __future__ import annotations

import logging
import os
from typing import Optional

from .config import CONFIG

_LOGGING_CONFIGURED = False


def configure_logging(log_path: Optional[str] = None) -> None:
    """Idempotently attach a file + stream handler to the dandi_scraper and
    pyAPisolation loggers. Set env var ``DANDI_SCRAPER_DEBUG=1`` for DEBUG level."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    log_path = log_path or CONFIG.log_path
    level = logging.DEBUG if os.environ.get("DANDI_SCRAPER_DEBUG") == "1" else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    for name in ("dandi_scraper", "pyAPisolation"):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        existing = {type(h).__name__ + getattr(h, "baseFilename", "") for h in lg.handlers}
        if "FileHandler" + os.path.abspath(log_path) not in existing:
            lg.addHandler(fh)
        if "StreamHandler" not in {type(h).__name__ for h in lg.handlers if not isinstance(h, logging.FileHandler)}:
            lg.addHandler(sh)
        lg.propagate = False
    _LOGGING_CONFIGURED = True
    logging.getLogger("dandi_scraper").info(
        "logging configured (level=%s, file=%s)", logging.getLevelName(level), log_path
    )


# Backwards-compatible private alias used by older imports.
_configure_logging = configure_logging
