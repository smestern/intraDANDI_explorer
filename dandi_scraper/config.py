"""Configuration defaults and runtime overrides for `dandi_scraper`.

All hardcoded paths and tuning constants previously inlined in
`dandi_scraper.py` live here. Defaults can be overridden by:

1. Setting environment variables (see `_ENV_MAP` below).
2. Calling `apply_overrides(**kwargs)` (used by `cli.py`).

Secrets (e.g. ``COHERE_KEY``) are kept in the gitignored `secrets.py`
sibling module; access them via `get_cohere_key()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Paths & runtime config
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = "/media/smestern/Expansion/dandi"
_DEFAULT_OUTPUT_DIR = "."
_DEFAULT_TRACES_DIR = "./data/traces"
_DEFAULT_LOG_PATH = "dandi_scraper_run.log"


@dataclass
class Config:
    """Mutable runtime configuration. Module-level singleton: `CONFIG`."""

    cache_dir: str = _DEFAULT_CACHE_DIR
    output_dir: str = _DEFAULT_OUTPUT_DIR
    traces_dir: str = _DEFAULT_TRACES_DIR
    log_path: str = _DEFAULT_LOG_PATH

    # ---- derived paths ----
    @property
    def merged_csv(self) -> str:
        return os.path.join(self.output_dir, "all_new.csv")

    @property
    def merged_errors_csv(self) -> str:
        return os.path.join(self.output_dir, "all_new.errors.csv")

    @property
    def shuffled_csv(self) -> str:
        return os.path.join(self.output_dir, "all2_2.csv")

    @property
    def dataset_numeric_pkl(self) -> str:
        return os.path.join(self.output_dir, "dataset_numeric.pkl")

    @property
    def dataset_numeric_norm_pkl(self) -> str:
        return os.path.join(self.output_dir, "dataset_numeric_norm.pkl")


# Env-var overrides applied at import time.
_ENV_MAP = {
    "cache_dir": "DANDI_CACHE_DIR",
    "output_dir": "DANDI_OUTPUT_DIR",
    "traces_dir": "DANDI_TRACES_DIR",
    "log_path": "DANDI_LOG_PATH",
}


def _load_from_env() -> Config:
    cfg = Config()
    for attr, env_var in _ENV_MAP.items():
        val = os.environ.get(env_var)
        if val:
            setattr(cfg, attr, val)
    return cfg


CONFIG: Config = _load_from_env()


def apply_overrides(**kwargs) -> Config:
    """Mutate the singleton `CONFIG` with any non-None values in `kwargs`.

    Unknown keys are ignored so the CLI can pass through namespace dicts.
    Returns the (mutated) singleton for convenience.
    """
    valid = {f.name for f in fields(Config)}
    for key, val in kwargs.items():
        if val is None or key not in valid:
            continue
        setattr(CONFIG, key, val)
    return CONFIG


# ---------------------------------------------------------------------------
# Analysis constants (lifted verbatim from the previous monofile)
# ---------------------------------------------------------------------------

COLS_TO_KEEP: List[str] = [
    "input_resistance", "tau", "v_baseline", "sag_nearest_minus_100",
    "ap_1_threshold_v_0_long_square", "ap_1_peak_v_0_long_square",
    "ap_1_upstroke_0_long_square",
    "ap_1_width_0_long_square", "ap_1_fast_trough_v_0_long_square",
    "ap_mean_threshold_v_0_long_square", "ap_mean_peak_v_0_long_square",
    "ap_mean_upstroke_0_long_square",
    "ap_mean_width_0_long_square", "ap_mean_fast_trough_v_0_long_square",
    "avg_rate_0_long_square", "latency_0_long_square",
]

DANDISETS_TO_SKIP: List[str] = [
    "000012", "000013",
    "000008",
    "000020",
    "000005",   # mostly in vivo continuous data
    "000117", "000168",
    "000362",   # appears to be some lfp or something
    "000717",   # test dandiset
    "000293",   # superseded by 000297
    "000292",   # superseded by 000297
    "000341",   # superseded by 000297
]

DANDISETS_TO_INCLUDE: List[str] = [
    "001776",   # iCEphys dataset not labeled as such
]

QC_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "input_resistance": (0, 1e9),
    "sag_nearest_minus_100": (-1, 1),
    "ap_1_threshold_v_0_long_square": (-100, 100),
    "tau": (0.01 / 1000, 0.4),
    "ap_1_width_0_long_square": (0.01 / 1000, 10 / 1000),
    "ap_1_fast_trough_v_0_long_square": (-100, 0),
    "avg_rate_0_long_square": (0, 200),
}

SCALE_FEATURES: Dict[str, str] = {
    "input_resistance": "log",
    "tau": "log-1000",
    "ap_1_width_0_long_square": "log-1000",
    "ap_mean_width_0_long_square": "log-1000",
}

CODES_TO_PLOT_THRESHOLD: float = 1455.9

SPECIES_REPLACEMENT: Dict[str, str] = {
    "Mus musculus - House mouse": "House mouse",
    "Rattus norvegicus - Norway rat": "Rat",
    "Brown rat": "Rat",
    "Rat; norway rat; rats; brown rat": "Rat",
    "Homo sapiens - Human": "Human",
    "Drosophila melanogaster - Fruit fly": "Fruit fly",
}

NEURODATA_TYPE_MAP: Dict[str, List[str]] = dict(
    ecephys=["LFP", "Units", "ElectricalSeries"],
    ophys=["PlaneSegmentation", "TwoPhotonSeries", "ImageSegmentation"],
    icephys=["PatchClampSeries", "VoltageClampSeries", "CurrentClampSeries"],
)


# ---------------------------------------------------------------------------
# Secrets accessor
# ---------------------------------------------------------------------------

def get_cohere_key() -> str:
    """Resolve the Cohere API key.

    Precedence: `COHERE_API_KEY` env var > `secrets.COHERE_KEY` (gitignored
    sibling module) > raise.
    """
    env_val = os.environ.get("COHERE_API_KEY")
    if env_val:
        return env_val
    try:
        from . import secrets as _secrets  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "No Cohere API key configured. Set the COHERE_API_KEY env var or "
            "create dandi_scraper/secrets.py with COHERE_KEY=<your key>."
        ) from exc
    key = getattr(_secrets, "COHERE_KEY", None)
    if not key:
        raise RuntimeError(
            "dandi_scraper/secrets.py is present but COHERE_KEY is empty. "
            "Set it or use the COHERE_API_KEY env var."
        )
    return key


__all__ = [
    "Config", "CONFIG", "apply_overrides",
    "COLS_TO_KEEP", "DANDISETS_TO_SKIP", "DANDISETS_TO_INCLUDE",
    "QC_FEATURE_BOUNDS", "SCALE_FEATURES", "CODES_TO_PLOT_THRESHOLD",
    "SPECIES_REPLACEMENT", "NEURODATA_TYPE_MAP",
    "get_cohere_key",
]
