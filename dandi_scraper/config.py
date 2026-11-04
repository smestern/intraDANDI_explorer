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

_DEFAULT_CACHE_DIR = "/media/inouelinux/sgbackup/"
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

# The ap_mean_* versions of the AP features are left out: they correlate r >= 0.97 with
# ap_1_*, so keeping both double-weights spike shape. The f-I slope stands in for
# avg_rate_0_long_square, which for the ~54% of cells that fire one spike at rheobase is
# just 1 / step duration (see figures/umap_workup/REPORT.md).
COLS_TO_KEEP: List[str] = [
    "input_resistance", "tau", "v_baseline", "sag_nearest_minus_100",
    "ap_1_threshold_v_0_long_square", "ap_1_peak_v_0_long_square",
    "ap_1_upstroke_0_long_square",
    "ap_1_width_0_long_square", "ap_1_fast_trough_v_0_long_square",
    "fi_linear_fit_slope", "latency_0_long_square",
]

DANDISETS_TO_SKIP: List[str] = [
    #"000012", "000013",
    #"000008",
    "000168",
    "000005",   # mostly in vivo continuous data
    #"000117", 
    "000168",   # loose cell-attached recording
    "000362",   # appears to be some lfp or something
    "000717",   # test dandiset
    "000293",   # superseded by 000297
    "000292",   # superseded by 000297
    "000341",   # In vivo recording
    "000570"
]

DANDISETS_TO_INCLUDE: List[str] = [
    "001776",   # iCEphys dataset not labeled as such
]

QC_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = { #These are generally very liberal. for example, input_resistance shouldn't actually measure 0
    "input_resistance": (10, 1e9), # ohms
    "sag_nearest_minus_100": (-5, 5), # unitless, ratio
    # Thresholds below -70 mV sit on v_baseline: the detector picked up stim onset, not a spike.
    "ap_1_threshold_v_0_long_square": (-70, 100), # mV
    "tau": (0.01 / 1000, 0.4), # seconds
    "ap_1_width_0_long_square": (0.1 / 1000, 10 / 1000), # seconds
    "ap_1_fast_trough_v_0_long_square": (-100, 0), # mV
    "avg_rate_0_long_square": (0, 500), # Hz probably shouldnt be above 500, but maybe?
    "ap_1_upstroke_0_long_square": (0, 2000), # mV/ms, above this is a detection/scaling artifact
    "ap_mean_upstroke_0_long_square": (0, 2000), # mV/ms
    # A bump on the AP peak double-detected as a 2nd spike gives a mean threshold >0 mV and
    # a sub-0.12 ms mean width (gap in the distribution between 0.12 and 0.2 ms).
    "ap_mean_threshold_v_0_long_square": (-100, 0), # mV
    "ap_mean_width_0_long_square": (0.15 / 1000, 10 / 1000), # seconds
    "ap_1_downstroke_0_long_square": (-2000, 0), # mV/ms, a positive downstroke is a detection error
    "ap_1_upstroke_downstroke_ratio_0_long_square": (0, 50), # unitless, negative only if downstroke > 0
    "ap_mean_upstroke_downstroke_ratio_0_long_square": (0, 50), # unitless
    # First spike <1 ms after step onset is a stim/capacitive artifact. These cells also
    # carry the implausible 250-290 Hz rheobase rates.
    "latency_0_long_square": (0.001, 5), # seconds
}

SCALE_FEATURES: Dict[str, str] = {
    "input_resistance": "log",
    "tau": "log-1000",
    "ap_1_width_0_long_square": "log-1000",
    "ap_mean_width_0_long_square": "log-1000",
    "ap_1_upstroke_0_long_square": "log",
    "ap_mean_upstroke_0_long_square": "log",
    "avg_rate_0_long_square": "log",
}

# After RobustScaler, clip each feature to +/- this many IQR-units so no single
# feature can dominate the cosine distance used by UMAP.
ROBUST_CLIP: float = 5.0

# Two-tier GMM clustering (see clustering.py): N_COARSE_CLUSTERS gross clusters,
# each split into up to N_FINE_CLUSTERS sub-clusters. A coarse cluster is only
# split into as many sub-clusters as it has MIN_FINE_CLUSTER_SIZE cells for.
N_COARSE_CLUSTERS: int = 6
N_FINE_CLUSTERS: int = 5
MIN_FINE_CLUSTER_SIZE: int = 30
CLUSTER_RANDOM_STATE: int = 0

CODES_TO_PLOT_THRESHOLD: float = 0 #if you are updating you don't need to 

SPECIES_REPLACEMENT: Dict[str, str] = {
    "Mus musculus - House mouse": "House mouse",
    "Rattus norvegicus - Norway rat": "Rat",
    "Brown rat": "Rat",
    "Rat; norway rat; rats; brown rat": "Rat",
    "Homo sapiens - Human": "Human",
    "Drosophila melanogaster - Fruit fly": "Fruit fly",
    "Macaca mulatta - Rhesus monkey": "Rhesus macaque",
    "Macaca fascicularis - Cynomolgus monkeys": "Cynomolgus macaque",
    "Macaca nemestrina - Pigtail macaque": "Pigtail macaque",
    "Macaca nemestrina": "Pigtail macaque",
    "Callithrix jacchus - Common marmoset": "Common marmoset",
}

# NCBITaxon id (trailing digits of the DANDI species identifier URL) -> display
# name. Preferred over SPECIES_REPLACEMENT since names vary between uploads.
SPECIES_BY_TAXON: Dict[str, str] = {
    "10090": "House mouse",
    "9606": "Human",
    "10116": "Rat",
    "9544": "Rhesus macaque",
    "9541": "Cynomolgus macaque",
    "9545": "Pigtail macaque",
    "9483": "Common marmoset",
    "7227": "Fruit fly",
}

# ---------------------------------------------------------------------------
# Brain-region vocabulary
# ---------------------------------------------------------------------------

# Coarse, closed vocabulary for the `brain_region` column. Finer detail goes in
# the free-text `brain_subregion` column.
BRAIN_REGIONS: List[str] = [
    "Neocortex", "Hippocampus", "Amygdala", "Striatum", "Pallidum",
    "Basal forebrain", "Thalamus", "Hypothalamus", "Midbrain", "Brainstem",
    "Cerebellum", "Spinal cord", "Olfactory bulb", "Retina", "Unknown",
]

# Lowercased free-text region (legacy LLM output, keywords, ...) ->
# (brain_region, brain_subregion).
BRAIN_REGION_SYNONYMS: Dict[str, Tuple[str, str]] = {
    "neocortex": ("Neocortex", ""),
    "cortex": ("Neocortex", ""),
    "cortical tissue": ("Neocortex", ""),
    "cerebral cortex": ("Neocortex", ""),
    "neocortical l1": ("Neocortex", "L1"),
    "motor cortex": ("Neocortex", "motor cortex"),
    "primary motor cortex": ("Neocortex", "motor cortex"),
    "premotor cortex": ("Neocortex", "premotor cortex"),
    "visual cortex": ("Neocortex", "visual cortex"),
    "primary visual cortex": ("Neocortex", "visual cortex"),
    "temporal cortex": ("Neocortex", "temporal cortex"),
    "middle temporal gyrus": ("Neocortex", "middle temporal gyrus"),
    "anterior insular cortex": ("Neocortex", "anterior insula"),
    "hippocampus": ("Hippocampus", ""),
    "dentate gyrus": ("Hippocampus", "dentate gyrus"),
    "ca1": ("Hippocampus", "CA1"),
    "ca3": ("Hippocampus", "CA3"),
    "amygdala": ("Amygdala", ""),
    "basolateral amygdala": ("Amygdala", "basolateral amygdala"),
    "striatum": ("Striatum", ""),
    "dorsal striatum": ("Striatum", "dorsal striatum"),
    "nucleus accumbens": ("Striatum", "nucleus accumbens"),
    "ventral pallidum": ("Pallidum", "ventral pallidum"),
    "globus pallidus": ("Pallidum", "globus pallidus"),
    "basal forebrain": ("Basal forebrain", ""),
    "thalamus": ("Thalamus", ""),
    "hypothalamus": ("Hypothalamus", ""),
    "dorsal raphe nucleus": ("Midbrain", "dorsal raphe nucleus"),
    "dorsal raphe": ("Midbrain", "dorsal raphe nucleus"),
    "ventral tegmental area": ("Midbrain", "ventral tegmental area"),
    "substantia nigra": ("Midbrain", "substantia nigra"),
    "locus coeruleus": ("Brainstem", "locus coeruleus"),
    "brainstem": ("Brainstem", ""),
    "cerebellum": ("Cerebellum", ""),
    "spinal cord": ("Spinal cord", ""),
    "olfactory bulb": ("Olfactory bulb", ""),
    "retina": ("Retina", ""),
    "unknown": ("Unknown", ""),
}

# Cohere model used to suggest regions for dandisets missing from
# `dandiset_metadata.csv`.
COHERE_MODEL: str = "command-a-03-2025"

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
    "QC_FEATURE_BOUNDS", "SCALE_FEATURES", "ROBUST_CLIP", "CODES_TO_PLOT_THRESHOLD",
    "N_COARSE_CLUSTERS", "N_FINE_CLUSTERS", "MIN_FINE_CLUSTER_SIZE", "CLUSTER_RANDOM_STATE",
    "SPECIES_REPLACEMENT", "SPECIES_BY_TAXON", "NEURODATA_TYPE_MAP",
    "BRAIN_REGIONS", "BRAIN_REGION_SYNONYMS", "COHERE_MODEL",
    "get_cohere_key",
]
