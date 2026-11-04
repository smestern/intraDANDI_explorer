"""Two-tier (coarse -> fine) GMM clustering of the scaled feature matrix.

A GMM with ``n_coarse`` components splits all cells into gross clusters; then a
separate GMM is fit within each gross cluster to split it into up to ``n_fine``
sub-clusters. Fine clusters therefore always nest inside one coarse cluster,
which a single flat GMM does not guarantee.

Labels are ordered by size (largest first) so they are stable across reruns
with the same data: coarse clusters are ``1..n_coarse`` and fine clusters are
``"<coarse>.<sub>"``, e.g. ``"2.3"``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

from . import config

logger = logging.getLogger("dandi_scraper.clustering")


@dataclass
class HierarchicalClusters:
    coarse: pd.Series  # int coarse label per cell, 1-based
    fine: pd.Series  # "<coarse>.<sub>" label per cell
    coarse_model: GaussianMixture
    fine_models: Dict[int, GaussianMixture] = field(default_factory=dict)


def _fit_gmm(X: np.ndarray, n_components: int, random_state: int) -> tuple[GaussianMixture, np.ndarray]:
    """Fit a GMM and return it with labels renumbered by size (0 = largest)."""
    gmm = GaussianMixture(n_components=n_components, n_init=5, random_state=random_state)
    raw = gmm.fit_predict(X)
    counts = np.bincount(raw, minlength=n_components)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(n_components)
    return gmm, rank[raw]

def _fit_aggo(X: np.ndarray, n_clusters: int, metric: str, linkage: str = None) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Fit agglomerative clustering and return raw and ranked labels.

    ``linkage`` defaults to ward for euclidean and average otherwise. Euclidean with
    average linkage only peels off outliers here (15518 of 15665 cells in one cluster).
    """
    if linkage is None:
        linkage = "ward" if metric == "euclidean" else "average"
    agg = AgglomerativeClustering(metric=metric, n_clusters=n_clusters, linkage=linkage)
    raw = agg.fit_predict(X)
    counts = np.bincount(raw)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return agg, rank[raw]


def hierarchical_gmm(
    X: pd.DataFrame,
    n_coarse: int = None,
    n_fine: int = None,
    min_fine_size: int = None,
    random_state: int = None,
) -> HierarchicalClusters:
    """Cluster rows of ``X`` into coarse clusters, then sub-cluster each one.

    A coarse cluster with fewer than ``n_fine * min_fine_size`` cells is split
    into fewer sub-clusters (``len // min_fine_size``, at least 1).
    """
    n_coarse = config.N_COARSE_CLUSTERS if n_coarse is None else n_coarse
    n_fine = config.N_FINE_CLUSTERS if n_fine is None else n_fine
    min_fine_size = config.MIN_FINE_CLUSTER_SIZE if min_fine_size is None else min_fine_size
    random_state = config.CLUSTER_RANDOM_STATE if random_state is None else random_state

    values = X.to_numpy()
    coarse_model, coarse = _fit_gmm(values, n_coarse, random_state)
    coarse += 1

    fine = np.empty(len(X), dtype=object)
    fine_models = {}
    for c in range(1, n_coarse + 1):
        members = np.flatnonzero(coarse == c)
        n_sub = max(1, min(n_fine, len(members) // min_fine_size))
        if n_sub == 1:
            fine[members] = f"{c}.1"
        else:
            fine_models[c], sub = _fit_gmm(values[members], n_sub, random_state)
            fine[members] = [f"{c}.{s + 1}" for s in sub]
        logger.info(
            "coarse cluster %d: %d cells -> %d sub-clusters %s",
            c, len(members), n_sub,
            pd.Series(fine[members]).value_counts().sort_index().to_dict(),
        )

    return HierarchicalClusters(
        coarse=pd.Series(coarse, index=X.index, name="coarse cluster"),
        fine=pd.Series(fine, index=X.index, name="fine cluster"),
        coarse_model=coarse_model,
        fine_models=fine_models,
    )

def hierarchical_aggo(
    X: pd.DataFrame,
    metric: str = "cosine", #match umap?
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster rows of ``X`` using agglomerative clustering and return raw and ranked labels."""

    n_coarse = config.N_COARSE_CLUSTERS
    n_fine = config.N_FINE_CLUSTERS
    min_fine_size = config.MIN_FINE_CLUSTER_SIZE
    random_state = config.CLUSTER_RANDOM_STATE

    values = X.to_numpy()
    agg_coarse, coarse_ranked = _fit_aggo(values, n_coarse, metric)
    coarse = coarse_ranked + 1

    fine = np.empty(len(X), dtype=object)
    fine_models = {}
    for c in range(1, n_coarse + 1):
        members = np.flatnonzero(coarse == c)
        n_sub = max(1, min(n_fine, len(members) // min_fine_size))
        if n_sub == 1:
            fine[members] = f"{c}.1"
        else:
            agg_fine, sub_ranked = _fit_aggo(values[members], n_sub, metric)
            fine[members] = [f"{c}.{s + 1}" for s in sub_ranked]
            fine_models[c] = agg_fine

    return HierarchicalClusters(
        coarse=pd.Series(coarse, index=X.index, name="coarse cluster"),
        fine=pd.Series(fine, index=X.index, name="fine cluster"),
        coarse_model=agg_coarse,
        fine_models=fine_models,
    )
