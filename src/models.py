"""Unsupervised regime models and automatic regime labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

ModelName = Literal["KMeans", "GMM", "HMM"]

N_REGIMES = 4
RANDOM_STATE = 42


@dataclass
class RegimeResult:
    """Container for fitted model outputs."""

    model_name: ModelName
    cluster_ids: pd.Series
    regime_labels: pd.Series
    transition_matrix: np.ndarray | None
    cluster_stats: pd.DataFrame


def _fit_kmeans(X: np.ndarray) -> np.ndarray:
    model = KMeans(n_clusters=N_REGIMES, random_state=RANDOM_STATE, n_init=10)
    return model.fit_predict(X)


def _fit_gmm(X: np.ndarray) -> np.ndarray:
    model = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=5,
        max_iter=500,
    )
    return model.fit_predict(X)


def _fit_hmm(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_iter=500,
        tol=1e-3,
    )
    model.fit(X)
    states = model.predict(X)
    return states, model.transmat_


def _cluster_summary(features: pd.DataFrame, cluster_ids: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean return, volatility, and drawdown."""
    work = features.copy()
    work["cluster"] = cluster_ids
    summary = work.groupby("cluster").agg(
        mean_return=("daily_return", "mean"),
        mean_volatility=("volatility_20", "mean"),
        mean_drawdown=("drawdown", "mean"),
        mean_momentum=("momentum_20", "mean"),
        count=("daily_return", "count"),
    )
    return summary.sort_index()


def auto_label_regimes(cluster_stats: pd.DataFrame) -> dict[int, str]:
    """
    Map cluster IDs to economic regime names using centroid statistics.

    Heuristic priority:
    1. Crisis — highest volatility (typically sharp selloffs)
    2. Bear — lowest mean return among remaining clusters
    3. Bull — highest mean return among remaining clusters
    4. Sideways — remaining cluster (muted return/vol profile)

    Recovery is derived at backtest time for clusters with positive return
    but still elevated drawdown; clusters labeled Bull with deep drawdown
    are relabeled Recovery when appropriate.
    """
    stats = cluster_stats.copy()
    remaining = set(stats.index.tolist())
    labels: dict[int, str] = {}

    crisis_id = stats["mean_volatility"].idxmax()
    labels[crisis_id] = "Crisis"
    remaining.remove(crisis_id)

    bear_id = stats.loc[list(remaining), "mean_return"].idxmin()
    labels[bear_id] = "Bear"
    remaining.remove(bear_id)

    bull_id = stats.loc[list(remaining), "mean_return"].idxmax()
    labels[bull_id] = "Bull"
    remaining.remove(bull_id)

    sideways_id = remaining.pop()
    labels[sideways_id] = "Sideways"

    # Promote Bull -> Recovery when returns are positive but drawdown is still material.
    bull_stats = stats.loc[bull_id]
    if bull_stats["mean_return"] > 0 and bull_stats["mean_drawdown"] < -0.08:
        labels[bull_id] = "Recovery"

    return labels


def fit_regime_model(
    features: pd.DataFrame,
    scaled_features: pd.DataFrame,
    model_name: ModelName,
) -> RegimeResult:
    """Train the selected clustering / HMM model and return labeled regimes."""
    X = scaled_features.values
    transition_matrix: np.ndarray | None = None

    if model_name == "KMeans":
        cluster_ids = _fit_kmeans(X)
    elif model_name == "GMM":
        cluster_ids = _fit_gmm(X)
    elif model_name == "HMM":
        cluster_ids, transition_matrix = _fit_hmm(X)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    cluster_stats = _cluster_summary(features, cluster_ids)
    label_map = auto_label_regimes(cluster_stats)

    regime_labels = pd.Series(
        [label_map[c] for c in cluster_ids],
        index=scaled_features.index,
        name="regime",
    )
    cluster_series = pd.Series(cluster_ids, index=scaled_features.index, name="cluster")

    return RegimeResult(
        model_name=model_name,
        cluster_ids=cluster_series,
        regime_labels=regime_labels,
        transition_matrix=transition_matrix,
        cluster_stats=cluster_stats.assign(regime_label=cluster_stats.index.map(label_map)),
    )
