"""Feature engineering for market regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "daily_return",
    "log_return",
    "volatility_20",
    "momentum_20",
    "momentum_50",
    "drawdown",
    "volume_zscore",
]


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build regime-detection features from OHLCV data."""
    df = ohlcv.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["volatility_20"] = df["daily_return"].rolling(20).std() * np.sqrt(252)
    df["momentum_20"] = close.pct_change(20)
    df["momentum_50"] = close.pct_change(50)

    rolling_max = close.cummax()
    df["drawdown"] = close / rolling_max - 1.0

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    return df.dropna(subset=FEATURE_COLUMNS)


def standardize_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score standardize feature matrix; returns scaled frame and fitted scaler."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[FEATURE_COLUMNS])
    scaled_df = pd.DataFrame(scaled, index=features.index, columns=FEATURE_COLUMNS)
    return scaled_df, scaler
