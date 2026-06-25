"""OHLCV data acquisition via yfinance with Streamlit caching."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and enforce a consistent OHLCV schema."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(how="any")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def download_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Download daily OHLCV bars for *ticker* between *start* and *end* (inclusive).

    Results are cached for one hour to avoid redundant API calls.
    """
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
    )

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} in the selected date range.")

    return _normalize_ohlcv(raw)
