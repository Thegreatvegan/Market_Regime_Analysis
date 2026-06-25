"""Regime-aware and buy-and-hold backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLD_REGIMES = frozenset({"Bull", "Recovery", "Sideways"})
CASH_REGIMES = frozenset({"Crisis", "Bear"})


def _align_returns(price: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Align close prices and regime labels on a common index."""
    frame = pd.DataFrame({"close": price, "regime": regimes}).dropna()
    frame["daily_return"] = frame["close"].pct_change()
    return frame.dropna()


def regime_positions(regimes: pd.Series) -> pd.Series:
    """
    Binary position: 1 when holding the asset, 0 in cash.

    Hold during Bull, Recovery, and Sideways; move to cash in Crisis and Bear.
    Positions are shifted by one day to avoid look-ahead bias.
    """
    signal = regimes.map(lambda r: 1.0 if r in HOLD_REGIMES else 0.0)
    return signal.shift(1).fillna(0.0)


def buy_and_hold_returns(aligned: pd.DataFrame) -> pd.Series:
    """Daily strategy returns for a fully invested buy-and-hold portfolio."""
    return aligned["daily_return"].copy()


def regime_strategy_returns(aligned: pd.DataFrame) -> pd.Series:
    """Daily strategy returns using regime-based exposure."""
    position = regime_positions(aligned["regime"])
    return position * aligned["daily_return"]


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Compound daily returns into a normalized equity curve starting at 1.0."""
    return (1.0 + daily_returns).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Underwater equity curve (peak-to-trough decline)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def compute_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Annualized performance statistics from a daily return series."""
    if daily_returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    equity = equity_curve(daily_returns)
    total_return = float(equity.iloc[-1] - 1.0)
    n_days = len(daily_returns)
    years = n_days / periods_per_year

    ann_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    ann_vol = float(daily_returns.std() * np.sqrt(periods_per_year))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    max_dd = float(drawdown_series(equity).min())

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def run_backtest(
    price: pd.Series,
    regimes: pd.Series,
) -> dict[str, pd.Series | dict[str, float]]:
    """Run buy-and-hold and regime strategies; return curves and metrics."""
    aligned = _align_returns(price, regimes)

    bh_returns = buy_and_hold_returns(aligned)
    regime_returns = regime_strategy_returns(aligned)

    bh_equity = equity_curve(bh_returns)
    regime_equity = equity_curve(regime_returns)

    return {
        "aligned": aligned,
        "bh_returns": bh_returns,
        "regime_returns": regime_returns,
        "bh_equity": bh_equity,
        "regime_equity": regime_equity,
        "bh_drawdown": drawdown_series(bh_equity),
        "regime_drawdown": drawdown_series(regime_equity),
        "bh_metrics": compute_metrics(bh_returns),
        "regime_metrics": compute_metrics(regime_returns),
        "position": regime_positions(aligned["regime"]),
    }


def regime_distribution(regimes: pd.Series) -> pd.DataFrame:
    """Count and percentage of days spent in each regime."""
    counts = regimes.value_counts().rename("days")
    pct = (counts / counts.sum() * 100.0).rename("pct")
    return pd.concat([counts, pct], axis=1).sort_values("days", ascending=False)
