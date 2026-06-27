"""Market Regime Lab — Streamlit application entry point."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import regime_distribution, run_backtest
from src.data import download_ohlcv
from src.features import FEATURE_COLUMNS, compute_features, standardize_features
from src.models import AVAILABLE_MODELS, HMM_AVAILABLE, ModelName, fit_regime_model
from src.plots import (
    cluster_stats_bar,
    drawdown_chart,
    equity_curve_chart,
    feature_scatter,
    metrics_comparison_table,
    price_shaded_by_regime,
    regime_pie_chart,
    regime_timeline,
    transition_matrix_heatmap,
)

TICKERS = ["SPY", "QQQ", "BTC-USD", "NVDA", "TSLA", "AAPL"]
MODELS: list[ModelName] = ["KMeans", "GMM", "HMM"]
DEFAULT_START = date.today() - timedelta(days=365 * 5)
DEFAULT_END = date.today()

st.set_page_config(
    page_title="Market Regime Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }
        .app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
        .app-subtitle { color: #64748b; margin-bottom: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _pipeline(
    ticker: str,
    start: date,
    end: date,
    model_name: ModelName,
) -> dict:
    """End-to-end data, feature, model, and backtest pipeline (cached)."""
    ohlcv = download_ohlcv(ticker, start, end)
    features = compute_features(ohlcv)
    scaled, _ = standardize_features(features)
    regime_result = fit_regime_model(features, scaled, model_name)
    backtest = run_backtest(ohlcv["Close"], regime_result.regime_labels)

    return {
        "ohlcv": ohlcv,
        "features": features,
        "scaled": scaled,
        "regime_result": regime_result,
        "backtest": backtest,
        "distribution": regime_distribution(regime_result.regime_labels),
    }


def _render_sidebar() -> tuple[str, date, date, ModelName]:
    st.sidebar.header("Configuration")
    ticker = st.sidebar.selectbox("Ticker", TICKERS, index=0)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start = st.date_input("Start", value=DEFAULT_START)
    with col2:
        end = st.date_input("End", value=DEFAULT_END)
    default_model_idx = MODELS.index("HMM") if "HMM" in MODELS else len(MODELS) - 1
    model_name = st.sidebar.selectbox("Regime Model", MODELS, index=default_model_idx)

    if not HMM_AVAILABLE:
        st.sidebar.warning(
            "HMM is disabled: hmmlearn is not compatible with this Python version. "
            "Use KMeans or GMM, or redeploy on Streamlit Cloud with **Python 3.12**."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Regime strategy**
        - **Hold:** Bull, Recovery, Sideways
        - **Cash:** Crisis, Bear
        """
    )
    return ticker, start, end, model_name


def _metric_row(metrics: dict[str, float], prefix: str) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{prefix} Total Return", f"{metrics['total_return']:.2%}")
    c2.metric(f"{prefix} Ann. Return", f"{metrics['annualized_return']:.2%}")
    c3.metric(f"{prefix} Ann. Vol", f"{metrics['annualized_volatility']:.2%}")
    c4.metric(f"{prefix} Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    c5.metric(f"{prefix} Max DD", f"{metrics['max_drawdown']:.2%}")


def main() -> None:
    _inject_styles()
    st.markdown('<p class="app-title">Market Regime Lab</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Unsupervised regime detection, visualization, and backtesting</p>',
        unsafe_allow_html=True,
    )

    ticker, start, end, model_name = _render_sidebar()

    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner("Downloading data and training models…"):
        try:
            result = _pipeline(ticker, start, end, model_name)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    ohlcv = result["ohlcv"]
    features = result["features"]
    regime_result = result["regime_result"]
    backtest = result["backtest"]
    distribution = result["distribution"]
    regimes = regime_result.regime_labels

    tab_overview, tab_model, tab_backtest, tab_stats = st.tabs(
        ["Overview", "Regime Model", "Backtest", "Research Stats"]
    )

    with tab_overview:
        st.subheader(f"{ticker} — Market Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trading Days", f"{len(features):,}")
        m2.metric("Latest Close", f"${ohlcv['Close'].iloc[-1]:,.2f}")
        m3.metric("Active Regimes", str(regimes.nunique()))
        m4.metric("Model", model_name)

        st.plotly_chart(
            price_shaded_by_regime(ohlcv["Close"], regimes, title=f"{ticker} Price by Regime"),
            use_container_width=True,
        )

        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(regime_timeline(regimes), use_container_width=True)
        with c_right:
            st.plotly_chart(regime_pie_chart(distribution), use_container_width=True)

    with tab_model:
        st.subheader("Regime Model Diagnostics")
        st.plotly_chart(
            feature_scatter(features, regimes),
            use_container_width=True,
        )

        st.plotly_chart(cluster_stats_bar(regime_result.cluster_stats), use_container_width=True)

        with st.expander("Cluster summary statistics"):
            display_stats = regime_result.cluster_stats.copy()
            for col in ("mean_return", "mean_volatility", "mean_drawdown", "mean_momentum"):
                display_stats[col] = display_stats[col].map(lambda x: f"{x:.4f}")
            st.dataframe(display_stats, use_container_width=True)

        if regime_result.transition_matrix is not None:
            ordered_labels = (
                regime_result.cluster_stats.assign(
                    regime=regime_result.cluster_stats.index.map(
                        dict(
                            zip(
                                regime_result.cluster_stats.index,
                                regime_result.cluster_stats["regime_label"],
                            )
                        )
                    )
                )
                .sort_index()["regime_label"]
                .tolist()
            )
            st.plotly_chart(
                transition_matrix_heatmap(
                    regime_result.transition_matrix,
                    ordered_labels,
                    title=f"{model_name} Transition Matrix",
                ),
                use_container_width=True,
            )
        else:
            st.info("Transition matrix is available only for the Hidden Markov Model (HMM).")

    with tab_backtest:
        st.subheader("Strategy Backtest")
        st.plotly_chart(
            equity_curve_chart(backtest["bh_equity"], backtest["regime_equity"]),
            use_container_width=True,
        )
        st.plotly_chart(
            drawdown_chart(backtest["bh_drawdown"], backtest["regime_drawdown"]),
            use_container_width=True,
        )
        st.plotly_chart(
            metrics_comparison_table(backtest["bh_metrics"], backtest["regime_metrics"]),
            use_container_width=True,
        )

        st.markdown("#### Buy & Hold")
        _metric_row(backtest["bh_metrics"], "BH")
        st.markdown("#### Regime Strategy")
        _metric_row(backtest["regime_metrics"], "RS")

    with tab_stats:
        st.subheader("Research Statistics")
        st.markdown("#### Feature correlations")
        corr = features[FEATURE_COLUMNS].corr()
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

        st.markdown("#### Regime occupancy")
        st.dataframe(
            distribution.assign(pct=distribution["pct"].map(lambda x: f"{x:.1f}%")),
            use_container_width=True,
        )

        st.markdown("#### Per-regime return statistics")
        joined = features.join(regimes, how="inner")
        regime_stats = joined.groupby("regime").agg(
            mean_daily_return=("daily_return", "mean"),
            vol_20=("volatility_20", "mean"),
            avg_drawdown=("drawdown", "mean"),
            days=("daily_return", "count"),
        )
        st.dataframe(regime_stats, use_container_width=True)

        st.markdown("#### Sample feature rows")
        st.dataframe(features.tail(10), use_container_width=True)


if __name__ == "__main__":
    main()
