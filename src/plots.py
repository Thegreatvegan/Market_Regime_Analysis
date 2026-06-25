"""Plotly visualization helpers for Market Regime Lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGIME_COLORS = {
    "Bull": "#2ecc71",
    "Bear": "#e74c3c",
    "Crisis": "#8e44ad",
    "Sideways": "#95a5a6",
    "Recovery": "#3498db",
}


def _regime_color_map(regimes: pd.Series) -> dict[str, str]:
    unique = sorted(regimes.dropna().unique())
    return {r: REGIME_COLORS.get(r, "#bdc3c7") for r in unique}


def price_shaded_by_regime(
    price: pd.Series,
    regimes: pd.Series,
    title: str = "Price by Market Regime",
) -> go.Figure:
    """Line chart of close price with background shading per regime."""
    frame = pd.DataFrame({"close": price, "regime": regimes}).dropna()
    colors = _regime_color_map(frame["regime"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["close"],
            mode="lines",
            name="Close",
            line=dict(color="#2c3e50", width=2),
        )
    )

    for regime in frame["regime"].unique():
        mask = frame["regime"] == regime
        fig.add_trace(
            go.Scatter(
                x=frame.index[mask],
                y=frame["close"][mask],
                mode="markers",
                name=regime,
                marker=dict(size=4, color=colors[regime]),
                showlegend=True,
            )
        )

    shapes = []
    regime_values = frame["regime"].values
    dates = frame.index
    if len(dates) > 1:
        start = 0
        for i in range(1, len(regime_values)):
            if regime_values[i] != regime_values[start]:
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=dates[start],
                        x1=dates[i - 1],
                        y0=0,
                        y1=1,
                        fillcolor=colors.get(regime_values[start], "#ecf0f1"),
                        opacity=0.18,
                        layer="below",
                        line_width=0,
                    )
                )
                start = i
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=dates[start],
                x1=dates[-1],
                y0=0,
                y1=1,
                fillcolor=colors.get(regime_values[start], "#ecf0f1"),
                opacity=0.18,
                layer="below",
                line_width=0,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        shapes=shapes,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def feature_scatter(
    features: pd.DataFrame,
    regimes: pd.Series,
    x_col: str = "volatility_20",
    y_col: str = "momentum_20",
    title: str = "Feature Space by Regime",
) -> go.Figure:
    """2-D scatter of two features colored by regime label."""
    frame = features[[x_col, y_col]].copy()
    frame["regime"] = regimes
    frame = frame.dropna()

    fig = px.scatter(
        frame,
        x=x_col,
        y=y_col,
        color="regime",
        color_discrete_map=REGIME_COLORS,
        opacity=0.65,
        title=title,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
    )
    fig.update_layout(template="plotly_white")
    return fig


def regime_timeline(regimes: pd.Series, title: str = "Regime Timeline") -> go.Figure:
    """Step plot showing regime assignment over time."""
    regime_order = ["Bull", "Recovery", "Sideways", "Bear", "Crisis"]
    present = [r for r in regime_order if r in regimes.unique()]
    mapping = {r: i for i, r in enumerate(present)}

    y_vals = regimes.map(mapping)
    colors = _regime_color_map(regimes)

    fig = go.Figure()
    for regime in present:
        mask = regimes == regime
        fig.add_trace(
            go.Scatter(
                x=regimes.index[mask],
                y=y_vals[mask],
                mode="markers",
                name=regime,
                marker=dict(color=colors[regime], size=6),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(present))),
            ticktext=present,
            title="Regime",
        ),
        template="plotly_white",
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def transition_matrix_heatmap(
    transition_matrix: np.ndarray,
    regime_labels: list[str],
    title: str = "HMM Transition Matrix",
) -> go.Figure:
    """Heatmap of regime-to-regime transition probabilities."""
    fig = go.Figure(
        data=go.Heatmap(
            z=transition_matrix,
            x=regime_labels,
            y=regime_labels,
            colorscale="Blues",
            text=np.round(transition_matrix, 3),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title="Probability"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        template="plotly_white",
    )
    return fig


def equity_curve_chart(
    bh_equity: pd.Series,
    regime_equity: pd.Series,
    title: str = "Equity Curve Comparison",
) -> go.Figure:
    """Compare buy-and-hold vs regime-filtered equity curves."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bh_equity.index,
            y=bh_equity.values,
            name="Buy & Hold",
            line=dict(color="#34495e", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=regime_equity.index,
            y=regime_equity.values,
            name="Regime Strategy",
            line=dict(color="#2980b9", width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def drawdown_chart(
    bh_dd: pd.Series,
    regime_dd: pd.Series,
    title: str = "Drawdown Comparison",
) -> go.Figure:
    """Underwater plot for both strategies."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bh_dd.index,
            y=bh_dd.values,
            name="Buy & Hold",
            fill="tozeroy",
            line=dict(color="#c0392b", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=regime_dd.index,
            y=regime_dd.values,
            name="Regime Strategy",
            fill="tozeroy",
            line=dict(color="#16a085", width=1.5),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def metrics_comparison_table(
    bh_metrics: dict[str, float],
    regime_metrics: dict[str, float],
) -> go.Figure:
    """Side-by-side metrics as a Plotly table."""
    labels = [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
    ]
    keys = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]

    def fmt(key: str, val: float) -> str:
        if key in ("sharpe_ratio",):
            return f"{val:.2f}"
        return f"{val:.2%}"

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Buy & Hold", "Regime Strategy"],
                    fill_color="#2c3e50",
                    font=dict(color="white", size=13),
                    align="left",
                ),
                cells=dict(
                    values=[
                        labels,
                        [fmt(k, bh_metrics[k]) for k in keys],
                        [fmt(k, regime_metrics[k]) for k in keys],
                    ],
                    fill_color=[["#ecf0f1"] * len(labels)] * 3,
                    align="left",
                    font=dict(size=12),
                ),
            )
        ]
    )
    fig.update_layout(title="Performance Metrics", template="plotly_white")
    return fig


def regime_pie_chart(distribution: pd.DataFrame) -> go.Figure:
    """Pie chart of time spent in each regime."""
    fig = px.pie(
        distribution.reset_index().rename(columns={"index": "regime"}),
        names="regime",
        values="days",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        title="Regime Occupancy",
    )
    fig.update_layout(template="plotly_white")
    return fig


def cluster_stats_bar(cluster_stats: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of per-cluster summary statistics."""
    plot_df = cluster_stats.reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=plot_df["regime_label"], y=plot_df["mean_return"], name="Mean Return"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=plot_df["regime_label"], y=plot_df["mean_volatility"], name="Mean Volatility"),
        secondary_y=True,
    )
    fig.update_layout(
        title="Cluster Centroid Statistics",
        template="plotly_white",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Mean Daily Return", secondary_y=False, tickformat=".2%")
    fig.update_yaxes(title_text="Annualized Vol (20d)", secondary_y=True, tickformat=".0%")
    return fig
