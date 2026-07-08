"""
dashboard/charts.py
===================

Reusable Plotly charts for the Stockout Risk Dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.config import (
    GAUGE_COLORS,
    GAUGE_HEIGHT,
    GAUGE_MAX,
    GAUGE_MIN,
    GAUGE_STEPS,
    HISTOGRAM_BINS,
    HISTOGRAM_COLOR,
    HISTOGRAM_HEIGHT,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    SHAP_CHART_HEIGHT,
    SHAP_NEGATIVE_COLOR,
    SHAP_POSITIVE_COLOR,
)


# =============================================================================
# Gauge Chart
# =============================================================================

def create_gauge_chart(probability: float) -> go.Figure:
    """
    Create stockout probability gauge.

    Parameters
    ----------
    probability : float

    Returns
    -------
    go.Figure
    """

    probability_pct = probability * 100

    if probability < LOW_RISK_THRESHOLD:
        color = GAUGE_COLORS["low"]

    elif probability < MEDIUM_RISK_THRESHOLD:
        color = GAUGE_COLORS["medium"]

    else:
        color = GAUGE_COLORS["high"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_pct,
            number={"suffix": "%"},
            title={"text": "30-Day Stockout Probability"},
            gauge={
                "axis": {
                    "range": [GAUGE_MIN, GAUGE_MAX],
                },
                "bar": {
                    "color": color,
                },
                "steps": GAUGE_STEPS,
            },
        )
    )

    fig.update_layout(
        height=GAUGE_HEIGHT,
        margin=dict(
            t=50,
            b=10,
            l=10,
            r=10,
        ),
    )

    return fig


# =============================================================================
# Monte Carlo Histogram
# =============================================================================

def create_simulation_histogram(
    days_until_stockout,
    simulation_days: int,
) -> go.Figure:
    """
    Create histogram of simulated stockout days.

    Parameters
    ----------
    days_until_stockout : np.ndarray

    simulation_days : int

    Returns
    -------
    go.Figure
    """

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=days_until_stockout[
                days_until_stockout <= simulation_days
            ],
            nbinsx=HISTOGRAM_BINS,
            marker_color=HISTOGRAM_COLOR,
            name="Simulation",
        )
    )

    fig.add_vline(
        x=30,
        line_dash="dash",
        line_color="red",
        annotation_text="30-Day Threshold",
        annotation_position="top",
    )

    fig.update_layout(
        height=HISTOGRAM_HEIGHT,
        xaxis_title="Days Until Stockout",
        yaxis_title="Simulation Count",
        margin=dict(
            t=30,
            b=10,
            l=10,
            r=10,
        ),
    )

    return fig


# =============================================================================
# SHAP Bar Chart
# =============================================================================

def create_shap_chart(
    contributions: pd.DataFrame,
) -> go.Figure:
    """
    Create SHAP contribution bar chart.

    Parameters
    ----------
    contributions : pd.DataFrame

    Returns
    -------
    go.Figure
    """

    colors = [
        SHAP_POSITIVE_COLOR
        if value > 0
        else SHAP_NEGATIVE_COLOR
        for value in contributions["shap_value"]
    ]

    fig = go.Figure(
        go.Bar(
            x=contributions["shap_value"],
            y=contributions["feature"],
            orientation="h",
            marker_color=colors,
        )
    )

    fig.update_layout(
        height=SHAP_CHART_HEIGHT,
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        margin=dict(
            t=20,
            b=10,
            l=10,
            r=10,
        ),
    )

    return fig