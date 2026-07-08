"""
dashboard/simulation.py
=======================

Monte Carlo simulation module for the Stockout Risk Dashboard.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard.charts import create_simulation_histogram
from dashboard.config import (
    HISTOGRAM_HEIGHT,
    N_SIMULATIONS,
    RANDOM_STATE,
    SIMULATION_DAYS,
)


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def run_simulation(
    row,
    simulation_row,
):
    """
    Run Monte Carlo inventory simulation.

    Parameters
    ----------
    row : pd.Series
        Original SKU record.

    simulation_row : pd.Series
        User-adjusted scenario.

    Returns
    -------
    dict
    """

    rng = np.random.default_rng(RANDOM_STATE)

    daily_mean = (
        simulation_row["avg_daily_sales"]
        * simulation_row["seasonality_index"]
    )

    daily_std = max(
        simulation_row["sales_volatility"],
        0.01,
    )

    daily_sales = rng.normal(
        loc=daily_mean,
        scale=daily_std,
        size=(N_SIMULATIONS, SIMULATION_DAYS),
    )

    daily_sales = np.clip(
        daily_sales,
        0,
        None,
    )

    cumulative_sales = np.cumsum(
        daily_sales,
        axis=1,
    )

    inventory_curve = (
        simulation_row["current_stock"]
        + simulation_row["safety_stock_level"]
        - cumulative_sales
    )

    stockout_mask = inventory_curve <= 0

    days_until_stockout = np.where(
        stockout_mask.any(axis=1),
        stockout_mask.argmax(axis=1) + 1,
        SIMULATION_DAYS + 1,
    )

    probability_30d = (
        days_until_stockout <= 30
    ).mean()

    expected_days = np.mean(days_until_stockout)

    median_days = np.median(days_until_stockout)

    return {
        "days_until_stockout": days_until_stockout,
        "probability_30d": probability_30d,
        "expected_days": expected_days,
        "median_days": median_days,
    }


# =============================================================================
# Render
# =============================================================================

def render_simulation(results):
    """
    Render Monte Carlo simulation section.

    Parameters
    ----------
    results : dict
    """

    st.subheader(
        "B. Monte Carlo Simulation — Days Until Stockout"
    )

    fig = create_simulation_histogram(
        results["days_until_stockout"],
        SIMULATION_DAYS,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "30-Day Stockout",
            f"{results['probability_30d']:.1%}",
        )

    with col2:
        st.metric(
            "Expected Days",
            f"{results['expected_days']:.1f}",
        )

    with col3:
        st.metric(
            "Median Days",
            f"{results['median_days']:.0f}",
        )

    st.caption(
        f"""
Simulation performed using **{N_SIMULATIONS:,} Monte Carlo runs**
over a **{SIMULATION_DAYS}-day horizon**.
Sales variability is sampled from a normal distribution based on the
SKU's historical average daily sales and sales volatility.
"""
    )