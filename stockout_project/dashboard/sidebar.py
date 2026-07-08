"""
dashboard/sidebar.py
====================

Sidebar components for the Stockout Risk Dashboard.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import (
    AVG_DAILY_SALES,
    FEATURES,
    LEAD_TIME,
    SAFETY_STOCK,
)


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar(df: pd.DataFrame):
    """
    Render the dashboard sidebar.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple
        (
            sku_id,
            original_row,
            simulation_row,
            X_sim
        )
    """

    st.sidebar.header("Inventory Scenario")

    sku_id = st.sidebar.selectbox(
        label="Select SKU",
        options=df["sku_id"].tolist(),
    )

    row = df.loc[df["sku_id"] == sku_id].iloc[0].copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Parameters")

    safety_stock = st.sidebar.slider(
        label="Safety Stock Level",
        min_value=SAFETY_STOCK["min"],
        max_value=SAFETY_STOCK["max"],
        value=int(row["safety_stock_level"]),
        step=SAFETY_STOCK["step"],
    )

    lead_time = st.sidebar.slider(
        label="Supplier Lead Time (Days)",
        min_value=LEAD_TIME["min"],
        max_value=LEAD_TIME["max"],
        value=int(row["supplier_lead_time_days"]),
        step=LEAD_TIME["step"],
    )

    avg_daily_sales = st.sidebar.slider(
        label="Average Daily Sales",
        min_value=AVG_DAILY_SALES["min"],
        max_value=AVG_DAILY_SALES["max"],
        value=float(row["avg_daily_sales"]),
        step=AVG_DAILY_SALES["step"],
    )

    sim_row = row.copy()

    sim_row["safety_stock_level"] = safety_stock
    sim_row["supplier_lead_time_days"] = lead_time
    sim_row["avg_daily_sales"] = avg_daily_sales

    X_sim = pd.DataFrame([sim_row[FEATURES]])

    return (
        sku_id,
        row,
        sim_row,
        X_sim,
    )