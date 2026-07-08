"""
dashboard/prediction.py
=======================

Prediction module for the Stockout Risk Dashboard.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import create_gauge_chart
from dashboard.config import (
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)


# =============================================================================
# Prediction
# =============================================================================

def predict_probability(
    model,
    X: pd.DataFrame,
) -> float:
    """
    Predict calibrated stockout probability.

    Parameters
    ----------
    model
        Trained calibrated classifier.

    X : pd.DataFrame
        Feature dataframe.

    Returns
    -------
    float
        Probability of stockout.
    """

    return float(model.predict_proba(X)[0, 1])


# =============================================================================
# Risk Level
# =============================================================================

def get_risk_level(probability: float) -> str:
    """
    Convert probability into business risk level.

    Parameters
    ----------
    probability : float

    Returns
    -------
    str
    """

    if probability < LOW_RISK_THRESHOLD:
        return "Low"

    if probability < MEDIUM_RISK_THRESHOLD:
        return "Medium"

    return "High"


# =============================================================================
# Recommendation
# =============================================================================

def get_recommendation(probability: float) -> str:
    """
    Generate a short business recommendation.

    Parameters
    ----------
    probability : float

    Returns
    -------
    str
    """

    if probability < LOW_RISK_THRESHOLD:
        return (
            "Inventory level is healthy. Continue monitoring demand "
            "and replenish according to the existing policy."
        )

    if probability < MEDIUM_RISK_THRESHOLD:
        return (
            "Inventory risk is increasing. Review replenishment plans "
            "and monitor supplier performance."
        )

    return (
        "High stockout risk detected. Immediate replenishment or "
        "supplier intervention is recommended."
    )


# =============================================================================
# Render
# =============================================================================

def render_prediction(
    probability: float,
) -> None:
    """
    Render prediction section.

    Parameters
    ----------
    probability : float
    """

    st.subheader("A. Calibrated Stockout Probability")

    fig = create_gauge_chart(probability)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    risk_level = get_risk_level(probability)

    recommendation = get_recommendation(probability)

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            label="Risk Level",
            value=risk_level,
        )

    with metric_col2:
        st.metric(
            label="Probability",
            value=f"{probability:.1%}",
        )

    if risk_level == "Low":
        st.success(recommendation)

    elif risk_level == "Medium":
        st.warning(recommendation)

    else:
        st.error(recommendation)