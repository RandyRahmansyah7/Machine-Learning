"""
dashboard/utils.py
==================

Shared utility functions for the Stockout Risk Dashboard.

This module contains reusable business logic used across
multiple dashboard components.
"""

from __future__ import annotations

import pandas as pd

from dashboard.config import (
    FEATURES,
    GAUGE_COLORS,
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

    Returns
    -------
    float
        Stockout probability.
    """

    return float(model.predict_proba(X)[0, 1])


# =============================================================================
# Risk Assessment
# =============================================================================

def get_risk_level(
    probability: float,
) -> str:
    """
    Convert probability into a business risk level.
    """

    if probability < LOW_RISK_THRESHOLD:
        return "Low"

    if probability < MEDIUM_RISK_THRESHOLD:
        return "Medium"

    return "High"


def get_risk_color(
    probability: float,
) -> str:
    """
    Return dashboard color for a probability.
    """

    if probability < LOW_RISK_THRESHOLD:
        return GAUGE_COLORS["low"]

    if probability < MEDIUM_RISK_THRESHOLD:
        return GAUGE_COLORS["medium"]

    return GAUGE_COLORS["high"]


def get_business_recommendation(
    probability: float,
) -> str:
    """
    Generate a short business recommendation.
    """

    level = get_risk_level(probability)

    recommendations = {
        "Low": (
            "Inventory level is healthy. Continue monitoring demand and "
            "follow the current replenishment policy."
        ),
        "Medium": (
            "Inventory risk is increasing. Review replenishment planning "
            "and monitor supplier performance."
        ),
        "High": (
            "Immediate action is recommended. Increase inventory, reduce "
            "supplier lead time, or secure an alternative supplier."
        ),
    }

    return recommendations[level]


# =============================================================================
# Formatting Helpers
# =============================================================================

def format_probability(
    probability: float,
) -> str:
    """
    Format probability as percentage.
    """

    return f"{probability:.1%}"


def format_days(
    days: float,
) -> str:
    """
    Format simulated days.
    """

    return f"{days:.1f} days"


def format_units(
    units: int,
) -> str:
    """
    Format inventory units.
    """

    return f"{units:,} units"


# =============================================================================
# Feature Builder
# =============================================================================

def build_feature_dataframe(
    simulation_row: pd.Series,
) -> pd.DataFrame:
    """
    Build a model-ready feature dataframe from a simulated SKU.

    Parameters
    ----------
    simulation_row : pd.Series

    Returns
    -------
    pd.DataFrame
    """

    return pd.DataFrame(
        [simulation_row[FEATURES]]
    )


# =============================================================================
# Prediction Summary
# =============================================================================

def build_prediction_summary(
    probability: float,
) -> dict:
    """
    Create a compact prediction summary used by dashboard components.

    Parameters
    ----------
    probability : float

    Returns
    -------
    dict
    """

    return {
        "probability": probability,
        "probability_text": format_probability(probability),
        "risk_level": get_risk_level(probability),
        "risk_color": get_risk_color(probability),
        "recommendation": get_business_recommendation(probability),
    }


# =============================================================================
# Inventory Metrics
# =============================================================================

def estimate_inventory_coverage(
    current_stock: float,
    safety_stock: float,
    avg_daily_sales: float,
) -> float:
    """
    Estimate how many days the inventory can cover.

    Parameters
    ----------
    current_stock : float

    safety_stock : float

    avg_daily_sales : float

    Returns
    -------
    float
    """

    if avg_daily_sales <= 0:
        return float("inf")

    available_inventory = current_stock + safety_stock

    return available_inventory / avg_daily_sales


def build_inventory_summary(
    simulation_row: pd.Series,
) -> dict:
    """
    Build inventory business metrics.

    Parameters
    ----------
    simulation_row : pd.Series

    Returns
    -------
    dict
    """

    coverage = estimate_inventory_coverage(
        current_stock=simulation_row["current_stock"],
        safety_stock=simulation_row["safety_stock_level"],
        avg_daily_sales=simulation_row["avg_daily_sales"],
    )

    return {
        "current_stock": simulation_row["current_stock"],
        "safety_stock": simulation_row["safety_stock_level"],
        "lead_time": simulation_row["supplier_lead_time_days"],
        "avg_daily_sales": simulation_row["avg_daily_sales"],
        "inventory_coverage_days": coverage,
    }