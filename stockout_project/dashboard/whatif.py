"""
dashboard/whatif.py
===================

What-if recommendation engine for the Stockout Risk Dashboard.

The objective is to search for the smallest operational change that
reduces the calibrated stockout probability below the target threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import (
    FEATURES,
    LEAD_TIME_SEARCH,
    SAFETY_STOCK_SEARCH,
    TARGET_PROBABILITY,
)


# =============================================================================
# Prediction Helper
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
        Trained calibrated model.

    X : pd.DataFrame

    Returns
    -------
    float
    """

    return float(model.predict_proba(X)[0, 1])


# =============================================================================
# Recommendation Search
# =============================================================================

def search_recommendation(
    model,
    simulation_row: pd.Series,
):
    """
    Search for the minimum operational adjustment that
    achieves the target probability.

    Search priority

    1. Increase safety stock.
    2. Reduce supplier lead time.

    Parameters
    ----------
    model

    simulation_row : pd.Series

    Returns
    -------
    dict | None
    """

    current_probability = predict_probability(
        model,
        pd.DataFrame([simulation_row[FEATURES]]),
    )

    if current_probability <= TARGET_PROBABILITY:

        return {
            "parameter": None,
            "value": None,
            "probability": current_probability,
            "status": "already_safe",
        }

    # ---------------------------------------------------------------------
    # Safety Stock Search
    # ---------------------------------------------------------------------

    current_ss = int(simulation_row["safety_stock_level"])

    safety_candidates = np.arange(
        current_ss,
        current_ss + SAFETY_STOCK_SEARCH["max_increase"] + 1,
        SAFETY_STOCK_SEARCH["step"],
    )

    for ss in safety_candidates:

        candidate = simulation_row.copy()
        candidate["safety_stock_level"] = ss

        probability = predict_probability(
            model,
            pd.DataFrame([candidate[FEATURES]]),
        )

        if probability <= TARGET_PROBABILITY:

            return {
                "parameter": "safety_stock_level",
                "value": int(ss),
                "probability": probability,
                "status": "found",
            }

    # ---------------------------------------------------------------------
    # Lead Time Search
    # ---------------------------------------------------------------------

    current_lt = int(simulation_row["supplier_lead_time_days"])

    for lt in range(
        current_lt,
        LEAD_TIME_SEARCH["min"] - 1,
        -1,
    ):

        candidate = simulation_row.copy()
        candidate["supplier_lead_time_days"] = lt

        probability = predict_probability(
            model,
            pd.DataFrame([candidate[FEATURES]]),
        )

        if probability <= TARGET_PROBABILITY:

            return {
                "parameter": "supplier_lead_time_days",
                "value": int(lt),
                "probability": probability,
                "status": "found",
            }

    return {
        "parameter": None,
        "value": None,
        "probability": current_probability,
        "status": "not_found",
    }


# =============================================================================
# Render
# =============================================================================

def render_what_if(
    recommendation: dict,
):
    """
    Render What-if Recommendation section.

    Parameters
    ----------
    recommendation : dict
    """

    st.subheader("D. What-If Recommendation")

    status = recommendation["status"]

    if status == "already_safe":

        st.success(
            f"""
Current stockout probability is already below
**{TARGET_PROBABILITY:.0%}**.

No additional inventory action is required.
"""
        )

        return

    if status == "found":

        parameter = recommendation["parameter"]

        value = recommendation["value"]

        probability = recommendation["probability"]

        if parameter == "safety_stock_level":

            st.success(
                f"""
### Recommended Action

Increase **Safety Stock Level**
to **{value} units**.

Estimated stockout probability:

**{probability:.1%}**
"""
            )

        else:

            st.success(
                f"""
### Recommended Action

Reduce **Supplier Lead Time**
to **{value} days**.

Estimated stockout probability:

**{probability:.1%}**
"""
            )

        return

    st.warning(
        """
No feasible recommendation was found using
Safety Stock or Supplier Lead Time alone.

Consider one or more of the following:

- Increase current inventory
- Reduce demand variability
- Improve demand forecasting
- Source an alternative supplier
- Split replenishment across multiple suppliers
"""
    )