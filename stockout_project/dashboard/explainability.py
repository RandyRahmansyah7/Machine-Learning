"""
dashboard/explainability.py
===========================

SHAP explainability module for the Stockout Risk Dashboard.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import create_shap_chart
from dashboard.config import (
    FEATURES,
    SHAP_NEGATIVE_COLOR,
    SHAP_POSITIVE_COLOR,
)


# =============================================================================
# SHAP Values
# =============================================================================

def compute_shap_values(
    explainer,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SHAP values for a single prediction.

    Parameters
    ----------
    explainer : shap.TreeExplainer

    X : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    shap_values = explainer.shap_values(X)

    # Older versions of SHAP return a list for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contributions = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "shap_value": shap_values[0],
            }
        )
        .sort_values(
            by="shap_value",
            key=lambda s: s.abs(),
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return contributions


# =============================================================================
# Business Summary
# =============================================================================

def summarize_contributions(
    contributions: pd.DataFrame,
):
    """
    Split SHAP contributions into risk-increasing and risk-reducing features.

    Parameters
    ----------
    contributions : pd.DataFrame

    Returns
    -------
    tuple
        (
            positive_features,
            negative_features,
        )
    """

    positive = (
        contributions[contributions["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
    )

    negative = (
        contributions[contributions["shap_value"] < 0]
        .sort_values("shap_value")
    )

    return positive, negative


# =============================================================================
# Render
# =============================================================================

def render_explainability(
    contributions: pd.DataFrame,
):
    """
    Render SHAP explainability section.

    Parameters
    ----------
    contributions : pd.DataFrame
    """

    st.subheader("C. Feature Contribution (SHAP)")

    fig = create_shap_chart(contributions)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    positive, negative = summarize_contributions(contributions)

    left, right = st.columns(2)

    with left:

        st.markdown("#### 🔴 Main Risk Drivers")

        if positive.empty:
            st.success("No feature is significantly increasing stockout risk.")

        else:
            for _, row in positive.head(3).iterrows():
                st.write(
                    f"• **{row['feature']}** "
                    f"({row['shap_value']:.3f})"
                )

    with right:

        st.markdown("#### 🟢 Main Protective Factors")

        if negative.empty:
            st.info("No feature is significantly reducing stockout risk.")

        else:
            for _, row in negative.head(3).iterrows():
                st.write(
                    f"• **{row['feature']}** "
                    f"({row['shap_value']:.3f})"
                )

    st.caption(
        f"""
**Interpretation**

- <span style="color:{SHAP_POSITIVE_COLOR};"><b>Positive SHAP values</b></span>
  increase the predicted probability of stockout.

- <span style="color:{SHAP_NEGATIVE_COLOR};"><b>Negative SHAP values</b></span>
  decrease the predicted probability of stockout.

The larger the absolute SHAP value, the greater the influence of that
feature on the prediction.
""",
        unsafe_allow_html=True,
    )