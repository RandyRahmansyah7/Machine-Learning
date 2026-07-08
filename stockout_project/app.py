from __future__ import annotations

import streamlit as st

from dashboard.config import (
    LAYOUT,
    PAGE_ICON,
    PAGE_TITLE,
)

from dashboard.loaders import load_dashboard_resources
from dashboard.sidebar import render_sidebar

from dashboard.prediction import (
    predict_probability,
    render_prediction,
)

from dashboard.simulation import (
    run_simulation,
    render_simulation,
)

from dashboard.explainability import (
    compute_shap_values,
    render_explainability,
)

from dashboard.whatif import (
    search_recommendation,
    render_what_if,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
)


# =============================================================================
# Load Resources
# =============================================================================

(
    df,
    calibrated_model,
    shap_model,
    explainer,
    features,
) = load_dashboard_resources()


# =============================================================================
# Sidebar
# =============================================================================

(
    sku_id,
    original_row,
    simulation_row,
    X_sim,
) = render_sidebar(df)


# =============================================================================
# Header
# =============================================================================

st.title(
    f"📦 Stockout Risk Dashboard — {sku_id}"
)

st.caption(
    "Predict → Simulate → Explain → What-If Recommendation"
)


# =============================================================================
# Prediction
# =============================================================================

probability = predict_probability(
    calibrated_model,
    X_sim,
)


left, right = st.columns(2)


with left:

    render_prediction(
        probability
    )


with right:

    simulation_result = run_simulation(
        original_row,
        simulation_row,
    )

    render_simulation(
        simulation_result
    )


# =============================================================================
# Explainability
# =============================================================================

st.markdown("---")

shap_contributions = compute_shap_values(
    explainer,
    X_sim,
)

render_explainability(
    shap_contributions
)


# =============================================================================
# What-If Recommendation
# =============================================================================

st.markdown("---")

recommendation = search_recommendation(
    calibrated_model,
    simulation_row,
)

render_what_if(
    recommendation
)


# =============================================================================
# Raw SKU Data
# =============================================================================

st.markdown("---")

with st.expander(
    "View Selected SKU Data"
):

    st.dataframe(
        original_row.to_frame().T,
        use_container_width=True,
    )