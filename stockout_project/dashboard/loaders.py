"""
dashboard/loaders.py
====================

Cached data and model loaders for the Stockout Risk Dashboard.
"""

from __future__ import annotations

import joblib
import pandas as pd
import shap
import streamlit as st

from dashboard.config import ARTIFACT_PATH, DATA_PATH


# =============================================================================
# Model Artifacts
# =============================================================================

@st.cache_resource(show_spinner="Loading trained models...")
def load_artifacts() -> dict:
    """
    Load trained model artifacts.

    Returns
    -------
    dict
        Dictionary containing:
        - calibrated_model
        - shap_model
        - features
    """

    return joblib.load(ARTIFACT_PATH)


# =============================================================================
# Dataset
# =============================================================================

@st.cache_data(show_spinner="Loading inventory dataset...")
def load_dataset() -> pd.DataFrame:
    """
    Load inventory dataset.

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(DATA_PATH)


# =============================================================================
# SHAP Explainer
# =============================================================================

@st.cache_resource(show_spinner="Initializing SHAP explainer...")
def load_shap_explainer(_shap_model):
    """
    Create SHAP TreeExplainer.

    Parameter diberi underscore agar Streamlit
    tidak melakukan hashing terhadap XGBoost model object.
    """

    return shap.TreeExplainer(_shap_model)


# =============================================================================
# Combined Loader
# =============================================================================

def load_dashboard_resources():
    """
    Load everything required by dashboard.

    Returns
    -------
    tuple
        (
            dataframe,
            calibrated_model,
            shap_model,
            explainer,
            feature_names
        )
    """

    artifacts = load_artifacts()

    df = load_dataset()

    calibrated_model = artifacts["calibrated_model"]
    shap_model = artifacts["shap_model"]
    feature_names = artifacts["features"]

    explainer = load_shap_explainer(shap_model)

    return (
        df,
        calibrated_model,
        shap_model,
        explainer,
        feature_names,
    )