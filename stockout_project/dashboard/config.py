"""
dashboard/config.py
===================

Central configuration for the Stockout Risk Dashboard.

This module stores:
- File paths
- Dashboard settings
- Visualization settings
- Business thresholds
- Simulation parameters
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "inventory_dataset_clean.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifacts.joblib"

# =============================================================================
# Dashboard
# =============================================================================

PAGE_TITLE = "Stockout Risk Dashboard"
PAGE_ICON = "📦"
LAYOUT = "wide"

# =============================================================================
# Risk Thresholds
# =============================================================================

LOW_RISK_THRESHOLD = 0.30
MEDIUM_RISK_THRESHOLD = 0.60
TARGET_PROBABILITY = 0.10

# =============================================================================
# Gauge Chart
# =============================================================================

GAUGE_MIN = 0
GAUGE_MAX = 100

GAUGE_HEIGHT = 320

GAUGE_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red",
}

GAUGE_STEPS = [
    {
        "range": [0, 30],
        "color": "#d4f4dd",
    },
    {
        "range": [30, 60],
        "color": "#fff3cd",
    },
    {
        "range": [60, 100],
        "color": "#f8d7da",
    },
]

# =============================================================================
# Monte Carlo Simulation
# =============================================================================

N_SIMULATIONS = 1_000
SIMULATION_DAYS = 60
RANDOM_STATE = 42

# =============================================================================
# Sidebar Slider Limits
# =============================================================================

SAFETY_STOCK = {
    "min": 0,
    "max": 300,
    "step": 1,
}

LEAD_TIME = {
    "min": 1,
    "max": 40,
    "step": 1,
}

AVG_DAILY_SALES = {
    "min": 0.0,
    "max": 40.0,
    "step": 0.1,
}

# =============================================================================
# What-If Search
# =============================================================================

SAFETY_STOCK_SEARCH = {
    "max_increase": 250,
    "step": 10,
}

LEAD_TIME_SEARCH = {
    "min": 1,
}

# =============================================================================
# SHAP
# =============================================================================

SHAP_POSITIVE_COLOR = "#d9534f"
SHAP_NEGATIVE_COLOR = "#5cb85c"
SHAP_CHART_HEIGHT = 350

# =============================================================================
# Histogram
# =============================================================================

HISTOGRAM_HEIGHT = 320
HISTOGRAM_BINS = 30
HISTOGRAM_COLOR = "steelblue"

# =============================================================================
# Feature Columns
# =============================================================================

FEATURES = [
    "current_stock",
    "avg_daily_sales",
    "sales_volatility",
    "supplier_lead_time_days",
    "safety_stock_level",
    "reorder_point",
    "seasonality_index",
    "historical_stockout_count",
]