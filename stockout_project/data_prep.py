"""
data_prep.py
============

Validate and clean the raw inventory dataset before model training.

Responsibilities
----------------
- Validate required schema
- Validate data types
- Handle missing values
- Remove duplicate SKU IDs
- Validate business rules and value ranges
- Save a clean dataset for training

Run:
    python data_prep.py

Input:
    inventory_dataset.csv

Output:
    inventory_dataset_clean.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
)

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "inventory_dataset.csv"
CLEAN_PATH = BASE_DIR / "inventory_dataset_clean.csv"

EXPECTED_COLUMNS = {
    "sku_id": "string",
    "current_stock": "int",
    "avg_daily_sales": "float",
    "sales_volatility": "float",
    "supplier_lead_time_days": "int",
    "safety_stock_level": "int",
    "reorder_point": "int",
    "seasonality_index": "float",
    "historical_stockout_count": "int",
    "will_stockout_next_30_days": "int",
}

NON_NEGATIVE_COLUMNS = [
    "current_stock",
    "avg_daily_sales",
    "sales_volatility",
    "supplier_lead_time_days",
    "safety_stock_level",
    "reorder_point",
    "historical_stockout_count",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Load Dataset
# =============================================================================


def load_raw(path: Path) -> pd.DataFrame:
    """
    Load raw dataset.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


# =============================================================================
# Schema Validation
# =============================================================================


def check_schema(df: pd.DataFrame) -> None:
    """
    Validate required columns and data types.
    """

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        logger.warning("Extra columns detected (ignored): %s", sorted(extra_cols))

    for col, expected in EXPECTED_COLUMNS.items():

        if expected == "int":
            if not is_integer_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be integer.")

        elif expected == "float":
            if not (is_float_dtype(df[col]) or is_integer_dtype(df[col])):
                raise TypeError(f"Column '{col}' must be numeric.")

        elif expected == "string":
            if not is_object_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be string/object.")

    logger.info("Schema validation passed.")


# =============================================================================
# Missing Values
# =============================================================================


def remove_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows containing missing values.
    """

    missing_summary = df.isna().sum()

    if missing_summary.sum() > 0:

        logger.warning("Missing values detected:")

        for col, n in missing_summary.items():
            if n > 0:
                logger.warning("  %-30s %d", col, n)

        before = len(df)
        df = df.dropna().copy()
        removed = before - len(df)

        logger.info("%d rows removed due to missing values.", removed)

        return df, removed

    logger.info("No missing values found.")

    return df, 0


# =============================================================================
# Duplicate SKU
# =============================================================================


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicated SKU IDs.
    """

    before = len(df)

    df = df.drop_duplicates(subset="sku_id", keep="first").copy()

    removed = before - len(df)

    if removed:
        logger.warning("%d duplicate SKU IDs removed.", removed)
    else:
        logger.info("No duplicate SKU IDs found.")

    return df, removed


# =============================================================================
# Business Rules
# =============================================================================


def validate_business_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows violating business rules.
    """

    invalid = pd.Series(False, index=df.index)

    # Non-negative values
    for col in NON_NEGATIVE_COLUMNS:
        invalid |= df[col] < 0

    # Binary target
    invalid |= ~df["will_stockout_next_30_days"].isin([0, 1])

    # Seasonality range
    invalid |= ~df["seasonality_index"].between(0.3, 2.0)

    # Logical relationships
    invalid |= df["reorder_point"] < df["safety_stock_level"]

    invalid |= df["supplier_lead_time_days"] > 365

    removed = int(invalid.sum())

    if removed:
        logger.warning("%d invalid rows removed.", removed)
        df = df.loc[~invalid].copy()
    else:
        logger.info("Business rule validation passed.")

    return df, removed


# =============================================================================
# Save Dataset
# =============================================================================


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """
    Save cleaned dataset.
    """

    df.to_csv(path, index=False)

    logger.info("Clean dataset saved to: %s", path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    logger.info("Loading dataset...")
    df = load_raw(RAW_PATH)

    initial_rows = len(df)

    logger.info("Initial rows: %d", initial_rows)

    check_schema(df)

    df, missing_removed = remove_missing(df)

    df, duplicate_removed = remove_duplicates(df)

    df, invalid_removed = validate_business_rules(df)

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Dataset is empty after cleaning. Training cannot continue."
        )

    save_dataset(df, CLEAN_PATH)

    logger.info("")
    logger.info("=" * 45)
    logger.info("DATA PREPARATION SUMMARY")
    logger.info("=" * 45)
    logger.info("Initial rows          : %d", initial_rows)
    logger.info("Missing removed       : %d", missing_removed)
    logger.info("Duplicates removed    : %d", duplicate_removed)
    logger.info("Invalid rows removed  : %d", invalid_removed)
    logger.info("Final rows            : %d", len(df))
    logger.info("=" * 45)


if __name__ == "__main__":
    main()