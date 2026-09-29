"""EDA utilities for ECON 5200 labs.

Functions for data quality checking, distribution shift detection,
and automated EDA summaries.
"""

import numpy as np
import pandas as pd


def check_impossible_values(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Check columns against domain constraints and flag violations.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to validate.
    constraints : dict
        Column name -> (min_val, max_val) or callable.
        If a tuple, flags rows outside [min_val, max_val].
        If a callable, flags rows where callable(value) returns False.

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean columns '{col}_flagged' for each
        constrained column. True = violation detected.
    """
    result = df.copy()

    for col, rule in constraints.items():
        if col not in df.columns:
            continue

        if callable(rule):
            result[f"{col}_flagged"] = ~rule(df[col])
        else:
            min_val, max_val = rule
            result[f"{col}_flagged"] = (
                (df[col] < min_val) | (df[col] > max_val)
            )

    flag_cols = [c for c in result.columns if c.endswith("_flagged")]
    n_flagged = result[flag_cols].any(axis=1).sum()
    print(f"Flagged {n_flagged} rows with constraint violations.")
    for fc in flag_cols:
        n = result[fc].sum()
        if n > 0:
            print(f"  {fc}: {n} violations")

    return result


def detect_distribution_shift(
    train_col, inference_col, n_bins: int = 10
) -> tuple[float, str]:
    """Compute Population Stability Index between two distributions.

    Parameters
    ----------
    train_col : array-like
        Reference distribution.
    inference_col : array-like
        New distribution.
    n_bins : int, default 10
        Number of histogram bins.

    Returns
    -------
    tuple[float, str]
        PSI value and its interpretation.
    """
    train_arr = np.asarray(train_col, dtype=float)
    inf_arr = np.asarray(inference_col, dtype=float)

    breakpoints = np.linspace(
        min(train_arr.min(), inf_arr.min()),
        max(train_arr.max(), inf_arr.max()),
        n_bins + 1,
    )

    expected_counts = np.histogram(train_arr, bins=breakpoints)[0]
    actual_counts = np.histogram(inf_arr, bins=breakpoints)[0]

    eps = 1e-4
    expected_pct = expected_counts / len(train_arr) + eps
    actual_pct = actual_counts / len(inf_arr) + eps

    psi = float(np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    ))

    if psi < 0.1:
        interpretation = "stable"
    elif psi < 0.25:
        interpretation = "moderate_shift"
    else:
        interpretation = "significant_shift"

    return psi, interpretation


def eda_summary(df: pd.DataFrame) -> dict:
    """Return a structured EDA report for a DataFrame."""
    report = {}

    report["shape"] = df.shape
    report["dtypes"] = df.dtypes.to_dict()

    missing = df.isnull().sum()
    report["missing"] = missing[missing > 0].to_dict()
    report["duplicates"] = int(df.duplicated().sum())

    numeric_df = df.select_dtypes(include=[np.number])
    report["numeric_summary"] = numeric_df.describe().to_dict()

    outlier_counts = {}
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(
            ((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum()
        )
        if n_outliers > 0:
            outlier_counts[col] = n_outliers
    report["outlier_counts"] = outlier_counts

    print(f"Shape: {report['shape']}")
    print(f"Duplicates: {report['duplicates']}")
    if report["missing"]:
        print(f"Missing values: {report['missing']}")
    else:
        print("Missing values: None")
    if report["outlier_counts"]:
        print(f"Outliers (IQR): {report['outlier_counts']}")
    else:
        print("Outliers (IQR): None detected")

    return report
