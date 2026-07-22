"""Real-time data drift and prediction drift monitoring engine.

Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics
to detect distribution shifts between baseline training data and production inference batches.
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10
) -> float:
    """Compute the Population Stability Index (PSI) between baseline and actual arrays.

    PSI Interpretation:
    * PSI < 0.10: No significant distribution change.
    * 0.10 <= PSI < 0.25: Moderate drift — warning state.
    * PSI >= 0.25: Actionable shift — model retraining required.
    """
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)

    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Replace zero percentages with small epsilon to prevent log(0)
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def evaluate_ks_drift(
    baseline: Union[np.ndarray, pd.Series],
    production: Union[np.ndarray, pd.Series],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Perform a two-sample Kolmogorov-Smirnov test for continuous variable drift."""
    baseline = np.array(baseline, dtype=float)
    production = np.array(production, dtype=float)

    baseline = baseline[~np.isnan(baseline)]
    production = production[~np.isnan(production)]

    if len(baseline) < 5 or len(production) < 5:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "status": "insufficient_data",
        }

    ks_stat, p_val = stats.ks_2samp(baseline, production)
    drift_detected = bool(p_val < alpha)

    return {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 6),
        "drift_detected": drift_detected,
        "status": "drift_detected" if drift_detected else "stable",
    }


def evaluate_batch_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str] = None
) -> Dict[str, Any]:
    """Evaluate drift across multiple numeric features between baseline and current data."""
    if features is None:
        features = ["tenure", "MonthlyCharges", "TotalCharges"]

    results = {}
    overall_drift = False

    for feat in features:
        if feat in baseline_df.columns and feat in current_df.columns:
            base_col = pd.to_numeric(baseline_df[feat], errors="coerce").dropna()
            curr_col = pd.to_numeric(current_df[feat], errors="coerce").dropna()

            psi_score = calculate_psi(base_col.values, curr_col.values)
            ks_res = evaluate_ks_drift(base_col.values, curr_col.values)

            feat_drift = (psi_score >= 0.25) or ks_res["drift_detected"]
            if feat_drift:
                overall_drift = True

            results[feat] = {
                "psi": round(psi_score, 4),
                "ks_stat": ks_res["ks_statistic"],
                "p_value": ks_res["p_value"],
                "drift_detected": feat_drift,
            }

    return {
        "overall_drift_detected": overall_drift,
        "features_evaluated": len(results),
        "feature_metrics": results,
    }
