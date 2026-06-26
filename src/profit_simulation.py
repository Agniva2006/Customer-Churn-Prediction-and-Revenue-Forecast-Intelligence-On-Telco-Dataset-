"""Profit-optimized threshold selection and Monte Carlo revenue simulation."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def simulate_profit(
    y_true,
    y_probs,
    retention_cost: float = 500,
    annual_revenue: float = 6000,
    save_rate: float = 0.6,
) -> pd.DataFrame:
    """Compute net profit across a range of classification thresholds.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_probs : array-like
        Predicted churn probabilities.
    retention_cost : float
        Cost to attempt retaining one customer (₹).
    annual_revenue : float
        Annual revenue per customer (₹).
    save_rate : float
        Probability a retention offer succeeds.

    Returns
    -------
    pd.DataFrame
        Columns: threshold, TP, FP, FN, TN, net_profit.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        saved_revenue = tp * annual_revenue * save_rate
        cost = (tp + fp) * retention_cost
        net_profit = saved_revenue - cost

        results.append({
            "threshold": round(t, 3),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "net_profit": round(net_profit, 2),
        })

    return pd.DataFrame(results)


def find_optimal_threshold(
    y_true,
    y_probs,
    retention_cost: float = 500,
    annual_revenue: float = 6000,
    save_rate: float = 0.6,
) -> Tuple[float, float]:
    """Return the threshold that maximises net profit.

    Returns
    -------
    tuple
        (optimal_threshold, max_net_profit)
    """
    df = simulate_profit(y_true, y_probs, retention_cost, annual_revenue, save_rate)
    best = df.loc[df["net_profit"].idxmax()]
    return float(best["threshold"]), float(best["net_profit"])


def monte_carlo_revenue(
    n_customers: int,
    avg_revenue: float,
    churn_rate_mean: float,
    churn_rate_std: float,
    n_simulations: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Simulate total revenue under churn uncertainty.

    Draws ``n_simulations`` churn rates from a clipped normal distribution
    and computes total revenue for each scenario.

    Parameters
    ----------
    n_customers : int
        Current customer base size.
    avg_revenue : float
        Average annual revenue per customer.
    churn_rate_mean : float
        Expected churn rate (e.g. 0.26).
    churn_rate_std : float
        Standard deviation of churn rate.
    n_simulations : int
        Number of Monte Carlo iterations.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    np.ndarray
        Array of simulated total revenues (length ``n_simulations``).
    """
    rng = np.random.default_rng(random_state)
    churn_rates = rng.normal(churn_rate_mean, churn_rate_std, n_simulations)
    churn_rates = np.clip(churn_rates, 0, 1)
    retained = n_customers * (1 - churn_rates)
    return retained * avg_revenue