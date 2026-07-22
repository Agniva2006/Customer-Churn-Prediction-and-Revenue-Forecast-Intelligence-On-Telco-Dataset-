"""Profit-optimized threshold selection, CLV-based individualized action recommendations, and Monte Carlo risk simulation."""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_individualized_profit(
    monthly_charge: float,
    prob: float,
    threshold: float = 0.15,
    retention_cost: float = 500.0,
    save_rate: float = 0.6,
    multiplier: float = 12.0
) -> Dict[str, Any]:
    """Compute individual customer lifetime profit and assign retention action quadrant.

    Parameters
    ----------
    monthly_charge : float
        Customer's monthly billing rate.
    prob : float
        Predicted probability of churn.
    threshold : float
        Targeting threshold.
    retention_cost : float
        Cost of retention intervention (₹).
    save_rate : float
        Success probability of retention offer.
    multiplier : float
        Annualization multiplier (default 12 months).

    Returns
    -------
    dict
        Contains expected_clv, expected_profit, decision, action_quadrant, priority.
    """
    individual_clv = monthly_charge * multiplier

    if prob >= threshold:
        decision = "retain"
        expected_profit = (save_rate * individual_clv) - retention_cost
    else:
        decision = "no_action"
        expected_profit = 0.0

    # Assign Action Matrix Quadrant
    if prob >= 0.6:
        if monthly_charge >= 75.0:
            quadrant = "VIP Concierge Outreach & High-Value Offer"
            priority = "P1 — Critical"
        else:
            quadrant = "Automated Digital Retention Discount"
            priority = "P2 — High"
    elif prob >= 0.3:
        quadrant = "Targeted Loyalty Campaign & Service Check-in"
        priority = "P3 — Medium"
    else:
        quadrant = "Standard Account Service — No Intervention Required"
        priority = "P4 — Low"

    return {
        "clv": round(individual_clv, 2),
        "expected_profit": round(expected_profit, 2),
        "decision": decision,
        "action_quadrant": quadrant,
        "priority": priority,
    }


def simulate_profit(
    y_true,
    y_probs,
    retention_cost: float = 500,
    annual_revenue: float = 6000,
    save_rate: float = 0.6,
) -> pd.DataFrame:
    """Compute net profit across a range of classification thresholds."""
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
    """Return the threshold that maximises net profit."""
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
    """Simulate total revenue under churn uncertainty via Monte Carlo drawing."""
    rng = np.random.default_rng(random_state)
    churn_rates = rng.normal(churn_rate_mean, churn_rate_std, n_simulations)
    churn_rates = np.clip(churn_rates, 0, 1)
    retained = n_customers * (1 - churn_rates)
    return retained * avg_revenue