"""Unit tests for individualized profit simulation and threshold optimization."""

import numpy as np
from src.profit_simulation import compute_individualized_profit, find_optimal_threshold


def test_compute_individualized_profit():
    # High risk, high monthly charge -> Action: VIP Concierge
    res1 = compute_individualized_profit(monthly_charge=85.0, prob=0.75, threshold=0.15)
    assert res1["decision"] == "retain"
    assert res1["expected_profit"] > 0
    assert "VIP" in res1["action_quadrant"]

    # Low risk -> Action: No action
    res2 = compute_individualized_profit(monthly_charge=50.0, prob=0.05, threshold=0.15)
    assert res2["decision"] == "no_action"
    assert res2["expected_profit"] == 0.0


def test_find_optimal_threshold():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.15, 0.05, 0.3, 0.6, 0.2])

    opt_thresh, max_profit = find_optimal_threshold(y_true, y_probs)
    assert 0.05 <= opt_thresh <= 0.95
    assert isinstance(max_profit, float)
