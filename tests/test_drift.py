"""Unit tests for Population Stability Index (PSI) and KS drift statistics."""

import numpy as np
from src.drift import calculate_psi, evaluate_ks_drift


def test_calculate_psi_identical():
    baseline = np.random.normal(100, 15, 1000)
    current = baseline.copy()
    psi_val = calculate_psi(baseline, current)
    assert psi_val < 0.05


def test_calculate_psi_shifted():
    baseline = np.random.normal(100, 15, 1000)
    current = np.random.normal(150, 15, 1000)
    psi_val = calculate_psi(baseline, current)
    assert psi_val > 0.25


def test_evaluate_ks_drift():
    baseline = np.random.normal(50, 5, 500)
    shifted = np.random.normal(80, 5, 500)
    ks_res = evaluate_ks_drift(baseline, shifted)

    assert ks_res["drift_detected"] is True
    assert ks_res["p_value"] < 0.05
