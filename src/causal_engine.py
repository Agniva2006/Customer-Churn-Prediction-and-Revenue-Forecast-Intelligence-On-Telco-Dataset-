#!/usr/bin/env python3
"""
causal_engine.py
TelcoPulse: Causal Prescriptive AI Engine (Double Machine Learning / CATE Estimation).
Implements Robinson's Double Machine Learning (DML) R-Learner to estimate
Individual Heterogeneous Treatment Effects (CATE: tau(X)) and optimal budget-constrained interventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold


class DoubleMLCausalEngine:
    """
    Double Machine Learning (DML) for Heterogeneous Treatment Effect (CATE) Estimation.
    Decomposition:
        Y - E[Y|X] = tau(X) * (T - E[T|X]) + epsilon
    where:
        Y = Retention indicator (1 = Retained, 0 = Churned)
        T = Intervention offer (1 = Retention Offer, 0 = Control)
        X = Confounder / customer feature vector
        tau(X) = Conditional Average Treatment Effect (CATE uplift)
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        # Nuisance models (Stage 1)
        self.outcome_model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=42)
        self.propensity_model = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        # Uplift / CATE model (Stage 2)
        self.cate_model = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)

        self.is_fitted = False
        self._fit_default_baseline()

    def _fit_default_baseline(self):
        """Pre-fit on synthetic observational telco data with known heterogeneous treatment effects."""
        np.random.seed(42)
        n = 1200

        # Features: tenure (months), monthly_charges ($), support_calls, contract_is_month_to_month
        tenure = np.random.uniform(1, 72, n)
        charges = np.random.uniform(20, 120, n)
        calls = np.random.poisson(1.5, n)
        is_m2m = np.random.binomial(1, 0.55, n)
        X = np.column_stack([tenure, charges, calls, is_m2m])

        # Propensity score e(X): treatment assignment probability
        propensity_logits = -0.5 + 0.02 * charges + 0.3 * calls - 0.01 * tenure
        propensity = 1.0 / (1.0 + np.exp(-propensity_logits))
        T = np.random.binomial(1, np.clip(propensity, 0.1, 0.9))

        # True CATE tau(X): high-charge, high-call customers on month-to-month contracts have high uplift!
        true_tau = 0.15 + 0.002 * charges + 0.04 * calls * is_m2m - 0.002 * tenure
        true_tau = np.clip(true_tau, -0.05, 0.45)

        # Baseline retention without treatment
        base_retention = 0.85 - 0.003 * charges - 0.08 * calls - 0.15 * is_m2m + 0.004 * tenure
        base_retention = np.clip(base_retention, 0.1, 0.95)

        # Observed retention Y
        retention_prob = np.clip(base_retention + T * true_tau, 0.01, 0.99)
        Y = np.random.binomial(1, retention_prob)

        # Stage 1: Cross-fitting nuisance estimates
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        mu_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr = Y[train_idx]
            T_tr = T[train_idx]

            m_model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=42)
            p_model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)

            m_model.fit(X_tr, Y_tr)
            p_model.fit(X_tr, T_tr)

            mu_hat[val_idx] = m_model.predict(X_val)
            e_hat[val_idx] = p_model.predict_proba(X_val)[:, 1]

        # Stage 2: Orthogonalized CATE regression
        # Y_tilde = Y - mu_hat(X)
        # T_tilde = T - e_hat(X)
        Y_tilde = Y - mu_hat
        T_tilde = T - np.clip(e_hat, 0.05, 0.95)

        # Modified target for R-learner: Y_tilde / T_tilde with sample weights (T_tilde)^2
        sample_weights = T_tilde ** 2
        pseudo_outcomes = Y_tilde / np.where(np.abs(T_tilde) < 1e-4, 1e-4, T_tilde)

        self.cate_model.fit(X, pseudo_outcomes, sample_weight=sample_weights)
        self.is_fitted = True

    def estimate_uplift(self, features: np.ndarray) -> np.ndarray:
        """
        Estimate Individual Treatment Effect tau_i(X) for customer feature vectors.
        Features shape: (N, 4) -> [tenure, monthly_charges, support_calls, is_month_to_month]
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        raw_uplift = self.cate_model.predict(features)
        return np.clip(raw_uplift, -0.15, 0.50)

    def prescribe_intervention(
        self,
        customer_id: str,
        tenure: float,
        monthly_charges: float,
        support_calls: int,
        is_month_to_month: int,
        clv_estimate: float = 850.0
    ) -> Dict[str, Any]:
        """
        Prescribe the optimal marketing/retention action based on causal uplift segmentation.
        Segments:
          - Persuadable (tau > 0.08): Target with retention intervention.
          - Sure Thing (tau in [-0.02, 0.08], base churn low): Do not discount.
          - Lost Cause (tau in [-0.02, 0.08], base churn high): Do not waste budget.
          - Sleeping Dog (tau < -0.02): Do NOT contact! Intervention triggers churn.
        """
        feat = np.array([[tenure, monthly_charges, float(support_calls), float(is_month_to_month)]])
        tau_val = float(self.estimate_uplift(feat)[0])

        # Baseline churn probability heuristic
        base_churn = np.clip(0.55 + 0.003 * monthly_charges + 0.06 * support_calls - 0.007 * tenure, 0.05, 0.95)

        # Action Catalog
        # 1. No Action ($0 cost)
        # 2. Standard Retention ($15 coupon)
        # 3. VIP Concierge Service ($50 dedicated agent package)
        if tau_val < -0.02:
            segment = "SLEEPING_DOG"
            recommended_action = "DO_NOT_DISTURB"
            action_cost = 0.0
            expected_net_value = 0.0
            rationale = "Customer is sensitive to contact. Intervention triggers churn awareness."
        elif tau_val > 0.12:
            segment = "PERSUADABLE_HIGH_VALUE"
            recommended_action = "VIP_CONCIERGE_RETENTION"
            action_cost = 50.0
            expected_net_value = (tau_val * clv_estimate) - action_cost
            rationale = "High responsiveness to personalized outreach. Justifies premium retention tier."
        elif tau_val > 0.05:
            segment = "PERSUADABLE_STANDARD"
            recommended_action = "TARGETED_DISCOUNT_COUPON"
            action_cost = 15.0
            expected_net_value = (tau_val * clv_estimate) - action_cost
            rationale = "Moderate positive responsiveness. Standard discount yields positive ROI."
        elif base_churn > 0.65:
            segment = "LOST_CAUSE"
            recommended_action = "NO_ACTION_LOW_ROI"
            action_cost = 0.0
            expected_net_value = 0.0
            rationale = "High churn risk but low treatment elasticity. Budget better allocated elsewhere."
        else:
            segment = "SURE_THING"
            recommended_action = "ORGANIC_NURTURE"
            action_cost = 0.0
            expected_net_value = 0.0
            rationale = "Customer organically stays. Giving discounts cannibalizes revenue."

        roi = round((expected_net_value / max(1.0, action_cost)), 2) if action_cost > 0 else 0.0

        return {
            "customer_id": customer_id,
            "cate_uplift": round(tau_val, 4),
            "baseline_churn_prob": round(float(base_churn), 3),
            "causal_segment": segment,
            "prescribed_action": recommended_action,
            "action_cost_usd": action_cost,
            "expected_clv_preserved_usd": round(tau_val * clv_estimate, 2) if expected_net_value > 0 else 0.0,
            "expected_net_gain_usd": round(expected_net_value, 2),
            "campaign_roi_multiple": roi,
            "clinical_decision_rationale": rationale,
        }


# Singleton causal engine instance
causal_engine = DoubleMLCausalEngine()
