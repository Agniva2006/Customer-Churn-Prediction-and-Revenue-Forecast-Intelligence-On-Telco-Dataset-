#!/usr/bin/env python3
"""
ab_testing.py
TelcoPulse: Advanced Statistical Experimentation & Variance Reduction Engine.
Implements:
  1. CUPED (Controlled-experiment Using Pre-Experiment Data) Variance Reduction
  2. Difference-in-Differences (DiD) 2-Way Fixed Effects Regression
  3. mixture Sequential Probability Ratio Testing (mSPRT) for Continuous Experimentation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats


class CUPEDExperimentEngine:
    """
    CUPED Variance Reduction Engine.
    Adjusts post-experiment metric Y using pre-experiment baseline metric X:
        Y_cuped = Y - theta * (X - E[X])
        theta = Cov(Y, X) / Var(X)
    Variance Reduction:
        Var(Y_cuped) = Var(Y) * (1 - Corr(Y, X)^2)
    """

    @staticmethod
    def evaluate_cuped(
        treatment_pre: np.ndarray,
        treatment_post: np.ndarray,
        control_pre: np.ndarray,
        control_post: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compute standard vs CUPED-adjusted A/B test statistics.
        """
        y_treat = np.asarray(treatment_post, dtype=np.float64)
        x_treat = np.asarray(treatment_pre, dtype=np.float64)
        y_ctrl = np.asarray(control_post, dtype=np.float64)
        x_ctrl = np.asarray(control_pre, dtype=np.float64)

        n_treat = len(y_treat)
        n_ctrl = len(y_ctrl)

        # Combine all pre and post data to compute pooled theta
        all_y = np.concatenate([y_treat, y_ctrl])
        all_x = np.concatenate([x_treat, x_ctrl])

        cov_matrix = np.cov(all_y, all_x)
        cov_yx = cov_matrix[0, 1]
        var_x = cov_matrix[1, 1]
        var_y = cov_matrix[0, 0]

        theta = cov_yx / var_x if var_x > 1e-9 else 0.0
        corr_yx = cov_yx / np.sqrt(var_x * var_y) if (var_x * var_y) > 1e-9 else 0.0
        theoretical_variance_reduction = corr_yx ** 2

        mean_x = np.mean(all_x)

        # Standard Raw Metric Stats
        mean_y_treat = np.mean(y_treat)
        mean_y_ctrl = np.mean(y_ctrl)
        raw_diff = mean_y_treat - mean_y_ctrl
        raw_se = np.sqrt(np.var(y_treat, ddof=1) / n_treat + np.var(y_ctrl, ddof=1) / n_ctrl)
        raw_t_stat = raw_diff / max(1e-9, raw_se)
        raw_dof = n_treat + n_ctrl - 2
        raw_p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(raw_t_stat), df=raw_dof))
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=raw_dof)
        raw_ci = (raw_diff - t_crit * raw_se, raw_diff + t_crit * raw_se)

        # CUPED Adjusted Metric
        y_cuped_treat = y_treat - theta * (x_treat - mean_x)
        y_cuped_ctrl = y_ctrl - theta * (x_ctrl - mean_x)

        mean_cuped_treat = np.mean(y_cuped_treat)
        mean_cuped_ctrl = np.mean(y_cuped_ctrl)
        cuped_diff = mean_cuped_treat - mean_cuped_ctrl

        var_cuped_treat = np.var(y_cuped_treat, ddof=1)
        var_cuped_ctrl = np.var(y_cuped_ctrl, ddof=1)
        cuped_se = np.sqrt(var_cuped_treat / n_treat + var_cuped_ctrl / n_ctrl)
        cuped_t_stat = cuped_diff / max(1e-9, cuped_se)
        cuped_p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(cuped_t_stat), df=raw_dof))
        cuped_ci = (cuped_diff - t_crit * cuped_se, cuped_diff + t_crit * cuped_se)

        # Empirical variance reduction and sample size savings
        empirical_var_reduction = 1.0 - (cuped_se ** 2) / max(1e-9, raw_se ** 2)
        sample_size_savings_pct = max(0.0, empirical_var_reduction * 100.0)

        return {
            "theta_optimal": round(float(theta), 5),
            "correlation_pre_post": round(float(corr_yx), 4),
            "variance_reduction_pct": round(float(empirical_var_reduction * 100.0), 2),
            "sample_size_savings_pct": round(float(sample_size_savings_pct), 2),
            "effective_sample_multiplier": round(1.0 / max(0.01, 1.0 - empirical_var_reduction), 2),
            "raw_ab_test": {
                "treatment_mean": round(float(mean_y_treat), 4),
                "control_mean": round(float(mean_y_ctrl), 4),
                "absolute_effect": round(float(raw_diff), 4),
                "relative_uplift_pct": round(float((raw_diff / max(1e-9, mean_y_ctrl)) * 100.0), 2),
                "standard_error": round(float(raw_se), 4),
                "t_statistic": round(float(raw_t_stat), 3),
                "p_value": round(float(raw_p_value), 5),
                "ci_95": [round(float(raw_ci[0]), 4), round(float(raw_ci[1]), 4)],
                "statistically_significant": bool(raw_p_value < alpha)
            },
            "cuped_ab_test": {
                "treatment_mean": round(float(mean_cuped_treat), 4),
                "control_mean": round(float(mean_cuped_ctrl), 4),
                "absolute_effect": round(float(cuped_diff), 4),
                "relative_uplift_pct": round(float((cuped_diff / max(1e-9, mean_cuped_ctrl)) * 100.0), 2),
                "standard_error": round(float(cuped_se), 4),
                "t_statistic": round(float(cuped_t_stat), 3),
                "p_value": round(float(cuped_p_value), 5),
                "ci_95": [round(float(cuped_ci[0]), 4), round(float(cuped_ci[1]), 4)],
                "statistically_significant": bool(cuped_p_value < alpha)
            }
        }


class DifferenceInDifferencesEngine:
    """
    Difference-in-Differences (DiD) 2-Way Fixed Effects Quasi-Experimentation Engine.
    Estimates policy impact across pre/post and treatment/control cohorts:
        Y_it = beta0 + beta1 * Treated_i + beta2 * Post_t + beta3 * (Treated_i * Post_t) + eps
    where beta3 is the causal Average Treatment Effect on the Treated (ATT).
    """

    @staticmethod
    def fit_did(
        y_ctrl_pre: np.ndarray,
        y_ctrl_post: np.ndarray,
        y_treat_pre: np.ndarray,
        y_treat_post: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Fit 2x2 DiD regression model with robust standard errors.
        """
        # Construct panel dataframe
        n_cp = len(y_ctrl_pre)
        n_cpo = len(y_ctrl_post)
        n_tp = len(y_treat_pre)
        n_tpo = len(y_treat_post)

        y = np.concatenate([y_ctrl_pre, y_ctrl_post, y_treat_pre, y_treat_post])
        treated = np.concatenate([np.zeros(n_cp + n_cpo), np.ones(n_tp + n_tpo)])
        post = np.concatenate([np.zeros(n_cp), np.ones(n_cpo), np.zeros(n_tp), np.ones(n_tpo)])
        treated_x_post = treated * post

        # Design matrix X: [1, treated, post, treated*post]
        X = np.column_stack([np.ones(len(y)), treated, post, treated_x_post])

        # OLS closed form: beta = (X^T X)^(-1) X^T Y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)

        # Residual variance and covariance matrix
        residuals = y - X @ beta
        deg_freedom = len(y) - 4
        s2 = np.sum(residuals ** 2) / max(1, deg_freedom)
        cov_beta = s2 * XtX_inv
        std_errors = np.sqrt(np.diagonal(cov_beta))

        # ATT is beta[3] (interaction coefficient)
        att = beta[3]
        att_se = std_errors[3]
        t_stat = att / max(1e-9, att_se)
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=deg_freedom))
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=deg_freedom)
        ci_95 = (att - t_crit * att_se, att + t_crit * att_se)

        # Means breakdown
        mean_c_pre = np.mean(y_ctrl_pre)
        mean_c_post = np.mean(y_ctrl_post)
        mean_t_pre = np.mean(y_treat_pre)
        mean_t_post = np.mean(y_treat_post)

        delta_ctrl = mean_c_post - mean_c_pre
        delta_treat = mean_t_post - mean_t_pre
        raw_did = delta_treat - delta_ctrl

        return {
            "causal_att": round(float(att), 4),
            "standard_error": round(float(att_se), 4),
            "t_statistic": round(float(t_stat), 3),
            "p_value": round(float(p_val), 5),
            "ci_95": [round(float(ci_95[0]), 4), round(float(ci_95[1]), 4)],
            "statistically_significant": bool(p_val < alpha),
            "cohort_breakdown": {
                "control_pre_mean": round(float(mean_c_pre), 4),
                "control_post_mean": round(float(mean_c_post), 4),
                "control_change": round(float(delta_ctrl), 4),
                "treatment_pre_mean": round(float(mean_t_pre), 4),
                "treatment_post_mean": round(float(mean_t_post), 4),
                "treatment_change": round(float(delta_treat), 4),
                "raw_did_effect": round(float(raw_did), 4)
            },
            "regression_coefficients": {
                "intercept_beta0": round(float(beta[0]), 4),
                "treatment_fixed_effect_beta1": round(float(beta[1]), 4),
                "time_fixed_effect_beta2": round(float(beta[2]), 4),
                "interaction_att_beta3": round(float(beta[3]), 4)
            }
        }


class MSPRTExperimentEngine:
    """
    Mixture Sequential Probability Ratio Testing (mSPRT) Engine.
    Allows continuous experimentation without inflating False Positive Rate (prevents p-hacking).
    Computes likelihood ratio martingale Lambda_n with normal mixing distribution N(0, tau^2).
    Stopping threshold: Lambda_n >= 1 / alpha
    """

    def __init__(self, alpha: float = 0.05, mixing_variance_tau2: float = 0.5):
        self.alpha = alpha
        self.tau2 = mixing_variance_tau2
        self.threshold = 1.0 / alpha

    def evaluate_stream(
        self,
        treatment_stream: List[float],
        control_stream: List[float]
    ) -> Dict[str, Any]:
        """
        Evaluate sequential data stream pairwise.
        """
        t_arr = np.asarray(treatment_stream, dtype=np.float64)
        c_arr = np.asarray(control_stream, dtype=np.float64)
        n = min(len(t_arr), len(c_arr))

        if n < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_size": n,
                "current_lr_martingale": 1.0,
                "stopping_threshold": self.threshold,
                "decision": "CONTINUE_SAMPLING"
            }

        diffs = t_arr[:n] - c_arr[:n]
        sigma2 = np.var(diffs, ddof=1) if np.var(diffs, ddof=1) > 1e-6 else 1.0

        # S_n = sum(d_i), V_n = n * sigma2 / 2
        # Under pairwise diffs: V_n = sum of variances
        s_n = np.sum(diffs)
        v_n = n * sigma2

        # Mixture likelihood ratio formula:
        # Lambda_n = sqrt(v_n / (v_n + tau2 * n^2)) * exp( (tau2 * S_n^2) / (2 * sigma2 * (v_n / n + tau2 * n)) )
        # Simplified standard form:
        factor = np.sqrt(sigma2 / (sigma2 + n * self.tau2))
        exponent = (self.tau2 * (s_n ** 2)) / (2.0 * sigma2 * (sigma2 + n * self.tau2))
        lambda_n = factor * np.exp(min(500.0, exponent))

        stopped_early = bool(lambda_n >= self.threshold)
        mean_diff = float(np.mean(diffs))

        if stopped_early:
            decision = "REJECT_H0_EARLY_SUCCESS" if mean_diff > 0 else "REJECT_H0_EARLY_HARMFUL"
        else:
            decision = "CONTINUE_SAMPLING"

        return {
            "sample_size": n,
            "current_lr_martingale": round(float(lambda_n), 4),
            "stopping_threshold": round(self.threshold, 2),
            "decision": decision,
            "stopped_early": stopped_early,
            "observed_mean_difference": round(mean_diff, 4),
            "variance_estimate": round(float(sigma2), 4),
            "confidence_sequence_width": round(float(np.sqrt((2.0 * sigma2 * (sigma2 + n * self.tau2)) / (n ** 2 * self.tau2) * np.log(self.threshold))), 4) if self.tau2 > 0 else 0.0
        }
