"""
Telecom Customer Churn & Revenue Intelligence — Source Package
==============================================================
Reusable modules for data processing, feature engineering, transformers,
modeling, explainability, profit simulation, drift monitoring, forecasting, and utilities.
"""

from .data_processing import load_data, clean_data, save_processed
from .feature_engineering import create_features
from .transformers import TelcoCleanerTransformer, TelcoFeatureTransformer
from .modeling import (
    build_unified_pipeline, split_data, train_pipeline,
    cross_validate_pipeline, evaluate_pipeline, save_model, load_model
)
from .explainability import get_top_churn_drivers
from .drift import calculate_psi, evaluate_ks_drift, evaluate_batch_drift
from .profit_simulation import (
    compute_individualized_profit, simulate_profit,
    find_optimal_threshold, monte_carlo_revenue
)
from .forecasting import create_monthly_revenue, arima_forecast
from .utils import plot_feature_importance, setup_logging

__all__ = [
    "load_data", "clean_data", "save_processed",
    "create_features",
    "TelcoCleanerTransformer", "TelcoFeatureTransformer",
    "build_unified_pipeline", "split_data", "train_pipeline",
    "cross_validate_pipeline", "evaluate_pipeline", "save_model", "load_model",
    "get_top_churn_drivers",
    "calculate_psi", "evaluate_ks_drift", "evaluate_batch_drift",
    "compute_individualized_profit", "simulate_profit",
    "find_optimal_threshold", "monte_carlo_revenue",
    "create_monthly_revenue", "arima_forecast",
    "plot_feature_importance", "setup_logging",
]
