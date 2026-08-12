"""Reproducible training script for the Telecom Churn & Revenue Intelligence System.

Executes data loading, stratified train/test split, 5-fold cross-validation,
calibrated pipeline fitting, evaluation, and serialization to models/churn_pipeline.pkl.
Also exports model metadata to models/model_metadata.json for production tracking.
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from src.utils import RAW_CSV, MODEL_PATH, setup_logging
from src.modeling import train_pipeline, cross_validate_pipeline, evaluate_pipeline, save_model, split_data
from src.profit_simulation import find_optimal_threshold

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

METADATA_PATH = os.path.join(os.path.dirname(MODEL_PATH), "model_metadata.json")


def main():
    logger.info("=== Telecom Churn Pipeline Training ===")
    
    if not os.path.exists(RAW_CSV):
        logger.error("Raw CSV file not found at: %s", RAW_CSV)
        sys.exit(1)

    logger.info("Loading raw dataset from %s", RAW_CSV)
    df = pd.read_csv(RAW_CSV)

    if "Churn" not in df.columns:
        logger.error("Target column 'Churn' missing from dataset.")
        sys.exit(1)

    # Encode target vector
    y = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # Stratified Train/Test Split
    logger.info("Splitting data (80%% train, 20%% test)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Cross-Validation
    logger.info("Running 5-Fold Stratified Cross-Validation...")
    cv_scores = cross_validate_pipeline(X_train, y_train, n_splits=5, random_state=42)
    logger.info("5-Fold CV ROC-AUC Scores: %s", [round(s, 4) for s in cv_scores])
    logger.info("Mean CV ROC-AUC: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    # Fit Production Pipeline
    logger.info("Fitting unified calibrated stacking ensemble pipeline on training set...")
    pipeline = train_pipeline(X_train, y_train)

    # Evaluate Holdout Test Set
    test_auc, test_probs = evaluate_pipeline(pipeline, X_test, y_test)
    logger.info("Holdout Test ROC-AUC Score: %.4f", test_auc)

    # Optimal Threshold Optimization
    opt_thresh, max_profit = find_optimal_threshold(y_test, test_probs)
    logger.info("Optimal Retention Profit Threshold: %.2f (Max Net Profit: ₹%.2f)", opt_thresh, max_profit)

    # Serialize Pipeline Artifact
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(pipeline, MODEL_PATH)
    logger.info("Successfully exported unified pipeline to %s", MODEL_PATH)

    # Export Model Metadata for Production Tracking
    metadata = {
        "model_version": "3.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_type": "CalibratedStackingEnsemble",
        "base_estimators": ["XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"],
        "meta_learner": "LogisticRegression",
        "calibration_method": "isotonic",
        "dataset": {
            "source": os.path.basename(RAW_CSV),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "churn_rate": round(float(y.mean()), 4),
            "feature_count": X.shape[1],
        },
        "performance": {
            "cv_roc_auc_scores": [round(float(s), 4) for s in cv_scores],
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "holdout_test_roc_auc": round(float(test_auc), 4),
        },
        "business": {
            "optimal_threshold": round(float(opt_thresh), 4),
            "max_net_profit_at_threshold": round(float(max_profit), 2),
            "retention_cost": 500,
            "save_rate": 0.6,
            "annual_revenue_assumption": 6000,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Exported model metadata to %s", METADATA_PATH)

    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
