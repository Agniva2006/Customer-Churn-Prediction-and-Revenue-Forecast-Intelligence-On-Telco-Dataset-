"""Reproducible training script for the Telecom Churn & Revenue Intelligence System.

Executes data loading, stratified train/test split, 5-fold cross-validation,
calibrated pipeline fitting, evaluation, and serialization to models/churn_pipeline.pkl.
"""

import os
import sys
import logging
import pandas as pd

from src.utils import RAW_CSV, MODEL_PATH, setup_logging
from src.modeling import train_pipeline, cross_validate_pipeline, evaluate_pipeline, save_model, split_data
from src.profit_simulation import find_optimal_threshold

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Cross-Validation
    logger.info("Running 5-Fold Stratified Cross-Validation...")
    cv_scores = cross_validate_pipeline(X_train, y_train, n_splits=5, random_state=42)
    logger.info("5-Fold CV ROC-AUC Scores: %s", [round(s, 4) for s in cv_scores])
    logger.info("Mean CV ROC-AUC: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    # Fit Production Pipeline
    logger.info("Fitting unified calibrated XGBoost pipeline on training set...")
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
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
