"""Model pipeline construction, training, probability calibration, cross-validation, and serialization."""

import logging
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.transformers import TelcoCleanerTransformer, TelcoFeatureTransformer

logger = logging.getLogger(__name__)

# Default XGBoost Hyperparameters
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "service_count", "avg_revenue_per_month",
    "contract_risk_score", "high_value_customer",
    "auto_payment", "monthly_to_total_ratio"
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"
]


def build_unified_pipeline(params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline incorporating cleaning,
    feature engineering, column preprocessing, and calibrated XGBoost classification.
    """
    xgb_params = {**DEFAULT_PARAMS, **(params or {})}
    base_xgb = XGBClassifier(**xgb_params)
    calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method="isotonic", cv=3)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("cleaner", TelcoCleanerTransformer(drop_customer_id=True)),
            ("feature_engineer", TelcoFeatureTransformer()),
            ("preprocessor", preprocessor),
            ("classifier", calibrated_clf),
        ]
    )

    return pipeline


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple:
    """Stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """Train the full unified pipeline on raw training data."""
    pipeline = build_unified_pipeline(params)
    pipeline.fit(X_train, y_train)
    logger.info("Fitted unified calibrated pipeline successfully.")
    return pipeline


def cross_validate_pipeline(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> np.ndarray:
    """Perform 5-Fold Stratified Cross-Validation on the full pipeline."""
    pipeline = build_unified_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return scores


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, np.ndarray]:
    """Evaluate pipeline predictions and return ROC-AUC score + probabilities."""
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return float(auc), probs


def save_model(model: Any, path: str) -> None:
    """Serialize the pipeline model to disk."""
    joblib.dump(model, path)
    logger.info("Saved pipeline to %s", path)


def load_model(path: str) -> Any:
    """Load a serialized pipeline model from disk."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run 'python train.py' to generate the production pipeline."
        )