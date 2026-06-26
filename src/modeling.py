"""Model training, evaluation, cross-validation, and persistence."""

from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Default hyperparameters (tuned via RandomizedSearchCV in notebooks)
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}


def split_data(
    X, y, test_size: float = 0.2, random_state: int = 42
) -> Tuple:
    """Stratified train/test split.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Binary target vector.
    test_size : float
        Proportion held out for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_xgb(
    X_train, y_train, params: Optional[Dict[str, Any]] = None
) -> XGBClassifier:
    """Train an XGBoost classifier.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data.
    params : dict, optional
        Hyperparameters; falls back to ``DEFAULT_PARAMS``.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    final_params = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBClassifier(**final_params)
    model.fit(X_train, y_train)
    return model


def cross_validate(
    X, y, params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5, random_state: int = 42
) -> np.ndarray:
    """Stratified K-Fold cross-validation reporting ROC-AUC per fold.

    Parameters
    ----------
    X, y : array-like
        Full dataset (not yet split).
    params : dict, optional
        Hyperparameters for XGBoost.
    n_splits : int
        Number of folds (default 5).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    np.ndarray
        Array of ROC-AUC scores, one per fold.
    """
    final_params = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBClassifier(**final_params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return scores


def evaluate(model, X_test, y_test) -> Tuple[float, np.ndarray]:
    """Evaluate a trained model and return ROC-AUC + predicted probabilities."""
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs), probs


def save_model(model, path: str) -> None:
    """Serialize a model to disk."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load a serialized model from disk.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Please train the model first or check the path."
        )