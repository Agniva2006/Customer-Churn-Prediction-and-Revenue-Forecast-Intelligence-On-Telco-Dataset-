"""Unit tests for the new Stacking Ensemble classifier and SQLite prediction logging audit pipeline."""

import os
import sqlite3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

from src.modeling import build_unified_pipeline
from src.database import init_db, log_prediction, get_recent_predictions, DB_PATH

def test_pipeline_uses_stacking_classifier():
    """Verify that build_unified_pipeline includes a calibrated StackingClassifier."""
    pipeline = build_unified_pipeline()
    assert isinstance(pipeline, Pipeline)
    
    clf = pipeline.named_steps["classifier"]
    assert isinstance(clf, CalibratedClassifierCV)
    
    base_estimator = clf.estimator
    assert isinstance(base_estimator, StackingClassifier)
    
    # Check that XGB, RF, and GB are base estimators
    estimators_dict = dict(base_estimator.estimators)
    assert "xgb" in estimators_dict
    assert "rf" in estimators_dict
    assert "gb" in estimators_dict

def test_database_logging_and_retrieval(tmp_path):
    """Test SQLite database logging saves features and retrieves logs correctly."""
    # Use a test database path inside a temp directory
    test_db_dir = tmp_path / "logs"
    test_db_dir.mkdir()
    test_db_path = test_db_dir / "predictions.db"
    
    # Patch database path variables for isolation
    import src.database
    original_db_path = src.database.DB_PATH
    original_db_dir = src.database.DB_DIR
    src.database.DB_PATH = test_db_path
    src.database.DB_DIR = test_db_dir
    
    try:
        # Initialize
        init_db()
        assert test_db_path.exists()
        
        # Log a prediction record
        log_prediction(
            monthly_charges=75.5,
            total_charges=900.0,
            tenure=12,
            contract="One year",
            risk_probability=0.25,
            risk_level="medium",
            expected_profit=150.0,
            clv=3500.0,
            action_quadrant="High CLV - Low Churn (Proactive)"
        )
        
        # Retrieve logs
        df = get_recent_predictions()
        assert len(df) == 1
        assert df.iloc[0]["monthly_charges"] == 75.5
        assert df.iloc[0]["total_charges"] == 900.0
        assert df.iloc[0]["tenure"] == 12
        assert df.iloc[0]["contract"] == "One year"
        assert df.iloc[0]["risk_probability"] == 0.25
        assert df.iloc[0]["risk_level"] == "medium"
        assert df.iloc[0]["expected_profit"] == 150.0
        assert df.iloc[0]["clv"] == 3500.0
        assert df.iloc[0]["action_quadrant"] == "High CLV - Low Churn (Proactive)"
        
    finally:
        # Restore original paths
        src.database.DB_PATH = original_db_path
        src.database.DB_DIR = original_db_dir
