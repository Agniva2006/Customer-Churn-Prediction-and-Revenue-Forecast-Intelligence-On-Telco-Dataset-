"""Unit tests for pipeline transformers and unified modeling pipeline."""

import pandas as pd
import numpy as np
from src.transformers import TelcoCleanerTransformer, TelcoFeatureTransformer
from src.modeling import build_unified_pipeline


def get_sample_df():
    return pd.DataFrame([
        {
            "customerID": "7590-VHVEG",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": "29.85",
            "Churn": "No",
        },
        {
            "customerID": "5575-GNVDE",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 34,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 56.95,
            "TotalCharges": "1889.5",
            "Churn": "No",
        }
    ])


def test_cleaner_transformer():
    df = get_sample_df()
    cleaner = TelcoCleanerTransformer()
    cleaner.fit(df)
    df_clean = cleaner.transform(df)

    assert "customerID" not in df_clean.columns
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])


def test_feature_transformer():
    df = get_sample_df()
    cleaner = TelcoCleanerTransformer()
    feature_eng = TelcoFeatureTransformer()

    df_clean = cleaner.fit_transform(df)
    df_feat = feature_eng.fit_transform(df_clean)

    assert "tenure_group" in df_feat.columns
    assert "service_count" in df_feat.columns
    assert "avg_revenue_per_month" in df_feat.columns
    assert "contract_risk_score" in df_feat.columns
    assert "high_value_customer" in df_feat.columns
    assert "auto_payment" in df_feat.columns


def test_pipeline_build():
    pipeline = build_unified_pipeline()
    assert pipeline is not None
    assert "cleaner" in pipeline.named_steps
    assert "feature_engineer" in pipeline.named_steps
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
