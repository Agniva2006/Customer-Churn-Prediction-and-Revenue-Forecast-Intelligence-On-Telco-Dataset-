"""Custom scikit-learn transformers for Telco Churn cleaning and feature engineering.

Provides leak-free, reproducible preprocessing suitable for encapsulation in sklearn Pipelines.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelcoCleanerTransformer(BaseEstimator, TransformerMixin):
    """Transformer for raw Telco data cleaning.

    Applies median imputation to ``TotalCharges``, strips whitespace,
    and drops non-predictive identifier columns like ``customerID``.
    """

    def __init__(self, drop_customer_id: bool = True):
        self.drop_customer_id = drop_customer_id
        self.median_total_charges_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        if "TotalCharges" in X_df.columns:
            coerced = pd.to_numeric(X_df["TotalCharges"], errors="coerce")
            self.median_total_charges_ = float(coerced.median())
        else:
            self.median_total_charges_ = 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()

        # Drop ID if present and requested
        if self.drop_customer_id and "customerID" in X_df.columns:
            X_df = X_df.drop(columns=["customerID"])

        # Coerce TotalCharges and fill missing using fitted median
        if "TotalCharges" in X_df.columns:
            X_df["TotalCharges"] = pd.to_numeric(X_df["TotalCharges"], errors="coerce")
            fill_val = self.median_total_charges_ if self.median_total_charges_ is not None else 0.0
            X_df["TotalCharges"] = X_df["TotalCharges"].fillna(fill_val)

        # Map Churn target if present in X (for safety during dataset transform)
        if "Churn" in X_df.columns and X_df["Churn"].dtype == object:
            X_df["Churn"] = X_df["Churn"].str.strip().map({"Yes": 1, "No": 0})

        return X_df


class TelcoFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transformer for domain-specific feature engineering.

    Creates domain-driven features:
    * ``tenure_group``: lifecycle segmentation
    * ``service_count``: product stickiness proxy
    * ``avg_revenue_per_month``: monthly financial norm
    * ``contract_risk_score``: contract stability index
    * ``high_value_customer``: 75th percentile monthly charge flag
    * ``auto_payment``: automatic payment method flag
    * ``monthly_to_total_ratio``: tenure intensity metric
    """

    def __init__(self, monthly_charge_quantile: float = 0.75):
        self.monthly_charge_quantile = monthly_charge_quantile
        self.high_value_threshold_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        if "MonthlyCharges" in X.columns:
            self.high_value_threshold_ = float(X["MonthlyCharges"].quantile(self.monthly_charge_quantile))
        else:
            self.high_value_threshold_ = 70.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Ensure numeric tenure
        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=["0-12", "12-24", "24-48", "48+"],
            ).astype(str)
        else:
            df["tenure_group"] = "0-12"

        # Service count
        service_cols = [
            "PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies",
        ]
        present_service_cols = [col for col in service_cols if col in df.columns]
        if present_service_cols:
            df["service_count"] = (df[present_service_cols] == "Yes").sum(axis=1)
        else:
            df["service_count"] = 0

        # Avg revenue per month & ratio
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
            df["avg_revenue_per_month"] = df["TotalCharges"] / np.maximum(df["tenure"], 1)
            df["monthly_to_total_ratio"] = df["MonthlyCharges"] / np.maximum(df["TotalCharges"], 1.0)
        else:
            df["avg_revenue_per_month"] = df.get("MonthlyCharges", 0)
            df["monthly_to_total_ratio"] = 0.0

        # Contract risk score
        contract_risk_map = {"Month-to-month": 3, "One year": 2, "Two year": 1}
        if "Contract" in df.columns:
            df["contract_risk_score"] = df["Contract"].map(contract_risk_map).fillna(0).astype(int)
        else:
            df["contract_risk_score"] = 0

        # High-value customer
        thresh = self.high_value_threshold_ if self.high_value_threshold_ is not None else 70.0
        if "MonthlyCharges" in df.columns:
            df["high_value_customer"] = (df["MonthlyCharges"] >= thresh).astype(int)
        else:
            df["high_value_customer"] = 0

        # Auto payment
        auto_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
        if "PaymentMethod" in df.columns:
            df["auto_payment"] = df["PaymentMethod"].isin(auto_methods).astype(int)
        else:
            df["auto_payment"] = 0

        return df
