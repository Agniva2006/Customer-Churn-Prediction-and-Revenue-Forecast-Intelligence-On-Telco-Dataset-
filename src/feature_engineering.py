"""Business-driven feature engineering for churn prediction.

Creates six domain-specific features that capture customer lifecycle,
service dependency, financial value, contract risk, and payment stability.
"""

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataframe with engineered features.

    Features created
    ----------------
    * ``tenure_group`` — lifecycle segment (0-12, 12-24, 24-48, 48+)
    * ``service_count`` — number of optional services subscribed
    * ``avg_revenue_per_month`` — TotalCharges / tenure (guarded against div-by-zero)
    * ``contract_risk_score`` — ordinal risk by contract type
    * ``high_value_customer`` — 1 if MonthlyCharges ≥ 75th percentile
    * ``auto_payment`` — 1 if payment method is automatic

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from the data-processing stage.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns appended.
    """
    df = df.copy()

    # --- Tenure group ---
    df["tenure"] = df["tenure"].astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48+"],
    )

    # --- Service count ---
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)

    # --- Average revenue per month ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["avg_revenue_per_month"] = df["TotalCharges"] / np.maximum(df["tenure"], 1)

    # --- Contract risk score (higher = riskier) ---
    contract_risk_map = {
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1,
    }
    df["contract_risk_score"] = df["Contract"].map(contract_risk_map).fillna(0).astype(int)

    # --- High-value customer flag ---
    threshold_75 = df["MonthlyCharges"].quantile(0.75)
    df["high_value_customer"] = (df["MonthlyCharges"] >= threshold_75).astype(int)

    # --- Auto-payment flag ---
    auto_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
    df["auto_payment"] = df["PaymentMethod"].isin(auto_methods).astype(int)

    return df