import pandas as pd

def create_features(df):
    
    # --- Cleaning ---
    df["tenure"] = df["tenure"].astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # --- Tenure Group ---
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48+"]
    )
    
    # --- Service Count ---
    services = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    
    df["service_count"] = (df[services] == "Yes").sum(axis=1)
    
    return df