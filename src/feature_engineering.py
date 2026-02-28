import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Clean target
    df["Churn"] = df["Churn"].str.strip()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def save_processed(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)