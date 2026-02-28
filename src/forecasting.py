import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def create_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = pd.cut(df["tenure"], bins=12, labels=False)

    monthly = df.groupby("Month").agg({
        "customerID": "count",
        "Churn": "sum",
        "MonthlyCharges": "mean"
    }).reset_index()

    monthly.columns = ["Month", "Active_Customers", "Churned", "Avg_Monthly_Charges"]
    monthly["Revenue"] = monthly["Active_Customers"] * monthly["Avg_Monthly_Charges"]

    return monthly


def arima_forecast(revenue_series, steps=6):
    model = ARIMA(revenue_series, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(steps=steps)
    return forecast