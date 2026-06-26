"""ARIMA-based revenue forecasting with confidence intervals."""

from typing import Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def create_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer data into monthly revenue buckets.

    Uses ``tenure`` as a proxy for customer lifetime months and bins
    them into 12 groups.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing at least ``tenure``, ``customerID``,
        ``Churn``, and ``MonthlyCharges``.

    Returns
    -------
    pd.DataFrame
        Columns: Month, Active_Customers, Churned, Avg_Monthly_Charges, Revenue.
    """
    df = df.copy()
    df["Month"] = pd.cut(df["tenure"], bins=12, labels=False)

    monthly = df.groupby("Month").agg({
        "customerID": "count",
        "Churn": "sum",
        "MonthlyCharges": "mean",
    }).reset_index()

    monthly.columns = ["Month", "Active_Customers", "Churned", "Avg_Monthly_Charges"]
    monthly["Revenue"] = monthly["Active_Customers"] * monthly["Avg_Monthly_Charges"]

    return monthly


def arima_forecast(
    revenue_series: pd.Series,
    steps: int = 6,
    order: Tuple[int, int, int] = (1, 1, 1),
    conf_level: float = 0.05,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Fit an ARIMA model and forecast future revenue.

    Parameters
    ----------
    revenue_series : pd.Series
        Historical revenue values.
    steps : int
        Number of periods to forecast.
    order : tuple
        ARIMA (p, d, q) order.
    conf_level : float
        Significance level for confidence intervals (default 0.05 → 95 %).

    Returns
    -------
    tuple
        (forecast_values, confidence_intervals_df)
        The CI dataframe has columns ``lower`` and ``upper``.
    """
    model = ARIMA(revenue_series, order=order)
    fit = model.fit()

    forecast = fit.get_forecast(steps=steps)
    predicted = forecast.predicted_mean
    ci = forecast.conf_int(alpha=conf_level)
    ci.columns = ["lower", "upper"]

    return predicted, ci