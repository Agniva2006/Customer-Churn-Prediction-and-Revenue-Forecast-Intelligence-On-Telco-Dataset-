"""Unit tests for the revenue forecasting module (ARIMA & Monte Carlo)."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import create_monthly_revenue, arima_forecast
from src.profit_simulation import monte_carlo_revenue


def _make_sample_df(n: int = 200) -> pd.DataFrame:
    """Generate a synthetic customer dataframe suitable for forecasting tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "tenure": rng.integers(1, 72, size=n),
        "MonthlyCharges": rng.uniform(20, 100, size=n).round(2),
        "Churn": rng.choice([0, 1], size=n, p=[0.73, 0.27]),
    })


class TestCreateMonthlyRevenue:
    def test_output_columns(self):
        df = _make_sample_df()
        monthly = create_monthly_revenue(df)
        expected_cols = {"Month", "Active_Customers", "Churned", "Avg_Monthly_Charges", "Revenue"}
        assert expected_cols == set(monthly.columns)

    def test_output_has_rows(self):
        df = _make_sample_df()
        monthly = create_monthly_revenue(df)
        assert len(monthly) > 0

    def test_revenue_is_positive(self):
        df = _make_sample_df()
        monthly = create_monthly_revenue(df)
        assert (monthly["Revenue"] >= 0).all()


class TestArimaForecast:
    def test_forecast_returns_correct_steps(self):
        series = pd.Series([100, 120, 130, 110, 140, 150, 160, 170, 180, 190, 200, 210])
        steps = 4
        predicted, ci = arima_forecast(series, steps=steps)
        assert len(predicted) == steps
        assert len(ci) == steps

    def test_confidence_interval_columns(self):
        series = pd.Series([100, 120, 130, 110, 140, 150, 160, 170, 180, 190, 200, 210])
        _, ci = arima_forecast(series, steps=3)
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    def test_lower_bound_less_than_upper(self):
        series = pd.Series([100, 120, 130, 110, 140, 150, 160, 170, 180, 190, 200, 210])
        _, ci = arima_forecast(series, steps=3)
        assert (ci["lower"] <= ci["upper"]).all()


class TestMonteCarloRevenue:
    def test_output_shape(self):
        result = monte_carlo_revenue(
            n_customers=1000, avg_revenue=6000,
            churn_rate_mean=0.27, churn_rate_std=0.05,
            n_simulations=500
        )
        assert len(result) == 500

    def test_all_positive_revenue(self):
        result = monte_carlo_revenue(
            n_customers=1000, avg_revenue=6000,
            churn_rate_mean=0.27, churn_rate_std=0.05,
            n_simulations=500
        )
        assert (result >= 0).all()

    def test_mean_revenue_reasonable(self):
        result = monte_carlo_revenue(
            n_customers=1000, avg_revenue=6000,
            churn_rate_mean=0.27, churn_rate_std=0.05,
            n_simulations=5000
        )
        # Expected ~ 1000 * 6000 * (1 - 0.27) = 4,380,000
        mean_rev = result.mean()
        assert 3_500_000 < mean_rev < 5_500_000
