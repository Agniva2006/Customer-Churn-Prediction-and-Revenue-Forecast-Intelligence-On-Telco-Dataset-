#!/usr/bin/env python3
"""
test_telcopulse_v2.py
Automated Unit and Integration Tests for TelcoPulse v2 Enhancements:
- Industrial Causal Engine (CATE 95% CIs, Asymptotic Inference, Knapsack Allocation)
- A/B Testing & Variance Reduction (CUPED, DiD, mSPRT)
- Feature Store & Caching Layer
- FastAPI Endpoints
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal_engine_v2 import industrial_causal_engine
from src.ab_testing import CUPEDExperimentEngine, DifferenceInDifferencesEngine, MSPRTExperimentEngine
from src.feature_store import feature_store
from fastapi.testclient import TestClient
from api import app


class TestTelcoPulseV2(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_causal_engine_v2_ci(self):
        """Test CATE uplift inference with 95% confidence bounds."""
        features = np.array([[24.0, 75.0, 2.0, 1.0], [60.0, 30.0, 0.0, 0.0]])
        res = industrial_causal_engine.predict_uplift_with_ci(features)
        
        self.assertIn("cate_uplift", res)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertIn("std_error", res)
        self.assertIn("p_value", res)
        self.assertEqual(len(res["cate_uplift"]), 2)
        # Verify lower bound <= uplift <= upper bound
        for i in range(2):
            self.assertLessEqual(res["ci_lower"][i], res["cate_uplift"][i] + 1e-4)
            self.assertGreaterEqual(res["ci_upper"][i], res["cate_uplift"][i] - 1e-4)

    def test_prescribe_intervention_v2(self):
        """Test individual prescription with 4-quadrant policy."""
        rx = industrial_causal_engine.prescribe_intervention_v2(
            customer_id="CUST_TEST_01",
            tenure=6.0,
            monthly_charges=95.0,
            support_calls=4,
            is_month_to_month=1
        )
        self.assertEqual(rx["customer_id"], "CUST_TEST_01")
        self.assertIn("ci_95", rx)
        self.assertIn(rx["causal_segment"], [
            "PERSUADABLE_HIGH_VALUE", "PERSUADABLE_STANDARD", "SURE_THING", "LOST_CAUSE", "SLEEPING_DOG"
        ])
        self.assertGreaterEqual(rx["campaign_roi_multiple"], 0.0)

    def test_knapsack_budget_allocation(self):
        """Test portfolio budget knapsack optimization."""
        batch = [
            {"customer_id": f"C_{i}", "tenure": 5 + i * 2, "monthly_charges": 50 + i * 5, "support_calls": i % 5, "is_month_to_month": 1}
            for i in range(15)
        ]
        alloc = industrial_causal_engine.allocate_campaign_budget(batch, total_budget_usd=200.0)
        self.assertLessEqual(alloc["spent_budget_usd"], 200.0)
        self.assertGreaterEqual(alloc["interventions_funded"], 1)
        self.assertGreaterEqual(alloc["portfolio_roi_multiple"], 0.0)

    def test_cuped_variance_reduction(self):
        """Test CUPED variance reduction mathematics."""
        np.random.seed(42)
        n = 1000
        x = np.random.normal(50, 10, n)
        # Strong correlation between pre and post
        y_c = x + np.random.normal(0, 3, n)
        y_t = x + 3.0 + np.random.normal(0, 3, n)

        cuped_res = CUPEDExperimentEngine.evaluate_cuped(x[:500], y_t[:500], x[500:], y_c[500:])
        self.assertGreater(cuped_res["variance_reduction_pct"], 50.0)
        self.assertTrue(cuped_res["cuped_ab_test"]["statistically_significant"])
        self.assertLess(cuped_res["cuped_ab_test"]["standard_error"], cuped_res["raw_ab_test"]["standard_error"])

    def test_did_quasi_experiment(self):
        """Test Difference-in-Differences 2-Way Fixed Effects."""
        np.random.seed(42)
        n = 500
        c_pre = np.random.normal(40, 5, n)
        c_post = c_pre + 2.0 + np.random.normal(0, 2, n)
        t_pre = np.random.normal(42, 5, n)
        t_post = t_pre + 2.0 + 5.0 + np.random.normal(0, 2, n) # true ATT = +5.0

        did_res = DifferenceInDifferencesEngine.fit_did(c_pre, c_post, t_pre, t_post)
        self.assertAlmostEqual(did_res["causal_att"], 5.0, delta=0.5)
        self.assertTrue(did_res["statistically_significant"])

    def test_msprt_sequential_testing(self):
        """Test mSPRT continuous experimentation."""
        msprt = MSPRTExperimentEngine(alpha=0.05)
        c_stream = np.random.normal(0, 1, 300).tolist()
        t_stream = (np.random.normal(0.4, 1, 300)).tolist() # strong effect
        res = msprt.evaluate_stream(t_stream, c_stream)
        self.assertIn(res["decision"], ["REJECT_H0_EARLY_SUCCESS", "CONTINUE_SAMPLING"])

    def test_feature_store_caching(self):
        """Test Feature Store set and get with TTL."""
        feature_store.cache_cate_prescription("CUST_CACHE_TEST", {"action": "DISCOUNT", "uplift": 0.15}, ttl=10)
        cached = feature_store.get_cached_cate_prescription("CUST_CACHE_TEST")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["action"], "DISCOUNT")

    def test_api_endpoints(self):
        """Test FastAPI endpoints for Causal V2, Experiments, and Metrics."""
        # 1. Causal Prescribe V2
        resp = self.client.post("/causal/prescribe-v2", json={
            "customer_id": "API_TEST_CUST",
            "tenure": 10.0,
            "monthly_charges": 80.0,
            "support_calls": 3,
            "is_month_to_month": 1,
            "clv_estimate": 900.0
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("ci_95", data["prescription"])

        # 2. Prometheus Metrics
        metrics_resp = self.client.get("/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        self.assertIn("telcopulse_events_total", metrics_resp.text)
        self.assertIn("telcopulse_p99_latency_ms", metrics_resp.text)


if __name__ == "__main__":
    unittest.main()
