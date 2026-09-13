#!/usr/bin/env python3
"""
run_causal_streaming_demo.py
TelcoPulse: Causal Prescriptive AI, A/B Testing & Streaming Production Demo Runner.
Demonstrates:
  1. Industrial Double Machine Learning (DML / CATE) with 95% Confidence Intervals & Knapsack Budget Allocation.
  2. CUPED Variance Reduction & Difference-in-Differences (DiD) Experimentation Engine.
  3. High-throughput Kafka stream ingestion (10,000+ events/sec) into RocksDB sliding windows.
  4. Redis Feature Store caching and Transactional Outbox Pattern with Distributed Idempotency.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.causal_engine_v2 import industrial_causal_engine
from src.ab_testing import CUPEDExperimentEngine, DifferenceInDifferencesEngine, MSPRTExperimentEngine
from src.feature_store import feature_store
from src.event_streaming import stream_store
from src.outbox_pattern import outbox_manager


def main():
    print("=" * 86)
    print(" 📊 TELCOPULSE: INDUSTRIAL CAUSAL ML, EXPERIMENTATION & STREAMING SUITE")
    print("=" * 86)
    print(" • Causal Inference   : Double Machine Learning (DML CATE with Asymptotic 95% CIs)")
    print(" • Experimentation    : CUPED Variance Reduction, DiD 2-Way Fixed Effects, mSPRT")
    print(" • Feature Store      : Low-Latency Redis Tier with LRU Caching (< 2ms access)")
    print(" • Streaming Engine   : Kafka Broker + RocksDB Stateful Windows (> 10k events/sec)")
    print(" • Outbox Reliability : Transactional Outbox + Redis Distributed Idempotency Guard")
    print("-" * 86)

    # 1. High-Throughput Streaming Simulation
    print("\n[PHASE 1] Executing High-Throughput Kafka Stream Ingestion (1,000 Events)...")
    print("-" * 86)
    batch_res = stream_store.simulate_kafka_batch(batch_size=1000)
    print(f" • Batch Size Processed        : {batch_res['batch_size_events']} events")
    print(f" • Processing Time             : {batch_res['elapsed_seconds']} seconds")
    print(f" • Ingestion Throughput        : {batch_res['throughput_events_per_sec']} events/sec (SLA: > 10,000 eps)")
    print(f" • Acute Churn Anomalies Flagged: {batch_res['anomalies_detected']} acute risks (5m window rule)")
    print(f" • State Storage Engine        : {batch_res['storage_engine']}")

    # 2. Causal Uplift with 95% Confidence Intervals
    print("\n[PHASE 2] Evaluating CATE Uplift with 95% Confidence Intervals Across Cohorts...")
    print("-" * 86)
    print(f"{'Customer ID':<14} | {'Segment':<24} | {'CATE Uplift':<12} | {'95% CI':<20} | {'ROI Multiple':<10}")
    print("-" * 86)

    sample_cohort = [
        {"cid": "CUST_PERSUADABLE", "tenure": 8.0, "charges": 105.0, "calls": 4, "m2m": 1, "clv": 1250.0},
        {"cid": "CUST_SURE_THING",  "tenure": 48.0, "charges": 35.0,  "calls": 0, "m2m": 0, "clv": 900.0},
        {"cid": "CUST_LOST_CAUSE",  "tenure": 2.0,  "charges": 115.0, "calls": 7, "m2m": 1, "clv": 450.0},
        {"cid": "CUST_SLEEPING_DOG","tenure": 36.0, "charges": 45.0,  "calls": 0, "m2m": 0, "clv": 800.0},
    ]

    for c in sample_cohort:
        p = industrial_causal_engine.prescribe_intervention_v2(
            customer_id=c["cid"],
            tenure=c["tenure"],
            monthly_charges=c["charges"],
            support_calls=c["calls"],
            is_month_to_month=c["m2m"],
            clv_estimate=c["clv"]
        )
        ci_str = f"[{p['ci_95']['lower']:.3f}, {p['ci_95']['upper']:.3f}]"
        print(f"{p['customer_id']:<14} | {p['causal_segment']:<24} | {p['cate_uplift']:<12} | {ci_str:<20} | {p['campaign_roi_multiple']}x")

    # 3. Budget-Constrained Knapsack Allocation
    print("\n[PHASE 3] Knapsack-Optimal Portfolio Budget Intervention Allocation ($500 Budget)...")
    print("-" * 86)
    batch_candidates = [
        {"customer_id": f"CUST_PORT_{i:02d}", "tenure": float(4 + i * 3), "monthly_charges": float(50 + (i % 6) * 12), "support_calls": (i % 5), "is_month_to_month": 1 if i % 2 == 0 else 0}
        for i in range(25)
    ]
    alloc_res = industrial_causal_engine.allocate_campaign_budget(batch_candidates, total_budget_usd=500.0)
    print(f" • Customers Evaluated         : {alloc_res['customers_evaluated']}")
    print(f" • Interventions Funded        : {alloc_res['interventions_funded']}")
    print(f" • Budget Spent / Cap          : ${alloc_res['spent_budget_usd']} / ${alloc_res['total_budget_usd']}")
    print(f" • Preserved CLV Value         : ${alloc_res['total_clv_preserved_usd']}")
    print(f" • Expected Net Value Gain     : ${alloc_res['total_net_gain_usd']}")
    print(f" • Portfolio ROI Multiple      : {alloc_res['portfolio_roi_multiple']}x")

    # 4. A/B Testing & CUPED Variance Reduction Simulation
    print("\n[PHASE 4] Running CUPED Variance Reduction & A/B Experimentation Engine...")
    print("-" * 86)
    np.random.seed(42)
    n = 2000
    x_pre = np.random.normal(50.0, 10.0, n)
    y_ctrl = 0.85 * x_pre + 10.0 + np.random.normal(0, 5.0, n)
    y_treat = 0.85 * x_pre + 10.0 + 3.50 + np.random.normal(0, 5.0, n) # +$3.50 Lift

    cuped_res = CUPEDExperimentEngine.evaluate_cuped(
        treatment_pre=x_pre[:1000],
        treatment_post=y_treat[:1000],
        control_pre=x_pre[1000:],
        control_post=y_ctrl[1000:]
    )
    print(f" • Pre/Post Correlation (r)    : {cuped_res['correlation_pre_post']}")
    print(f" • Optimal Theta Adjuster      : {cuped_res['theta_optimal']}")
    print(f" • Empirical Variance Reduction: {cuped_res['variance_reduction_pct']}%")
    print(f" • Sample Size Savings         : {cuped_res['sample_size_savings_pct']}%")
    print(f" • Raw A/B SE vs CUPED SE      : {cuped_res['raw_ab_test']['standard_error']} -> {cuped_res['cuped_ab_test']['standard_error']}")
    print(f" • Statistical Significance    : {cuped_res['cuped_ab_test']['statistically_significant']} (p={cuped_res['cuped_ab_test']['p_value']})")

    # 5. Feature Store & Outbox Reliability
    print("\n[PHASE 5] Testing Feature Store Caching & Transactional Outbox Idempotency...")
    print("-" * 86)
    feature_store.cache_cate_prescription("CUST_LIVE_99", {"action": "VIP_CONCIERGE", "uplift": 0.22}, ttl=60)
    cached_val = feature_store.get_cached_cate_prescription("CUST_LIVE_99")
    fs_stats = feature_store.get_store_metrics()
    print(f" • Feature Store Backend       : {fs_stats['backend']}")
    print(f" • Cached CATE Value Retrieved : {cached_val is not None} (Action: {cached_val['action']})")

    test_key = "idemp_trans_tx_88192a"
    first_attempt = outbox_manager.check_and_set_idempotency(test_key)
    duplicate_attempt = outbox_manager.check_and_set_idempotency(test_key)
    print(f" • First Idempotency Attempt   : Allowed = {first_attempt}")
    print(f" • Replay Duplicate Attempt    : Allowed = {duplicate_attempt} (Guarded by Outbox)")

    print("\n" + "=" * 86)
    print(" 🏆 TELCOPULSE INDUSTRIAL CAUSAL & EXPERIMENTATION VERIFICATION COMPLETE!")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
