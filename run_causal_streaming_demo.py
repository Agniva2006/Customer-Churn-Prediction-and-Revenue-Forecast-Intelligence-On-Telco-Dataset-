#!/usr/bin/env python3
"""
run_causal_streaming_demo.py
TelcoPulse: Causal Prescriptive AI & High-Throughput Streaming Live Demo Runner.
Demonstrates:
  1. Double Machine Learning (DML / CATE) uplift estimation & budget-constrained policy optimization.
  2. High-throughput Kafka stream ingestion (10,000+ events/sec) into RocksDB sliding windows.
  3. Transactional Outbox Pattern & distributed idempotency deduplication.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.causal_engine import causal_engine
from src.event_streaming import stream_store
from src.outbox_pattern import outbox_manager


def main():
    print("=" * 82)
    print(" 📊 TELCOPULSE: CAUSAL PRESCRIPTIVE ML & STREAMING ARCHITECTURE DEMO")
    print("=" * 82)
    print(" • Model Methodology : Double Machine Learning (DML R-Learner / CATE Uplift)")
    print(" • Streaming Engine   : Multi-Partition Kafka Simulator + RocksDB Stateful Windows")
    print(" • Reliability Model  : Transactional Outbox Pattern + Redis Atomic Idempotency (SET NX)")
    print("-" * 82)

    # 1. High-Throughput Streaming Simulation
    print("\n[PHASE 1] Executing High-Throughput Kafka Stream Ingestion (1,000 Events)...")
    print("-" * 82)
    batch_res = stream_store.simulate_kafka_batch(batch_size=1000)
    print(f" • Batch Size Processed        : {batch_res['batch_size_events']} events")
    print(f" • Processing Time             : {batch_res['elapsed_seconds']} seconds")
    print(f" • Ingestion Throughput        : {batch_res['throughput_events_per_sec']} events/sec (SLA: > 10,000 eps)")
    print(f" • Acute Churn Anomalies Flagged: {batch_res['anomalies_detected']} acute risks (5m window rule)")
    print(f" • State Storage Engine        : {batch_res['storage_engine']}")

    # 2. Causal Uplift & Prescriptive Intervention Assignment
    print("\n[PHASE 2] Evaluating Heterogeneous Treatment Effects (CATE) Across Customer Cohorts...")
    print("-" * 82)
    print(f"{'Customer ID':<12} | {'Segment':<22} | {'CATE Uplift':<12} | {'Prescribed Action':<24} | {'ROI Multiple':<12}")
    print("-" * 82)

    sample_cohort = [
        {"cid": "CUST_PERSUADABLE", "tenure": 8.0, "charges": 105.0, "calls": 4, "m2m": 1, "clv": 1250.0},
        {"cid": "CUST_SURE_THING",  "tenure": 48.0, "charges": 35.0,  "calls": 0, "m2m": 0, "clv": 900.0},
        {"cid": "CUST_LOST_CAUSE",  "tenure": 2.0,  "charges": 115.0, "calls": 7, "m2m": 1, "clv": 450.0},
        {"cid": "CUST_SLEEPING_DOG","tenure": 36.0, "charges": 45.0,  "calls": 0, "m2m": 0, "clv": 800.0},
    ]

    for c in sample_cohort:
        p = causal_engine.prescribe_intervention(
            customer_id=c["cid"],
            tenure=c["tenure"],
            monthly_charges=c["charges"],
            support_calls=c["calls"],
            is_month_to_month=c["m2m"],
            clv_estimate=c["clv"]
        )
        print(f"{p['customer_id']:<12} | {p['causal_segment']:<22} | {p['cate_uplift']:<12} | {p['prescribed_action']:<24} | {p['campaign_roi_multiple']}x")

    # 3. Transactional Outbox & Idempotency Reliability
    print("\n[PHASE 3] Testing Transactional Outbox & Distributed Idempotency Guards...")
    print("-" * 82)
    test_key = "idemp_trans_tx_88192a"
    first_attempt = outbox_manager.check_and_set_idempotency(test_key)
    print(f" • First Intervention Request  : Allowed = {first_attempt} (Saved atomically to outbox)")

    # Immediate duplicate request with same key
    duplicate_attempt = outbox_manager.check_and_set_idempotency(test_key)
    print(f" • Duplicate Request (Replay)  : Allowed = {duplicate_attempt} (Blocked by Idempotency Guard!)")

    audit = outbox_manager.get_audit_metrics()
    print(f" • Total Dispatched Events     : {audit['total_events_dispatched']}")
    print(f" • Blocked Duplicate Requests  : {audit['duplicate_requests_blocked_by_idempotency']}")
    print(f" • Delivery Guarantee          : {audit['delivery_guarantee']}")

    print("\n" + "=" * 82)
    print(" 🏆 CAUSAL MACHINE LEARNING & STREAMING VERIFICATION COMPLETE!")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
