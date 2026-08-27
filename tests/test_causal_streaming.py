"""Unit tests for 0.01% tier Causal Prescriptive Engine, Event Streaming, and Transactional Outbox."""

import pytest
from src.causal_engine import causal_engine
from src.event_streaming import stream_store
from src.outbox_pattern import outbox_manager


def test_causal_uplift_estimation():
    res = causal_engine.prescribe_intervention(
        customer_id="CUST_TEST_01",
        tenure=10.0,
        monthly_charges=90.0,
        support_calls=3,
        is_month_to_month=1,
        clv_estimate=1000.0
    )
    assert "cate_uplift" in res
    assert "causal_segment" in res
    assert "prescribed_action" in res
    assert res["action_cost_usd"] >= 0.0


def test_streaming_ingest_and_windows():
    batch_res = stream_store.simulate_kafka_batch(batch_size=500)
    assert batch_res["batch_size_events"] == 500
    assert batch_res["throughput_events_per_sec"] > 5000.0

    metrics = stream_store.get_metrics()
    assert metrics["status"] == "STREAMING_ACTIVE"
    assert metrics["total_events_ingested"] >= 500


def test_outbox_idempotency():
    key = "test_key_unique_99"
    first = outbox_manager.check_and_set_idempotency(key)
    second = outbox_manager.check_and_set_idempotency(key)

    assert first is True
    assert second is False  # Duplicate blocked!


def test_transactional_outbox_write():
    record = outbox_manager.write_transactional_outbox(
        aggregate_id="CUST_TEST_OUTBOX",
        event_type="TEST_EVENT",
        payload={"data": 123}
    )
    assert record["status"] == "DISPATCHED"
    assert record["aggregate_id"] == "CUST_TEST_OUTBOX"
