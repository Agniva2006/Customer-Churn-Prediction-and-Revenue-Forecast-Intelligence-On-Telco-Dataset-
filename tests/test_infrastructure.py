"""Tests for auth, feature store, event streaming, outbox pattern, and config."""
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from src import config
from src.auth import (
    _hash_password, _verify_password, register_user, login_user,
    get_user_by_token, logout_user,
)
from src.feature_store import FeatureStore
from src.event_streaming import stream_store
from src.outbox_pattern import TransactionalOutboxManager


class TestPasswordHashing(unittest.TestCase):
    def test_hash_format(self):
        h = _hash_password("s3cret")
        parts = h.split("$")
        self.assertEqual(parts[0], "pbkdf2_sha256")
        self.assertEqual(len(parts), 4)

    def test_verify_correct_password(self):
        h = _hash_password("s3cret")
        self.assertTrue(_verify_password("s3cret", h))
        self.assertFalse(_verify_password("wrong", h))

    def test_unique_salts(self):
        self.assertNotEqual(_hash_password("same"), _hash_password("same"))

    def test_constant_time_compare_rejects_malformed(self):
        self.assertFalse(_verify_password("x", "not-a-valid-hash"))
        self.assertFalse(_verify_password("x", "pbkdf2_sha256$bad$zz$gg"))


class TestAuthFlow(unittest.TestCase):
    def test_register_login_token_roundtrip(self):
        email = f"user_{time.time_ns()}@test.com"
        reg = register_user(email, "secret123", "Test User")
        self.assertNotIn("error", reg)
        self.assertEqual(len(reg["token"]), 64)

        login = login_user(email, "secret123")
        self.assertNotIn("error", login)
        self.assertEqual(login["email"], email)

        user = get_user_by_token(login["token"])
        self.assertIsNotNone(user)
        self.assertEqual(user["email"], email)

    def test_wrong_password_rejected(self):
        email = f"bad_{time.time_ns()}@test.com"
        register_user(email, "secret123", "T")
        self.assertIn("error", login_user(email, "wrongpass"))

    def test_duplicate_register_rejected(self):
        email = f"dup_{time.time_ns()}@test.com"
        first = register_user(email, "secret123", "T")
        second = register_user(email, "secret123", "T")
        self.assertIn("error", second)

    def test_unknown_token_rejected(self):
        self.assertIsNone(get_user_by_token("deadbeef" * 8))

    def test_logout_invalidates_token(self):
        email = f"out_{time.time_ns()}@test.com"
        reg = register_user(email, "secret123", "T")
        self.assertTrue(logout_user(reg["token"]))
        self.assertIsNone(get_user_by_token(reg["token"]))

    def test_expired_token_rejected(self):
        email = f"exp_{time.time_ns()}@test.com"
        reg = register_user(email, "secret123", "T")
        # Force expiry in DB directly
        import sqlite3
        from src.auth import AUTH_DB_PATH
        conn = sqlite3.connect(str(AUTH_DB_PATH))
        conn.execute("UPDATE users SET token_expires_at = ? WHERE token = ?",
                     ("2000-01-01T00:00:00+00:00", reg["token"]))
        conn.commit()
        conn.close()
        self.assertIsNone(get_user_by_token(reg["token"]))


class TestFeatureStore(unittest.TestCase):
    def setUp(self):
        self.fs = FeatureStore()  # in-memory fallback (no Redis in CI)

    def test_put_get_roundtrip(self):
        self.fs.put_features("C1", {"tenure": 12})
        self.assertEqual(self.fs.get_features("C1")["tenure"], 12)

    def test_ttl_expiry(self):
        self.fs.put_features("C2", {"x": 1}, ttl=1)
        # Manually expire
        key = "features:C2"
        self.fs._memory_store[key]["expires_at"] = time.time() - 1
        self.assertIsNone(self.fs.get_features("C2"))

    def test_cate_cache_roundtrip(self):
        self.fs.cache_cate_prescription("C3", {"action": "DISCOUNT", "uplift": 0.2}, ttl=60)
        got = self.fs.get_cached_cate_prescription("C3")
        self.assertIsNotNone(got)
        self.assertEqual(got["action"], "DISCOUNT")

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.fs.get_features("NOPE"))


class TestEventStreaming(unittest.TestCase):
    def test_ingest_and_metrics(self):
        before = stream_store.get_metrics().get("total_events_ingested", 0)
        stream_store.simulate_kafka_batch(batch_size=50)
        after = stream_store.get_metrics().get("total_events_ingested", 0)
        self.assertGreaterEqual(after, before + 50)

    def test_metrics_keys(self):
        m = stream_store.get_metrics()
        self.assertIn("total_events_ingested", m)


class TestOutboxPattern(unittest.TestCase):
    def test_idempotency_blocks_duplicates(self):
        mgr = TransactionalOutboxManager()
        self.assertTrue(mgr.check_and_set_idempotency("key-1"))
        self.assertFalse(mgr.check_and_set_idempotency("key-1"))

    def test_idempotency_ttl_expiry(self):
        mgr = TransactionalOutboxManager()
        self.assertTrue(mgr.check_and_set_idempotency("key-2", ttl_seconds=-1))
        self.assertTrue(mgr.check_and_set_idempotency("key-2", ttl_seconds=-1))

    def test_outbox_write_and_dispatch(self):
        mgr = TransactionalOutboxManager()
        rec = mgr.write_transactional_outbox("AGG-1", "TEST_EVENT", {"a": 1})
        self.assertEqual(rec["status"], "DISPATCHED")
        metrics = mgr.get_audit_metrics()
        self.assertEqual(metrics["total_events_dispatched"], 1)


class TestConfig(unittest.TestCase):
    def test_business_constants_present(self):
        self.assertIsInstance(config.DEFAULT_THRESHOLD, float)
        self.assertIsInstance(config.RETENTION_COST, float)
        self.assertIsInstance(config.SAVE_RATE, float)
        self.assertGreater(config.SAVE_RATE, 0)
        self.assertLessEqual(config.SAVE_RATE, 1)

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"RETENTION_COST": "999"}):
            import importlib
            importlib.reload(config)
            self.assertEqual(config.RETENTION_COST, 999.0)
            importlib.reload(config)  # restore


if __name__ == "__main__":
    unittest.main()
