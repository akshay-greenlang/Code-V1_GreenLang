# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.connectors framework (F060)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from greenlang.factors.connectors.base import (
    BaseConnector,
    ConnectorCapabilities,
    ConnectorHealthResult,
    ConnectorStatus,
)
from greenlang.factors.connectors.config import ConnectorConfig, RetryPolicy
from greenlang.factors.connectors.registry import ConnectorRegistry, get_global_registry
from greenlang.factors.connectors.license_manager import LicenseManager, LicenseKeyRecord
from greenlang.factors.connectors.audit_log import ConnectorAuditEntry, ConnectorAuditLog
from greenlang.factors.connectors.metrics import (
    get_fallback_metrics,
    record_error,
    record_factors_fetched,
    record_latency,
    record_request,
    set_quota_remaining,
    track_connector_call,
)


# ---------------------------------------------------------------------------
# Concrete test connector
# ---------------------------------------------------------------------------

class _MockConnector(BaseConnector):
    @property
    def source_id(self) -> str:
        return "mock_source"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities(typical_factor_count=100)

    def fetch_metadata(self) -> List[Dict[str, Any]]:
        self._track_request()
        return [{"factor_id": "EF:MOCK:1"}, {"factor_id": "EF:MOCK:2"}]

    def fetch_factors(self, factor_ids: List[str], *, license_key: Optional[str] = None) -> List[Dict[str, Any]]:
        self.get_license_key(license_key)
        self._track_request()
        return [{"factor_id": fid, "co2e_total": 1.0} for fid in factor_ids]

    def health_check(self) -> ConnectorHealthResult:
        return ConnectorHealthResult(status=ConnectorStatus.HEALTHY, latency_ms=10)


class _FailingConnector(BaseConnector):
    @property
    def source_id(self) -> str:
        return "failing_source"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        return ConnectorCapabilities()

    def fetch_metadata(self) -> List[Dict[str, Any]]:
        raise RuntimeError("API down")

    def fetch_factors(self, factor_ids: List[str], *, license_key: Optional[str] = None) -> List[Dict[str, Any]]:
        raise RuntimeError("API down")

    def health_check(self) -> ConnectorHealthResult:
        return ConnectorHealthResult(status=ConnectorStatus.UNAVAILABLE, message="down")


# ---------------------------------------------------------------------------
# BaseConnector tests
# ---------------------------------------------------------------------------

class TestBaseConnector:
    def test_fetch_metadata(self):
        c = _MockConnector(license_key="test-key")
        meta = c.fetch_metadata()
        assert len(meta) == 2
        assert meta[0]["factor_id"] == "EF:MOCK:1"

    def test_fetch_factors(self):
        c = _MockConnector(license_key="test-key")
        factors = c.fetch_factors(["EF:1", "EF:2"])
        assert len(factors) == 2

    def test_fetch_factors_no_key(self):
        c = _MockConnector()
        with pytest.raises(Exception, match="No license key"):
            c.fetch_factors(["EF:1"])

    def test_health_check(self):
        c = _MockConnector()
        result = c.health_check()
        assert result.status == ConnectorStatus.HEALTHY

    def test_stats(self):
        c = _MockConnector(license_key="k")
        c.fetch_metadata()
        c.fetch_factors(["EF:1"])
        assert c.stats["request_count"] == 2
        assert c.stats["error_count"] == 0

    def test_batched_fetch(self):
        c = _MockConnector(license_key="k")
        ids = [f"EF:{i}" for i in range(5)]
        results = c.fetch_factors_batched(ids, batch_size=2)
        assert len(results) == 5

    def test_repr(self):
        c = _MockConnector()
        assert "MockConnector" in repr(c)
        assert "mock_source" in repr(c)


# ---------------------------------------------------------------------------
# ConnectorConfig tests
# ---------------------------------------------------------------------------

class TestConnectorConfig:
    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = ConnectorConfig.from_env("iea")
        assert cfg.source_id == "iea"
        assert cfg.timeout_sec == 30
        assert cfg.batch_size == 1000

    def test_from_env_custom(self):
        env = {
            "GL_FACTORS_IEA_API_ENDPOINT": "https://custom.api/v1",
            "GL_FACTORS_IEA_LICENSE_KEY": "secret123",
            "GL_FACTORS_IEA_TIMEOUT_SEC": "60",
            "GL_FACTORS_IEA_BATCH_SIZE": "500",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = ConnectorConfig.from_env("iea")
        assert cfg.api_endpoint == "https://custom.api/v1"
        assert cfg.license_key == "secret123"
        assert cfg.timeout_sec == 60
        assert cfg.batch_size == 500
        assert cfg.has_license is True

    def test_to_dict(self):
        cfg = ConnectorConfig(source_id="test", license_key="secret")
        d = cfg.to_dict()
        assert d["source_id"] == "test"
        assert d["has_license"] is True
        assert "license_key" not in d  # Should not expose key

    def test_retry_policy(self):
        rp = RetryPolicy(max_retries=5)
        assert rp.max_retries == 5
        assert 429 in rp.retry_on_status


# ---------------------------------------------------------------------------
# ConnectorRegistry tests
# ---------------------------------------------------------------------------

class TestConnectorRegistry:
    def test_register_and_get_instance(self):
        reg = ConnectorRegistry()
        c = _MockConnector(license_key="k")
        reg.register_instance(c)
        assert reg.get("mock_source") is c

    def test_register_and_get_class(self):
        reg = ConnectorRegistry()
        reg.register_class("mock_source", _MockConnector)
        c = reg.get("mock_source", license_key="k")
        assert c is not None
        assert c.source_id == "mock_source"

    def test_get_unknown(self):
        reg = ConnectorRegistry()
        assert reg.get("nonexistent") is None

    def test_list_source_ids(self):
        reg = ConnectorRegistry()
        reg.register_class("a", _MockConnector)
        reg.register_instance(_FailingConnector())
        ids = reg.list_source_ids()
        assert "a" in ids
        assert "failing_source" in ids

    def test_unregister(self):
        reg = ConnectorRegistry()
        reg.register_instance(_MockConnector())
        assert reg.unregister("mock_source") is True
        assert reg.get("mock_source") is None
        assert reg.unregister("mock_source") is False

    def test_clear(self):
        reg = ConnectorRegistry()
        reg.register_class("a", _MockConnector)
        reg.register_instance(_FailingConnector())
        reg.clear()
        assert len(reg) == 0

    def test_contains(self):
        reg = ConnectorRegistry()
        reg.register_class("iea", _MockConnector)
        assert "iea" in reg
        assert "unknown" not in reg

    def test_global_registry_singleton(self):
        r1 = get_global_registry()
        r2 = get_global_registry()
        assert r1 is r2


# ---------------------------------------------------------------------------
# LicenseManager tests
# ---------------------------------------------------------------------------

class TestLicenseManager:
    def test_resolve_from_env(self):
        lm = LicenseManager()
        env = {"GL_FACTORS_LICENSE_IEA": "my-secret-key"}
        with patch.dict("os.environ", env, clear=True):
            key = lm.resolve_key("iea")
        assert key == "my-secret-key"

    def test_resolve_tenant_specific(self):
        lm = LicenseManager()
        env = {"GL_FACTORS_LICENSE_IEA_ACME": "acme-key"}
        with patch.dict("os.environ", env, clear=True):
            key = lm.resolve_key("iea", "acme")
        assert key == "acme-key"

    def test_resolve_no_key(self):
        lm = LicenseManager()
        with patch.dict("os.environ", {}, clear=True):
            key = lm.resolve_key("iea")
        assert key is None

    def test_register_key(self):
        lm = LicenseManager()
        record = lm.register_key("iea", "default", "new-key-123")
        assert record.connector_id == "iea"
        assert record.tenant_id == "default"
        assert len(record.key_hash) == 64  # SHA-256

    def test_rotate_key(self):
        lm = LicenseManager()
        lm.register_key("iea", "default", "old-key")
        record = lm.rotate_key("iea", "default", "new-key")
        assert record.rotated_from is not None

    def test_revoke_key(self):
        lm = LicenseManager()
        lm.register_key("iea", "default", "key-to-revoke")
        assert lm.revoke_key("iea", "default") is True
        # Should now resolve to None (from cache)
        with patch.dict("os.environ", {}, clear=True):
            key = lm.resolve_key("iea")
        assert key is None

    def test_validate_key(self):
        lm = LicenseManager()
        assert lm.validate_key("iea", "valid-key-123") is True
        assert lm.validate_key("iea", "") is False
        assert lm.validate_key("iea", "short") is False

    def test_audit_log(self):
        lm = LicenseManager()
        lm.register_key("iea", "default", "key1")
        lm.resolve_key("iea", "default")
        assert len(lm.audit_log) >= 2


# ---------------------------------------------------------------------------
# ConnectorAuditLog tests
# ---------------------------------------------------------------------------

class TestConnectorAuditLog:
    def test_in_memory_log(self):
        log = ConnectorAuditLog()
        entry = ConnectorAuditEntry(
            connector_id="iea",
            operation="fetch_metadata",
            timestamp="2026-04-17T00:00:00Z",
            response_factor_count=100,
        )
        log.log(entry)
        assert len(log) == 1
        assert log.entries[0].connector_id == "iea"

    def test_sqlite_persistence(self, tmp_path):
        db = tmp_path / "audit.db"
        log = ConnectorAuditLog(db_path=db)
        entry = ConnectorAuditEntry(
            connector_id="ecoinvent",
            operation="fetch_factors",
            timestamp="2026-04-17T01:00:00Z",
            success=False,
            error="timeout",
        )
        log.log(entry)
        assert db.exists()
        assert len(log) == 1

    def test_query_by_connector(self):
        log = ConnectorAuditLog()
        log.log(ConnectorAuditEntry(connector_id="iea", operation="fetch", timestamp="t1"))
        log.log(ConnectorAuditEntry(connector_id="eci", operation="fetch", timestamp="t2"))
        results = log.query(connector_id="iea")
        assert len(results) == 1
        assert results[0].connector_id == "iea"

    def test_query_success_only(self):
        log = ConnectorAuditLog()
        log.log(ConnectorAuditEntry(connector_id="a", operation="f", timestamp="t1", success=True))
        log.log(ConnectorAuditEntry(connector_id="a", operation="f", timestamp="t2", success=False))
        results = log.query(success_only=True)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_record_request(self):
        record_request("test_conn", "fetch", success=True)
        m = get_fallback_metrics()
        assert m.requests.get("test_conn:fetch:ok", 0) >= 1

    def test_record_error(self):
        record_error("test_conn", "ConnectionTimeout")
        m = get_fallback_metrics()
        assert m.errors.get("test_conn:ConnectionTimeout", 0) >= 1

    def test_record_latency(self):
        record_latency("test_conn", "fetch", 0.5)
        m = get_fallback_metrics()
        assert len(m.latencies.get("test_conn:fetch", [])) >= 1

    def test_record_factors_fetched(self):
        record_factors_fetched("test_conn", 100)
        m = get_fallback_metrics()
        assert m.factors_fetched.get("test_conn", 0) >= 100

    def test_set_quota(self):
        set_quota_remaining("test_conn", 500)
        m = get_fallback_metrics()
        assert m.quota_remaining.get("test_conn") == 500

    def test_track_connector_call_success(self):
        with track_connector_call("test_conn", "test_op"):
            pass  # Simulates successful call

    def test_track_connector_call_failure(self):
        with pytest.raises(ValueError):
            with track_connector_call("test_conn", "test_op"):
                raise ValueError("test error")
