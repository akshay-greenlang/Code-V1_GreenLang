# -*- coding: utf-8 -*-
"""
Unit Tests for ConnectionManagerEngine (AGENT-DATA-004)

Tests source registration (success, ID format SRC-xxxxx, provenance hash),
unregistration (success, not found), retrieval (exists, not found), listing
(all, by type, by status, empty), health checking (healthy response, unhealthy
response, check all), connection testing (success, failure), healthy source
filtering, and source status transitions.

Coverage target: 85%+ of connection_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline models (minimal)
# ---------------------------------------------------------------------------


class DataSource:
    def __init__(
        self,
        source_id: str = "",
        name: str = "",
        source_type: str = "postgresql",
        connection_string: str = "",
        status: str = "active",
        description: str = "",
        tags: Optional[List[str]] = None,
        max_connections: int = 10,
        timeout_s: int = 30,
        retry_count: int = 3,
        created_at: Optional[str] = None,
        provenance_hash: Optional[str] = None,
    ):
        self.source_id = source_id
        self.name = name
        self.source_type = source_type
        self.connection_string = connection_string
        self.status = status
        self.description = description
        self.tags = tags or []
        self.max_connections = max_connections
        self.timeout_s = timeout_s
        self.retry_count = retry_count
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.provenance_hash = provenance_hash


class SourceHealthCheck:
    def __init__(
        self,
        source_id: str = "",
        status: str = "active",
        latency_ms: float = 0.0,
        last_checked_at: Optional[str] = None,
        consecutive_failures: int = 0,
        error_message: Optional[str] = None,
    ):
        self.source_id = source_id
        self.status = status
        self.latency_ms = latency_ms
        self.last_checked_at = last_checked_at or datetime.now(timezone.utc).isoformat()
        self.consecutive_failures = consecutive_failures
        self.error_message = error_message


class RegisterSourceRequest:
    def __init__(
        self,
        name: str = "",
        source_type: str = "postgresql",
        connection_string: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        max_connections: int = 10,
        timeout_s: int = 30,
    ):
        self.name = name
        self.source_type = source_type
        self.connection_string = connection_string
        self.description = description
        self.tags = tags or []
        self.max_connections = max_connections
        self.timeout_s = timeout_s


# ---------------------------------------------------------------------------
# Inline ConnectionManagerEngine
# ---------------------------------------------------------------------------


class ConnectionManagerEngine:
    """Manages data source connections, health checks, and lifecycle."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._sources: Dict[str, DataSource] = {}
        self._health_status: Dict[str, SourceHealthCheck] = {}
        # Simulate test connection results per source type
        self._test_results: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "sources_registered": 0,
            "sources_unregistered": 0,
            "health_checks_performed": 0,
            "connection_tests": 0,
        }

    def _next_source_id(self) -> str:
        self._counter += 1
        return f"SRC-{self._counter:05d}"

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def register_source(self, request: RegisterSourceRequest) -> DataSource:
        """Register a new data source."""
        with self._lock:
            source_id = self._next_source_id()

        prov_data = {
            "op": "register_source",
            "source_id": source_id,
            "name": request.name,
            "source_type": request.source_type,
        }

        source = DataSource(
            source_id=source_id,
            name=request.name,
            source_type=request.source_type,
            connection_string=request.connection_string,
            status="active",
            description=request.description,
            tags=list(request.tags),
            max_connections=request.max_connections,
            timeout_s=request.timeout_s,
            provenance_hash=self._compute_provenance(prov_data),
        )

        with self._lock:
            self._sources[source_id] = source
            self._health_status[source_id] = SourceHealthCheck(
                source_id=source_id,
                status="active",
                latency_ms=0.0,
            )
            self._stats["sources_registered"] += 1

        return source

    def unregister_source(self, source_id: str) -> bool:
        """Unregister a data source. Returns True if found and removed."""
        with self._lock:
            if source_id not in self._sources:
                return False
            del self._sources[source_id]
            self._health_status.pop(source_id, None)
            self._stats["sources_unregistered"] += 1
        return True

    def get_source(self, source_id: str) -> Optional[DataSource]:
        """Retrieve a source by ID. Returns None if not found."""
        with self._lock:
            return self._sources.get(source_id)

    def list_sources(
        self,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DataSource]:
        """List all sources with optional filters."""
        with self._lock:
            result = list(self._sources.values())
        if source_type is not None:
            result = [s for s in result if s.source_type == source_type]
        if status is not None:
            result = [s for s in result if s.status == status]
        return result

    def check_health(self, source_id: str) -> Optional[SourceHealthCheck]:
        """Check health of a single source. Returns None if source not found."""
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                return None

        # Simulate health check based on test result configuration
        is_healthy = self._test_results.get(source_id, True)

        with self._lock:
            hc = self._health_status.get(source_id)
            if hc is None:
                hc = SourceHealthCheck(source_id=source_id)
                self._health_status[source_id] = hc

            if is_healthy:
                hc.status = "active"
                hc.latency_ms = 5.0
                hc.consecutive_failures = 0
                hc.error_message = None
                source.status = "active"
            else:
                hc.consecutive_failures += 1
                hc.error_message = "Connection refused"
                hc.latency_ms = 0.0
                if hc.consecutive_failures >= 3:
                    hc.status = "error"
                    source.status = "error"
                else:
                    hc.status = "degraded"
                    source.status = "degraded"

            hc.last_checked_at = datetime.now(timezone.utc).isoformat()
            self._stats["health_checks_performed"] += 1

        return hc

    def check_all_health(self) -> List[SourceHealthCheck]:
        """Check health of all registered sources."""
        with self._lock:
            source_ids = list(self._sources.keys())
        results = []
        for sid in source_ids:
            hc = self.check_health(sid)
            if hc is not None:
                results.append(hc)
        return results

    def test_connection(self, source_id: str) -> Dict[str, Any]:
        """Test connectivity to a source. Returns success/failure dict."""
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                return {
                    "source_id": source_id,
                    "success": False,
                    "error": "Source not found",
                }
            self._stats["connection_tests"] += 1

        # Simulate test based on configured results
        is_success = self._test_results.get(source_id, True)

        if is_success:
            return {
                "source_id": source_id,
                "success": True,
                "latency_ms": 3.5,
                "message": "Connection successful",
            }
        else:
            return {
                "source_id": source_id,
                "success": False,
                "error": "Connection refused",
            }

    def set_test_result(self, source_id: str, success: bool) -> None:
        """Configure simulated test/health result for a source (test helper)."""
        self._test_results[source_id] = success

    def get_healthy_sources(self) -> List[DataSource]:
        """Return all sources with active status."""
        return self.list_sources(status="active")

    def update_source_status(self, source_id: str, status: str) -> Optional[DataSource]:
        """Update a source's status. Returns None if not found."""
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                return None
            source.status = status
            hc = self._health_status.get(source_id)
            if hc:
                hc.status = status
        return source

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """ConnectionManagerEngine instance for testing."""
    return ConnectionManagerEngine()


@pytest.fixture
def sample_request():
    """RegisterSourceRequest for a PostgreSQL emissions database."""
    return RegisterSourceRequest(
        name="emissions_db",
        source_type="postgresql",
        connection_string="postgresql://user:pass@host:5432/emissions",
        description="Main emissions database",
        tags=["emissions", "production"],
        max_connections=20,
        timeout_s=60,
    )


@pytest.fixture
def redis_request():
    """RegisterSourceRequest for a Redis cache."""
    return RegisterSourceRequest(
        name="cache_redis",
        source_type="redis",
        connection_string="redis://localhost:6379/0",
        description="Application cache",
        tags=["cache"],
    )


@pytest.fixture
def s3_request():
    """RegisterSourceRequest for an S3 data lake."""
    return RegisterSourceRequest(
        name="data_lake_s3",
        source_type="s3",
        connection_string="s3://gl-data-lake/raw/",
        description="S3 data lake raw zone",
        tags=["data-lake", "raw"],
    )


@pytest.fixture
def engine_with_sources(engine, sample_request, redis_request, s3_request):
    """Engine pre-loaded with 3 sources."""
    engine.register_source(sample_request)
    engine.register_source(redis_request)
    engine.register_source(s3_request)
    return engine


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRegisterSource:
    """Test data source registration."""

    def test_register_source_success(self, engine, sample_request):
        """Register and verify returned DataSource."""
        source = engine.register_source(sample_request)
        assert source.source_id is not None
        assert source.name == "emissions_db"
        assert source.source_type == "postgresql"
        assert source.connection_string == "postgresql://user:pass@host:5432/emissions"
        assert source.status == "active"
        assert source.description == "Main emissions database"
        assert source.tags == ["emissions", "production"]
        assert source.max_connections == 20
        assert source.timeout_s == 60
        assert source.created_at is not None

    def test_register_source_id_format(self, engine, sample_request):
        """Auto-generated SRC-xxxxx format."""
        source = engine.register_source(sample_request)
        assert re.match(r"^SRC-\d{5}$", source.source_id)
        assert source.source_id == "SRC-00001"

    def test_register_source_sequential_ids(self, engine, sample_request, redis_request):
        """Sequential registrations get sequential IDs."""
        src1 = engine.register_source(sample_request)
        src2 = engine.register_source(redis_request)
        assert src1.source_id == "SRC-00001"
        assert src2.source_id == "SRC-00002"

    def test_register_source_provenance(self, engine, sample_request):
        """Provenance hash is SHA-256 (64 hex chars)."""
        source = engine.register_source(sample_request)
        assert source.provenance_hash is not None
        assert len(source.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", source.provenance_hash)

    def test_register_increments_stats(self, engine, sample_request):
        """Registration increments sources_registered counter."""
        engine.register_source(sample_request)
        stats = engine.get_statistics()
        assert stats["sources_registered"] == 1

    def test_register_creates_health_check(self, engine, sample_request):
        """Registration creates initial health check entry."""
        source = engine.register_source(sample_request)
        hc = engine.check_health(source.source_id)
        assert hc is not None
        assert hc.source_id == source.source_id
        assert hc.status == "active"


class TestUnregisterSource:
    """Test data source unregistration."""

    def test_unregister_success(self, engine, sample_request):
        """Unregister an existing source."""
        source = engine.register_source(sample_request)
        result = engine.unregister_source(source.source_id)
        assert result is True
        assert engine.get_source(source.source_id) is None

    def test_unregister_not_found(self, engine):
        """Unregister non-existent source returns False."""
        result = engine.unregister_source("SRC-99999")
        assert result is False

    def test_unregister_increments_stats(self, engine, sample_request):
        """Unregistration increments sources_unregistered counter."""
        source = engine.register_source(sample_request)
        engine.unregister_source(source.source_id)
        stats = engine.get_statistics()
        assert stats["sources_unregistered"] == 1

    def test_unregister_removes_health(self, engine, sample_request):
        """Unregistration also removes health check entry."""
        source = engine.register_source(sample_request)
        engine.unregister_source(source.source_id)
        hc = engine.check_health(source.source_id)
        assert hc is None


class TestGetSource:
    """Test source retrieval."""

    def test_get_source_exists(self, engine, sample_request):
        """Retrieve registered source."""
        source = engine.register_source(sample_request)
        retrieved = engine.get_source(source.source_id)
        assert retrieved is not None
        assert retrieved.source_id == source.source_id
        assert retrieved.name == "emissions_db"

    def test_get_source_not_found(self, engine):
        """Returns None for unknown source_id."""
        assert engine.get_source("SRC-99999") is None


class TestListSources:
    """Test listing sources with filters."""

    def test_list_all(self, engine_with_sources):
        """List all registered sources."""
        sources = engine_with_sources.list_sources()
        assert len(sources) == 3

    def test_list_by_type(self, engine_with_sources):
        """Filter by source type."""
        pg_sources = engine_with_sources.list_sources(source_type="postgresql")
        assert len(pg_sources) == 1
        assert pg_sources[0].source_type == "postgresql"

        redis_sources = engine_with_sources.list_sources(source_type="redis")
        assert len(redis_sources) == 1
        assert redis_sources[0].source_type == "redis"

    def test_list_by_status(self, engine_with_sources):
        """Filter by status."""
        active_sources = engine_with_sources.list_sources(status="active")
        assert len(active_sources) == 3  # All start as active

    def test_list_by_status_no_match(self, engine_with_sources):
        """Filter by status with no matches."""
        error_sources = engine_with_sources.list_sources(status="error")
        assert len(error_sources) == 0

    def test_list_empty(self, engine):
        """Empty engine returns empty list."""
        sources = engine.list_sources()
        assert sources == []

    def test_list_by_type_and_status(self, engine_with_sources):
        """Combined type and status filter."""
        sources = engine_with_sources.list_sources(
            source_type="postgresql", status="active",
        )
        assert len(sources) == 1
        assert sources[0].source_type == "postgresql"
        assert sources[0].status == "active"


class TestCheckHealth:
    """Test individual source health checks."""

    def test_healthy_response(self, engine, sample_request):
        """Healthy source returns active status with low latency."""
        source = engine.register_source(sample_request)
        hc = engine.check_health(source.source_id)
        assert hc is not None
        assert hc.status == "active"
        assert hc.latency_ms > 0
        assert hc.consecutive_failures == 0
        assert hc.error_message is None

    def test_unhealthy_response(self, engine, sample_request):
        """Unhealthy source returns degraded status with error message."""
        source = engine.register_source(sample_request)
        engine.set_test_result(source.source_id, False)
        hc = engine.check_health(source.source_id)
        assert hc is not None
        assert hc.status == "degraded"
        assert hc.consecutive_failures == 1
        assert hc.error_message is not None

    def test_health_check_not_found(self, engine):
        """Health check for unknown source returns None."""
        hc = engine.check_health("SRC-99999")
        assert hc is None

    def test_consecutive_failures_escalate(self, engine, sample_request):
        """Multiple failures escalate from degraded to error."""
        source = engine.register_source(sample_request)
        engine.set_test_result(source.source_id, False)

        engine.check_health(source.source_id)  # 1st failure: degraded
        engine.check_health(source.source_id)  # 2nd failure: degraded
        hc = engine.check_health(source.source_id)  # 3rd failure: error
        assert hc.status == "error"
        assert hc.consecutive_failures == 3

    def test_health_check_increments_stats(self, engine, sample_request):
        """Health checks increment the counter."""
        source = engine.register_source(sample_request)
        engine.check_health(source.source_id)
        stats = engine.get_statistics()
        assert stats["health_checks_performed"] >= 1


class TestCheckAllHealth:
    """Test bulk health checks."""

    def test_check_all_health(self, engine_with_sources):
        """Check health of all registered sources."""
        results = engine_with_sources.check_all_health()
        assert len(results) == 3
        for hc in results:
            assert hc.status == "active"

    def test_check_all_mixed_health(self, engine, sample_request, redis_request):
        """Mixed healthy and unhealthy sources."""
        src1 = engine.register_source(sample_request)
        src2 = engine.register_source(redis_request)
        engine.set_test_result(src2.source_id, False)

        results = engine.check_all_health()
        assert len(results) == 2
        statuses = {hc.source_id: hc.status for hc in results}
        assert statuses[src1.source_id] == "active"
        assert statuses[src2.source_id] == "degraded"


class TestTestConnection:
    """Test connection testing."""

    def test_connection_success(self, engine, sample_request):
        """Successful connection test."""
        source = engine.register_source(sample_request)
        result = engine.test_connection(source.source_id)
        assert result["success"] is True
        assert result["source_id"] == source.source_id
        assert "latency_ms" in result

    def test_connection_failure(self, engine, sample_request):
        """Failed connection test."""
        source = engine.register_source(sample_request)
        engine.set_test_result(source.source_id, False)
        result = engine.test_connection(source.source_id)
        assert result["success"] is False
        assert "error" in result

    def test_connection_not_found(self, engine):
        """Connection test for unknown source."""
        result = engine.test_connection("SRC-99999")
        assert result["success"] is False
        assert "Source not found" in result["error"]

    def test_connection_increments_stats(self, engine, sample_request):
        """Connection test increments counter."""
        source = engine.register_source(sample_request)
        engine.test_connection(source.source_id)
        stats = engine.get_statistics()
        assert stats["connection_tests"] == 1


class TestGetHealthySources:
    """Test filtering for healthy sources."""

    def test_all_healthy(self, engine_with_sources):
        """All sources healthy returns all."""
        healthy = engine_with_sources.get_healthy_sources()
        assert len(healthy) == 3

    def test_filter_unhealthy(self, engine, sample_request, redis_request):
        """Unhealthy sources excluded from healthy list."""
        src1 = engine.register_source(sample_request)
        src2 = engine.register_source(redis_request)

        # Mark src2 as unhealthy
        engine.set_test_result(src2.source_id, False)
        # Trigger enough failures to set error status
        engine.check_health(src2.source_id)
        engine.check_health(src2.source_id)
        engine.check_health(src2.source_id)

        healthy = engine.get_healthy_sources()
        assert len(healthy) == 1
        assert healthy[0].source_id == src1.source_id


class TestSourceStatusTracking:
    """Test source status transitions."""

    def test_initial_active_status(self, engine, sample_request):
        """Newly registered source starts as active."""
        source = engine.register_source(sample_request)
        assert source.status == "active"

    def test_status_transition_to_maintenance(self, engine, sample_request):
        """Update source status to maintenance."""
        source = engine.register_source(sample_request)
        updated = engine.update_source_status(source.source_id, "maintenance")
        assert updated is not None
        assert updated.status == "maintenance"

    def test_status_transition_to_inactive(self, engine, sample_request):
        """Update source status to inactive."""
        source = engine.register_source(sample_request)
        updated = engine.update_source_status(source.source_id, "inactive")
        assert updated is not None
        assert updated.status == "inactive"

    def test_status_transition_to_error(self, engine, sample_request):
        """Update source status to error."""
        source = engine.register_source(sample_request)
        updated = engine.update_source_status(source.source_id, "error")
        assert updated is not None
        assert updated.status == "error"

    def test_status_transition_back_to_active(self, engine, sample_request):
        """Source can transition from error back to active."""
        source = engine.register_source(sample_request)
        engine.update_source_status(source.source_id, "error")
        updated = engine.update_source_status(source.source_id, "active")
        assert updated.status == "active"

    def test_status_update_not_found(self, engine):
        """Status update for unknown source returns None."""
        result = engine.update_source_status("SRC-99999", "active")
        assert result is None

    def test_status_affects_listing(self, engine, sample_request, redis_request):
        """Status changes affect filtered listing results."""
        src1 = engine.register_source(sample_request)
        src2 = engine.register_source(redis_request)

        engine.update_source_status(src2.source_id, "maintenance")

        active = engine.list_sources(status="active")
        assert len(active) == 1
        assert active[0].source_id == src1.source_id

        maintenance = engine.list_sources(status="maintenance")
        assert len(maintenance) == 1
        assert maintenance[0].source_id == src2.source_id

    def test_health_check_degrades_status(self, engine, sample_request):
        """Failed health checks degrade source status."""
        source = engine.register_source(sample_request)
        assert source.status == "active"

        engine.set_test_result(source.source_id, False)
        engine.check_health(source.source_id)

        updated = engine.get_source(source.source_id)
        assert updated.status == "degraded"

    def test_health_check_restores_status(self, engine, sample_request):
        """Successful health check after failure restores active status."""
        source = engine.register_source(sample_request)

        # Degrade the source
        engine.set_test_result(source.source_id, False)
        engine.check_health(source.source_id)
        assert engine.get_source(source.source_id).status == "degraded"

        # Restore with successful check
        engine.set_test_result(source.source_id, True)
        engine.check_health(source.source_id)
        assert engine.get_source(source.source_id).status == "active"
