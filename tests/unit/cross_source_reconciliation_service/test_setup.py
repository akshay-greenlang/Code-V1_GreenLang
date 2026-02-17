# -*- coding: utf-8 -*-
"""
Unit Tests for CrossSourceReconciliationService (AGENT-DATA-015)
=================================================================

Comprehensive test suite for ``greenlang.cross_source_reconciliation.setup``
covering the ``CrossSourceReconciliationService`` facade, all public methods,
lifecycle, statistics, provenance, configuration helpers, singleton
management, and full end-to-end workflows.

Target: 40+ tests with 85%+ coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.cross_source_reconciliation.setup import (
    CrossSourceReconciliationService,
    configure_reconciliation,
    get_reconciliation,
    get_router,
    get_service,
    reset_service,
    _compute_hash,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_service(**config_overrides: Any) -> CrossSourceReconciliationService:
    """Create a CrossSourceReconciliationService with optional config overrides.

    Relies on the autouse ``_patch_engines_and_reset`` fixture having
    set all engine class references to None so that startup() skips
    engine construction.
    """
    svc = CrossSourceReconciliationService(config=config_overrides or {})
    svc.startup()
    return svc


def _sample_records_a() -> List[Dict[str, Any]]:
    """Return a list of sample records for source A."""
    return [
        {"entity_id": "facility-001", "period": "2025-Q1", "electricity_kwh": 12500.0, "gas_m3": 3400.0},
        {"entity_id": "facility-002", "period": "2025-Q1", "electricity_kwh": 9800.0, "gas_m3": 2200.0},
        {"entity_id": "facility-003", "period": "2025-Q2", "electricity_kwh": 15000.0, "gas_m3": 4100.0},
    ]


def _sample_records_b() -> List[Dict[str, Any]]:
    """Return a list of sample records for source B (with discrepancies)."""
    return [
        {"entity_id": "facility-001", "period": "2025-Q1", "electricity_kwh": 12650.0, "gas_m3": 3400.0},
        {"entity_id": "facility-002", "period": "2025-Q1", "electricity_kwh": 9800.0, "gas_m3": 2250.0},
        {"entity_id": "facility-004", "period": "2025-Q2", "electricity_kwh": 7700.0, "gas_m3": 1800.0},
    ]


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def _patch_engines_and_reset():
    """Patch engine classes to None and reset the singleton before each test.

    The actual engine constructors do not accept ``config=`` kwargs, so
    we set the module-level references to None so that startup() skips
    engine construction. All facade methods exercise the fallback code
    paths instead.
    """
    import greenlang.cross_source_reconciliation.setup as setup_mod

    engine_names = [
        "SourceRegistryEngine",
        "MatchingEngine",
        "ComparisonEngine",
        "DiscrepancyDetectorEngine",
        "ResolutionEngine",
        "AuditTrailEngine",
        "ReconciliationPipelineEngine",
    ]

    originals = {}
    for name in engine_names:
        originals[name] = getattr(setup_mod, name)
        setattr(setup_mod, name, None)

    setup_mod._service_instance = None

    yield

    # Restore and reset
    for name, original in originals.items():
        setattr(setup_mod, name, original)
    setup_mod._service_instance = None


@pytest.fixture
def service() -> CrossSourceReconciliationService:
    """Create a fresh CrossSourceReconciliationService for each test."""
    return _make_service()


# ===================================================================
# 1. Service Lifecycle Tests
# ===================================================================


class TestServiceLifecycle:
    """Tests for service creation, startup, shutdown, and singleton."""

    def test_service_creates_successfully(self):
        """Service can be created with default config."""
        svc = CrossSourceReconciliationService()
        assert svc is not None
        assert svc.config is not None
        assert svc._started is False

    def test_service_creates_with_custom_config(self):
        """Service accepts a custom config dictionary."""
        cfg = {"batch_size": 500, "max_records": 50000}
        svc = CrossSourceReconciliationService(config=cfg)
        assert svc.config["batch_size"] == 500
        assert svc.config["max_records"] == 50000

    def test_service_creates_with_none_config(self):
        """Service handles None config gracefully."""
        svc = CrossSourceReconciliationService(config=None)
        assert svc.config == {}

    def test_startup_sets_started_flag(self):
        """startup() sets the _started flag to True."""
        svc = CrossSourceReconciliationService()
        assert svc._started is False
        svc.startup()
        assert svc._started is True

    def test_shutdown_clears_started_flag(self, service):
        """shutdown() sets the _started flag to False."""
        assert service._started is True
        service.shutdown()
        assert service._started is False

    def test_get_service_singleton_returns_same_instance(self):
        """get_service() returns the same singleton on repeated calls."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_reset_service_creates_new_instance(self):
        """reset_service() replaces the singleton with a new instance."""
        svc1 = get_service()
        svc2 = reset_service()
        assert svc1 is not svc2
        assert svc2._started is True

    def test_service_has_provenance_tracker(self, service):
        """Service has a provenance tracker."""
        assert service._provenance is not None

    def test_service_initializes_empty_stores(self, service):
        """Service initializes all in-memory stores as empty dicts."""
        assert isinstance(service._jobs, dict)
        assert len(service._jobs) == 0
        assert isinstance(service._sources, dict)
        assert len(service._sources) == 0
        assert isinstance(service._matches, dict)
        assert len(service._matches) == 0
        assert isinstance(service._discrepancies, dict)
        assert len(service._discrepancies) == 0
        assert isinstance(service._golden_records, dict)
        assert len(service._golden_records) == 0

    def test_service_initializes_stats_counters(self, service):
        """Service initializes aggregate stat counters at zero."""
        for key in (
            "total_jobs", "total_sources", "total_matches",
            "total_comparisons", "total_discrepancies",
            "total_resolutions", "total_golden_records", "total_pipelines",
        ):
            assert service._stats[key] == 0

    def test_health_check_returns_valid_response(self, service):
        """health_check() returns a dict with required keys."""
        result = service.health_check()
        assert isinstance(result, dict)
        assert "status" in result
        assert "service" in result
        assert "engines" in result
        assert "stores" in result
        assert "timestamp" in result

    def test_health_check_status_healthy(self, service):
        """health_check() reports healthy when started."""
        result = service.health_check()
        assert result["status"] == "healthy"

    def test_health_check_status_starting_when_not_started(self):
        """health_check() reports starting when not started."""
        svc = CrossSourceReconciliationService()
        result = svc.health_check()
        assert result["status"] == "starting"

    def test_health_check_service_name(self, service):
        """health_check() includes the correct service name."""
        result = service.health_check()
        assert result["service"] == "cross_source_reconciliation"

    def test_health_check_engines_are_booleans(self, service):
        """health_check() engines are all boolean values."""
        result = service.health_check()
        for key, val in result["engines"].items():
            assert isinstance(val, bool), f"Engine {key} is not bool"

    def test_health_check_stores_are_ints(self, service):
        """health_check() stores are all integer counts."""
        result = service.health_check()
        for key, val in result["stores"].items():
            assert isinstance(val, int), f"Store {key} is not int"

    def test_get_stats_returns_statistics(self, service):
        """get_stats() returns a dict with counter keys."""
        result = service.get_stats()
        assert isinstance(result, dict)
        assert "total_jobs" in result
        assert "total_sources" in result
        assert "total_matches" in result
        assert "total_comparisons" in result
        assert "total_discrepancies" in result
        assert "total_resolutions" in result
        assert "total_golden_records" in result
        assert "total_pipelines" in result
        assert "timestamp" in result

    def test_get_stats_includes_stored_counts(self, service):
        """get_stats() includes stored item counts."""
        result = service.get_stats()
        assert "jobs_stored" in result
        assert "sources_stored" in result
        assert "matches_stored" in result
        assert "provenance_entries" in result

    def test_get_statistics_alias(self, service):
        """get_statistics() is an alias for get_stats()."""
        stats1 = service.get_stats()
        stats2 = service.get_statistics()
        assert stats1["total_jobs"] == stats2["total_jobs"]
        assert stats1["total_sources"] == stats2["total_sources"]

    def test_get_health_alias(self, service):
        """get_health() is an alias for health_check()."""
        h1 = service.health_check()
        h2 = service.get_health()
        assert h1["status"] == h2["status"]
        assert h1["service"] == h2["service"]


# ===================================================================
# 2. Module-Level Function Tests
# ===================================================================


class TestModuleFunctions:
    """Tests for configure_reconciliation, get_reconciliation, get_router."""

    def test_configure_reconciliation_attaches_to_app_state(self):
        """configure_reconciliation(app) sets app.state attribute."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        svc = configure_reconciliation(app)
        assert app.state.cross_source_reconciliation_service is svc

    def test_get_reconciliation_retrieves_service(self):
        """get_reconciliation(app) retrieves the service from app.state."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        svc = configure_reconciliation(app)
        retrieved = get_reconciliation(app)
        assert retrieved is svc

    def test_get_reconciliation_returns_none_when_not_configured(self):
        """get_reconciliation() returns None when service not configured."""
        app = MagicMock()
        app.state = MagicMock(spec=[])
        result = get_reconciliation(app)
        assert result is None

    def test_get_router_returns_api_router(self):
        """get_router() returns a FastAPI APIRouter."""
        router = get_router()
        assert router is not None

    def test_compute_hash_returns_sha256(self):
        """_compute_hash returns a 64-char hex SHA-256 hash."""
        h = _compute_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_compute_hash_deterministic(self):
        """_compute_hash returns the same hash for same input."""
        data = {"a": 1, "b": 2}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_compute_hash_key_order_independent(self):
        """_compute_hash sorts keys, so insertion order does not matter."""
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})
        assert h1 == h2


# ===================================================================
# 3. Job Management Tests
# ===================================================================


class TestJobManagement:
    """Tests for create_job, list_jobs, get_job, delete_job."""

    def test_create_job_returns_dict(self, service):
        """create_job returns a dict with job_id."""
        result = service.create_job(name="test-job")
        assert isinstance(result, dict)
        assert "job_id" in result
        assert result["status"] == "pending"

    def test_create_job_default_name(self, service):
        """create_job auto-generates name when not provided."""
        result = service.create_job()
        assert result["name"].startswith("reconciliation-")

    def test_create_job_custom_strategy(self, service):
        """create_job respects custom strategy."""
        result = service.create_job(strategy="weighted_average")
        assert result["strategy"] == "weighted_average"

    def test_create_job_with_source_ids(self, service):
        """create_job stores source_ids."""
        result = service.create_job(source_ids=["src-1", "src-2"])
        assert result["source_ids"] == ["src-1", "src-2"]

    def test_create_job_increments_stats(self, service):
        """create_job increments total_jobs counter."""
        assert service._stats["total_jobs"] == 0
        service.create_job()
        assert service._stats["total_jobs"] == 1

    def test_create_job_has_provenance_hash(self, service):
        """create_job result includes a 64-char provenance_hash."""
        result = service.create_job()
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_create_job_stores_in_jobs_dict(self, service):
        """create_job stores the job in the internal jobs dict."""
        result = service.create_job(name="stored-job")
        assert result["job_id"] in service._jobs

    def test_list_jobs_returns_all(self, service):
        """list_jobs returns all stored jobs."""
        service.create_job(name="job-1")
        service.create_job(name="job-2")
        result = service.list_jobs()
        assert result["total"] == 2
        assert result["count"] == 2

    def test_list_jobs_supports_status_filter(self, service):
        """list_jobs filters by status."""
        service.create_job(name="job-1")
        result = service.list_jobs(status="pending")
        assert result["count"] == 1
        result_empty = service.list_jobs(status="completed")
        assert result_empty["count"] == 0

    def test_list_jobs_pagination(self, service):
        """list_jobs supports limit and offset."""
        for i in range(5):
            service.create_job(name=f"job-{i}")
        result = service.list_jobs(limit=2, offset=1)
        assert result["count"] == 2
        assert result["total"] == 5

    def test_get_job_found(self, service):
        """get_job returns the job dict for a known ID."""
        created = service.create_job(name="get-me")
        fetched = service.get_job(created["job_id"])
        assert fetched is not None
        assert fetched["name"] == "get-me"

    def test_get_job_not_found(self, service):
        """get_job returns None for an unknown ID."""
        result = service.get_job("nonexistent-id")
        assert result is None

    def test_delete_job_removes_job(self, service):
        """delete_job removes the job from storage."""
        created = service.create_job(name="delete-me")
        result = service.delete_job(created["job_id"])
        assert result["status"] == "cancelled"
        assert service.get_job(created["job_id"]) is None

    def test_delete_job_not_found_raises_value_error(self, service):
        """delete_job raises ValueError for unknown ID."""
        with pytest.raises(ValueError, match="not found"):
            service.delete_job("nonexistent-id")


# ===================================================================
# 4. Source Management Tests
# ===================================================================


class TestSourceManagement:
    """Tests for register_source, list_sources, get_source, update_source."""

    def test_register_source_returns_dict(self, service):
        """register_source returns a dict with source_id."""
        result = service.register_source(name="SAP ERP")
        assert isinstance(result, dict)
        assert "source_id" in result
        assert result["name"] == "SAP ERP"
        assert result["status"] == "active"

    def test_register_source_default_type(self, service):
        """register_source defaults to manual type."""
        result = service.register_source(name="My Source")
        assert result["source_type"] == "manual"

    def test_register_source_custom_priority(self, service):
        """register_source respects custom priority."""
        result = service.register_source(name="High Priority", priority=1)
        assert result["priority"] == 1

    def test_register_source_credibility_score(self, service):
        """register_source stores credibility_score."""
        result = service.register_source(
            name="Meter", credibility_score=0.95,
        )
        assert result["credibility_score"] == 0.95

    def test_register_source_increments_stats(self, service):
        """register_source increments total_sources counter."""
        assert service._stats["total_sources"] == 0
        service.register_source(name="Test Source")
        assert service._stats["total_sources"] == 1

    def test_register_source_has_provenance_hash(self, service):
        """register_source result includes provenance_hash."""
        result = service.register_source(name="Test")
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_list_sources_returns_all(self, service):
        """list_sources returns all registered sources."""
        service.register_source(name="Source A")
        service.register_source(name="Source B")
        result = service.list_sources()
        assert result["total"] == 2
        assert result["count"] == 2

    def test_list_sources_pagination(self, service):
        """list_sources supports limit and offset."""
        for i in range(5):
            service.register_source(name=f"Source-{i}")
        result = service.list_sources(limit=2, offset=1)
        assert result["count"] == 2
        assert result["total"] == 5

    def test_get_source_found(self, service):
        """get_source returns source for known ID."""
        created = service.register_source(name="Find Me")
        fetched = service.get_source(created["source_id"])
        assert fetched is not None
        assert fetched["name"] == "Find Me"

    def test_get_source_not_found(self, service):
        """get_source returns None for unknown ID."""
        result = service.get_source("nonexistent-id")
        assert result is None

    def test_update_source_name(self, service):
        """update_source updates the source name."""
        created = service.register_source(name="Old Name")
        updated = service.update_source(
            source_id=created["source_id"], name="New Name",
        )
        assert updated["name"] == "New Name"

    def test_update_source_priority(self, service):
        """update_source updates priority."""
        created = service.register_source(name="Source", priority=5)
        updated = service.update_source(
            source_id=created["source_id"], priority=1,
        )
        assert updated["priority"] == 1

    def test_update_source_credibility_score(self, service):
        """update_source updates credibility_score."""
        created = service.register_source(name="Source", credibility_score=0.8)
        updated = service.update_source(
            source_id=created["source_id"], credibility_score=0.95,
        )
        assert updated["credibility_score"] == 0.95

    def test_update_source_not_found_raises(self, service):
        """update_source raises ValueError for unknown ID."""
        with pytest.raises(ValueError, match="not found"):
            service.update_source(source_id="nonexistent", name="X")

    def test_update_source_updates_provenance_hash(self, service):
        """update_source generates a new provenance_hash."""
        created = service.register_source(name="Source")
        old_hash = created["provenance_hash"]
        updated = service.update_source(
            source_id=created["source_id"], name="Updated",
        )
        assert updated["provenance_hash"] != old_hash

    def test_update_source_merges_metadata(self, service):
        """update_source merges new metadata into existing."""
        created = service.register_source(
            name="Source", metadata={"key1": "val1"},
        )
        updated = service.update_source(
            source_id=created["source_id"],
            metadata={"key2": "val2"},
        )
        assert updated["metadata"]["key1"] == "val1"
        assert updated["metadata"]["key2"] == "val2"


# ===================================================================
# 5. Record Matching Tests
# ===================================================================


class TestRecordMatching:
    """Tests for match_records, list_matches, get_match."""

    def test_match_records_returns_dict(self, service):
        """match_records returns a dict with match_id."""
        result = service.match_records(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
        )
        assert isinstance(result, dict)
        assert "match_id" in result
        assert "total_matched" in result

    def test_match_records_exact_keys(self, service):
        """match_records with default keys matches on entity_id and period."""
        result = service.match_records(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        # facility-001/Q1 and facility-002/Q1 should match
        assert result["total_matched"] == 2

    def test_match_records_no_overlap(self, service):
        """match_records with no overlap returns zero matched."""
        result = service.match_records(
            records_a=[{"entity_id": "x", "period": "Q1"}],
            records_b=[{"entity_id": "y", "period": "Q2"}],
        )
        assert result["total_matched"] == 0

    def test_match_records_increments_stats(self, service):
        """match_records increments total_matches counter."""
        assert service._stats["total_matches"] == 0
        service.match_records(records_a=[], records_b=[])
        assert service._stats["total_matches"] == 1

    def test_match_records_has_provenance_hash(self, service):
        """match_records result includes provenance_hash."""
        result = service.match_records(records_a=[], records_b=[])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_match_records_stores_result(self, service):
        """match_records stores result in matches dict."""
        result = service.match_records(records_a=[], records_b=[])
        assert result["match_id"] in service._matches

    def test_match_records_includes_processing_time(self, service):
        """match_records result includes processing_time_ms."""
        result = service.match_records(records_a=[], records_b=[])
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_list_matches_returns_all(self, service):
        """list_matches returns all stored matches."""
        service.match_records(records_a=[], records_b=[])
        service.match_records(records_a=[], records_b=[])
        result = service.list_matches()
        assert result["total"] == 2

    def test_list_matches_pagination(self, service):
        """list_matches supports limit and offset."""
        for _ in range(5):
            service.match_records(records_a=[], records_b=[])
        result = service.list_matches(limit=2, offset=1)
        assert result["count"] == 2
        assert result["total"] == 5

    def test_get_match_found(self, service):
        """get_match returns match for known ID."""
        created = service.match_records(records_a=[], records_b=[])
        fetched = service.get_match(created["match_id"])
        assert fetched is not None

    def test_get_match_not_found(self, service):
        """get_match returns None for unknown ID."""
        result = service.get_match("nonexistent-id")
        assert result is None

    def test_match_records_unmatched_counts(self, service):
        """match_records reports unmatched counts correctly."""
        result = service.match_records(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        # 3 in A, 2 matched => 1 unmatched_a
        assert result["total_unmatched_a"] == 1
        # 3 in B, 2 matched => 1 unmatched_b
        assert result["total_unmatched_b"] == 1


# ===================================================================
# 6. Record Comparison Tests
# ===================================================================


class TestRecordComparison:
    """Tests for compare_records and fallback comparison logic."""

    def test_compare_records_returns_dict(self, service):
        """compare_records returns a dict with comparison_id."""
        result = service.compare_records(
            record_a={"electricity_kwh": 12500.0},
            record_b={"electricity_kwh": 12650.0},
            fields=["electricity_kwh"],
        )
        assert isinstance(result, dict)
        assert "comparison_id" in result

    def test_compare_records_numeric_match_within_tolerance(self, service):
        """compare_records classifies within-tolerance as match."""
        result = service.compare_records(
            record_a={"val": 100.0},
            record_b={"val": 104.0},
            fields=["val"],
            tolerance_pct=5.0,
        )
        field_result = result["fields_compared"][0]
        assert field_result["result"] == "match"

    def test_compare_records_numeric_mismatch_beyond_tolerance(self, service):
        """compare_records classifies beyond-tolerance as mismatch."""
        result = service.compare_records(
            record_a={"val": 100.0},
            record_b={"val": 200.0},
            fields=["val"],
            tolerance_pct=5.0,
            tolerance_abs=0.01,
        )
        field_result = result["fields_compared"][0]
        assert field_result["result"] == "mismatch"

    def test_compare_records_string_match(self, service):
        """compare_records matches strings case-insensitively."""
        result = service.compare_records(
            record_a={"name": "Facility A"},
            record_b={"name": "facility a"},
            fields=["name"],
        )
        field_result = result["fields_compared"][0]
        assert field_result["result"] == "match"

    def test_compare_records_string_mismatch(self, service):
        """compare_records detects string mismatches."""
        result = service.compare_records(
            record_a={"name": "Facility A"},
            record_b={"name": "Facility B"},
            fields=["name"],
        )
        field_result = result["fields_compared"][0]
        assert field_result["result"] == "mismatch"

    def test_compare_records_missing_field(self, service):
        """compare_records classifies missing values correctly."""
        result = service.compare_records(
            record_a={"val": 100.0},
            record_b={"val": None},
            fields=["val"],
        )
        field_result = result["fields_compared"][0]
        assert field_result["result"] == "missing"

    def test_compare_records_increments_stats(self, service):
        """compare_records increments total_comparisons counter."""
        assert service._stats["total_comparisons"] == 0
        service.compare_records(record_a={}, record_b={})
        assert service._stats["total_comparisons"] == 1

    def test_compare_records_match_rate(self, service):
        """compare_records calculates match_rate correctly."""
        result = service.compare_records(
            record_a={"a": 1, "b": "hello"},
            record_b={"a": 1, "b": "hello"},
            fields=["a", "b"],
        )
        assert result["match_rate"] == 1.0

    def test_compare_records_auto_detect_shared_fields(self, service):
        """compare_records auto-detects shared fields when fields=None."""
        result = service.compare_records(
            record_a={"x": 1, "y": 2, "z": 3},
            record_b={"x": 1, "y": 2, "w": 4},
        )
        # x and y are shared, z and w are not
        assert result["total_fields"] == 2


# ===================================================================
# 7. Discrepancy Detection Tests
# ===================================================================


class TestDiscrepancyDetection:
    """Tests for detect_discrepancies and list/get discrepancy."""

    def test_detect_discrepancies_returns_dict(self, service):
        """detect_discrepancies returns a dict with detection_id."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "val", "result": "mismatch", "value_a": 100, "value_b": 200, "rel_diff_pct": 50.0},
            ],
        )
        assert isinstance(result, dict)
        assert "detection_id" in result
        assert result["total_discrepancies"] >= 1

    def test_detect_discrepancies_no_mismatches(self, service):
        """detect_discrepancies returns zero for all matching fields."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "val", "result": "match", "value_a": 100, "value_b": 100},
            ],
        )
        assert result["total_discrepancies"] == 0

    def test_detect_discrepancies_severity_classification(self, service):
        """detect_discrepancies classifies severity by relative difference."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "high_diff", "result": "mismatch", "value_a": 100, "value_b": 200, "rel_diff_pct": 60.0},
                {"field": "low_diff", "result": "mismatch", "value_a": 100, "value_b": 105, "rel_diff_pct": 5.0},
            ],
        )
        severities = [d["severity"] for d in result["discrepancies"]]
        assert "critical" in severities

    def test_detect_discrepancies_missing_type(self, service):
        """detect_discrepancies classifies missing fields as missing_in_source."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "missing_field", "result": "missing", "value_a": 100, "value_b": None},
            ],
        )
        disc_types = [d["type"] for d in result["discrepancies"]]
        assert "missing_in_source" in disc_types

    def test_detect_discrepancies_stores_individually(self, service):
        """detect_discrepancies stores each discrepancy by its ID."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "a", "result": "mismatch", "value_a": 1, "value_b": 2, "rel_diff_pct": 50.0},
                {"field": "b", "result": "mismatch", "value_a": 3, "value_b": 4, "rel_diff_pct": 25.0},
            ],
        )
        for disc in result["discrepancies"]:
            assert disc["discrepancy_id"] in service._discrepancies

    def test_detect_discrepancies_increments_stats(self, service):
        """detect_discrepancies increments total_discrepancies counter."""
        assert service._stats["total_discrepancies"] == 0
        service.detect_discrepancies(
            comparisons=[
                {"field": "x", "result": "mismatch", "value_a": 1, "value_b": 2, "rel_diff_pct": 50.0},
            ],
        )
        assert service._stats["total_discrepancies"] == 1

    def test_list_discrepancies_returns_all(self, service):
        """list_discrepancies returns all stored discrepancies."""
        service.detect_discrepancies(
            comparisons=[
                {"field": "a", "result": "mismatch", "value_a": 1, "value_b": 2, "rel_diff_pct": 50.0},
                {"field": "b", "result": "mismatch", "value_a": 3, "value_b": 4, "rel_diff_pct": 30.0},
            ],
        )
        result = service.list_discrepancies()
        assert result["total"] == 2

    def test_list_discrepancies_filter_by_severity(self, service):
        """list_discrepancies filters by severity."""
        service.detect_discrepancies(
            comparisons=[
                {"field": "a", "result": "mismatch", "value_a": 1, "value_b": 2, "rel_diff_pct": 60.0},
                {"field": "b", "result": "mismatch", "value_a": 3, "value_b": 4, "rel_diff_pct": 3.0},
            ],
        )
        # rel_diff 60% => critical, rel_diff 3% => info
        critical = service.list_discrepancies(severity="critical")
        assert critical["total"] >= 1

    def test_get_discrepancy_found(self, service):
        """get_discrepancy returns discrepancy for known ID."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "x", "result": "mismatch", "value_a": 1, "value_b": 2, "rel_diff_pct": 50.0},
            ],
        )
        disc_id = result["discrepancies"][0]["discrepancy_id"]
        fetched = service.get_discrepancy(disc_id)
        assert fetched is not None

    def test_get_discrepancy_not_found(self, service):
        """get_discrepancy returns None for unknown ID."""
        result = service.get_discrepancy("nonexistent-id")
        assert result is None


# ===================================================================
# 8. Resolution Tests
# ===================================================================


class TestResolution:
    """Tests for resolve_discrepancies."""

    def _create_discrepancies(self, service):
        """Helper: create some discrepancies and return their IDs."""
        result = service.detect_discrepancies(
            comparisons=[
                {"field": "electricity_kwh", "result": "mismatch",
                 "value_a": 12500.0, "value_b": 12650.0, "rel_diff_pct": 1.2},
                {"field": "gas_m3", "result": "mismatch",
                 "value_a": 3400.0, "value_b": 3600.0, "rel_diff_pct": 5.6},
            ],
        )
        return [d["discrepancy_id"] for d in result["discrepancies"]]

    def test_resolve_priority_wins(self, service):
        """resolve_discrepancies with priority_wins picks value_a."""
        disc_ids = self._create_discrepancies(service)
        result = service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="priority_wins",
        )
        assert result["total_resolved"] == 2
        for res in result["resolutions"]:
            assert res["winning_source"] == "source_a"

    def test_resolve_most_recent(self, service):
        """resolve_discrepancies with most_recent picks value_b."""
        disc_ids = self._create_discrepancies(service)
        result = service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="most_recent",
        )
        for res in result["resolutions"]:
            assert res["winning_source"] == "source_b"

    def test_resolve_weighted_average(self, service):
        """resolve_discrepancies with weighted_average averages numeric values."""
        disc_ids = self._create_discrepancies(service)
        result = service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="weighted_average",
        )
        for res in result["resolutions"]:
            assert res["winning_source"] == "weighted_average"
            assert res["resolved_value"] == pytest.approx(
                (res["original_value_a"] + res["original_value_b"]) / 2.0,
            )

    def test_resolve_manual_override(self, service):
        """resolve_discrepancies with manual_override uses manual values."""
        disc_ids = self._create_discrepancies(service)
        result = service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="manual_override",
            manual_values={"electricity_kwh": 12575.0, "gas_m3": 3500.0},
        )
        for res in result["resolutions"]:
            assert res["winning_source"] == "manual"

    def test_resolve_marks_discrepancies_resolved(self, service):
        """resolve_discrepancies marks each discrepancy as resolved."""
        disc_ids = self._create_discrepancies(service)
        service.resolve_discrepancies(discrepancy_ids=disc_ids)
        for disc_id in disc_ids:
            disc = service.get_discrepancy(disc_id)
            assert disc["status"] == "resolved"

    def test_resolve_increments_stats(self, service):
        """resolve_discrepancies increments total_resolutions counter."""
        disc_ids = self._create_discrepancies(service)
        assert service._stats["total_resolutions"] == 0
        service.resolve_discrepancies(discrepancy_ids=disc_ids)
        assert service._stats["total_resolutions"] == 2

    def test_resolve_has_provenance_hash(self, service):
        """resolve_discrepancies result includes provenance_hash."""
        disc_ids = self._create_discrepancies(service)
        result = service.resolve_discrepancies(discrepancy_ids=disc_ids)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_resolve_empty_ids(self, service):
        """resolve_discrepancies with empty list resolves zero."""
        result = service.resolve_discrepancies(discrepancy_ids=[])
        assert result["total_resolved"] == 0

    def test_resolve_unknown_ids_skipped(self, service):
        """resolve_discrepancies skips unknown discrepancy IDs."""
        result = service.resolve_discrepancies(
            discrepancy_ids=["nonexistent-1", "nonexistent-2"],
        )
        assert result["total_resolved"] == 0


# ===================================================================
# 9. Golden Records Tests
# ===================================================================


class TestGoldenRecords:
    """Tests for get_golden_records, get_golden_record."""

    def test_get_golden_records_empty(self, service):
        """get_golden_records returns empty list when none exist."""
        result = service.get_golden_records()
        assert result["count"] == 0
        assert result["total"] == 0

    def test_get_golden_records_after_pipeline(self, service):
        """get_golden_records returns records after pipeline execution."""
        service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        result = service.get_golden_records()
        assert result["total"] >= 1

    def test_get_golden_record_by_id(self, service):
        """get_golden_record returns a record by known ID."""
        service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        all_records = service.get_golden_records()
        if all_records["count"] > 0:
            record_id = all_records["golden_records"][0]["record_id"]
            fetched = service.get_golden_record(record_id)
            assert fetched is not None
            assert fetched["record_id"] == record_id

    def test_get_golden_record_not_found(self, service):
        """get_golden_record returns None for unknown ID."""
        result = service.get_golden_record("nonexistent-id")
        assert result is None

    def test_golden_records_pagination(self, service):
        """get_golden_records supports limit and offset."""
        # Run pipeline to create golden records
        service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        result = service.get_golden_records(limit=1, offset=0)
        assert result["count"] <= 1


# ===================================================================
# 10. Pipeline Tests
# ===================================================================


class TestPipeline:
    """Tests for run_pipeline end-to-end."""

    def test_run_pipeline_returns_complete_result(self, service):
        """run_pipeline returns a dict with all pipeline stages."""
        result = service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        assert isinstance(result, dict)
        assert "pipeline_id" in result
        assert "match_result" in result
        assert "discrepancy_result" in result
        assert "resolution_result" in result
        assert "golden_records" in result
        assert result["status"] == "completed"

    def test_run_pipeline_matching_phase(self, service):
        """run_pipeline performs record matching."""
        result = service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
        )
        assert result["match_result"]["total_matched"] == 2

    def test_run_pipeline_golden_records_created(self, service):
        """run_pipeline creates golden records when requested."""
        result = service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
            generate_golden_records=True,
        )
        assert result["golden_record_count"] >= 1

    def test_run_pipeline_no_golden_records(self, service):
        """run_pipeline skips golden records when disabled."""
        result = service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
            generate_golden_records=False,
        )
        assert result["golden_record_count"] == 0

    def test_run_pipeline_empty_records(self, service):
        """run_pipeline handles empty record sets gracefully."""
        result = service.run_pipeline(
            records_a=[],
            records_b=[],
        )
        assert result["status"] == "completed"
        assert result["match_result"]["total_matched"] == 0

    def test_run_pipeline_increments_stats(self, service):
        """run_pipeline increments total_pipelines counter."""
        assert service._stats["total_pipelines"] == 0
        service.run_pipeline(records_a=[], records_b=[])
        assert service._stats["total_pipelines"] == 1

    def test_run_pipeline_has_provenance_hash(self, service):
        """run_pipeline result includes provenance_hash."""
        result = service.run_pipeline(records_a=[], records_b=[])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_run_pipeline_processing_time(self, service):
        """run_pipeline result includes total_processing_time_ms."""
        result = service.run_pipeline(records_a=[], records_b=[])
        assert "total_processing_time_ms" in result
        assert result["total_processing_time_ms"] >= 0

    def test_run_pipeline_stores_result(self, service):
        """run_pipeline stores the result in pipeline_results."""
        result = service.run_pipeline(records_a=[], records_b=[])
        assert result["pipeline_id"] in service._pipeline_results

    def test_run_pipeline_resolution_strategy(self, service):
        """run_pipeline uses the specified resolution strategy."""
        result = service.run_pipeline(
            records_a=_sample_records_a(),
            records_b=_sample_records_b(),
            match_keys=["entity_id", "period"],
            resolution_strategy="most_recent",
        )
        assert result["resolution_result"]["strategy"] == "most_recent"
