# -*- coding: utf-8 -*-
"""
Unit tests for Data Freshness Monitor Service Setup - AGENT-DATA-016

Tests DataFreshnessMonitorService facade, lifecycle management,
dataset CRUD, SLA CRUD, freshness checking, batch checking, breach
management, alert listing, predictions, pipeline execution, health/stats,
singleton pattern, and FastAPI integration helpers.

Target: 80+ tests at 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

import importlib
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_freshness_monitor.setup import (
    DataFreshnessMonitorService,
    _compute_hash,
    configure_freshness_monitor,
    get_freshness_monitor,
    get_router,
    get_service,
    reset_service,
    set_service,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def service() -> DataFreshnessMonitorService:
    """Return a fresh DataFreshnessMonitorService instance."""
    return DataFreshnessMonitorService()


@pytest.fixture()
def started_service() -> DataFreshnessMonitorService:
    """Return a started DataFreshnessMonitorService instance."""
    svc = DataFreshnessMonitorService()
    svc.startup()
    return svc


@pytest.fixture()
def service_with_config() -> DataFreshnessMonitorService:
    """Return a service initialized with custom config."""
    return DataFreshnessMonitorService(config={"env": "test", "region": "us-east-1"})


@pytest.fixture()
def dataset_id(started_service: DataFreshnessMonitorService) -> str:
    """Register a dataset and return its ID."""
    result = started_service.register_dataset(
        name="test-dataset",
        source="erp",
        owner="qa-team",
        refresh_cadence="hourly",
        priority=3,
        tags=["scope1", "energy"],
        metadata={"region": "EU"},
    )
    return result["dataset_id"]


@pytest.fixture()
def sla_id(started_service: DataFreshnessMonitorService, dataset_id: str) -> str:
    """Create an SLA and return its ID."""
    result = started_service.create_sla(
        dataset_id=dataset_id,
        name="test-sla",
        warning_hours=12.0,
        critical_hours=48.0,
        severity="high",
    )
    return result["sla_id"]


@pytest.fixture()
def _reset_singleton():
    """Reset the module-level singleton after the test."""
    import greenlang.data_freshness_monitor.setup as setup_mod

    original = setup_mod._service_instance
    yield
    setup_mod._service_instance = original


# ===================================================================
# TestDataFreshnessMonitorService -- Instantiation
# ===================================================================


class TestServiceInstantiation:
    """Tests for DataFreshnessMonitorService.__init__."""

    def test_default_instantiation(self, service: DataFreshnessMonitorService):
        """Service creates with empty config dict."""
        assert service.config == {}

    def test_custom_config(self, service_with_config: DataFreshnessMonitorService):
        """Service stores provided config."""
        assert service_with_config.config["env"] == "test"
        assert service_with_config.config["region"] == "us-east-1"

    def test_provenance_tracker_initialized(self, service: DataFreshnessMonitorService):
        """Provenance tracker is set on construction."""
        assert service._provenance is not None

    def test_engines_are_none_before_startup(self, service: DataFreshnessMonitorService):
        """All engine slots are None before startup()."""
        assert service._dataset_registry is None
        assert service._sla_definition is None
        assert service._freshness_checker is None
        assert service._staleness_detector is None
        assert service._refresh_predictor is None
        assert service._alert_manager is None
        assert service._pipeline is None

    def test_started_flag_false(self, service: DataFreshnessMonitorService):
        """Service is not started immediately after construction."""
        assert service._started is False

    def test_stores_empty_at_init(self, service: DataFreshnessMonitorService):
        """All in-memory stores are empty at construction."""
        assert len(service._datasets) == 0
        assert len(service._sla_definitions) == 0
        assert len(service._checks) == 0
        assert len(service._breaches) == 0
        assert len(service._alerts) == 0
        assert len(service._predictions) == 0
        assert len(service._pipeline_results) == 0

    def test_stats_counters_zero(self, service: DataFreshnessMonitorService):
        """All aggregate counters start at zero."""
        for key, value in service._stats.items():
            assert value == 0, f"Expected {key} to be 0, got {value}"

    def test_none_config_defaults_to_empty(self):
        """Passing None config produces empty dict."""
        svc = DataFreshnessMonitorService(config=None)
        assert svc.config == {}


# ===================================================================
# TestServiceLifecycle
# ===================================================================


class TestServiceLifecycle:
    """Tests for startup() and shutdown()."""

    def test_startup_sets_started(self, service: DataFreshnessMonitorService):
        """startup() sets _started to True."""
        service.startup()
        assert service._started is True

    def test_shutdown_clears_started(self, started_service: DataFreshnessMonitorService):
        """shutdown() sets _started to False."""
        assert started_service._started is True
        started_service.shutdown()
        assert started_service._started is False

    def test_double_startup(self, service: DataFreshnessMonitorService):
        """Calling startup() twice does not raise."""
        service.startup()
        service.startup()
        assert service._started is True

    def test_shutdown_then_startup(self, started_service: DataFreshnessMonitorService):
        """Service can be restarted after shutdown."""
        started_service.shutdown()
        started_service.startup()
        assert started_service._started is True

    def test_startup_initializes_engines_when_available(
        self, service: DataFreshnessMonitorService,
    ):
        """startup() tries to instantiate each engine class."""
        service.startup()
        # Engines may or may not be available depending on imports.
        # The important thing is startup() does not raise.
        assert service._started is True


# ===================================================================
# TestDatasetManagement
# ===================================================================


class TestDatasetManagement:
    """Tests for register, list, get, update, delete dataset methods."""

    def test_register_dataset_returns_dict(
        self, started_service: DataFreshnessMonitorService,
    ):
        """register_dataset returns a dictionary with required keys."""
        result = started_service.register_dataset(name="ds1")
        assert isinstance(result, dict)
        assert "dataset_id" in result
        assert result["name"] == "ds1"

    def test_register_dataset_defaults(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Default values are applied when optional args omitted."""
        result = started_service.register_dataset(name="ds-default")
        assert result["source"] == ""
        assert result["owner"] == ""
        assert result["refresh_cadence"] == "daily"
        assert result["priority"] == 5
        assert result["tags"] == []
        assert result["metadata"] == {}
        assert result["status"] == "active"

    def test_register_dataset_custom_fields(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Custom arguments are stored correctly."""
        result = started_service.register_dataset(
            name="custom",
            source="api",
            owner="data-eng",
            refresh_cadence="weekly",
            priority=2,
            tags=["scope3"],
            metadata={"k": "v"},
        )
        assert result["source"] == "api"
        assert result["owner"] == "data-eng"
        assert result["refresh_cadence"] == "weekly"
        assert result["priority"] == 2
        assert result["tags"] == ["scope3"]
        assert result["metadata"] == {"k": "v"}

    def test_register_dataset_provenance_hash(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Registered dataset has a 64-char provenance hash."""
        result = started_service.register_dataset(name="prov")
        assert len(result["provenance_hash"]) == 64

    def test_register_dataset_increments_stats(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Registering a dataset increments total_datasets counter."""
        started_service.register_dataset(name="ds-a")
        started_service.register_dataset(name="ds-b")
        assert started_service._stats["total_datasets"] == 2

    def test_register_dataset_unique_ids(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Each registration produces a unique dataset_id."""
        ids = {
            started_service.register_dataset(name=f"ds-{i}")["dataset_id"]
            for i in range(5)
        }
        assert len(ids) == 5

    def test_list_datasets_empty(self, started_service: DataFreshnessMonitorService):
        """list_datasets returns empty set when none registered."""
        result = started_service.list_datasets()
        assert result["datasets"] == []
        assert result["total"] == 0

    def test_list_datasets_returns_registered(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_datasets includes previously registered datasets."""
        started_service.register_dataset(name="ds1")
        started_service.register_dataset(name="ds2")
        result = started_service.list_datasets()
        assert result["total"] == 2
        assert result["count"] == 2

    def test_list_datasets_status_filter(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_datasets filters by status."""
        ds = started_service.register_dataset(name="ds-filter")
        started_service.update_dataset(ds["dataset_id"], status="archived")
        result = started_service.list_datasets(status="archived")
        assert result["total"] == 1

    def test_list_datasets_source_filter(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_datasets filters by source."""
        started_service.register_dataset(name="ds-erp", source="erp")
        started_service.register_dataset(name="ds-api", source="api")
        result = started_service.list_datasets(source="erp")
        assert result["total"] == 1
        assert result["datasets"][0]["source"] == "erp"

    def test_list_datasets_pagination_limit(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_datasets respects limit parameter."""
        for i in range(5):
            started_service.register_dataset(name=f"ds-{i}")
        result = started_service.list_datasets(limit=2)
        assert result["count"] == 2
        assert result["total"] == 5
        assert result["limit"] == 2

    def test_list_datasets_pagination_offset(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_datasets respects offset parameter."""
        for i in range(5):
            started_service.register_dataset(name=f"ds-{i}")
        result = started_service.list_datasets(limit=2, offset=3)
        assert result["count"] == 2
        assert result["offset"] == 3

    def test_get_dataset_found(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """get_dataset returns dataset when it exists."""
        ds = started_service.get_dataset(dataset_id)
        assert ds is not None
        assert ds["dataset_id"] == dataset_id

    def test_get_dataset_not_found(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_dataset returns None for unknown ID."""
        assert started_service.get_dataset("nonexistent") is None

    def test_update_dataset_name(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset changes the name."""
        result = started_service.update_dataset(dataset_id, name="renamed")
        assert result["name"] == "renamed"

    def test_update_dataset_source(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset changes the source."""
        result = started_service.update_dataset(dataset_id, source="new-src")
        assert result["source"] == "new-src"

    def test_update_dataset_owner(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset changes the owner."""
        result = started_service.update_dataset(dataset_id, owner="new-owner")
        assert result["owner"] == "new-owner"

    def test_update_dataset_refresh_cadence(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset changes refresh_cadence."""
        result = started_service.update_dataset(dataset_id, refresh_cadence="monthly")
        assert result["refresh_cadence"] == "monthly"

    def test_update_dataset_priority(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset changes the priority."""
        result = started_service.update_dataset(dataset_id, priority=1)
        assert result["priority"] == 1

    def test_update_dataset_tags(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset replaces tags."""
        result = started_service.update_dataset(dataset_id, tags=["new-tag"])
        assert result["tags"] == ["new-tag"]

    def test_update_dataset_metadata_merges(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset merges metadata into existing."""
        started_service.update_dataset(dataset_id, metadata={"a": 1})
        result = started_service.update_dataset(dataset_id, metadata={"b": 2})
        assert result["metadata"]["a"] == 1
        assert result["metadata"]["b"] == 2

    def test_update_dataset_status(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset changes the status."""
        result = started_service.update_dataset(dataset_id, status="inactive")
        assert result["status"] == "inactive"

    def test_update_dataset_not_found_raises(
        self, started_service: DataFreshnessMonitorService,
    ):
        """update_dataset raises ValueError for unknown ID."""
        with pytest.raises(ValueError, match="not found"):
            started_service.update_dataset("missing-id", name="x")

    def test_update_dataset_provenance_hash_changes(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset recomputes the provenance hash."""
        original = started_service.get_dataset(dataset_id)["provenance_hash"]
        result = started_service.update_dataset(dataset_id, name="changed")
        assert result["provenance_hash"] != original

    def test_update_dataset_updated_at(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_dataset refreshes the updated_at timestamp."""
        original = started_service.get_dataset(dataset_id)["updated_at"]
        result = started_service.update_dataset(dataset_id, name="ts-check")
        assert result["updated_at"] >= original

    def test_delete_dataset(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """delete_dataset removes the dataset from the store."""
        result = started_service.delete_dataset(dataset_id)
        assert result["status"] == "deleted"
        assert result["dataset_id"] == dataset_id
        assert started_service.get_dataset(dataset_id) is None

    def test_delete_dataset_not_found_raises(
        self, started_service: DataFreshnessMonitorService,
    ):
        """delete_dataset raises ValueError for unknown ID."""
        with pytest.raises(ValueError, match="not found"):
            started_service.delete_dataset("nonexistent")


# ===================================================================
# TestSLAManagement
# ===================================================================


class TestSLAManagement:
    """Tests for create, list, get, update SLA methods."""

    def test_create_sla_returns_dict(
        self, started_service: DataFreshnessMonitorService,
    ):
        """create_sla returns a dictionary with required keys."""
        result = started_service.create_sla(name="sla-1")
        assert isinstance(result, dict)
        assert "sla_id" in result
        assert result["name"] == "sla-1"

    def test_create_sla_defaults(
        self, started_service: DataFreshnessMonitorService,
    ):
        """create_sla applies correct default values."""
        result = started_service.create_sla()
        assert result["dataset_id"] == ""
        assert result["warning_hours"] == 24.0
        assert result["critical_hours"] == 72.0
        assert result["severity"] == "high"
        assert result["status"] == "active"

    def test_create_sla_auto_name(
        self, started_service: DataFreshnessMonitorService,
    ):
        """When name is empty, an auto-generated name is used."""
        result = started_service.create_sla()
        assert result["name"].startswith("sla-")

    def test_create_sla_custom_values(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """create_sla stores all custom arguments."""
        result = started_service.create_sla(
            dataset_id=dataset_id,
            name="custom-sla",
            warning_hours=6.0,
            critical_hours=24.0,
            severity="critical",
            escalation_policy={"notify": "ops"},
            metadata={"env": "prod"},
        )
        assert result["dataset_id"] == dataset_id
        assert result["warning_hours"] == 6.0
        assert result["critical_hours"] == 24.0
        assert result["severity"] == "critical"
        assert result["escalation_policy"] == {"notify": "ops"}
        assert result["metadata"] == {"env": "prod"}

    def test_create_sla_increments_stats(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Creating SLAs increments total_sla_definitions counter."""
        started_service.create_sla(name="s1")
        started_service.create_sla(name="s2")
        assert started_service._stats["total_sla_definitions"] == 2

    def test_create_sla_provenance_hash(
        self, started_service: DataFreshnessMonitorService,
    ):
        """SLA has a 64-char provenance hash."""
        result = started_service.create_sla(name="hash-test")
        assert len(result["provenance_hash"]) == 64

    def test_list_slas_empty(self, started_service: DataFreshnessMonitorService):
        """list_slas returns empty list when none created."""
        result = started_service.list_slas()
        assert result["sla_definitions"] == []
        assert result["total"] == 0

    def test_list_slas_returns_created(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_slas returns previously created SLAs."""
        started_service.create_sla(name="sla-a")
        started_service.create_sla(name="sla-b")
        result = started_service.list_slas()
        assert result["total"] == 2

    def test_list_slas_dataset_filter(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_slas filters by dataset_id."""
        started_service.create_sla(dataset_id=dataset_id, name="linked")
        started_service.create_sla(dataset_id="other", name="unlinked")
        result = started_service.list_slas(dataset_id=dataset_id)
        assert result["total"] == 1

    def test_list_slas_pagination(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_slas respects limit and offset."""
        for i in range(5):
            started_service.create_sla(name=f"sla-{i}")
        result = started_service.list_slas(limit=2, offset=1)
        assert result["count"] == 2
        assert result["offset"] == 1
        assert result["total"] == 5

    def test_get_sla_found(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """get_sla returns SLA when it exists."""
        sla = started_service.get_sla(sla_id)
        assert sla is not None
        assert sla["sla_id"] == sla_id

    def test_get_sla_not_found(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_sla returns None for unknown ID."""
        assert started_service.get_sla("nonexistent") is None

    def test_update_sla_name(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla changes the name."""
        result = started_service.update_sla(sla_id, name="new-name")
        assert result["name"] == "new-name"

    def test_update_sla_warning_hours(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla changes warning_hours."""
        result = started_service.update_sla(sla_id, warning_hours=8.0)
        assert result["warning_hours"] == 8.0

    def test_update_sla_critical_hours(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla changes critical_hours."""
        result = started_service.update_sla(sla_id, critical_hours=100.0)
        assert result["critical_hours"] == 100.0

    def test_update_sla_severity(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla changes severity."""
        result = started_service.update_sla(sla_id, severity="low")
        assert result["severity"] == "low"

    def test_update_sla_escalation_policy(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla changes escalation_policy."""
        policy = {"level1": "email", "level2": "page"}
        result = started_service.update_sla(sla_id, escalation_policy=policy)
        assert result["escalation_policy"] == policy

    def test_update_sla_metadata_merges(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla merges metadata."""
        started_service.update_sla(sla_id, metadata={"a": 1})
        result = started_service.update_sla(sla_id, metadata={"b": 2})
        assert result["metadata"]["a"] == 1
        assert result["metadata"]["b"] == 2

    def test_update_sla_status(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla changes status."""
        result = started_service.update_sla(sla_id, status="inactive")
        assert result["status"] == "inactive"

    def test_update_sla_not_found_raises(
        self, started_service: DataFreshnessMonitorService,
    ):
        """update_sla raises ValueError for unknown ID."""
        with pytest.raises(ValueError, match="not found"):
            started_service.update_sla("nonexistent", name="x")

    def test_update_sla_provenance_hash_changes(
        self,
        started_service: DataFreshnessMonitorService,
        sla_id: str,
    ):
        """update_sla recomputes provenance hash."""
        original = started_service.get_sla(sla_id)["provenance_hash"]
        result = started_service.update_sla(sla_id, name="hash-change")
        assert result["provenance_hash"] != original


# ===================================================================
# TestFreshnessChecking
# ===================================================================


class TestFreshnessChecking:
    """Tests for run_check, run_batch_check, and list_checks."""

    def test_run_check_valid_dataset(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check returns check result for known dataset."""
        result = started_service.run_check(dataset_id=dataset_id)
        assert "check_id" in result
        assert result["dataset_id"] == dataset_id
        assert "freshness_score" in result
        assert "freshness_level" in result
        assert "sla_status" in result
        assert "processing_time_ms" in result

    def test_run_check_unknown_dataset_raises(
        self, started_service: DataFreshnessMonitorService,
    ):
        """run_check raises ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="not found"):
            started_service.run_check(dataset_id="nonexistent")

    def test_run_check_with_last_refreshed_at(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check accepts external last_refreshed_at timestamp."""
        ts = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["age_hours"] >= 0.0

    def test_run_check_freshness_excellent(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """Recent refresh yields 'excellent' freshness."""
        ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["freshness_level"] == "excellent"
        assert result["freshness_score"] == 1.0

    def test_run_check_freshness_good(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """2-hour age yields 'good' freshness."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["freshness_level"] == "good"
        assert result["freshness_score"] == 0.8

    def test_run_check_freshness_fair(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """10-hour age yields 'fair' freshness."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["freshness_level"] == "fair"
        assert result["freshness_score"] == 0.6

    def test_run_check_freshness_poor(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """30-hour age yields 'poor' freshness."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["freshness_level"] == "poor"
        assert result["freshness_score"] == 0.4

    def test_run_check_freshness_stale(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """100-hour age yields 'stale' freshness."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["freshness_level"] == "stale"
        assert result["freshness_score"] == 0.2

    def test_run_check_no_refresh_time(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check handles None last_refreshed_at gracefully."""
        result = started_service.run_check(dataset_id=dataset_id)
        assert result["age_hours"] == 0.0

    def test_run_check_provenance_hash(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check returns a 64-char provenance hash."""
        result = started_service.run_check(dataset_id=dataset_id)
        assert len(result["provenance_hash"]) == 64

    def test_run_check_increments_stats(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check increments total_checks counter."""
        started_service.run_check(dataset_id=dataset_id)
        started_service.run_check(dataset_id=dataset_id)
        assert started_service._stats["total_checks"] == 2

    def test_run_check_updates_dataset_state(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check updates the dataset's freshness state."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        ds = started_service.get_dataset(dataset_id)
        assert ds["freshness_level"] == "good"
        assert ds["last_checked_at"] is not None

    def test_run_check_sla_compliant(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
        sla_id: str,
    ):
        """Fresh dataset is SLA compliant."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["sla_status"] == "compliant"
        assert result["sla_breach"] is None

    def test_run_check_sla_warning_breach(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
        sla_id: str,
    ):
        """Dataset exceeding warning threshold triggers warning breach."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=15)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["sla_status"] == "warning"
        assert result["sla_breach"] is not None
        assert result["sla_breach"]["severity"] == "warning"

    def test_run_check_sla_critical_breach(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
        sla_id: str,
    ):
        """Dataset exceeding critical threshold triggers critical breach."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        assert result["sla_status"] == "critical"
        assert result["sla_breach"] is not None
        assert result["sla_breach"]["severity"] == "critical"

    def test_run_check_stores_breach(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
        sla_id: str,
    ):
        """Breach detected during check is stored in breaches store."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
        started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=ts,
        )
        breaches = started_service.list_breaches()
        assert breaches["total"] >= 1

    def test_run_check_invalid_timestamp_handled(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_check handles unparseable timestamps gracefully."""
        result = started_service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at="not-a-timestamp",
        )
        assert result["age_hours"] == 0.0

    def test_run_batch_check_all_datasets(
        self, started_service: DataFreshnessMonitorService,
    ):
        """run_batch_check checks all datasets when IDs not specified."""
        started_service.register_dataset(name="b1")
        started_service.register_dataset(name="b2")
        result = started_service.run_batch_check()
        assert result["total_checked"] == 2
        assert result["total_errors"] == 0
        assert len(result["results"]) == 2

    def test_run_batch_check_specific_ids(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_batch_check checks only specified IDs."""
        started_service.register_dataset(name="extra")
        result = started_service.run_batch_check(dataset_ids=[dataset_id])
        assert result["total_checked"] == 1

    def test_run_batch_check_with_errors(
        self, started_service: DataFreshnessMonitorService,
    ):
        """run_batch_check reports errors for unknown IDs."""
        result = started_service.run_batch_check(dataset_ids=["missing-1"])
        assert result["total_errors"] == 1
        assert result["errors"][0]["dataset_id"] == "missing-1"

    def test_run_batch_check_provenance_hash(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Batch check has a 64-char provenance hash."""
        result = started_service.run_batch_check()
        assert len(result["provenance_hash"]) == 64

    def test_list_checks_empty(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_checks returns empty before any checks run."""
        result = started_service.list_checks()
        assert result["checks"] == []
        assert result["total"] == 0

    def test_list_checks_returns_run_checks(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_checks returns previously run checks."""
        started_service.run_check(dataset_id=dataset_id)
        result = started_service.list_checks()
        assert result["total"] == 1

    def test_list_checks_dataset_filter(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_checks filters by dataset_id."""
        ds2 = started_service.register_dataset(name="other")
        started_service.run_check(dataset_id=dataset_id)
        started_service.run_check(dataset_id=ds2["dataset_id"])
        result = started_service.list_checks(dataset_id=dataset_id)
        assert result["total"] == 1

    def test_list_checks_pagination(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_checks respects limit and offset."""
        for _ in range(5):
            started_service.run_check(dataset_id=dataset_id)
        result = started_service.list_checks(limit=2, offset=1)
        assert result["count"] == 2
        assert result["offset"] == 1
        assert result["total"] == 5


# ===================================================================
# TestBreachManagement
# ===================================================================


class TestBreachManagement:
    """Tests for list_breaches, get_breach, update_breach."""

    def _create_breach(
        self,
        svc: DataFreshnessMonitorService,
        dataset_id: str,
    ) -> str:
        """Helper to create a breach via a stale check."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        result = svc.run_check(dataset_id=dataset_id, last_refreshed_at=ts)
        return result["sla_breach"]["breach_id"]

    def test_list_breaches_empty(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_breaches returns empty when no breaches exist."""
        result = started_service.list_breaches()
        assert result["breaches"] == []
        assert result["total"] == 0

    def test_list_breaches_returns_detected(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_breaches returns detected breaches."""
        self._create_breach(started_service, dataset_id)
        result = started_service.list_breaches()
        assert result["total"] >= 1

    def test_list_breaches_severity_filter(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_breaches filters by severity."""
        self._create_breach(started_service, dataset_id)
        result = started_service.list_breaches(severity="critical")
        assert all(b["severity"] == "critical" for b in result["breaches"])

    def test_list_breaches_status_filter(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_breaches filters by status."""
        self._create_breach(started_service, dataset_id)
        result = started_service.list_breaches(status="detected")
        assert all(b["status"] == "detected" for b in result["breaches"])

    def test_list_breaches_pagination(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """list_breaches respects limit and offset."""
        for _ in range(3):
            self._create_breach(started_service, dataset_id)
        result = started_service.list_breaches(limit=1, offset=0)
        assert result["count"] == 1

    def test_get_breach_found(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """get_breach returns breach when it exists."""
        breach_id = self._create_breach(started_service, dataset_id)
        breach = started_service.get_breach(breach_id)
        assert breach is not None
        assert breach["breach_id"] == breach_id

    def test_get_breach_not_found(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_breach returns None for unknown ID."""
        assert started_service.get_breach("nonexistent") is None

    def test_update_breach_status(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_breach changes the status."""
        breach_id = self._create_breach(started_service, dataset_id)
        result = started_service.update_breach(breach_id, status="acknowledged")
        assert result["status"] == "acknowledged"

    def test_update_breach_resolved(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_breach with 'resolved' sets resolved_at timestamp."""
        breach_id = self._create_breach(started_service, dataset_id)
        result = started_service.update_breach(breach_id, status="resolved")
        assert result["status"] == "resolved"
        assert result["resolved_at"] is not None

    def test_update_breach_resolution_notes(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_breach stores resolution_notes."""
        breach_id = self._create_breach(started_service, dataset_id)
        result = started_service.update_breach(
            breach_id, resolution_notes="Pipeline restarted",
        )
        assert result["resolution_notes"] == "Pipeline restarted"

    def test_update_breach_metadata(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """update_breach merges metadata."""
        breach_id = self._create_breach(started_service, dataset_id)
        result = started_service.update_breach(
            breach_id, metadata={"ticket": "INC-123"},
        )
        assert result["metadata"]["ticket"] == "INC-123"

    def test_update_breach_not_found_raises(
        self, started_service: DataFreshnessMonitorService,
    ):
        """update_breach raises ValueError for unknown ID."""
        with pytest.raises(ValueError, match="not found"):
            started_service.update_breach("nonexistent", status="resolved")


# ===================================================================
# TestAlerts
# ===================================================================


class TestAlerts:
    """Tests for list_alerts."""

    def test_list_alerts_empty(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_alerts returns empty when no alerts exist."""
        result = started_service.list_alerts()
        assert result["alerts"] == []
        assert result["total"] == 0

    def test_list_alerts_severity_filter(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_alerts filters by severity."""
        result = started_service.list_alerts(severity="critical")
        assert result["total"] == 0

    def test_list_alerts_status_filter(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_alerts filters by status."""
        result = started_service.list_alerts(status="open")
        assert result["total"] == 0

    def test_list_alerts_pagination(
        self, started_service: DataFreshnessMonitorService,
    ):
        """list_alerts uses limit and offset."""
        result = started_service.list_alerts(limit=10, offset=0)
        assert result["limit"] == 10
        assert result["offset"] == 0


# ===================================================================
# TestPredictions
# ===================================================================


class TestPredictions:
    """Tests for get_predictions."""

    def test_get_predictions_empty(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_predictions returns empty when none available."""
        result = started_service.get_predictions()
        assert result["predictions"] == []
        assert result["total"] == 0

    def test_get_predictions_dataset_filter(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_predictions filters by dataset_id."""
        result = started_service.get_predictions(dataset_id="ds-123")
        assert result["total"] == 0

    def test_get_predictions_pagination(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_predictions uses limit and offset."""
        result = started_service.get_predictions(limit=5, offset=2)
        assert result["limit"] == 5
        assert result["offset"] == 2


# ===================================================================
# TestPipeline
# ===================================================================


class TestPipeline:
    """Tests for run_pipeline."""

    def test_run_pipeline_empty(
        self, started_service: DataFreshnessMonitorService,
    ):
        """run_pipeline works with no registered datasets."""
        result = started_service.run_pipeline()
        assert result["status"] == "completed"
        assert "pipeline_id" in result
        assert result["batch_result"]["total_checked"] == 0

    def test_run_pipeline_with_datasets(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_pipeline checks registered datasets."""
        result = started_service.run_pipeline()
        assert result["batch_result"]["total_checked"] >= 1

    def test_run_pipeline_specific_ids(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_pipeline accepts explicit dataset_ids."""
        result = started_service.run_pipeline(dataset_ids=[dataset_id])
        assert result["dataset_ids"] == [dataset_id]

    def test_run_pipeline_provenance_hash(
        self, started_service: DataFreshnessMonitorService,
    ):
        """run_pipeline result has a 64-char provenance hash."""
        result = started_service.run_pipeline()
        assert len(result["provenance_hash"]) == 64

    def test_run_pipeline_increments_stats(
        self, started_service: DataFreshnessMonitorService,
    ):
        """run_pipeline increments total_pipelines counter."""
        started_service.run_pipeline()
        assert started_service._stats["total_pipelines"] == 1

    def test_run_pipeline_no_predictions(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_pipeline with run_predictions=False skips predictions."""
        result = started_service.run_pipeline(run_predictions=False)
        # With no predictor engine, predictions should be empty either way.
        assert isinstance(result["predictions"], list)

    def test_run_pipeline_no_alerts(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """run_pipeline with generate_alerts=False skips alert generation."""
        result = started_service.run_pipeline(generate_alerts=False)
        assert result["alerts"] == []

    def test_run_pipeline_staleness_patterns(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
        sla_id: str,
    ):
        """Pipeline detects staleness patterns for critical SLA breaches."""
        # Force a stale dataset
        ts = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        started_service.update_dataset(dataset_id)
        ds = started_service.get_dataset(dataset_id)
        ds["last_refreshed_at"] = ts
        result = started_service.run_pipeline(dataset_ids=[dataset_id])
        # Fallback path creates patterns for critical SLA status
        assert isinstance(result["staleness_patterns"], list)

    def test_run_pipeline_generates_alerts_for_breaches(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
        sla_id: str,
    ):
        """Pipeline generates alerts when breaches are detected."""
        ds = started_service.get_dataset(dataset_id)
        ds["last_refreshed_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=100)
        ).isoformat()
        result = started_service.run_pipeline(
            dataset_ids=[dataset_id],
            generate_alerts=True,
        )
        assert len(result["alerts"]) >= 1
        assert started_service._stats["total_alerts"] >= 1


# ===================================================================
# TestHealthAndStats
# ===================================================================


class TestHealthAndStats:
    """Tests for health_check, get_health, get_stats, get_statistics."""

    def test_health_check_before_startup(
        self, service: DataFreshnessMonitorService,
    ):
        """health_check returns 'starting' when not started."""
        result = service.health_check()
        assert result["status"] == "starting"

    def test_health_check_after_startup(
        self, started_service: DataFreshnessMonitorService,
    ):
        """health_check returns 'healthy' when started."""
        result = started_service.health_check()
        assert result["status"] == "healthy"

    def test_health_check_service_name(
        self, started_service: DataFreshnessMonitorService,
    ):
        """health_check reports correct service name."""
        result = started_service.health_check()
        assert result["service"] == "data_freshness_monitor"

    def test_health_check_engines_dict(
        self, started_service: DataFreshnessMonitorService,
    ):
        """health_check includes engine availability map."""
        result = started_service.health_check()
        assert "engines" in result
        expected_keys = {
            "dataset_registry", "sla_definition", "freshness_checker",
            "staleness_detector", "refresh_predictor", "alert_manager",
            "pipeline",
        }
        assert set(result["engines"].keys()) == expected_keys

    def test_health_check_stores_dict(
        self, started_service: DataFreshnessMonitorService,
    ):
        """health_check includes store size counts."""
        result = started_service.health_check()
        assert "stores" in result
        assert "datasets" in result["stores"]
        assert "sla_definitions" in result["stores"]

    def test_health_check_timestamp(
        self, started_service: DataFreshnessMonitorService,
    ):
        """health_check includes a timestamp."""
        result = started_service.health_check()
        assert "timestamp" in result

    def test_get_health_alias(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_health is an alias for health_check."""
        h1 = started_service.health_check()
        h2 = started_service.get_health()
        assert h1["status"] == h2["status"]
        assert h1["service"] == h2["service"]

    def test_get_stats_structure(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_stats returns expected keys."""
        result = started_service.get_stats()
        expected_keys = {
            "total_datasets", "total_sla_definitions", "total_checks",
            "total_breaches", "total_alerts", "total_predictions",
            "total_pipelines", "datasets_stored", "sla_definitions_stored",
            "checks_stored", "breaches_stored", "alerts_stored",
            "predictions_stored", "pipeline_results_stored",
            "provenance_entries", "timestamp",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_get_stats_reflects_operations(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """get_stats reflects dataset registrations."""
        stats = started_service.get_stats()
        assert stats["total_datasets"] >= 1
        assert stats["datasets_stored"] >= 1

    def test_get_statistics_alias(
        self, started_service: DataFreshnessMonitorService,
    ):
        """get_statistics is an alias for get_stats."""
        s1 = started_service.get_stats()
        s2 = started_service.get_statistics()
        assert s1["total_datasets"] == s2["total_datasets"]


# ===================================================================
# TestSingleton
# ===================================================================


class TestSingleton:
    """Tests for get_service, set_service, reset_service."""

    @pytest.fixture(autouse=True)
    def _clean_singleton(self):
        """Reset singleton state before and after each test."""
        import greenlang.data_freshness_monitor.setup as setup_mod

        original = setup_mod._service_instance
        setup_mod._service_instance = None
        yield
        setup_mod._service_instance = original

    def test_get_service_returns_instance(self):
        """get_service returns a DataFreshnessMonitorService."""
        svc = get_service()
        assert isinstance(svc, DataFreshnessMonitorService)

    def test_get_service_is_started(self):
        """get_service returns a started service."""
        svc = get_service()
        assert svc._started is True

    def test_get_service_singleton(self):
        """get_service returns the same instance on repeated calls."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_set_service_replaces_instance(self):
        """set_service installs a new service instance."""
        custom = DataFreshnessMonitorService(config={"custom": True})
        set_service(custom)
        svc = get_service()
        assert svc is custom

    def test_reset_service_creates_new(self):
        """reset_service creates a fresh started instance."""
        original = get_service()
        new = reset_service()
        assert new is not original
        assert new._started is True

    def test_reset_service_replaces_singleton(self):
        """reset_service replaces the global singleton."""
        original = get_service()
        reset_service()
        current = get_service()
        assert current is not original


# ===================================================================
# TestConfigureAndGetFreshnessMonitor
# ===================================================================


class TestConfigureAndGetFreshnessMonitor:
    """Tests for configure_freshness_monitor and get_freshness_monitor."""

    @pytest.fixture(autouse=True)
    def _clean_singleton(self):
        """Reset singleton state before and after each test."""
        import greenlang.data_freshness_monitor.setup as setup_mod

        original = setup_mod._service_instance
        setup_mod._service_instance = None
        yield
        setup_mod._service_instance = original

    def _make_app(self):
        """Create a minimal app-like object with state."""
        app = SimpleNamespace()
        app.state = SimpleNamespace()
        app.include_router = MagicMock()
        return app

    def test_configure_attaches_to_state(self):
        """configure_freshness_monitor sets app.state.data_freshness_monitor_service."""
        app = self._make_app()
        svc = configure_freshness_monitor(app)
        assert app.state.data_freshness_monitor_service is svc
        assert isinstance(svc, DataFreshnessMonitorService)

    def test_configure_returns_service(self):
        """configure_freshness_monitor returns the service."""
        app = self._make_app()
        svc = configure_freshness_monitor(app)
        assert isinstance(svc, DataFreshnessMonitorService)

    def test_get_freshness_monitor_returns_service(self):
        """get_freshness_monitor retrieves the attached service."""
        app = self._make_app()
        configure_freshness_monitor(app)
        svc = get_freshness_monitor(app)
        assert isinstance(svc, DataFreshnessMonitorService)

    def test_get_freshness_monitor_returns_none_when_unconfigured(self):
        """get_freshness_monitor returns None when not configured."""
        app = self._make_app()
        result = get_freshness_monitor(app)
        assert result is None


# ===================================================================
# TestGetRouter
# ===================================================================


class TestGetRouter:
    """Tests for get_router() function."""

    def test_get_router_returns_router(self):
        """get_router returns an APIRouter (or None if FastAPI unavailable)."""
        router = get_router()
        if router is not None:
            assert hasattr(router, "routes")

    def test_get_router_has_20_routes(self):
        """Router has exactly 20 registered routes."""
        router = get_router()
        if router is not None:
            route_count = len(router.routes)
            assert route_count == 20, (
                f"Expected 20 routes, got {route_count}"
            )


# ===================================================================
# TestComputeHash
# ===================================================================


class TestComputeHash:
    """Tests for the _compute_hash helper function."""

    def test_deterministic_hash(self):
        """Same input produces same hash."""
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_hash_length(self):
        """Hash is 64 characters (SHA-256 hex)."""
        h = _compute_hash({"test": "data"})
        assert len(h) == 64

    def test_different_input_different_hash(self):
        """Different input produces different hash."""
        h1 = _compute_hash({"x": 1})
        h2 = _compute_hash({"x": 2})
        assert h1 != h2

    def test_key_order_invariant(self):
        """Hash is independent of key insertion order."""
        h1 = _compute_hash({"b": 2, "a": 1})
        h2 = _compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_hash_handles_datetime(self):
        """Hash handles datetime objects via default=str."""
        h = _compute_hash({"ts": datetime.now(timezone.utc)})
        assert len(h) == 64


# ===================================================================
# TestInternalHelpers
# ===================================================================


class TestInternalHelpers:
    """Tests for _compute_freshness and _evaluate_sla internal methods."""

    def test_compute_freshness_boundary_0(
        self, started_service: DataFreshnessMonitorService,
    ):
        """0 hours -> excellent."""
        result = started_service._compute_freshness(0.0)
        assert result["level"] == "excellent"
        assert result["score"] == 1.0

    def test_compute_freshness_boundary_1(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Exactly 1 hour -> good (boundary)."""
        result = started_service._compute_freshness(1.0)
        assert result["level"] == "good"

    def test_compute_freshness_boundary_6(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Exactly 6 hours -> fair (boundary)."""
        result = started_service._compute_freshness(6.0)
        assert result["level"] == "fair"

    def test_compute_freshness_boundary_24(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Exactly 24 hours -> poor (boundary)."""
        result = started_service._compute_freshness(24.0)
        assert result["level"] == "poor"

    def test_compute_freshness_boundary_72(
        self, started_service: DataFreshnessMonitorService,
    ):
        """Exactly 72 hours -> stale (boundary)."""
        result = started_service._compute_freshness(72.0)
        assert result["level"] == "stale"

    def test_evaluate_sla_no_sla_defined(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """With no SLA, default thresholds are used."""
        # Dataset has no SLA attached
        result = started_service._evaluate_sla(dataset_id, 0.5)
        assert result["sla_status"] == "compliant"

    def test_evaluate_sla_default_warning(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """Default warning threshold is 24h without explicit SLA."""
        result = started_service._evaluate_sla(dataset_id, 25.0)
        assert result["sla_status"] == "warning"

    def test_evaluate_sla_default_critical(
        self,
        started_service: DataFreshnessMonitorService,
        dataset_id: str,
    ):
        """Default critical threshold is 72h without explicit SLA."""
        result = started_service._evaluate_sla(dataset_id, 73.0)
        assert result["sla_status"] == "critical"
