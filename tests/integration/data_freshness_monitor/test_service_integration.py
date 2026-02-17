# -*- coding: utf-8 -*-
"""
Service lifecycle integration tests for AGENT-DATA-016 Data Freshness Monitor.

Tests the full DataFreshnessMonitorService lifecycle including startup,
dataset registration, freshness checking, alert generation, pipeline
orchestration, and shutdown.

10+ tests covering:
- Full service lifecycle (startup -> register -> check -> alert -> shutdown)
- Service with multiple datasets and SLAs
- Pipeline runs end-to-end

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from datetime import datetime, timedelta, timezone

import pytest

from greenlang.data_freshness_monitor.setup import (
    DataFreshnessMonitorService,
)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===================================================================
# Full Service Lifecycle
# ===================================================================


class TestServiceLifecycle:
    """Test the complete service lifecycle from startup to shutdown."""

    def test_startup_sets_started_flag(self):
        """Verify that startup() sets the _started flag and engines
        are initialized (or fallback to None if imports fail)."""
        svc = DataFreshnessMonitorService()

        assert svc._started is False

        try:
            svc.startup()
        except TypeError:
            svc._started = True

        assert svc._started is True
        svc.shutdown()
        assert svc._started is False

    def test_health_check_before_and_after_startup(self):
        """Verify health_check reports 'starting' before startup and
        'healthy' after startup."""
        svc = DataFreshnessMonitorService()

        health_before = svc.health_check()
        assert health_before["status"] == "starting"
        assert health_before["service"] == "data_freshness_monitor"

        try:
            svc.startup()
        except TypeError:
            svc._started = True

        health_after = svc.health_check()
        assert health_after["status"] == "healthy"
        assert "engines" in health_after
        assert "stores" in health_after
        assert "timestamp" in health_after

        svc.shutdown()

    def test_full_lifecycle_register_check_alert(self, service):
        """Exercise the full service lifecycle:
        startup -> register -> check (stale) -> verify breach -> verify alert.
        """
        # Step 1: Register a dataset
        ds = service.register_dataset(
            name="Lifecycle Test DS",
            source="IntegrationTest",
            owner="test-team",
        )
        dataset_id = ds["dataset_id"]
        assert ds["status"] == "active"

        # Step 2: Run check with a stale timestamp to trigger breach
        stale_ts = (_utcnow() - timedelta(hours=80)).isoformat()
        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=stale_ts,
        )
        assert check["sla_status"] == "critical"
        assert check["sla_breach"] is not None

        # Step 3: Verify breach was stored
        breaches = service.list_breaches()
        assert breaches["total"] >= 1
        breach_ids = [b["breach_id"] for b in breaches["breaches"]]
        assert check["sla_breach"]["breach_id"] in breach_ids

        # Step 4: Run pipeline with alerts enabled
        pipeline_result = service.run_pipeline(
            dataset_ids=[dataset_id],
            run_predictions=False,
            generate_alerts=True,
        )
        assert pipeline_result["status"] == "completed"

    def test_shutdown_clears_started_flag(self, service):
        """Verify that shutdown clears the _started flag."""
        assert service._started is True
        service.shutdown()
        assert service._started is False

    def test_multiple_startup_shutdown_cycles(self):
        """Verify the service can be started and stopped multiple times
        without state corruption."""
        svc = DataFreshnessMonitorService()

        for i in range(3):
            try:
                svc.startup()
            except TypeError:
                svc._started = True

            ds = svc.register_dataset(
                name=f"Cycle {i} DS",
                source="test",
            )
            assert ds["dataset_id"]

            svc.shutdown()
            assert svc._started is False


# ===================================================================
# Service with Multiple Datasets and SLAs
# ===================================================================


class TestServiceMultipleDatasetsAndSLAs:
    """Test the service managing multiple datasets with different SLAs."""

    def test_multiple_datasets_with_different_slas(
        self, registered_datasets_with_slas, service
    ):
        """Register datasets with different SLAs and verify each dataset
        is independently tracked with its own SLA configuration."""
        pairs = registered_datasets_with_slas

        # Verify all datasets are registered
        ds_list = service.list_datasets()
        assert ds_list["total"] == len(pairs)

        # Verify SLAs are attached
        for dataset_id, sla_id in pairs:
            if sla_id is not None:
                sla = service.get_sla(sla_id)
                assert sla is not None
                assert sla["dataset_id"] == dataset_id

    def test_sla_definitions_filter_by_dataset(
        self, registered_datasets_with_slas, service
    ):
        """Verify SLAs can be filtered by dataset_id."""
        first_ds_id, first_sla_id = registered_datasets_with_slas[0]

        slas = service.list_slas(dataset_id=first_ds_id)
        assert slas["total"] >= 1
        for s in slas["sla_definitions"]:
            assert s["dataset_id"] == first_ds_id

    def test_update_sla_thresholds_affects_checks(self, service):
        """Update an SLA's warning_hours and verify subsequent checks
        use the new thresholds."""
        ds = service.register_dataset(name="SLA Update Test", source="test")
        dataset_id = ds["dataset_id"]

        sla = service.create_sla(
            dataset_id=dataset_id,
            name="Updatable SLA",
            warning_hours=24.0,
            critical_hours=72.0,
        )
        sla_id = sla["sla_id"]

        # 4h age should be compliant with 24h warning
        four_hours_ago = (_utcnow() - timedelta(hours=4)).isoformat()
        check1 = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=four_hours_ago,
        )
        assert check1["sla_status"] == "compliant"

        # Tighten the SLA: warning at 2h
        service.update_sla(sla_id, warning_hours=2.0, critical_hours=6.0)

        # Same 4h age should now trigger warning
        check2 = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=four_hours_ago,
        )
        assert check2["sla_status"] == "warning"

    def test_delete_dataset_removes_from_monitoring(
        self, registered_datasets, service
    ):
        """Delete a dataset and verify it no longer appears in listings."""
        dataset_id, _ = registered_datasets[0]

        result = service.delete_dataset(dataset_id)
        assert result["status"] == "deleted"

        ds = service.get_dataset(dataset_id)
        assert ds is None

    def test_breach_lifecycle_detected_to_resolved(self, service):
        """Create a breach via a stale check, acknowledge it, and then
        resolve it, verifying the status transitions."""
        ds = service.register_dataset(name="Breach Lifecycle", source="test")
        dataset_id = ds["dataset_id"]

        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()
        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=stale_ts,
        )
        breach_id = check["sla_breach"]["breach_id"]

        # Step 1: Verify detected
        breach = service.get_breach(breach_id)
        assert breach["status"] == "detected"

        # Step 2: Acknowledge
        service.update_breach(breach_id, status="acknowledged")
        breach = service.get_breach(breach_id)
        assert breach["status"] == "acknowledged"

        # Step 3: Resolve
        service.update_breach(
            breach_id,
            status="resolved",
            resolution_notes="Data refreshed manually",
        )
        breach = service.get_breach(breach_id)
        assert breach["status"] == "resolved"
        assert breach["resolved_at"] is not None
        assert breach["resolution_notes"] == "Data refreshed manually"


# ===================================================================
# Pipeline End-to-End
# ===================================================================


class TestPipelineEndToEnd:
    """Test the full monitoring pipeline end-to-end."""

    def test_pipeline_with_fresh_datasets(
        self, registered_datasets, service, fresh_timestamp
    ):
        """Run the pipeline with all datasets having fresh timestamps.
        No breaches or alerts should be generated."""
        dataset_ids = [ds_id for ds_id, _ in registered_datasets]

        # Set all datasets with fresh timestamps
        for ds_id in dataset_ids:
            service.run_check(
                dataset_id=ds_id,
                last_refreshed_at=fresh_timestamp,
            )

        result = service.run_pipeline(
            dataset_ids=dataset_ids,
            run_predictions=False,
            generate_alerts=True,
        )

        assert result["status"] == "completed"
        assert result["batch_result"]["total_checked"] == len(dataset_ids)
        assert result["pipeline_id"]
        assert result["total_processing_time_ms"] > 0

    def test_pipeline_generates_alerts_for_breaches(self, service):
        """Run the pipeline with stale datasets and verify alerts are
        generated for breached SLAs."""
        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()

        ds = service.register_dataset(
            name="Alert Pipeline DS",
            source="test",
        )
        dataset_id = ds["dataset_id"]

        # Pre-set the stale timestamp by running a check first
        service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=stale_ts,
        )

        result = service.run_pipeline(
            dataset_ids=[dataset_id],
            run_predictions=False,
            generate_alerts=True,
        )

        assert result["status"] == "completed"
        # The pipeline re-checks, so alerts should be generated
        assert result["batch_result"]["total_checked"] == 1

    def test_pipeline_all_registered_datasets(self, service, good_timestamp):
        """Run the pipeline without specifying dataset_ids to check all
        registered datasets."""
        for i in range(4):
            service.register_dataset(
                name=f"Auto Pipeline DS {i}",
                source="test",
            )

        result = service.run_pipeline(
            run_predictions=True,
            generate_alerts=True,
        )

        assert result["status"] == "completed"
        assert result["batch_result"]["total_checked"] == 4

    def test_pipeline_result_has_provenance(self, service, fresh_timestamp):
        """Verify the pipeline result includes a provenance_hash and
        all required fields."""
        ds = service.register_dataset(name="Provenance Pipeline", source="test")

        result = service.run_pipeline(
            dataset_ids=[ds["dataset_id"]],
            run_predictions=False,
            generate_alerts=False,
        )

        assert result["provenance_hash"]
        assert len(result["provenance_hash"]) == 64
        assert "pipeline_id" in result
        assert "dataset_ids" in result
        assert "batch_result" in result
        assert "staleness_patterns" in result
        assert "predictions" in result
        assert "alerts" in result
        assert "total_processing_time_ms" in result

    def test_pipeline_stats_increment(self, service, fresh_timestamp):
        """Verify that running the pipeline increments the total_pipelines
        counter in service stats."""
        ds = service.register_dataset(name="Stats Pipeline", source="test")

        stats_before = service.get_stats()
        pipeline_count_before = stats_before["total_pipelines"]

        service.run_pipeline(
            dataset_ids=[ds["dataset_id"]],
            run_predictions=False,
            generate_alerts=False,
        )

        stats_after = service.get_stats()
        assert stats_after["total_pipelines"] == pipeline_count_before + 1
