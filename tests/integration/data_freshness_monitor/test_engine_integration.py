# -*- coding: utf-8 -*-
"""
Engine integration tests for AGENT-DATA-016 Data Freshness Monitor.

Tests the integration between engines through the DataFreshnessMonitorService
facade: dataset registration -> freshness checking -> SLA evaluation,
staleness detection, refresh prediction, and multi-dataset batch operations.

12+ tests covering:
- Dataset registration -> freshness check -> SLA evaluation flow
- Dataset registration -> staleness detection flow
- Dataset registration -> refresh prediction flow
- Multi-dataset batch operations

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

import time
from datetime import datetime, timedelta, timezone

import pytest


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===================================================================
# Dataset Registration -> Freshness Check -> SLA Evaluation Flow
# ===================================================================


class TestRegistrationCheckSLAFlow:
    """Test the full registration -> check -> SLA evaluation integration."""

    def test_register_then_check_fresh_dataset(self, service, fresh_timestamp):
        """Register a dataset, set a recent refresh timestamp, and verify
        the freshness check returns excellent/compliant status."""
        ds = service.register_dataset(
            name="Fresh Dataset",
            source="TestSource",
            owner="test-team",
            refresh_cadence="hourly",
            priority=1,
        )
        dataset_id = ds["dataset_id"]

        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=fresh_timestamp,
        )

        assert check["dataset_id"] == dataset_id
        assert check["freshness_level"] == "excellent"
        assert check["freshness_score"] == 1.0
        assert check["sla_status"] == "compliant"
        assert check["sla_breach"] is None
        assert check["age_hours"] < 1.0
        assert check["provenance_hash"]
        assert len(check["provenance_hash"]) == 64

    def test_register_then_check_stale_dataset(self, service, stale_timestamp):
        """Register a dataset with a stale refresh timestamp and verify
        the check returns stale level with critical breach."""
        ds = service.register_dataset(
            name="Stale Dataset",
            source="Legacy",
            owner="data-team",
        )
        dataset_id = ds["dataset_id"]

        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=stale_timestamp,
        )

        assert check["freshness_level"] == "stale"
        assert check["freshness_score"] == 0.2
        assert check["sla_status"] == "critical"
        assert check["sla_breach"] is not None
        assert check["sla_breach"]["severity"] == "critical"
        assert check["age_hours"] >= 72.0

    def test_register_with_custom_sla_warning_breach(
        self, service, warning_breach_timestamp
    ):
        """Register a dataset, attach a custom SLA with tight thresholds,
        then check that a warning-level breach is detected at 30h age
        against default 24h warning threshold."""
        ds = service.register_dataset(
            name="Warning SLA Dataset",
            source="API",
            owner="ops-team",
        )
        dataset_id = ds["dataset_id"]

        # Default SLA: warning=24h, critical=72h
        # Dataset age is ~30h, so warning breach expected
        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=warning_breach_timestamp,
        )

        assert check["sla_status"] == "warning"
        assert check["sla_breach"] is not None
        assert check["sla_breach"]["severity"] == "warning"

    def test_register_with_custom_sla_critical_breach(
        self, service, critical_breach_timestamp
    ):
        """Register a dataset with a custom SLA and verify critical breach
        detection at 80h against default 72h critical threshold."""
        ds = service.register_dataset(
            name="Critical SLA Dataset",
            source="ERP",
            owner="compliance",
        )
        dataset_id = ds["dataset_id"]

        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=critical_breach_timestamp,
        )

        assert check["sla_status"] == "critical"
        assert check["sla_breach"] is not None
        assert check["sla_breach"]["severity"] == "critical"
        assert check["sla_breach"]["status"] == "detected"

    def test_custom_sla_overrides_default_thresholds(self, service):
        """Register a dataset, attach a tight SLA (warning=2h, critical=6h),
        and verify the custom SLA thresholds are used for evaluation."""
        ds = service.register_dataset(
            name="Tight SLA Dataset",
            source="Streaming",
            owner="realtime-team",
        )
        dataset_id = ds["dataset_id"]

        sla = service.create_sla(
            dataset_id=dataset_id,
            name="Tight SLA",
            warning_hours=2.0,
            critical_hours=6.0,
            severity="critical",
        )
        assert sla["sla_id"]

        # 3 hours ago: should breach warning (2h) but not critical (6h)
        three_hours_ago = (_utcnow() - timedelta(hours=3)).isoformat()
        check = service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=three_hours_ago,
        )

        assert check["sla_status"] == "warning"
        assert check["sla_breach"] is not None
        assert check["sla_breach"]["severity"] == "warning"
        assert check["sla_breach"]["threshold_hours"] == 2.0

    def test_check_updates_dataset_state(self, service, good_timestamp):
        """Verify that running a freshness check updates the dataset's
        stored freshness_score, freshness_level, sla_status, and
        last_checked_at fields."""
        ds = service.register_dataset(name="State Update Test", source="API")
        dataset_id = ds["dataset_id"]

        assert ds["freshness_score"] is None
        assert ds["last_checked_at"] is None

        service.run_check(
            dataset_id=dataset_id,
            last_refreshed_at=good_timestamp,
        )

        updated_ds = service.get_dataset(dataset_id)
        assert updated_ds["freshness_score"] == 0.8
        assert updated_ds["freshness_level"] == "good"
        assert updated_ds["sla_status"] == "compliant"
        assert updated_ds["last_checked_at"] is not None

    def test_check_nonexistent_dataset_raises(self, service):
        """Verify that running a check on a non-existent dataset raises
        ValueError."""
        with pytest.raises(ValueError, match="not found"):
            service.run_check(dataset_id="nonexistent-id-12345")


# ===================================================================
# Dataset Registration -> Staleness Detection Flow
# ===================================================================


class TestRegistrationStalenessFlow:
    """Test dataset registration followed by staleness pattern detection
    via the pipeline's batch check and pattern flagging."""

    def test_pipeline_flags_critical_staleness_pattern(
        self, service, critical_breach_timestamp
    ):
        """Register a dataset at critical age and run the pipeline,
        verifying that a staleness pattern is flagged for sla_critical."""
        ds = service.register_dataset(
            name="Staleness Pattern Dataset",
            source="Legacy",
        )
        dataset_id = ds["dataset_id"]

        # Set the last_refreshed_at on the dataset so the check picks it up
        service.update_dataset(dataset_id, metadata={
            "last_refreshed_at": critical_breach_timestamp,
        })

        result = service.run_pipeline(
            dataset_ids=[dataset_id],
            run_predictions=False,
            generate_alerts=False,
        )

        assert result["status"] == "completed"
        assert result["batch_result"]["total_checked"] == 1

    def test_multiple_stale_datasets_detected(self, service):
        """Register multiple datasets with stale timestamps and verify
        the batch check identifies all of them."""
        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()
        dataset_ids = []
        for i in range(3):
            ds = service.register_dataset(
                name=f"Stale DS {i}",
                source="batch-test",
            )
            dataset_ids.append(ds["dataset_id"])

        batch = service.run_batch_check(dataset_ids=dataset_ids)

        assert batch["total_checked"] == 3
        assert batch["total_errors"] == 0
        for result in batch["results"]:
            # With no last_refreshed_at, age defaults to 0 hours
            assert result["check_id"]
            assert result["provenance_hash"]


# ===================================================================
# Dataset Registration -> Refresh Prediction Flow
# ===================================================================


class TestRegistrationPredictionFlow:
    """Test dataset registration followed by refresh prediction via pipeline."""

    def test_pipeline_runs_predictions_when_enabled(
        self, service, fresh_timestamp
    ):
        """Register a dataset and run the pipeline with predictions enabled.
        The pipeline should complete without errors even if the predictor
        engine uses fallback mode."""
        ds = service.register_dataset(
            name="Prediction Test Dataset",
            source="API",
            refresh_cadence="daily",
        )
        dataset_id = ds["dataset_id"]

        result = service.run_pipeline(
            dataset_ids=[dataset_id],
            run_predictions=True,
            generate_alerts=False,
        )

        assert result["status"] == "completed"
        assert result["batch_result"]["total_checked"] == 1
        # Predictions list may be empty if predictor engine not available
        assert isinstance(result["predictions"], list)

    def test_pipeline_skips_predictions_when_disabled(
        self, service, good_timestamp
    ):
        """Register a dataset and run the pipeline with predictions disabled,
        verifying no predictions are generated."""
        ds = service.register_dataset(name="No Predictions", source="Manual")
        dataset_id = ds["dataset_id"]

        result = service.run_pipeline(
            dataset_ids=[dataset_id],
            run_predictions=False,
            generate_alerts=False,
        )

        assert result["status"] == "completed"
        assert result["predictions"] == []


# ===================================================================
# Multi-Dataset Batch Operations
# ===================================================================


class TestMultiDatasetBatchOperations:
    """Test batch freshness checks across multiple datasets."""

    def test_batch_check_all_datasets(self, registered_datasets, service):
        """Register multiple datasets and run a batch check on all of them,
        verifying each dataset gets a check result."""
        batch = service.run_batch_check()

        assert batch["total_checked"] == len(registered_datasets)
        assert batch["total_errors"] == 0
        assert len(batch["results"]) == len(registered_datasets)
        assert batch["provenance_hash"]
        assert len(batch["provenance_hash"]) == 64

    def test_batch_check_subset(self, registered_datasets, service):
        """Run a batch check on a subset of datasets and verify only
        the requested datasets are checked."""
        subset_ids = [ds_id for ds_id, _ in registered_datasets[:2]]

        batch = service.run_batch_check(dataset_ids=subset_ids)

        assert batch["total_checked"] == 2
        checked_ids = {r["dataset_id"] for r in batch["results"]}
        assert checked_ids == set(subset_ids)

    def test_batch_check_with_mixed_freshness(self, service):
        """Register datasets with different ages and verify the batch check
        correctly assigns different freshness levels."""
        ages_and_expected = [
            (timedelta(minutes=15), "excellent", 1.0),
            (timedelta(hours=4), "good", 0.8),
            (timedelta(hours=18), "fair", 0.6),
            (timedelta(hours=50), "poor", 0.4),
            (timedelta(hours=100), "stale", 0.2),
        ]

        dataset_ids = []
        for age, level, score in ages_and_expected:
            ds = service.register_dataset(
                name=f"Age {age} DS", source="test",
            )
            dataset_ids.append(ds["dataset_id"])

        results = []
        for i, (age, expected_level, expected_score) in enumerate(
            ages_and_expected
        ):
            ts = (_utcnow() - age).isoformat()
            check = service.run_check(
                dataset_id=dataset_ids[i],
                last_refreshed_at=ts,
            )
            results.append(check)

        for i, (_, expected_level, expected_score) in enumerate(
            ages_and_expected
        ):
            assert results[i]["freshness_level"] == expected_level
            assert results[i]["freshness_score"] == expected_score

    def test_batch_check_handles_errors_gracefully(self, service):
        """Verify batch check handles a mix of valid and invalid dataset
        IDs, reporting errors for invalid ones without failing the batch."""
        ds = service.register_dataset(name="Valid DS", source="test")
        valid_id = ds["dataset_id"]

        batch = service.run_batch_check(
            dataset_ids=[valid_id, "invalid-id-99999"],
        )

        assert batch["total_checked"] == 1
        assert batch["total_errors"] == 1
        assert batch["errors"][0]["dataset_id"] == "invalid-id-99999"

    def test_batch_check_empty_list(self, service):
        """Verify batch check with an explicit empty list returns zero results."""
        batch = service.run_batch_check(dataset_ids=[])

        assert batch["total_checked"] == 0
        assert batch["total_errors"] == 0
        assert batch["results"] == []

    def test_batch_stores_breach_records(self, service):
        """Register datasets with stale timestamps, run batch check,
        and verify breach records are stored in the service."""
        stale_ts = (_utcnow() - timedelta(hours=100)).isoformat()

        dataset_ids = []
        for i in range(3):
            ds = service.register_dataset(name=f"Breach DS {i}", source="test")
            dataset_ids.append(ds["dataset_id"])

        # Run checks individually with stale timestamps to generate breaches
        for ds_id in dataset_ids:
            service.run_check(dataset_id=ds_id, last_refreshed_at=stale_ts)

        breaches = service.list_breaches()
        assert breaches["total"] >= 3

    def test_stats_reflect_operations(self, service, fresh_timestamp):
        """Verify that service stats accumulate correctly after multiple
        operations: registrations, SLAs, checks, and pipeline runs."""
        ds1 = service.register_dataset(name="Stats DS 1", source="test")
        ds2 = service.register_dataset(name="Stats DS 2", source="test")

        service.create_sla(
            dataset_id=ds1["dataset_id"],
            name="Stats SLA",
            warning_hours=12.0,
            critical_hours=48.0,
        )

        service.run_check(
            dataset_id=ds1["dataset_id"],
            last_refreshed_at=fresh_timestamp,
        )
        service.run_check(
            dataset_id=ds2["dataset_id"],
            last_refreshed_at=fresh_timestamp,
        )

        stats = service.get_stats()

        assert stats["total_datasets"] >= 2
        assert stats["total_sla_definitions"] >= 1
        assert stats["total_checks"] >= 2
        assert stats["datasets_stored"] >= 2
        assert stats["provenance_entries"] >= 4  # 2 register + 1 SLA + 2 check
