# -*- coding: utf-8 -*-
"""
Metrics integration tests for AGENT-DATA-015 Cross-Source Reconciliation.

Tests the 12 Prometheus metrics (or their Dummy fallbacks) are properly
wired into the reconciliation pipeline:
- Pipeline run increments job counter
- Match operations increment match counter
- Discrepancy detection increments discrepancy counter
- All 12 metric objects are non-None after module import
- Helper functions execute without error
- Metric counters increment correctly for each pipeline stage

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from typing import Any, Dict, List

import pytest

from greenlang.cross_source_reconciliation.metrics import (
    PROMETHEUS_AVAILABLE,
    csr_jobs_processed_total,
    csr_records_matched_total,
    csr_comparisons_total,
    csr_discrepancies_detected_total,
    csr_resolutions_applied_total,
    csr_golden_records_created_total,
    csr_processing_errors_total,
    csr_match_confidence,
    csr_processing_duration_seconds,
    csr_discrepancy_magnitude,
    csr_active_jobs,
    csr_pending_reviews,
    inc_jobs_processed,
    inc_records_matched,
    inc_comparisons,
    inc_discrepancies,
    inc_resolutions,
    inc_golden_records,
    observe_confidence,
    observe_duration,
    observe_magnitude,
    set_active_jobs,
    set_pending_reviews,
    inc_errors,
)


# =========================================================================
# Test class: Metric object availability
# =========================================================================


class TestMetricObjectAvailability:
    """Test that all 12 metric objects are importable and non-None."""

    def test_all_12_counters_and_gauges_are_non_none(self):
        """All 12 metric objects should be non-None regardless of prometheus_client."""
        metrics = [
            csr_jobs_processed_total,
            csr_records_matched_total,
            csr_comparisons_total,
            csr_discrepancies_detected_total,
            csr_resolutions_applied_total,
            csr_golden_records_created_total,
            csr_processing_errors_total,
            csr_match_confidence,
            csr_processing_duration_seconds,
            csr_discrepancy_magnitude,
            csr_active_jobs,
            csr_pending_reviews,
        ]

        for i, metric in enumerate(metrics):
            assert metric is not None, f"Metric #{i + 1} is None"

    def test_prometheus_available_is_boolean(self):
        """PROMETHEUS_AVAILABLE flag should be a boolean."""
        assert isinstance(PROMETHEUS_AVAILABLE, bool)


# =========================================================================
# Test class: Helper function safety
# =========================================================================


class TestMetricHelperFunctions:
    """Test that all metric helper functions execute without error."""

    def test_inc_jobs_processed_no_error(self):
        """inc_jobs_processed executes without raising."""
        inc_jobs_processed("completed")
        inc_jobs_processed("failed")
        inc_jobs_processed("cancelled")

    def test_inc_records_matched_no_error(self):
        """inc_records_matched executes without raising."""
        inc_records_matched("exact", count=5)
        inc_records_matched("fuzzy", count=3)
        inc_records_matched("composite", count=10)

    def test_inc_comparisons_no_error(self):
        """inc_comparisons executes without raising."""
        inc_comparisons("match", count=10)
        inc_comparisons("mismatch", count=2)
        inc_comparisons("missing_left", count=1)

    def test_inc_discrepancies_no_error(self):
        """inc_discrepancies executes without raising."""
        inc_discrepancies("value_mismatch", "critical", count=1)
        inc_discrepancies("missing_record", "low", count=5)

    def test_inc_resolutions_no_error(self):
        """inc_resolutions executes without raising."""
        inc_resolutions("priority_wins", count=3)
        inc_resolutions("weighted_average", count=2)

    def test_inc_golden_records_no_error(self):
        """inc_golden_records executes without raising."""
        inc_golden_records("created", count=5)
        inc_golden_records("updated", count=1)

    def test_observe_functions_no_error(self):
        """observe_confidence, observe_duration, observe_magnitude execute safely."""
        observe_confidence(0.95)
        observe_duration(1.5)
        observe_magnitude(15.0)

    def test_set_gauge_functions_no_error(self):
        """set_active_jobs and set_pending_reviews execute safely."""
        set_active_jobs(5)
        set_pending_reviews(3)

    def test_inc_errors_no_error(self):
        """inc_errors executes without raising."""
        inc_errors("validation")
        inc_errors("timeout")
        inc_errors("unknown")


# =========================================================================
# Test class: Pipeline metrics integration
# =========================================================================


class TestPipelineMetricsIntegration:
    """Test metrics are emitted during pipeline operations."""

    def test_pipeline_run_does_not_raise_metric_errors(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Full pipeline run should not raise any metric-related errors.

        Even when prometheus_client is not installed, the Dummy fallbacks
        should silently absorb all metric calls.
        """
        result = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            generate_golden_records=True,
        )

        assert result["status"] == "completed"

    def test_stats_reflect_pipeline_operations(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Service stats counters increase after pipeline operations."""
        stats_before = service.get_stats()

        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        stats_after = service.get_stats()

        assert stats_after["total_matches"] > stats_before["total_matches"]
        assert stats_after["total_pipelines"] > stats_before["total_pipelines"]

    def test_match_operation_increments_match_counter(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """match_records increments the total_matches stat."""
        stats_before = service.get_stats()

        service.match_records(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        stats_after = service.get_stats()
        assert stats_after["total_matches"] == stats_before["total_matches"] + 1

    def test_discrepancy_detection_increments_counter(
        self, service, records_with_large_discrepancy,
    ):
        """detect_discrepancies increments total_discrepancies stat."""
        recs = records_with_large_discrepancy
        match_result = service.match_records(
            records_a=recs["records_a"],
            records_b=recs["records_b"],
        )

        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
        )

        stats_before = service.get_stats()
        service.detect_discrepancies(
            comparison_id=comparison["comparison_id"],
        )
        stats_after = service.get_stats()

        assert stats_after["total_discrepancies"] > stats_before["total_discrepancies"]

    def test_resolution_increments_counter(
        self, service, records_with_large_discrepancy,
    ):
        """resolve_discrepancies increments total_resolutions stat."""
        recs = records_with_large_discrepancy
        match_result = service.match_records(
            records_a=recs["records_a"],
            records_b=recs["records_b"],
        )

        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
        )

        detection = service.detect_discrepancies(
            comparison_id=comparison["comparison_id"],
        )

        disc_ids = [
            d["discrepancy_id"]
            for d in detection.get("discrepancies", [])
        ]

        stats_before = service.get_stats()
        service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="priority_wins",
        )
        stats_after = service.get_stats()

        assert stats_after["total_resolutions"] > stats_before["total_resolutions"]

    def test_golden_record_creation_increments_counter(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Pipeline with golden_records=True increments golden_records counter."""
        stats_before = service.get_stats()

        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            generate_golden_records=True,
        )

        stats_after = service.get_stats()
        assert stats_after["total_golden_records"] > stats_before["total_golden_records"]

    def test_source_registration_increments_counter(self, service):
        """register_source increments total_sources stat."""
        stats_before = service.get_stats()

        service.register_source(name="Metric Test Source", source_type="erp")

        stats_after = service.get_stats()
        assert stats_after["total_sources"] == stats_before["total_sources"] + 1

    def test_comparison_increments_counter(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """compare_records increments total_comparisons stat."""
        match_result = service.match_records(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        pair = match_result["matched_pairs"][0]
        stats_before = service.get_stats()

        service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
        )

        stats_after = service.get_stats()
        assert stats_after["total_comparisons"] == stats_before["total_comparisons"] + 1
