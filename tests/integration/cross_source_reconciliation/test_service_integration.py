# -*- coding: utf-8 -*-
"""
Service-level integration tests for AGENT-DATA-015 Cross-Source Reconciliation.

Tests the CrossSourceReconciliationService facade as a single entry point:
- Full pipeline execution with 2-3 sources
- Source registration and retrieval
- Match -> compare -> detect -> resolve flow
- Golden record quality checks
- Job lifecycle management
- Service health and statistics

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from typing import Any, Dict, List

import pytest

from greenlang.cross_source_reconciliation.setup import (
    CrossSourceReconciliationService,
)


# =========================================================================
# Test class: Service-level pipeline and lifecycle
# =========================================================================


class TestServicePipeline:
    """Integration tests for service-level pipeline operations."""

    def test_service_startup_sets_started_flag(self, service):
        """Service startup sets the internal _started flag."""
        assert service._started is True

    def test_service_health_check_after_startup(self, service):
        """Service health check returns healthy after startup."""
        health = service.health_check()

        assert health["status"] == "healthy"
        assert health["service"] == "cross_source_reconciliation"
        assert isinstance(health["engines"], dict)
        assert isinstance(health["stores"], dict)
        assert "timestamp" in health

    def test_full_pipeline_execution_two_sources(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Full pipeline execution with two sources returns completed status."""
        result = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            match_keys=["entity_id", "period"],
            match_threshold=0.85,
            tolerance_pct=5.0,
            resolution_strategy="priority_wins",
            generate_golden_records=True,
        )

        assert result["status"] == "completed"
        assert result["pipeline_id"] != ""
        assert result["match_result"]["total_matched"] >= 1
        assert result["golden_record_count"] >= 1

    def test_pipeline_discrepancy_detection_works(
        self, service, records_with_large_discrepancy,
    ):
        """Pipeline detects discrepancies for records with large differences."""
        recs = records_with_large_discrepancy
        result = service.run_pipeline(
            records_a=recs["records_a"],
            records_b=recs["records_b"],
            tolerance_pct=5.0,
        )

        assert result["status"] == "completed"
        disc_result = result["discrepancy_result"]
        assert disc_result["total_discrepancies"] >= 1


# =========================================================================
# Test class: Source registration and retrieval
# =========================================================================


class TestServiceSourceManagement:
    """Integration tests for source registration and retrieval."""

    def test_register_and_retrieve_source(self, service):
        """Register a source and retrieve it by ID."""
        source = service.register_source(
            name="ERP System",
            source_type="erp",
            priority=1,
            credibility_score=0.95,
            refresh_cadence="daily",
        )

        assert source["source_id"] != ""
        assert source["name"] == "ERP System"
        assert source["source_type"] == "erp"

        retrieved = service.get_source(source["source_id"])
        assert retrieved is not None
        assert retrieved["name"] == "ERP System"

    def test_list_sources_returns_all_registered(self, service):
        """Listing sources returns all previously registered sources."""
        service.register_source(name="Source A", source_type="erp")
        service.register_source(name="Source B", source_type="utility")
        service.register_source(name="Source C", source_type="meter")

        result = service.list_sources()
        assert result["total"] >= 3
        assert result["count"] >= 3

    def test_update_source_modifies_fields(self, service):
        """Updating a source modifies the stored fields."""
        source = service.register_source(
            name="Original Name", source_type="erp", priority=5,
        )
        sid = source["source_id"]

        updated = service.update_source(
            sid, name="Updated Name", priority=1, credibility_score=0.99,
        )

        assert updated["name"] == "Updated Name"
        assert updated["priority"] == 1
        assert updated["credibility_score"] == 0.99

    def test_get_nonexistent_source_returns_none(self, service):
        """Getting a source with a fake ID returns None."""
        result = service.get_source("nonexistent-id")
        assert result is None


# =========================================================================
# Test class: Step-by-step match -> compare -> detect -> resolve flow
# =========================================================================


class TestServiceStepByStep:
    """Integration tests for step-by-step reconciliation flow."""

    def test_step_by_step_reconciliation(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Execute each pipeline step individually and verify continuity."""
        # Step 1: Match
        match_result = service.match_records(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            match_keys=["entity_id", "period"],
        )
        assert match_result["total_matched"] >= 1
        match_id = match_result["match_id"]

        # Step 2: Compare first matched pair
        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
        )
        assert comparison["total_fields"] >= 1
        comparison_id = comparison["comparison_id"]

        # Step 3: Detect discrepancies
        detection = service.detect_discrepancies(
            comparison_id=comparison_id,
        )
        assert isinstance(detection["total_discrepancies"], int)

        # Step 4: Resolve (if there are discrepancies)
        disc_ids = [
            d["discrepancy_id"]
            for d in detection.get("discrepancies", [])
        ]
        if disc_ids:
            resolution = service.resolve_discrepancies(
                discrepancy_ids=disc_ids,
                strategy="priority_wins",
            )
            assert resolution["total_resolved"] == len(disc_ids)

    def test_golden_record_quality_fields(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Golden records contain required quality fields."""
        result = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            generate_golden_records=True,
        )

        for golden in result.get("golden_records", []):
            assert "record_id" in golden
            assert "entity_id" in golden
            assert "period" in golden
            assert "field_values" in golden
            assert "field_sources" in golden
            assert "field_confidence" in golden
            assert "overall_confidence" in golden
            assert "provenance_hash" in golden

            # Overall confidence should be between 0 and 1
            assert 0.0 <= golden["overall_confidence"] <= 1.0

            # Field values should not be empty
            assert len(golden["field_values"]) >= 1
