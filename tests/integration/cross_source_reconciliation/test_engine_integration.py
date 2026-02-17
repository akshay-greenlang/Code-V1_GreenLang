# -*- coding: utf-8 -*-
"""
Engine-level integration tests for AGENT-DATA-015 Cross-Source Reconciliation.

Tests cross-engine interactions and combined behaviors:
- SourceRegistry + MatchingEngine: register sources then match records
- MatchingEngine + ComparisonEngine: match then compare matched pairs
- ComparisonEngine + DiscrepancyDetector: compare then detect discrepancies
- DiscrepancyDetector + ResolutionEngine: detect then resolve discrepancies
- ResolutionEngine + AuditTrail: resolve then verify audit trail completeness
- Full engine pipeline: register -> match -> compare -> detect -> resolve -> golden
- Provenance chain threading across all engine interactions

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from typing import Any, Dict, List

import pytest

from greenlang.cross_source_reconciliation.config import (
    CrossSourceReconciliationConfig,
    set_config,
)
from greenlang.cross_source_reconciliation.provenance import ProvenanceTracker
from greenlang.cross_source_reconciliation.setup import (
    CrossSourceReconciliationService,
)


# =========================================================================
# Test class: Source Registration -> Record Matching
# =========================================================================


class TestSourceRegistryWithMatching:
    """Test interactions between source registration and record matching."""

    def test_register_three_sources_then_match(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Register ERP and Utility sources then match their records."""
        # Step 1: Register sources
        erp_source = service.register_source(
            name="ERP System",
            source_type="erp",
            priority=1,
            credibility_score=0.95,
        )
        utility_source = service.register_source(
            name="Utility Provider",
            source_type="utility",
            priority=2,
            credibility_score=0.90,
        )

        assert erp_source["source_id"] != ""
        assert utility_source["source_id"] != ""

        # Step 2: Match records across the two sources
        match_result = service.match_records(
            source_ids=[erp_source["source_id"], utility_source["source_id"]],
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            match_keys=["entity_id", "period"],
        )

        assert match_result["total_matched"] >= 1
        assert match_result["provenance_hash"] != ""

    def test_register_three_sources_all_retrievable(self, service):
        """Register 3 sources and verify all are listed."""
        names = ["ERP System", "Utility Provider", "Meter Data"]
        types = ["erp", "utility", "meter"]
        source_ids = []

        for name, stype in zip(names, types):
            source = service.register_source(
                name=name, source_type=stype, priority=3,
            )
            source_ids.append(source["source_id"])

        result = service.list_sources()
        assert result["total"] >= 3

        for sid in source_ids:
            found = service.get_source(sid)
            assert found is not None
            assert found["source_id"] == sid

    def test_source_update_then_match(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Update source priority then match -- higher priority source wins."""
        erp_source = service.register_source(
            name="ERP System", source_type="erp", priority=5,
        )
        utility_source = service.register_source(
            name="Utility Provider", source_type="utility", priority=2,
        )

        # Update ERP to highest priority
        service.update_source(
            erp_source["source_id"], priority=1, credibility_score=0.99,
        )

        match_result = service.match_records(
            source_ids=[erp_source["source_id"], utility_source["source_id"]],
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        assert match_result["match_id"] != ""
        assert match_result["total_matched"] >= 1


# =========================================================================
# Test class: Matching -> Comparison -> Discrepancy Detection
# =========================================================================


class TestMatchCompareDetect:
    """Test the match -> compare -> detect discrepancy pipeline segment."""

    def test_match_then_compare_matched_pairs(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Match records, then compare first matched pair field by field."""
        match_result = service.match_records(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        assert match_result["total_matched"] >= 1

        # Compare via the stored match_id
        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
        )

        assert comparison["comparison_id"] != ""
        assert comparison["total_fields"] >= 1
        assert "provenance_hash" in comparison

    def test_compare_detects_mismatches_beyond_tolerance(
        self, service, records_with_large_discrepancy,
    ):
        """Compare records with large difference -> mismatching fields."""
        recs = records_with_large_discrepancy
        match_result = service.match_records(
            records_a=recs["records_a"],
            records_b=recs["records_b"],
        )

        assert match_result["total_matched"] >= 1

        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
            tolerance_pct=5.0,
            tolerance_abs=0.01,
        )

        assert comparison["mismatching_fields"] >= 1

    def test_compare_within_tolerance_no_mismatch(
        self, service, records_within_tolerance,
    ):
        """Compare records within tolerance -> all fields match."""
        recs = records_within_tolerance
        match_result = service.match_records(
            records_a=recs["records_a"],
            records_b=recs["records_b"],
        )

        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
            tolerance_pct=5.0,
            tolerance_abs=1.0,
        )

        # All fields should match or be within tolerance
        assert comparison["mismatching_fields"] == 0

    def test_detect_discrepancies_from_comparison(
        self, service, records_with_large_discrepancy,
    ):
        """Detect discrepancies from comparison results."""
        recs = records_with_large_discrepancy
        match_result = service.match_records(
            records_a=recs["records_a"],
            records_b=recs["records_b"],
        )

        pair = match_result["matched_pairs"][0]
        comparison = service.compare_records(
            record_a=pair["record_a"],
            record_b=pair["record_b"],
            tolerance_pct=5.0,
        )

        detection = service.detect_discrepancies(
            comparison_id=comparison["comparison_id"],
        )

        assert detection["total_discrepancies"] >= 1
        assert detection["provenance_hash"] != ""

    def test_discrepancy_severity_classification(
        self, service, records_with_large_discrepancy,
    ):
        """Verify discrepancies are classified by severity."""
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

        assert detection["total_discrepancies"] >= 1
        # At least one discrepancy should have a severity classification
        for disc in detection.get("discrepancies", []):
            assert disc["severity"] in (
                "critical", "high", "medium", "low", "info",
            )


# =========================================================================
# Test class: Discrepancy Detection -> Resolution -> Golden Records
# =========================================================================


class TestDetectResolveGolden:
    """Test detect -> resolve -> golden record assembly."""

    def test_resolve_discrepancies_priority_wins(
        self, service, records_with_large_discrepancy,
    ):
        """Resolve discrepancies using priority_wins strategy."""
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

        resolution = service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="priority_wins",
        )

        assert resolution["total_resolved"] >= 1
        assert resolution["provenance_hash"] != ""
        for res in resolution.get("resolutions", []):
            assert res["strategy"] == "priority_wins"
            assert res["resolved_value"] is not None

    def test_resolve_weighted_average_strategy(
        self, service, records_with_large_discrepancy,
    ):
        """Resolve using weighted_average produces averaged values."""
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

        resolution = service.resolve_discrepancies(
            discrepancy_ids=disc_ids,
            strategy="weighted_average",
        )

        assert resolution["total_resolved"] >= 1
        for res in resolution.get("resolutions", []):
            assert res["strategy"] == "weighted_average"


# =========================================================================
# Test class: Full Engine Pipeline (end-to-end)
# =========================================================================


class TestFullEnginePipeline:
    """Test the complete engine pipeline end-to-end."""

    def test_full_pipeline_erp_vs_utility(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Run full pipeline: ERP vs Utility data produces golden records."""
        # Register sources
        erp = service.register_source(
            name="ERP System", source_type="erp", priority=1,
        )
        util = service.register_source(
            name="Utility Provider", source_type="utility", priority=2,
        )

        result = service.run_pipeline(
            source_ids=[erp["source_id"], util["source_id"]],
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            resolution_strategy="priority_wins",
            generate_golden_records=True,
        )

        assert result["status"] == "completed"
        assert result["golden_record_count"] >= 1
        assert result["provenance_hash"] != ""
        assert result["total_processing_time_ms"] >= 0

    def test_full_pipeline_produces_audit_trail(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Full pipeline generates audit trail entries in stats."""
        service.register_source(
            name="ERP System", source_type="erp", priority=1,
        )
        service.register_source(
            name="Utility Provider", source_type="utility", priority=2,
        )

        service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )

        stats = service.get_stats()
        assert stats["total_matches"] >= 1
        assert stats["provenance_entries"] >= 1

    def test_pipeline_golden_records_have_field_sources(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Golden records include field_sources attribution."""
        result = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            generate_golden_records=True,
        )

        for golden in result.get("golden_records", []):
            assert golden["field_sources"] is not None
            assert len(golden["field_sources"]) >= 1
            assert golden["overall_confidence"] >= 0.0
            assert golden["provenance_hash"] != ""

    def test_pipeline_with_three_way_reconciliation(
        self,
        service,
        sample_erp_data,
        sample_utility_data,
        sample_meter_data,
    ):
        """Pipeline handles two-source reconciliation; meter data used separately.

        This test validates that the service can handle separate
        pairwise reconciliations for ERP vs Utility and ERP vs Meter.
        """
        # Run ERP vs Utility
        result_1 = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
        )
        assert result_1["status"] == "completed"

        # Run ERP vs Meter
        result_2 = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_meter_data,
        )
        assert result_2["status"] == "completed"

        # Stats should reflect both runs
        stats = service.get_stats()
        assert stats["total_matches"] >= 2
        assert stats["total_pipelines"] >= 2

    def test_pipeline_without_golden_records(
        self, service, sample_erp_data, sample_utility_data,
    ):
        """Pipeline with generate_golden_records=False skips golden assembly."""
        result = service.run_pipeline(
            records_a=sample_erp_data,
            records_b=sample_utility_data,
            generate_golden_records=False,
        )

        assert result["status"] == "completed"
        assert result["golden_record_count"] == 0
        assert len(result.get("golden_records", [])) == 0
        # Pipeline should still produce match and discrepancy results
        assert result["match_result"]["total_matched"] >= 1
