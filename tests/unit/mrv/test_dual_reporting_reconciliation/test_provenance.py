# -*- coding: utf-8 -*-
"""
Unit tests for Dual Reporting Reconciliation provenance tracking.

AGENT-MRV-013: Dual Reporting Reconciliation Agent
Target: 20 tests covering DualReportingProvenanceTracker and hash helpers.
"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime

import pytest

from greenlang.agents.mrv.dual_reporting_reconciliation.provenance import (
    DualReportingReconciliationProvenance,
    DualReportingProvenanceTracker,
    ProvenanceStage,
    ProvenanceEntry,
    ProvenanceChain,
    hash_upstream_result,
    hash_discrepancy,
    hash_quality_assessment,
    hash_framework_table,
    hash_trend_point,
    hash_compliance_result,
    hash_reconciliation_report,
    hash_waterfall,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before each test for isolation."""
    DualReportingProvenanceTracker._instance = None
    yield
    DualReportingProvenanceTracker._instance = None


# ===========================================================================
# 1. Initialization Tests
# ===========================================================================


class TestProvenanceInit:
    """Test provenance tracker initialization."""

    def test_create_instance(self):
        prov = DualReportingProvenanceTracker.get_instance()
        assert prov is not None

    def test_singleton_pattern(self):
        p1 = DualReportingProvenanceTracker.get_instance()
        p2 = DualReportingProvenanceTracker.get_instance()
        assert p1 is p2

    def test_alias_matches_class(self):
        assert DualReportingReconciliationProvenance is DualReportingProvenanceTracker


# ===========================================================================
# 2. Hash Functions
# ===========================================================================


class TestProvenanceHashing:
    """Test provenance hash computation."""

    def test_hash_upstream_result(self):
        h = hash_upstream_result(
            agent="GL-MRV-007",
            method="location_based",
            energy_type="electricity_grid",
            emissions=Decimal("1250.50"),
            ef_used=Decimal("0.2501"),
            provenance_hash="abc123",
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_deterministic(self):
        h1 = hash_upstream_result(
            agent="GL-MRV-007",
            method="location_based",
            energy_type="electricity_grid",
            emissions=Decimal("1250.50"),
            ef_used=Decimal("0.2501"),
            provenance_hash="abc123",
        )
        h2 = hash_upstream_result(
            agent="GL-MRV-007",
            method="location_based",
            energy_type="electricity_grid",
            emissions=Decimal("1250.50"),
            ef_used=Decimal("0.2501"),
            provenance_hash="abc123",
        )
        assert h1 == h2

    def test_hash_discrepancy(self):
        h = hash_discrepancy(
            type="rec_go_impact",
            direction="location_higher",
            materiality="MATERIAL",
            absolute=Decimal("625.25"),
            percentage=Decimal("50.0"),
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_quality_assessment(self):
        h = hash_quality_assessment(
            composite_score=Decimal("85.0"),
            grade="B",
            dimension_scores={
                "completeness": Decimal("90.0"),
                "consistency": Decimal("80.0"),
            },
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_framework_table(self):
        h = hash_framework_table(
            framework="ghg_protocol",
            row_count=10,
            footnote_count=3,
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_trend_point(self):
        h = hash_trend_point(
            period="2024",
            location=Decimal("1800"),
            market=Decimal("1000"),
            pif=Decimal("0.44"),
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_compliance_result(self):
        h = hash_compliance_result(
            framework="ghg_protocol",
            status="compliant",
            met=10,
            total=13,
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_different_data_different_hash(self):
        h1 = hash_upstream_result(
            agent="GL-MRV-007",
            method="location_based",
            energy_type="electricity_grid",
            emissions=Decimal("1250.50"),
            ef_used=Decimal("0.2501"),
            provenance_hash="abc123",
        )
        h2 = hash_upstream_result(
            agent="GL-MRV-007",
            method="market_based",
            energy_type="electricity_grid",
            emissions=Decimal("625.25"),
            ef_used=Decimal("0.12505"),
            provenance_hash="def456",
        )
        assert h1 != h2

    def test_hash_reconciliation_report(self):
        h = hash_reconciliation_report(
            reconciliation_id="REC-001",
            location_total=Decimal("1250.50"),
            market_total=Decimal("625.25"),
            pif=Decimal("0.50"),
            quality_grade="B",
        )
        assert isinstance(h, str)
        assert len(h) == 64

    def test_hash_waterfall(self):
        h = hash_waterfall(
            total_discrepancy=Decimal("625.25"),
            item_count=5,
        )
        assert isinstance(h, str)
        assert len(h) == 64


# ===========================================================================
# 3. Stage Tracking
# ===========================================================================


class TestProvenanceStages:
    """Test provenance stage tracking."""

    def test_provenance_stages_enum(self):
        assert len(ProvenanceStage) == 17

    def test_collect_location_stage(self):
        assert ProvenanceStage.COLLECT_LOCATION_RESULTS is not None

    def test_seal_provenance_stage(self):
        assert ProvenanceStage.SEAL_PROVENANCE is not None

    def test_all_stage_values(self):
        expected_stages = {
            "collect_location_results",
            "collect_market_results",
            "align_boundaries",
            "map_energy_types",
            "calculate_totals",
            "analyze_discrepancies",
            "classify_materiality",
            "waterfall_decomposition",
            "score_completeness",
            "score_consistency",
            "score_accuracy",
            "score_transparency",
            "generate_tables",
            "analyze_trends",
            "check_compliance",
            "assemble_report",
            "seal_provenance",
        }
        actual_stages = {s.value for s in ProvenanceStage}
        assert actual_stages == expected_stages

    def test_create_chain_and_add_stage(self):
        tracker = DualReportingProvenanceTracker.get_instance()
        chain_id = tracker.create_chain("REC-001", "ORG-123", "2024-Q1")
        chain_hash = tracker.add_stage(
            chain_id,
            ProvenanceStage.COLLECT_LOCATION_RESULTS,
            {"upstream_count": 5},
            {"total": "1250.50"},
        )
        assert isinstance(chain_hash, str)
        assert len(chain_hash) == 64

    def test_seal_chain(self):
        tracker = DualReportingProvenanceTracker.get_instance()
        chain_id = tracker.create_chain("REC-002", "ORG-123", "2024-Q2")
        tracker.add_stage(
            chain_id,
            ProvenanceStage.COLLECT_LOCATION_RESULTS,
            {"upstream_count": 3},
            {"total": "800.00"},
        )
        seal_hash = tracker.seal_chain(chain_id)
        assert isinstance(seal_hash, str)
        assert len(seal_hash) == 64

    def test_verify_chain(self):
        tracker = DualReportingProvenanceTracker.get_instance()
        chain_id = tracker.create_chain("REC-003", "ORG-123", "2024-Q3")
        tracker.add_stage(
            chain_id,
            ProvenanceStage.COLLECT_LOCATION_RESULTS,
            {"upstream_count": 2},
            {"total": "500.00"},
        )
        tracker.add_stage(
            chain_id,
            ProvenanceStage.ANALYZE_DISCREPANCIES,
            {"discrepancy_count": 1},
            {"pct": "10.0"},
        )
        is_valid, error = tracker.verify_chain(chain_id)
        assert is_valid is True
        assert error is None
