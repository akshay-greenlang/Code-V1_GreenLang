# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage.change_detector_engine - AGENT-MRV-030.

Tests Engine 5: ChangeDetectorEngine -- change tracking, recalculation
triggering, and version comparison for the Audit Trail & Lineage Agent
(GL-MRV-X-042).

Coverage:
- detect_change for each change type (EMISSION_FACTOR_UPDATE,
  ACTIVITY_DATA_CORRECTION, METHODOLOGY_CHANGE, SCOPE_BOUNDARY_CHANGE,
  REPORTING_YEAR_RESTATEMENT, MANUAL_ADJUSTMENT)
- Severity auto-assessment (LOW, MEDIUM, HIGH, CRITICAL)
- Materiality calculation (percentage change threshold)
- Impact analysis (downstream affected calculations)
- Version comparison (before vs after snapshots)
- Cascade impact tracking
- Recalculation triggering
- Change timeline retrieval
- Pending recalculations listing
- Change statistics

Target: ~80 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.change_detector_engine import (
        ChangeDetectorEngine,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="ChangeDetectorEngine not available",
)

ORG_ID = "org-test-change"
YEAR = 2025

CHANGE_TYPES = [
    "EMISSION_FACTOR_UPDATE",
    "ACTIVITY_DATA_CORRECTION",
    "METHODOLOGY_CHANGE",
    "SCOPE_BOUNDARY_CHANGE",
    "REPORTING_YEAR_RESTATEMENT",
    "MANUAL_ADJUSTMENT",
]


# ==============================================================================
# DETECT CHANGE TESTS
# ==============================================================================


@_SKIP
class TestDetectChange:
    """Test change detection functionality."""

    @pytest.mark.parametrize("change_type", CHANGE_TYPES)
    def test_detect_change_each_type(self, change_detector_engine, change_type):
        """Test detecting each supported change type."""
        result = change_detector_engine.detect_change(
            change_type=change_type,
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef_value": "2.68"},
            new_value={"ef_value": "2.71"},
            reason="Test change",
        )
        assert result["success"] is True

    def test_detect_change_returns_id(self, change_detector_engine):
        """Test detect_change returns a change_id."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="DEFRA update",
        )
        assert "change_id" in result
        assert result["change_id"] is not None

    def test_detect_change_invalid_type(self, change_detector_engine):
        """Test invalid change type raises ValueError."""
        with pytest.raises(ValueError):
            change_detector_engine.detect_change(
                change_type="INVALID_TYPE",
                organization_id=ORG_ID,
                reporting_year=YEAR,
                scope="scope_1",
                agent_id="GL-MRV-S1-001",
                previous_value={},
                new_value={},
                reason="test",
            )

    def test_detect_change_empty_org(self, change_detector_engine):
        """Test empty organization_id raises ValueError."""
        with pytest.raises(ValueError):
            change_detector_engine.detect_change(
                change_type="EMISSION_FACTOR_UPDATE",
                organization_id="",
                reporting_year=YEAR,
                scope="scope_1",
                agent_id="GL-MRV-S1-001",
                previous_value={},
                new_value={},
                reason="test",
            )

    def test_detect_change_has_timestamp(self, change_detector_engine):
        """Test change detection result includes timestamp."""
        result = change_detector_engine.detect_change(
            change_type="ACTIVITY_DATA_CORRECTION",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"quantity": "1000"},
            new_value={"quantity": "1050"},
            reason="Meter recalibration",
        )
        assert "detected_at" in result or "timestamp" in result

    def test_detect_change_with_metadata(self, change_detector_engine):
        """Test change detection with additional metadata."""
        result = change_detector_engine.detect_change(
            change_type="METHODOLOGY_CHANGE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"method": "tier_1"},
            new_value={"method": "tier_2"},
            reason="Upgraded to Tier 2",
            metadata={"approved_by": "compliance-team"},
        )
        assert result["success"] is True


# ==============================================================================
# SEVERITY ASSESSMENT TESTS
# ==============================================================================


@_SKIP
class TestSeverityAssessment:
    """Test automatic severity classification."""

    def test_severity_in_result(self, change_detector_engine):
        """Test change result includes severity."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="Minor update",
        )
        assert "severity" in result
        assert result["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_small_change_low_severity(self, change_detector_engine):
        """Test small changes are classified as LOW severity."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.680"},
            new_value={"ef": "2.681"},
            reason="Rounding adjustment",
        )
        assert result["severity"] in ["LOW", "MEDIUM"]

    def test_methodology_change_high_severity(self, change_detector_engine):
        """Test methodology changes tend to be HIGH severity."""
        result = change_detector_engine.detect_change(
            change_type="METHODOLOGY_CHANGE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"method": "spend_based"},
            new_value={"method": "supplier_specific"},
            reason="Full methodology change",
        )
        assert result["severity"] in ["HIGH", "CRITICAL"]


# ==============================================================================
# MATERIALITY CALCULATION TESTS
# ==============================================================================


@_SKIP
class TestMaterialityCalculation:
    """Test materiality threshold evaluation."""

    def test_materiality_in_result(self, change_detector_engine):
        """Test change result includes materiality assessment."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"total_co2e": "1000"},
            new_value={"total_co2e": "1060"},
            reason="Updated factors",
        )
        assert "is_material" in result or "materiality" in result

    def test_large_change_is_material(self, change_detector_engine):
        """Test change > 5% is flagged as material."""
        result = change_detector_engine.detect_change(
            change_type="ACTIVITY_DATA_CORRECTION",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"total_co2e": "1000"},
            new_value={"total_co2e": "1200"},
            reason="Major correction",
        )
        is_material = result.get("is_material", result.get("materiality", {}).get("is_material"))
        assert is_material is True


# ==============================================================================
# IMPACT ANALYSIS TESTS
# ==============================================================================


@_SKIP
class TestImpactAnalysis:
    """Test downstream impact analysis."""

    def test_impact_analysis(self, change_detector_engine):
        """Test impact analysis identifies affected calculations."""
        # First detect a change
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="DEFRA update",
        )
        change_id = result["change_id"]
        impact = change_detector_engine.analyze_impact(change_id)
        assert impact["success"] is True

    def test_impact_returns_affected_list(self, change_detector_engine):
        """Test impact analysis returns list of affected items."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="update",
        )
        impact = change_detector_engine.analyze_impact(result["change_id"])
        assert "affected" in impact or "affected_calculations" in impact

    def test_impact_nonexistent_change(self, change_detector_engine):
        """Test impact analysis for nonexistent change_id."""
        with pytest.raises((ValueError, KeyError)):
            change_detector_engine.analyze_impact("nonexistent-change")


# ==============================================================================
# VERSION COMPARISON TESTS
# ==============================================================================


@_SKIP
class TestVersionComparison:
    """Test version comparison between change snapshots."""

    def test_compare_versions(self, change_detector_engine):
        """Test comparing previous and new values."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68", "source": "DEFRA_2023"},
            new_value={"ef": "2.71", "source": "DEFRA_2024"},
            reason="Annual update",
        )
        comparison = change_detector_engine.compare_versions(result["change_id"])
        assert comparison["success"] is True

    def test_comparison_shows_diffs(self, change_detector_engine):
        """Test version comparison identifies field-level differences."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="update",
        )
        comparison = change_detector_engine.compare_versions(result["change_id"])
        assert "differences" in comparison or "diffs" in comparison


# ==============================================================================
# CASCADE IMPACT TESTS
# ==============================================================================


@_SKIP
class TestCascadeImpact:
    """Test cascade impact tracking through calculation chains."""

    def test_cascade_tracking(self, change_detector_engine):
        """Test cascade impact tracking returns results."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="update",
        )
        cascade = change_detector_engine.get_cascade_impact(result["change_id"])
        assert cascade["success"] is True


# ==============================================================================
# RECALCULATION TESTS
# ==============================================================================


@_SKIP
class TestRecalculation:
    """Test recalculation triggering and tracking."""

    def test_trigger_recalculation(self, change_detector_engine):
        """Test triggering a recalculation."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="update",
        )
        recalc = change_detector_engine.trigger_recalculation(result["change_id"])
        assert recalc["success"] is True

    def test_list_pending_recalculations(self, change_detector_engine):
        """Test listing pending recalculations."""
        pending = change_detector_engine.list_pending_recalculations(ORG_ID, YEAR)
        assert isinstance(pending, (list, dict))


# ==============================================================================
# CHANGE TIMELINE TESTS
# ==============================================================================


@_SKIP
class TestChangeTimeline:
    """Test change timeline retrieval."""

    def test_timeline_empty(self, change_detector_engine):
        """Test timeline is empty when no changes detected."""
        timeline = change_detector_engine.get_change_timeline(ORG_ID, YEAR)
        assert timeline["success"] is True
        assert len(timeline.get("changes", [])) == 0

    def test_timeline_after_changes(self, change_detector_engine):
        """Test timeline returns detected changes."""
        change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"},
            new_value={"ef": "2.71"},
            reason="update 1",
        )
        change_detector_engine.detect_change(
            change_type="ACTIVITY_DATA_CORRECTION",
            organization_id=ORG_ID,
            reporting_year=YEAR,
            scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={"qty": "1000"},
            new_value={"qty": "1050"},
            reason="update 2",
        )
        timeline = change_detector_engine.get_change_timeline(ORG_ID, YEAR)
        assert len(timeline.get("changes", [])) >= 2

    def test_timeline_chronological_order(self, change_detector_engine):
        """Test timeline entries are in chronological order."""
        for i in range(3):
            change_detector_engine.detect_change(
                change_type="EMISSION_FACTOR_UPDATE",
                organization_id=ORG_ID,
                reporting_year=YEAR,
                scope="scope_1",
                agent_id="GL-MRV-S1-001",
                previous_value={"ef": str(2.68 + i * 0.01)},
                new_value={"ef": str(2.69 + i * 0.01)},
                reason=f"update {i}",
            )
        timeline = change_detector_engine.get_change_timeline(ORG_ID, YEAR)
        changes = timeline.get("changes", [])
        if len(changes) >= 2:
            timestamps = [c.get("detected_at", c.get("timestamp", "")) for c in changes]
            assert timestamps == sorted(timestamps)


# ==============================================================================
# CHANGE STATISTICS TESTS
# ==============================================================================


@_SKIP
class TestChangeStatistics:
    """Test change statistics computation."""

    def test_statistics_empty(self, change_detector_engine):
        """Test statistics when no changes exist."""
        stats = change_detector_engine.get_statistics(ORG_ID, YEAR)
        assert stats["total_changes"] == 0

    def test_statistics_by_type(self, change_detector_engine):
        """Test statistics grouped by change type."""
        change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
            agent_id="GL-MRV-S1-001", previous_value={}, new_value={},
            reason="test",
        )
        stats = change_detector_engine.get_statistics(ORG_ID, YEAR)
        assert stats["total_changes"] >= 1
        assert "by_type" in stats


# ==============================================================================
# RESET TESTS
# ==============================================================================


@_SKIP
class TestChangeDetectorReset:
    """Test engine reset functionality."""

    def test_reset_clears_changes(self, change_detector_engine):
        """Test reset clears all detected changes."""
        change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
            agent_id="GL-MRV-S1-001", previous_value={}, new_value={},
            reason="test",
        )
        change_detector_engine.reset()
        stats = change_detector_engine.get_statistics(ORG_ID, YEAR)
        assert stats["total_changes"] == 0


# ==============================================================================
# ADDITIONAL CHANGE DETECTOR EDGE CASE TESTS
# ==============================================================================


@_SKIP
class TestChangeDetectorEdgeCases:
    """Additional edge case tests for change detector engine."""

    @pytest.mark.parametrize("scope", ["scope_1", "scope_2", "scope_3"])
    def test_detect_change_all_scopes(self, change_detector_engine, scope):
        """Test detecting changes in all GHG scopes."""
        result = change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID, reporting_year=YEAR, scope=scope,
            agent_id="GL-MRV-S1-001",
            previous_value={"ef": "2.68"}, new_value={"ef": "2.71"},
            reason=f"Update for {scope}",
        )
        assert result["success"] is True

    def test_detect_multiple_changes_same_scope(self, change_detector_engine):
        """Test detecting multiple changes in the same scope."""
        for i in range(5):
            result = change_detector_engine.detect_change(
                change_type="EMISSION_FACTOR_UPDATE",
                organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
                agent_id="GL-MRV-S1-001",
                previous_value={"ef": str(2.68 + i * 0.01)},
                new_value={"ef": str(2.69 + i * 0.01)},
                reason=f"Update {i}",
            )
            assert result["success"] is True
        stats = change_detector_engine.get_statistics(ORG_ID, YEAR)
        assert stats["total_changes"] >= 5

    def test_detect_change_across_orgs(self, change_detector_engine):
        """Test detecting changes in different organizations."""
        for org in ["org-A", "org-B"]:
            result = change_detector_engine.detect_change(
                change_type="ACTIVITY_DATA_CORRECTION",
                organization_id=org, reporting_year=YEAR, scope="scope_1",
                agent_id="GL-MRV-S1-001",
                previous_value={"qty": "1000"}, new_value={"qty": "1100"},
                reason="Correction",
            )
            assert result["success"] is True

    def test_detect_change_with_complex_values(self, change_detector_engine):
        """Test detecting change with complex nested values."""
        result = change_detector_engine.detect_change(
            change_type="METHODOLOGY_CHANGE",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
            agent_id="GL-MRV-S1-001",
            previous_value={
                "method": "tier_1",
                "parameters": {"fuel_type": "diesel", "ef_source": "DEFRA_2023"},
            },
            new_value={
                "method": "tier_2",
                "parameters": {"fuel_type": "diesel", "ef_source": "DEFRA_2024", "calorific_value": "35.8"},
            },
            reason="Upgrade to Tier 2 methodology",
        )
        assert result["success"] is True

    def test_scope_boundary_change_severity(self, change_detector_engine):
        """Test scope boundary changes are classified appropriately."""
        result = change_detector_engine.detect_change(
            change_type="SCOPE_BOUNDARY_CHANGE",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_3",
            agent_id="GL-MRV-X-042",
            previous_value={"categories": [1, 2, 3, 6]},
            new_value={"categories": [1, 2, 3, 4, 5, 6, 7]},
            reason="Extended Scope 3 boundary",
        )
        assert result["severity"] in ["MEDIUM", "HIGH", "CRITICAL"]

    def test_reporting_year_restatement(self, change_detector_engine):
        """Test reporting year restatement detection."""
        result = change_detector_engine.detect_change(
            change_type="REPORTING_YEAR_RESTATEMENT",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
            agent_id="GL-MRV-X-042",
            previous_value={"total_co2e": "50000"},
            new_value={"total_co2e": "52500"},
            reason="Restatement due to methodology correction",
        )
        assert result["success"] is True

    def test_manual_adjustment_tracked(self, change_detector_engine):
        """Test manual adjustment is tracked."""
        result = change_detector_engine.detect_change(
            change_type="MANUAL_ADJUSTMENT",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_2",
            agent_id="GL-MRV-S2-001",
            previous_value={"residual_mix_ef": "0.45"},
            new_value={"residual_mix_ef": "0.42"},
            reason="Manual correction by compliance team",
        )
        assert result["success"] is True

    def test_statistics_by_severity(self, change_detector_engine):
        """Test statistics include severity breakdown."""
        change_detector_engine.detect_change(
            change_type="EMISSION_FACTOR_UPDATE",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
            agent_id="GL-MRV-S1-001", previous_value={"ef": "2.68"},
            new_value={"ef": "2.69"}, reason="Minor",
        )
        change_detector_engine.detect_change(
            change_type="METHODOLOGY_CHANGE",
            organization_id=ORG_ID, reporting_year=YEAR, scope="scope_1",
            agent_id="GL-MRV-S1-001", previous_value={"method": "spend"},
            new_value={"method": "supplier"}, reason="Major",
        )
        stats = change_detector_engine.get_statistics(ORG_ID, YEAR)
        assert stats["total_changes"] >= 2
        assert "by_severity" in stats or "by_type" in stats
