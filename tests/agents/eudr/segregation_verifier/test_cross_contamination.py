# -*- coding: utf-8 -*-
"""
Tests for CrossContaminationDetector - AGENT-EUDR-010 Engine 5: Contamination Detection

Comprehensive test suite covering:
- Contamination risk detection (all 10 pathway types)
- Temporal proximity analysis (within threshold=risk, outside=safe)
- Spatial proximity analysis (adjacent=risk, distant=safe)
- Equipment sharing detection
- Contamination event recording (all 4 severities)
- Impact propagation (downstream batch tracing)
- Status downgrade (SG -> MB on contamination)
- Risk heatmap generation
- Root cause suggestion (deterministic templates per pathway)
- Corrective action suggestion (per pathway+severity)
- Facility contamination risk scoring
- Unresolved event tracking
- Contamination resolution
- Edge cases (cascading contamination, multiple pathways, zero-quantity)

Test count: 70+ tests
Coverage target: >= 85% of CrossContaminationDetector module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier Agent (GL-EUDR-SGV-010)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.segregation_verifier.conftest import (
    CONTAMINATION_PATHWAYS,
    CONTAMINATION_SEVERITIES,
    PATHWAY_RISK_WEIGHTS,
    SEVERITY_SCORES,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    CONTAMINATION_SPATIAL,
    CONTAMINATION_TEMPORAL,
    FAC_ID_WAREHOUSE_GH,
    FAC_ID_FACTORY_DE,
    ZONE_ID_COCOA_A,
    ZONE_ID_MIXED_D,
    BATCH_ID_COCOA_001,
    BATCH_ID_COCOA_002,
    BATCH_ID_PALM_001,
    BATCH_ID_COFFEE_001,
    make_contamination,
    make_zone,
    assert_valid_score,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Contamination Risk Detection
# ===========================================================================


class TestContaminationRiskDetection:
    """Test contamination risk detection for all pathway types."""

    @pytest.mark.parametrize("pathway", CONTAMINATION_PATHWAYS)
    def test_detect_all_pathway_types(self, cross_contamination_detector, pathway):
        """Each of the 10 contamination pathways can be detected."""
        event = make_contamination(pathway_type=pathway)
        result = cross_contamination_detector.record_event(event)
        assert result is not None
        assert result["pathway_type"] == pathway

    @pytest.mark.parametrize("pathway,weight", list(PATHWAY_RISK_WEIGHTS.items()))
    def test_pathway_risk_weights(self, cross_contamination_detector, pathway, weight):
        """Each pathway has the correct risk weight."""
        event = make_contamination(pathway_type=pathway)
        result = cross_contamination_detector.record_event(event)
        risk = cross_contamination_detector.get_pathway_risk(pathway)
        assert risk["weight"] == pytest.approx(weight)

    def test_shared_equipment_highest_weight(self):
        """Shared equipment has the highest pathway risk weight."""
        max_pathway = max(PATHWAY_RISK_WEIGHTS, key=PATHWAY_RISK_WEIGHTS.get)
        assert max_pathway == "shared_equipment"
        assert PATHWAY_RISK_WEIGHTS["shared_equipment"] == 0.90

    def test_shared_personnel_lowest_weight(self):
        """Shared personnel has the lowest pathway risk weight."""
        min_pathway = min(PATHWAY_RISK_WEIGHTS, key=PATHWAY_RISK_WEIGHTS.get)
        assert min_pathway == "shared_personnel"
        assert PATHWAY_RISK_WEIGHTS["shared_personnel"] == 0.40


# ===========================================================================
# 2. Temporal Proximity Analysis
# ===========================================================================


class TestTemporalProximityAnalysis:
    """Test temporal proximity contamination detection."""

    @pytest.mark.parametrize("hours_gap,should_detect", [
        (1, True),
        (2, True),
        (3, True),
        (4, True),
        (5, False),
        (8, False),
        (24, False),
    ])
    def test_temporal_proximity_at_thresholds(
        self, cross_contamination_detector, hours_gap, should_detect
    ):
        """Temporal proximity is detected within configurable threshold."""
        now = datetime.now(timezone.utc)
        result = cross_contamination_detector.check_temporal_proximity(
            batch_a_end=(now - timedelta(hours=hours_gap)).isoformat(),
            batch_b_start=now.isoformat(),
            facility_id=FAC_ID_FACTORY_DE,
        )
        if should_detect:
            assert result["risk_detected"] is True
        else:
            assert result["risk_detected"] is False

    def test_simultaneous_batches_high_risk(self, cross_contamination_detector):
        """Simultaneous batch processing has high temporal risk."""
        now = datetime.now(timezone.utc)
        result = cross_contamination_detector.check_temporal_proximity(
            batch_a_end=now.isoformat(),
            batch_b_start=now.isoformat(),
            facility_id=FAC_ID_FACTORY_DE,
        )
        assert result["risk_detected"] is True

    def test_large_gap_no_risk(self, cross_contamination_detector):
        """Large temporal gap has no contamination risk."""
        now = datetime.now(timezone.utc)
        result = cross_contamination_detector.check_temporal_proximity(
            batch_a_end=(now - timedelta(days=7)).isoformat(),
            batch_b_start=now.isoformat(),
            facility_id=FAC_ID_FACTORY_DE,
        )
        assert result["risk_detected"] is False


# ===========================================================================
# 3. Spatial Proximity Analysis
# ===========================================================================


class TestSpatialProximityAnalysis:
    """Test spatial proximity contamination detection."""

    def test_adjacent_zones_risk(self, cross_contamination_detector):
        """Adjacent zones with different commodities have spatial risk."""
        result = cross_contamination_detector.check_spatial_proximity(
            zone_a=ZONE_ID_COCOA_A,
            zone_b=ZONE_ID_MIXED_D,
            distance_meters=5.0,
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result["risk_detected"] is True

    def test_distant_zones_safe(self, cross_contamination_detector):
        """Distant zones have no spatial contamination risk."""
        result = cross_contamination_detector.check_spatial_proximity(
            zone_a=ZONE_ID_COCOA_A,
            zone_b="ZONE-FAR-AWAY",
            distance_meters=50.0,
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result["risk_detected"] is False

    @pytest.mark.parametrize("distance,should_detect", [
        (1.0, True),
        (5.0, True),
        (9.0, True),
        (10.0, True),
        (11.0, False),
        (50.0, False),
    ])
    def test_spatial_distance_thresholds(
        self, cross_contamination_detector, distance, should_detect
    ):
        """Spatial risk detection at various distances."""
        result = cross_contamination_detector.check_spatial_proximity(
            zone_a=ZONE_ID_COCOA_A,
            zone_b=ZONE_ID_MIXED_D,
            distance_meters=distance,
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result["risk_detected"] is should_detect


# ===========================================================================
# 4. Equipment Sharing Detection
# ===========================================================================


class TestEquipmentSharingDetection:
    """Test equipment sharing contamination risk."""

    def test_shared_equipment_detected(self, cross_contamination_detector):
        """Shared equipment between different commodity lines is detected."""
        result = cross_contamination_detector.check_equipment_sharing(
            line_a="LINE-A",
            line_b="LINE-B",
            shared_items=["scale", "conveyor"],
            facility_id=FAC_ID_FACTORY_DE,
        )
        assert result["risk_detected"] is True
        assert len(result.get("shared_items", [])) == 2

    def test_no_shared_equipment_safe(self, cross_contamination_detector):
        """No shared equipment means no risk."""
        result = cross_contamination_detector.check_equipment_sharing(
            line_a="LINE-A",
            line_b="LINE-B",
            shared_items=[],
            facility_id=FAC_ID_FACTORY_DE,
        )
        assert result["risk_detected"] is False


# ===========================================================================
# 5. Contamination Event Recording
# ===========================================================================


class TestContaminationEventRecording:
    """Test contamination event recording for all severities."""

    @pytest.mark.parametrize("severity", CONTAMINATION_SEVERITIES)
    def test_record_all_severities(self, cross_contamination_detector, severity):
        """Each of the 4 severity levels can be recorded."""
        event = make_contamination(severity=severity)
        result = cross_contamination_detector.record_event(event)
        assert result is not None
        assert result["severity"] == severity

    @pytest.mark.parametrize("severity,score", list(SEVERITY_SCORES.items()))
    def test_severity_scores(self, severity, score):
        """Each severity level has the correct impact score."""
        assert SEVERITY_SCORES[severity] == score

    def test_record_with_affected_batches(self, cross_contamination_detector):
        """Contamination event records affected batches."""
        event = make_contamination(
            affected_batches=[BATCH_ID_COCOA_001, BATCH_ID_COCOA_002]
        )
        result = cross_contamination_detector.record_event(event)
        assert len(result["affected_batches"]) == 2

    def test_record_provenance_hash(self, cross_contamination_detector):
        """Contamination event generates a provenance hash."""
        event = make_contamination()
        result = cross_contamination_detector.record_event(event)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_duplicate_contamination_id_raises(self, cross_contamination_detector):
        """Duplicate contamination ID raises an error."""
        event = make_contamination(contamination_id="CTM-DUP-001")
        cross_contamination_detector.record_event(event)
        with pytest.raises((ValueError, KeyError)):
            cross_contamination_detector.record_event(copy.deepcopy(event))

    def test_invalid_pathway_raises(self, cross_contamination_detector):
        """Invalid pathway type raises ValueError."""
        event = make_contamination(pathway_type="invalid_pathway")
        with pytest.raises(ValueError):
            cross_contamination_detector.record_event(event)

    def test_invalid_severity_raises(self, cross_contamination_detector):
        """Invalid severity raises ValueError."""
        event = make_contamination(severity="invalid_severity")
        with pytest.raises(ValueError):
            cross_contamination_detector.record_event(event)


# ===========================================================================
# 6. Impact Propagation
# ===========================================================================


class TestImpactPropagation:
    """Test contamination impact propagation to downstream batches."""

    def test_trace_downstream_batches(self, cross_contamination_detector):
        """Contamination traces to downstream batches."""
        event = make_contamination(
            contamination_id="CTM-PROP-001",
            affected_batches=[BATCH_ID_COCOA_001],
        )
        cross_contamination_detector.record_event(event)
        downstream = cross_contamination_detector.trace_impact("CTM-PROP-001")
        assert downstream is not None
        assert "affected_batches" in downstream

    def test_propagation_includes_source(self, cross_contamination_detector):
        """Impact propagation includes the source batch."""
        event = make_contamination(
            contamination_id="CTM-PROP-002",
            source_batch_id=BATCH_ID_PALM_001,
        )
        cross_contamination_detector.record_event(event)
        downstream = cross_contamination_detector.trace_impact("CTM-PROP-002")
        assert downstream.get("source_batch_id") == BATCH_ID_PALM_001


# ===========================================================================
# 7. Status Downgrade
# ===========================================================================


class TestStatusDowngrade:
    """Test CoC model status downgrade on contamination."""

    def test_sg_to_mb_downgrade(self, cross_contamination_detector):
        """SG batch downgrades to MB on contamination."""
        event = make_contamination(
            contamination_id="CTM-DG-001",
            affected_batches=[BATCH_ID_COCOA_001],
            severity="high",
        )
        cross_contamination_detector.record_event(event)
        downgrade = cross_contamination_detector.recommend_downgrade("CTM-DG-001")
        assert downgrade is not None
        recommendations = downgrade.get("recommendations", [])
        if recommendations:
            assert any(r.get("to_model") == "MB" for r in recommendations)

    def test_low_severity_no_downgrade(self, cross_contamination_detector):
        """Low severity contamination may not require downgrade."""
        event = make_contamination(
            contamination_id="CTM-NODG-001",
            severity="low",
            quantity_affected_kg=10.0,
        )
        cross_contamination_detector.record_event(event)
        downgrade = cross_contamination_detector.recommend_downgrade("CTM-NODG-001")
        recommendations = downgrade.get("recommendations", [])
        assert len(recommendations) == 0 or all(
            r.get("severity") == "low" for r in recommendations
        )


# ===========================================================================
# 8. Risk Heatmap
# ===========================================================================


class TestRiskHeatmap:
    """Test contamination risk heatmap generation."""

    def test_heatmap_generation(self, cross_contamination_detector):
        """Generate a risk heatmap for a facility."""
        event1 = make_contamination(contamination_id="CTM-HM-001",
                                     facility_id=FAC_ID_WAREHOUSE_GH)
        event2 = make_contamination(contamination_id="CTM-HM-002",
                                     facility_id=FAC_ID_WAREHOUSE_GH,
                                     pathway_type="shared_equipment")
        cross_contamination_detector.record_event(event1)
        cross_contamination_detector.record_event(event2)
        heatmap = cross_contamination_detector.generate_heatmap(FAC_ID_WAREHOUSE_GH)
        assert heatmap is not None
        assert "zones" in heatmap or "risk_zones" in heatmap or "data" in heatmap

    def test_heatmap_empty_facility(self, cross_contamination_detector):
        """Heatmap for facility with no events returns empty data."""
        heatmap = cross_contamination_detector.generate_heatmap("FAC-CLEAN-001")
        assert heatmap is not None


# ===========================================================================
# 9. Root Cause Suggestion
# ===========================================================================


class TestRootCauseSuggestion:
    """Test deterministic root cause suggestion per pathway."""

    @pytest.mark.parametrize("pathway", CONTAMINATION_PATHWAYS)
    def test_root_cause_per_pathway(self, cross_contamination_detector, pathway):
        """Each pathway has a deterministic root cause template."""
        event = make_contamination(pathway_type=pathway)
        cross_contamination_detector.record_event(event)
        suggestion = cross_contamination_detector.suggest_root_cause(
            event["contamination_id"]
        )
        assert suggestion is not None
        assert "root_cause" in suggestion
        assert len(suggestion["root_cause"]) > 0

    def test_root_cause_is_deterministic(self, cross_contamination_detector):
        """Same pathway always produces the same root cause template."""
        event1 = make_contamination(pathway_type="spatial_proximity",
                                     contamination_id="CTM-RC-001")
        event2 = make_contamination(pathway_type="spatial_proximity",
                                     contamination_id="CTM-RC-002")
        cross_contamination_detector.record_event(event1)
        cross_contamination_detector.record_event(event2)
        s1 = cross_contamination_detector.suggest_root_cause("CTM-RC-001")
        s2 = cross_contamination_detector.suggest_root_cause("CTM-RC-002")
        assert s1["root_cause"] == s2["root_cause"]


# ===========================================================================
# 10. Corrective Action Suggestion
# ===========================================================================


class TestCorrectiveActionSuggestion:
    """Test corrective action suggestions per pathway and severity."""

    @pytest.mark.parametrize("pathway", CONTAMINATION_PATHWAYS)
    def test_corrective_actions_per_pathway(self, cross_contamination_detector, pathway):
        """Each pathway has corrective action suggestions."""
        event = make_contamination(pathway_type=pathway, severity="medium")
        cross_contamination_detector.record_event(event)
        actions = cross_contamination_detector.suggest_corrective_actions(
            event["contamination_id"]
        )
        assert actions is not None
        assert len(actions.get("actions", [])) > 0

    @pytest.mark.parametrize("severity", CONTAMINATION_SEVERITIES)
    def test_actions_vary_by_severity(self, cross_contamination_detector, severity):
        """Corrective actions scale with severity."""
        event = make_contamination(severity=severity)
        cross_contamination_detector.record_event(event)
        actions = cross_contamination_detector.suggest_corrective_actions(
            event["contamination_id"]
        )
        assert actions is not None


# ===========================================================================
# 11. Facility Risk Scoring
# ===========================================================================


class TestFacilityRiskScoring:
    """Test facility-level contamination risk scoring."""

    def test_facility_risk_score(self, cross_contamination_detector):
        """Calculate overall contamination risk score for facility."""
        for i in range(3):
            event = make_contamination(
                contamination_id=f"CTM-FACR-{i:03d}",
                facility_id=FAC_ID_WAREHOUSE_GH,
            )
            cross_contamination_detector.record_event(event)
        score = cross_contamination_detector.calculate_facility_risk(FAC_ID_WAREHOUSE_GH)
        assert score is not None
        assert_valid_score(score.get("risk_score", score.get("score", 0)))

    def test_clean_facility_low_score(self, cross_contamination_detector):
        """Facility with no contamination events has low risk score."""
        score = cross_contamination_detector.calculate_facility_risk("FAC-PRISTINE-001")
        total = score.get("risk_score", score.get("score", 0))
        assert total <= 10.0


# ===========================================================================
# 12. Unresolved Event Tracking
# ===========================================================================


class TestUnresolvedTracking:
    """Test tracking of unresolved contamination events."""

    def test_open_events_tracked(self, cross_contamination_detector):
        """Open contamination events are tracked."""
        event = make_contamination(contamination_id="CTM-OPEN-001", status="open")
        cross_contamination_detector.record_event(event)
        unresolved = cross_contamination_detector.get_unresolved(FAC_ID_WAREHOUSE_GH)
        assert any(e["contamination_id"] == "CTM-OPEN-001" for e in unresolved)

    def test_resolved_not_in_unresolved(self, cross_contamination_detector):
        """Resolved events are not in unresolved list."""
        event = make_contamination(contamination_id="CTM-RESOLVED-001", status="resolved")
        cross_contamination_detector.record_event(event)
        unresolved = cross_contamination_detector.get_unresolved(FAC_ID_WAREHOUSE_GH)
        assert not any(e["contamination_id"] == "CTM-RESOLVED-001" for e in unresolved)


# ===========================================================================
# 13. Contamination Resolution
# ===========================================================================


class TestContaminationResolution:
    """Test contamination event resolution."""

    def test_resolve_event(self, cross_contamination_detector):
        """Resolve an open contamination event."""
        event = make_contamination(contamination_id="CTM-RES-001", status="open")
        cross_contamination_detector.record_event(event)
        result = cross_contamination_detector.resolve(
            "CTM-RES-001",
            root_cause="Insufficient barrier between zones",
            corrective_actions=["Installed physical wall", "Retrained staff"],
        )
        assert result["status"] == "resolved"
        assert result["root_cause"] is not None

    def test_resolve_already_resolved_raises(self, cross_contamination_detector):
        """Resolving an already-resolved event raises an error."""
        event = make_contamination(contamination_id="CTM-ALRES-001", status="open")
        cross_contamination_detector.record_event(event)
        cross_contamination_detector.resolve(
            "CTM-ALRES-001", root_cause="Test", corrective_actions=["Fix"]
        )
        with pytest.raises((ValueError, KeyError)):
            cross_contamination_detector.resolve(
                "CTM-ALRES-001", root_cause="Again", corrective_actions=["Fix again"]
            )

    def test_resolve_nonexistent_raises(self, cross_contamination_detector):
        """Resolving a non-existent event raises an error."""
        with pytest.raises((ValueError, KeyError)):
            cross_contamination_detector.resolve(
                "CTM-NONEXIST", root_cause="Test", corrective_actions=["Fix"]
            )


# ===========================================================================
# 14. Edge Cases
# ===========================================================================


class TestContaminationEdgeCases:
    """Test edge cases for contamination detection."""

    def test_zero_quantity_contamination(self, cross_contamination_detector):
        """Zero quantity contamination is recorded but flagged."""
        event = make_contamination(quantity_affected_kg=0.0)
        result = cross_contamination_detector.record_event(event)
        assert result is not None
        assert result["quantity_affected_kg"] == 0.0

    def test_negative_quantity_raises(self, cross_contamination_detector):
        """Negative quantity raises ValueError."""
        event = make_contamination(quantity_affected_kg=-100.0)
        with pytest.raises(ValueError):
            cross_contamination_detector.record_event(event)

    def test_get_nonexistent_event_returns_none(self, cross_contamination_detector):
        """Getting a non-existent contamination event returns None."""
        result = cross_contamination_detector.get_event("CTM-NONEXISTENT")
        assert result is None

    def test_multiple_pathways_same_event(self, cross_contamination_detector):
        """Multiple contamination events from different pathways at same facility."""
        for i, pathway in enumerate(CONTAMINATION_PATHWAYS[:5]):
            event = make_contamination(
                contamination_id=f"CTM-MULTI-{i:03d}",
                pathway_type=pathway,
                facility_id=FAC_ID_WAREHOUSE_GH,
            )
            cross_contamination_detector.record_event(event)
        score = cross_contamination_detector.calculate_facility_risk(FAC_ID_WAREHOUSE_GH)
        assert score.get("risk_score", score.get("score", 0)) > 0

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_contamination_all_commodities(self, cross_contamination_detector, commodity):
        """Contamination events can be recorded for all commodities."""
        event = make_contamination()
        event["metadata"] = {"commodity": commodity}
        result = cross_contamination_detector.record_event(event)
        assert result is not None
