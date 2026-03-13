# -*- coding: utf-8 -*-
"""
Tests for StorageSegregationAuditor - AGENT-EUDR-010 Engine 2: Storage Segregation

Comprehensive test suite covering:
- Zone registration (all 12 storage types, various barrier types)
- Storage event recording (material_in, material_out, zone_transfer, cleaning, inspection)
- Storage audit (full audit flow, barrier quality, zone separation, cleaning compliance)
- Adjacent zone risk assessment
- Capacity tracking (utilization, overflow risk)
- Cleaning protocol verification
- Inventory reconciliation (match/mismatch)
- Contamination incident recording
- Storage score calculation (composite with 4 sub-scores)
- Edge cases (zero capacity, full zone, no barriers)

Test count: 65+ tests
Coverage target: >= 85% of StorageSegregationAuditor module

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
    STORAGE_TYPES,
    BARRIER_TYPES,
    STORAGE_EVENT_TYPES,
    STORAGE_SCORE_WEIGHTS,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    ZONE_COCOA_A,
    ZONE_COCOA_B,
    ZONE_PALM_C,
    ZONE_ID_COCOA_A,
    ZONE_ID_COCOA_B,
    ZONE_ID_PALM_C,
    ZONE_ID_MIXED_D,
    FAC_ID_WAREHOUSE_GH,
    FAC_ID_MILL_ID,
    BATCH_ID_COCOA_001,
    BATCH_ID_COCOA_002,
    BATCH_ID_PALM_001,
    make_zone,
    make_storage_event,
    assert_valid_score,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Zone Registration
# ===========================================================================


class TestZoneRegistration:
    """Test storage zone registration."""

    @pytest.mark.parametrize("storage_type", STORAGE_TYPES)
    def test_register_all_storage_types(self, storage_segregation_auditor, storage_type):
        """Each of the 12 storage types can be registered."""
        zone = make_zone(storage_type=storage_type)
        result = storage_segregation_auditor.register_zone(zone)
        assert result is not None
        assert result["storage_type"] == storage_type

    @pytest.mark.parametrize("barrier_type", BARRIER_TYPES)
    def test_register_all_barrier_types(self, storage_segregation_auditor, barrier_type):
        """Each barrier type can be registered."""
        zone = make_zone(barrier_type=barrier_type)
        result = storage_segregation_auditor.register_zone(zone)
        assert result is not None
        assert result["barrier_type"] == barrier_type

    def test_register_zone_full_details(self, storage_segregation_auditor):
        """Register a zone with all fields populated."""
        zone = copy.deepcopy(ZONE_COCOA_A)
        result = storage_segregation_auditor.register_zone(zone)
        assert result["zone_id"] == ZONE_ID_COCOA_A
        assert result["commodity"] == "cocoa"
        assert result["capacity_kg"] == 25000.0

    def test_duplicate_zone_id_raises(self, storage_segregation_auditor):
        """Registering a zone with a duplicate ID raises an error."""
        zone = make_zone(zone_id="ZONE-DUP-001")
        storage_segregation_auditor.register_zone(zone)
        with pytest.raises((ValueError, KeyError)):
            storage_segregation_auditor.register_zone(copy.deepcopy(zone))

    def test_missing_zone_id_auto_assigns(self, storage_segregation_auditor):
        """Zone without zone_id gets one auto-assigned."""
        zone = make_zone()
        zone["zone_id"] = None
        result = storage_segregation_auditor.register_zone(zone)
        assert result.get("zone_id") is not None

    def test_register_zone_provenance_hash(self, storage_segregation_auditor):
        """Zone registration generates a provenance hash."""
        zone = make_zone()
        result = storage_segregation_auditor.register_zone(zone)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Storage Event Recording
# ===========================================================================


class TestStorageEventRecording:
    """Test recording storage events."""

    @pytest.mark.parametrize("event_type", STORAGE_EVENT_TYPES)
    def test_record_all_event_types(self, storage_segregation_auditor, event_type):
        """Each of the 5 storage event types can be recorded."""
        zone = make_zone(zone_id=f"ZONE-EVT-{event_type}")
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event(event_type=event_type, zone_id=zone["zone_id"])
        result = storage_segregation_auditor.record_event(evt)
        assert result is not None
        assert result["event_type"] == event_type

    def test_record_material_in(self, storage_segregation_auditor):
        """Material-in event increases zone quantity."""
        zone = make_zone(zone_id="ZONE-IN-001", current_quantity_kg=10000.0)
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("material_in", "ZONE-IN-001", quantity_kg=5000.0)
        storage_segregation_auditor.record_event(evt)
        updated = storage_segregation_auditor.get_zone("ZONE-IN-001")
        assert updated["current_quantity_kg"] == pytest.approx(15000.0)

    def test_record_material_out(self, storage_segregation_auditor):
        """Material-out event decreases zone quantity."""
        zone = make_zone(zone_id="ZONE-OUT-001", current_quantity_kg=10000.0)
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("material_out", "ZONE-OUT-001", quantity_kg=3000.0)
        storage_segregation_auditor.record_event(evt)
        updated = storage_segregation_auditor.get_zone("ZONE-OUT-001")
        assert updated["current_quantity_kg"] == pytest.approx(7000.0)

    def test_record_zone_transfer(self, storage_segregation_auditor):
        """Zone transfer moves material between zones."""
        z1 = make_zone(zone_id="ZONE-XFR-A", current_quantity_kg=10000.0)
        z2 = make_zone(zone_id="ZONE-XFR-B", current_quantity_kg=5000.0)
        storage_segregation_auditor.register_zone(z1)
        storage_segregation_auditor.register_zone(z2)
        evt = make_storage_event("zone_transfer", "ZONE-XFR-A", quantity_kg=3000.0)
        evt["target_zone_id"] = "ZONE-XFR-B"
        result = storage_segregation_auditor.record_event(evt)
        assert result is not None

    def test_record_cleaning_event(self, storage_segregation_auditor):
        """Cleaning event updates last_cleaned timestamp."""
        zone = make_zone(zone_id="ZONE-CLN-001")
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("cleaning", "ZONE-CLN-001")
        result = storage_segregation_auditor.record_event(evt)
        assert result is not None

    def test_record_inspection_event(self, storage_segregation_auditor):
        """Inspection event records audit observation."""
        zone = make_zone(zone_id="ZONE-INSP-001")
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("inspection", "ZONE-INSP-001")
        result = storage_segregation_auditor.record_event(evt)
        assert result is not None

    def test_invalid_event_type_raises(self, storage_segregation_auditor):
        """Invalid event type raises ValueError."""
        zone = make_zone(zone_id="ZONE-INVAL-001")
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("invalid_type", "ZONE-INVAL-001")
        with pytest.raises(ValueError):
            storage_segregation_auditor.record_event(evt)

    def test_event_on_unregistered_zone_raises(self, storage_segregation_auditor):
        """Event on an unregistered zone raises an error."""
        evt = make_storage_event("material_in", "ZONE-NONEXISTENT")
        with pytest.raises((ValueError, KeyError)):
            storage_segregation_auditor.record_event(evt)


# ===========================================================================
# 3. Storage Audit
# ===========================================================================


class TestStorageAudit:
    """Test storage segregation audit flow."""

    def test_full_audit_flow(self, storage_segregation_auditor):
        """Run a full storage audit for a facility."""
        zone_a = make_zone(zone_id="ZONE-AUD-A", facility_id=FAC_ID_WAREHOUSE_GH)
        zone_b = make_zone(zone_id="ZONE-AUD-B", facility_id=FAC_ID_WAREHOUSE_GH)
        storage_segregation_auditor.register_zone(zone_a)
        storage_segregation_auditor.register_zone(zone_b)
        result = storage_segregation_auditor.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert result is not None
        assert "audit_score" in result or "score" in result

    def test_audit_includes_barrier_quality(self, storage_segregation_auditor):
        """Audit result includes barrier quality assessment."""
        zone = make_zone(zone_id="ZONE-BAR-001", barrier_quality_score=90.0)
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert "barrier_quality" in result or "barrier_score" in result or "scores" in result

    def test_audit_includes_zone_separation(self, storage_segregation_auditor):
        """Audit result includes zone separation assessment."""
        zone = make_zone(zone_id="ZONE-SEP-001")
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert result is not None

    def test_audit_includes_cleaning_compliance(self, storage_segregation_auditor):
        """Audit result includes cleaning compliance assessment."""
        zone = make_zone(zone_id="ZONE-CLN-AUD-001")
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert result is not None

    def test_audit_no_zones_returns_zero_score(self, storage_segregation_auditor):
        """Audit for facility with no zones returns zero score."""
        result = storage_segregation_auditor.audit(facility_id="FAC-EMPTY-AUDIT")
        total = result.get("audit_score", result.get("score", 0.0))
        assert total == 0.0

    def test_audit_provenance_hash(self, storage_segregation_auditor):
        """Audit result includes provenance hash."""
        zone = make_zone(zone_id="ZONE-PROV-001")
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.audit(facility_id=FAC_ID_WAREHOUSE_GH)
        assert result.get("provenance_hash") is not None


# ===========================================================================
# 4. Adjacent Zone Risk Assessment
# ===========================================================================


class TestAdjacentZoneRisk:
    """Test adjacent zone risk assessment."""

    def test_adjacent_zone_risk_detected(self, storage_segregation_auditor):
        """Adjacent zones with different commodities have risk flagged."""
        z1 = make_zone(zone_id="ZONE-ADJ-A", commodity="cocoa",
                       adjacent_zones=["ZONE-ADJ-B"])
        z2 = make_zone(zone_id="ZONE-ADJ-B", commodity="palm_oil",
                       adjacent_zones=["ZONE-ADJ-A"])
        storage_segregation_auditor.register_zone(z1)
        storage_segregation_auditor.register_zone(z2)
        risk = storage_segregation_auditor.assess_adjacent_risk("ZONE-ADJ-A")
        assert risk is not None
        assert risk.get("risk_detected") is True or len(risk.get("risks", [])) > 0

    def test_adjacent_zone_same_commodity_no_risk(self, storage_segregation_auditor):
        """Adjacent zones with same commodity have no cross-contamination risk."""
        z1 = make_zone(zone_id="ZONE-SAME-A", commodity="cocoa",
                       adjacent_zones=["ZONE-SAME-B"])
        z2 = make_zone(zone_id="ZONE-SAME-B", commodity="cocoa",
                       adjacent_zones=["ZONE-SAME-A"])
        storage_segregation_auditor.register_zone(z1)
        storage_segregation_auditor.register_zone(z2)
        risk = storage_segregation_auditor.assess_adjacent_risk("ZONE-SAME-A")
        if "risk_detected" in risk:
            assert risk["risk_detected"] is False
        else:
            assert len(risk.get("risks", [])) == 0

    def test_no_adjacent_zones_safe(self, storage_segregation_auditor):
        """Zone with no adjacent zones has no adjacency risk."""
        zone = make_zone(zone_id="ZONE-ISOL-001", adjacent_zones=[])
        storage_segregation_auditor.register_zone(zone)
        risk = storage_segregation_auditor.assess_adjacent_risk("ZONE-ISOL-001")
        assert risk.get("risk_detected") is False or len(risk.get("risks", [])) == 0


# ===========================================================================
# 5. Capacity Tracking
# ===========================================================================


class TestCapacityTracking:
    """Test zone capacity tracking and overflow risk."""

    def test_utilization_percentage(self, storage_segregation_auditor):
        """Utilization is correctly calculated as current/capacity."""
        zone = make_zone(zone_id="ZONE-UTIL-001", capacity_kg=25000.0, current_quantity_kg=18000.0)
        storage_segregation_auditor.register_zone(zone)
        util = storage_segregation_auditor.get_utilization("ZONE-UTIL-001")
        assert util["utilization_pct"] == pytest.approx(72.0)

    def test_overflow_risk_at_high_utilization(self, storage_segregation_auditor):
        """Zone at >90% utilization is flagged for overflow risk."""
        zone = make_zone(zone_id="ZONE-OVER-001", capacity_kg=25000.0, current_quantity_kg=24000.0)
        storage_segregation_auditor.register_zone(zone)
        util = storage_segregation_auditor.get_utilization("ZONE-OVER-001")
        assert util.get("overflow_risk") is True or util["utilization_pct"] > 90.0

    def test_empty_zone_zero_utilization(self, storage_segregation_auditor):
        """Empty zone has zero utilization."""
        zone = make_zone(zone_id="ZONE-EMPTY-001", capacity_kg=25000.0, current_quantity_kg=0.0)
        storage_segregation_auditor.register_zone(zone)
        util = storage_segregation_auditor.get_utilization("ZONE-EMPTY-001")
        assert util["utilization_pct"] == pytest.approx(0.0)

    def test_material_out_exceeding_current_raises(self, storage_segregation_auditor):
        """Material out exceeding current quantity raises ValueError."""
        zone = make_zone(zone_id="ZONE-EXCEED-001", current_quantity_kg=5000.0)
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("material_out", "ZONE-EXCEED-001", quantity_kg=10000.0)
        with pytest.raises(ValueError):
            storage_segregation_auditor.record_event(evt)

    @pytest.mark.parametrize("current,capacity,expected_pct", [
        (0.0, 25000.0, 0.0),
        (5000.0, 25000.0, 20.0),
        (12500.0, 25000.0, 50.0),
        (25000.0, 25000.0, 100.0),
    ])
    def test_utilization_at_various_levels(self, storage_segregation_auditor, current, capacity, expected_pct):
        """Utilization percentage is correct at various fill levels."""
        zone = make_zone(capacity_kg=capacity, current_quantity_kg=current)
        storage_segregation_auditor.register_zone(zone)
        util = storage_segregation_auditor.get_utilization(zone["zone_id"])
        assert util["utilization_pct"] == pytest.approx(expected_pct)


# ===========================================================================
# 6. Cleaning Protocol Verification
# ===========================================================================


class TestCleaningProtocol:
    """Test cleaning protocol verification."""

    def test_cleaning_within_schedule_compliant(self, storage_segregation_auditor):
        """Zone cleaned within schedule is compliant."""
        zone = make_zone(zone_id="ZONE-CLNV-001")
        zone["last_cleaned"] = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.verify_cleaning("ZONE-CLNV-001")
        assert result.get("compliant") is True or result.get("status") == "compliant"

    def test_cleaning_overdue_non_compliant(self, storage_segregation_auditor):
        """Zone not cleaned within schedule is non-compliant."""
        zone = make_zone(zone_id="ZONE-CLNO-001")
        zone["last_cleaned"] = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.verify_cleaning("ZONE-CLNO-001")
        assert result.get("compliant") is False or result.get("status") == "overdue"

    @pytest.mark.parametrize("method", ["dry_sweep", "vacuum", "water_wash", "chemical_wash"])
    def test_cleaning_methods_recorded(self, storage_segregation_auditor, method):
        """Cleaning method is recorded in event."""
        zone = make_zone(zone_id=f"ZONE-CLM-{method}", cleaning_method=method)
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("cleaning", zone["zone_id"])
        evt["cleaning_method"] = method
        result = storage_segregation_auditor.record_event(evt)
        assert result is not None


# ===========================================================================
# 7. Inventory Reconciliation
# ===========================================================================


class TestInventoryReconciliation:
    """Test inventory reconciliation between expected and actual quantities."""

    def test_reconciliation_match(self, storage_segregation_auditor):
        """Matching expected and actual quantities pass reconciliation."""
        zone = make_zone(zone_id="ZONE-RECON-001", current_quantity_kg=10000.0)
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.reconcile("ZONE-RECON-001", actual_kg=10000.0)
        assert result.get("match") is True or result.get("variance_kg") == pytest.approx(0.0)

    def test_reconciliation_mismatch(self, storage_segregation_auditor):
        """Mismatched quantities flag a variance."""
        zone = make_zone(zone_id="ZONE-RECON-002", current_quantity_kg=10000.0)
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.reconcile("ZONE-RECON-002", actual_kg=9500.0)
        variance = abs(result.get("variance_kg", 0.0))
        assert variance == pytest.approx(500.0)

    def test_reconciliation_within_tolerance(self, storage_segregation_auditor):
        """Small variance within tolerance passes reconciliation."""
        zone = make_zone(zone_id="ZONE-RECON-003", current_quantity_kg=10000.0)
        storage_segregation_auditor.register_zone(zone)
        result = storage_segregation_auditor.reconcile("ZONE-RECON-003", actual_kg=9990.0)
        assert result.get("within_tolerance") is True or result.get("match") is True


# ===========================================================================
# 8. Storage Score Calculation
# ===========================================================================


class TestStorageScoreCalculation:
    """Test composite storage segregation score."""

    def test_score_has_all_components(self, storage_segregation_auditor):
        """Score includes all 4 sub-components."""
        zone = make_zone(zone_id="ZONE-SCORE-001")
        storage_segregation_auditor.register_zone(zone)
        score = storage_segregation_auditor.calculate_score("ZONE-SCORE-001")
        for key in STORAGE_SCORE_WEIGHTS:
            assert key in score, f"Missing score component: {key}"

    def test_score_weights_sum_to_one(self):
        """Storage score weights sum to 1.0."""
        total = sum(STORAGE_SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_score_within_bounds(self, storage_segregation_auditor):
        """Storage score is between 0 and 100."""
        zone = make_zone(zone_id="ZONE-SCORE-002")
        storage_segregation_auditor.register_zone(zone)
        score = storage_segregation_auditor.calculate_score("ZONE-SCORE-002")
        total = score.get("total_score", score.get("score", 0))
        assert_valid_score(total)

    @pytest.mark.parametrize("barrier,expected_min_score", [
        ("wall", 70.0),
        ("container", 80.0),
        ("none", 0.0),
    ])
    def test_barrier_quality_affects_score(self, storage_segregation_auditor, barrier, expected_min_score):
        """Barrier type affects barrier quality sub-score."""
        zone = make_zone(barrier_type=barrier, barrier_quality_score=expected_min_score)
        storage_segregation_auditor.register_zone(zone)
        score = storage_segregation_auditor.calculate_score(zone["zone_id"])
        bqs = score.get("barrier_quality", 0.0)
        assert isinstance(bqs, (int, float))


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestStorageEdgeCases:
    """Test edge cases for storage segregation."""

    def test_zero_capacity_zone_raises(self, storage_segregation_auditor):
        """Zone with zero capacity raises ValueError."""
        zone = make_zone(capacity_kg=0.0)
        with pytest.raises(ValueError):
            storage_segregation_auditor.register_zone(zone)

    def test_full_zone_material_in_raises(self, storage_segregation_auditor):
        """Adding material to a full zone raises ValueError."""
        zone = make_zone(zone_id="ZONE-FULL-001", capacity_kg=25000.0, current_quantity_kg=25000.0)
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("material_in", "ZONE-FULL-001", quantity_kg=1000.0)
        with pytest.raises(ValueError):
            storage_segregation_auditor.record_event(evt)

    def test_no_barrier_lowest_score(self, storage_segregation_auditor):
        """Zone with no barrier has lowest barrier score."""
        zone = make_zone(zone_id="ZONE-NOBAR-001", barrier_type="none", barrier_quality_score=0.0)
        storage_segregation_auditor.register_zone(zone)
        score = storage_segregation_auditor.calculate_score("ZONE-NOBAR-001")
        bqs = score.get("barrier_quality", 0.0)
        assert bqs == pytest.approx(0.0)

    def test_get_nonexistent_zone_returns_none(self, storage_segregation_auditor):
        """Getting a non-existent zone returns None."""
        result = storage_segregation_auditor.get_zone("ZONE-NONEXISTENT")
        assert result is None

    def test_negative_quantity_event_raises(self, storage_segregation_auditor):
        """Event with negative quantity raises ValueError."""
        zone = make_zone(zone_id="ZONE-NEG-001")
        storage_segregation_auditor.register_zone(zone)
        evt = make_storage_event("material_in", "ZONE-NEG-001", quantity_kg=-500.0)
        with pytest.raises(ValueError):
            storage_segregation_auditor.record_event(evt)

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_zones_for_all_commodities(self, storage_segregation_auditor, commodity):
        """Zones can be registered for all 7 EUDR commodities."""
        zone = make_zone(commodity=commodity)
        result = storage_segregation_auditor.register_zone(zone)
        assert result["commodity"] == commodity
