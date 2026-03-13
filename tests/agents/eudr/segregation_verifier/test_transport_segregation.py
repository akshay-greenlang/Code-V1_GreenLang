# -*- coding: utf-8 -*-
"""
Tests for TransportSegregationTracker - AGENT-EUDR-010 Engine 3: Transport Segregation

Comprehensive test suite covering:
- Vehicle registration (all 10 transport types, dedicated vs shared)
- Transport verification (dedicated vehicle=pass, shared+cleaned=pass, shared+unclean=fail)
- Cleaning verification (all cleaning methods, duration validation, certificate check)
- Previous cargo tracking (clean history=low risk, contaminated history=high risk)
- Seal integrity verification (applied, intact, broken)
- Dedicated vehicle bonus scoring
- Route segregation analysis (no non-compliant stops=pass)
- Transport score calculation (composite)
- Multi-modal transport verification
- Edge cases (no history, expired cleaning, missing seals)

Test count: 65+ tests
Coverage target: >= 85% of TransportSegregationTracker module

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
    TRANSPORT_TYPES,
    CLEANING_METHODS,
    TRANSPORT_SCORE_WEIGHTS,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    VEHICLE_DEDICATED_TRUCK,
    VEHICLE_SHARED_CONTAINER,
    VEHICLE_ID_TRUCK_01,
    VEHICLE_ID_CONTAINER_01,
    FAC_ID_WAREHOUSE_GH,
    FAC_ID_MILL_ID,
    BATCH_ID_COCOA_001,
    BATCH_ID_PALM_001,
    make_vehicle,
    assert_valid_score,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Vehicle Registration
# ===========================================================================


class TestVehicleRegistration:
    """Test transport vehicle registration."""

    @pytest.mark.parametrize("vehicle_type", TRANSPORT_TYPES)
    def test_register_all_transport_types(self, transport_segregation_tracker, vehicle_type):
        """Each of the 10 transport types can be registered."""
        vehicle = make_vehicle(vehicle_type=vehicle_type)
        result = transport_segregation_tracker.register_vehicle(vehicle)
        assert result is not None
        assert result["vehicle_type"] == vehicle_type

    def test_register_dedicated_truck(self, transport_segregation_tracker):
        """Register a dedicated truck with full details."""
        vehicle = copy.deepcopy(VEHICLE_DEDICATED_TRUCK)
        result = transport_segregation_tracker.register_vehicle(vehicle)
        assert result["vehicle_id"] == VEHICLE_ID_TRUCK_01
        assert result["dedicated"] is True
        assert result["dedicated_commodity"] == "cocoa"

    def test_register_shared_container(self, transport_segregation_tracker):
        """Register a shared container."""
        vehicle = copy.deepcopy(VEHICLE_SHARED_CONTAINER)
        result = transport_segregation_tracker.register_vehicle(vehicle)
        assert result["vehicle_id"] == VEHICLE_ID_CONTAINER_01
        assert result["dedicated"] is False

    def test_duplicate_vehicle_id_raises(self, transport_segregation_tracker):
        """Registering a vehicle with duplicate ID raises an error."""
        vehicle = make_vehicle(vehicle_id="VEH-DUP-001")
        transport_segregation_tracker.register_vehicle(vehicle)
        with pytest.raises((ValueError, KeyError)):
            transport_segregation_tracker.register_vehicle(copy.deepcopy(vehicle))

    def test_missing_vehicle_type_raises(self, transport_segregation_tracker):
        """Vehicle without vehicle_type raises ValueError."""
        vehicle = make_vehicle()
        vehicle["vehicle_type"] = None
        with pytest.raises(ValueError):
            transport_segregation_tracker.register_vehicle(vehicle)

    def test_register_provenance_hash(self, transport_segregation_tracker):
        """Vehicle registration generates a provenance hash."""
        vehicle = make_vehicle()
        result = transport_segregation_tracker.register_vehicle(vehicle)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH

    @pytest.mark.parametrize("dedicated,commodity", [
        (True, "cocoa"),
        (True, "palm_oil"),
        (True, "coffee"),
        (False, None),
    ])
    def test_dedicated_status_combinations(self, transport_segregation_tracker, dedicated, commodity):
        """Dedicated and non-dedicated vehicles are registered correctly."""
        vehicle = make_vehicle(dedicated=dedicated, dedicated_commodity=commodity)
        result = transport_segregation_tracker.register_vehicle(vehicle)
        assert result["dedicated"] is dedicated


# ===========================================================================
# 2. Transport Verification
# ===========================================================================


class TestTransportVerification:
    """Test transport segregation verification."""

    def test_dedicated_vehicle_passes(self, transport_segregation_tracker):
        """Dedicated vehicle passes verification for its commodity."""
        vehicle = make_vehicle(vehicle_id="VEH-DED-PASS", dedicated=True,
                               dedicated_commodity="cocoa")
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_transport(
            "VEH-DED-PASS", commodity="cocoa", batch_id=BATCH_ID_COCOA_001
        )
        assert result["compliant"] is True

    def test_shared_cleaned_passes(self, transport_segregation_tracker):
        """Shared vehicle with cleaning certificate passes verification."""
        vehicle = make_vehicle(vehicle_id="VEH-SHR-CLN", dedicated=False,
                               cleaning_method="water_wash", cleaning_duration_minutes=90)
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_transport(
            "VEH-SHR-CLN", commodity="cocoa", batch_id=BATCH_ID_COCOA_001
        )
        assert result["compliant"] is True

    def test_shared_uncleaned_fails(self, transport_segregation_tracker):
        """Shared vehicle without proper cleaning fails verification."""
        vehicle = make_vehicle(vehicle_id="VEH-SHR-UNCLN", dedicated=False,
                               cleaning_method="dry_sweep", cleaning_duration_minutes=5)
        vehicle["cleaning_certificate"] = None
        vehicle["previous_cargo"] = [
            {"commodity": "soya", "batch_id": "BATCH-X", "compliant": False},
        ]
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_transport(
            "VEH-SHR-UNCLN", commodity="cocoa", batch_id=BATCH_ID_COCOA_001
        )
        assert result["compliant"] is False

    def test_dedicated_wrong_commodity_fails(self, transport_segregation_tracker):
        """Dedicated vehicle used for wrong commodity fails verification."""
        vehicle = make_vehicle(vehicle_id="VEH-DED-WRONG", dedicated=True,
                               dedicated_commodity="palm_oil")
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_transport(
            "VEH-DED-WRONG", commodity="cocoa", batch_id=BATCH_ID_COCOA_001
        )
        assert result["compliant"] is False

    def test_verify_nonexistent_vehicle_raises(self, transport_segregation_tracker):
        """Verifying a non-existent vehicle raises an error."""
        with pytest.raises((ValueError, KeyError)):
            transport_segregation_tracker.verify_transport(
                "VEH-NONEXISTENT", commodity="cocoa", batch_id=BATCH_ID_COCOA_001
            )


# ===========================================================================
# 3. Cleaning Verification
# ===========================================================================


class TestCleaningVerification:
    """Test vehicle cleaning verification."""

    @pytest.mark.parametrize("method", CLEANING_METHODS)
    def test_all_cleaning_methods(self, transport_segregation_tracker, method):
        """Each cleaning method can be recorded and verified."""
        vehicle = make_vehicle(vehicle_id=f"VEH-CLN-{method}", cleaning_method=method)
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_cleaning(f"VEH-CLN-{method}")
        assert result is not None
        assert result.get("cleaning_method") == method

    def test_cleaning_with_certificate_passes(self, transport_segregation_tracker):
        """Vehicle with valid cleaning certificate passes."""
        vehicle = make_vehicle(vehicle_id="VEH-CERT-OK")
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_cleaning("VEH-CERT-OK")
        assert result.get("certificate_valid") is True

    def test_cleaning_without_certificate_flagged(self, transport_segregation_tracker):
        """Vehicle without cleaning certificate is flagged."""
        vehicle = make_vehicle(vehicle_id="VEH-NO-CERT")
        vehicle["cleaning_certificate"] = None
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_cleaning("VEH-NO-CERT")
        assert result.get("certificate_valid") is False

    @pytest.mark.parametrize("duration,expected_adequate", [
        (10, False),
        (30, True),
        (45, True),
        (90, True),
        (120, True),
    ])
    def test_cleaning_duration_validation(self, transport_segregation_tracker, duration, expected_adequate):
        """Cleaning duration is validated against minimum requirement."""
        vehicle = make_vehicle(cleaning_duration_minutes=duration)
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_cleaning(vehicle["vehicle_id"])
        if expected_adequate:
            assert result.get("duration_adequate") is True or result.get("compliant") is True
        else:
            assert result.get("duration_adequate") is False or result.get("compliant") is False

    def test_expired_cleaning_flagged(self, transport_segregation_tracker):
        """Vehicle with expired cleaning is flagged."""
        vehicle = make_vehicle(vehicle_id="VEH-EXP-CLN")
        vehicle["last_cleaned"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_cleaning("VEH-EXP-CLN")
        assert result.get("cleaning_current") is False or result.get("expired") is True


# ===========================================================================
# 4. Previous Cargo Tracking
# ===========================================================================


class TestPreviousCargoTracking:
    """Test previous cargo risk assessment."""

    def test_clean_history_low_risk(self, transport_segregation_tracker):
        """Vehicle with only compliant cargo history has low risk."""
        vehicle = make_vehicle(vehicle_id="VEH-CLEAN-HIST", previous_cargo=[
            {"commodity": "cocoa", "batch_id": "B1", "compliant": True},
            {"commodity": "cocoa", "batch_id": "B2", "compliant": True},
        ])
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.assess_cargo_history("VEH-CLEAN-HIST")
        assert result["risk_level"] in ("low", "none")

    def test_contaminated_history_high_risk(self, transport_segregation_tracker):
        """Vehicle with non-compliant cargo history has high risk."""
        vehicle = make_vehicle(vehicle_id="VEH-DIRTY-HIST", previous_cargo=[
            {"commodity": "soya", "batch_id": "B1", "compliant": False},
            {"commodity": "palm_oil", "batch_id": "B2", "compliant": False},
        ])
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.assess_cargo_history("VEH-DIRTY-HIST")
        assert result["risk_level"] in ("high", "critical")

    def test_empty_history_medium_risk(self, transport_segregation_tracker):
        """Vehicle with no cargo history has medium risk (unknown)."""
        vehicle = make_vehicle(vehicle_id="VEH-NO-HIST", previous_cargo=[])
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.assess_cargo_history("VEH-NO-HIST")
        assert result["risk_level"] in ("medium", "unknown")

    def test_mixed_history(self, transport_segregation_tracker):
        """Vehicle with mixed cargo history has elevated risk."""
        vehicle = make_vehicle(vehicle_id="VEH-MIX-HIST", previous_cargo=[
            {"commodity": "cocoa", "batch_id": "B1", "compliant": True},
            {"commodity": "soya", "batch_id": "B2", "compliant": False},
            {"commodity": "cocoa", "batch_id": "B3", "compliant": True},
        ])
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.assess_cargo_history("VEH-MIX-HIST")
        assert result["risk_level"] not in ("low", "none")


# ===========================================================================
# 5. Seal Integrity Verification
# ===========================================================================


class TestSealIntegrity:
    """Test transport seal integrity verification."""

    def test_intact_seal_passes(self, transport_segregation_tracker):
        """Intact seal passes verification."""
        vehicle = make_vehicle(vehicle_id="VEH-SEAL-OK", seal_status="intact")
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_seal("VEH-SEAL-OK")
        assert result["seal_intact"] is True

    def test_broken_seal_fails(self, transport_segregation_tracker):
        """Broken seal fails verification."""
        vehicle = make_vehicle(vehicle_id="VEH-SEAL-BRK", seal_status="broken")
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_seal("VEH-SEAL-BRK")
        assert result["seal_intact"] is False

    def test_missing_seal_fails(self, transport_segregation_tracker):
        """Missing seal fails verification."""
        vehicle = make_vehicle(vehicle_id="VEH-NO-SEAL")
        vehicle["seal_number"] = None
        vehicle["seal_status"] = None
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_seal("VEH-NO-SEAL")
        assert result["seal_intact"] is False

    @pytest.mark.parametrize("seal_status,expected_intact", [
        ("intact", True),
        ("applied", True),
        ("broken", False),
        ("tampered", False),
        ("missing", False),
    ])
    def test_seal_status_outcomes(self, transport_segregation_tracker, seal_status, expected_intact):
        """Various seal statuses produce correct verification outcomes."""
        vehicle = make_vehicle(seal_status=seal_status)
        transport_segregation_tracker.register_vehicle(vehicle)
        result = transport_segregation_tracker.verify_seal(vehicle["vehicle_id"])
        assert result["seal_intact"] is expected_intact


# ===========================================================================
# 6. Dedicated Vehicle Bonus
# ===========================================================================


class TestDedicatedVehicleBonus:
    """Test dedicated vehicle bonus scoring."""

    def test_dedicated_gets_bonus(self, transport_segregation_tracker):
        """Dedicated vehicle gets bonus in transport score."""
        vehicle = make_vehicle(vehicle_id="VEH-BONUS", dedicated=True,
                               dedicated_commodity="cocoa")
        transport_segregation_tracker.register_vehicle(vehicle)
        score = transport_segregation_tracker.calculate_score("VEH-BONUS")
        assert score.get("vehicle_dedication", 0) >= 80.0

    def test_shared_no_bonus(self, transport_segregation_tracker):
        """Shared vehicle does not get dedication bonus."""
        vehicle = make_vehicle(vehicle_id="VEH-NO-BONUS", dedicated=False)
        transport_segregation_tracker.register_vehicle(vehicle)
        score = transport_segregation_tracker.calculate_score("VEH-NO-BONUS")
        assert score.get("vehicle_dedication", 0) < 80.0


# ===========================================================================
# 7. Route Segregation Analysis
# ===========================================================================


class TestRouteSegregation:
    """Test route segregation analysis."""

    def test_compliant_route_passes(self, transport_segregation_tracker):
        """Route with no non-compliant stops passes."""
        vehicle = make_vehicle(vehicle_id="VEH-ROUTE-OK")
        transport_segregation_tracker.register_vehicle(vehicle)
        route = {
            "stops": [
                {"facility_id": FAC_ID_WAREHOUSE_GH, "compliant": True},
                {"facility_id": FAC_ID_MILL_ID, "compliant": True},
            ]
        }
        result = transport_segregation_tracker.verify_route("VEH-ROUTE-OK", route)
        assert result["compliant"] is True

    def test_non_compliant_stop_fails(self, transport_segregation_tracker):
        """Route with a non-compliant stop fails."""
        vehicle = make_vehicle(vehicle_id="VEH-ROUTE-BAD")
        transport_segregation_tracker.register_vehicle(vehicle)
        route = {
            "stops": [
                {"facility_id": FAC_ID_WAREHOUSE_GH, "compliant": True},
                {"facility_id": "FAC-NONCOMPLIANT", "compliant": False},
            ]
        }
        result = transport_segregation_tracker.verify_route("VEH-ROUTE-BAD", route)
        assert result["compliant"] is False

    def test_empty_route_passes(self, transport_segregation_tracker):
        """Direct route with no intermediate stops passes."""
        vehicle = make_vehicle(vehicle_id="VEH-DIRECT")
        transport_segregation_tracker.register_vehicle(vehicle)
        route = {"stops": []}
        result = transport_segregation_tracker.verify_route("VEH-DIRECT", route)
        assert result["compliant"] is True


# ===========================================================================
# 8. Transport Score Calculation
# ===========================================================================


class TestTransportScore:
    """Test composite transport score calculation."""

    def test_score_has_all_components(self, transport_segregation_tracker):
        """Transport score has all 4 sub-components."""
        vehicle = make_vehicle(vehicle_id="VEH-SCORE-001")
        transport_segregation_tracker.register_vehicle(vehicle)
        score = transport_segregation_tracker.calculate_score("VEH-SCORE-001")
        for key in TRANSPORT_SCORE_WEIGHTS:
            assert key in score, f"Missing score component: {key}"

    def test_score_weights_sum_to_one(self):
        """Transport score weights sum to 1.0."""
        total = sum(TRANSPORT_SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_score_within_bounds(self, transport_segregation_tracker):
        """Transport score is between 0 and 100."""
        vehicle = make_vehicle(vehicle_id="VEH-SCORE-002")
        transport_segregation_tracker.register_vehicle(vehicle)
        score = transport_segregation_tracker.calculate_score("VEH-SCORE-002")
        total = score.get("total_score", score.get("score", 0))
        assert_valid_score(total)


# ===========================================================================
# 9. Multi-Modal Transport
# ===========================================================================


class TestMultiModalTransport:
    """Test multi-modal transport verification."""

    def test_multi_modal_chain_verification(self, transport_segregation_tracker):
        """Verify segregation across multi-modal transport chain."""
        truck = make_vehicle(vehicle_id="VEH-MM-TRK", vehicle_type="truck", dedicated=True)
        container = make_vehicle(vehicle_id="VEH-MM-CNT", vehicle_type="container", dedicated=True)
        transport_segregation_tracker.register_vehicle(truck)
        transport_segregation_tracker.register_vehicle(container)
        chain = [
            {"vehicle_id": "VEH-MM-TRK", "leg": 1, "mode": "road"},
            {"vehicle_id": "VEH-MM-CNT", "leg": 2, "mode": "sea"},
        ]
        result = transport_segregation_tracker.verify_multi_modal(chain, commodity="cocoa")
        assert result is not None

    def test_multi_modal_one_non_compliant_fails(self, transport_segregation_tracker):
        """Multi-modal chain fails if any leg is non-compliant."""
        truck = make_vehicle(vehicle_id="VEH-MMF-TRK", vehicle_type="truck", dedicated=True)
        container = make_vehicle(vehicle_id="VEH-MMF-CNT", vehicle_type="container",
                                 dedicated=False)
        container["cleaning_certificate"] = None
        container["previous_cargo"] = [
            {"commodity": "soya", "batch_id": "X", "compliant": False},
        ]
        transport_segregation_tracker.register_vehicle(truck)
        transport_segregation_tracker.register_vehicle(container)
        chain = [
            {"vehicle_id": "VEH-MMF-TRK", "leg": 1, "mode": "road"},
            {"vehicle_id": "VEH-MMF-CNT", "leg": 2, "mode": "sea"},
        ]
        result = transport_segregation_tracker.verify_multi_modal(chain, commodity="cocoa")
        assert result.get("compliant") is False


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestTransportEdgeCases:
    """Test edge cases for transport segregation."""

    def test_get_nonexistent_vehicle_returns_none(self, transport_segregation_tracker):
        """Getting a non-existent vehicle returns None."""
        result = transport_segregation_tracker.get_vehicle("VEH-NONEXISTENT")
        assert result is None

    def test_zero_capacity_vehicle_raises(self, transport_segregation_tracker):
        """Vehicle with zero capacity raises ValueError."""
        vehicle = make_vehicle(capacity_kg=0.0)
        with pytest.raises(ValueError):
            transport_segregation_tracker.register_vehicle(vehicle)

    def test_negative_cleaning_duration_raises(self, transport_segregation_tracker):
        """Vehicle with negative cleaning duration raises ValueError."""
        vehicle = make_vehicle(cleaning_duration_minutes=-10)
        with pytest.raises(ValueError):
            transport_segregation_tracker.register_vehicle(vehicle)

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_vehicles_for_all_commodities(self, transport_segregation_tracker, commodity):
        """Vehicles can be registered and verified for all 7 EUDR commodities."""
        vehicle = make_vehicle(dedicated=True, dedicated_commodity=commodity)
        result = transport_segregation_tracker.register_vehicle(vehicle)
        assert result is not None
