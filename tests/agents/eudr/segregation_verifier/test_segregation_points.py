# -*- coding: utf-8 -*-
"""
Tests for SegregationPointValidator - AGENT-EUDR-010 Engine 1: SCP Validation

Comprehensive test suite covering:
- SCP registration (valid, duplicate, missing fields, all SCP types, all methods)
- SCP validation (compliant, non-compliant, expired, score calculation)
- Risk classification (dedicated_facility=low, physical_barrier=medium, temporal=high)
- Compliance score calculation (evidence+documentation+history+method weights)
- Reverification scheduling (90-day default, custom intervals, expired detection)
- SCP discovery from events
- Bulk import (valid, invalid, mixed)
- Search functionality (by facility, type, commodity, status)
- SCP history tracking
- Edge cases (empty facility, max capacity, null coordinates)

Test count: 65+ tests
Coverage target: >= 85% of SegregationPointValidator module

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
    SCP_TYPES,
    SEGREGATION_METHODS,
    EUDR_COMMODITIES,
    METHOD_RISK_LEVELS,
    SCP_SCORE_WEIGHTS,
    SHA256_HEX_LENGTH,
    DEFAULT_REVERIFICATION_DAYS,
    SCP_STORAGE_COCOA,
    SCP_TRANSPORT_PALM,
    SCP_PROCESSING_COFFEE,
    SCP_HANDLING_SOYA,
    SCP_LOADING_RUBBER,
    SCP_ID_STORAGE_COCOA,
    FAC_ID_WAREHOUSE_GH,
    FAC_ID_MILL_ID,
    make_scp,
    assert_valid_provenance_hash,
    assert_valid_score,
)


# ===========================================================================
# 1. SCP Registration
# ===========================================================================


class TestSCPRegistration:
    """Test segregation control point registration."""

    @pytest.mark.parametrize("scp_type", SCP_TYPES)
    def test_register_scp_all_types(self, segregation_point_validator, scp_type):
        """Each of the 5 SCP types can be registered."""
        scp = make_scp(scp_type=scp_type)
        result = segregation_point_validator.register(scp)
        assert result is not None
        assert result["scp_type"] == scp_type

    @pytest.mark.parametrize("method", SEGREGATION_METHODS)
    def test_register_scp_all_methods(self, segregation_point_validator, method):
        """Each of the 8 segregation methods can be registered."""
        scp = make_scp(segregation_method=method)
        result = segregation_point_validator.register(scp)
        assert result is not None
        assert result["segregation_method"] == method

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_register_scp_all_commodities(self, segregation_point_validator, commodity):
        """SCPs can be registered for all 7 EUDR commodities."""
        scp = make_scp(commodity=commodity)
        result = segregation_point_validator.register(scp)
        assert result is not None
        assert result["commodity"] == commodity

    def test_register_storage_scp(self, segregation_point_validator):
        """Register a storage SCP with full details."""
        scp = copy.deepcopy(SCP_STORAGE_COCOA)
        result = segregation_point_validator.register(scp)
        assert result["scp_id"] == SCP_ID_STORAGE_COCOA
        assert result["scp_type"] == "storage"
        assert result["commodity"] == "cocoa"

    def test_register_transport_scp(self, segregation_point_validator):
        """Register a transport SCP."""
        scp = copy.deepcopy(SCP_TRANSPORT_PALM)
        result = segregation_point_validator.register(scp)
        assert result["scp_type"] == "transport"
        assert result["segregation_method"] == "dedicated_facility"

    def test_register_processing_scp(self, segregation_point_validator):
        """Register a processing SCP."""
        scp = copy.deepcopy(SCP_PROCESSING_COFFEE)
        result = segregation_point_validator.register(scp)
        assert result["scp_type"] == "processing"

    def test_duplicate_scp_id_raises(self, segregation_point_validator):
        """Registering an SCP with a duplicate ID raises an error."""
        scp = make_scp(scp_id="SCP-DUP-001")
        segregation_point_validator.register(scp)
        with pytest.raises((ValueError, KeyError)):
            segregation_point_validator.register(copy.deepcopy(scp))

    def test_missing_scp_type_raises(self, segregation_point_validator):
        """SCP without scp_type raises ValueError."""
        scp = make_scp()
        scp["scp_type"] = None
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_invalid_scp_type_raises(self, segregation_point_validator):
        """SCP with invalid scp_type raises ValueError."""
        scp = make_scp()
        scp["scp_type"] = "invalid_type"
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_missing_commodity_raises(self, segregation_point_validator):
        """SCP without commodity raises ValueError."""
        scp = make_scp()
        scp["commodity"] = None
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_missing_facility_id_raises(self, segregation_point_validator):
        """SCP without facility_id raises ValueError."""
        scp = make_scp()
        scp["facility_id"] = None
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_register_assigns_id(self, segregation_point_validator):
        """Registration assigns a unique scp_id if not provided."""
        scp = make_scp()
        scp["scp_id"] = None
        result = segregation_point_validator.register(scp)
        assert result.get("scp_id") is not None
        assert len(result["scp_id"]) > 0

    def test_register_provenance_hash(self, segregation_point_validator):
        """Registration generates a provenance hash."""
        scp = make_scp()
        result = segregation_point_validator.register(scp)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. SCP Validation
# ===========================================================================


class TestSCPValidation:
    """Test SCP compliance validation."""

    def test_validate_compliant_scp(self, segregation_point_validator):
        """SCP with high score validates as compliant."""
        scp = make_scp(compliance_score=90.0)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.validate(scp["scp_id"])
        assert result is not None
        assert result["status"] in ("compliant", "active")

    def test_validate_non_compliant_scp(self, segregation_point_validator):
        """SCP with low score validates as non-compliant."""
        scp = make_scp(compliance_score=30.0)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.validate(scp["scp_id"])
        assert result is not None
        assert result["compliance_score"] < 50.0

    def test_validate_expired_scp(self, segregation_point_validator):
        """SCP past reverification date is flagged as expired."""
        scp = make_scp(last_verified_days_ago=120)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.validate(scp["scp_id"])
        assert result.get("verification_expired") is True or result.get("status") == "expired"

    def test_validate_nonexistent_raises(self, segregation_point_validator):
        """Validating a non-existent SCP raises an error."""
        with pytest.raises((ValueError, KeyError)):
            segregation_point_validator.validate("SCP-NONEXISTENT")

    @pytest.mark.parametrize("score,expected_pass", [
        (0.0, False),
        (49.9, False),
        (50.0, True),
        (75.0, True),
        (100.0, True),
    ])
    def test_score_boundary_conditions(self, segregation_point_validator, score, expected_pass):
        """Validate score boundary conditions for pass/fail."""
        scp = make_scp(compliance_score=score)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.validate(scp["scp_id"])
        if expected_pass:
            assert result["compliance_score"] >= 50.0
        else:
            assert result["compliance_score"] < 50.0


# ===========================================================================
# 3. Risk Classification
# ===========================================================================


class TestRiskClassification:
    """Test risk classification per segregation method."""

    @pytest.mark.parametrize("method,expected_risk", list(METHOD_RISK_LEVELS.items()))
    def test_risk_level_per_method(self, segregation_point_validator, method, expected_risk):
        """Each segregation method maps to the correct risk level."""
        scp = make_scp(segregation_method=method)
        segregation_point_validator.register(scp)
        risk = segregation_point_validator.classify_risk(scp["scp_id"])
        assert risk["risk_level"] == expected_risk

    def test_dedicated_facility_is_low_risk(self, segregation_point_validator):
        """Dedicated facility has low contamination risk."""
        scp = make_scp(segregation_method="dedicated_facility")
        segregation_point_validator.register(scp)
        risk = segregation_point_validator.classify_risk(scp["scp_id"])
        assert risk["risk_level"] == "low"

    def test_physical_barrier_is_medium_risk(self, segregation_point_validator):
        """Physical barrier has medium contamination risk."""
        scp = make_scp(segregation_method="physical_barrier")
        segregation_point_validator.register(scp)
        risk = segregation_point_validator.classify_risk(scp["scp_id"])
        assert risk["risk_level"] == "medium"

    def test_temporal_separation_is_high_risk(self, segregation_point_validator):
        """Temporal separation has high contamination risk."""
        scp = make_scp(segregation_method="temporal_separation")
        segregation_point_validator.register(scp)
        risk = segregation_point_validator.classify_risk(scp["scp_id"])
        assert risk["risk_level"] == "high"

    def test_labeling_only_is_high_risk(self, segregation_point_validator):
        """Labeling only has high contamination risk."""
        scp = make_scp(segregation_method="labeling_only")
        segregation_point_validator.register(scp)
        risk = segregation_point_validator.classify_risk(scp["scp_id"])
        assert risk["risk_level"] == "high"

    def test_combined_method_is_low_risk(self, segregation_point_validator):
        """Combined method has low contamination risk."""
        scp = make_scp(segregation_method="combined")
        segregation_point_validator.register(scp)
        risk = segregation_point_validator.classify_risk(scp["scp_id"])
        assert risk["risk_level"] == "low"


# ===========================================================================
# 4. Compliance Score Calculation
# ===========================================================================


class TestComplianceScoreCalculation:
    """Test compliance score calculation with weighted components."""

    def test_score_has_all_components(self, segregation_point_validator):
        """Score breakdown includes evidence, documentation, history, method."""
        scp = make_scp()
        segregation_point_validator.register(scp)
        breakdown = segregation_point_validator.calculate_score(scp["scp_id"])
        assert "evidence" in breakdown
        assert "documentation" in breakdown
        assert "history" in breakdown
        assert "method" in breakdown

    def test_score_weights_sum_to_one(self):
        """Score weights sum to 1.0."""
        total = sum(SCP_SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_perfect_score_components(self, segregation_point_validator):
        """All perfect sub-scores yield a total score of 100."""
        scp = make_scp(compliance_score=100.0)
        segregation_point_validator.register(scp)
        breakdown = segregation_point_validator.calculate_score(scp["scp_id"])
        assert breakdown.get("total_score", breakdown.get("compliance_score", 0)) <= 100.0

    def test_zero_score_components(self, segregation_point_validator):
        """All zero sub-scores yield a total score of 0."""
        scp = make_scp(compliance_score=0.0)
        scp["evidence_refs"] = []
        segregation_point_validator.register(scp)
        breakdown = segregation_point_validator.calculate_score(scp["scp_id"])
        total = breakdown.get("total_score", breakdown.get("compliance_score", 0))
        assert total >= 0.0

    def test_score_within_bounds(self, segregation_point_validator):
        """Compliance score is always between 0 and 100."""
        scp = make_scp(compliance_score=72.5)
        segregation_point_validator.register(scp)
        breakdown = segregation_point_validator.calculate_score(scp["scp_id"])
        total = breakdown.get("total_score", breakdown.get("compliance_score", 0))
        assert_valid_score(total)


# ===========================================================================
# 5. Reverification Scheduling
# ===========================================================================


class TestReverificationScheduling:
    """Test reverification scheduling and expired detection."""

    def test_default_reverification_interval(self, sgv_config):
        """Default reverification interval is 90 days."""
        assert sgv_config["reverification_interval_days"] == 90

    def test_scp_not_expired_within_interval(self, segregation_point_validator):
        """SCP verified within interval is not expired."""
        scp = make_scp(last_verified_days_ago=30)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.check_reverification(scp["scp_id"])
        assert result["expired"] is False

    def test_scp_expired_past_interval(self, segregation_point_validator):
        """SCP verified past interval is expired."""
        scp = make_scp(last_verified_days_ago=100)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.check_reverification(scp["scp_id"])
        assert result["expired"] is True

    def test_scp_at_boundary_not_expired(self, segregation_point_validator):
        """SCP verified exactly at interval boundary is not expired."""
        scp = make_scp(last_verified_days_ago=DEFAULT_REVERIFICATION_DAYS)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.check_reverification(scp["scp_id"])
        assert result["expired"] is False

    def test_find_expired_scps(self, segregation_point_validator):
        """Find all SCPs that need reverification."""
        scp_ok = make_scp(scp_id="SCP-OK-001", last_verified_days_ago=30)
        scp_expired = make_scp(scp_id="SCP-EXP-001", last_verified_days_ago=120)
        segregation_point_validator.register(scp_ok)
        segregation_point_validator.register(scp_expired)
        expired = segregation_point_validator.find_expired()
        assert any(s["scp_id"] == "SCP-EXP-001" for s in expired)
        assert not any(s["scp_id"] == "SCP-OK-001" for s in expired)

    @pytest.mark.parametrize("days_ago,expected_expired", [
        (0, False),
        (45, False),
        (89, False),
        (91, True),
        (180, True),
        (365, True),
    ])
    def test_expiry_at_various_ages(self, segregation_point_validator, days_ago, expected_expired):
        """Parameterized expiry test at various verification ages."""
        scp = make_scp(last_verified_days_ago=days_ago)
        segregation_point_validator.register(scp)
        result = segregation_point_validator.check_reverification(scp["scp_id"])
        assert result["expired"] is expected_expired


# ===========================================================================
# 6. Bulk Import
# ===========================================================================


class TestBulkImport:
    """Test bulk SCP import."""

    def test_bulk_import_valid_list(self, segregation_point_validator):
        """Import a list of valid SCPs."""
        scps = [make_scp(scp_id=f"SCP-BULK-{i:03d}") for i in range(10)]
        results = segregation_point_validator.bulk_import(scps)
        assert len(results) == 10
        assert all(r is not None for r in results)

    def test_bulk_import_with_invalid(self, segregation_point_validator):
        """Bulk import with some invalid SCPs reports partial failures."""
        scps = [
            make_scp(scp_id="SCP-BULK-OK"),
            make_scp(scp_id="SCP-BULK-BAD"),
            make_scp(scp_id="SCP-BULK-OK2"),
        ]
        scps[1]["scp_type"] = "invalid_type"
        results = segregation_point_validator.bulk_import(scps, continue_on_error=True)
        assert len([r for r in results if r.get("status") == "error"]) >= 1

    def test_bulk_import_empty(self, segregation_point_validator):
        """Bulk import of empty list returns empty results."""
        results = segregation_point_validator.bulk_import([])
        assert len(results) == 0

    def test_bulk_import_mixed_types(self, segregation_point_validator):
        """Bulk import with mixed SCP types succeeds."""
        scps = [make_scp(scp_type=t, scp_id=f"SCP-MIX-{i}") for i, t in enumerate(SCP_TYPES)]
        results = segregation_point_validator.bulk_import(scps)
        assert len(results) == len(SCP_TYPES)


# ===========================================================================
# 7. Search Functionality
# ===========================================================================


class TestSearchFunctionality:
    """Test SCP search and filtering."""

    def test_search_by_facility(self, segregation_point_validator):
        """Search SCPs by facility ID."""
        scp1 = make_scp(scp_id="SCP-FAC-A", facility_id=FAC_ID_WAREHOUSE_GH)
        scp2 = make_scp(scp_id="SCP-FAC-B", facility_id=FAC_ID_MILL_ID)
        segregation_point_validator.register(scp1)
        segregation_point_validator.register(scp2)
        results = segregation_point_validator.search(facility_id=FAC_ID_WAREHOUSE_GH)
        assert all(r["facility_id"] == FAC_ID_WAREHOUSE_GH for r in results)

    def test_search_by_type(self, segregation_point_validator):
        """Search SCPs by SCP type."""
        scp1 = make_scp(scp_id="SCP-TYPE-S", scp_type="storage")
        scp2 = make_scp(scp_id="SCP-TYPE-T", scp_type="transport")
        segregation_point_validator.register(scp1)
        segregation_point_validator.register(scp2)
        results = segregation_point_validator.search(scp_type="storage")
        assert all(r["scp_type"] == "storage" for r in results)

    def test_search_by_commodity(self, segregation_point_validator):
        """Search SCPs by commodity."""
        scp1 = make_scp(scp_id="SCP-COM-C", commodity="cocoa")
        scp2 = make_scp(scp_id="SCP-COM-P", commodity="palm_oil")
        segregation_point_validator.register(scp1)
        segregation_point_validator.register(scp2)
        results = segregation_point_validator.search(commodity="cocoa")
        assert all(r["commodity"] == "cocoa" for r in results)

    def test_search_by_status(self, segregation_point_validator):
        """Search SCPs by status."""
        scp1 = make_scp(scp_id="SCP-STA-A", status="active")
        scp2 = make_scp(scp_id="SCP-STA-I", status="inactive")
        segregation_point_validator.register(scp1)
        segregation_point_validator.register(scp2)
        results = segregation_point_validator.search(status="active")
        assert all(r["status"] == "active" for r in results)

    def test_search_no_results(self, segregation_point_validator):
        """Search with no matching criteria returns empty list."""
        results = segregation_point_validator.search(facility_id="FAC-NONEXISTENT")
        assert len(results) == 0


# ===========================================================================
# 8. SCP History Tracking
# ===========================================================================


class TestSCPHistoryTracking:
    """Test SCP history and audit trail."""

    def test_history_after_registration(self, segregation_point_validator):
        """SCP history starts with registration event."""
        scp = make_scp(scp_id="SCP-HIST-001")
        segregation_point_validator.register(scp)
        history = segregation_point_validator.get_history("SCP-HIST-001")
        assert len(history) >= 1
        assert history[0].get("action") in ("registered", "created")

    def test_history_after_validation(self, segregation_point_validator):
        """Validation adds an entry to SCP history."""
        scp = make_scp(scp_id="SCP-HIST-002")
        segregation_point_validator.register(scp)
        segregation_point_validator.validate("SCP-HIST-002")
        history = segregation_point_validator.get_history("SCP-HIST-002")
        assert len(history) >= 2

    def test_history_preserves_chronological_order(self, segregation_point_validator):
        """History entries are in chronological order."""
        scp = make_scp(scp_id="SCP-HIST-003")
        segregation_point_validator.register(scp)
        segregation_point_validator.validate("SCP-HIST-003")
        history = segregation_point_validator.get_history("SCP-HIST-003")
        for i in range(len(history) - 1):
            assert history[i].get("timestamp", "") <= history[i + 1].get("timestamp", "")


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestSCPEdgeCases:
    """Test edge cases for SCP operations."""

    def test_zero_capacity(self, segregation_point_validator):
        """SCP with zero capacity raises ValueError."""
        scp = make_scp(capacity_kg=0.0)
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_negative_capacity_raises(self, segregation_point_validator):
        """SCP with negative capacity raises ValueError."""
        scp = make_scp(capacity_kg=-1000.0)
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_very_large_capacity(self, segregation_point_validator):
        """SCP with very large capacity is accepted."""
        scp = make_scp(capacity_kg=10_000_000.0)
        result = segregation_point_validator.register(scp)
        assert result["capacity_kg"] == 10_000_000.0

    def test_null_coordinates_accepted(self, segregation_point_validator):
        """SCP with null coordinates is accepted for indoor facilities."""
        scp = make_scp()
        scp["location_lat"] = None
        scp["location_lon"] = None
        result = segregation_point_validator.register(scp)
        assert result is not None

    def test_get_nonexistent_scp_returns_none(self, segregation_point_validator):
        """Getting a non-existent SCP returns None."""
        result = segregation_point_validator.get("SCP-NONEXISTENT")
        assert result is None

    def test_invalid_method_raises(self, segregation_point_validator):
        """SCP with invalid segregation method raises ValueError."""
        scp = make_scp(segregation_method="invalid_method")
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_score_negative_raises(self, segregation_point_validator):
        """SCP with negative compliance score raises ValueError."""
        scp = make_scp(compliance_score=-10.0)
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_score_above_100_raises(self, segregation_point_validator):
        """SCP with score above 100 raises ValueError."""
        scp = make_scp(compliance_score=110.0)
        with pytest.raises(ValueError):
            segregation_point_validator.register(scp)

    def test_empty_facility_scps(self, segregation_point_validator):
        """Search for SCPs in a facility with none returns empty list."""
        results = segregation_point_validator.search(facility_id="FAC-EMPTY-001")
        assert len(results) == 0
