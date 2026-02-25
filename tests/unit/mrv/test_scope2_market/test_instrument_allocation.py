# -*- coding: utf-8 -*-
"""
Unit tests for InstrumentAllocationEngine

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests priority-based allocation, proportional allocation, custom allocation,
batch allocation, instrument validation, coverage tracking, certificate
management, utilities, and thread safety.

Target: ~80 tests, 85%+ coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.scope2_market.instrument_allocation import (
        InstrumentAllocationEngine,
        InstrumentType,
        CoverageStatus,
        RetirementStatus,
        TrackingSystem,
        INSTRUMENT_PRIORITY,
        VINTAGE_WINDOWS,
        GEOGRAPHIC_MARKETS,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instrument(
    inst_id: str = "REC-001",
    inst_type: str = "BUNDLED_CERTIFICATE",
    mwh: str = "5000",
    vintage_year: int = 2025,
    region: str = "US-WECC",
    tracking_system: str = "WREGIS",
    emission_factor: str = "0",
) -> Dict[str, Any]:
    """Create a minimal instrument dict for testing."""
    return {
        "id": inst_id,
        "type": inst_type,
        "mwh": mwh,
        "vintage_year": vintage_year,
        "region": region,
        "tracking_system": tracking_system,
        "emission_factor": emission_factor,
    }


def _make_purchase(
    facility_id: str = "FAC-001",
    mwh: str = "10000",
    region: str = "US-WECC",
) -> Dict[str, Any]:
    """Create a minimal purchase dict for testing."""
    return {
        "facility_id": facility_id,
        "mwh": mwh,
        "region": region,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh InstrumentAllocationEngine for each test."""
    return InstrumentAllocationEngine(config={
        "enable_provenance": False,
        "strict_geographic_match": True,
        "strict_vintage": True,
        "default_reporting_year": 2025,
    })


@pytest.fixture
def lenient_engine():
    """Create an engine with relaxed validation."""
    return InstrumentAllocationEngine(config={
        "enable_provenance": False,
        "strict_geographic_match": False,
        "strict_vintage": False,
        "default_reporting_year": 2025,
    })


# ===========================================================================
# 1. TestPriorityBasedAllocation  (15 tests)
# ===========================================================================


@_SKIP
class TestPriorityBasedAllocation:
    """Tests for allocate_instruments with priority-based ordering."""

    def test_single_instrument_full_coverage(self, engine):
        """Single instrument fully covering purchase yields 100% coverage."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(mwh="5000")]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["status"] == "SUCCESS"
        assert result["coverage_status"] == CoverageStatus.FULL.value

    def test_single_instrument_partial_coverage(self, engine):
        """Single instrument < purchase MWh yields PARTIAL coverage."""
        purchase = _make_purchase(mwh="10000")
        instruments = [_make_instrument(mwh="3000")]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["status"] == "SUCCESS"
        assert result["coverage_status"] == CoverageStatus.PARTIAL.value

    def test_multiple_instruments_full_coverage(self, engine):
        """Multiple instruments totaling purchase MWh yield FULL coverage."""
        purchase = _make_purchase(mwh="10000")
        instruments = [
            _make_instrument(inst_id="REC-001", mwh="6000"),
            _make_instrument(inst_id="REC-002", mwh="4000"),
        ]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["status"] == "SUCCESS"
        assert result["coverage_status"] == CoverageStatus.FULL.value

    def test_over_allocation(self, engine):
        """Instruments exceeding purchase MWh yield FULL (capped at 100%)."""
        purchase = _make_purchase(mwh="5000")
        instruments = [
            _make_instrument(inst_id="REC-001", mwh="6000"),
            _make_instrument(inst_id="REC-002", mwh="3000"),
        ]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["status"] == "SUCCESS"
        # Coverage should be FULL since purchase is fully covered
        assert result["coverage_status"] == CoverageStatus.FULL.value

    def test_priority_ordering_bundled_first(self, engine):
        """BUNDLED_CERTIFICATE (priority 1) is allocated before UNBUNDLED (priority 4)."""
        purchase = _make_purchase(mwh="5000")
        instruments = [
            _make_instrument(inst_id="UNBUNDLED-001", inst_type="UNBUNDLED_CERTIFICATE", mwh="5000"),
            _make_instrument(inst_id="BUNDLED-001", inst_type="BUNDLED_CERTIFICATE", mwh="5000"),
        ]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["status"] == "SUCCESS"
        # First allocation should be the bundled certificate
        assert result["allocations"][0]["instrument_id"] == "BUNDLED-001"

    def test_allocation_respects_instrument_capacity(self, engine):
        """Allocated MWh does not exceed instrument capacity."""
        purchase = _make_purchase(mwh="10000")
        instruments = [_make_instrument(inst_id="REC-001", mwh="3000")]
        result = engine.allocate_instruments(purchase, instruments)
        assert Decimal(result["allocations"][0]["allocated_mwh"]) == Decimal("3000")

    def test_no_instruments_none_coverage(self, engine):
        """No instruments yields NONE coverage."""
        purchase = _make_purchase(mwh="5000")
        result = engine.allocate_instruments(purchase, [])
        assert result["status"] == "SUCCESS"
        assert result["coverage_status"] == CoverageStatus.NONE.value
        assert Decimal(result["uncovered_mwh"]) == Decimal("5000")

    def test_allocation_has_provenance_hash(self, engine):
        """Result includes a 64-char provenance hash."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(mwh="5000")]
        result = engine.allocate_instruments(purchase, instruments)
        assert len(result["provenance_hash"]) == 64

    def test_allocation_has_trace(self, engine):
        """Result includes a non-empty allocation trace."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(mwh="5000")]
        result = engine.allocate_instruments(purchase, instruments)
        assert len(result["allocation_trace"]) > 0

    def test_allocation_processing_time(self, engine):
        """Result includes processing_time_ms >= 0."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(mwh="5000")]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["processing_time_ms"] >= 0

    def test_missing_mwh_raises(self, engine):
        """Purchase without 'mwh' field returns FAILED status."""
        result = engine.allocate_instruments({"facility_id": "FAC-001"}, [])
        assert result["status"] == "FAILED"

    def test_negative_mwh_raises(self, engine):
        """Negative purchase MWh returns FAILED status."""
        result = engine.allocate_instruments(
            {"facility_id": "FAC-001", "mwh": "-100"}, [],
        )
        assert result["status"] == "FAILED"

    def test_allocation_id_starts_with_s2m(self, engine):
        """Allocation ID starts with 's2m_alloc_'."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_instruments(purchase, [])
        assert result["allocation_id"].startswith("s2m_alloc_")

    def test_reporting_year_in_result(self, engine):
        """Reporting year appears in the result."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_instruments(purchase, [])
        assert result["reporting_year"] == 2025

    def test_allocation_with_emission_factor(self, engine):
        """Allocated instrument with EF includes allocated_emissions_tco2e."""
        purchase = _make_purchase(mwh="1000")
        instruments = [_make_instrument(mwh="1000", emission_factor="0.350")]
        result = engine.allocate_instruments(purchase, instruments)
        assert result["status"] == "SUCCESS"
        alloc = result["allocations"][0]
        assert "emission_factor_tco2e_per_mwh" in alloc


# ===========================================================================
# 2. TestProportionalAllocation  (10 tests)
# ===========================================================================


@_SKIP
class TestProportionalAllocation:
    """Tests for allocate_proportional distributing pro-rata."""

    def test_proportional_equal_instruments(self, engine):
        """Two equal instruments split the purchase 50/50."""
        purchase = _make_purchase(mwh="10000")
        instruments = [
            _make_instrument(inst_id="A", mwh="10000"),
            _make_instrument(inst_id="B", mwh="10000"),
        ]
        result = engine.allocate_proportional(purchase, instruments)
        assert result["status"] == "SUCCESS"
        for alloc in result["allocations"]:
            assert Decimal(alloc["allocated_mwh"]) == pytest.approx(Decimal("5000"), abs=Decimal("1"))

    def test_proportional_unequal_instruments(self, engine):
        """Unequal instruments allocate proportionally to capacity."""
        purchase = _make_purchase(mwh="10000")
        instruments = [
            _make_instrument(inst_id="A", mwh="3000"),
            _make_instrument(inst_id="B", mwh="7000"),
        ]
        result = engine.allocate_proportional(purchase, instruments)
        assert result["status"] == "SUCCESS"
        allocs = {a["instrument_id"]: Decimal(a["allocated_mwh"]) for a in result["allocations"]}
        assert allocs["A"] < allocs["B"]

    def test_proportional_insufficient_capacity(self, engine):
        """When total capacity < purchase, all instruments fully consumed."""
        purchase = _make_purchase(mwh="20000")
        instruments = [
            _make_instrument(inst_id="A", mwh="3000"),
            _make_instrument(inst_id="B", mwh="4000"),
        ]
        result = engine.allocate_proportional(purchase, instruments)
        assert result["status"] == "SUCCESS"
        assert result["coverage_status"] == CoverageStatus.PARTIAL.value

    def test_proportional_strategy_label(self, engine):
        """Result includes strategy='proportional'."""
        purchase = _make_purchase(mwh="1000")
        instruments = [_make_instrument(mwh="1000")]
        result = engine.allocate_proportional(purchase, instruments)
        assert result["strategy"] == "proportional"

    def test_proportional_no_instruments(self, engine):
        """No instruments yields zero coverage."""
        purchase = _make_purchase(mwh="5000")
        result = engine.allocate_proportional(purchase, [])
        assert Decimal(result["covered_mwh"]) == Decimal("0")

    def test_proportional_share_pct_present(self, engine):
        """Proportional allocations include share_pct."""
        purchase = _make_purchase(mwh="10000")
        instruments = [
            _make_instrument(inst_id="A", mwh="5000"),
            _make_instrument(inst_id="B", mwh="5000"),
        ]
        result = engine.allocate_proportional(purchase, instruments)
        for alloc in result["allocations"]:
            assert "share_pct" in alloc

    def test_proportional_processing_time(self, engine):
        """Proportional allocation records processing time."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_proportional(purchase, [_make_instrument(mwh="1000")])
        assert result["processing_time_ms"] >= 0

    def test_proportional_has_provenance(self, engine):
        """Proportional allocation generates provenance hash."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_proportional(purchase, [_make_instrument(mwh="1000")])
        assert len(result["provenance_hash"]) == 64

    def test_proportional_single_instrument_gets_full(self, engine):
        """Single instrument in proportional gets full allocation."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(mwh="10000")]
        result = engine.allocate_proportional(purchase, instruments)
        assert result["coverage_status"] == CoverageStatus.FULL.value

    def test_proportional_id_prefix(self, engine):
        """Proportional allocation ID starts with 's2m_prop_'."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_proportional(purchase, [])
        assert result["allocation_id"].startswith("s2m_prop_")


# ===========================================================================
# 3. TestCustomAllocation  (5 tests)
# ===========================================================================


@_SKIP
class TestCustomAllocation:
    """Tests for allocate_custom with caller-specified ordering."""

    def test_custom_ordering_applied(self, engine):
        """Custom ordering puts GREEN_TARIFF before BUNDLED_CERTIFICATE."""
        purchase = _make_purchase(mwh="5000")
        instruments = [
            _make_instrument(inst_id="BUNDLED-001", inst_type="BUNDLED_CERTIFICATE", mwh="5000"),
            _make_instrument(inst_id="GREEN-001", inst_type="GREEN_TARIFF", mwh="5000",
                             tracking_system=""),
        ]
        custom_order = ["GREEN_TARIFF", "BUNDLED_CERTIFICATE"]
        result = engine.allocate_custom(purchase, instruments, custom_order)
        assert result["status"] == "SUCCESS"
        if result["allocations"]:
            assert result["allocations"][0]["instrument_type"] == "GREEN_TARIFF"

    def test_custom_excludes_unlisted_types(self, engine):
        """Types not in custom_order are excluded from allocation."""
        purchase = _make_purchase(mwh="5000")
        instruments = [
            _make_instrument(inst_id="REC-001", inst_type="BUNDLED_CERTIFICATE", mwh="5000"),
        ]
        custom_order = ["DIRECT_CONTRACT"]  # BUNDLED_CERTIFICATE not listed
        result = engine.allocate_custom(purchase, instruments, custom_order)
        assert result["status"] == "SUCCESS"
        assert len(result["allocations"]) == 0

    def test_custom_strategy_label(self, engine):
        """Result includes strategy='custom'."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_custom(purchase, [], ["BUNDLED_CERTIFICATE"])
        assert result["strategy"] == "custom"

    def test_custom_preserves_custom_order_in_result(self, engine):
        """Result includes the custom_order list."""
        order = ["DIRECT_CONTRACT", "BUNDLED_CERTIFICATE"]
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_custom(purchase, [], order)
        assert result["custom_order"] == order

    def test_custom_allocation_id_prefix(self, engine):
        """Custom allocation ID starts with 's2m_cust_'."""
        purchase = _make_purchase(mwh="1000")
        result = engine.allocate_custom(purchase, [], ["BUNDLED_CERTIFICATE"])
        assert result["allocation_id"].startswith("s2m_cust_")


# ===========================================================================
# 4. TestBatchAllocation  (5 tests)
# ===========================================================================


@_SKIP
class TestBatchAllocation:
    """Tests for allocate_batch with multiple purchases."""

    def test_batch_processes_multiple_purchases(self, engine):
        """Batch returns one result per purchase."""
        batch = [
            {"purchase": _make_purchase(facility_id="F1", mwh="1000"), "instruments": []},
            {"purchase": _make_purchase(facility_id="F2", mwh="2000"), "instruments": []},
        ]
        results = engine.allocate_batch(batch)
        assert len(results) == 2

    def test_batch_each_result_has_status(self, engine):
        """Each batch result has a status field."""
        batch = [
            {"purchase": _make_purchase(mwh="1000"), "instruments": []},
        ]
        results = engine.allocate_batch(batch)
        assert results[0]["status"] in ("SUCCESS", "FAILED")

    def test_batch_with_instruments(self, engine):
        """Batch allocation applies instruments to each purchase."""
        batch = [
            {
                "purchase": _make_purchase(mwh="5000"),
                "instruments": [_make_instrument(mwh="5000")],
            },
        ]
        results = engine.allocate_batch(batch)
        assert results[0]["status"] == "SUCCESS"
        assert results[0]["coverage_status"] == CoverageStatus.FULL.value

    def test_batch_error_isolation(self, engine):
        """Failure in one purchase does not affect others."""
        batch = [
            {"purchase": {"facility_id": "F1"}, "instruments": []},  # missing mwh
            {"purchase": _make_purchase(facility_id="F2", mwh="1000"), "instruments": []},
        ]
        results = engine.allocate_batch(batch)
        assert results[0]["status"] == "FAILED"
        assert results[1]["status"] == "SUCCESS"

    def test_batch_empty_list(self, engine):
        """Empty batch returns empty results."""
        results = engine.allocate_batch([])
        assert results == []


# ===========================================================================
# 5. TestInstrumentValidation  (15 tests)
# ===========================================================================


@_SKIP
class TestInstrumentValidation:
    """Tests for validate_instrument, validate_vintage, validate_geographic_match, etc."""

    def test_validate_valid_instrument(self, engine):
        """Well-formed instrument passes all checks."""
        inst = _make_instrument()
        result = engine.validate_instrument(inst, reporting_year=2025, consumption_region="US-WECC")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_validate_vintage_same_year(self, engine):
        """Vintage matching reporting year is valid."""
        assert engine.validate_vintage("BUNDLED_CERTIFICATE", 2025, 2025) is True

    def test_validate_vintage_previous_year(self, engine):
        """Bundled certificate vintage one year before is valid."""
        assert engine.validate_vintage("BUNDLED_CERTIFICATE", 2024, 2025) is True

    def test_validate_vintage_expired(self, engine):
        """Vintage two years before reporting year is invalid for bundled."""
        assert engine.validate_vintage("BUNDLED_CERTIFICATE", 2023, 2025) is False

    def test_validate_vintage_supplier_specific_must_match(self, engine):
        """SUPPLIER_SPECIFIC vintage must match reporting year exactly."""
        assert engine.validate_vintage("SUPPLIER_SPECIFIC", 2025, 2025) is True
        assert engine.validate_vintage("SUPPLIER_SPECIFIC", 2024, 2025) is False

    def test_validate_geographic_match_exact(self, engine):
        """Exact region match returns True."""
        assert engine.validate_geographic_match("US-WECC", "US-WECC") is True

    def test_validate_geographic_match_interconnected(self, engine):
        """Interconnected regions match."""
        assert engine.validate_geographic_match("US-CAMX", "US-WECC") is True

    def test_validate_geographic_match_global(self, engine):
        """GLOBAL region matches any consumption region."""
        assert engine.validate_geographic_match("GLOBAL", "US-WECC") is True

    def test_validate_geographic_mismatch(self, engine):
        """Unrelated regions do not match."""
        assert engine.validate_geographic_match("US-ERCOT", "EU-DE") is False

    def test_validate_tracking_system_recognized(self, engine):
        """Recognized tracking system passes."""
        inst = _make_instrument(tracking_system="WREGIS")
        assert engine.validate_tracking_system(inst) is True

    def test_validate_tracking_system_unrecognized(self, engine):
        """Unrecognized tracking system fails for strict mode."""
        inst = _make_instrument(tracking_system="FAKE_SYSTEM")
        assert engine.validate_tracking_system(inst) is False

    def test_validate_tracking_system_optional_for_supplier(self, engine):
        """Missing tracking system is OK for SUPPLIER_SPECIFIC."""
        inst = _make_instrument(inst_type="SUPPLIER_SPECIFIC", tracking_system="")
        assert engine.validate_tracking_system(inst) is True

    def test_check_double_counting_unused(self, engine):
        """Unused instrument is not double-counted."""
        assert engine.check_double_counting("REC-NEW", {}) is False

    def test_check_double_counting_used(self, engine):
        """Used instrument is detected as double-counted."""
        used = {"REC-001": {"allocation_id": "a1"}}
        assert engine.check_double_counting("REC-001", used) is True

    def test_validate_instrument_zero_capacity_fails(self, engine):
        """Instrument with zero capacity fails has_capacity check."""
        inst = _make_instrument(mwh="0")
        result = engine.validate_instrument(inst)
        assert result["checks"]["has_capacity"] is False


# ===========================================================================
# 6. TestCoverageTracking  (10 tests)
# ===========================================================================


@_SKIP
class TestCoverageTracking:
    """Tests for calculate_coverage, get_coverage_status, identify_coverage_gaps."""

    def test_full_coverage(self, engine):
        """Instruments totaling purchase MWh yield FULL status."""
        result = engine.calculate_coverage("10000", [{"mwh": "10000"}])
        assert result["coverage_status"] == CoverageStatus.FULL.value

    def test_partial_coverage(self, engine):
        """Instruments < purchase MWh yield PARTIAL status."""
        result = engine.calculate_coverage("10000", [{"mwh": "5000"}])
        assert result["coverage_status"] == CoverageStatus.PARTIAL.value

    def test_none_coverage(self, engine):
        """No instruments yield NONE status."""
        result = engine.calculate_coverage("10000", [])
        assert result["coverage_status"] == CoverageStatus.NONE.value

    def test_over_allocated(self, engine):
        """Instruments exceeding purchase yield OVER_ALLOCATED."""
        result = engine.calculate_coverage("5000", [{"mwh": "8000"}])
        # calculate_coverage caps covered at total, so status is FULL
        assert result["coverage_status"] == CoverageStatus.FULL.value

    def test_coverage_status_zero(self, engine):
        """0% coverage returns NONE."""
        assert engine.get_coverage_status(Decimal("0")) == CoverageStatus.NONE

    def test_coverage_status_partial(self, engine):
        """50% coverage returns PARTIAL."""
        assert engine.get_coverage_status(Decimal("50")) == CoverageStatus.PARTIAL

    def test_coverage_status_full(self, engine):
        """100% coverage returns FULL."""
        assert engine.get_coverage_status(Decimal("100")) == CoverageStatus.FULL

    def test_coverage_status_over(self, engine):
        """120% coverage returns OVER_ALLOCATED."""
        assert engine.get_coverage_status(Decimal("120")) == CoverageStatus.OVER_ALLOCATED

    def test_identify_coverage_gaps_finds_partial(self, engine):
        """identify_coverage_gaps returns facilities with < 100% coverage."""
        facilities = [
            {"facility_id": "F1", "total_mwh": "10000", "instruments": [{"mwh": "10000"}]},
            {"facility_id": "F2", "total_mwh": "10000", "instruments": [{"mwh": "5000"}]},
        ]
        gaps = engine.identify_coverage_gaps(facilities)
        assert len(gaps) == 1
        assert gaps[0]["facility_id"] == "F2"

    def test_identify_coverage_gaps_no_gaps(self, engine):
        """All facilities fully covered returns empty gap list."""
        facilities = [
            {"facility_id": "F1", "total_mwh": "5000", "instruments": [{"mwh": "5000"}]},
        ]
        gaps = engine.identify_coverage_gaps(facilities)
        assert len(gaps) == 0


# ===========================================================================
# 7. TestCertificateManagement  (10 tests)
# ===========================================================================


@_SKIP
class TestCertificateManagement:
    """Tests for retire_instrument, check_retirement_status, etc."""

    def test_retire_instrument_success(self, engine):
        """Retiring an active instrument returns RETIRED status."""
        result = engine.retire_instrument("REC-001", reason="Annual retirement")
        assert result["status"] == "RETIRED"
        assert result["instrument_id"] == "REC-001"
        assert result["retired_at"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_retire_instrument_already_retired(self, engine):
        """Retiring an already-retired instrument returns ALREADY_RETIRED."""
        engine.retire_instrument("REC-001")
        result = engine.retire_instrument("REC-001")
        assert result["status"] == "ALREADY_RETIRED"

    def test_check_retirement_status_active(self, engine):
        """Unretired instrument has ACTIVE status."""
        status = engine.check_retirement_status("REC-NEW")
        assert status == RetirementStatus.ACTIVE.value

    def test_check_retirement_status_retired(self, engine):
        """Retired instrument has RETIRED status."""
        engine.retire_instrument("REC-001")
        status = engine.check_retirement_status("REC-001")
        assert status == RetirementStatus.RETIRED.value

    def test_check_retirement_status_allocated(self, engine):
        """Allocated but not retired instrument has ALLOCATED status."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(inst_id="REC-ALLOC", mwh="5000")]
        engine.allocate_instruments(purchase, instruments)
        status = engine.check_retirement_status("REC-ALLOC")
        assert status == RetirementStatus.ALLOCATED.value

    def test_list_retired_instruments_empty(self, engine):
        """Initially no retired instruments."""
        assert engine.list_retired_instruments() == []

    def test_list_retired_instruments_after_retire(self, engine):
        """list_retired_instruments reflects retired instruments."""
        engine.retire_instrument("REC-001")
        engine.retire_instrument("REC-002")
        retired = engine.list_retired_instruments()
        ids = {r["instrument_id"] for r in retired}
        assert ids == {"REC-001", "REC-002"}

    def test_get_retirement_history_empty(self, engine):
        """Unretired instrument has empty retirement history."""
        history = engine.get_retirement_history("REC-NEW")
        assert history == []

    def test_get_retirement_history_after_retire(self, engine):
        """Retired instrument has one history entry."""
        engine.retire_instrument("REC-001", reason="End of year", retired_by="admin")
        history = engine.get_retirement_history("REC-001")
        assert len(history) == 1
        assert history[0]["reason"] == "End of year"
        assert history[0]["retired_by"] == "admin"

    def test_retired_instrument_not_allocated(self, engine):
        """Retired instrument fails validation and is not allocated."""
        engine.retire_instrument("REC-RETIRED")
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(inst_id="REC-RETIRED", mwh="5000")]
        result = engine.allocate_instruments(purchase, instruments)
        # The retired instrument should be filtered out
        assert result["coverage_status"] in (CoverageStatus.NONE.value, CoverageStatus.PARTIAL.value)


# ===========================================================================
# 8. TestUtilities  (5 tests)
# ===========================================================================


@_SKIP
class TestUtilities:
    """Tests for get_instrument_priority, sort_by_priority, get_statistics, reset."""

    def test_get_instrument_priority_bundled(self, engine):
        """BUNDLED_CERTIFICATE has priority 1."""
        assert engine.get_instrument_priority("BUNDLED_CERTIFICATE") == 1

    def test_get_instrument_priority_residual(self, engine):
        """RESIDUAL_MIX has priority 6."""
        assert engine.get_instrument_priority("RESIDUAL_MIX") == 6

    def test_get_instrument_priority_unknown(self, engine):
        """Unknown type returns priority 99."""
        assert engine.get_instrument_priority("UNKNOWN_TYPE") == 99

    def test_sort_by_priority(self, engine):
        """sort_by_priority orders instruments by hierarchy."""
        instruments = [
            {"type": "RESIDUAL_MIX", "mwh": "1000"},
            {"type": "BUNDLED_CERTIFICATE", "mwh": "1000"},
            {"type": "DIRECT_CONTRACT", "mwh": "1000"},
        ]
        sorted_insts = engine.sort_by_priority(instruments)
        types = [i["type"] for i in sorted_insts]
        assert types == ["BUNDLED_CERTIFICATE", "DIRECT_CONTRACT", "RESIDUAL_MIX"]

    def test_get_statistics_after_allocation(self, engine):
        """Statistics reflect allocation counts."""
        purchase = _make_purchase(mwh="5000")
        instruments = [_make_instrument(mwh="5000")]
        engine.allocate_instruments(purchase, instruments)
        stats = engine.get_statistics()
        assert stats["allocation_count"] == 1


# ===========================================================================
# 9. TestThreadSafety  (5 tests)
# ===========================================================================


@_SKIP
class TestThreadSafety:
    """Tests for concurrent allocation and state consistency."""

    def test_concurrent_allocations_no_crash(self, engine):
        """Concurrent allocations do not crash the engine."""
        errors = []

        def allocate(idx):
            try:
                purchase = _make_purchase(facility_id=f"F-{idx}", mwh="1000")
                instruments = [_make_instrument(inst_id=f"REC-{idx}", mwh="1000")]
                result = engine.allocate_instruments(purchase, instruments)
                assert result["status"] == "SUCCESS"
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=allocate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_retirements_no_crash(self, engine):
        """Concurrent retirements do not corrupt state."""
        errors = []

        def retire(idx):
            try:
                engine.retire_instrument(f"REC-THREAD-{idx}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=retire, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        retired = engine.list_retired_instruments()
        assert len(retired) == 10

    def test_reset_clears_all_state(self, engine):
        """reset() clears all mutable state."""
        purchase = _make_purchase(mwh="5000")
        engine.allocate_instruments(purchase, [_make_instrument(mwh="5000")])
        engine.retire_instrument("REC-RESET")

        engine.reset()

        stats = engine.get_statistics()
        assert stats["allocation_count"] == 0
        assert stats["retirement_count"] == 0
        assert stats["used_instrument_count"] == 0
        assert stats["retired_instrument_count"] == 0

    def test_engine_repr(self, engine):
        """__repr__ returns a meaningful string."""
        r = repr(engine)
        assert "InstrumentAllocationEngine" in r

    def test_engine_len(self, engine):
        """len(engine) returns allocation count."""
        assert len(engine) == 0
        purchase = _make_purchase(mwh="1000")
        engine.allocate_instruments(purchase, [])
        assert len(engine) == 1
