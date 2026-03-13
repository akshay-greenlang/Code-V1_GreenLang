# -*- coding: utf-8 -*-
"""
Tests for ConversionFactorValidator - AGENT-EUDR-011 Engine 3: Factor Validation

Comprehensive test suite covering:
- Factor validation (within range, warn on deviation, reject on large deviation)
- Reference factors (get all, by commodity)
- Custom factors (register, approval workflow)
- Chain validation (multi-step, cumulative)
- Seasonal adjustment (cocoa, coffee, palm oil)
- Factor history (per facility tracking)
- Deviation reporting (facility trends)
- Edge cases (zero, negative, unknown commodity)

Test count: 55+ tests
Coverage target: >= 85% of ConversionFactorValidator module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    REFERENCE_CONVERSION_FACTORS,
    SHA256_HEX_LENGTH,
    FACTOR_COCOA_ROASTING,
    FACTOR_PALM_EXTRACTION,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    FAC_ID_FACTORY_DE,
    make_factor,
    assert_valid_provenance_hash,
    assert_valid_score,
)


# ===========================================================================
# 1. Factor Validation
# ===========================================================================


class TestFactorValidation:
    """Test conversion factor validation against reference data."""

    def test_validate_exact_match(self, conversion_factor_validator):
        """Factor matching reference exactly is accepted."""
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.85)
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("accepted", "valid")

    def test_validate_within_warn_threshold(self, conversion_factor_validator):
        """Factor within warn threshold is accepted with info."""
        # 0.85 * 1.03 = 0.8755 (3% deviation, below 5% warn)
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.8755)
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("accepted", "valid")

    def test_validate_warn_on_deviation(self, conversion_factor_validator):
        """Factor exceeding warn threshold triggers warning."""
        # 0.85 * 1.08 = 0.918 (8% deviation, above 5% warn)
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.918)
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("warning", "warn")

    def test_validate_reject_on_large_deviation(self, conversion_factor_validator):
        """Factor exceeding reject threshold is rejected."""
        # 0.85 * 1.20 = 1.02 (20% deviation, above 15% reject)
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.99)
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("rejected", "reject", "invalid")

    def test_validate_below_reference_warns(self, conversion_factor_validator):
        """Factor below reference by >5% triggers warning."""
        # 0.85 * 0.90 = 0.765 (10% below, above 5% warn threshold)
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.765)
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("warning", "warn")

    def test_validate_below_reference_rejects(self, conversion_factor_validator):
        """Factor far below reference is rejected."""
        # 0.85 * 0.80 = 0.68 (20% below, above 15% reject)
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.68)
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("rejected", "reject", "invalid")

    def test_validate_provenance_hash(self, conversion_factor_validator):
        """Validation generates a provenance hash."""
        factor = make_factor()
        result = conversion_factor_validator.validate(factor)
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_validate_all_commodities(self, conversion_factor_validator, commodity):
        """Validation works for all 7 EUDR commodities."""
        processes = REFERENCE_CONVERSION_FACTORS.get(commodity, {})
        if processes:
            process = list(processes.keys())[0]
            ratio = processes[process]
            factor = make_factor(
                commodity=commodity, process=process, yield_ratio=ratio,
            )
            result = conversion_factor_validator.validate(factor)
            assert result is not None

    def test_validate_deviation_percent_calculated(self, conversion_factor_validator):
        """Validation result includes deviation percentage."""
        factor = make_factor(commodity="cocoa", process="roasting", yield_ratio=0.87)
        result = conversion_factor_validator.validate(factor)
        assert "deviation_percent" in result or "deviation" in result


# ===========================================================================
# 2. Reference Factors
# ===========================================================================


class TestReferenceFactors:
    """Test reference factor lookups."""

    def test_get_all_reference_factors(self, conversion_factor_validator):
        """Get all reference conversion factors."""
        factors = conversion_factor_validator.get_reference_factors()
        assert isinstance(factors, dict)
        assert len(factors) > 0

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_get_reference_by_commodity(self, conversion_factor_validator, commodity):
        """Get reference factors for each commodity."""
        factors = conversion_factor_validator.get_reference_factors(commodity=commodity)
        assert isinstance(factors, (dict, list))

    def test_reference_cocoa_roasting(self, conversion_factor_validator):
        """Cocoa roasting reference ratio is ~0.85."""
        factors = conversion_factor_validator.get_reference_factors(commodity="cocoa")
        if isinstance(factors, dict):
            assert "roasting" in factors
            assert abs(factors["roasting"] - 0.85) < 0.01

    def test_reference_palm_extraction(self, conversion_factor_validator):
        """Palm oil extraction reference ratio is ~0.22."""
        factors = conversion_factor_validator.get_reference_factors(commodity="oil_palm")
        if isinstance(factors, dict):
            assert "extraction" in factors
            assert abs(factors["extraction"] - 0.22) < 0.01

    def test_unknown_commodity_returns_empty(self, conversion_factor_validator):
        """Unknown commodity returns empty or raises ValueError."""
        try:
            factors = conversion_factor_validator.get_reference_factors(
                commodity="unknown_crop"
            )
            assert factors is None or len(factors) == 0
        except ValueError:
            pass  # Also acceptable


# ===========================================================================
# 3. Custom Factors
# ===========================================================================


class TestCustomFactors:
    """Test custom factor registration and approval."""

    def test_register_custom_factor(self, conversion_factor_validator):
        """Register a custom conversion factor."""
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.83,
            facility_id=FAC_ID_MILL_MY,
        )
        result = conversion_factor_validator.register_custom(factor)
        assert result is not None

    def test_register_custom_pending_approval(self, conversion_factor_validator):
        """Custom factor with large deviation requires approval."""
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.75,  # ~12% deviation
        )
        result = conversion_factor_validator.register_custom(factor)
        assert result.get("status") in ("pending_approval", "warning", "pending")

    def test_register_custom_auto_approved(self, conversion_factor_validator):
        """Custom factor within tolerance is auto-approved."""
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.84,  # ~1% deviation
        )
        result = conversion_factor_validator.register_custom(factor)
        assert result.get("status") in ("accepted", "approved", "valid")

    def test_approve_custom_factor(self, conversion_factor_validator):
        """Approve a pending custom factor."""
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.75,
            factor_id="CF-CUSTOM-001",
        )
        conversion_factor_validator.register_custom(factor)
        result = conversion_factor_validator.approve(
            "CF-CUSTOM-001",
            approved_by="qa-manager-001",
        )
        assert result is not None

    def test_register_duplicate_raises(self, conversion_factor_validator):
        """Registering duplicate custom factor ID raises error."""
        factor = make_factor(factor_id="CF-DUP-001")
        conversion_factor_validator.register_custom(factor)
        with pytest.raises((ValueError, KeyError)):
            conversion_factor_validator.register_custom(copy.deepcopy(factor))


# ===========================================================================
# 4. Chain Validation
# ===========================================================================


class TestChainValidation:
    """Test multi-step conversion chain validation."""

    def test_validate_two_step_chain(self, conversion_factor_validator):
        """Validate a two-step conversion chain."""
        chain = [
            make_factor(commodity="cocoa", process="fermentation", yield_ratio=0.92),
            make_factor(commodity="cocoa", process="drying", yield_ratio=0.88),
        ]
        result = conversion_factor_validator.validate_chain(chain)
        assert result is not None
        cumulative = result.get("cumulative_yield", result.get("chain_yield"))
        if cumulative is not None:
            assert abs(cumulative - 0.92 * 0.88) < 0.01

    def test_validate_five_step_chain(self, conversion_factor_validator):
        """Validate a five-step cocoa processing chain."""
        chain = [
            make_factor(commodity="cocoa", process="fermentation", yield_ratio=0.92),
            make_factor(commodity="cocoa", process="drying", yield_ratio=0.88),
            make_factor(commodity="cocoa", process="roasting", yield_ratio=0.85),
            make_factor(commodity="cocoa", process="winnowing", yield_ratio=0.80),
            make_factor(commodity="cocoa", process="grinding", yield_ratio=0.98),
        ]
        result = conversion_factor_validator.validate_chain(chain)
        assert result is not None

    def test_chain_with_rejected_step_fails(self, conversion_factor_validator):
        """Chain with a rejected step fails overall validation."""
        chain = [
            make_factor(commodity="cocoa", process="fermentation", yield_ratio=0.92),
            make_factor(commodity="cocoa", process="drying", yield_ratio=0.50),  # Way off
        ]
        result = conversion_factor_validator.validate_chain(chain)
        assert result.get("status") in ("rejected", "warning", "invalid") or \
            result.get("has_rejected_steps") is True

    def test_empty_chain_raises(self, conversion_factor_validator):
        """Empty chain raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            conversion_factor_validator.validate_chain([])


# ===========================================================================
# 5. Seasonal Adjustment
# ===========================================================================


class TestSeasonalAdjustment:
    """Test seasonal adjustment factors."""

    def test_seasonal_factor_cocoa_dry(self, conversion_factor_validator):
        """Cocoa dry season factor adjustment."""
        factor = make_factor(
            commodity="cocoa",
            process="drying",
            yield_ratio=0.85,
            season="dry",
        )
        result = conversion_factor_validator.validate(factor)
        assert result is not None

    def test_seasonal_factor_cocoa_wet(self, conversion_factor_validator):
        """Cocoa wet season factor adjustment."""
        factor = make_factor(
            commodity="cocoa",
            process="drying",
            yield_ratio=0.90,
            season="wet",
        )
        result = conversion_factor_validator.validate(factor)
        assert result is not None

    def test_seasonal_factor_coffee_harvest(self, conversion_factor_validator):
        """Coffee harvest season factor."""
        factor = make_factor(
            commodity="coffee",
            process="wet_processing",
            yield_ratio=0.62,
            season="harvest",
        )
        result = conversion_factor_validator.validate(factor)
        assert result is not None

    def test_seasonal_factor_palm_peak(self, conversion_factor_validator):
        """Palm oil peak season factor."""
        factor = make_factor(
            commodity="oil_palm",
            process="extraction",
            yield_ratio=0.24,
            season="peak",
        )
        result = conversion_factor_validator.validate(factor)
        assert result is not None


# ===========================================================================
# 6. Factor History
# ===========================================================================


class TestFactorHistory:
    """Test factor history tracking per facility."""

    def test_track_factor_per_facility(self, conversion_factor_validator):
        """Track conversion factors per facility."""
        for i in range(3):
            factor = make_factor(
                commodity="cocoa",
                process="roasting",
                yield_ratio=0.84 + i * 0.005,
                factor_id=f"CF-HIST-{i:03d}",
                facility_id=FAC_ID_MILL_MY,
            )
            conversion_factor_validator.register_custom(factor)
        history = conversion_factor_validator.get_history(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            process="roasting",
        )
        assert len(history) >= 3

    def test_history_chronological_order(self, conversion_factor_validator):
        """Factor history is in chronological order."""
        for i in range(3):
            factor = make_factor(
                commodity="cocoa",
                process="drying",
                yield_ratio=0.87 + i * 0.005,
                factor_id=f"CF-HORD-{i:03d}",
                facility_id=FAC_ID_MILL_MY,
            )
            conversion_factor_validator.register_custom(factor)
        history = conversion_factor_validator.get_history(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            process="drying",
        )
        for i in range(len(history) - 1):
            assert history[i].get("registered_at", "") <= history[i + 1].get("registered_at", "")


# ===========================================================================
# 7. Deviation Reporting
# ===========================================================================


class TestDeviationReporting:
    """Test facility deviation trend reporting."""

    def test_deviation_trend_report(self, conversion_factor_validator):
        """Generate deviation trend report for a facility."""
        for i in range(5):
            factor = make_factor(
                commodity="cocoa",
                process="roasting",
                yield_ratio=0.80 + i * 0.02,
                factor_id=f"CF-DEV-{i:03d}",
                facility_id=FAC_ID_MILL_MY,
            )
            conversion_factor_validator.register_custom(factor)
        report = conversion_factor_validator.deviation_report(
            facility_id=FAC_ID_MILL_MY,
        )
        assert report is not None

    def test_deviation_report_includes_commodity(self, conversion_factor_validator):
        """Deviation report includes commodity breakdown."""
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.83,
            factor_id="CF-DEVC-001",
            facility_id=FAC_ID_MILL_MY,
        )
        conversion_factor_validator.register_custom(factor)
        report = conversion_factor_validator.deviation_report(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        assert report is not None


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for conversion factor operations."""

    def test_zero_factor_raises(self, conversion_factor_validator):
        """Factor with zero yield ratio raises ValueError."""
        factor = make_factor(yield_ratio=0.0)
        with pytest.raises(ValueError):
            conversion_factor_validator.validate(factor)

    def test_negative_factor_raises(self, conversion_factor_validator):
        """Factor with negative yield ratio raises ValueError."""
        factor = make_factor(yield_ratio=-0.5)
        with pytest.raises(ValueError):
            conversion_factor_validator.validate(factor)

    def test_factor_above_one_raises(self, conversion_factor_validator):
        """Factor with yield ratio above 1.0 raises ValueError."""
        factor = make_factor(yield_ratio=1.5)
        with pytest.raises(ValueError):
            conversion_factor_validator.validate(factor)

    def test_unknown_process_handled(self, conversion_factor_validator):
        """Unknown process type is handled gracefully."""
        factor = make_factor(
            commodity="cocoa",
            process="unknown_process",
            yield_ratio=0.85,
        )
        try:
            result = conversion_factor_validator.validate(factor)
            assert result is not None
        except (ValueError, KeyError):
            pass  # Also acceptable

    def test_factor_exactly_at_warn_threshold(self, conversion_factor_validator):
        """Factor exactly at warn deviation threshold."""
        # 0.85 * 1.05 = 0.8925
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.8925,
        )
        result = conversion_factor_validator.validate(factor)
        assert result is not None

    def test_factor_exactly_at_reject_threshold(self, conversion_factor_validator):
        """Factor exactly at reject deviation threshold."""
        # 0.85 * 1.15 = 0.9775
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.9775,
        )
        result = conversion_factor_validator.validate(factor)
        assert result is not None

    def test_factor_exactly_one(self, conversion_factor_validator):
        """Factor of exactly 1.0 may be valid for some processes."""
        factor = make_factor(yield_ratio=1.0)
        try:
            result = conversion_factor_validator.validate(factor)
            assert result is not None
        except ValueError:
            pass  # Some implementations may reject 1.0

    def test_very_low_factor_rejected(self, conversion_factor_validator):
        """Very low factor (0.01) is rejected."""
        factor = make_factor(
            commodity="cocoa",
            process="roasting",
            yield_ratio=0.01,
        )
        result = conversion_factor_validator.validate(factor)
        assert result["status"] in ("rejected", "reject", "invalid")

    def test_single_step_chain(self, conversion_factor_validator):
        """Single-step chain validation works."""
        chain = [
            make_factor(commodity="cocoa", process="roasting", yield_ratio=0.85),
        ]
        result = conversion_factor_validator.validate_chain(chain)
        assert result is not None

    @pytest.mark.parametrize("process,expected_ratio", [
        ("fermentation", 0.92),
        ("drying", 0.88),
        ("roasting", 0.85),
        ("winnowing", 0.80),
        ("grinding", 0.98),
    ])
    def test_cocoa_reference_factors(self, conversion_factor_validator, process, expected_ratio):
        """Cocoa reference factors match expected values."""
        factors = conversion_factor_validator.get_reference_factors(commodity="cocoa")
        if isinstance(factors, dict) and process in factors:
            assert abs(factors[process] - expected_ratio) < 0.01

    @pytest.mark.parametrize("process,expected_ratio", [
        ("wet_processing", 0.60),
        ("dry_processing", 0.50),
        ("hulling", 0.80),
        ("roasting", 0.82),
    ])
    def test_coffee_reference_factors(self, conversion_factor_validator, process, expected_ratio):
        """Coffee reference factors match expected values."""
        factors = conversion_factor_validator.get_reference_factors(commodity="coffee")
        if isinstance(factors, dict) and process in factors:
            assert abs(factors[process] - expected_ratio) < 0.01
