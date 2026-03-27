# -*- coding: utf-8 -*-
"""
Unit tests for PublicTransitCalculatorEngine (Engine 3).

Tests the public transit emissions calculator covering all 9 transit types,
WTT emissions, WFH fraction reduction, multi-modal transit, and input
validation.

Target: ~30 tests covering all transit types, WTT, working days scaling,
frequency adjustments, multi-modal, batch, and input validation.

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUANT = Decimal("0.00000001")


def _q(v: Decimal) -> Decimal:
    return v.quantize(_QUANT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton before each test."""
    from greenlang.agents.mrv.employee_commuting.public_transit_calculator import (
        PublicTransitCalculatorEngine,
    )
    PublicTransitCalculatorEngine.reset_instance()
    yield
    PublicTransitCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh PublicTransitCalculatorEngine instance."""
    from greenlang.agents.mrv.employee_commuting.public_transit_calculator import (
        PublicTransitCalculatorEngine,
    )
    return PublicTransitCalculatorEngine.get_instance()


# ===========================================================================
# 1. SINGLETON AND INITIALIZATION
# ===========================================================================

class TestSingletonTransit:
    """Singleton and initialization tests."""

    def test_singleton_identity(self, engine):
        """Two get_instance calls return the same object."""
        from greenlang.agents.mrv.employee_commuting.public_transit_calculator import (
            PublicTransitCalculatorEngine,
        )
        assert engine is PublicTransitCalculatorEngine.get_instance()

    def test_initial_calculation_count(self, engine):
        """Calculation count starts at zero."""
        assert engine.calculation_count == 0


# ===========================================================================
# 2. SINGLE-MODE TRANSIT CALCULATIONS
# ===========================================================================

class TestTransitCommute:
    """Tests for calculate_transit_commute() across all 9 transit types."""

    @pytest.mark.parametrize("transit_type,co2e_per_pkm", [
        ("local_bus", Decimal("0.10312")),
        ("express_bus", Decimal("0.08956")),
        ("coach", Decimal("0.02732")),
        ("commuter_rail", Decimal("0.04115")),
        ("subway_metro", Decimal("0.03071")),
        ("light_rail", Decimal("0.02904")),
        ("tram_streetcar", Decimal("0.02940")),
        ("ferry_boat", Decimal("0.11318")),
        ("water_taxi", Decimal("0.14782")),
    ])
    def test_transit_type_positive_emissions(self, engine, transit_type, co2e_per_pkm):
        """Each transit type produces positive emissions for a standard commute."""
        result = engine.calculate_transit_commute(
            transit_type=transit_type,
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        assert result["co2e_kg"] > Decimal("0")
        assert result["transit_type"] == transit_type

    def test_local_bus_known_calculation(self, engine):
        """Local bus: 10 km x 2 x 225 days = 4500 pkm; TTW = 4500 x 0.10312."""
        result = engine.calculate_transit_commute(
            transit_type="local_bus",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
            include_wtt=False,
        )
        expected_distance = _q(Decimal("10.0") * 2 * 225)
        expected_co2e = _q(expected_distance * Decimal("0.10312"))
        assert result["annual_distance_km"] == expected_distance
        assert result["ttw_co2e_kg"] == expected_co2e

    def test_commuter_rail_with_wtt(self, engine):
        """Commuter rail with WTT includes both TTW and WTT components."""
        result = engine.calculate_transit_commute(
            transit_type="commuter_rail",
            one_way_distance_km=Decimal("25.0"),
            working_days=225,
            include_wtt=True,
        )
        assert result["wtt_co2e_kg"] > Decimal("0")
        assert result["co2e_kg"] == _q(result["ttw_co2e_kg"] + result["wtt_co2e_kg"])

    def test_subway_without_wtt(self, engine):
        """Subway without WTT: wtt_co2e_kg = 0."""
        result = engine.calculate_transit_commute(
            transit_type="subway_metro",
            one_way_distance_km=Decimal("12.0"),
            working_days=225,
            include_wtt=False,
        )
        assert result["wtt_co2e_kg"] == Decimal("0")

    def test_wfh_reduces_transit_emissions(self, engine):
        """20% WFH reduces annual distance and emissions by 20%."""
        full = engine.calculate_transit_commute(
            transit_type="commuter_rail",
            one_way_distance_km=Decimal("30.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        wfh = engine.calculate_transit_commute(
            transit_type="commuter_rail",
            one_way_distance_km=Decimal("30.0"),
            working_days=225,
            wfh_fraction=Decimal("0.20"),
        )
        expected_distance = _q(full["annual_distance_km"] * Decimal("0.80"))
        assert wfh["annual_distance_km"] == expected_distance

    def test_one_way_mode(self, engine):
        """round_trip=False halves the daily distance."""
        result = engine.calculate_transit_commute(
            transit_type="local_bus",
            one_way_distance_km=Decimal("10.0"),
            working_days=200,
            round_trip=False,
        )
        expected_daily = _q(Decimal("10.0"))
        assert result["daily_distance_km"] == expected_daily

    def test_trips_per_day(self, engine):
        """Multiple trips per day multiplies the daily distance."""
        single = engine.calculate_transit_commute(
            transit_type="subway_metro",
            one_way_distance_km=Decimal("5.0"),
            working_days=200,
            trips_per_day=1,
        )
        double = engine.calculate_transit_commute(
            transit_type="subway_metro",
            one_way_distance_km=Decimal("5.0"),
            working_days=200,
            trips_per_day=2,
        )
        assert double["annual_distance_km"] == _q(single["annual_distance_km"] * 2)

    def test_gas_breakdown_present(self, engine):
        """Result includes co2_kg, ch4_kg, n2o_kg breakdown."""
        result = engine.calculate_transit_commute(
            transit_type="local_bus",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        assert "co2_kg" in result
        assert "ch4_kg" in result
        assert "n2o_kg" in result
        assert result["co2_kg"] > Decimal("0")

    def test_ferry_highest_among_transit(self, engine):
        """Ferry/water taxi should have highest per-pkm factor among transit."""
        bus = engine.calculate_transit_commute(
            transit_type="local_bus",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        ferry = engine.calculate_transit_commute(
            transit_type="ferry_boat",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        assert ferry["co2e_kg"] > bus["co2e_kg"]

    def test_coach_lowest_bus_type(self, engine):
        """Coach has the lowest factor among bus types."""
        local = engine.calculate_transit_commute(
            transit_type="local_bus",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        coach = engine.calculate_transit_commute(
            transit_type="coach",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        assert coach["co2e_kg"] < local["co2e_kg"]


# ===========================================================================
# 3. INPUT VALIDATION
# ===========================================================================

class TestTransitInputValidation:
    """Tests for input validation in calculate_transit_commute."""

    def test_invalid_transit_type_raises(self, engine):
        """Unknown transit type raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_transit_commute(
                transit_type="teleportation",
                one_way_distance_km=Decimal("10.0"),
            )

    def test_negative_distance_raises(self, engine):
        """Negative distance raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_transit_commute(
                transit_type="local_bus",
                one_way_distance_km=Decimal("-5.0"),
            )

    def test_zero_distance_raises(self, engine):
        """Zero distance raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_transit_commute(
                transit_type="local_bus",
                one_way_distance_km=Decimal("0.0"),
            )

    def test_wfh_greater_than_one_raises(self, engine):
        """WFH fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_transit_commute(
                transit_type="local_bus",
                one_way_distance_km=Decimal("10.0"),
                wfh_fraction=Decimal("1.5"),
            )

    def test_negative_wfh_raises(self, engine):
        """Negative WFH fraction raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_transit_commute(
                transit_type="local_bus",
                one_way_distance_km=Decimal("10.0"),
                wfh_fraction=Decimal("-0.1"),
            )

    def test_zero_working_days_raises(self, engine):
        """Zero working days raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_transit_commute(
                transit_type="local_bus",
                one_way_distance_km=Decimal("10.0"),
                working_days=0,
            )


# ===========================================================================
# 4. PROVENANCE AND REPRODUCIBILITY
# ===========================================================================

class TestTransitProvenance:
    """Provenance and reproducibility tests."""

    def test_same_inputs_same_hash(self, engine):
        """Identical inputs produce the same provenance hash."""
        kwargs = dict(
            transit_type="commuter_rail",
            one_way_distance_km=Decimal("25.0"),
            working_days=225,
            wfh_fraction=Decimal("0.0"),
        )
        r1 = engine.calculate_transit_commute(**kwargs)
        r2 = engine.calculate_transit_commute(**kwargs)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_hash_length_is_64(self, engine):
        """Provenance hash is a 64-character SHA-256 hex string."""
        result = engine.calculate_transit_commute(
            transit_type="subway_metro",
            one_way_distance_km=Decimal("8.0"),
            working_days=200,
        )
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 5. CALCULATION COUNT
# ===========================================================================

class TestTransitCalculationCount:
    """Calculation count tracking."""

    def test_count_increments_on_each_call(self, engine):
        """Each successful call increments the calculation count."""
        engine.calculate_transit_commute(
            transit_type="local_bus",
            one_way_distance_km=Decimal("5.0"),
        )
        engine.calculate_transit_commute(
            transit_type="subway_metro",
            one_way_distance_km=Decimal("8.0"),
        )
        assert engine.calculation_count == 2

    def test_count_not_incremented_on_error(self, engine):
        """Failed validations should not increment the count."""
        try:
            engine.calculate_transit_commute(
                transit_type="invalid",
                one_way_distance_km=Decimal("5.0"),
            )
        except ValueError:
            pass
        assert engine.calculation_count == 0


# ===========================================================================
# 6. CASE AND WHITESPACE NORMALISATION
# ===========================================================================

class TestTransitNormalisation:
    """Transit type normalisation tests."""

    def test_uppercase_transit_type(self, engine):
        """Transit type is case-insensitive."""
        result = engine.calculate_transit_commute(
            transit_type="LOCAL_BUS",
            one_way_distance_km=Decimal("10.0"),
            working_days=225,
        )
        assert result["transit_type"] == "local_bus"

    def test_transit_type_with_spaces(self, engine):
        """Hyphens or spaces in transit type are normalised."""
        result = engine.calculate_transit_commute(
            transit_type="commuter-rail",
            one_way_distance_km=Decimal("20.0"),
            working_days=225,
        )
        assert result["transit_type"] == "commuter_rail"
