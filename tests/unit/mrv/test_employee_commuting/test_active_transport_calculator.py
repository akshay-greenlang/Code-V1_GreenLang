# -*- coding: utf-8 -*-
"""
Unit tests for ActiveTransportCalculatorEngine (Engine 5).

Tests active transport (cycling, walking) and micro-mobility (e-bike, e-scooter)
emissions calculations. Active modes have zero operational emissions with
optional lifecycle; electric modes have grid-based operational emissions plus
lifecycle.

Target: ~20 tests covering cycling, walking, e-bike, e-scooter, lifecycle,
mode share tracking, grid factor resolution, and input validation.

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
_ZERO = Decimal("0")


def _q(v: Decimal) -> Decimal:
    return v.quantize(_QUANT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton before each test."""
    from greenlang.agents.mrv.employee_commuting.active_transport_calculator import (
        ActiveTransportCalculatorEngine,
    )
    ActiveTransportCalculatorEngine.reset_instance()
    yield
    ActiveTransportCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh ActiveTransportCalculatorEngine instance."""
    from greenlang.agents.mrv.employee_commuting.active_transport_calculator import (
        ActiveTransportCalculatorEngine,
    )
    return ActiveTransportCalculatorEngine.get_instance()


# ===========================================================================
# 1. SINGLETON AND PROPERTIES
# ===========================================================================

class TestSingletonActiveTransport:
    """Singleton and property tests."""

    def test_singleton_identity(self, engine):
        """Two get_instance calls return the same object."""
        from greenlang.agents.mrv.employee_commuting.active_transport_calculator import (
            ActiveTransportCalculatorEngine,
        )
        assert engine is ActiveTransportCalculatorEngine.get_instance()

    def test_engine_id(self, engine):
        """Engine ID is correct."""
        assert engine.engine_id == "active_transport_calculator_engine"

    def test_engine_version(self, engine):
        """Engine version is 1.0.0."""
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# 2. CYCLING (ZERO OPERATIONAL EMISSIONS)
# ===========================================================================

class TestCycling:
    """Tests for calculate_cycling()."""

    def test_cycling_zero_operational_emissions(self, engine):
        """Cycling has zero operational CO2e."""
        result = engine.calculate_cycling(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
        )
        assert result["operational_co2e_kg"] == _q(_ZERO)
        assert result["mode"] == "cycling"
        assert result["mode_category"] == "active_transport"

    def test_cycling_with_lifecycle(self, engine):
        """Cycling with lifecycle includes manufacturing and maintenance."""
        result = engine.calculate_cycling(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
            include_lifecycle=True,
        )
        assert result["lifecycle_co2e_kg"] > _ZERO
        assert result["total_co2e_kg"] == result["lifecycle_co2e_kg"]

    def test_cycling_without_lifecycle(self, engine):
        """Cycling without lifecycle has zero total emissions."""
        result = engine.calculate_cycling(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
            include_lifecycle=False,
        )
        assert result["lifecycle_co2e_kg"] == _q(_ZERO)
        assert result["total_co2e_kg"] == _q(_ZERO)

    def test_cycling_annual_distance(self, engine):
        """5 km x 2 x 240 days = 2400 km annual distance."""
        result = engine.calculate_cycling(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
            wfh_fraction=Decimal("0.0"),
        )
        expected = _q(Decimal("5.0") * 2 * 240)
        assert Decimal(result["annual_distance_km"]) == expected

    def test_cycling_known_lifecycle_calculation(self, engine):
        """Verify lifecycle: 2400 km x 0.00700 = 16.80 kg CO2e."""
        result = engine.calculate_cycling(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
            wfh_fraction=Decimal("0.0"),
            include_lifecycle=True,
        )
        annual_dist = Decimal("5.0") * 2 * 240  # 2400
        expected_lifecycle = _q(annual_dist * Decimal("0.00700"))
        assert result["lifecycle_co2e_kg"] == expected_lifecycle


# ===========================================================================
# 3. WALKING (ZERO OPERATIONAL EMISSIONS)
# ===========================================================================

class TestWalking:
    """Tests for calculate_walking()."""

    def test_walking_zero_operational_emissions(self, engine):
        """Walking has zero operational CO2e."""
        result = engine.calculate_walking(
            one_way_distance_km=Decimal("2.0"),
            working_days=240,
        )
        assert result["operational_co2e_kg"] == _q(_ZERO)
        assert result["mode"] == "walking"

    def test_walking_lifecycle_lower_than_cycling(self, engine):
        """Walking lifecycle factor (0.003) is lower than cycling (0.007)."""
        walk = engine.calculate_walking(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
            include_lifecycle=True,
        )
        cycle = engine.calculate_cycling(
            one_way_distance_km=Decimal("5.0"),
            working_days=240,
            include_lifecycle=True,
        )
        assert walk["lifecycle_co2e_kg"] < cycle["lifecycle_co2e_kg"]

    def test_walking_wfh_reduces_distance(self, engine):
        """50% WFH halves the annual distance."""
        full = engine.calculate_walking(
            one_way_distance_km=Decimal("2.0"),
            working_days=240,
            wfh_fraction=Decimal("0.0"),
        )
        wfh = engine.calculate_walking(
            one_way_distance_km=Decimal("2.0"),
            working_days=240,
            wfh_fraction=Decimal("0.50"),
        )
        assert Decimal(wfh["annual_distance_km"]) == _q(
            Decimal(full["annual_distance_km"]) * Decimal("0.50")
        )


# ===========================================================================
# 4. E-BIKE (ELECTRIC MICRO-MOBILITY)
# ===========================================================================

class TestEBike:
    """Tests for calculate_e_bike()."""

    def test_e_bike_positive_operational_emissions(self, engine):
        """E-bike has positive operational emissions from electricity."""
        result = engine.calculate_e_bike(
            one_way_distance_km=Decimal("8.0"),
            country_code="US",
        )
        assert result["operational_co2e_kg"] > _ZERO
        assert result["mode"] == "e_bike"
        assert result["mode_category"] == "micro_mobility"

    def test_e_bike_energy_consumption(self, engine):
        """E-bike standard: 0.011 kWh/km."""
        result = engine.calculate_e_bike(
            one_way_distance_km=Decimal("10.0"),
            e_bike_type="standard",
            working_days=240,
            wfh_fraction=Decimal("0.0"),
            country_code="US",
        )
        annual_dist = _q(Decimal("10.0") * 2 * 240)
        expected_kwh = _q(annual_dist * Decimal("0.01100"))
        assert result["energy_kwh"] == expected_kwh

    def test_e_bike_with_lifecycle(self, engine):
        """E-bike with lifecycle includes manufacturing, battery, maintenance."""
        result = engine.calculate_e_bike(
            one_way_distance_km=Decimal("8.0"),
            include_lifecycle=True,
            country_code="US",
        )
        assert result["lifecycle_co2e_kg"] > _ZERO
        assert result["total_co2e_kg"] > result["operational_co2e_kg"]

    def test_e_bike_without_lifecycle(self, engine):
        """E-bike without lifecycle has only operational emissions."""
        result = engine.calculate_e_bike(
            one_way_distance_km=Decimal("8.0"),
            include_lifecycle=False,
            country_code="US",
        )
        assert result["lifecycle_co2e_kg"] == _q(_ZERO)
        assert result["total_co2e_kg"] == result["operational_co2e_kg"]

    @pytest.mark.parametrize("bike_type", ["standard", "cargo", "speed_pedelec"])
    def test_e_bike_types_valid(self, engine, bike_type):
        """All three e-bike types are accepted."""
        result = engine.calculate_e_bike(
            one_way_distance_km=Decimal("8.0"),
            e_bike_type=bike_type,
            country_code="US",
        )
        assert result["e_bike_type"] == bike_type

    def test_e_bike_invalid_type_raises(self, engine):
        """Unknown e-bike type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown e_bike_type"):
            engine.calculate_e_bike(
                one_way_distance_km=Decimal("8.0"),
                e_bike_type="flying",
                country_code="US",
            )

    def test_e_bike_green_grid_lower(self, engine):
        """E-bike in Norway (low grid) has lower operational emissions than in India."""
        no = engine.calculate_e_bike(
            one_way_distance_km=Decimal("10.0"),
            country_code="NO",
            include_lifecycle=False,
        )
        india = engine.calculate_e_bike(
            one_way_distance_km=Decimal("10.0"),
            country_code="IN",
            include_lifecycle=False,
        )
        assert no["operational_co2e_kg"] < india["operational_co2e_kg"]


# ===========================================================================
# 5. E-SCOOTER
# ===========================================================================

class TestEScooter:
    """Tests for calculate_e_scooter()."""

    def test_e_scooter_positive_emissions(self, engine):
        """E-scooter has positive emissions."""
        result = engine.calculate_e_scooter(
            one_way_distance_km=Decimal("5.0"),
            country_code="US",
        )
        assert result["total_co2e_kg"] > _ZERO
        assert result["mode"] == "e_scooter"

    @pytest.mark.parametrize("scooter_type", ["personal", "shared"])
    def test_e_scooter_types_valid(self, engine, scooter_type):
        """Both personal and shared scooter types are accepted."""
        result = engine.calculate_e_scooter(
            one_way_distance_km=Decimal("5.0"),
            scooter_type=scooter_type,
            country_code="US",
        )
        assert result["scooter_type"] == scooter_type

    def test_shared_scooter_higher_lifecycle(self, engine):
        """Shared scooters have higher lifecycle emissions per km than personal."""
        personal = engine.calculate_e_scooter(
            one_way_distance_km=Decimal("5.0"),
            scooter_type="personal",
            country_code="US",
            include_lifecycle=True,
        )
        shared = engine.calculate_e_scooter(
            one_way_distance_km=Decimal("5.0"),
            scooter_type="shared",
            country_code="US",
            include_lifecycle=True,
        )
        assert shared["lifecycle_co2e_kg"] > personal["lifecycle_co2e_kg"]

    def test_e_scooter_invalid_type_raises(self, engine):
        """Unknown scooter type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scooter_type"):
            engine.calculate_e_scooter(
                one_way_distance_km=Decimal("5.0"),
                scooter_type="hoverboard",
                country_code="US",
            )


# ===========================================================================
# 6. INPUT VALIDATION
# ===========================================================================

class TestActiveTransportValidation:
    """Input validation tests across active transport modes."""

    def test_negative_distance_cycling_raises(self, engine):
        """Negative distance raises ValueError for cycling."""
        with pytest.raises(ValueError):
            engine.calculate_cycling(one_way_distance_km=Decimal("-1.0"))

    def test_zero_distance_walking_raises(self, engine):
        """Zero distance raises ValueError for walking."""
        with pytest.raises(ValueError):
            engine.calculate_walking(one_way_distance_km=Decimal("0.0"))

    def test_wfh_greater_than_one_raises(self, engine):
        """WFH fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_cycling(
                one_way_distance_km=Decimal("5.0"),
                wfh_fraction=Decimal("1.5"),
            )


# ===========================================================================
# 7. PROVENANCE AND COUNT
# ===========================================================================

class TestActiveTransportProvenance:
    """Provenance and count tests."""

    def test_provenance_hash_64_chars(self, engine):
        """Provenance hash is 64-character SHA-256 hex."""
        result = engine.calculate_cycling(one_way_distance_km=Decimal("5.0"))
        assert len(result["provenance_hash"]) == 64

    def test_calculation_count_tracks(self, engine):
        """Calculation count increments for each call."""
        engine.calculate_cycling(one_way_distance_km=Decimal("5.0"))
        engine.calculate_walking(one_way_distance_km=Decimal("2.0"))
        engine.calculate_e_bike(
            one_way_distance_km=Decimal("8.0"), country_code="US",
        )
        assert engine.calculation_count == 3
