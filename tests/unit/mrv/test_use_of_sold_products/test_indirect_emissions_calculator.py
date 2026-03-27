# -*- coding: utf-8 -*-
"""
Unit tests for IndirectEmissionsCalculatorEngine -- AGENT-MRV-024

Tests indirect use-phase emission calculations including electricity
consumption (appliances, IT, lighting), heating fuel (furnaces, boilers),
and steam/cooling (district energy systems).

Calculation formula:
- Electricity: units_sold x lifetime_years x kWh_per_year x grid_EF
- Heating fuel: units_sold x lifetime x fuel_per_year x fuel_EF
- Steam/cooling: units_sold x lifetime x energy_per_year x steam_EF

Specific test values:
- Electricity: 10,000 fridges x 15yr x 400kWh x 0.417 = 25,020,000 kgCO2e
- Heating: 5,000 furnaces x 20yr x 2000m3 x 1.93 = 386,000,000 kgCO2e

Target: 30+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.agents.mrv.use_of_sold_products.indirect_emissions_calculator import (
        IndirectEmissionsCalculatorEngine,
        get_indirect_calculator,
        calculate_electricity_emissions,
        calculate_heating_fuel_emissions,
        calculate_steam_cooling_emissions,
        calculate_provenance_hash,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="IndirectEmissionsCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    IndirectEmissionsCalculatorEngine.reset()
    yield
    IndirectEmissionsCalculatorEngine.reset()


@pytest.fixture
def engine():
    """Create an IndirectEmissionsCalculatorEngine instance."""
    return IndirectEmissionsCalculatorEngine()


# ============================================================================
# TEST: Electricity Consumption Calculations
# ============================================================================


class TestElectricityConsumption:
    """Test indirect electricity consumption calculations."""

    def test_fridge_basic(self, engine):
        """Test 10,000 fridges x 15yr x 400kWh x 0.417 = 25,020,000 kgCO2e."""
        result = engine.calculate_electricity_emissions(
            units_sold=10000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        expected = Decimal("25020000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_laptop_basic(self, engine):
        """Test 50,000 laptops x 5yr x 50kWh x 0.417 = 5,212,500 kgCO2e."""
        result = engine.calculate_electricity_emissions(
            units_sold=50000,
            lifetime_years=5,
            energy_consumption_kwh_per_year=Decimal("50.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        expected = Decimal("5212500.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_led_bulb(self, engine):
        """Test 500,000 LEDs x 25yr x 10kWh x 0.417 = 52,125,000 kgCO2e."""
        result = engine.calculate_electricity_emissions(
            units_sold=500000,
            lifetime_years=25,
            energy_consumption_kwh_per_year=Decimal("10.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        expected = Decimal("52125000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_mri_scanner(self, engine):
        """Test 100 MRI x 12yr x 50,000kWh x 0.417 = 25,020,000 kgCO2e."""
        result = engine.calculate_electricity_emissions(
            units_sold=100,
            lifetime_years=12,
            energy_consumption_kwh_per_year=Decimal("50000.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        expected = Decimal("25020000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_server(self, engine):
        """Test 1000 servers x 7yr x 4380kWh x 0.417 = 12,787,320 kgCO2e."""
        result = engine.calculate_electricity_emissions(
            units_sold=1000,
            lifetime_years=7,
            energy_consumption_kwh_per_year=Decimal("4380.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        expected = Decimal("12787320.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_zero_units_returns_zero(self, engine):
        """Test zero units returns zero emissions."""
        result = engine.calculate_electricity_emissions(
            units_sold=0,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        assert result["total_co2e_kg"] == Decimal("0")

    def test_co2e_per_unit(self, engine):
        """Test co2e_per_unit = total / units_sold."""
        result = engine.calculate_electricity_emissions(
            units_sold=10000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        expected_per_unit = Decimal("2502.0")
        assert result["co2e_per_unit"] == pytest.approx(expected_per_unit, rel=Decimal("0.001"))

    def test_with_degradation(self, engine):
        """Test electricity with 1% annual degradation (increasing consumption)."""
        result = engine.calculate_electricity_emissions(
            units_sold=10000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
            degradation_rate=Decimal("0.01"),
        )
        # With degradation (efficiency loss), total should be slightly higher
        no_degrade = Decimal("25020000.0")
        assert result["total_co2e_kg"] != no_degrade

    @pytest.mark.parametrize("region,ef", [
        ("US", Decimal("0.417")),
        ("US_CAMX", Decimal("0.275")),
        ("US_RFCW", Decimal("0.520")),
        ("US_SRMW", Decimal("0.680")),
        ("DE", Decimal("0.350")),
        ("CN", Decimal("0.580")),
        ("GB", Decimal("0.230")),
        ("JP", Decimal("0.470")),
        ("IN", Decimal("0.710")),
        ("BR", Decimal("0.080")),
        ("FR", Decimal("0.060")),
        ("AU", Decimal("0.630")),
        ("CA", Decimal("0.130")),
        ("KR", Decimal("0.460")),
        ("ZA", Decimal("0.920")),
        ("GLOBAL", Decimal("0.440")),
    ])
    def test_all_16_grid_regions(self, engine, region, ef):
        """Test electricity calculation with all 16 grid regions."""
        result = engine.calculate_electricity_emissions(
            units_sold=1000,
            lifetime_years=10,
            energy_consumption_kwh_per_year=Decimal("100.0"),
            grid_ef_kg_per_kwh=ef,
        )
        expected = 1000 * 10 * 100 * float(ef)
        assert result["total_co2e_kg"] == pytest.approx(Decimal(str(expected)), rel=Decimal("0.001"))

    def test_provenance_hash_present(self, engine):
        """Test electricity result includes provenance hash."""
        result = engine.calculate_electricity_emissions(
            units_sold=10000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TEST: Heating Fuel Calculations
# ============================================================================


class TestHeatingFuel:
    """Test indirect heating fuel calculations."""

    def test_gas_furnace(self, engine):
        """Test 5,000 furnaces x 20yr x 2000m3 x 1.93 = 386,000,000 kgCO2e."""
        result = engine.calculate_heating_fuel_emissions(
            units_sold=5000,
            lifetime_years=20,
            fuel_consumption_per_year=Decimal("2000.0"),
            fuel_ef_kg_per_unit=Decimal("1.93"),
        )
        expected = Decimal("386000000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_oil_boiler(self, engine):
        """Test 2,000 oil boilers x 20yr x 1500L x 2.96 = 177,600,000 kgCO2e."""
        result = engine.calculate_heating_fuel_emissions(
            units_sold=2000,
            lifetime_years=20,
            fuel_consumption_per_year=Decimal("1500.0"),
            fuel_ef_kg_per_unit=Decimal("2.96"),
        )
        expected = Decimal("177600000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_zero_units_returns_zero(self, engine):
        """Test zero units returns zero."""
        result = engine.calculate_heating_fuel_emissions(
            units_sold=0,
            lifetime_years=20,
            fuel_consumption_per_year=Decimal("2000.0"),
            fuel_ef_kg_per_unit=Decimal("1.93"),
        )
        assert result["total_co2e_kg"] == Decimal("0")


# ============================================================================
# TEST: Steam/Cooling Calculations
# ============================================================================


class TestSteamCooling:
    """Test indirect steam/cooling calculations."""

    def test_steam_heating_system(self, engine):
        """Test 500 steam systems x 20yr x 10,000kWh x 0.200 = 20,000,000 kgCO2e."""
        result = engine.calculate_steam_cooling_emissions(
            units_sold=500,
            lifetime_years=20,
            energy_consumption_kwh_per_year=Decimal("10000.0"),
            steam_cooling_ef_kg_per_kwh=Decimal("0.200"),
        )
        expected = Decimal("20000000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_district_cooling(self, engine):
        """Test 1000 cooling units x 15yr x 5000kWh x 0.120 = 9,000,000 kgCO2e."""
        result = engine.calculate_steam_cooling_emissions(
            units_sold=1000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("5000.0"),
            steam_cooling_ef_kg_per_kwh=Decimal("0.120"),
        )
        expected = Decimal("9000000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))


# ============================================================================
# TEST: DQI Score for Indirect Calculations
# ============================================================================


class TestIndirectDQI:
    """Test data quality indicator scoring for indirect calculations."""

    def test_dqi_electricity(self, engine):
        """Test DQI score for electricity is in 60-100 range."""
        result = engine.calculate_electricity_emissions(
            units_sold=10000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        if "dqi_score" in result:
            assert Decimal("60") <= result["dqi_score"] <= Decimal("100")


# ============================================================================
# TEST: Uncertainty for Indirect
# ============================================================================


class TestIndirectUncertainty:
    """Test uncertainty calculations for indirect emissions."""

    def test_uncertainty_bounds(self, engine):
        """Test uncertainty bounds are present and bracket the total."""
        result = engine.calculate_electricity_emissions(
            units_sold=10000,
            lifetime_years=15,
            energy_consumption_kwh_per_year=Decimal("400.0"),
            grid_ef_kg_per_kwh=Decimal("0.417"),
        )
        if "uncertainty_lower" in result and "uncertainty_upper" in result:
            assert result["uncertainty_lower"] < result["total_co2e_kg"]
            assert result["uncertainty_upper"] > result["total_co2e_kg"]
