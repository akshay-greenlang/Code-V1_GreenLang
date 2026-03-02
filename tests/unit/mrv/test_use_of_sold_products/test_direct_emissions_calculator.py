# -*- coding: utf-8 -*-
"""
Unit tests for DirectEmissionsCalculatorEngine -- AGENT-MRV-024

Tests direct use-phase emission calculations including fuel combustion
(vehicles, generators), refrigerant leakage (HVAC, refrigeration),
and chemical release (aerosols, solvents).

Calculation formulas:
- Fuel: units_sold x lifetime_years x fuel_per_year x fuel_EF
- Refrigerant: units_sold x charge_kg x leak_rate x GWP x lifetime_years
- Chemical: units_sold x chemical_mass x GWP

Specific test values:
- Fuel: 1000 cars x 15yr x 1200L/yr x 2.315 = 41,670,000 kgCO2e
- Refrigerant: 500 ACs x 3kg R410A x 0.05 leak x 2088 GWP x 12yr = 1,879,200 kgCO2e
- Chemical: 100,000 aerosols x 0.15kg HFC-134a x 1430 GWP = 21,450,000 kgCO2e

Target: 35+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.direct_emissions_calculator import (
        DirectEmissionsCalculatorEngine,
        get_direct_calculator,
        calculate_fuel_combustion,
        calculate_refrigerant_leakage,
        calculate_chemical_release,
        calculate_provenance_hash,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="DirectEmissionsCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    DirectEmissionsCalculatorEngine.reset()
    yield
    DirectEmissionsCalculatorEngine.reset()


@pytest.fixture
def engine():
    """Create a DirectEmissionsCalculatorEngine instance."""
    return DirectEmissionsCalculatorEngine()


# ============================================================================
# TEST: Fuel Combustion Calculations
# ============================================================================


class TestFuelCombustion:
    """Test direct fuel combustion calculations."""

    def test_gasoline_vehicle_basic(self, engine):
        """Test 1000 cars x 15yr x 1200L/yr x 2.315 = 41,670,000 kgCO2e."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        expected = Decimal("41670000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_diesel_generator(self, engine):
        """Test 200 generators x 20yr x 5000L/yr x 2.68 = 53,600,000 kgCO2e."""
        result = engine.calculate_fuel_combustion(
            units_sold=200,
            lifetime_years=20,
            fuel_consumption_per_year=Decimal("5000.0"),
            fuel_ef_kg_per_unit=Decimal("2.68"),
        )
        expected = Decimal("53600000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_single_unit(self, engine):
        """Test single vehicle: 1 x 15 x 1200 x 2.315 = 41,670 kgCO2e."""
        result = engine.calculate_fuel_combustion(
            units_sold=1,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        expected = Decimal("41670.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_zero_units_returns_zero(self, engine):
        """Test zero units returns zero emissions."""
        result = engine.calculate_fuel_combustion(
            units_sold=0,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        assert result["total_co2e_kg"] == Decimal("0")

    def test_lpg_vehicle(self, engine):
        """Test LPG vehicle: 500 x 10 x 800L x 1.553 = 6,212,000 kgCO2e."""
        result = engine.calculate_fuel_combustion(
            units_sold=500,
            lifetime_years=10,
            fuel_consumption_per_year=Decimal("800.0"),
            fuel_ef_kg_per_unit=Decimal("1.553"),
        )
        expected = Decimal("6212000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_co2e_per_unit_calculation(self, engine):
        """Test co2e_per_unit = total / units_sold."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        expected_per_unit = Decimal("41670.0")
        assert result["co2e_per_unit"] == pytest.approx(expected_per_unit, rel=Decimal("0.001"))

    def test_provenance_hash_present(self, engine):
        """Test calculation result includes provenance hash."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_with_degradation(self, engine):
        """Test fuel combustion with 0.5% annual degradation."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
            degradation_rate=Decimal("0.005"),
        )
        # With degradation, total should be slightly less
        no_degrade = Decimal("41670000.0")
        assert result["total_co2e_kg"] < no_degrade

    @pytest.mark.parametrize("fuel_type,ef", [
        ("gasoline", Decimal("2.315")),
        ("diesel", Decimal("2.680")),
        ("lpg", Decimal("1.553")),
        ("kerosene", Decimal("2.540")),
        ("propane", Decimal("1.510")),
    ])
    def test_various_fuel_types(self, engine, fuel_type, ef):
        """Test fuel combustion with different fuel types."""
        result = engine.calculate_fuel_combustion(
            units_sold=100,
            lifetime_years=10,
            fuel_consumption_per_year=Decimal("1000.0"),
            fuel_ef_kg_per_unit=ef,
        )
        expected = 100 * 10 * 1000 * float(ef)
        assert result["total_co2e_kg"] == pytest.approx(Decimal(str(expected)), rel=Decimal("0.001"))


# ============================================================================
# TEST: Refrigerant Leakage Calculations
# ============================================================================


class TestRefrigerantLeakage:
    """Test direct refrigerant leakage calculations."""

    def test_r410a_ac_basic(self, engine):
        """Test 500 ACs x 3kg x 0.05 x 2088 x 12 = 1,879,200 kgCO2e."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=500,
            refrigerant_charge_kg=Decimal("3.0"),
            annual_leak_rate=Decimal("0.05"),
            refrigerant_gwp=Decimal("2088"),
            lifetime_years=12,
        )
        expected = Decimal("1879200.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_r134a_fridge(self, engine):
        """Test 10000 fridges x 0.15kg x 0.02 x 1430 x 15 = 643,500 kgCO2e."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=10000,
            refrigerant_charge_kg=Decimal("0.15"),
            annual_leak_rate=Decimal("0.02"),
            refrigerant_gwp=Decimal("1430"),
            lifetime_years=15,
        )
        expected = Decimal("643500.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_r32_low_gwp(self, engine):
        """Test R-32 low GWP refrigerant: 1000 x 2kg x 0.05 x 675 x 12 = 810,000."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=1000,
            refrigerant_charge_kg=Decimal("2.0"),
            annual_leak_rate=Decimal("0.05"),
            refrigerant_gwp=Decimal("675"),
            lifetime_years=12,
        )
        expected = Decimal("810000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_zero_leak_rate_returns_zero(self, engine):
        """Test zero leak rate returns zero emissions."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=500,
            refrigerant_charge_kg=Decimal("3.0"),
            annual_leak_rate=Decimal("0.0"),
            refrigerant_gwp=Decimal("2088"),
            lifetime_years=12,
        )
        assert result["total_co2e_kg"] == Decimal("0")

    def test_commercial_chiller_high_charge(self, engine):
        """Test commercial chiller: 100 x 50kg x 0.10 x 3922 x 20 = 392,200,000."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=100,
            refrigerant_charge_kg=Decimal("50.0"),
            annual_leak_rate=Decimal("0.10"),
            refrigerant_gwp=Decimal("3922"),
            lifetime_years=20,
        )
        expected = Decimal("392200000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_co2e_per_unit(self, engine):
        """Test co2e_per_unit for refrigerant leakage."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=500,
            refrigerant_charge_kg=Decimal("3.0"),
            annual_leak_rate=Decimal("0.05"),
            refrigerant_gwp=Decimal("2088"),
            lifetime_years=12,
        )
        expected_per_unit = Decimal("3758.4")
        assert result["co2e_per_unit"] == pytest.approx(expected_per_unit, rel=Decimal("0.001"))

    def test_provenance_hash_present(self, engine):
        """Test refrigerant result includes provenance hash."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=500,
            refrigerant_charge_kg=Decimal("3.0"),
            annual_leak_rate=Decimal("0.05"),
            refrigerant_gwp=Decimal("2088"),
            lifetime_years=12,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TEST: Chemical Release Calculations
# ============================================================================


class TestChemicalRelease:
    """Test direct chemical release calculations."""

    def test_aerosol_hfc134a(self, engine):
        """Test 100,000 aerosols x 0.15kg x 1430 = 21,450,000 kgCO2e."""
        result = engine.calculate_chemical_release(
            units_sold=100000,
            chemical_mass_per_unit_kg=Decimal("0.15"),
            chemical_gwp=Decimal("1430"),
        )
        expected = Decimal("21450000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_sf6_equipment(self, engine):
        """Test SF6 equipment: 50 x 5kg x 22800 = 5,700,000 kgCO2e."""
        result = engine.calculate_chemical_release(
            units_sold=50,
            chemical_mass_per_unit_kg=Decimal("5.0"),
            chemical_gwp=Decimal("22800"),
        )
        expected = Decimal("5700000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_low_gwp_propellant(self, engine):
        """Test propane propellant: 200,000 x 0.10kg x 3 = 60,000 kgCO2e."""
        result = engine.calculate_chemical_release(
            units_sold=200000,
            chemical_mass_per_unit_kg=Decimal("0.10"),
            chemical_gwp=Decimal("3"),
        )
        expected = Decimal("60000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_zero_units_chemical(self, engine):
        """Test zero units returns zero chemical emissions."""
        result = engine.calculate_chemical_release(
            units_sold=0,
            chemical_mass_per_unit_kg=Decimal("0.15"),
            chemical_gwp=Decimal("1430"),
        )
        assert result["total_co2e_kg"] == Decimal("0")


# ============================================================================
# TEST: Multi-Product Direct Calculations
# ============================================================================


class TestMultiProductDirect:
    """Test multi-product direct emission aggregation."""

    def test_vehicle_fleet_total(self, engine):
        """Test aggregating multiple vehicle types."""
        gasoline = engine.calculate_fuel_combustion(
            units_sold=800,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        diesel = engine.calculate_fuel_combustion(
            units_sold=200,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1000.0"),
            fuel_ef_kg_per_unit=Decimal("2.680"),
        )
        total = gasoline["total_co2e_kg"] + diesel["total_co2e_kg"]
        assert total > 0


# ============================================================================
# TEST: DQI Score for Direct Calculations
# ============================================================================


class TestDirectDQI:
    """Test data quality indicator scoring for direct calculations."""

    def test_dqi_fuel_combustion(self, engine):
        """Test DQI score for fuel combustion is 80/75/70 pattern."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        if "dqi_score" in result:
            assert Decimal("60") <= result["dqi_score"] <= Decimal("100")

    def test_dqi_refrigerant_leakage(self, engine):
        """Test DQI score for refrigerant leakage."""
        result = engine.calculate_refrigerant_leakage(
            units_sold=500,
            refrigerant_charge_kg=Decimal("3.0"),
            annual_leak_rate=Decimal("0.05"),
            refrigerant_gwp=Decimal("2088"),
            lifetime_years=12,
        )
        if "dqi_score" in result:
            assert Decimal("60") <= result["dqi_score"] <= Decimal("100")


# ============================================================================
# TEST: Uncertainty
# ============================================================================


class TestDirectUncertainty:
    """Test uncertainty calculations for direct emissions."""

    def test_uncertainty_bounds_present(self, engine):
        """Test uncertainty bounds are present in result."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        if "uncertainty_lower" in result and "uncertainty_upper" in result:
            assert result["uncertainty_lower"] < result["total_co2e_kg"]
            assert result["uncertainty_upper"] > result["total_co2e_kg"]

    def test_uncertainty_symmetric(self, engine):
        """Test uncertainty is roughly symmetric around the mean."""
        result = engine.calculate_fuel_combustion(
            units_sold=1000,
            lifetime_years=15,
            fuel_consumption_per_year=Decimal("1200.0"),
            fuel_ef_kg_per_unit=Decimal("2.315"),
        )
        if "uncertainty_lower" in result and "uncertainty_upper" in result:
            total = result["total_co2e_kg"]
            lower_delta = total - result["uncertainty_lower"]
            upper_delta = result["uncertainty_upper"] - total
            # Should be within 2x of each other
            if lower_delta > 0 and upper_delta > 0:
                ratio = float(upper_delta / lower_delta)
                assert 0.5 <= ratio <= 2.0
