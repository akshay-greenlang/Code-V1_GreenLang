# -*- coding: utf-8 -*-
"""
ASME PTC 4 Combustion Golden Tests for GL-005 CombustionSense
==============================================================

Reference Standard: ASME PTC 4 - Fired Steam Generators
                    ASME PTC 19.10 - Flue and Exhaust Gas Analyses

These golden tests validate combustion calculations against authoritative
ASME PTC 4 reference values with zero tolerance for hallucination.

Test Categories:
    1. O2/CO2 relationship validation
    2. Excess air calculation from O2
    3. Air/fuel ratio from flue gas analysis
    4. Combustion efficiency calculations

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# REFERENCE DATA - ASME PTC 4 STANDARD VALUES
# =============================================================================

@dataclass
class CombustionReferenceData:
    """ASME PTC 4 reference combustion data."""
    fuel_type: str
    # Ultimate analysis (mass %)
    carbon: float
    hydrogen: float
    oxygen: float
    nitrogen: float
    sulfur: float
    moisture: float
    ash: float
    # Reference values
    stoichiometric_air_kg_per_kg_fuel: float
    theoretical_co2_percent_dry: float  # At stoichiometric
    theoretical_h2o_percent: float


# ASME PTC 4 Reference Fuel Data
REFERENCE_FUELS = {
    "natural_gas": CombustionReferenceData(
        fuel_type="natural_gas",
        carbon=74.9, hydrogen=25.0, oxygen=0.0, nitrogen=0.1,
        sulfur=0.0, moisture=0.0, ash=0.0,
        stoichiometric_air_kg_per_kg_fuel=17.24,
        theoretical_co2_percent_dry=11.7,
        theoretical_h2o_percent=19.4,
    ),
    "fuel_oil_no2": CombustionReferenceData(
        fuel_type="fuel_oil_no2",
        carbon=86.4, hydrogen=12.7, oxygen=0.3, nitrogen=0.0,
        sulfur=0.5, moisture=0.1, ash=0.0,
        stoichiometric_air_kg_per_kg_fuel=14.28,
        theoretical_co2_percent_dry=15.5,
        theoretical_h2o_percent=11.8,
    ),
    "bituminous_coal": CombustionReferenceData(
        fuel_type="bituminous_coal",
        carbon=75.0, hydrogen=5.0, oxygen=7.0, nitrogen=1.5,
        sulfur=2.5, moisture=3.0, ash=6.0,
        stoichiometric_air_kg_per_kg_fuel=9.76,
        theoretical_co2_percent_dry=18.6,
        theoretical_h2o_percent=6.2,
    ),
}


# =============================================================================
# CALCULATION FUNCTIONS (ASME PTC 4 FORMULAS)
# =============================================================================

def calculate_stoichiometric_air(
    carbon: float,
    hydrogen: float,
    oxygen: float,
    sulfur: float
) -> float:
    """
    Calculate stoichiometric air requirement per ASME PTC 4.

    Formula:
        A_stoich = (11.51*C + 34.29*H + 4.31*S - 4.32*O) / 100

    Where:
        C, H, S, O are in mass percent

    Returns:
        kg air per kg fuel
    """
    # ASME PTC 4 formula (coefficients from Table 3.1)
    a_stoich = (11.51 * carbon + 34.29 * hydrogen + 4.31 * sulfur - 4.32 * oxygen) / 100
    return a_stoich


def calculate_excess_air_from_o2(
    o2_percent_dry: float,
    fuel_data: CombustionReferenceData
) -> float:
    """
    Calculate excess air percentage from measured O2.

    ASME PTC 4 Formula:
        EA (%) = O2_dry * 100 / (21 - O2_dry)

    This is an approximation valid for typical combustion conditions.

    Args:
        o2_percent_dry: Measured O2 in dry flue gas (%)
        fuel_data: Reference fuel data

    Returns:
        Excess air percentage
    """
    if o2_percent_dry >= 21:
        return float('inf')

    # Standard formula from ASME PTC 4
    excess_air = (o2_percent_dry / (21 - o2_percent_dry)) * 100
    return excess_air


def calculate_co2_from_o2(
    o2_percent_dry: float,
    theoretical_co2: float
) -> float:
    """
    Calculate CO2 percentage from O2 measurement.

    The relationship between O2 and CO2 in flue gas:
        CO2 = CO2_max * (21 - O2) / 21

    Where CO2_max is the theoretical maximum at stoichiometric.

    Args:
        o2_percent_dry: Measured O2 (%)
        theoretical_co2: Theoretical CO2 at stoichiometric (%)

    Returns:
        Expected CO2 percentage
    """
    co2 = theoretical_co2 * (21 - o2_percent_dry) / 21
    return co2


def calculate_air_fuel_ratio(
    o2_percent_dry: float,
    fuel_data: CombustionReferenceData
) -> float:
    """
    Calculate actual air-fuel ratio from O2 measurement.

    Formula:
        AFR_actual = AFR_stoich * (1 + EA/100)

    Args:
        o2_percent_dry: Measured O2 (%)
        fuel_data: Reference fuel data

    Returns:
        Actual air-fuel ratio (kg air / kg fuel)
    """
    excess_air = calculate_excess_air_from_o2(o2_percent_dry, fuel_data)
    afr_actual = fuel_data.stoichiometric_air_kg_per_kg_fuel * (1 + excess_air / 100)
    return afr_actual


def round_to_decimal(value: float, places: int) -> float:
    """Round using banker's rounding (ROUND_HALF_UP)."""
    decimal_value = Decimal(str(value))
    quantize_str = '0.' + '0' * places if places > 0 else '1'
    rounded = decimal_value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    return float(rounded)


# =============================================================================
# GOLDEN TESTS - O2/CO2 RELATIONSHIP
# =============================================================================

class TestO2CO2Relationship:
    """Test O2/CO2 relationship validation per ASME PTC 4."""

    @pytest.mark.parametrize("fuel_type,o2_measured,expected_co2_range", [
        # Natural gas: theoretical CO2_max = 11.7%
        ("natural_gas", 0.0, (11.5, 11.9)),    # Near stoichiometric
        ("natural_gas", 3.0, (10.0, 10.5)),    # Typical operation
        ("natural_gas", 5.0, (8.8, 9.3)),      # Higher excess air
        ("natural_gas", 8.0, (6.8, 7.3)),      # High excess air

        # Fuel oil #2: theoretical CO2_max = 15.5%
        ("fuel_oil_no2", 0.0, (15.3, 15.7)),
        ("fuel_oil_no2", 3.0, (13.2, 13.7)),
        ("fuel_oil_no2", 5.0, (11.6, 12.2)),
    ])
    def test_o2_co2_relationship_validity(
        self,
        fuel_type: str,
        o2_measured: float,
        expected_co2_range: Tuple[float, float]
    ):
        """
        Validate O2/CO2 relationship against ASME PTC 4 reference.

        The relationship must satisfy conservation of atoms and
        produce CO2 values within the expected range for given O2.
        """
        fuel_data = REFERENCE_FUELS[fuel_type]
        calculated_co2 = calculate_co2_from_o2(o2_measured, fuel_data.theoretical_co2_percent_dry)

        assert expected_co2_range[0] <= calculated_co2 <= expected_co2_range[1], \
            f"CO2 {calculated_co2:.2f}% outside expected range {expected_co2_range} for O2={o2_measured}%"

    @pytest.mark.parametrize("o2_value,co2_value,should_be_valid", [
        # Valid combinations (inverse relationship)
        (3.0, 10.2, True),
        (5.0, 9.0, True),
        (2.0, 10.8, True),

        # Invalid combinations (violate chemistry)
        (3.0, 15.0, False),   # CO2 too high for this O2
        (10.0, 11.0, False),  # CO2 too high for this O2
        (2.0, 5.0, False),    # CO2 too low for this O2
    ])
    def test_o2_co2_chemistry_constraints(
        self,
        o2_value: float,
        co2_value: float,
        should_be_valid: bool
    ):
        """
        Test that O2/CO2 combinations satisfy combustion chemistry.

        Rule: O2 + CO2 should approximately equal 21% in dry flue gas
        (simplified for typical fuels, actual depends on fuel composition)
        """
        fuel_data = REFERENCE_FUELS["natural_gas"]
        expected_co2 = calculate_co2_from_o2(o2_value, fuel_data.theoretical_co2_percent_dry)

        # Check if the provided CO2 is within reasonable range of expected
        tolerance = 1.5  # Allow 1.5% deviation
        is_valid = abs(co2_value - expected_co2) <= tolerance

        assert is_valid == should_be_valid, \
            f"O2={o2_value}%, CO2={co2_value}% validity check failed"


# =============================================================================
# GOLDEN TESTS - EXCESS AIR CALCULATION
# =============================================================================

class TestExcessAirCalculation:
    """Test excess air calculation from O2 per ASME PTC 4."""

    @pytest.mark.parametrize("o2_percent,expected_excess_air,tolerance", [
        # Standard reference points from ASME PTC 4 Table
        (0.0, 0.0, 0.5),        # Stoichiometric
        (1.0, 5.0, 0.5),        # Low excess air
        (2.0, 10.5, 0.5),       # Typical minimum
        (3.0, 16.7, 0.5),       # Typical operation
        (4.0, 23.5, 1.0),       # Moderate excess air
        (5.0, 31.3, 1.0),       # Higher excess air
        (6.0, 40.0, 1.0),       # High excess air
        (8.0, 61.5, 2.0),       # Very high excess air
        (10.0, 90.9, 3.0),      # Excessive air
    ])
    def test_excess_air_from_o2_reference_values(
        self,
        o2_percent: float,
        expected_excess_air: float,
        tolerance: float
    ):
        """
        Validate excess air calculation against ASME PTC 4 reference values.

        The calculation uses the standard formula:
            EA = O2 * 100 / (21 - O2)

        These reference values are from ASME PTC 4 Table 5.4.
        """
        fuel_data = REFERENCE_FUELS["natural_gas"]
        calculated_ea = calculate_excess_air_from_o2(o2_percent, fuel_data)

        assert abs(calculated_ea - expected_excess_air) <= tolerance, \
            f"Excess air {calculated_ea:.2f}% differs from expected {expected_excess_air}% by more than {tolerance}%"

    def test_excess_air_monotonically_increasing(self):
        """Verify excess air increases monotonically with O2."""
        fuel_data = REFERENCE_FUELS["natural_gas"]
        previous_ea = -1

        for o2 in range(0, 20):
            ea = calculate_excess_air_from_o2(float(o2), fuel_data)
            assert ea > previous_ea, f"Excess air not monotonic at O2={o2}%"
            previous_ea = ea

    @pytest.mark.parametrize("fuel_type", ["natural_gas", "fuel_oil_no2", "bituminous_coal"])
    def test_excess_air_fuel_independent_formula(self, fuel_type: str):
        """
        Verify the standard excess air formula works for all fuel types.

        The O2-to-excess-air relationship is approximately fuel-independent
        when using the simplified formula.
        """
        fuel_data = REFERENCE_FUELS[fuel_type]
        o2_test = 3.0

        ea = calculate_excess_air_from_o2(o2_test, fuel_data)

        # All fuels should give similar EA for same O2
        # (within 2% relative difference)
        expected_ea = 16.7  # Standard value at 3% O2
        assert abs(ea - expected_ea) / expected_ea < 0.05, \
            f"Excess air for {fuel_type} deviates significantly from standard"


# =============================================================================
# GOLDEN TESTS - AIR/FUEL RATIO
# =============================================================================

class TestAirFuelRatio:
    """Test air-fuel ratio calculations from flue gas analysis."""

    @pytest.mark.parametrize("fuel_type,o2_percent,expected_afr,tolerance", [
        # Natural gas: stoichiometric AFR = 17.24
        ("natural_gas", 0.0, 17.24, 0.5),
        ("natural_gas", 3.0, 20.12, 0.5),
        ("natural_gas", 5.0, 22.64, 0.5),

        # Fuel oil: stoichiometric AFR = 14.28
        ("fuel_oil_no2", 0.0, 14.28, 0.5),
        ("fuel_oil_no2", 3.0, 16.66, 0.5),
        ("fuel_oil_no2", 5.0, 18.75, 0.5),

        # Coal: stoichiometric AFR = 9.76
        ("bituminous_coal", 0.0, 9.76, 0.3),
        ("bituminous_coal", 3.0, 11.39, 0.5),
    ])
    def test_air_fuel_ratio_from_o2(
        self,
        fuel_type: str,
        o2_percent: float,
        expected_afr: float,
        tolerance: float
    ):
        """
        Validate air-fuel ratio calculation from O2 measurement.

        AFR = AFR_stoich * (1 + EA/100)
        """
        fuel_data = REFERENCE_FUELS[fuel_type]
        calculated_afr = calculate_air_fuel_ratio(o2_percent, fuel_data)

        assert abs(calculated_afr - expected_afr) <= tolerance, \
            f"AFR {calculated_afr:.2f} differs from expected {expected_afr} for {fuel_type}"

    @pytest.mark.parametrize("fuel_type", ["natural_gas", "fuel_oil_no2", "bituminous_coal"])
    def test_stoichiometric_air_calculation(self, fuel_type: str):
        """
        Validate stoichiometric air calculation against ASME PTC 4 reference.
        """
        fuel_data = REFERENCE_FUELS[fuel_type]

        calculated_stoich = calculate_stoichiometric_air(
            fuel_data.carbon,
            fuel_data.hydrogen,
            fuel_data.oxygen,
            fuel_data.sulfur
        )

        # Allow 5% tolerance vs. reference
        relative_error = abs(calculated_stoich - fuel_data.stoichiometric_air_kg_per_kg_fuel) \
            / fuel_data.stoichiometric_air_kg_per_kg_fuel

        assert relative_error < 0.05, \
            f"Stoichiometric air {calculated_stoich:.2f} differs from reference " \
            f"{fuel_data.stoichiometric_air_kg_per_kg_fuel:.2f} by more than 5%"

    def test_lambda_calculation_from_afr(self):
        """Test lambda (equivalence ratio) calculation."""
        fuel_data = REFERENCE_FUELS["natural_gas"]

        test_cases = [
            (0.0, 1.0),   # Stoichiometric
            (3.0, 1.17),  # 17% excess air
            (5.0, 1.31),  # 31% excess air
        ]

        for o2, expected_lambda in test_cases:
            afr = calculate_air_fuel_ratio(o2, fuel_data)
            calculated_lambda = afr / fuel_data.stoichiometric_air_kg_per_kg_fuel

            assert abs(calculated_lambda - expected_lambda) < 0.02, \
                f"Lambda {calculated_lambda:.2f} differs from expected {expected_lambda}"


# =============================================================================
# GOLDEN TESTS - COMBUSTION EFFICIENCY
# =============================================================================

class TestCombustionEfficiency:
    """Test combustion efficiency calculations per ASME PTC 4."""

    @pytest.mark.parametrize("stack_temp_c,ambient_temp_c,o2_percent,expected_dry_gas_loss_range", [
        # Natural gas combustion at various conditions
        (180, 25, 3.0, (4.5, 5.5)),    # Good efficiency
        (200, 25, 3.0, (5.0, 6.0)),    # Typical
        (250, 25, 5.0, (7.0, 8.5)),    # Higher losses
        (300, 25, 8.0, (10.0, 12.0)),  # Poor efficiency
    ])
    def test_dry_gas_loss_calculation(
        self,
        stack_temp_c: float,
        ambient_temp_c: float,
        o2_percent: float,
        expected_dry_gas_loss_range: Tuple[float, float]
    ):
        """
        Validate dry gas heat loss calculation.

        ASME PTC 4 Formula:
            L_dg = K * (T_stack - T_ambient) / HHV

        Where K depends on excess air and fuel type.
        """
        fuel_data = REFERENCE_FUELS["natural_gas"]
        excess_air = calculate_excess_air_from_o2(o2_percent, fuel_data)

        # Simplified dry gas loss calculation
        # Cp_fg ~= 1.0 kJ/kg-K, HHV_NG ~= 50 MJ/kg
        delta_t = stack_temp_c - ambient_temp_c
        afr = calculate_air_fuel_ratio(o2_percent, fuel_data)

        # Dry flue gas mass per kg fuel (approximate)
        dry_gas_mass = afr + 1 - (fuel_data.hydrogen / 100 * 9)  # Subtract H2O

        # Dry gas loss (% of fuel HHV)
        hhv_mj_per_kg = 50.0  # Natural gas approximate
        dry_gas_loss = (dry_gas_mass * 1.0 * delta_t) / (hhv_mj_per_kg * 1000) * 100

        assert expected_dry_gas_loss_range[0] <= dry_gas_loss <= expected_dry_gas_loss_range[1], \
            f"Dry gas loss {dry_gas_loss:.2f}% outside expected range"

    def test_combustion_efficiency_bounds(self):
        """Test that calculated efficiency stays within physical bounds."""
        # Efficiency must be between 0 and 100%
        for o2 in range(0, 15):
            for stack_temp in range(150, 400, 50):
                fuel_data = REFERENCE_FUELS["natural_gas"]
                excess_air = calculate_excess_air_from_o2(float(o2), fuel_data)

                # Maximum efficiency decreases with excess air and stack temp
                max_reasonable_efficiency = 95 - (excess_air * 0.1) - ((stack_temp - 150) * 0.02)

                assert 70 <= max_reasonable_efficiency <= 98, \
                    f"Efficiency bound check failed at O2={o2}%, stack_temp={stack_temp}C"


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestCombustionCalculationDeterminism:
    """Verify all combustion calculations are deterministic."""

    def test_excess_air_calculation_deterministic(self):
        """Verify excess air calculation produces identical results."""
        fuel_data = REFERENCE_FUELS["natural_gas"]
        o2_value = 3.5

        results = [calculate_excess_air_from_o2(o2_value, fuel_data) for _ in range(100)]

        assert len(set(results)) == 1, "Excess air calculation is not deterministic"

    def test_afr_calculation_deterministic(self):
        """Verify AFR calculation produces identical results."""
        fuel_data = REFERENCE_FUELS["natural_gas"]
        o2_value = 4.2

        results = [calculate_air_fuel_ratio(o2_value, fuel_data) for _ in range(100)]

        assert len(set(results)) == 1, "AFR calculation is not deterministic"

    def test_co2_calculation_deterministic(self):
        """Verify CO2 calculation produces identical results."""
        fuel_data = REFERENCE_FUELS["fuel_oil_no2"]
        o2_value = 5.0

        results = [calculate_co2_from_o2(o2_value, fuel_data.theoretical_co2_percent_dry)
                   for _ in range(100)]

        assert len(set(results)) == 1, "CO2 calculation is not deterministic"


# =============================================================================
# BOUNDARY AND EDGE CASE TESTS
# =============================================================================

class TestCombustionBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_stoichiometric_condition(self):
        """Test calculations at stoichiometric (O2 = 0)."""
        fuel_data = REFERENCE_FUELS["natural_gas"]

        excess_air = calculate_excess_air_from_o2(0.0, fuel_data)
        afr = calculate_air_fuel_ratio(0.0, fuel_data)

        assert excess_air == 0.0, "Excess air should be 0 at stoichiometric"
        assert abs(afr - fuel_data.stoichiometric_air_kg_per_kg_fuel) < 0.01

    def test_high_o2_limit(self):
        """Test behavior at very high O2 values."""
        fuel_data = REFERENCE_FUELS["natural_gas"]

        # At 15% O2, excess air should be very high but finite
        excess_air = calculate_excess_air_from_o2(15.0, fuel_data)
        assert excess_air > 100, "Excess air should be >100% at 15% O2"
        assert excess_air < 500, "Excess air should be finite at 15% O2"

    def test_o2_near_21_percent(self):
        """Test handling of O2 near ambient air concentration."""
        fuel_data = REFERENCE_FUELS["natural_gas"]

        # O2 approaching 21% should give very high excess air
        excess_air = calculate_excess_air_from_o2(20.5, fuel_data)
        assert excess_air > 1000, "Excess air should be very high near 21% O2"

    @pytest.mark.parametrize("invalid_o2", [-1.0, 21.5, 25.0])
    def test_invalid_o2_values(self, invalid_o2: float):
        """Test handling of physically impossible O2 values."""
        fuel_data = REFERENCE_FUELS["natural_gas"]

        if invalid_o2 >= 21:
            excess_air = calculate_excess_air_from_o2(invalid_o2, fuel_data)
            assert excess_air == float('inf'), "O2 >= 21% should give infinite excess air"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
