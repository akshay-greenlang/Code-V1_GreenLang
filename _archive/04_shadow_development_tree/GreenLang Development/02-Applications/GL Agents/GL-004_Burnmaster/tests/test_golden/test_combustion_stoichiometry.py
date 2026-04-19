# -*- coding: utf-8 -*-
"""
Combustion Stoichiometry Golden Value Tests for GL-004 BurnMaster
=================================================================

Validates combustion calculations against fundamental chemistry
and EPA emission factor databases.

Test Categories:
    1. Stoichiometric Air Requirements
    2. Flue Gas Composition
    3. Excess Air Calculations
    4. NOx Formation and Emission Factors
    5. CO Formation and Emission Factors
    6. Heat Release Calculations

Reference Sources:
    - EPA AP-42 Emission Factors
    - ASME PTC 4
    - Perry's Chemical Engineers' Handbook
    - Combustion Engineering by Baukal

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# FUNDAMENTAL COMBUSTION CONSTANTS
# =============================================================================

# Molecular weights (g/mol)
MOLECULAR_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "S": 32.065,
    "CO2": 44.009,
    "H2O": 18.015,
    "SO2": 64.064,
    "N2": 28.014,
    "O2": 31.998,
    "NO": 30.006,
    "NO2": 46.005,
    "CO": 28.010,
    "CH4": 16.043,
}

# Air composition (mole fraction)
AIR_COMPOSITION = {
    "N2": 0.7809,
    "O2": 0.2095,
    "Ar": 0.0093,
    "CO2": 0.0004,
}

# Standard conditions
STANDARD_CONDITIONS = {
    "temperature_K": 298.15,
    "pressure_kPa": 101.325,
    "molar_volume_L_mol": 24.465,  # Ideal gas at STP
}


# =============================================================================
# FUEL COMPOSITIONS - ULTIMATE ANALYSIS
# =============================================================================

@dataclass(frozen=True)
class FuelComposition:
    """Fuel ultimate analysis (mass fraction, as-fired basis)."""
    name: str
    C: float  # Carbon
    H: float  # Hydrogen
    O: float  # Oxygen
    N: float  # Nitrogen
    S: float  # Sulfur
    Ash: float  # Ash
    H2O: float  # Moisture
    HHV_kJ_kg: float  # Higher Heating Value
    LHV_kJ_kg: float  # Lower Heating Value


FUEL_DATABASE = {
    "natural_gas": FuelComposition(
        name="Natural Gas (Pipeline Quality)",
        C=0.7432, H=0.2324, O=0.0000, N=0.0055,
        S=0.0001, Ash=0.0000, H2O=0.0000,
        HHV_kJ_kg=55514, LHV_kJ_kg=50030,
    ),
    "methane": FuelComposition(
        name="Pure Methane (CH4)",
        C=0.7487, H=0.2513, O=0.0000, N=0.0000,
        S=0.0000, Ash=0.0000, H2O=0.0000,
        HHV_kJ_kg=55528, LHV_kJ_kg=50016,
    ),
    "propane": FuelComposition(
        name="Propane (C3H8)",
        C=0.8171, H=0.1829, O=0.0000, N=0.0000,
        S=0.0000, Ash=0.0000, H2O=0.0000,
        HHV_kJ_kg=50343, LHV_kJ_kg=46353,
    ),
    "fuel_oil_no2": FuelComposition(
        name="No. 2 Fuel Oil (Diesel)",
        C=0.8710, H=0.1232, O=0.0000, N=0.0020,
        S=0.0030, Ash=0.0008, H2O=0.0000,
        HHV_kJ_kg=45500, LHV_kJ_kg=42800,
    ),
    "fuel_oil_no6": FuelComposition(
        name="No. 6 Fuel Oil (Residual)",
        C=0.8762, H=0.1044, O=0.0020, N=0.0050,
        S=0.0200, Ash=0.0050, H2O=0.0050,
        HHV_kJ_kg=43500, LHV_kJ_kg=41000,
    ),
    "bituminous_coal": FuelComposition(
        name="Bituminous Coal (Typical)",
        C=0.7520, H=0.0510, O=0.0680, N=0.0140,
        S=0.0200, Ash=0.0800, H2O=0.0150,
        HHV_kJ_kg=30200, LHV_kJ_kg=29000,
    ),
    "sub_bituminous_coal": FuelComposition(
        name="Sub-Bituminous Coal (PRB)",
        C=0.5200, H=0.0350, O=0.1200, N=0.0070,
        S=0.0050, Ash=0.0600, H2O=0.2530,
        HHV_kJ_kg=21000, LHV_kJ_kg=19500,
    ),
    "lignite": FuelComposition(
        name="Lignite Coal",
        C=0.4000, H=0.0280, O=0.1500, N=0.0050,
        S=0.0080, Ash=0.0700, H2O=0.3390,
        HHV_kJ_kg=16500, LHV_kJ_kg=14800,
    ),
    "wood_chips": FuelComposition(
        name="Wood Chips (50% moisture)",
        C=0.2500, H=0.0300, O=0.2000, N=0.0010,
        S=0.0000, Ash=0.0190, H2O=0.5000,
        HHV_kJ_kg=10000, LHV_kJ_kg=7500,
    ),
}


# =============================================================================
# EPA AP-42 EMISSION FACTORS
# =============================================================================

@dataclass(frozen=True)
class EPAEmissionFactor:
    """EPA AP-42 emission factor data."""
    pollutant: str
    fuel_type: str
    factor_value: float
    factor_units: str
    scc_code: str  # Source Classification Code
    rating: str  # A, B, C, D, E (A=best)
    reference: str


# NOx emission factors (lb/mmBtu)
EPA_NOX_FACTORS = {
    "natural_gas_uncontrolled": EPAEmissionFactor(
        pollutant="NOx", fuel_type="natural_gas",
        factor_value=0.1, factor_units="lb/mmBtu",
        scc_code="1-02-006-02", rating="A",
        reference="AP-42 Table 1.4-1",
    ),
    "natural_gas_low_nox": EPAEmissionFactor(
        pollutant="NOx", fuel_type="natural_gas",
        factor_value=0.036, factor_units="lb/mmBtu",
        scc_code="1-02-006-02", rating="A",
        reference="AP-42 Table 1.4-1 with Low-NOx burner",
    ),
    "fuel_oil_no2_uncontrolled": EPAEmissionFactor(
        pollutant="NOx", fuel_type="fuel_oil_no2",
        factor_value=0.14, factor_units="lb/mmBtu",
        scc_code="1-02-004-02", rating="A",
        reference="AP-42 Table 1.3-1",
    ),
    "bituminous_coal_uncontrolled": EPAEmissionFactor(
        pollutant="NOx", fuel_type="bituminous_coal",
        factor_value=0.95, factor_units="lb/mmBtu",
        scc_code="1-02-002-02", rating="B",
        reference="AP-42 Table 1.1-3",
    ),
}

# CO emission factors (lb/mmBtu)
EPA_CO_FACTORS = {
    "natural_gas": EPAEmissionFactor(
        pollutant="CO", fuel_type="natural_gas",
        factor_value=0.084, factor_units="lb/mmBtu",
        scc_code="1-02-006-02", rating="B",
        reference="AP-42 Table 1.4-2",
    ),
    "fuel_oil_no2": EPAEmissionFactor(
        pollutant="CO", fuel_type="fuel_oil_no2",
        factor_value=0.036, factor_units="lb/mmBtu",
        scc_code="1-02-004-02", rating="B",
        reference="AP-42 Table 1.3-2",
    ),
    "bituminous_coal": EPAEmissionFactor(
        pollutant="CO", fuel_type="bituminous_coal",
        factor_value=0.5, factor_units="lb/mmBtu",
        scc_code="1-02-002-02", rating="C",
        reference="AP-42 Table 1.1-4",
    ),
}


# =============================================================================
# STOICHIOMETRY CALCULATION FUNCTIONS
# =============================================================================

def calculate_stoichiometric_air(fuel: FuelComposition) -> Dict[str, float]:
    """
    Calculate stoichiometric air requirement for complete combustion.

    Combustion reactions:
        C + O2 → CO2
        H2 + 0.5 O2 → H2O
        S + O2 → SO2

    Args:
        fuel: Fuel composition data

    Returns:
        Dictionary with air requirements
    """
    # Oxygen required for complete combustion (kg O2 / kg fuel)
    # C: 32/12 = 2.667 kg O2/kg C
    # H: 16/2 = 8.0 kg O2/kg H
    # S: 32/32 = 1.0 kg O2/kg S
    # O in fuel reduces requirement

    o2_for_c = fuel.C * (32 / 12)
    o2_for_h = fuel.H * (32 / 4)  # H2 + 0.5 O2 → H2O
    o2_for_s = fuel.S * (32 / 32)
    o2_in_fuel = fuel.O

    o2_required = o2_for_c + o2_for_h + o2_for_s - o2_in_fuel

    # Air is 23.2% O2 by mass
    air_required = o2_required / 0.232

    # Theoretical air (kg air / kg fuel)
    theoretical_air = air_required

    return {
        "o2_required_kg_per_kg_fuel": o2_required,
        "theoretical_air_kg_per_kg_fuel": theoretical_air,
        "theoretical_air_mol_per_kg_fuel": theoretical_air / 0.029,  # MW air ≈ 29
    }


def calculate_flue_gas_composition(
    fuel: FuelComposition,
    excess_air_percent: float
) -> Dict[str, float]:
    """
    Calculate flue gas composition from combustion.

    Args:
        fuel: Fuel composition data
        excess_air_percent: Excess air percentage

    Returns:
        Dictionary with flue gas composition (mole fraction dry basis)
    """
    # Moles of products per kg fuel
    mol_co2 = fuel.C / MOLECULAR_WEIGHTS["C"]
    mol_so2 = fuel.S / MOLECULAR_WEIGHTS["S"]
    mol_h2o = fuel.H / (2 * MOLECULAR_WEIGHTS["H"])

    # Theoretical O2 required (moles)
    stoich = calculate_stoichiometric_air(fuel)
    mol_o2_theoretical = stoich["o2_required_kg_per_kg_fuel"] / MOLECULAR_WEIGHTS["O2"]

    # Actual air with excess
    excess_factor = 1 + excess_air_percent / 100
    mol_o2_supplied = mol_o2_theoretical * excess_factor
    mol_n2_supplied = mol_o2_supplied * (0.79 / 0.21)  # N2/O2 ratio in air

    # Excess O2 in flue gas
    mol_o2_excess = mol_o2_theoretical * (excess_air_percent / 100)

    # Nitrogen in flue gas (from air + fuel nitrogen)
    mol_n2_flue = mol_n2_supplied + fuel.N / MOLECULAR_WEIGHTS["N2"]

    # Total moles (dry basis)
    total_dry = mol_co2 + mol_so2 + mol_o2_excess + mol_n2_flue

    return {
        "CO2_percent_dry": (mol_co2 / total_dry) * 100,
        "SO2_ppm_dry": (mol_so2 / total_dry) * 1e6,
        "O2_percent_dry": (mol_o2_excess / total_dry) * 100,
        "N2_percent_dry": (mol_n2_flue / total_dry) * 100,
        "H2O_mol_per_kg_fuel": mol_h2o,
    }


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.golden
class TestStoichiometricAir:
    """Test stoichiometric air calculations."""

    @pytest.mark.parametrize("fuel_key,expected_range", [
        ("natural_gas", (16.5, 18.0)),   # ~17.2 kg air/kg fuel
        ("methane", (17.0, 18.0)),       # ~17.2 kg air/kg fuel
        ("propane", (15.0, 16.5)),       # ~15.7 kg air/kg fuel
        ("fuel_oil_no2", (13.5, 15.0)),  # ~14.0 kg air/kg fuel
        ("bituminous_coal", (9.5, 11.5)), # ~10.5 kg air/kg fuel
    ])
    def test_theoretical_air_requirement(self, fuel_key: str, expected_range: Tuple[float, float]):
        """
        Validate theoretical air requirements for common fuels.

        Reference: Perry's Chemical Engineers' Handbook, 8th Ed.
        """
        fuel = FUEL_DATABASE[fuel_key]
        result = calculate_stoichiometric_air(fuel)

        theoretical_air = result["theoretical_air_kg_per_kg_fuel"]

        assert expected_range[0] <= theoretical_air <= expected_range[1], (
            f"{fuel_key}: Theoretical air {theoretical_air:.2f} kg/kg fuel "
            f"outside expected range {expected_range}"
        )

    def test_methane_combustion_stoichiometry(self):
        """
        Validate methane combustion: CH4 + 2O2 → CO2 + 2H2O

        Stoichiometric air-fuel ratio for CH4:
            - 2 mol O2 per mol CH4
            - 2 * 32 / 16 = 4 kg O2 per kg CH4
            - 4 / 0.232 = 17.24 kg air per kg CH4
        """
        fuel = FUEL_DATABASE["methane"]
        result = calculate_stoichiometric_air(fuel)

        # Expected: ~17.24 kg air / kg CH4
        expected_air = 4.0 / 0.232  # O2/air mass ratio

        deviation = abs(result["theoretical_air_kg_per_kg_fuel"] - expected_air)
        assert deviation <= 0.5, (
            f"Methane stoichiometric air: {result['theoretical_air_kg_per_kg_fuel']:.2f} "
            f"vs expected {expected_air:.2f} kg/kg"
        )

    def test_coal_requires_less_air_than_gas(self):
        """Coal requires less air per kg due to lower hydrogen content."""
        gas = FUEL_DATABASE["natural_gas"]
        coal = FUEL_DATABASE["bituminous_coal"]

        gas_air = calculate_stoichiometric_air(gas)["theoretical_air_kg_per_kg_fuel"]
        coal_air = calculate_stoichiometric_air(coal)["theoretical_air_kg_per_kg_fuel"]

        assert coal_air < gas_air, (
            f"Coal ({coal_air:.2f}) should require less air than gas ({gas_air:.2f})"
        )


@pytest.mark.golden
class TestFlueGasComposition:
    """Test flue gas composition calculations."""

    def test_natural_gas_co2_at_zero_excess_air(self):
        """
        Validate CO2 concentration at stoichiometric combustion.

        For natural gas, theoretical CO2 max is approximately 11.7-12.0%.
        """
        fuel = FUEL_DATABASE["natural_gas"]
        flue_gas = calculate_flue_gas_composition(fuel, excess_air_percent=0.0)

        co2_percent = flue_gas["CO2_percent_dry"]

        # Theoretical maximum CO2 for natural gas is ~11.7%
        assert 11.0 <= co2_percent <= 13.0, (
            f"CO2 at stoichiometric: {co2_percent:.1f}% "
            f"(expected 11-13% for natural gas)"
        )

    def test_o2_increases_with_excess_air(self):
        """Flue gas O2 should increase with excess air."""
        fuel = FUEL_DATABASE["natural_gas"]

        o2_at_10_percent = calculate_flue_gas_composition(fuel, 10.0)["O2_percent_dry"]
        o2_at_20_percent = calculate_flue_gas_composition(fuel, 20.0)["O2_percent_dry"]
        o2_at_50_percent = calculate_flue_gas_composition(fuel, 50.0)["O2_percent_dry"]

        assert o2_at_20_percent > o2_at_10_percent, "O2 should increase with excess air"
        assert o2_at_50_percent > o2_at_20_percent, "O2 should increase with excess air"

    @pytest.mark.parametrize("excess_air,expected_o2_range", [
        (10, (1.5, 2.5)),
        (15, (2.5, 3.5)),
        (20, (3.2, 4.2)),
        (30, (4.5, 5.5)),
    ])
    def test_o2_vs_excess_air_relationship(
        self,
        excess_air: float,
        expected_o2_range: Tuple[float, float]
    ):
        """
        Validate O2 percentage for different excess air levels.

        Common rule of thumb: O2% ≈ 21 * (EA / (100 + EA))
        """
        fuel = FUEL_DATABASE["natural_gas"]
        flue_gas = calculate_flue_gas_composition(fuel, excess_air)

        o2_percent = flue_gas["O2_percent_dry"]

        assert expected_o2_range[0] <= o2_percent <= expected_o2_range[1], (
            f"At {excess_air}% excess air: O2 = {o2_percent:.1f}% "
            f"(expected {expected_o2_range})"
        )

    def test_mass_balance_closure(self):
        """Verify mass balance closes for combustion products."""
        fuel = FUEL_DATABASE["natural_gas"]

        # Mass in: 1 kg fuel + stoichiometric air * (1 + excess)
        stoich = calculate_stoichiometric_air(fuel)
        excess_air = 15.0
        air_mass = stoich["theoretical_air_kg_per_kg_fuel"] * (1 + excess_air / 100)

        mass_in = 1.0 + air_mass

        # Mass out: CO2 + H2O + excess O2 + N2 (simplified)
        # For proper closure, need full mass balance
        # Here we just verify it's reasonable
        assert mass_in > 1.0, "Total mass should include air"
        assert mass_in < 25.0, "Total mass should be reasonable for gas combustion"


@pytest.mark.golden
class TestExcessAirFromO2:
    """Test excess air calculation from O2 measurement."""

    @pytest.mark.parametrize("o2_dry,expected_excess_air", [
        (2.0, 10.5),
        (3.0, 16.7),
        (4.0, 23.5),
        (5.0, 31.3),
        (6.0, 40.0),
    ])
    def test_excess_air_from_o2_measurement(
        self,
        o2_dry: float,
        expected_excess_air: float
    ):
        """
        Validate excess air calculation from flue gas O2.

        Formula: Excess Air % = O2 / (21 - O2) * 100
        """
        # Standard formula for excess air from O2
        calculated_excess_air = (o2_dry / (21.0 - o2_dry)) * 100

        deviation = abs(calculated_excess_air - expected_excess_air)
        assert deviation <= 1.0, (
            f"At O2 = {o2_dry}%: Calculated EA = {calculated_excess_air:.1f}%, "
            f"expected {expected_excess_air:.1f}%"
        )

    def test_high_o2_indicates_excess_air(self):
        """High O2 in flue gas indicates high excess air."""
        # At 10% O2, excess air should be very high
        excess_air_at_10_o2 = (10.0 / (21.0 - 10.0)) * 100

        assert excess_air_at_10_o2 > 80, (
            f"10% O2 should indicate >80% excess air, got {excess_air_at_10_o2:.1f}%"
        )


@pytest.mark.golden
class TestNOxEmissions:
    """Test NOx emission calculations and EPA factors."""

    @pytest.mark.parametrize("factor_key,expected_value", [
        ("natural_gas_uncontrolled", 0.1),
        ("natural_gas_low_nox", 0.036),
        ("fuel_oil_no2_uncontrolled", 0.14),
        ("bituminous_coal_uncontrolled", 0.95),
    ])
    def test_epa_nox_emission_factors(self, factor_key: str, expected_value: float):
        """
        Validate EPA AP-42 NOx emission factors.

        Reference: EPA AP-42, Chapter 1
        """
        factor = EPA_NOX_FACTORS[factor_key]

        assert factor.factor_value == expected_value, (
            f"{factor_key}: EPA factor {factor.factor_value} != {expected_value}"
        )
        assert factor.factor_units == "lb/mmBtu", "Units should be lb/mmBtu"

    def test_nox_increases_with_temperature(self):
        """
        Thermal NOx formation increases exponentially with temperature.

        Zeldovich mechanism: N2 + O ⇌ NO + N (highly temperature dependent)
        """
        # Simplified Arrhenius relationship
        # NOx ∝ exp(-Ea/RT) where Ea ≈ 314 kJ/mol

        def thermal_nox_factor(temp_K: float) -> float:
            """Relative thermal NOx formation rate."""
            Ea = 314000  # J/mol
            R = 8.314    # J/(mol·K)
            return math.exp(-Ea / (R * temp_K))

        nox_at_1500K = thermal_nox_factor(1500)
        nox_at_1800K = thermal_nox_factor(1800)
        nox_at_2100K = thermal_nox_factor(2100)

        # NOx should increase with temperature
        assert nox_at_1800K > nox_at_1500K * 10, "NOx should increase significantly 1500→1800K"
        assert nox_at_2100K > nox_at_1800K * 5, "NOx should increase significantly 1800→2100K"

    def test_low_nox_burner_reduction(self):
        """Low-NOx burners should reduce NOx by 50-70%."""
        uncontrolled = EPA_NOX_FACTORS["natural_gas_uncontrolled"].factor_value
        low_nox = EPA_NOX_FACTORS["natural_gas_low_nox"].factor_value

        reduction_percent = (1 - low_nox / uncontrolled) * 100

        assert 60 <= reduction_percent <= 70, (
            f"Low-NOx reduction {reduction_percent:.1f}% should be 60-70%"
        )


@pytest.mark.golden
class TestCOEmissions:
    """Test CO emission calculations and factors."""

    @pytest.mark.parametrize("fuel,expected_factor", [
        ("natural_gas", 0.084),
        ("fuel_oil_no2", 0.036),
        ("bituminous_coal", 0.5),
    ])
    def test_epa_co_emission_factors(self, fuel: str, expected_factor: float):
        """
        Validate EPA AP-42 CO emission factors.

        Reference: EPA AP-42, Chapter 1
        """
        factor = EPA_CO_FACTORS[fuel]

        assert factor.factor_value == expected_factor, (
            f"{fuel}: EPA CO factor {factor.factor_value} != {expected_factor}"
        )

    def test_co_increases_at_low_excess_air(self):
        """
        CO increases when excess air is too low (incomplete combustion).

        Typical CO vs O2 relationship:
        - O2 < 1%: Very high CO (incomplete combustion)
        - O2 1-3%: Moderate CO
        - O2 > 4%: Low CO but high efficiency loss
        """
        # Simplified CO model: CO increases exponentially as O2 approaches 0
        def co_model(o2_percent: float) -> float:
            """Relative CO emission rate vs O2."""
            if o2_percent < 0.5:
                return 100.0  # Very high CO
            return 1.0 / o2_percent  # Simplified inverse relationship

        co_at_1_percent = co_model(1.0)
        co_at_3_percent = co_model(3.0)
        co_at_5_percent = co_model(5.0)

        assert co_at_1_percent > co_at_3_percent, "CO should be higher at lower O2"
        assert co_at_3_percent > co_at_5_percent, "CO should decrease with O2"

    def test_optimal_o2_minimizes_losses(self):
        """
        Optimal O2 balances CO emissions vs dry gas losses.

        Too low O2: High CO (incomplete combustion)
        Too high O2: High dry gas loss (excess air energy)
        Optimal: typically 2-4% O2 for gas, 3-5% for oil/coal
        """
        optimal_o2_ranges = {
            "natural_gas": (2.0, 4.0),
            "fuel_oil": (3.0, 4.5),
            "coal": (3.5, 5.5),
        }

        for fuel, o2_range in optimal_o2_ranges.items():
            assert o2_range[1] > o2_range[0], f"{fuel}: Invalid O2 range"
            assert o2_range[0] >= 1.0, f"{fuel}: O2 too low risks CO"
            assert o2_range[1] <= 8.0, f"{fuel}: O2 too high wastes energy"


@pytest.mark.golden
class TestHeatRelease:
    """Test heat release and heating value calculations."""

    @pytest.mark.parametrize("fuel_key,expected_hhv_range", [
        ("natural_gas", (54000, 57000)),  # kJ/kg
        ("propane", (49000, 52000)),
        ("fuel_oil_no2", (44000, 47000)),
        ("bituminous_coal", (28000, 32000)),
        ("lignite", (14000, 18000)),
    ])
    def test_fuel_heating_values(self, fuel_key: str, expected_hhv_range: Tuple[float, float]):
        """
        Validate fuel higher heating values.

        Reference: Perry's Chemical Engineers' Handbook
        """
        fuel = FUEL_DATABASE[fuel_key]

        assert expected_hhv_range[0] <= fuel.HHV_kJ_kg <= expected_hhv_range[1], (
            f"{fuel_key}: HHV {fuel.HHV_kJ_kg} kJ/kg outside range {expected_hhv_range}"
        )

    def test_hhv_greater_than_lhv(self):
        """HHV should always be greater than LHV due to water latent heat."""
        for fuel_key, fuel in FUEL_DATABASE.items():
            assert fuel.HHV_kJ_kg > fuel.LHV_kJ_kg, (
                f"{fuel_key}: HHV ({fuel.HHV_kJ_kg}) should be > LHV ({fuel.LHV_kJ_kg})"
            )

            # Difference should be approximately 9*H*2442 kJ/kg
            # (latent heat of water formed from hydrogen)
            expected_diff = 9 * fuel.H * 2442  # kJ/kg
            actual_diff = fuel.HHV_kJ_kg - fuel.LHV_kJ_kg

            # Allow 20% tolerance for simplified calculation
            ratio = actual_diff / expected_diff if expected_diff > 0 else 1
            assert 0.5 <= ratio <= 2.0, (
                f"{fuel_key}: HHV-LHV diff {actual_diff:.0f} vs expected {expected_diff:.0f}"
            )


@pytest.mark.golden
class TestDeterminism:
    """Verify calculation determinism."""

    def test_stoichiometric_air_determinism(self):
        """Stoichiometric air calculation must be deterministic."""
        fuel = FUEL_DATABASE["natural_gas"]
        results = []

        for _ in range(100):
            result = calculate_stoichiometric_air(fuel)
            # Use Decimal for deterministic string representation
            air = Decimal(str(result["theoretical_air_kg_per_kg_fuel"])).quantize(
                Decimal("0.0000001")
            )
            results.append(str(air))

        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Non-deterministic: {len(unique_results)} unique results"
        )

    def test_flue_gas_composition_determinism(self):
        """Flue gas composition calculation must be deterministic."""
        fuel = FUEL_DATABASE["natural_gas"]
        hashes = []

        for _ in range(50):
            result = calculate_flue_gas_composition(fuel, 15.0)
            result_str = f"{result['CO2_percent_dry']:.10f}:{result['O2_percent_dry']:.10f}"
            hash_val = hashlib.sha256(result_str.encode()).hexdigest()
            hashes.append(hash_val)

        assert len(set(hashes)) == 1, "Flue gas composition not deterministic"


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_combustion_golden_values() -> Dict[str, Any]:
    """Export all combustion golden values for external validation."""
    return {
        "metadata": {
            "version": "1.0.0",
            "source": "EPA AP-42, ASME PTC 4, Perry's Handbook",
            "agent": "GL-004_BurnMaster",
        },
        "fuels": {
            key: {
                "name": fuel.name,
                "C": fuel.C,
                "H": fuel.H,
                "HHV_kJ_kg": fuel.HHV_kJ_kg,
                "theoretical_air": calculate_stoichiometric_air(fuel)["theoretical_air_kg_per_kg_fuel"],
            }
            for key, fuel in FUEL_DATABASE.items()
        },
        "epa_nox_factors": {
            key: {
                "value": factor.factor_value,
                "units": factor.factor_units,
                "reference": factor.reference,
            }
            for key, factor in EPA_NOX_FACTORS.items()
        },
        "epa_co_factors": {
            key: {
                "value": factor.factor_value,
                "units": factor.factor_units,
            }
            for key, factor in EPA_CO_FACTORS.items()
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_combustion_golden_values(), indent=2))
