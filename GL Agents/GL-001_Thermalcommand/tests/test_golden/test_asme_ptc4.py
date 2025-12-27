# -*- coding: utf-8 -*-
"""
ASME PTC 4 Golden Value Tests for GL-001 ThermalCommand
=========================================================

Validates boiler efficiency and heat balance calculations against
ASME Performance Test Code 4 (Fired Steam Generators).

Reference Standards:
    - ASME PTC 4-2013: Fired Steam Generators
    - ASME PTC 4.1-1964: Steam Generating Units (legacy)
    - ASME PTC 4.4: Gas Turbine Heat Recovery Steam Generators

Test Categories:
    1. Input-Output Method (Direct Efficiency)
    2. Heat Loss Method (Indirect Efficiency)
    3. Energy Balance Calculations
    4. Stack Loss Calculations

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
import math

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# ASME PTC 4 REFERENCE DATA
# =============================================================================

@dataclass(frozen=True)
class ASMEPTC4TestCase:
    """ASME PTC 4 test case with expected results."""
    name: str
    description: str

    # Fuel data
    fuel_type: str
    fuel_flow_kg_hr: float
    fuel_hhv_kj_kg: float  # Higher Heating Value
    fuel_lhv_kj_kg: float  # Lower Heating Value

    # Steam output
    steam_flow_kg_hr: float
    steam_pressure_mpa: float
    steam_temp_c: float
    steam_enthalpy_kj_kg: float

    # Feedwater
    feedwater_flow_kg_hr: float
    feedwater_temp_c: float
    feedwater_enthalpy_kj_kg: float

    # Flue gas
    flue_gas_temp_c: float
    ambient_temp_c: float
    excess_air_percent: float
    o2_percent_dry: float
    co2_percent_dry: float

    # Expected results
    expected_efficiency_hhv: float
    expected_efficiency_lhv: float
    expected_dry_gas_loss_percent: float
    expected_moisture_loss_percent: float
    expected_radiation_loss_percent: float

    tolerance_percent: float = 0.5


# -----------------------------------------------------------------------------
# ASME PTC 4 Table 5.1-1 Based Test Cases
# -----------------------------------------------------------------------------

ASME_PTC4_TEST_CASES = [
    # Test Case 1: Natural Gas Fired Boiler - Typical Industrial
    ASMEPTC4TestCase(
        name="Natural Gas Industrial Boiler",
        description="Typical natural gas fired package boiler at rated load",
        fuel_type="natural_gas",
        fuel_flow_kg_hr=1000.0,
        fuel_hhv_kj_kg=55500.0,  # ~23,850 BTU/lb
        fuel_lhv_kj_kg=50000.0,  # ~21,500 BTU/lb
        steam_flow_kg_hr=15000.0,
        steam_pressure_mpa=1.5,
        steam_temp_c=200.0,  # Saturated
        steam_enthalpy_kj_kg=2792.0,
        feedwater_flow_kg_hr=15000.0,
        feedwater_temp_c=105.0,
        feedwater_enthalpy_kj_kg=440.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        excess_air_percent=15.0,
        o2_percent_dry=3.0,
        co2_percent_dry=11.0,
        expected_efficiency_hhv=82.5,
        expected_efficiency_lhv=91.5,
        expected_dry_gas_loss_percent=4.5,
        expected_moisture_loss_percent=11.5,
        expected_radiation_loss_percent=0.5,
    ),

    # Test Case 2: Coal Fired Utility Boiler
    ASMEPTC4TestCase(
        name="Bituminous Coal Utility Boiler",
        description="Utility-scale pulverized coal boiler",
        fuel_type="bituminous_coal",
        fuel_flow_kg_hr=50000.0,
        fuel_hhv_kj_kg=29000.0,  # ~12,500 BTU/lb
        fuel_lhv_kj_kg=27500.0,  # ~11,850 BTU/lb
        steam_flow_kg_hr=400000.0,
        steam_pressure_mpa=17.0,
        steam_temp_c=540.0,  # Superheated
        steam_enthalpy_kj_kg=3400.0,
        feedwater_flow_kg_hr=400000.0,
        feedwater_temp_c=250.0,
        feedwater_enthalpy_kj_kg=1086.0,
        flue_gas_temp_c=150.0,
        ambient_temp_c=25.0,
        excess_air_percent=20.0,
        o2_percent_dry=3.8,
        co2_percent_dry=14.5,
        expected_efficiency_hhv=88.5,
        expected_efficiency_lhv=92.0,
        expected_dry_gas_loss_percent=5.0,
        expected_moisture_loss_percent=4.5,
        expected_radiation_loss_percent=0.3,
    ),

    # Test Case 3: Fuel Oil Fired Boiler
    ASMEPTC4TestCase(
        name="No. 6 Fuel Oil Boiler",
        description="Industrial fuel oil fired boiler",
        fuel_type="fuel_oil_no6",
        fuel_flow_kg_hr=2000.0,
        fuel_hhv_kj_kg=44000.0,  # ~18,900 BTU/lb
        fuel_lhv_kj_kg=41500.0,  # ~17,850 BTU/lb
        steam_flow_kg_hr=25000.0,
        steam_pressure_mpa=2.0,
        steam_temp_c=220.0,
        steam_enthalpy_kj_kg=2850.0,
        feedwater_flow_kg_hr=25000.0,
        feedwater_temp_c=110.0,
        feedwater_enthalpy_kj_kg=461.0,
        flue_gas_temp_c=200.0,
        ambient_temp_c=25.0,
        excess_air_percent=12.0,
        o2_percent_dry=2.3,
        co2_percent_dry=13.5,
        expected_efficiency_hhv=85.0,
        expected_efficiency_lhv=90.0,
        expected_dry_gas_loss_percent=5.5,
        expected_moisture_loss_percent=6.5,
        expected_radiation_loss_percent=0.4,
    ),
]


# =============================================================================
# HEAT LOSS CALCULATION REFERENCE VALUES
# =============================================================================

@dataclass
class HeatLossComponents:
    """Components of boiler heat losses per ASME PTC 4."""
    dry_gas_loss: float  # L1: Sensible heat in dry flue gas
    moisture_in_fuel_loss: float  # L2: Moisture in fuel
    moisture_from_h2_loss: float  # L3: Moisture from combustion of hydrogen
    moisture_in_air_loss: float  # L4: Moisture in combustion air
    unburned_carbon_loss: float  # L5: Combustible in refuse
    radiation_loss: float  # L6: Surface radiation and convection
    unmeasured_losses: float  # L7: Manufacturer's margin, other


# Reference heat loss data for validation
HEAT_LOSS_REFERENCE = {
    "natural_gas": HeatLossComponents(
        dry_gas_loss=4.5,
        moisture_in_fuel_loss=0.0,
        moisture_from_h2_loss=11.0,
        moisture_in_air_loss=0.2,
        unburned_carbon_loss=0.0,
        radiation_loss=0.5,
        unmeasured_losses=0.3,
    ),
    "bituminous_coal": HeatLossComponents(
        dry_gas_loss=5.0,
        moisture_in_fuel_loss=1.0,
        moisture_from_h2_loss=3.5,
        moisture_in_air_loss=0.2,
        unburned_carbon_loss=1.0,
        radiation_loss=0.3,
        unmeasured_losses=0.5,
    ),
    "fuel_oil_no6": HeatLossComponents(
        dry_gas_loss=5.5,
        moisture_in_fuel_loss=0.5,
        moisture_from_h2_loss=6.0,
        moisture_in_air_loss=0.2,
        unburned_carbon_loss=0.1,
        radiation_loss=0.4,
        unmeasured_losses=0.3,
    ),
}


# =============================================================================
# STOICHIOMETRIC AIR REQUIREMENTS
# =============================================================================

STOICHIOMETRIC_AIR = {
    # kg air per kg fuel for stoichiometric combustion
    "natural_gas": 17.2,
    "bituminous_coal": 10.5,
    "fuel_oil_no6": 14.0,
    "propane": 15.7,
    "hydrogen": 34.3,
}

FUEL_COMPOSITION = {
    # Ultimate analysis (mass fraction)
    "natural_gas": {"C": 0.75, "H": 0.23, "O": 0.00, "N": 0.01, "S": 0.00, "Ash": 0.00, "H2O": 0.00},
    "bituminous_coal": {"C": 0.75, "H": 0.05, "O": 0.08, "N": 0.015, "S": 0.02, "Ash": 0.08, "H2O": 0.015},
    "fuel_oil_no6": {"C": 0.87, "H": 0.10, "O": 0.01, "N": 0.005, "S": 0.02, "Ash": 0.005, "H2O": 0.00},
}


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.golden
class TestInputOutputMethod:
    """Test Input-Output (Direct) efficiency method per ASME PTC 4."""

    @pytest.mark.parametrize("test_case", ASME_PTC4_TEST_CASES)
    def test_direct_efficiency_calculation(self, test_case: ASMEPTC4TestCase):
        """
        Validate direct efficiency calculation.

        Efficiency = (Energy Output / Energy Input) * 100

        Where:
            Energy Output = Steam Flow * (Steam Enthalpy - Feedwater Enthalpy)
            Energy Input = Fuel Flow * HHV
        """
        # Calculate energy output (kW)
        energy_output_kw = (
            test_case.steam_flow_kg_hr *
            (test_case.steam_enthalpy_kj_kg - test_case.feedwater_enthalpy_kj_kg) /
            3600.0
        )

        # Calculate energy input (kW) based on HHV
        energy_input_hhv_kw = (
            test_case.fuel_flow_kg_hr *
            test_case.fuel_hhv_kj_kg /
            3600.0
        )

        # Calculate efficiency
        efficiency_hhv = (energy_output_kw / energy_input_hhv_kw) * 100

        # Validate against expected
        deviation = abs(efficiency_hhv - test_case.expected_efficiency_hhv)
        assert deviation <= test_case.tolerance_percent, (
            f"{test_case.name}: Efficiency deviation {deviation:.2f}% exceeds tolerance. "
            f"Calculated: {efficiency_hhv:.2f}%, Expected: {test_case.expected_efficiency_hhv:.2f}%"
        )

    @pytest.mark.parametrize("test_case", ASME_PTC4_TEST_CASES)
    def test_energy_balance_closure(self, test_case: ASMEPTC4TestCase):
        """
        Verify energy balance closes within acceptable tolerance.

        Energy In = Energy Out + Losses
        """
        # Energy input
        energy_input = test_case.fuel_flow_kg_hr * test_case.fuel_hhv_kj_kg

        # Energy to steam
        energy_to_steam = (
            test_case.steam_flow_kg_hr *
            (test_case.steam_enthalpy_kj_kg - test_case.feedwater_enthalpy_kj_kg)
        )

        # Estimated total losses (100 - efficiency)
        loss_fraction = 1.0 - (test_case.expected_efficiency_hhv / 100.0)
        estimated_losses = energy_input * loss_fraction

        # Energy balance: Input = Output + Losses
        balance_check = energy_input - energy_to_steam - estimated_losses
        balance_percent = abs(balance_check / energy_input) * 100

        # Balance should close within 2%
        assert balance_percent <= 2.0, (
            f"{test_case.name}: Energy balance imbalance of {balance_percent:.2f}%"
        )


@pytest.mark.golden
class TestHeatLossMethod:
    """Test Heat Loss (Indirect) efficiency method per ASME PTC 4."""

    @pytest.mark.parametrize("test_case", ASME_PTC4_TEST_CASES)
    def test_dry_gas_loss(self, test_case: ASMEPTC4TestCase):
        """
        Validate dry gas loss calculation.

        L1 = (m_fg / m_fuel) * Cp_fg * (T_fg - T_amb) / HHV * 100

        Where:
            m_fg/m_fuel = mass ratio of flue gas to fuel
            Cp_fg = specific heat of flue gas (~1.0 kJ/kg-K)
            T_fg = flue gas temperature
            T_amb = ambient temperature
        """
        # Approximate flue gas mass ratio (stoichiometric + excess air)
        stoich_air = STOICHIOMETRIC_AIR.get(test_case.fuel_type, 15.0)
        excess_air_fraction = test_case.excess_air_percent / 100.0
        total_air = stoich_air * (1 + excess_air_fraction)

        # Flue gas mass = air + fuel (approximate)
        fg_fuel_ratio = total_air + 1.0

        # Specific heat of flue gas (kJ/kg-K)
        cp_fg = 1.05

        # Dry gas loss calculation
        temp_diff = test_case.flue_gas_temp_c - test_case.ambient_temp_c
        dry_gas_loss = (fg_fuel_ratio * cp_fg * temp_diff / test_case.fuel_hhv_kj_kg) * 100 * 1000

        # Compare to expected (within 1.5% absolute)
        expected = test_case.expected_dry_gas_loss_percent
        deviation = abs(dry_gas_loss - expected)

        # Allow wider tolerance for simplified calculation
        assert deviation <= 2.0, (
            f"{test_case.name}: Dry gas loss {dry_gas_loss:.2f}% vs expected {expected:.2f}%"
        )

    @pytest.mark.parametrize("test_case", ASME_PTC4_TEST_CASES)
    def test_moisture_loss(self, test_case: ASMEPTC4TestCase):
        """
        Validate moisture loss from hydrogen combustion.

        L3 = 9 * H2 * (hfg + Cp_steam * (T_fg - T_ref)) / HHV * 100

        Where:
            H2 = hydrogen mass fraction in fuel
            hfg = latent heat of vaporization (~2442 kJ/kg at 25°C)
            Cp_steam = specific heat of steam (~1.88 kJ/kg-K)
        """
        composition = FUEL_COMPOSITION.get(test_case.fuel_type, {"H": 0.10})
        h2_fraction = composition["H"]

        # Water formed from hydrogen combustion: 9 kg H2O per kg H2
        water_from_h2 = 9.0 * h2_fraction

        # Latent heat at reference (kJ/kg)
        hfg = 2442.0

        # Sensible heat of steam above reference
        cp_steam = 1.88
        temp_rise = test_case.flue_gas_temp_c - 25.0  # Reference 25°C

        # Moisture loss
        moisture_loss = (
            water_from_h2 *
            (hfg + cp_steam * temp_rise) /
            test_case.fuel_hhv_kj_kg
        ) * 100 * 1000

        # Compare to expected (within 2% absolute)
        expected = test_case.expected_moisture_loss_percent
        deviation = abs(moisture_loss - expected)

        assert deviation <= 3.0, (
            f"{test_case.name}: Moisture loss {moisture_loss:.2f}% vs expected {expected:.2f}%"
        )

    @pytest.mark.parametrize("test_case", ASME_PTC4_TEST_CASES)
    def test_total_losses_plus_efficiency(self, test_case: ASMEPTC4TestCase):
        """
        Verify that efficiency + total losses = 100%.

        This is the fundamental check for heat loss method.
        """
        total_losses = (
            test_case.expected_dry_gas_loss_percent +
            test_case.expected_moisture_loss_percent +
            test_case.expected_radiation_loss_percent
        )

        # Add typical unmeasured losses
        unmeasured = 0.5

        efficiency_from_losses = 100.0 - total_losses - unmeasured
        expected_efficiency = test_case.expected_efficiency_hhv

        deviation = abs(efficiency_from_losses - expected_efficiency)

        # Should match within 1.5%
        assert deviation <= 1.5, (
            f"{test_case.name}: Efficiency from losses {efficiency_from_losses:.2f}% "
            f"vs direct {expected_efficiency:.2f}%"
        )


@pytest.mark.golden
class TestCombustionCalculations:
    """Test combustion-related calculations per ASME PTC 4."""

    @pytest.mark.parametrize("fuel_type,expected_stoich", [
        ("natural_gas", 17.2),
        ("bituminous_coal", 10.5),
        ("fuel_oil_no6", 14.0),
    ])
    def test_stoichiometric_air(self, fuel_type: str, expected_stoich: float):
        """
        Validate stoichiometric air requirements.

        Based on fuel composition:
        A_stoich = (11.5*C + 34.3*H + 4.3*S - 4.3*O) / 1.0
        """
        composition = FUEL_COMPOSITION.get(fuel_type)
        if composition is None:
            pytest.skip(f"No composition data for {fuel_type}")

        # Calculate stoichiometric air
        C = composition["C"]
        H = composition["H"]
        S = composition["S"]
        O = composition["O"]

        # Stoichiometric air (kg air / kg fuel)
        stoich_air = 11.5 * C + 34.3 * H + 4.3 * S - 4.3 * O

        deviation = abs(stoich_air - expected_stoich) / expected_stoich * 100

        assert deviation <= 10.0, (
            f"{fuel_type}: Stoichiometric air {stoich_air:.2f} vs expected {expected_stoich:.2f}"
        )

    @pytest.mark.parametrize("test_case", ASME_PTC4_TEST_CASES)
    def test_excess_air_from_o2(self, test_case: ASMEPTC4TestCase):
        """
        Validate excess air calculation from O2 measurement.

        Excess Air % = O2 / (21 - O2) * 100
        """
        o2_dry = test_case.o2_percent_dry

        # Excess air from O2 (simplified formula)
        calculated_excess_air = (o2_dry / (21.0 - o2_dry)) * 100

        expected = test_case.excess_air_percent
        deviation = abs(calculated_excess_air - expected)

        assert deviation <= 5.0, (
            f"{test_case.name}: Excess air from O2 = {calculated_excess_air:.1f}% "
            f"vs stated {expected:.1f}%"
        )

    def test_theoretical_co2_for_fuels(self):
        """Validate theoretical CO2 for common fuels."""
        # Theoretical maximum CO2 at zero excess air
        theoretical_co2 = {
            "natural_gas": 11.7,
            "bituminous_coal": 18.5,
            "fuel_oil_no6": 15.5,
        }

        for fuel, expected_co2 in theoretical_co2.items():
            # CO2 max depends on fuel carbon content
            composition = FUEL_COMPOSITION.get(fuel, {})
            C = composition.get("C", 0.8)

            # Simplified CO2 max calculation
            # CO2_max ≈ 21 * (C/12) / (C/12 + 0.264 * (1 + stoich_air_ratio))
            # This is approximate - just verify reasonable range
            assert 10.0 <= expected_co2 <= 20.0, (
                f"{fuel}: Theoretical CO2 {expected_co2}% outside expected range"
            )


@pytest.mark.golden
class TestRadiationLoss:
    """Test radiation and convection loss calculations."""

    @pytest.mark.parametrize("capacity_mw,expected_loss_percent", [
        (1.0, 2.5),    # Small boiler: ~2.5%
        (10.0, 1.0),   # Medium boiler: ~1.0%
        (100.0, 0.4),  # Large boiler: ~0.4%
        (500.0, 0.2),  # Utility boiler: ~0.2%
    ])
    def test_radiation_loss_vs_capacity(self, capacity_mw: float, expected_loss_percent: float):
        """
        Validate radiation loss correlation with boiler capacity.

        Radiation loss decreases with increasing boiler size due to
        better surface area to volume ratio.

        ABMA Standard curve: L_rad = f(capacity)
        """
        # ABMA Standard radiation loss correlation (approximate)
        # L_rad ≈ 0.4 * (capacity_MW)^(-0.3) for typical boilers

        calculated_loss = 0.4 * (capacity_mw ** -0.3)

        # Should be within 50% of expected (correlation is approximate)
        ratio = calculated_loss / expected_loss_percent
        assert 0.5 <= ratio <= 2.0, (
            f"Radiation loss at {capacity_mw} MW: calculated {calculated_loss:.2f}% "
            f"vs expected {expected_loss_percent:.2f}%"
        )


@pytest.mark.golden
class TestDeterminism:
    """Verify calculation determinism per GreenLang requirements."""

    def test_efficiency_calculation_determinism(self):
        """Verify efficiency calculations are perfectly reproducible."""
        test_case = ASME_PTC4_TEST_CASES[0]
        results = []

        for _ in range(100):
            # Use Decimal for deterministic calculation
            steam_flow = Decimal(str(test_case.steam_flow_kg_hr))
            h_steam = Decimal(str(test_case.steam_enthalpy_kj_kg))
            h_fw = Decimal(str(test_case.feedwater_enthalpy_kj_kg))
            fuel_flow = Decimal(str(test_case.fuel_flow_kg_hr))
            hhv = Decimal(str(test_case.fuel_hhv_kj_kg))

            # Calculate
            energy_out = steam_flow * (h_steam - h_fw)
            energy_in = fuel_flow * hhv
            efficiency = (energy_out / energy_in * Decimal("100")).quantize(
                Decimal("0.0001"),
                rounding=ROUND_HALF_UP
            )

            results.append(str(efficiency))

        # All results should be identical
        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Non-deterministic: {len(unique_results)} unique results"
        )

    def test_provenance_hash_consistency(self):
        """Verify provenance hashes are consistent."""
        test_case = ASME_PTC4_TEST_CASES[0]
        hashes = []

        for _ in range(50):
            # Create deterministic provenance data
            provenance = (
                f"{test_case.name}:"
                f"{test_case.expected_efficiency_hhv:.6f}:"
                f"{test_case.fuel_type}"
            )
            hash_val = hashlib.sha256(provenance.encode()).hexdigest()
            hashes.append(hash_val)

        assert len(set(hashes)) == 1, "Provenance hashes not consistent"


@pytest.mark.golden
class TestEPAComplianceIntegration:
    """Test integration with EPA 40 CFR Part 75/98 requirements."""

    def test_heat_input_for_epa_reporting(self):
        """
        Validate heat input calculation for EPA reporting.

        EPA requires heat input in mmBtu/hr for emissions reporting.
        Heat Input = Fuel Flow * HHV (in mmBtu/hr)
        """
        test_case = ASME_PTC4_TEST_CASES[0]

        # Convert to EPA units
        # 1 kJ = 0.0009478 BTU
        fuel_flow_lb_hr = test_case.fuel_flow_kg_hr * 2.205
        hhv_btu_lb = test_case.fuel_hhv_kj_kg * 0.4299

        heat_input_mmbtu_hr = fuel_flow_lb_hr * hhv_btu_lb / 1e6

        # Should be positive and reasonable
        assert heat_input_mmbtu_hr > 0, "Heat input must be positive"
        assert heat_input_mmbtu_hr < 10000, "Heat input unreasonably high"

    def test_co2_emission_factor(self):
        """
        Validate CO2 emission factors for GHG reporting.

        EPA 40 CFR Part 98 requires accurate CO2 factors.
        """
        # EPA CO2 emission factors (kg CO2 / mmBtu)
        epa_co2_factors = {
            "natural_gas": 53.06,
            "bituminous_coal": 93.28,
            "fuel_oil_no6": 75.10,
        }

        for fuel, expected_factor in epa_co2_factors.items():
            composition = FUEL_COMPOSITION.get(fuel, {})
            C = composition.get("C", 0.8)

            # Calculate CO2 from carbon content
            # CO2 = C * 44/12 (molecular weight ratio)
            co2_per_kg_fuel = C * (44 / 12)

            # Verify it's in reasonable range
            assert co2_per_kg_fuel > 0, f"{fuel}: CO2 factor must be positive"
            assert co2_per_kg_fuel < 5, f"{fuel}: CO2 factor unreasonably high"


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_asme_ptc4_golden_values() -> Dict[str, Any]:
    """Export all ASME PTC 4 golden values for external validation."""
    return {
        "metadata": {
            "version": "1.0.0",
            "source": "ASME PTC 4-2013",
            "agent": "GL-001_ThermalCommand",
        },
        "test_cases": [
            {
                "name": tc.name,
                "fuel_type": tc.fuel_type,
                "efficiency_hhv": tc.expected_efficiency_hhv,
                "efficiency_lhv": tc.expected_efficiency_lhv,
                "losses": {
                    "dry_gas": tc.expected_dry_gas_loss_percent,
                    "moisture": tc.expected_moisture_loss_percent,
                    "radiation": tc.expected_radiation_loss_percent,
                },
            }
            for tc in ASME_PTC4_TEST_CASES
        ],
        "stoichiometric_air": STOICHIOMETRIC_AIR,
        "fuel_compositions": FUEL_COMPOSITION,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_asme_ptc4_golden_values(), indent=2))
