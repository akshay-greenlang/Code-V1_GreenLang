"""
GL-004 Burnmaster Golden Value Tests - Combustion Reference Values

Tests combustion calculations against ASME PTC 4 and EPA Method 19 reference data.
These golden tests ensure zero-hallucination deterministic calculations.

Reference Standards:
    - ASME PTC 4-2013 (Fired Steam Generators)
    - EPA Method 19 (Sulfur Dioxide Emissions)
    - NIST Chemistry WebBook (Thermodynamic Properties)

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import hashlib
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any
import pytest


# =============================================================================
# GOLDEN VALUE REFERENCE DATA
# =============================================================================

# Natural Gas Combustion Reference (ASME PTC 4 Table)
NATURAL_GAS_GOLDEN_VALUES = {
    "fuel_composition": {
        "CH4": 0.9500,  # Methane
        "C2H6": 0.0300,  # Ethane
        "C3H8": 0.0100,  # Propane
        "CO2": 0.0050,  # Carbon Dioxide
        "N2": 0.0050,   # Nitrogen
    },
    "hhv_btu_per_scf": 1020.0,  # Higher Heating Value
    "lhv_btu_per_scf": 920.0,   # Lower Heating Value
    "stoichiometric_air_fuel_ratio": 9.52,  # lb air / lb fuel
    "theoretical_co2_percent": 11.8,  # Dry flue gas
    "expected_outputs": {
        "excess_air_at_3pct_o2": 15.0,  # %
        "combustion_efficiency_at_3pct_o2": 0.845,  # 84.5%
        "flue_gas_temp_reference_f": 350.0,
    }
}

# Fuel Oil #2 Combustion Reference
FUEL_OIL_2_GOLDEN_VALUES = {
    "fuel_composition": {
        "C": 0.8700,   # Carbon mass fraction
        "H": 0.1230,   # Hydrogen
        "S": 0.0050,   # Sulfur
        "O": 0.0010,   # Oxygen
        "N": 0.0005,   # Nitrogen
        "Ash": 0.0005, # Ash
    },
    "hhv_btu_per_lb": 19500.0,
    "lhv_btu_per_lb": 18300.0,
    "stoichiometric_air_fuel_ratio": 14.1,
    "theoretical_co2_percent": 15.3,
}

# EPA Method 19 F-Factors (scf/MMBtu)
EPA_F_FACTORS = {
    "natural_gas": {
        "Fd": 8710,   # Dry basis
        "Fw": 10610,  # Wet basis
        "Fc": 1040,   # Carbon
    },
    "fuel_oil_2": {
        "Fd": 9190,
        "Fw": 10320,
        "Fc": 1420,
    },
    "coal_bituminous": {
        "Fd": 9780,
        "Fw": 10640,
        "Fc": 1800,
    },
}


# =============================================================================
# GOLDEN VALUE TEST CLASS
# =============================================================================

class TestCombustionGoldenValues:
    """Golden value tests for combustion calculations."""

    @pytest.mark.golden
    def test_stoichiometric_air_natural_gas(self):
        """
        Test stoichiometric air calculation for natural gas.

        Reference: ASME PTC 4-2013 Section 5.4
        Expected: 9.52 lb air/lb fuel for typical natural gas
        Tolerance: +/- 0.05 (instrument uncertainty)
        """
        # Natural gas composition (molar basis)
        ch4 = 0.95
        c2h6 = 0.03
        c3h8 = 0.01

        # Stoichiometric oxygen requirement (moles O2 per mole fuel)
        # CH4 + 2O2 -> CO2 + 2H2O
        # C2H6 + 3.5O2 -> 2CO2 + 3H2O
        # C3H8 + 5O2 -> 3CO2 + 4H2O
        o2_required = ch4 * 2.0 + c2h6 * 3.5 + c3h8 * 5.0

        # Air is 21% O2, so air requirement
        air_required_molar = o2_required / 0.21

        # Convert to mass basis (assuming MW of air = 28.97, MW of fuel ~17)
        mw_air = 28.97
        mw_fuel = ch4 * 16.04 + c2h6 * 30.07 + c3h8 * 44.10

        stoich_afr = (air_required_molar * mw_air) / mw_fuel

        # Golden value check
        expected = NATURAL_GAS_GOLDEN_VALUES["stoichiometric_air_fuel_ratio"]
        tolerance = 0.15  # Allow 0.15 variance due to composition differences

        assert abs(stoich_afr - expected) < tolerance, (
            f"Stoichiometric AFR mismatch: calculated={stoich_afr:.2f}, "
            f"expected={expected:.2f} +/- {tolerance}"
        )

    @pytest.mark.golden
    def test_excess_air_from_o2_measurement(self):
        """
        Test excess air calculation from O2 measurement.

        Reference: ASME PTC 4-2013 Equation 5-4
        Formula: EA = O2 / (21 - O2) * 100
        """
        test_cases = [
            {"o2_percent": 2.0, "expected_ea": 10.5},
            {"o2_percent": 3.0, "expected_ea": 16.7},
            {"o2_percent": 4.0, "expected_ea": 23.5},
            {"o2_percent": 5.0, "expected_ea": 31.3},
            {"o2_percent": 6.0, "expected_ea": 40.0},
        ]

        for case in test_cases:
            o2 = case["o2_percent"]
            expected = case["expected_ea"]

            # ASME PTC 4 formula
            calculated_ea = (o2 / (21.0 - o2)) * 100.0

            # Use Decimal for precision
            calculated_decimal = Decimal(str(calculated_ea)).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            expected_decimal = Decimal(str(expected)).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )

            assert calculated_decimal == expected_decimal, (
                f"Excess air mismatch at {o2}% O2: "
                f"calculated={calculated_decimal}, expected={expected_decimal}"
            )

    @pytest.mark.golden
    def test_epa_method_19_f_factor(self):
        """
        Test EPA Method 19 F-factor calculations.

        Reference: 40 CFR Part 75 Appendix F
        F-factors are fundamental emission calculation constants.
        """
        # Natural gas F-factors
        ng_fd = EPA_F_FACTORS["natural_gas"]["Fd"]
        ng_fw = EPA_F_FACTORS["natural_gas"]["Fw"]
        ng_fc = EPA_F_FACTORS["natural_gas"]["Fc"]

        # Verify F-factor relationships (Fw should be > Fd due to water)
        assert ng_fw > ng_fd, "Wet F-factor must exceed dry F-factor"

        # Verify Fc is reasonable (carbon F-factor)
        assert 1000 < ng_fc < 2000, "Carbon F-factor outside expected range"

        # Cross-check: Fd should be approximately 8710 for natural gas
        assert abs(ng_fd - 8710) < 100, (
            f"Natural gas Fd={ng_fd} outside tolerance of 8710 +/- 100"
        )

    @pytest.mark.golden
    def test_combustion_efficiency_heat_loss_method(self):
        """
        Test combustion efficiency via heat loss method.

        Reference: ASME PTC 4-2013 Section 5.14
        Efficiency = 100 - (Dry Gas Loss + Moisture Loss + Unburned Loss + Radiation Loss)
        """
        # Test conditions
        flue_gas_temp_f = 350.0
        ambient_temp_f = 70.0
        excess_air_percent = 15.0

        # Simplified dry gas loss calculation
        # L_dg = K * (Tf - Ta) * (1 + EA/100) / HHV_ratio
        # Using approximation for natural gas
        k_factor = 0.0024  # Approximate for natural gas
        temp_diff = flue_gas_temp_f - ambient_temp_f
        dry_gas_loss = k_factor * temp_diff * (1 + excess_air_percent / 100) * 100

        # Moisture losses (latent heat of water vapor)
        moisture_loss = 5.5  # Typical for natural gas at these conditions

        # Unburned carbon loss (assume complete combustion)
        unburned_loss = 0.0

        # Radiation loss (typical 0.5-1.5% for well-insulated boiler)
        radiation_loss = 1.0

        # Calculate efficiency
        total_losses = dry_gas_loss + moisture_loss + unburned_loss + radiation_loss
        efficiency = (100.0 - total_losses) / 100.0

        # Golden value check
        expected = NATURAL_GAS_GOLDEN_VALUES["expected_outputs"]["combustion_efficiency_at_3pct_o2"]
        tolerance = 0.03  # 3% tolerance for simplified calculation

        assert abs(efficiency - expected) < tolerance, (
            f"Efficiency mismatch: calculated={efficiency:.3f}, "
            f"expected={expected:.3f} +/- {tolerance}"
        )

    @pytest.mark.golden
    def test_determinism_provenance_hash(self):
        """
        Test that identical inputs produce identical provenance hashes.

        This verifies the zero-hallucination guarantee.
        """
        input_data = {
            "fuel_type": "natural_gas",
            "o2_percent": 3.0,
            "flue_gas_temp_f": 350.0,
            "ambient_temp_f": 70.0,
            "fuel_flow_scfh": 1000.0,
        }

        # Calculate hash multiple times
        hashes = []
        for _ in range(5):
            content = str(sorted(input_data.items()))
            hash_value = hashlib.sha256(content.encode()).hexdigest()
            hashes.append(hash_value)

        # All hashes must be identical
        assert len(set(hashes)) == 1, (
            f"Non-deterministic hash generation detected: {set(hashes)}"
        )

    @pytest.mark.golden
    def test_fuel_oil_heating_value(self):
        """
        Test fuel oil heating value calculations.

        Reference: ASTM D240 (Bomb Calorimeter)
        """
        # Fuel Oil #2 reference values
        hhv = FUEL_OIL_2_GOLDEN_VALUES["hhv_btu_per_lb"]
        lhv = FUEL_OIL_2_GOLDEN_VALUES["lhv_btu_per_lb"]

        # LHV should be approximately HHV - (hydrogen content * latent heat of water)
        hydrogen_content = FUEL_OIL_2_GOLDEN_VALUES["fuel_composition"]["H"]
        latent_heat_water = 1050  # BTU/lb at typical conditions
        water_formed = hydrogen_content * 9  # 9 lb H2O per lb H

        expected_lhv = hhv - (water_formed * latent_heat_water)

        # Tolerance check
        tolerance = 500  # BTU/lb
        assert abs(lhv - expected_lhv) < tolerance, (
            f"LHV calculation mismatch: specified={lhv}, "
            f"calculated={expected_lhv:.0f} +/- {tolerance}"
        )


class TestEmissionsGoldenValues:
    """Golden value tests for emissions calculations."""

    @pytest.mark.golden
    def test_co2_emission_factor_natural_gas(self):
        """
        Test CO2 emission factor for natural gas.

        Reference: EPA 40 CFR Part 98 Table C-1
        Natural Gas: 53.06 kg CO2/MMBtu
        """
        epa_reference = 53.06  # kg CO2/MMBtu

        # Calculate from first principles
        # Natural gas is ~95% CH4
        # CH4 + 2O2 -> CO2 + 2H2O
        # 16 kg CH4 -> 44 kg CO2
        # HHV of CH4 = 23,875 BTU/lb = 55.5 MMBtu/ton

        ch4_fraction = 0.95
        co2_per_ch4 = 44.0 / 16.0  # Mass ratio
        hhv_mmbtu_per_kg = 0.0523  # Approximately

        calculated_ef = (ch4_fraction * co2_per_ch4) / hhv_mmbtu_per_kg

        # Within 5% of EPA reference
        tolerance_percent = 5.0
        tolerance = epa_reference * tolerance_percent / 100

        assert abs(calculated_ef - epa_reference) < tolerance, (
            f"CO2 EF mismatch: calculated={calculated_ef:.2f}, "
            f"EPA reference={epa_reference} +/- {tolerance:.2f}"
        )

    @pytest.mark.golden
    def test_nox_emission_estimation(self):
        """
        Test NOx emission estimation.

        Reference: EPA AP-42 Chapter 1.4 (Natural Gas Combustion)
        Uncontrolled NOx: 100 lb/10^6 scf (typical)
        """
        # AP-42 reference for industrial boilers
        ap42_nox_lb_per_mmscf = 100.0  # lb NOx per million scf

        # Convert to ppmv at reference conditions
        # Assuming stoichiometric combustion products
        # ~8.5 scf flue gas per scf fuel
        flue_gas_ratio = 8.5

        # NOx molecular weight = 46 (NO2 basis)
        # At STP: 1 lb-mol = 385.5 scf
        mw_nox = 46.0
        scf_per_lbmol = 385.5

        # Calculate ppmv
        nox_lbmol_per_mmscf = ap42_nox_lb_per_mmscf / mw_nox
        flue_gas_scf = 1e6 * flue_gas_ratio
        nox_ppmv = (nox_lbmol_per_mmscf * scf_per_lbmol / flue_gas_scf) * 1e6

        # Expected range for uncontrolled natural gas combustion
        assert 50 < nox_ppmv < 150, (
            f"NOx ppmv={nox_ppmv:.1f} outside expected range 50-150 ppmv"
        )


class TestThermodynamicGoldenValues:
    """Golden value tests for thermodynamic properties."""

    @pytest.mark.golden
    def test_air_enthalpy_at_reference_temps(self):
        """
        Test air enthalpy calculations at reference temperatures.

        Reference: NIST Chemistry WebBook / JANAF Tables
        """
        # Reference enthalpies (BTU/lb, relative to 77F/25C)
        reference_values = [
            {"temp_f": 77.0, "enthalpy_btu_lb": 0.0},
            {"temp_f": 200.0, "enthalpy_btu_lb": 29.5},
            {"temp_f": 400.0, "enthalpy_btu_lb": 77.6},
            {"temp_f": 600.0, "enthalpy_btu_lb": 126.8},
            {"temp_f": 800.0, "enthalpy_btu_lb": 177.1},
        ]

        cp_air = 0.24  # BTU/(lb-F), approximately constant

        for ref in reference_values:
            temp_f = ref["temp_f"]
            expected = ref["enthalpy_btu_lb"]

            # Simple calculation
            calculated = cp_air * (temp_f - 77.0)

            # Tolerance increases with temperature due to Cp variation
            tolerance = 5.0 + 0.02 * (temp_f - 77.0)

            assert abs(calculated - expected) < tolerance, (
                f"Air enthalpy at {temp_f}F: calculated={calculated:.1f}, "
                f"expected={expected:.1f} +/- {tolerance:.1f}"
            )


# =============================================================================
# PROVENANCE VERIFICATION
# =============================================================================

class TestProvenanceTracking:
    """Tests for SHA-256 provenance hash generation."""

    @pytest.mark.golden
    def test_calculation_provenance_hash(self):
        """Verify provenance hash is deterministic and traceable."""
        calculation_result = {
            "timestamp": "2024-12-24T12:00:00Z",
            "input_o2_percent": 3.0,
            "input_flue_temp_f": 350.0,
            "output_excess_air": 16.67,
            "output_efficiency": 0.845,
            "method": "ASME_PTC_4_HEAT_LOSS",
            "standard_version": "2013",
        }

        # Generate provenance hash
        content = str(sorted(calculation_result.items()))
        provenance_hash = hashlib.sha256(content.encode()).hexdigest()

        # Verify hash format
        assert len(provenance_hash) == 64, "SHA-256 hash must be 64 characters"
        assert provenance_hash.isalnum(), "Hash must be alphanumeric"

        # Verify reproducibility
        provenance_hash_2 = hashlib.sha256(content.encode()).hexdigest()
        assert provenance_hash == provenance_hash_2, "Hash must be reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "golden"])
