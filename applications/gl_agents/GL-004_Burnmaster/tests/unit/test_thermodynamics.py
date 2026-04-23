"""
Unit Tests for GL-004 BURNMASTER Thermodynamics Module

Comprehensive test coverage for thermodynamic property calculations:
- Shomate equation heat capacity (Cp)
- Enthalpy calculations
- Entropy calculations
- Flue gas mixture properties
- Stack loss calculations
- Efficiency calculations (direct and indirect methods)
- Heat balance calculations
- NIST-JANAF reference data validation (golden tests)

Reference Standards:
- NIST-JANAF Thermochemical Tables (5th Edition)
- ASME PTC 4.1: Fired Steam Generators
- ISO 13443: Natural gas - Standard reference conditions

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import pytest
import math
import sys
from pathlib import Path
from typing import Dict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from combustion.thermodynamics import (
    compute_cp_shomate,
    compute_enthalpy_shomate,
    compute_entropy_shomate,
    compute_flue_gas_enthalpy,
    compute_flue_gas_cp,
    compute_stack_loss,
    compute_radiation_loss,
    compute_unburned_loss,
    compute_efficiency_indirect,
    compute_efficiency_direct,
    compute_heat_rate,
    compute_fuel_intensity,
    compute_heat_balance,
    GasProperties,
    HeatBalanceResult,
    EfficiencyResult,
    EfficiencyMethod,
    SHOMATE_COEFFICIENTS,
    MOLECULAR_WEIGHTS,
    HEATING_VALUES,
    R_UNIVERSAL,
)


# =============================================================================
# NIST-JANAF Golden Reference Data
# Source: NIST-JANAF Thermochemical Tables, 5th Edition (1998)
# =============================================================================

NIST_JANAF_CP_REFERENCE: Dict[str, Dict[float, float]] = {
    "N2": {300: 29.12, 500: 29.58, 1000: 32.70, 1500: 34.82, 2000: 35.97},
    "O2": {300: 29.38, 500: 31.09, 1000: 34.88, 1500: 36.56, 2000: 37.12},
    "CO2": {300: 37.22, 500: 44.63, 800: 51.44, 1000: 54.30, 1200: 56.36},
    "H2O": {500: 35.22, 1000: 41.27, 1500: 46.01, 1700: 47.45},
    "CO": {300: 29.14, 500: 29.79, 1000: 33.18, 1300: 34.56},
    "CH4": {300: 35.69, 500: 46.34, 800: 63.95, 1000: 73.60, 1300: 86.07},
}


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def typical_flue_gas_composition():
    """Typical flue gas composition from natural gas combustion."""
    return {
        "co2_mol_frac": 0.10, "h2o_mol_frac": 0.18,
        "n2_mol_frac": 0.70, "o2_mol_frac": 0.02,
    }


@pytest.fixture
def standard_combustion_conditions():
    """Standard operating conditions for a natural gas burner."""
    return {
        "stack_temp_c": 180.0, "ambient_temp_c": 25.0, "excess_o2_pct": 3.0,
        "co_ppm": 50.0, "furnace_rating_mw": 20.0, "load_fraction": 0.85,
        "fuel_type": "natural_gas",
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestShomateCoefficients:
    """Tests for Shomate equation coefficient database."""

    def test_all_species_have_required_coefficients(self):
        """Verify all species have complete Shomate coefficients."""
        required = ["A", "B", "C", "D", "E", "F", "G", "H", "T_min", "T_max"]
        for species, coefs in SHOMATE_COEFFICIENTS.items():
            for key in required:
                assert key in coefs, f"{species} missing {key}"

    @pytest.mark.parametrize("species", list(SHOMATE_COEFFICIENTS.keys()))
    def test_temperature_ranges_valid(self, species):
        """Verify temperature ranges are physically valid."""
        coefs = SHOMATE_COEFFICIENTS[species]
        assert coefs["T_min"] >= 200
        assert coefs["T_max"] <= 7000
        assert coefs["T_min"] < coefs["T_max"]

    @pytest.mark.parametrize("species", list(SHOMATE_COEFFICIENTS.keys()))
    def test_coefficients_produce_positive_cp(self, species):
        """Verify Cp is positive at standard temperature."""
        cp = compute_cp_shomate(species, 298.15)
        assert cp > 0


class TestComputeCpShomate:
    """Tests for Shomate equation Cp calculations."""

    @pytest.mark.parametrize("species,temp,expected_cp,tolerance", [
        ("N2", 300, 29.12, 0.5), ("N2", 500, 29.58, 0.5), ("N2", 1000, 32.70, 1.0),
        ("O2", 300, 29.38, 0.5), ("O2", 500, 31.09, 0.5), ("O2", 1000, 34.88, 1.0),
        ("CO2", 300, 37.22, 0.5), ("CO2", 500, 44.63, 1.0), ("CO2", 1000, 54.30, 1.5),
        ("CH4", 300, 35.69, 1.0), ("CH4", 500, 46.34, 2.0),
    ])
    def test_cp_against_nist_janaf(self, species, temp, expected_cp, tolerance):
        """Golden test: Validate Cp against NIST-JANAF reference data."""
        calculated_cp = compute_cp_shomate(species, temp)
        assert abs(calculated_cp - expected_cp) < tolerance

    def test_unknown_species_raises(self):
        """Test unknown species raises ValueError."""
        with pytest.raises(ValueError, match="Unknown species"):
            compute_cp_shomate("XenonTrifluoride", 300)

    def test_cp_determinism(self):
        """Test Cp calculation is deterministic."""
        assert compute_cp_shomate("N2", 500) == compute_cp_shomate("N2", 500)

    def test_cp_increases_with_temperature_for_polyatomics(self):
        """Test Cp increases with temperature for polyatomic molecules."""
        cp_300 = compute_cp_shomate("CO2", 300)
        cp_500 = compute_cp_shomate("CO2", 500)
        cp_1000 = compute_cp_shomate("CO2", 1000)
        assert cp_300 < cp_500 < cp_1000

    @pytest.mark.parametrize("species", ["N2", "O2", "CO2", "H2O", "CO", "CH4"])
    def test_cp_positive_at_high_temperature(self, species):
        """Test Cp remains positive at high temperatures."""
        assert compute_cp_shomate(species, 2000) > 0


class TestComputeEnthalpyShomate:
    """Tests for Shomate equation enthalpy calculations."""

    @pytest.mark.parametrize("species,temp,expected_h,tolerance", [
        ("N2", 500, 5.912, 0.3), ("N2", 1000, 21.463, 1.0),
        ("O2", 500, 6.086, 0.3), ("O2", 1000, 22.703, 1.0),
        ("CO2", 500, 8.305, 0.5), ("CO2", 1000, 33.397, 1.5),
    ])
    def test_enthalpy_against_nist_janaf(self, species, temp, expected_h, tolerance):
        """Golden test: Validate enthalpy against NIST-JANAF reference data."""
        h_temp = compute_enthalpy_shomate(species, temp)
        h_ref = compute_enthalpy_shomate(species, 298.15)
        delta_h = h_temp - h_ref
        assert abs(delta_h - expected_h) < tolerance

    def test_enthalpy_increases_with_temperature(self):
        """Test enthalpy increases monotonically with temperature."""
        h_300 = compute_enthalpy_shomate("N2", 300)
        h_500 = compute_enthalpy_shomate("N2", 500)
        h_1000 = compute_enthalpy_shomate("N2", 1000)
        assert h_300 < h_500 < h_1000

    def test_unknown_species_raises(self):
        """Test unknown species raises ValueError."""
        with pytest.raises(ValueError, match="Unknown species"):
            compute_enthalpy_shomate("UnknownGas", 300)


class TestComputeEntropyShomate:
    """Tests for Shomate equation entropy calculations."""

    def test_entropy_positive_at_standard_conditions(self):
        """Test entropy is positive at standard conditions."""
        for species in SHOMATE_COEFFICIENTS:
            assert compute_entropy_shomate(species, 298.15) > 0

    def test_entropy_increases_with_temperature(self):
        """Test entropy increases with temperature (Second Law)."""
        s_300 = compute_entropy_shomate("N2", 300)
        s_500 = compute_entropy_shomate("N2", 500)
        s_1000 = compute_entropy_shomate("N2", 1000)
        assert s_300 < s_500 < s_1000

    @pytest.mark.parametrize("species", ["N2", "O2", "CO2", "H2O"])
    def test_entropy_order_of_magnitude(self, species):
        """Test entropy is in expected range (150-300 J/mol/K for gases)."""
        s = compute_entropy_shomate(species, 298.15)
        assert 100 < s < 400


class TestFlueGasProperties:
    """Tests for flue gas mixture property calculations."""

    def test_flue_gas_enthalpy_positive_above_reference(self, typical_flue_gas_composition):
        """Test flue gas enthalpy is positive above reference temperature."""
        h = compute_flue_gas_enthalpy(temp_k=500, **typical_flue_gas_composition, reference_temp_k=298.15)
        assert h > 0

    def test_flue_gas_enthalpy_zero_at_reference(self, typical_flue_gas_composition):
        """Test flue gas enthalpy is zero at reference temperature."""
        h = compute_flue_gas_enthalpy(temp_k=298.15, **typical_flue_gas_composition, reference_temp_k=298.15)
        assert abs(h) < 0.001

    def test_flue_gas_cp_positive(self, typical_flue_gas_composition):
        """Test flue gas Cp is positive."""
        cp = compute_flue_gas_cp(temp_k=500, **typical_flue_gas_composition)
        assert cp > 0


class TestStackLoss:
    """Tests for stack (dry flue gas) heat loss calculations."""

    def test_stack_loss_increases_with_temperature_difference(self):
        """Test stack loss increases with temperature difference."""
        loss_low = compute_stack_loss(stack_temp_c=150, ambient_temp_c=25, excess_o2_pct=3.0)
        loss_high = compute_stack_loss(stack_temp_c=250, ambient_temp_c=25, excess_o2_pct=3.0)
        assert loss_high > loss_low

    def test_stack_loss_increases_with_excess_o2(self):
        """Test stack loss increases with excess O2."""
        loss_low_o2 = compute_stack_loss(stack_temp_c=180, ambient_temp_c=25, excess_o2_pct=2.0)
        loss_high_o2 = compute_stack_loss(stack_temp_c=180, ambient_temp_c=25, excess_o2_pct=6.0)
        assert loss_high_o2 > loss_low_o2

    def test_stack_loss_typical_range(self):
        """Test stack loss is in typical range (3-10% for natural gas)."""
        loss = compute_stack_loss(stack_temp_c=180, ambient_temp_c=25, excess_o2_pct=3.0, fuel_type="natural_gas")
        assert 3.0 < loss < 12.0

    @pytest.mark.parametrize("fuel_type", ["natural_gas", "propane", "fuel_oil_2", "fuel_oil_6", "coal_bituminous"])
    def test_stack_loss_positive_for_all_fuels(self, fuel_type):
        """Test stack loss is positive for all fuel types."""
        loss = compute_stack_loss(stack_temp_c=180, ambient_temp_c=25, excess_o2_pct=3.0, fuel_type=fuel_type)
        assert loss > 0


class TestRadiationLoss:
    """Tests for radiation and convection loss calculations."""

    def test_radiation_loss_positive_at_load(self):
        """Test radiation loss is positive when operating."""
        assert compute_radiation_loss(furnace_rating_mw=20.0, load_fraction=0.8) > 0

    def test_radiation_loss_zero_at_zero_load(self):
        """Test radiation loss is zero at zero load."""
        assert compute_radiation_loss(furnace_rating_mw=20.0, load_fraction=0.0) == 0.0

    def test_radiation_loss_increases_at_low_load(self):
        """Test radiation loss (as % of input) increases at low load."""
        loss_full = compute_radiation_loss(furnace_rating_mw=20.0, load_fraction=1.0)
        loss_half = compute_radiation_loss(furnace_rating_mw=20.0, load_fraction=0.5)
        loss_quarter = compute_radiation_loss(furnace_rating_mw=20.0, load_fraction=0.25)
        assert loss_full < loss_half < loss_quarter

    def test_radiation_loss_decreases_with_size(self):
        """Test larger furnaces have lower % radiation loss."""
        loss_small = compute_radiation_loss(furnace_rating_mw=5.0, load_fraction=1.0)
        loss_large = compute_radiation_loss(furnace_rating_mw=100.0, load_fraction=1.0)
        assert loss_small > loss_large


class TestUnburnedLoss:
    """Tests for unburned combustibles loss calculations."""

    def test_unburned_loss_increases_with_co(self):
        """Test unburned loss increases with CO concentration."""
        assert compute_unburned_loss(co_ppm=500) > compute_unburned_loss(co_ppm=50)

    def test_unburned_loss_zero_at_zero_co(self):
        """Test unburned loss is minimal at zero CO."""
        assert compute_unburned_loss(co_ppm=0, combustible_in_ash_pct=0) == 0.0


class TestEfficiencyIndirect:
    """Tests for indirect (heat loss) efficiency calculation."""

    def test_efficiency_typical_range(self, standard_combustion_conditions):
        """Test efficiency is in typical range for natural gas."""
        result = compute_efficiency_indirect(**standard_combustion_conditions)
        assert 80 < result.gross_efficiency_pct < 95

    def test_efficiency_decreases_with_stack_temp(self):
        """Test efficiency decreases with higher stack temperature."""
        result_low = compute_efficiency_indirect(stack_temp_c=150, ambient_temp_c=25, excess_o2_pct=3.0, co_ppm=50, furnace_rating_mw=20, load_fraction=0.8)
        result_high = compute_efficiency_indirect(stack_temp_c=250, ambient_temp_c=25, excess_o2_pct=3.0, co_ppm=50, furnace_rating_mw=20, load_fraction=0.8)
        assert result_low.gross_efficiency_pct > result_high.gross_efficiency_pct

    def test_net_efficiency_higher_than_gross(self, standard_combustion_conditions):
        """Test net efficiency (LHV) is higher than gross (HHV)."""
        result = compute_efficiency_indirect(**standard_combustion_conditions)
        assert result.net_efficiency_pct >= result.gross_efficiency_pct


class TestEfficiencyDirect:
    """Tests for direct (input-output) efficiency calculation."""

    def test_efficiency_correct_calculation(self):
        """Test direct efficiency calculation is correct."""
        efficiency = compute_efficiency_direct(useful_output_mw=8.5, fuel_flow_kg_s=0.2, fuel_type="natural_gas", use_lhv=False)
        expected = 8.5 / (0.2 * 55.5) * 100
        assert abs(efficiency - expected) < 0.1

    def test_efficiency_capped_at_100(self):
        """Test efficiency cannot exceed 100%."""
        assert compute_efficiency_direct(useful_output_mw=100.0, fuel_flow_kg_s=0.1, fuel_type="natural_gas") <= 100.0

    def test_efficiency_zero_at_zero_fuel(self):
        """Test efficiency is zero with zero fuel flow."""
        assert compute_efficiency_direct(useful_output_mw=10.0, fuel_flow_kg_s=0.0, fuel_type="natural_gas") == 0.0


class TestHeatRate:
    """Tests for heat rate calculations."""

    def test_heat_rate_positive(self):
        """Test heat rate is positive."""
        assert compute_heat_rate(power_output_mw=10.0, fuel_flow_kg_s=0.3, fuel_type="natural_gas") > 0

    def test_heat_rate_infinity_at_zero_output(self):
        """Test heat rate is infinity at zero output."""
        assert math.isinf(compute_heat_rate(power_output_mw=0.0, fuel_flow_kg_s=0.3, fuel_type="natural_gas"))


class TestHeatBalance:
    """Tests for complete heat balance calculations."""

    def test_heat_balance_closure(self):
        """Test heat balance closes (energy conservation)."""
        result = compute_heat_balance(fuel_flow_kg_s=0.3, useful_output_mw=12.0, stack_temp_c=180, ambient_temp_c=25, excess_o2_pct=3.0, furnace_rating_mw=20, co_ppm=50, fuel_type="natural_gas")
        total_output = result.useful_output_mw + result.stack_loss_mw + result.radiation_loss_mw + result.unburned_loss_mw + result.moisture_loss_mw + result.other_losses_mw
        assert abs(result.heat_input_mw - total_output) / result.heat_input_mw < 0.01

    def test_heat_balance_zero_fuel(self):
        """Test heat balance handles zero fuel flow."""
        result = compute_heat_balance(fuel_flow_kg_s=0.0, useful_output_mw=0.0, stack_temp_c=180, ambient_temp_c=25, excess_o2_pct=3.0, furnace_rating_mw=20)
        assert result.heat_input_mw == 0.0


class TestMolecularWeights:
    """Tests for molecular weight database."""

    @pytest.mark.parametrize("species,expected_mw,tolerance", [
        ("N2", 28.014, 0.001), ("O2", 31.998, 0.001), ("CO2", 44.01, 0.01),
        ("H2O", 18.015, 0.001), ("CO", 28.01, 0.01), ("CH4", 16.043, 0.001),
    ])
    def test_molecular_weights_correct(self, species, expected_mw, tolerance):
        """Test molecular weights match NIST reference values."""
        assert abs(MOLECULAR_WEIGHTS[species] - expected_mw) < tolerance


class TestHeatingValues:
    """Tests for heating value database."""

    def test_hhv_greater_than_lhv(self):
        """Test HHV is always greater than LHV."""
        for fuel, values in HEATING_VALUES.items():
            assert values["HHV"] > values["LHV"]


class TestDeterminism:
    """Tests for calculation determinism (zero-hallucination requirement)."""

    def test_cp_deterministic_across_calls(self):
        """Test Cp is deterministic across multiple calls."""
        results = [compute_cp_shomate("CO2", 500) for _ in range(100)]
        assert all(r == results[0] for r in results)


class TestASMEPTC4Golden:
    """Golden tests based on ASME PTC 4 example calculations."""

    def test_asme_ptc4_stack_loss_example(self):
        """Golden test: ASME PTC 4 stack loss calculation."""
        stack_loss = compute_stack_loss(stack_temp_c=177, ambient_temp_c=27, excess_o2_pct=3.0, fuel_type="natural_gas")
        assert 3.0 < stack_loss < 10.0

    def test_asme_ptc4_efficiency_example(self):
        """Golden test: ASME PTC 4 efficiency calculation."""
        result = compute_efficiency_indirect(stack_temp_c=177, ambient_temp_c=27, excess_o2_pct=3.0, co_ppm=50, furnace_rating_mw=50, load_fraction=0.9, fuel_type="natural_gas")
        assert 75 < result.gross_efficiency_pct < 95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
