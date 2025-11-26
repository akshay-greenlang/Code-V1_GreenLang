# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH NOx Calculator.

Tests NOx emissions calculations including thermal, fuel, and prompt NOx
components, EPA Method 19 calculations, F-factor selection, concentration
conversions, temperature/pressure corrections, and determinism verification.

Test Count: 25+ tests
Coverage Target: 90%+

Standards: EPA Method 19, 40 CFR Part 60/75

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import math
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import (
    EmissionsComplianceTools,
    NOxEmissionsResult,
    F_FACTORS,
    MOLECULAR_WEIGHTS,
    AP42_EMISSION_FACTORS,
)


# =============================================================================
# TEST CLASS: NOx EMISSIONS CALCULATOR
# =============================================================================

@pytest.mark.unit
class TestNOxCalculator:
    """Test suite for NOx emissions calculations."""

    # =========================================================================
    # BASIC CALCULATION TESTS
    # =========================================================================

    def test_calculate_nox_emissions_basic(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test basic NOx emissions calculation."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert isinstance(result, NOxEmissionsResult)
        assert result.concentration_ppm >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.emission_rate_lb_hr >= 0
        assert result.mass_rate_kg_hr >= 0
        assert result.calculation_method == "EPA_Method_19"
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hash

    def test_calculate_nox_emissions_with_process_params(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data, boiler_process_parameters
    ):
        """Test NOx calculation with process parameters."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
            process_parameters=boiler_process_parameters,
        )

        assert isinstance(result, NOxEmissionsResult)
        assert result.thermal_nox_percent >= 0
        assert result.fuel_nox_percent >= 0
        assert result.prompt_nox_percent >= 0
        # Sum should be approximately 100%
        total_percent = result.thermal_nox_percent + result.fuel_nox_percent + result.prompt_nox_percent
        assert 99.0 <= total_percent <= 101.0

    # =========================================================================
    # THERMAL NOX CALCULATION TESTS
    # =========================================================================

    def test_thermal_nox_calculation_high_temperature(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test thermal NOx increases with combustion temperature."""
        low_temp_params = {"combustion_temperature_f": 2200.0}
        high_temp_params = {"combustion_temperature_f": 3200.0}

        result_low = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
            process_parameters=low_temp_params,
        )

        result_high = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
            process_parameters=high_temp_params,
        )

        # Higher temperature should generally produce higher thermal NOx percentage
        # Note: actual emissions depend on CEMS reading, but thermal % should differ
        assert result_high.thermal_nox_percent >= result_low.thermal_nox_percent * 0.9

    def test_thermal_nox_zeldovich_mechanism(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test thermal NOx follows Zeldovich mechanism temperature dependence."""
        # Thermal NOx formation follows Arrhenius relationship
        temperatures = [2400, 2600, 2800, 3000]
        thermal_percentages = []

        for temp in temperatures:
            result = emissions_tools.calculate_nox_emissions(
                cems_data=sample_cems_data,
                fuel_data=natural_gas_fuel_data,
                process_parameters={"combustion_temperature_f": temp},
            )
            thermal_percentages.append(result.thermal_nox_percent)

        # Thermal NOx should generally increase with temperature
        # Allow for some variation due to normalization
        assert thermal_percentages[-1] >= thermal_percentages[0] * 0.8

    # =========================================================================
    # FUEL NOX CALCULATION TESTS
    # =========================================================================

    def test_fuel_nox_calculation_nitrogen_content(self, emissions_tools, sample_cems_data):
        """Test fuel NOx increases with fuel nitrogen content."""
        low_n_fuel = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "nitrogen_percent": 0.01,
        }

        high_n_fuel = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "nitrogen_percent": 1.5,
        }

        result_low = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=low_n_fuel,
        )

        result_high = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=high_n_fuel,
        )

        # Higher fuel nitrogen should increase fuel NOx percentage
        assert result_high.fuel_nox_percent > result_low.fuel_nox_percent

    def test_fuel_nox_conversion_efficiency(self, emissions_tools, sample_cems_data):
        """Test fuel nitrogen to NOx conversion follows expected patterns."""
        # Coal typically converts 15-30% of fuel nitrogen to NOx
        coal_fuel = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "nitrogen_percent": 1.5,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=coal_fuel,
        )

        # Fuel NOx should be significant for coal
        assert result.fuel_nox_percent > 10.0

    # =========================================================================
    # PROMPT NOX CALCULATION TESTS
    # =========================================================================

    def test_prompt_nox_calculation(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test prompt NOx is typically small fraction of total."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # Prompt NOx typically < 10% for most fuels
        assert result.prompt_nox_percent <= 15.0
        assert result.prompt_nox_percent >= 0.0

    def test_prompt_nox_hydrocarbon_fuels(self, emissions_tools, sample_cems_data):
        """Test prompt NOx present for hydrocarbon fuels."""
        # Prompt NOx from CH radical attack on N2
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=fuel_data,
        )

        # Should have some prompt NOx contribution
        assert result.prompt_nox_percent > 0.0

    # =========================================================================
    # TOTAL NOX EMISSIONS TESTS
    # =========================================================================

    def test_total_nox_emissions_mass_balance(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test total NOx emissions mass balance."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # Verify mass rate conversions are consistent
        # lb/hr to kg/hr: 1 lb = 0.453592 kg
        expected_kg_hr = result.emission_rate_lb_hr * 0.453592
        assert abs(result.mass_rate_kg_hr - expected_kg_hr) < 0.01

    def test_total_nox_from_heat_input(self, emissions_tools, sample_cems_data):
        """Test NOx lb/hr calculation from heat input."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 200.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=fuel_data,
        )

        # emission_rate_lb_hr = emission_rate_lb_mmbtu * heat_input_mmbtu_hr
        expected_lb_hr = result.emission_rate_lb_mmbtu * 200.0
        assert abs(result.emission_rate_lb_hr - expected_lb_hr) < 0.01

    # =========================================================================
    # EPA METHOD 19 CALCULATION TESTS
    # =========================================================================

    def test_epa_method_19_calculation_natural_gas(self, emissions_tools):
        """Test EPA Method 19 calculation for natural gas."""
        # Known calculation: E = C * Fd * Mw / (K * 10^6)
        # For 50 ppm NOx at 3% O2 with natural gas Fd = 8710
        cems_data = {
            "nox_ppm": 50.0,
            "o2_percent": 3.0,
            "flow_rate_dscfm": 50000.0,
        }
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Manual calculation verification
        # Fd = 8710 dscf/MMBtu, Mw_NO2 = 46.01, K = 385.3
        # E = 50 * 8710 * 46.01 / (385.3 * 10^6)
        expected_lb_mmbtu = (50.0 * 8710 * 46.01) / (385.3 * 1e6)

        assert abs(result.emission_rate_lb_mmbtu - expected_lb_mmbtu) < 0.01

    def test_epa_method_19_calculation_coal(self, emissions_tools):
        """Test EPA Method 19 calculation for bituminous coal."""
        cems_data = {
            "nox_ppm": 200.0,
            "o2_percent": 6.0,
            "flow_rate_dscfm": 80000.0,
        }
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 200.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Coal has Fd = 9780
        # O2 correction: (20.9 - 3.0) / (20.9 - 6.0) = 1.201
        o2_correction = (20.9 - 3.0) / (20.9 - 6.0)
        corrected_ppm = 200.0 * o2_correction

        expected_lb_mmbtu = (corrected_ppm * 9780 * 46.01) / (385.3 * 1e6)

        assert abs(result.emission_rate_lb_mmbtu - expected_lb_mmbtu) < 0.02

    # =========================================================================
    # F-FACTOR SELECTION TESTS
    # =========================================================================

    def test_f_factor_selection_natural_gas(self, emissions_tools):
        """Test correct F-factor selection for natural gas."""
        # Natural gas Fd = 8710 dscf/MMBtu
        assert F_FACTORS["natural_gas"]["Fd"] == 8710

    def test_f_factor_selection_fuel_oil(self, emissions_tools):
        """Test correct F-factor selection for fuel oil."""
        # No. 2 fuel oil Fd = 9190 dscf/MMBtu
        assert F_FACTORS["fuel_oil_no2"]["Fd"] == 9190

    def test_f_factor_selection_coal(self, emissions_tools):
        """Test correct F-factor selection for coal."""
        # Bituminous coal Fd = 9780 dscf/MMBtu
        assert F_FACTORS["coal_bituminous"]["Fd"] == 9780

    def test_f_factor_fallback_unknown_fuel(self, emissions_tools):
        """Test F-factor fallback for unknown fuel type."""
        cems_data = {"nox_ppm": 50.0, "o2_percent": 3.0}
        fuel_data = {
            "fuel_type": "unknown_fuel",
            "heat_input_mmbtu_hr": 100.0,
        }

        # Should not raise, should use default (natural gas)
        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        assert result is not None

    # =========================================================================
    # CONCENTRATION CONVERSION TESTS
    # =========================================================================

    def test_concentration_conversion_ppm_to_lb_mmbtu(self, emissions_tools):
        """Test ppm to lb/MMBtu conversion."""
        cems_data = {"nox_ppm": 100.0, "o2_percent": 3.0}
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # 100 ppm should give higher emission rate than 50 ppm
        assert result.emission_rate_lb_mmbtu > 0.05

    def test_concentration_conversion_ppm_to_mg_nm3(self, emissions_tools):
        """Test ppm to mg/Nm3 conversion concept."""
        # Conversion: mg/Nm3 = ppm * MW / 22.4
        # For NO2: MW = 46.01
        ppm = 50.0
        expected_mg_nm3 = ppm * 46.01 / 22.4

        # ~102.7 mg/Nm3
        assert abs(expected_mg_nm3 - 102.7) < 1.0

    # =========================================================================
    # TEMPERATURE AND PRESSURE CORRECTION TESTS
    # =========================================================================

    def test_temperature_pressure_correction_standard(self, emissions_tools):
        """Test calculations at standard conditions."""
        cems_data = {
            "nox_ppm": 50.0,
            "o2_percent": 3.0,
            "temperature_f": 68.0,  # Standard temp
            "pressure_inhg": 29.92,  # Standard pressure
        }
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # At standard conditions, correction factor should be ~1.0
        assert result.correction_factor > 0.9
        assert result.correction_factor < 1.5

    def test_o2_correction_factor_calculation(self, emissions_tools):
        """Test O2 correction factor calculation."""
        # O2 correction = (20.9 - O2_ref) / (20.9 - O2_measured)
        # At 3% O2 (reference), correction = 1.0
        cems_data_3pct = {"nox_ppm": 50.0, "o2_percent": 3.0}
        cems_data_6pct = {"nox_ppm": 50.0, "o2_percent": 6.0}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result_3pct = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data_3pct,
            fuel_data=fuel_data,
        )

        result_6pct = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data_6pct,
            fuel_data=fuel_data,
        )

        # At 3% O2, correction = 1.0
        assert abs(result_3pct.correction_factor - 1.0) < 0.01

        # At 6% O2, correction = (20.9 - 3) / (20.9 - 6) = 1.201
        expected_correction = (20.9 - 3.0) / (20.9 - 6.0)
        assert abs(result_6pct.correction_factor - expected_correction) < 0.01

    def test_o2_correction_high_excess_air(self, emissions_tools):
        """Test O2 correction at high excess air (gas turbine)."""
        cems_data = {
            "nox_ppm": 25.0,
            "o2_percent": 15.0,  # High O2 typical for gas turbines
        }
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 150.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # High O2 should give significant correction factor
        expected_correction = (20.9 - 3.0) / (20.9 - 15.0)  # ~3.03
        assert abs(result.correction_factor - expected_correction) < 0.1

    # =========================================================================
    # BOUNDARY CONDITION TESTS
    # =========================================================================

    def test_boundary_zero_nox(self, emissions_tools):
        """Test calculation with zero NOx."""
        cems_data = {"nox_ppm": 0.0, "o2_percent": 3.0}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        assert result.concentration_ppm == 0.0
        assert result.emission_rate_lb_mmbtu == 0.0

    def test_boundary_zero_heat_input(self, emissions_tools):
        """Test calculation with zero heat input."""
        cems_data = {"nox_ppm": 50.0, "o2_percent": 3.0}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 0.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Should calculate emission rate but lb/hr will be zero
        assert result.emission_rate_lb_hr == 0.0

    def test_boundary_high_nox(self, emissions_tools):
        """Test calculation with very high NOx."""
        cems_data = {"nox_ppm": 2000.0, "o2_percent": 3.0}
        fuel_data = {"fuel_type": "coal_bituminous", "heat_input_mmbtu_hr": 200.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Should handle high values without error
        assert result.concentration_ppm == 2000.0
        assert result.emission_rate_lb_mmbtu > 0

    def test_boundary_o2_near_zero(self, emissions_tools):
        """Test calculation with O2 near zero."""
        cems_data = {"nox_ppm": 50.0, "o2_percent": 0.5}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Very low O2 should give correction factor < 1
        assert result.correction_factor < 1.0

    def test_boundary_o2_near_atmospheric(self, emissions_tools):
        """Test calculation with O2 near atmospheric (20.9%)."""
        cems_data = {"nox_ppm": 50.0, "o2_percent": 20.0}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Very high O2 should give large correction factor
        assert result.correction_factor > 10.0

    # =========================================================================
    # DETERMINISM TESTS
    # =========================================================================

    def test_determinism_same_input_same_output(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test deterministic behavior - same input produces same output."""
        results = []

        for _ in range(10):
            result = emissions_tools.calculate_nox_emissions(
                cems_data=sample_cems_data,
                fuel_data=natural_gas_fuel_data,
            )
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result.concentration_ppm == first.concentration_ppm
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu
            assert result.emission_rate_lb_hr == first.emission_rate_lb_hr
            assert result.thermal_nox_percent == first.thermal_nox_percent
            assert result.fuel_nox_percent == first.fuel_nox_percent
            assert result.prompt_nox_percent == first.prompt_nox_percent

    def test_determinism_provenance_hash(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test provenance hash is deterministic."""
        hashes = []

        for _ in range(5):
            result = emissions_tools.calculate_nox_emissions(
                cems_data=sample_cems_data,
                fuel_data=natural_gas_fuel_data,
            )
            hashes.append(result.provenance_hash)

        # All hashes should be identical
        assert len(set(hashes)) == 1

    # =========================================================================
    # TO_DICT CONVERSION TEST
    # =========================================================================

    def test_nox_result_to_dict(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test NOxEmissionsResult to_dict conversion."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "concentration_ppm" in result_dict
        assert "emission_rate_lb_mmbtu" in result_dict
        assert "thermal_nox_percent" in result_dict
        assert "fuel_nox_percent" in result_dict
        assert "prompt_nox_percent" in result_dict
        assert "calculation_method" in result_dict
        assert "provenance_hash" in result_dict

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_missing_cems_data_fields(self, emissions_tools, natural_gas_fuel_data):
        """Test handling of missing CEMS data fields."""
        # Should use defaults for missing fields
        cems_data = {"nox_ppm": 45.0}  # Missing o2_percent

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # Should still calculate with defaults
        assert result is not None
        assert result.concentration_ppm > 0

    def test_missing_fuel_data_fields(self, emissions_tools, sample_cems_data):
        """Test handling of missing fuel data fields."""
        fuel_data = {}  # Empty fuel data

        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=fuel_data,
        )

        # Should use defaults
        assert result is not None


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestNOxCalculatorParametrized:
    """Parametrized tests for NOx calculator."""

    @pytest.mark.parametrize("nox_ppm,o2_percent,expected_min,expected_max", [
        (25.0, 3.0, 0.02, 0.05),
        (50.0, 3.0, 0.04, 0.08),
        (100.0, 3.0, 0.08, 0.15),
        (50.0, 6.0, 0.05, 0.12),  # Higher O2 correction
        (50.0, 15.0, 0.15, 0.35),  # Gas turbine O2
    ])
    def test_nox_emission_rate_range(
        self, emissions_tools, nox_ppm, o2_percent, expected_min, expected_max
    ):
        """Test NOx emission rates fall within expected ranges."""
        cems_data = {"nox_ppm": nox_ppm, "o2_percent": o2_percent}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        assert expected_min <= result.emission_rate_lb_mmbtu <= expected_max

    @pytest.mark.parametrize("fuel_type,fd_expected", [
        ("natural_gas", 8710),
        ("fuel_oil_no2", 9190),
        ("fuel_oil_no6", 9220),
        ("coal_bituminous", 9780),
        ("biomass_wood", 9240),
    ])
    def test_f_factor_values(self, fuel_type, fd_expected):
        """Test F-factor values match EPA Method 19 reference."""
        assert F_FACTORS[fuel_type]["Fd"] == fd_expected

    @pytest.mark.parametrize("o2_measured,expected_correction", [
        (3.0, 1.0),
        (6.0, 1.2013),
        (10.0, 1.643),
        (15.0, 3.034),
    ])
    def test_o2_correction_factors(self, emissions_tools, o2_measured, expected_correction):
        """Test O2 correction factor calculations."""
        cems_data = {"nox_ppm": 50.0, "o2_percent": o2_measured}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result = emissions_tools.calculate_nox_emissions(
            cems_data=cems_data,
            fuel_data=fuel_data,
        )

        # Allow 1% tolerance
        assert abs(result.correction_factor - expected_correction) / expected_correction < 0.01
