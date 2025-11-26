# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH SOx Calculator.

Tests SOx emissions calculations including fuel sulfur content, SO2/SO3
partitioning, scrubber efficiency, mass balance calculations, and
EPA Method 6 compliance.

Test Count: 20+ tests
Coverage Target: 90%+

Standards: EPA Method 6, 40 CFR Part 60

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import (
    EmissionsComplianceTools,
    SOxEmissionsResult,
    F_FACTORS,
    MOLECULAR_WEIGHTS,
    AP42_EMISSION_FACTORS,
)


# =============================================================================
# TEST CLASS: SOx EMISSIONS CALCULATOR
# =============================================================================

@pytest.mark.unit
class TestSOxCalculator:
    """Test suite for SOx emissions calculations."""

    # =========================================================================
    # BASIC CALCULATION TESTS
    # =========================================================================

    def test_calculate_sox_emissions_basic(self, emissions_tools, natural_gas_fuel_data):
        """Test basic SOx emissions calculation."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        assert isinstance(result, SOxEmissionsResult)
        assert result.concentration_ppm >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.emission_rate_lb_hr >= 0
        assert result.mass_rate_kg_hr >= 0
        assert result.calculation_method == "Stoichiometric_S_to_SO2"
        assert result.provenance_hash is not None

    def test_calculate_sox_emissions_coal(self, emissions_tools, coal_bituminous_data):
        """Test SOx calculation for high-sulfur coal."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
        )

        # Coal with 2% sulfur should have significant SOx
        assert result.emission_rate_lb_mmbtu > 0.5
        assert result.fuel_sulfur_percent == 2.0

    # =========================================================================
    # FUEL SULFUR CONTENT TESTS
    # =========================================================================

    def test_sox_from_fuel_sulfur_linear_relationship(self, emissions_tools):
        """Test SOx emissions scale linearly with fuel sulfur content."""
        fuel_low_s = {
            "fuel_type": "fuel_oil_no2",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 19500.0,
            "sulfur_percent": 0.25,
        }

        fuel_high_s = {
            "fuel_type": "fuel_oil_no2",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 19500.0,
            "sulfur_percent": 0.50,
        }

        result_low = emissions_tools.calculate_sox_emissions(fuel_data=fuel_low_s)
        result_high = emissions_tools.calculate_sox_emissions(fuel_data=fuel_high_s)

        # Double sulfur should approximately double emissions
        ratio = result_high.emission_rate_lb_hr / result_low.emission_rate_lb_hr
        assert 1.8 <= ratio <= 2.2

    def test_sox_from_fuel_sulfur_natural_gas(self, emissions_tools):
        """Test natural gas has negligible SOx."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 23000.0,
            "sulfur_percent": 0.0006,  # Typical natural gas
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # Natural gas should have very low SOx
        assert result.emission_rate_lb_mmbtu < 0.01

    def test_sox_from_fuel_sulfur_high_sulfur_fuel(self, emissions_tools):
        """Test high sulfur residual fuel oil."""
        fuel_data = {
            "fuel_type": "fuel_oil_no6",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 18000.0,
            "sulfur_percent": 3.0,  # High sulfur residual oil
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # High sulfur fuel should have significant SOx
        assert result.emission_rate_lb_mmbtu > 1.0

    # =========================================================================
    # SO2/SO3 PARTITIONING TESTS
    # =========================================================================

    def test_so2_so3_partitioning_typical(self, emissions_tools, fuel_oil_no2_data):
        """Test typical SO2/SO3 partitioning ratio."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data=fuel_oil_no2_data,
        )

        # Typically 95-99% SO2, 1-5% SO3
        assert 95.0 <= result.so2_so3_ratio <= 99.0

    def test_so2_so3_ratio_consistency(self, emissions_tools):
        """Test SO2/SO3 ratio is consistent across calculations."""
        fuel_types = ["natural_gas", "fuel_oil_no2", "coal_bituminous"]
        ratios = []

        for fuel_type in fuel_types:
            fuel_data = {
                "fuel_type": fuel_type,
                "heat_input_mmbtu_hr": 100.0,
                "sulfur_percent": 1.0,
            }
            result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)
            ratios.append(result.so2_so3_ratio)

        # All ratios should be in valid range
        for ratio in ratios:
            assert 95.0 <= ratio <= 99.0

    # =========================================================================
    # SCRUBBER EFFICIENCY TESTS
    # =========================================================================

    def test_scrubber_efficiency_zero(self, emissions_tools, coal_bituminous_data):
        """Test SOx calculation without scrubber."""
        process_params = {"fgd_efficiency_percent": 0.0}

        result = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
            process_parameters=process_params,
        )

        assert result.removal_efficiency_percent == 0.0

    def test_scrubber_efficiency_high(self, emissions_tools, coal_bituminous_data):
        """Test SOx reduction with high-efficiency FGD."""
        process_no_fgd = {"fgd_efficiency_percent": 0.0}
        process_with_fgd = {"fgd_efficiency_percent": 95.0}

        result_no_fgd = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
            process_parameters=process_no_fgd,
        )

        result_with_fgd = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
            process_parameters=process_with_fgd,
        )

        # 95% removal should reduce emissions by 95%
        expected_reduction = result_no_fgd.emission_rate_lb_hr * 0.05
        assert abs(result_with_fgd.emission_rate_lb_hr - expected_reduction) < 0.1

    def test_scrubber_efficiency_moderate(self, emissions_tools, coal_bituminous_data):
        """Test SOx reduction with moderate FGD efficiency."""
        process_params = {"fgd_efficiency_percent": 70.0}

        result = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
            process_parameters=process_params,
        )

        assert result.removal_efficiency_percent == 70.0

    # =========================================================================
    # MASS BALANCE CALCULATION TESTS
    # =========================================================================

    def test_mass_balance_sulfur_to_so2(self, emissions_tools):
        """Test mass balance: S -> SO2 stoichiometry."""
        # MW(SO2) = 64.07, MW(S) = 32.07
        # SO2/S ratio = 64.07/32.07 = 2.0
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": 2.0,
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # Calculate expected SO2 from fuel consumption
        fuel_rate_lb_hr = (100.0 * 1e6) / 12000.0  # ~8333 lb/hr
        sulfur_rate_lb_hr = fuel_rate_lb_hr * 0.02  # 2% sulfur
        expected_so2_lb_hr = sulfur_rate_lb_hr * 2.0  # S -> SO2 factor

        # Allow 5% tolerance for calculation differences
        assert abs(result.emission_rate_lb_hr - expected_so2_lb_hr) / expected_so2_lb_hr < 0.05

    def test_mass_balance_conservation(self, emissions_tools, coal_bituminous_data):
        """Test mass conservation in SOx calculations."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
        )

        # lb/hr to kg/hr conversion
        expected_kg_hr = result.emission_rate_lb_hr * 0.453592
        assert abs(result.mass_rate_kg_hr - expected_kg_hr) < 0.01

    def test_mass_balance_heat_input_scaling(self, emissions_tools):
        """Test SOx scales with heat input."""
        fuel_50 = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 50.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": 2.0,
        }

        fuel_100 = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": 2.0,
        }

        result_50 = emissions_tools.calculate_sox_emissions(fuel_data=fuel_50)
        result_100 = emissions_tools.calculate_sox_emissions(fuel_data=fuel_100)

        # Double heat input should double lb/hr
        ratio = result_100.emission_rate_lb_hr / result_50.emission_rate_lb_hr
        assert 1.9 <= ratio <= 2.1

        # But lb/MMBtu should be same
        assert abs(result_100.emission_rate_lb_mmbtu - result_50.emission_rate_lb_mmbtu) < 0.01

    # =========================================================================
    # EPA METHOD 6 CALCULATION TESTS
    # =========================================================================

    def test_epa_method_6_calculation(self, emissions_tools):
        """Test EPA Method 6 compliant calculation."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": 2.0,
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # Verify calculation method
        assert result.calculation_method == "Stoichiometric_S_to_SO2"

        # Verify result has all required fields
        assert result.concentration_ppm is not None
        assert result.emission_rate_lb_mmbtu is not None
        assert result.fuel_sulfur_percent == 2.0

    def test_concentration_ppm_calculation(self, emissions_tools):
        """Test SOx concentration in ppm calculation."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": 2.0,
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # Concentration should be positive for sulfur-containing fuel
        assert result.concentration_ppm > 0

    # =========================================================================
    # BOUNDARY CONDITION TESTS
    # =========================================================================

    def test_boundary_zero_sulfur(self, emissions_tools):
        """Test calculation with zero sulfur fuel."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 23000.0,
            "sulfur_percent": 0.0,
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        assert result.emission_rate_lb_hr == 0.0
        assert result.emission_rate_lb_mmbtu == 0.0

    def test_boundary_very_high_sulfur(self, emissions_tools):
        """Test calculation with very high sulfur content."""
        fuel_data = {
            "fuel_type": "fuel_oil_no6",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 18000.0,
            "sulfur_percent": 5.0,  # Very high sulfur
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # Should handle high values
        assert result.emission_rate_lb_mmbtu > 0
        assert result.fuel_sulfur_percent == 5.0

    def test_boundary_zero_heat_input(self, emissions_tools):
        """Test calculation with zero heat input."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 0.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": 2.0,
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        # Zero heat input means zero mass rate
        assert result.emission_rate_lb_hr == 0.0

    # =========================================================================
    # DETERMINISM TESTS
    # =========================================================================

    def test_determinism_sox_calculation(self, emissions_tools, fuel_oil_no2_data):
        """Test deterministic SOx calculation."""
        results = []

        for _ in range(10):
            result = emissions_tools.calculate_sox_emissions(
                fuel_data=fuel_oil_no2_data,
            )
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result.concentration_ppm == first.concentration_ppm
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu
            assert result.emission_rate_lb_hr == first.emission_rate_lb_hr
            assert result.so2_so3_ratio == first.so2_so3_ratio

    def test_determinism_provenance_hash_sox(self, emissions_tools, coal_bituminous_data):
        """Test SOx provenance hash is deterministic."""
        hashes = []

        for _ in range(5):
            result = emissions_tools.calculate_sox_emissions(
                fuel_data=coal_bituminous_data,
            )
            hashes.append(result.provenance_hash)

        assert len(set(hashes)) == 1

    # =========================================================================
    # TO_DICT CONVERSION TEST
    # =========================================================================

    def test_sox_result_to_dict(self, emissions_tools, fuel_oil_no2_data):
        """Test SOxEmissionsResult to_dict conversion."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data=fuel_oil_no2_data,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "concentration_ppm" in result_dict
        assert "emission_rate_lb_mmbtu" in result_dict
        assert "fuel_sulfur_percent" in result_dict
        assert "so2_so3_ratio" in result_dict
        assert "removal_efficiency_percent" in result_dict
        assert "calculation_method" in result_dict
        assert "provenance_hash" in result_dict


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestSOxCalculatorParametrized:
    """Parametrized tests for SOx calculator."""

    @pytest.mark.parametrize("sulfur_percent,expected_min,expected_max", [
        (0.1, 0.05, 0.15),
        (0.5, 0.25, 0.75),
        (1.0, 0.50, 1.50),
        (2.0, 1.00, 3.00),
        (3.0, 1.50, 4.50),
    ])
    def test_sox_emission_rate_range(
        self, emissions_tools, sulfur_percent, expected_min, expected_max
    ):
        """Test SOx emission rates for different sulfur contents."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 12000.0,
            "sulfur_percent": sulfur_percent,
        }

        result = emissions_tools.calculate_sox_emissions(fuel_data=fuel_data)

        assert expected_min <= result.emission_rate_lb_mmbtu <= expected_max

    @pytest.mark.parametrize("fgd_efficiency,expected_removal", [
        (0.0, 0.0),
        (50.0, 50.0),
        (90.0, 90.0),
        (95.0, 95.0),
        (99.0, 99.0),
    ])
    def test_fgd_efficiency_values(
        self, emissions_tools, coal_bituminous_data, fgd_efficiency, expected_removal
    ):
        """Test FGD efficiency correctly applied."""
        process_params = {"fgd_efficiency_percent": fgd_efficiency}

        result = emissions_tools.calculate_sox_emissions(
            fuel_data=coal_bituminous_data,
            process_parameters=process_params,
        )

        assert result.removal_efficiency_percent == expected_removal
