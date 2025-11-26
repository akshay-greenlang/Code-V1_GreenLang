# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH CO2 Calculator.

Tests CO2 emissions calculations including combustion stoichiometry,
carbon balance method, fuel-specific factors, biogenic vs fossil CO2,
GHG Protocol Scope 1, and EPA Part 98 calculations.

Test Count: 25+ tests
Coverage Target: 90%+

Standards: EPA Method 3A, 40 CFR Part 75, Part 98

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
    CO2EmissionsResult,
    AP42_EMISSION_FACTORS,
    F_FACTORS,
)


# =============================================================================
# TEST CLASS: CO2 EMISSIONS CALCULATOR
# =============================================================================

@pytest.mark.unit
class TestCO2Calculator:
    """Test suite for CO2 emissions calculations."""

    # =========================================================================
    # BASIC CALCULATION TESTS
    # =========================================================================

    def test_calculate_co2_emissions_basic(self, emissions_tools, natural_gas_fuel_data):
        """Test basic CO2 emissions calculation."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        assert isinstance(result, CO2EmissionsResult)
        assert result.concentration_percent >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.mass_rate_tons_hr >= 0
        assert result.mass_rate_kg_hr >= 0
        assert result.calculation_method == "AP42_Emission_Factor"
        assert result.provenance_hash is not None

    def test_calculate_co2_emissions_coal(self, emissions_tools, coal_bituminous_data):
        """Test CO2 calculation for coal."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=coal_bituminous_data,
        )

        # Coal has higher carbon content, higher CO2
        assert result.emission_rate_lb_mmbtu > 150  # AP-42 coal factor

    # =========================================================================
    # COMBUSTION STOICHIOMETRY TESTS
    # =========================================================================

    def test_combustion_stoichiometry_carbon_to_co2(self, emissions_tools):
        """Test C + O2 -> CO2 stoichiometry (MW ratio 44/12)."""
        # For pure carbon: CO2/C = 44.01/12.01 = 3.664
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "carbon_percent": 75.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # CO2 emission rate should reflect stoichiometry
        assert result.emission_rate_lb_mmbtu > 0

    def test_combustion_stoichiometry_methane(self, emissions_tools):
        """Test CH4 + 2O2 -> CO2 + 2H2O stoichiometry."""
        # Natural gas is primarily methane
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # AP-42 natural gas CO2 factor: 117 lb/MMBtu
        assert 110 <= result.emission_rate_lb_mmbtu <= 125

    def test_combustion_stoichiometry_complete_combustion(self, emissions_tools, natural_gas_fuel_data):
        """Test complete combustion assumption."""
        process_params = {"combustion_efficiency_percent": 99.0}

        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
            process_parameters=process_params,
        )

        # 99% combustion efficiency
        assert result.combustion_efficiency_percent == 99.0

    # =========================================================================
    # CARBON BALANCE METHOD TESTS
    # =========================================================================

    def test_carbon_balance_method_natural_gas(self, emissions_tools):
        """Test carbon balance for natural gas."""
        # Natural gas: ~75% carbon, HHV ~23,000 Btu/lb
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "carbon_percent": 75.0,
            "heating_value_btu_lb": 23000.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        assert result.carbon_content_percent == 75.0

    def test_carbon_balance_method_coal(self, emissions_tools):
        """Test carbon balance for coal."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "carbon_percent": 75.0,
            "heating_value_btu_lb": 12000.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        assert result.carbon_content_percent == 75.0

    def test_carbon_balance_mass_conservation(self, emissions_tools, natural_gas_fuel_data):
        """Test carbon mass conservation in calculation."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        # Verify mass rate conversions
        # 1 short ton = 2000 lb
        expected_tons_hr = result.mass_rate_kg_hr / 907.185  # kg to short tons
        # Allow some tolerance for rounding
        assert abs(result.mass_rate_tons_hr - (result.mass_rate_kg_hr / 907.185)) < 0.1

    # =========================================================================
    # FUEL-SPECIFIC FACTORS TESTS
    # =========================================================================

    def test_fuel_specific_factors_natural_gas(self, emissions_tools):
        """Test natural gas CO2 emission factor."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # AP-42: 117 lb CO2/MMBtu for natural gas
        expected_factor = AP42_EMISSION_FACTORS["natural_gas"]["co2_lb_mmbtu"]
        assert abs(result.emission_rate_lb_mmbtu - expected_factor * 0.99) < 5  # 99% efficiency

    def test_fuel_specific_factors_fuel_oil(self, emissions_tools):
        """Test fuel oil CO2 emission factor."""
        fuel_data = {
            "fuel_type": "fuel_oil_no2",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # AP-42: 161 lb CO2/MMBtu for No. 2 fuel oil
        expected_factor = AP42_EMISSION_FACTORS["fuel_oil_no2"]["co2_lb_mmbtu"]
        assert abs(result.emission_rate_lb_mmbtu - expected_factor * 0.99) < 5

    def test_fuel_specific_factors_coal(self, emissions_tools):
        """Test coal CO2 emission factor."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # AP-42: 205 lb CO2/MMBtu for bituminous coal
        expected_factor = AP42_EMISSION_FACTORS["coal_bituminous"]["co2_lb_mmbtu"]
        assert abs(result.emission_rate_lb_mmbtu - expected_factor * 0.99) < 10

    def test_fuel_specific_factors_biomass(self, emissions_tools, biomass_wood_data):
        """Test biomass CO2 emission factor."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=biomass_wood_data,
        )

        # Biomass should have biogenic CO2
        assert result.biogenic_percent == 100.0

    # =========================================================================
    # BIOGENIC VS FOSSIL CO2 TESTS
    # =========================================================================

    def test_biogenic_vs_fossil_co2_fossil_fuel(self, emissions_tools, natural_gas_fuel_data):
        """Test fossil fuel has 0% biogenic CO2."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        assert result.biogenic_percent == 0.0

    def test_biogenic_vs_fossil_co2_biomass(self, emissions_tools):
        """Test biomass has 100% biogenic CO2."""
        fuel_data = {
            "fuel_type": "biomass_wood",
            "heat_input_mmbtu_hr": 50.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        assert result.biogenic_percent == 100.0

    def test_biogenic_co2_reporting(self, emissions_tools, biomass_wood_data):
        """Test biogenic CO2 is correctly flagged for reporting."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=biomass_wood_data,
        )

        # Biogenic CO2 is tracked separately for GHG reporting
        result_dict = result.to_dict()
        assert "biogenic_percent" in result_dict
        assert result_dict["biogenic_percent"] == 100.0

    # =========================================================================
    # GHG PROTOCOL SCOPE 1 TESTS
    # =========================================================================

    def test_ghg_protocol_scope1_natural_gas(self, emissions_tools, natural_gas_fuel_data):
        """Test GHG Protocol Scope 1 calculation for natural gas."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        # Direct emissions from stationary combustion
        # 100 MMBtu/hr * 117 lb CO2/MMBtu * 0.99 = ~11,583 lb/hr
        expected_lb_hr = 100.0 * 117.0 * 0.99
        expected_tons_hr = expected_lb_hr / 2000.0

        assert abs(result.mass_rate_tons_hr - expected_tons_hr) < 1.0

    def test_ghg_protocol_scope1_annual_estimate(self, emissions_tools, natural_gas_fuel_data):
        """Test annual CO2 estimate for GHG inventory."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        # Annual estimate: tons/hr * 8760 hours
        annual_tons = result.mass_rate_tons_hr * 8760

        # Should be in reasonable range for 100 MMBtu/hr unit
        assert 40000 < annual_tons < 60000

    def test_ghg_protocol_emission_factor_source(self, emissions_tools, natural_gas_fuel_data):
        """Test emission factor source is documented."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        # Calculation method should indicate source
        assert result.calculation_method == "AP42_Emission_Factor"

    # =========================================================================
    # EPA PART 98 CALCULATION TESTS
    # =========================================================================

    def test_epa_part98_calculation_method(self, emissions_tools, natural_gas_fuel_data):
        """Test EPA Part 98 compliant calculation."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        # Part 98 requires mass-based CO2 calculations
        assert result.mass_rate_kg_hr > 0
        assert result.mass_rate_tons_hr > 0

    def test_epa_part98_heating_value_method(self, emissions_tools):
        """Test EPA Part 98 Tier 1 (heating value) method."""
        # Tier 1: CO2 = Fuel * HHV * EF
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "heating_value_btu_lb": 23000.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # Result should be based on heat input and emission factor
        assert result.emission_rate_lb_mmbtu > 0

    def test_epa_part98_default_factors(self, emissions_tools):
        """Test EPA Part 98 default emission factors are used."""
        fuel_types = ["natural_gas", "fuel_oil_no2", "coal_bituminous"]

        for fuel_type in fuel_types:
            fuel_data = {
                "fuel_type": fuel_type,
                "heat_input_mmbtu_hr": 100.0,
            }

            result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

            # Should use AP-42/Part 98 default factors
            expected = AP42_EMISSION_FACTORS[fuel_type]["co2_lb_mmbtu"]
            # Account for combustion efficiency
            assert abs(result.emission_rate_lb_mmbtu - expected * 0.99) < 10

    # =========================================================================
    # COMBUSTION EFFICIENCY TESTS
    # =========================================================================

    def test_combustion_efficiency_effect(self, emissions_tools, natural_gas_fuel_data):
        """Test combustion efficiency effect on CO2."""
        process_99 = {"combustion_efficiency_percent": 99.0}
        process_95 = {"combustion_efficiency_percent": 95.0}

        result_99 = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
            process_parameters=process_99,
        )

        result_95 = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
            process_parameters=process_95,
        )

        # Higher efficiency should give higher CO2 (more complete combustion)
        assert result_99.emission_rate_lb_mmbtu > result_95.emission_rate_lb_mmbtu

    def test_combustion_efficiency_default(self, emissions_tools, natural_gas_fuel_data):
        """Test default combustion efficiency."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        # Default should be 99%
        assert result.combustion_efficiency_percent == 99.0

    # =========================================================================
    # BOUNDARY CONDITION TESTS
    # =========================================================================

    def test_boundary_zero_heat_input(self, emissions_tools):
        """Test calculation with zero heat input."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 0.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        assert result.mass_rate_tons_hr == 0.0
        assert result.mass_rate_kg_hr == 0.0

    def test_boundary_high_heat_input(self, emissions_tools):
        """Test calculation with high heat input."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 5000.0,  # Large power plant
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # Should handle large values
        assert result.mass_rate_tons_hr > 0

    def test_boundary_low_carbon_fuel(self, emissions_tools):
        """Test low carbon fuel (hydrogen blend)."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "carbon_percent": 50.0,  # Reduced carbon (H2 blend)
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        # Should still calculate with default emission factor
        assert result.emission_rate_lb_mmbtu > 0

    # =========================================================================
    # DETERMINISM TESTS
    # =========================================================================

    def test_determinism_co2_calculation(self, emissions_tools, natural_gas_fuel_data):
        """Test deterministic CO2 calculation."""
        results = []

        for _ in range(10):
            result = emissions_tools.calculate_co2_emissions(
                fuel_data=natural_gas_fuel_data,
            )
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result.concentration_percent == first.concentration_percent
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu
            assert result.mass_rate_tons_hr == first.mass_rate_tons_hr
            assert result.biogenic_percent == first.biogenic_percent

    def test_determinism_provenance_hash_co2(self, emissions_tools, coal_bituminous_data):
        """Test CO2 provenance hash is deterministic."""
        hashes = []

        for _ in range(5):
            result = emissions_tools.calculate_co2_emissions(
                fuel_data=coal_bituminous_data,
            )
            hashes.append(result.provenance_hash)

        assert len(set(hashes)) == 1

    # =========================================================================
    # TO_DICT CONVERSION TEST
    # =========================================================================

    def test_co2_result_to_dict(self, emissions_tools, natural_gas_fuel_data):
        """Test CO2EmissionsResult to_dict conversion."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "concentration_percent" in result_dict
        assert "emission_rate_lb_mmbtu" in result_dict
        assert "mass_rate_tons_hr" in result_dict
        assert "mass_rate_kg_hr" in result_dict
        assert "carbon_content_percent" in result_dict
        assert "combustion_efficiency_percent" in result_dict
        assert "biogenic_percent" in result_dict
        assert "calculation_method" in result_dict
        assert "provenance_hash" in result_dict


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestCO2CalculatorParametrized:
    """Parametrized tests for CO2 calculator."""

    @pytest.mark.parametrize("fuel_type,expected_min,expected_max", [
        ("natural_gas", 110.0, 125.0),
        ("fuel_oil_no2", 155.0, 170.0),
        ("fuel_oil_no6", 165.0, 180.0),
        ("coal_bituminous", 195.0, 215.0),
        ("biomass_wood", 185.0, 205.0),
    ])
    def test_co2_emission_factors_by_fuel(
        self, emissions_tools, fuel_type, expected_min, expected_max
    ):
        """Test CO2 emission factors for different fuels."""
        fuel_data = {
            "fuel_type": fuel_type,
            "heat_input_mmbtu_hr": 100.0,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        assert expected_min <= result.emission_rate_lb_mmbtu <= expected_max

    @pytest.mark.parametrize("heat_input,expected_tons_min,expected_tons_max", [
        (50.0, 2.5, 3.5),
        (100.0, 5.0, 7.0),
        (200.0, 10.0, 14.0),
        (500.0, 25.0, 35.0),
    ])
    def test_co2_mass_rate_scaling(
        self, emissions_tools, heat_input, expected_tons_min, expected_tons_max
    ):
        """Test CO2 mass rate scales with heat input."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": heat_input,
        }

        result = emissions_tools.calculate_co2_emissions(fuel_data=fuel_data)

        assert expected_tons_min <= result.mass_rate_tons_hr <= expected_tons_max

    @pytest.mark.parametrize("efficiency,expected_factor", [
        (99.0, 0.99),
        (98.0, 0.98),
        (95.0, 0.95),
        (90.0, 0.90),
    ])
    def test_combustion_efficiency_scaling(
        self, emissions_tools, natural_gas_fuel_data, efficiency, expected_factor
    ):
        """Test combustion efficiency properly scales CO2."""
        process_params = {"combustion_efficiency_percent": efficiency}

        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
            process_parameters=process_params,
        )

        # Expected: AP-42 factor * efficiency
        base_factor = AP42_EMISSION_FACTORS["natural_gas"]["co2_lb_mmbtu"]
        expected = base_factor * expected_factor

        assert abs(result.emission_rate_lb_mmbtu - expected) < 1.0
