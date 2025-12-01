# -*- coding: utf-8 -*-
"""
Tests for carbon footprint calculator.

Tests the CarbonFootprintCalculator for:
- GHG Protocol compliance
- IPCC emission factor accuracy
- Scope 1/2/3 calculations
- Carbon intensity metrics
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.carbon_footprint_calculator import (
    CarbonFootprintCalculator,
    CarbonFootprintInput,
    CarbonFootprintOutput
)


class TestCarbonFootprintCalculator:
    """Test suite for CarbonFootprintCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return CarbonFootprintCalculator()

    @pytest.fixture
    def fuel_properties(self):
        """Sample fuel properties with emission factors."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_ch4_kg_gj': 0.001,
                'emission_factor_n2o_kg_gj': 0.0001
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_ch4_kg_gj': 0.001,
                'emission_factor_n2o_kg_gj': 0.0015
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,  # Biogenic carbon
                'emission_factor_ch4_kg_gj': 0.03,
                'emission_factor_n2o_kg_gj': 0.004
            },
            'fuel_oil': {
                'heating_value_mj_kg': 42.0,
                'emission_factor_co2_kg_gj': 77.4,
                'emission_factor_ch4_kg_gj': 0.003,
                'emission_factor_n2o_kg_gj': 0.0006
            }
        }

    def test_basic_co2_calculation(self, calculator, fuel_properties):
        """Test basic CO2 calculation."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},  # kg
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        # 1000 kg NG * 50 MJ/kg = 50,000 MJ = 50 GJ
        # 50 GJ * 56.1 kg CO2/GJ = 2805 kg CO2
        assert result.total_co2_kg > 2500
        assert result.total_co2_kg < 3200

    def test_co2e_includes_ch4_and_n2o(self, calculator, fuel_properties):
        """Test CO2e calculation includes CH4 and N2O."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        # CO2e should be greater than CO2 alone due to CH4 and N2O
        assert result.total_co2e_kg >= result.total_co2_kg

    def test_gwp_ar6_values(self, calculator):
        """Test GWP values match IPCC AR6."""
        # IPCC AR6 GWP100 values
        assert calculator.GWP_CO2 == 1.0
        assert calculator.GWP_CH4 == 29.8  # AR6 value (was 28 in AR5)
        assert calculator.GWP_N2O == 273.0  # AR6 value (was 265 in AR5)

    def test_multi_fuel_calculation(self, calculator, fuel_properties):
        """Test calculation with multiple fuels."""
        input_data = CarbonFootprintInput(
            fuel_quantities={
                'natural_gas': 500,
                'coal': 300,
                'fuel_oil': 200
            },
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        # Should have breakdown by fuel
        assert 'natural_gas' in result.emissions_by_fuel
        assert 'coal' in result.emissions_by_fuel
        assert 'fuel_oil' in result.emissions_by_fuel

        # Total should be sum of individual fuels
        total_from_breakdown = sum(result.emissions_by_fuel.values())
        assert abs(result.total_co2e_kg - total_from_breakdown) < 1.0

    def test_biomass_biogenic_carbon(self, calculator, fuel_properties):
        """Test biomass has zero fossil CO2 emissions."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'biomass': 1000},
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        # Biogenic CO2 should be zero (carbon neutral)
        assert result.total_co2_kg == 0.0
        # But CO2e may be non-zero due to CH4 and N2O
        assert result.total_co2e_kg >= 0.0

    def test_carbon_intensity_calculation(self, calculator, fuel_properties):
        """Test carbon intensity calculation (kg/MWh)."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        # Carbon intensity should be in reasonable range
        # Natural gas: ~200 kg CO2/MWh thermal
        assert 150 < result.carbon_intensity_kg_mwh < 300

    def test_coal_highest_intensity(self, calculator, fuel_properties):
        """Test coal has highest carbon intensity."""
        ng_input = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties
        )
        coal_input = CarbonFootprintInput(
            fuel_quantities={'coal': 1000},
            fuel_properties=fuel_properties
        )

        ng_result = calculator.calculate(ng_input)
        coal_result = calculator.calculate(coal_input)

        # Coal should have higher intensity than natural gas
        assert coal_result.carbon_intensity_kg_mwh > ng_result.carbon_intensity_kg_mwh

    def test_provenance_hash_generated(self, calculator, fuel_properties):
        """Test provenance hash is generated."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_determinism(self, calculator, fuel_properties):
        """Test calculation is deterministic."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000, 'coal': 500},
            fuel_properties=fuel_properties
        )

        result1 = calculator.calculate(input_data)
        result2 = calculator.calculate(input_data)

        assert result1.total_co2e_kg == result2.total_co2e_kg
        assert result1.carbon_intensity_kg_mwh == result2.carbon_intensity_kg_mwh
        assert result1.provenance_hash == result2.provenance_hash

    def test_zero_quantity_handling(self, calculator, fuel_properties):
        """Test handling of zero fuel quantity."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 0},
            fuel_properties=fuel_properties
        )

        result = calculator.calculate(input_data)

        assert result.total_co2e_kg == 0.0
        assert result.total_co2_kg == 0.0

    def test_scope_classification(self, calculator, fuel_properties):
        """Test emissions are classified by scope."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties,
            include_scope_breakdown=True
        )

        result = calculator.calculate(input_data)

        # Scope 1 (direct combustion) should be present
        assert hasattr(result, 'scope_1_emissions')
        assert result.scope_1_emissions > 0

    def test_emission_factors_ipcc_compliant(self, calculator):
        """Test emission factors match IPCC defaults."""
        # IPCC 2006 default emission factors for stationary combustion
        ipcc_factors = {
            'natural_gas': 56.1,  # kg CO2/GJ
            'coal': 94.6,  # Bituminous coal
            'fuel_oil': 77.4,  # Residual fuel oil
        }

        for fuel, expected_factor in ipcc_factors.items():
            factor = calculator.get_default_emission_factor(fuel)
            assert abs(factor - expected_factor) < 1.0, \
                f"{fuel} emission factor mismatch"

    def test_uncertainty_estimation(self, calculator, fuel_properties):
        """Test uncertainty estimation for GHG inventory."""
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000},
            fuel_properties=fuel_properties,
            include_uncertainty=True
        )

        result = calculator.calculate(input_data)

        # IPCC default uncertainty ranges
        assert hasattr(result, 'uncertainty_percent')
        assert 0 < result.uncertainty_percent < 50


class TestCarbonFootprintMinimization:
    """Test suite for carbon footprint minimization scenarios."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return CarbonFootprintCalculator()

    @pytest.fixture
    def fuel_properties(self):
        """Full fuel properties set."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_ch4_kg_gj': 0.001,
                'emission_factor_n2o_kg_gj': 0.0001
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_ch4_kg_gj': 0.001,
                'emission_factor_n2o_kg_gj': 0.0015
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_ch4_kg_gj': 0.03,
                'emission_factor_n2o_kg_gj': 0.004
            },
            'hydrogen': {
                'heating_value_mj_kg': 120.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_ch4_kg_gj': 0.0,
                'emission_factor_n2o_kg_gj': 0.0
            }
        }

    def test_fuel_switch_emissions_reduction(self, calculator, fuel_properties):
        """Test emissions reduction from fuel switching."""
        # Baseline: 100% coal
        coal_input = CarbonFootprintInput(
            fuel_quantities={'coal': 1000},
            fuel_properties=fuel_properties
        )
        coal_result = calculator.calculate(coal_input)

        # Switch to natural gas (same energy)
        # Coal: 1000 kg * 25 MJ/kg = 25,000 MJ
        # NG: 25,000 MJ / 50 MJ/kg = 500 kg
        ng_input = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 500},
            fuel_properties=fuel_properties
        )
        ng_result = calculator.calculate(ng_input)

        # Natural gas should have lower emissions for same energy
        assert ng_result.total_co2e_kg < coal_result.total_co2e_kg

    def test_renewable_blend_reduction(self, calculator, fuel_properties):
        """Test emissions reduction from renewable blending."""
        # 100% coal baseline
        coal_only = CarbonFootprintInput(
            fuel_quantities={'coal': 1000},
            fuel_properties=fuel_properties
        )
        coal_result = calculator.calculate(coal_only)

        # 50% coal + 50% biomass (energy equivalent)
        # Coal provides 1000 * 25 = 25,000 MJ
        # Half from coal = 500 kg coal
        # Half from biomass = 12,500 MJ / 18 = 694 kg biomass
        blend = CarbonFootprintInput(
            fuel_quantities={'coal': 500, 'biomass': 694},
            fuel_properties=fuel_properties
        )
        blend_result = calculator.calculate(blend)

        # Blend should have ~50% lower CO2 (biomass is carbon neutral)
        assert blend_result.total_co2_kg < coal_result.total_co2_kg * 0.6

    def test_hydrogen_zero_emissions(self, calculator, fuel_properties):
        """Test green hydrogen has zero combustion emissions."""
        h2_input = CarbonFootprintInput(
            fuel_quantities={'hydrogen': 100},
            fuel_properties=fuel_properties
        )
        result = calculator.calculate(h2_input)

        # Green hydrogen has zero direct emissions
        assert result.total_co2e_kg == 0.0
