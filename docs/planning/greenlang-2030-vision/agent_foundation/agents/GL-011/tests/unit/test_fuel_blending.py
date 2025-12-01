# -*- coding: utf-8 -*-
"""
Tests for fuel blending calculator.

Tests the FuelBlendingCalculator for:
- Blend ratio optimization
- Property constraints
- Compatibility checks
- Quality scoring
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.fuel_blending_calculator import (
    FuelBlendingCalculator,
    BlendingInput,
    BlendingOutput
)


class TestFuelBlendingCalculator:
    """Test suite for FuelBlendingCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return FuelBlendingCalculator()

    @pytest.fixture
    def fuel_properties(self):
        """Sample fuel properties."""
        return {
            'coal': {
                'heating_value_mj_kg': 25.0,
                'carbon_content_percent': 60.0,
                'moisture_content_percent': 8.0,
                'ash_content_percent': 10.0,
                'sulfur_content_percent': 2.0,
                'emission_factor_co2_kg_gj': 94.6
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'carbon_content_percent': 50.0,
                'moisture_content_percent': 25.0,
                'ash_content_percent': 2.0,
                'sulfur_content_percent': 0.1,
                'emission_factor_co2_kg_gj': 0.0
            },
            'wood_pellets': {
                'heating_value_mj_kg': 17.5,
                'carbon_content_percent': 50.0,
                'moisture_content_percent': 8.0,
                'ash_content_percent': 0.5,
                'sulfur_content_percent': 0.02,
                'emission_factor_co2_kg_gj': 0.0
            }
        }

    def test_basic_blend_optimization(self, calculator, fuel_properties):
        """Test basic blend optimization."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=20.0,
            max_ash=15.0,
            max_sulfur=2.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        assert result is not None
        assert sum(result.blend_ratios.values()) > 0.99  # Should sum to ~1
        assert result.blend_heating_value > 0

    def test_moisture_constraint(self, calculator, fuel_properties):
        """Test moisture constraint is respected."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=20.0,
            max_moisture=15.0,  # Strict moisture limit
            max_ash=15.0,
            max_sulfur=2.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        # Should generate warning if moisture exceeds limit
        if result.blend_moisture > 15.0:
            assert any('moisture' in w.lower() for w in result.warnings)

    def test_sulfur_constraint(self, calculator, fuel_properties):
        """Test sulfur constraint is respected."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=1.0,  # Strict sulfur limit
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        # High coal ratio would violate sulfur limit
        if result.blend_sulfur > 1.0:
            assert any('sulfur' in w.lower() for w in result.warnings)

    def test_incompatibility_check(self, calculator, fuel_properties):
        """Test incompatible fuel pairs are flagged."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=20.0,
            max_moisture=30.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='balanced',
            incompatible_pairs=[['coal', 'biomass']]  # Mark as incompatible
        )

        result = calculator.optimize_blend(input_data)

        # Compatibility should fail
        assert result.compatibility_ok is False
        assert any('incompatible' in w.lower() for w in result.warnings)

    def test_quality_score_range(self, calculator, fuel_properties):
        """Test quality score is in valid range."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        assert 0 <= result.quality_score <= 100

    def test_emissions_estimate(self, calculator, fuel_properties):
        """Test emissions are estimated for blend."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='minimize_emissions',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        assert 'co2_kg_gj' in result.estimated_emissions
        assert 'nox_g_gj' in result.estimated_emissions
        assert 'sox_g_gj' in result.estimated_emissions

    def test_emissions_objective_favors_renewables(self, calculator, fuel_properties):
        """Test emissions objective favors low-carbon fuels."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass', 'wood_pellets'],
            fuel_properties=fuel_properties,
            target_heating_value=18.0,
            max_moisture=30.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='minimize_emissions',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        # Biomass/wood pellets should be favored
        renewable_share = result.blend_ratios.get('biomass', 0) + result.blend_ratios.get('wood_pellets', 0)
        assert renewable_share > 0.3

    def test_provenance_hash(self, calculator, fuel_properties):
        """Test provenance hash is generated."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],
            fuel_properties=fuel_properties,
            target_heating_value=22.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result = calculator.optimize_blend(input_data)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_determinism(self, calculator, fuel_properties):
        """Test calculation is deterministic."""
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass', 'wood_pellets'],
            fuel_properties=fuel_properties,
            target_heating_value=20.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=2.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        result1 = calculator.optimize_blend(input_data)
        result2 = calculator.optimize_blend(input_data)

        assert result1.blend_ratios == result2.blend_ratios
        assert result1.blend_heating_value == result2.blend_heating_value
