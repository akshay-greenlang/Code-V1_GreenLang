# -*- coding: utf-8 -*-
"""
Tests for multi-fuel optimization calculator.

Tests the MultiFuelOptimizer for:
- Basic optimization scenarios
- Various optimization objectives
- Constraint satisfaction
- Edge cases
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.multi_fuel_optimizer import (
    MultiFuelOptimizer,
    MultiFuelOptimizationInput,
    MultiFuelOptimizationOutput
)


class TestMultiFuelOptimizer:
    """Test suite for MultiFuelOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return MultiFuelOptimizer()

    @pytest.fixture
    def fuel_properties(self):
        """Sample fuel properties."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'emission_factor_sox_g_gj': 0.3,
                'renewable': False
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_nox_g_gj': 250,
                'emission_factor_sox_g_gj': 500,
                'renewable': False
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_nox_g_gj': 150,
                'emission_factor_sox_g_gj': 20,
                'renewable': True
            },
            'hydrogen': {
                'heating_value_mj_kg': 120.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_nox_g_gj': 10,
                'emission_factor_sox_g_gj': 0,
                'renewable': True
            }
        }

    @pytest.fixture
    def market_prices(self):
        """Sample market prices (USD/kg)."""
        return {
            'natural_gas': 0.045,
            'coal': 0.035,
            'biomass': 0.08,
            'hydrogen': 5.00
        }

    def test_basic_optimization(self, optimizer, fuel_properties, market_prices):
        """Test basic multi-fuel optimization."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        assert result is not None
        assert result.total_cost_usd > 0
        assert result.efficiency_percent > 0
        assert sum(result.optimal_fuel_mix.values()) > 0.99  # Should sum to ~1

    def test_minimize_cost_objective(self, optimizer, fuel_properties, market_prices):
        """Test cost minimization selects cheapest fuel."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='minimize_cost'
        )

        result = optimizer.optimize(input_data)

        # Coal should be preferred (cheaper per GJ)
        assert result.optimal_fuel_mix.get('coal', 0) > 0

    def test_minimize_emissions_objective(self, optimizer, fuel_properties, market_prices):
        """Test emissions minimization selects low-carbon fuels."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass', 'hydrogen'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='minimize_emissions'
        )

        result = optimizer.optimize(input_data)

        # Renewable fuels should be preferred
        renewable_share = result.optimal_fuel_mix.get('biomass', 0) + result.optimal_fuel_mix.get('hydrogen', 0)
        assert renewable_share > 0

    def test_renewable_priority_objective(self, optimizer, fuel_properties, market_prices):
        """Test renewable priority selects renewable fuels."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'biomass', 'hydrogen'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='renewable_priority'
        )

        result = optimizer.optimize(input_data)

        assert result.renewable_share > 0.5  # Should prioritize renewables

    def test_emission_constraints(self, optimizer, fuel_properties, market_prices):
        """Test emission constraints exclude high-emission fuels."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={'nox_g_gj': 100, 'sox_g_gj': 100},  # Exclude coal
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        # Coal should be excluded due to emission limits
        assert result.optimal_fuel_mix.get('coal', 0) == 0

    def test_max_share_constraints(self, optimizer, fuel_properties, market_prices):
        """Test maximum share constraints."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={'coal_max_share': 0.3},  # Max 30% coal
            optimization_objective='minimize_cost'
        )

        result = optimizer.optimize(input_data)

        coal_share = result.optimal_fuel_mix.get('coal', 0)
        assert coal_share <= 0.31  # Allow small tolerance

    def test_provenance_hash_generated(self, optimizer, fuel_properties, market_prices):
        """Test that provenance hash is generated."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_determinism(self, optimizer, fuel_properties, market_prices):
        """Test optimization is deterministic."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result1 = optimizer.optimize(input_data)
        result2 = optimizer.optimize(input_data)

        assert result1.optimal_fuel_mix == result2.optimal_fuel_mix
        assert result1.total_cost_usd == result2.total_cost_usd

    def test_zero_demand_edge_case(self, optimizer, fuel_properties, market_prices):
        """Test handling of zero energy demand."""
        with pytest.raises(ValueError):
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=0,
                available_fuels=['natural_gas'],
                fuel_properties=fuel_properties,
                market_prices=market_prices,
                emission_limits={},
                constraints={},
                optimization_objective='balanced'
            )
            optimizer.optimize(input_data)

    def test_no_fuels_edge_case(self, optimizer, fuel_properties, market_prices):
        """Test handling of empty fuel list."""
        with pytest.raises(ValueError):
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=100,
                available_fuels=[],
                fuel_properties=fuel_properties,
                market_prices=market_prices,
                emission_limits={},
                constraints={},
                optimization_objective='balanced'
            )
            optimizer.optimize(input_data)

    def test_optimization_score_range(self, optimizer, fuel_properties, market_prices):
        """Test optimization score is in valid range."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        assert 0 <= result.optimization_score <= 100

    def test_carbon_intensity_calculation(self, optimizer, fuel_properties, market_prices):
        """Test carbon intensity is calculated correctly."""
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        # Carbon intensity should be in reasonable range for natural gas
        assert 100 < result.carbon_intensity_kg_mwh < 500
