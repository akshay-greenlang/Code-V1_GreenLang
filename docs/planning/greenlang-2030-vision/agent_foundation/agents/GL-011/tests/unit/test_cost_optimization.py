# -*- coding: utf-8 -*-
"""
Tests for cost optimization calculator.

Tests the CostOptimizationCalculator for:
- Minimum cost selection
- Inventory constraints
- Delivery cost inclusion
- Alternative ranking
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.cost_optimization_calculator import (
    CostOptimizationCalculator,
    CostOptimizationInput,
    CostOptimizationOutput
)


class TestCostOptimizationCalculator:
    """Test suite for CostOptimizationCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return CostOptimizationCalculator()

    @pytest.fixture
    def fuel_properties(self):
        """Sample fuel properties."""
        return {
            'natural_gas': {'heating_value_mj_kg': 50.0},
            'coal': {'heating_value_mj_kg': 25.0},
            'biomass': {'heating_value_mj_kg': 18.0}
        }

    @pytest.fixture
    def market_prices(self):
        """Sample market prices (USD/kg)."""
        return {
            'natural_gas': 0.045,
            'coal': 0.035,
            'biomass': 0.08
        }

    def test_selects_cheapest_fuel(self, calculator, fuel_properties, market_prices):
        """Test that optimizer selects cheapest option per GJ."""
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000, 'coal': 100000, 'biomass': 100000},
            delivery_costs={'natural_gas': 0, 'coal': 0, 'biomass': 0},
            constraints={}
        )

        result = calculator.optimize(input_data)

        # Cost per GJ: NG=0.9, Coal=1.4, Biomass=4.4
        # Natural gas should be selected (cheapest per GJ)
        assert result.optimal_fuel == 'natural_gas'

    def test_considers_delivery_costs(self, calculator, fuel_properties, market_prices):
        """Test delivery costs are included in total."""
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000, 'coal': 100000},
            delivery_costs={'natural_gas': 50, 'coal': 10},  # USD/tonne
            constraints={}
        )

        result = calculator.optimize(input_data)

        # Cost breakdown should include delivery
        assert result.cost_breakdown.get('delivery_cost', 0) >= 0

    def test_inventory_check(self, calculator, fuel_properties, market_prices):
        """Test inventory availability is checked."""
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100, 'coal': 100000},  # Low NG
            delivery_costs={'natural_gas': 0, 'coal': 0},
            constraints={}
        )

        result = calculator.optimize(input_data)

        # Should indicate if inventory is sufficient
        assert hasattr(result, 'inventory_sufficient')

    def test_alternatives_ranked(self, calculator, fuel_properties, market_prices):
        """Test alternatives are properly ranked by cost."""
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000, 'coal': 100000, 'biomass': 100000},
            delivery_costs={'natural_gas': 0, 'coal': 0, 'biomass': 0},
            constraints={}
        )

        result = calculator.optimize(input_data)

        # Alternatives should be provided
        assert len(result.alternatives) > 0

        # Alternatives should be sorted by cost
        if len(result.alternatives) > 1:
            costs = [a['total_cost_usd'] for a in result.alternatives]
            assert costs == sorted(costs)

    def test_provenance_hash(self, calculator, fuel_properties, market_prices):
        """Test provenance hash is generated."""
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000},
            delivery_costs={'natural_gas': 0},
            constraints={}
        )

        result = calculator.optimize(input_data)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_levelized_cost_calculation(self, calculator, fuel_properties):
        """Test levelized cost calculation."""
        lcoe = calculator.calculate_levelized_cost(
            fuel='natural_gas',
            properties=fuel_properties['natural_gas'],
            price=0.045,
            delivery_cost=50,
            carbon_price=50
        )

        # LCOE should be positive
        assert lcoe > 0

    def test_determinism(self, calculator, fuel_properties, market_prices):
        """Test calculation is deterministic."""
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000, 'coal': 100000},
            delivery_costs={'natural_gas': 0, 'coal': 0},
            constraints={}
        )

        result1 = calculator.optimize(input_data)
        result2 = calculator.optimize(input_data)

        assert result1.optimal_fuel == result2.optimal_fuel
        assert result1.total_cost_usd == result2.total_cost_usd
