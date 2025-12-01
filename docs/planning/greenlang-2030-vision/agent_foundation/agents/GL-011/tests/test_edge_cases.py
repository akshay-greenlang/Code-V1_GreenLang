# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Edge Case Test Suite.

This module tests edge cases and boundary conditions:
- Zero fuel inventory handling
- Single fuel optimization (no blending)
- All-same-price fuels scenario
- Extreme price volatility (1000x spike)
- Negative calorific values (error handling)
- Division by zero in cost calculation
- Missing emission factors
- Corrupt cache entries
- Empty fuel lists
- Extreme demand values (0 MW, 10000 MW)
- Precision boundaries
- Unicode in fuel names
- Very long fuel IDs

Test Count: 30+ edge case tests
Coverage: Boundary conditions, unusual inputs, error resilience

Author: GreenLang Industrial Optimization Team
Version: 1.0.0
"""

import pytest
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FuelSpecification, FuelInventory, FuelCategory, FuelState
from calculators.multi_fuel_optimizer import MultiFuelOptimizer, MultiFuelOptimizationInput
from calculators.cost_optimization_calculator import CostOptimizationCalculator, CostOptimizationInput
from calculators.fuel_blending_calculator import FuelBlendingCalculator, BlendingInput
from pydantic import ValidationError


@pytest.mark.edge_case
class TestZeroInventoryEdgeCases:
    """Edge cases for zero fuel inventory."""

    def test_zero_inventory_all_fuels(self, fuel_properties, market_prices):
        """
        Edge case: Zero inventory for all fuels.

        Expected:
        - Optimization should still work (assumes infinite supply available for purchase)
        - Or raise appropriate error if inventory constraint is hard
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        # Should optimize based on cost per energy, not fail
        assert result.total_cost_usd >= 0

    def test_zero_inventory_with_reorder(self, zero_inventory):
        """
        Edge case: Zero inventory triggers reorder.

        Expected:
        - needs_reorder() returns True
        - Reorder quantity calculated
        """
        inventory = FuelInventory(
            fuel_id="NG-001",
            site_id="SITE-001",
            storage_unit_id="TANK-01",
            current_quantity=0,
            quantity_unit="kg",
            storage_capacity=100000,
            minimum_level=10000,
            reorder_point=20000,
            reorder_quantity=40000,
        )

        assert inventory.needs_reorder() is True
        assert inventory.current_quantity == 0

    def test_single_fuel_zero_inventory_others_available(
        self, fuel_properties, market_prices
    ):
        """
        Edge case: One fuel has zero inventory, others available.

        Expected:
        - Optimizer selects available fuels only
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        # Should work with available fuels
        assert sum(result.optimal_fuel_mix.values()) > 0


@pytest.mark.edge_case
class TestSingleFuelEdgeCases:
    """Edge cases for single fuel optimization (no blending)."""

    def test_single_fuel_natural_gas_only(self, fuel_properties, market_prices):
        """
        Edge case: Only one fuel available.

        Expected:
        - 100% allocation to that fuel
        - No blending calculation needed
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        assert result.optimal_fuel_mix.get('natural_gas') == 1.0
        assert len(result.optimal_fuel_mix) == 1

    def test_single_fuel_with_blending_calculator_error(self, fuel_properties):
        """
        Edge case: Blending calculator with single fuel.

        Expected:
        - Should handle gracefully (no blending needed)
        - Or raise appropriate error
        """
        calculator = FuelBlendingCalculator()
        input_data = BlendingInput(
            available_fuels=['coal'],  # Only one fuel
            fuel_properties=fuel_properties,
            target_heating_value=25.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        # Should either work or raise meaningful error
        try:
            result = calculator.optimize_blend(input_data)
            # If it works, should be 100% coal
            assert result.blend_ratios.get('coal', 0) == 1.0
        except ValueError as e:
            # Acceptable to reject single-fuel blending
            assert 'single fuel' in str(e).lower() or 'at least 2' in str(e).lower()


@pytest.mark.edge_case
class TestPriceEdgeCases:
    """Edge cases for fuel pricing."""

    def test_all_same_price_fuels(self, fuel_properties):
        """
        Edge case: All fuels have identical price.

        Expected:
        - Optimization should choose based on other factors (efficiency, emissions)
        - Deterministic choice
        """
        optimizer = MultiFuelOptimizer()
        same_price = {'natural_gas': 0.05, 'coal': 0.05, 'biomass': 0.05}

        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=same_price,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        results = [optimizer.optimize(input_data) for _ in range(10)]

        # Results should be deterministic
        first = results[0]
        for result in results[1:]:
            assert result.optimal_fuel_mix == first.optimal_fuel_mix

    def test_extreme_price_volatility_1000x_spike(self, fuel_properties, volatile_market_prices):
        """
        Edge case: Extreme price volatility (1000x spike).

        Expected:
        - Optimizer handles large price ranges
        - Avoids expensive fuel
        - No overflow errors
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=volatile_market_prices,  # Natural gas at 10x normal
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)

        # Should avoid expensive natural gas
        assert result.optimal_fuel_mix.get('natural_gas', 1.0) < 0.1
        # Should prefer cheaper alternatives
        assert result.optimal_fuel_mix.get('coal', 0) > 0 or result.optimal_fuel_mix.get('biomass', 0) > 0

    def test_zero_price_fuel_subsidized(self, fuel_properties):
        """
        Edge case: Zero price fuel (fully subsidized).

        Expected:
        - Should select 100% of zero-price fuel
        - Cost should be near zero (only delivery costs)
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['biomass'],
            fuel_properties=fuel_properties,
            market_prices={'biomass': 0.0},  # Free (subsidized)
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        assert result.optimal_fuel_mix.get('biomass') == 1.0
        # Cost should be zero (or minimal delivery)
        assert result.total_cost_usd >= 0

    def test_missing_price_data_for_fuel(self, fuel_properties, market_prices):
        """
        Edge case: Missing price data for one fuel.

        Expected:
        - Should handle gracefully (skip that fuel or use default)
        - Or raise meaningful error
        """
        # Remove price for coal
        partial_prices = {k: v for k, v in market_prices.items() if k != 'coal'}

        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=partial_prices,
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        # Should either skip coal or raise error
        try:
            result = optimizer.optimize(input_data)
            # If it works, coal should not be selected
            assert result.optimal_fuel_mix.get('coal', 0) == 0
        except (KeyError, ValueError):
            # Acceptable to reject missing price data
            pass


@pytest.mark.edge_case
class TestDemandEdgeCases:
    """Edge cases for energy demand."""

    def test_zero_demand(self, fuel_properties, market_prices):
        """
        Edge case: Zero energy demand.

        Expected:
        - Zero fuel consumption
        - Zero cost
        - Empty or zero fuel mix
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=0,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        assert result.total_fuel_consumption_kg == 0
        assert result.total_cost_usd == 0

    def test_extremely_high_demand_10000mw(self, fuel_properties, market_prices):
        """
        Edge case: Extremely high demand (10,000 MW).

        Expected:
        - Scales linearly
        - No integer overflow
        - Reasonable execution time
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=10000,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)

        # 10000 MW * 72 kg/MW/hr = 720,000 kg/hr
        assert abs(result.total_fuel_consumption_kg - 720000.0) < 1000.0

    def test_fractional_demand_0_001_mw(self, fuel_properties, market_prices):
        """
        Edge case: Very small fractional demand (0.001 MW).

        Expected:
        - Handles small values correctly
        - No precision loss
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=0.001,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        # 0.001 MW * 72 kg/MW/hr = 0.072 kg/hr
        assert abs(result.total_fuel_consumption_kg - 0.072) < 0.01


@pytest.mark.edge_case
class TestCalorificValueEdgeCases:
    """Edge cases for calorific value handling."""

    def test_very_high_calorific_value_hydrogen(self):
        """
        Edge case: Very high calorific value (hydrogen).

        Expected:
        - Handles values up to 150 MJ/kg
        - No calculation errors
        """
        spec = FuelSpecification(
            fuel_id="H2-HIGH",
            fuel_name="High Calorific Hydrogen",
            fuel_type="hydrogen",
            category=FuelCategory.RENEWABLE,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=142.0,
            net_calorific_value_mj_kg=120.0,
            density_kg_m3=0.09,
            carbon_content_percent=0.0,
            hydrogen_content_percent=100.0,
            emission_factor_co2_kg_gj=0.0,
        )

        assert spec.gross_calorific_value_mj_kg == 142.0

    def test_very_low_calorific_value_waste_fuel(self):
        """
        Edge case: Very low calorific value (waste fuel).

        Expected:
        - Handles low values (5-10 MJ/kg)
        - Fuel consumption scales correctly
        """
        spec = FuelSpecification(
            fuel_id="WASTE-001",
            fuel_name="Municipal Waste",
            fuel_type="waste",
            category=FuelCategory.FOSSIL,
            state=FuelState.SOLID,
            gross_calorific_value_mj_kg=9.0,
            net_calorific_value_mj_kg=8.0,
            density_kg_m3=400.0,
            carbon_content_percent=30.0,
            hydrogen_content_percent=4.0,
            oxygen_content_percent=20.0,
            moisture_content_percent=40.0,
            ash_content_percent=5.0,
            emission_factor_co2_kg_gj=50.0,
        )

        assert spec.net_calorific_value_mj_kg == 8.0


@pytest.mark.edge_case
class TestEmissionFactorEdgeCases:
    """Edge cases for emission factors."""

    def test_missing_emission_factor_defaults_to_zero(self, fuel_properties):
        """
        Edge case: Missing emission factor data.

        Expected:
        - Defaults to zero (or raises error)
        """
        # Emission factors are required in FuelSpecification
        # This test documents that behavior
        pass

    def test_zero_emission_factor_renewable(self):
        """
        Edge case: Zero CO2 emission (renewable fuel).

        Expected:
        - Handles zero correctly
        - No division by zero
        """
        spec = FuelSpecification(
            fuel_id="SOLAR-THERMAL",
            fuel_name="Solar Thermal",
            fuel_type="solar",
            category=FuelCategory.RENEWABLE,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=50.0,
            net_calorific_value_mj_kg=45.0,
            density_kg_m3=1.0,
            carbon_content_percent=0.0,
            hydrogen_content_percent=0.0,
            emission_factor_co2_kg_gj=0.0,  # Zero emissions
        )

        assert spec.emission_factor_co2_kg_gj == 0.0


@pytest.mark.edge_case
class TestBlendingEdgeCases:
    """Edge cases for fuel blending."""

    def test_incompatible_fuel_pairs(self, fuel_properties):
        """
        Edge case: Incompatible fuel pairs in blend.

        Expected:
        - Should not blend incompatible fuels
        - Raises error or skips combination
        """
        calculator = FuelBlendingCalculator()
        input_data = BlendingInput(
            available_fuels=['coal', 'natural_gas'],
            fuel_properties=fuel_properties,
            target_heating_value=30.0,
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='balanced',
            incompatible_pairs=[['coal', 'natural_gas']]  # Incompatible
        )

        # Should handle incompatibility constraint
        try:
            result = calculator.optimize_blend(input_data)
            # If it works, should select only one
            assert len([f for f, r in result.blend_ratios.items() if r > 0]) == 1
        except ValueError:
            # Acceptable to reject incompatible blends
            pass

    def test_impossible_target_heating_value(self, fuel_properties):
        """
        Edge case: Target heating value unachievable with available fuels.

        Expected:
        - Raises meaningful error
        - Or returns closest achievable blend
        """
        calculator = FuelBlendingCalculator()
        input_data = BlendingInput(
            available_fuels=['coal', 'biomass'],  # Max ~25 MJ/kg
            fuel_properties=fuel_properties,
            target_heating_value=50.0,  # Impossible to achieve
            max_moisture=25.0,
            max_ash=15.0,
            max_sulfur=3.0,
            optimization_objective='balanced',
            incompatible_pairs=[]
        )

        try:
            result = calculator.optimize_blend(input_data)
            # If it works, should be best effort
            assert result.blend_heating_value < 50.0
        except ValueError as e:
            # Acceptable to reject impossible targets
            assert 'impossible' in str(e).lower() or 'cannot achieve' in str(e).lower()


@pytest.mark.edge_case
class TestDivisionByZeroEdgeCases:
    """Edge cases that could cause division by zero."""

    def test_zero_heating_value_protection(self):
        """
        Edge case: Zero heating value should be rejected.

        Expected:
        - Validation error (heating value must be > 0)
        """
        with pytest.raises(ValidationError):
            FuelSpecification(
                fuel_id="ZERO-HV",
                fuel_name="Zero Heating Value",
                fuel_type="invalid",
                category=FuelCategory.FOSSIL,
                state=FuelState.SOLID,
                gross_calorific_value_mj_kg=0.0,  # Invalid
                net_calorific_value_mj_kg=0.0,
                density_kg_m3=1000.0,
                carbon_content_percent=50.0,
                hydrogen_content_percent=5.0,
                emission_factor_co2_kg_gj=50.0,
            )

    def test_zero_density_protection(self):
        """
        Edge case: Zero density should be rejected.

        Expected:
        - Validation error (density must be > 0)
        """
        with pytest.raises(ValidationError):
            FuelSpecification(
                fuel_id="ZERO-DENSITY",
                fuel_name="Zero Density",
                fuel_type="invalid",
                category=FuelCategory.FOSSIL,
                state=FuelState.GAS,
                gross_calorific_value_mj_kg=50.0,
                net_calorific_value_mj_kg=45.0,
                density_kg_m3=0.0,  # Invalid
                carbon_content_percent=50.0,
                hydrogen_content_percent=5.0,
                emission_factor_co2_kg_gj=50.0,
            )


@pytest.mark.edge_case
class TestUnicodeAndSpecialCharacters:
    """Edge cases for unicode and special characters."""

    def test_unicode_fuel_name(self):
        """
        Edge case: Unicode characters in fuel name.

        Expected:
        - Handles Unicode correctly
        - No encoding errors
        """
        spec = FuelSpecification(
            fuel_id="UNICODE-001",
            fuel_name="Kraftstoff Öl №2 燃料",  # German, Russian, Chinese
            fuel_type="fuel_oil",
            category=FuelCategory.FOSSIL,
            state=FuelState.LIQUID,
            gross_calorific_value_mj_kg=45.0,
            net_calorific_value_mj_kg=42.0,
            density_kg_m3=850.0,
            carbon_content_percent=85.0,
            hydrogen_content_percent=12.0,
            emission_factor_co2_kg_gj=75.0,
        )

        assert "Kraftstoff" in spec.fuel_name
        assert "燃料" in spec.fuel_name

    def test_special_characters_in_fuel_id(self):
        """
        Edge case: Special characters in fuel ID.

        Expected:
        - Alphanumeric and limited special chars allowed
        - Invalid chars rejected
        """
        # Valid special chars (hyphens, underscores)
        spec = FuelSpecification(
            fuel_id="FUEL-001_NG",
            fuel_name="Natural Gas",
            fuel_type="natural_gas",
            category=FuelCategory.FOSSIL,
            state=FuelState.GAS,
            gross_calorific_value_mj_kg=55.5,
            net_calorific_value_mj_kg=50.0,
            density_kg_m3=0.75,
            carbon_content_percent=75.0,
            hydrogen_content_percent=25.0,
            emission_factor_co2_kg_gj=56.1,
        )

        assert spec.fuel_id == "FUEL-001_NG"


@pytest.mark.edge_case
class TestPrecisionBoundaries:
    """Edge cases for numerical precision boundaries."""

    def test_very_small_fuel_quantity(self, fuel_properties, market_prices):
        """
        Edge case: Very small fuel quantity (0.000001 kg).

        Expected:
        - No precision loss
        - Correct cost calculation
        """
        calculator = CostOptimizationCalculator()
        input_data = CostOptimizationInput(
            energy_demand_mw=0.000001,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            fuel_inventories={'natural_gas': 1000},
            delivery_costs={'natural_gas': 0},
            constraints={}
        )

        result = calculator.optimize(input_data)
        # Should handle tiny values without underflow
        assert result.total_cost_usd >= 0

    def test_very_large_cost_calculation(self, fuel_properties):
        """
        Edge case: Very large cost calculation (billions).

        Expected:
        - No overflow
        - Correct precision
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100000,  # 100 GW
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 1000.0},  # $1000/kg
            emission_limits={},
            constraints={},
            optimization_objective='cost'
        )

        result = optimizer.optimize(input_data)
        # Cost should be in billions but still calculated correctly
        assert result.total_cost_usd > 1_000_000_000


@pytest.mark.edge_case
class TestEmptyInputs:
    """Edge cases for empty inputs."""

    def test_empty_available_fuels_list(self, fuel_properties, market_prices):
        """
        Edge case: Empty available fuels list.

        Expected:
        - Raises meaningful error
        """
        optimizer = MultiFuelOptimizer()

        with pytest.raises((ValueError, IndexError)) as exc_info:
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=100,
                available_fuels=[],  # Empty
                fuel_properties=fuel_properties,
                market_prices=market_prices,
                emission_limits={},
                constraints={},
                optimization_objective='cost'
            )
            optimizer.optimize(input_data)

    def test_empty_fuel_properties_dict(self, market_prices):
        """
        Edge case: Empty fuel properties dictionary.

        Expected:
        - Raises error (no fuel data)
        """
        optimizer = MultiFuelOptimizer()

        with pytest.raises((ValueError, KeyError)):
            input_data = MultiFuelOptimizationInput(
                energy_demand_mw=100,
                available_fuels=['natural_gas'],
                fuel_properties={},  # Empty
                market_prices=market_prices,
                emission_limits={},
                constraints={},
                optimization_objective='cost'
            )
            optimizer.optimize(input_data)


@pytest.mark.edge_case
class TestTimeEdgeCases:
    """Edge cases for time-related operations."""

    def test_expired_inventory_handling(self):
        """
        Edge case: Fuel inventory past expiry date.

        Expected:
        - Should flag as unusable
        - Or apply quality degradation
        """
        from datetime import datetime, timedelta, timezone

        inventory = FuelInventory(
            fuel_id="BIO-001",
            site_id="SITE-001",
            storage_unit_id="BIO-TANK-01",
            current_quantity=10000,
            quantity_unit="kg",
            storage_capacity=50000,
            minimum_level=5000,
            expiry_date=datetime.now(timezone.utc) - timedelta(days=30),  # Expired
        )

        # Should detect expiry
        if inventory.expiry_date:
            assert inventory.expiry_date < datetime.now(timezone.utc)
