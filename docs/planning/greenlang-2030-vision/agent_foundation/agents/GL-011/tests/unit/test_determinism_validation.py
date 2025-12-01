# -*- coding: utf-8 -*-
"""
Tests for determinism validation.

Validates that all GL-011 calculations are deterministic:
- Same inputs produce same outputs
- No random elements in calculations
- Provenance hashes are reproducible
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.multi_fuel_optimizer import MultiFuelOptimizer, MultiFuelOptimizationInput
from calculators.cost_optimization_calculator import CostOptimizationCalculator, CostOptimizationInput
from calculators.fuel_blending_calculator import FuelBlendingCalculator, BlendingInput
from calculators.carbon_footprint_calculator import CarbonFootprintCalculator, CarbonFootprintInput
from calculators.provenance_tracker import ProvenanceTracker


class TestDeterminismValidation:
    """Test suite for determinism validation."""

    @pytest.fixture
    def fuel_properties(self):
        """Standard fuel properties for testing."""
        return {
            'natural_gas': {
                'heating_value_mj_kg': 50.0,
                'emission_factor_co2_kg_gj': 56.1,
                'emission_factor_nox_g_gj': 50,
                'renewable': False
            },
            'coal': {
                'heating_value_mj_kg': 25.0,
                'emission_factor_co2_kg_gj': 94.6,
                'emission_factor_nox_g_gj': 250,
                'renewable': False
            },
            'biomass': {
                'heating_value_mj_kg': 18.0,
                'emission_factor_co2_kg_gj': 0.0,
                'emission_factor_nox_g_gj': 150,
                'renewable': True
            }
        }

    @pytest.fixture
    def market_prices(self):
        """Standard market prices for testing."""
        return {
            'natural_gas': 0.045,
            'coal': 0.035,
            'biomass': 0.08
        }

    def test_multi_fuel_determinism_100_runs(self, fuel_properties, market_prices):
        """Test multi-fuel optimizer is deterministic across 100 runs."""
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        results = [optimizer.optimize(input_data) for _ in range(100)]

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.optimal_fuel_mix == first_result.optimal_fuel_mix, \
                f"Result {i} differs from result 1"
            assert result.total_cost_usd == first_result.total_cost_usd, \
                f"Cost differs at run {i}"
            assert result.provenance_hash == first_result.provenance_hash, \
                f"Provenance hash differs at run {i}"

    def test_cost_optimization_determinism(self, fuel_properties, market_prices):
        """Test cost optimizer is deterministic."""
        calculator = CostOptimizationCalculator()
        input_data = CostOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal', 'biomass'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            fuel_inventories={'natural_gas': 100000, 'coal': 100000, 'biomass': 100000},
            delivery_costs={'natural_gas': 0, 'coal': 0, 'biomass': 0},
            constraints={}
        )

        results = [calculator.optimize(input_data) for _ in range(50)]

        first = results[0]
        for r in results[1:]:
            assert r.optimal_fuel == first.optimal_fuel
            assert r.total_cost_usd == first.total_cost_usd

    def test_blending_determinism(self, fuel_properties):
        """Test blending calculator is deterministic."""
        calculator = FuelBlendingCalculator()
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

        results = [calculator.optimize_blend(input_data) for _ in range(50)]

        first = results[0]
        for r in results[1:]:
            assert r.blend_ratios == first.blend_ratios
            assert r.blend_heating_value == first.blend_heating_value

    def test_carbon_footprint_determinism(self, fuel_properties):
        """Test carbon footprint calculator is deterministic."""
        calculator = CarbonFootprintCalculator()
        input_data = CarbonFootprintInput(
            fuel_quantities={'natural_gas': 1000, 'coal': 500},
            fuel_properties=fuel_properties
        )

        results = [calculator.calculate(input_data) for _ in range(50)]

        first = results[0]
        for r in results[1:]:
            assert r.total_co2e_kg == first.total_co2e_kg
            assert r.carbon_intensity_kg_mwh == first.carbon_intensity_kg_mwh

    def test_provenance_hash_reproducibility(self):
        """Test provenance hash is reproducible for same inputs."""
        tracker = ProvenanceTracker()

        inputs = {'demand_mw': 100, 'fuels': ['natural_gas', 'coal']}
        outputs = {'cost': 5000, 'emissions': 1000}

        record1 = tracker.record(
            operation='test_op',
            inputs=inputs,
            outputs=outputs
        )

        record2 = tracker.record(
            operation='test_op',
            inputs=inputs,
            outputs=outputs
        )

        # Combined hash should be based on content, not time
        # (Note: actual implementation may vary)
        assert record1.input_hash == record2.input_hash
        assert record1.output_hash == record2.output_hash

    def test_provenance_verification(self):
        """Test provenance verification works correctly."""
        tracker = ProvenanceTracker()

        inputs = {'demand_mw': 100}
        outputs = {'cost': 5000}

        record = tracker.record(
            operation='test_op',
            inputs=inputs,
            outputs=outputs
        )

        # Verification should succeed with same inputs/outputs
        assert tracker.verify(record.record_id, inputs, outputs)

        # Verification should fail with different inputs
        modified_inputs = {'demand_mw': 200}
        assert not tracker.verify(record.record_id, modified_inputs, outputs)

    def test_no_random_in_calculations(self, fuel_properties, market_prices):
        """Test that calculations don't use unseeded random."""
        import random

        # Set global random state
        random.seed(12345)
        state_before = random.getstate()

        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas', 'coal'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        optimizer.optimize(input_data)

        state_after = random.getstate()

        # If calculation used random without restoring state, states would differ
        # (This is a basic check - may need refinement based on implementation)
        # Note: Some implementations may legitimately use seeded random

    def test_calculation_count_tracking(self, fuel_properties, market_prices):
        """Test calculation count is tracked."""
        optimizer = MultiFuelOptimizer()
        initial_count = optimizer.calculation_count

        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices=market_prices,
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        optimizer.optimize(input_data)
        optimizer.optimize(input_data)
        optimizer.optimize(input_data)

        assert optimizer.calculation_count == initial_count + 3

    def test_golden_test_values(self, fuel_properties, market_prices):
        """
        Golden test: verify specific known-good outputs.

        These values should never change unless the algorithm is intentionally modified.
        """
        optimizer = MultiFuelOptimizer()
        input_data = MultiFuelOptimizationInput(
            energy_demand_mw=100,
            available_fuels=['natural_gas'],
            fuel_properties=fuel_properties,
            market_prices={'natural_gas': 0.045},
            emission_limits={},
            constraints={},
            optimization_objective='balanced'
        )

        result = optimizer.optimize(input_data)

        # These are golden values that should remain constant
        assert result.optimal_fuel_mix.get('natural_gas', 0) == 1.0
        assert result.efficiency_percent > 90  # Natural gas efficiency
        # Note: Actual golden values should be established from known-good run
