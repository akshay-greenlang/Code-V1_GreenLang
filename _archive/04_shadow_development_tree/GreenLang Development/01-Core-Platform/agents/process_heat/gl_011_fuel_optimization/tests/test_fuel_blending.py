"""
GL-011 FUELCRAFT - Fuel Blending Optimizer Tests

Unit tests for FuelBlendingOptimizer including blend optimization,
Wobbe Index matching, cost optimization, and constraint satisfaction.
"""

import pytest
from datetime import datetime, timezone, timedelta

from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_blending import (
    FuelBlendingOptimizer,
    BlendInput,
    BlendOutput,
    BlendConstraints,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    HeatingValueCalculator,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelProperties,
    FuelPrice,
    BlendStatus,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    BlendingConfig,
)


class TestFuelBlendingOptimizer:
    """Tests for FuelBlendingOptimizer class."""

    def test_optimizer_initialization(self, fuel_blending_optimizer):
        """Test optimizer initialization."""
        assert fuel_blending_optimizer.config is not None
        assert fuel_blending_optimizer._hv_calculator is not None
        assert fuel_blending_optimizer.optimization_count == 0

    def test_optimizer_increments_count(self, fuel_blending_optimizer, blend_input):
        """Test optimization count increments."""
        initial = fuel_blending_optimizer.optimization_count

        fuel_blending_optimizer.optimize_blend(blend_input)

        assert fuel_blending_optimizer.optimization_count == initial + 1


class TestSingleFuelBlend:
    """Tests for single-fuel (100%) blends."""

    def test_single_fuel_blend(self, fuel_blending_optimizer, natural_gas_properties, natural_gas_price):
        """Test single fuel produces 100% blend."""
        input_data = BlendInput(
            available_fuels=["natural_gas"],
            fuel_properties={"natural_gas": natural_gas_properties},
            fuel_prices={"natural_gas": natural_gas_price},
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        assert result.blend_ratios["natural_gas"] == 100.0
        assert sum(result.blend_ratios.values()) == 100.0

    def test_single_fuel_wobbe_unchanged(self, fuel_blending_optimizer, natural_gas_properties, natural_gas_price):
        """Test single fuel Wobbe equals input Wobbe."""
        input_data = BlendInput(
            available_fuels=["natural_gas"],
            fuel_properties={"natural_gas": natural_gas_properties},
            fuel_prices={"natural_gas": natural_gas_price},
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        assert result.blended_wobbe == pytest.approx(natural_gas_properties.wobbe_index, rel=0.01)


class TestTwoFuelBlend:
    """Tests for two-fuel blending."""

    def test_two_fuel_cost_optimization(
        self,
        fuel_blending_optimizer,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test two-fuel blend optimizes for cost."""
        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        # Since natural gas is cheaper, it should dominate the blend
        assert result.blend_ratios["natural_gas"] >= 50.0
        assert sum(result.blend_ratios.values()) == pytest.approx(100.0, rel=0.01)

    def test_two_fuel_wobbe_in_range(
        self,
        fuel_blending_optimizer,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test two-fuel blend Wobbe is within range."""
        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        min_wobbe = fuel_blending_optimizer.config.min_wobbe_index
        max_wobbe = fuel_blending_optimizer.config.max_wobbe_index

        # Blended Wobbe should be in range (or as close as possible)
        if result.status == BlendStatus.OPTIMAL:
            assert min_wobbe <= result.blended_wobbe <= max_wobbe


class TestWobbeIndexConstraints:
    """Tests for Wobbe Index constraint handling."""

    def test_wobbe_target_optimization(self, blending_config, heating_value_calculator, all_fuel_properties, all_fuel_prices):
        """Test optimization targets Wobbe range."""
        config = BlendingConfig(
            min_wobbe_index=1340.0,
            max_wobbe_index=1360.0,
            wobbe_tolerance_pct=2.0,
        )

        optimizer = FuelBlendingOptimizer(config, heating_value_calculator)

        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            target_wobbe_index=1350.0,
            required_heat_input_mmbtu_hr=50.0,
        )

        result = optimizer.optimize_blend(input_data)

        # Should be close to target
        tolerance = 1350.0 * config.wobbe_tolerance_pct / 100.0
        assert abs(result.blended_wobbe - 1350.0) <= tolerance

    def test_wobbe_infeasible_marked(self, blending_config, heating_value_calculator, biogas_properties, natural_gas_price):
        """Test infeasible Wobbe blend is marked."""
        config = BlendingConfig(
            min_wobbe_index=1300.0,
            max_wobbe_index=1400.0,
        )

        optimizer = FuelBlendingOptimizer(config, heating_value_calculator)

        # Biogas alone cannot meet Wobbe requirements
        input_data = BlendInput(
            available_fuels=["biogas"],
            fuel_properties={"biogas": biogas_properties},
            fuel_prices={"biogas": natural_gas_price},  # Use any price
            required_heat_input_mmbtu_hr=50.0,
        )

        result = optimizer.optimize_blend(input_data)

        # Should indicate Wobbe not in range
        assert result.wobbe_in_range is False


class TestEmissionsConstraints:
    """Tests for emissions constraint handling."""

    def test_emissions_constraint(
        self,
        fuel_blending_optimizer,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test blend respects emissions constraint."""
        input_data = BlendInput(
            available_fuels=["natural_gas", "no2_fuel_oil"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "no2_fuel_oil": all_fuel_properties["no2_fuel_oil"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "no2_fuel_oil": all_fuel_prices["no2_fuel_oil"],
            },
            max_co2_kg_hr=3000.0,  # Constraint
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        # If constraint met, CO2 should be below limit
        if result.emissions_in_range:
            assert result.blended_co2_kg_hr <= 3000.0

    def test_low_carbon_preference(
        self,
        blending_config,
        heating_value_calculator,
        natural_gas_properties,
        hydrogen_properties,
        natural_gas_price,
    ):
        """Test preference for low-carbon fuels with tight constraint."""
        # Create hydrogen price
        h2_price = FuelPrice(
            fuel_type="hydrogen",
            price=15.00,
            commodity_price=14.00,
            source="test",
        )

        input_data = BlendInput(
            available_fuels=["natural_gas", "hydrogen"],
            fuel_properties={
                "natural_gas": natural_gas_properties,
                "hydrogen": hydrogen_properties,
            },
            fuel_prices={
                "natural_gas": natural_gas_price,
                "hydrogen": h2_price,
            },
            max_co2_kg_hr=1500.0,  # Very tight constraint
            required_heat_input_mmbtu_hr=50.0,
        )

        optimizer = FuelBlendingOptimizer(blending_config, heating_value_calculator)
        result = optimizer.optimize_blend(input_data)

        # Should include hydrogen to meet emissions target
        if "hydrogen" in result.blend_ratios:
            assert result.blend_ratios["hydrogen"] > 0


class TestMinimumPrimaryFuel:
    """Tests for minimum primary fuel constraint."""

    def test_minimum_primary_fuel_respected(
        self,
        blending_config,
        heating_value_calculator,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test minimum primary fuel percentage is respected."""
        config = BlendingConfig(
            min_primary_fuel_pct=60.0,
        )

        optimizer = FuelBlendingOptimizer(config, heating_value_calculator)

        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            primary_fuel="natural_gas",
            required_heat_input_mmbtu_hr=50.0,
        )

        result = optimizer.optimize_blend(input_data)

        # Primary fuel should be at least 60%
        assert result.blend_ratios["natural_gas"] >= 60.0


class TestMaxFuelsInBlend:
    """Tests for maximum fuels in blend constraint."""

    def test_max_fuels_respected(
        self,
        blending_config,
        heating_value_calculator,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test maximum number of fuels is respected."""
        config = BlendingConfig(max_fuels_in_blend=2)

        optimizer = FuelBlendingOptimizer(config, heating_value_calculator)

        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane", "no2_fuel_oil"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
                "no2_fuel_oil": all_fuel_properties["no2_fuel_oil"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
                "no2_fuel_oil": all_fuel_prices["no2_fuel_oil"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        result = optimizer.optimize_blend(input_data)

        # Count non-zero fuels
        active_fuels = sum(1 for pct in result.blend_ratios.values() if pct > 0)
        assert active_fuels <= 2


class TestCurrentBlendTransition:
    """Tests for blend transition from current blend."""

    def test_transition_from_current_blend(
        self,
        blending_config,
        heating_value_calculator,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test transition from current blend."""
        config = BlendingConfig(max_blend_change_rate_pct_min=5.0)

        optimizer = FuelBlendingOptimizer(config, heating_value_calculator)

        current_blend = {"natural_gas": 80.0, "lpg_propane": 20.0}

        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            current_blend=current_blend,
            required_heat_input_mmbtu_hr=50.0,
        )

        result = optimizer.optimize_blend(input_data)

        # Transition time should be calculated
        assert result.transition_time_minutes is not None


class TestBlendCostCalculation:
    """Tests for blend cost calculation."""

    def test_blend_cost_weighted_average(
        self,
        fuel_blending_optimizer,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test blend cost is weighted average."""
        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        # Calculate expected cost
        ng_pct = result.blend_ratios.get("natural_gas", 0) / 100.0
        lpg_pct = result.blend_ratios.get("lpg_propane", 0) / 100.0

        expected_cost = (
            ng_pct * all_fuel_prices["natural_gas"].price +
            lpg_pct * all_fuel_prices["lpg_propane"].price
        )

        assert result.blended_cost_usd_mmbtu == pytest.approx(expected_cost, rel=0.01)


class TestBlendHHVCalculation:
    """Tests for blended HHV calculation."""

    def test_blend_hhv_weighted_average(
        self,
        fuel_blending_optimizer,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test blended HHV is correctly calculated."""
        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        result = fuel_blending_optimizer.optimize_blend(input_data)

        # HHV should be between the two fuels
        ng_hhv = all_fuel_properties["natural_gas"].hhv_btu_scf
        lpg_hhv = all_fuel_properties["lpg_propane"].hhv_btu_scf

        assert min(ng_hhv, lpg_hhv) <= result.blended_hhv <= max(ng_hhv, lpg_hhv)


class TestBlendRecommendationCreation:
    """Tests for BlendRecommendation creation."""

    def test_create_recommendation(
        self,
        fuel_blending_optimizer,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test creating blend recommendation."""
        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        output = fuel_blending_optimizer.optimize_blend(input_data)

        current_blend = {"natural_gas": 100.0}
        recommendation = fuel_blending_optimizer.create_blend_recommendation(
            output,
            current_blend,
        )

        assert recommendation.blend_ratios == output.blend_ratios
        assert recommendation.blended_hhv == output.blended_hhv
        assert recommendation.blended_cost_usd_mmbtu == output.blended_cost_usd_mmbtu


class TestBlendStatus:
    """Tests for blend status determination."""

    def test_optimal_status(self, fuel_blending_optimizer, blend_input):
        """Test optimal status when constraints met."""
        result = fuel_blending_optimizer.optimize_blend(blend_input)

        # Should be optimal if all constraints met
        if result.wobbe_in_range and result.hhv_in_range and result.emissions_in_range:
            assert result.status == BlendStatus.OPTIMAL

    def test_suboptimal_status(
        self,
        blending_config,
        heating_value_calculator,
        all_fuel_properties,
        all_fuel_prices,
    ):
        """Test sub-optimal status when constraints partially met."""
        # Create strict constraints that may not be fully satisfiable
        config = BlendingConfig(
            min_wobbe_index=1310.0,
            max_wobbe_index=1320.0,  # Very narrow range
        )

        optimizer = FuelBlendingOptimizer(config, heating_value_calculator)

        input_data = BlendInput(
            available_fuels=["natural_gas", "lpg_propane"],
            fuel_properties={
                "natural_gas": all_fuel_properties["natural_gas"],
                "lpg_propane": all_fuel_properties["lpg_propane"],
            },
            fuel_prices={
                "natural_gas": all_fuel_prices["natural_gas"],
                "lpg_propane": all_fuel_prices["lpg_propane"],
            },
            required_heat_input_mmbtu_hr=50.0,
        )

        result = optimizer.optimize_blend(input_data)

        # May be sub-optimal if Wobbe can't be exactly matched
        assert result.status in [BlendStatus.OPTIMAL, BlendStatus.SUB_OPTIMAL]


class TestProvenanceTracking:
    """Tests for provenance tracking."""

    def test_provenance_hash_generated(self, fuel_blending_optimizer, blend_input):
        """Test provenance hash is generated."""
        result = fuel_blending_optimizer.optimize_blend(blend_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(self, fuel_blending_optimizer, blend_input):
        """Test same input produces same hash."""
        result1 = fuel_blending_optimizer.optimize_blend(blend_input)
        result2 = fuel_blending_optimizer.optimize_blend(blend_input)

        assert result1.provenance_hash == result2.provenance_hash


class TestBlendConstraintsClass:
    """Tests for BlendConstraints class."""

    def test_constraints_from_config(self, blending_config):
        """Test creating constraints from config."""
        constraints = BlendConstraints.from_config(blending_config)

        assert constraints.min_wobbe == blending_config.min_wobbe_index
        assert constraints.max_wobbe == blending_config.max_wobbe_index
        assert constraints.min_hhv == blending_config.min_hhv_btu_scf
        assert constraints.max_hhv == blending_config.max_hhv_btu_scf

    def test_constraints_with_overrides(self, blending_config):
        """Test constraints with manual overrides."""
        constraints = BlendConstraints(
            min_wobbe=1300.0,
            max_wobbe=1400.0,
            max_co2_kg_hr=2500.0,
        )

        assert constraints.min_wobbe == 1300.0
        assert constraints.max_co2_kg_hr == 2500.0
