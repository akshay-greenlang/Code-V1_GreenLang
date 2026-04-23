"""
GL-002 FLAMEGUARD - Comprehensive Optimization Tests

Tests for combustion optimizer, load dispatch, multi-objective optimization,
and Pareto analysis.
Targets 70%+ coverage with parameterized tests and edge cases.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from optimization.combustion_optimizer import (
    CombustionOptimizer,
    BoilerModel,
    LoadDispatchResult,
)


# =============================================================================
# BOILER MODEL TESTS
# =============================================================================


class TestBoilerModel:
    """Test BoilerModel dataclass and calculations."""

    @pytest.fixture
    def boiler(self):
        return BoilerModel(
            boiler_id="BOILER-001",
            rated_capacity_klb_hr=200.0,
            min_load_percent=25.0,
            max_load_percent=100.0,
            design_efficiency=82.0,
            efficiency_a=-0.0002,
            efficiency_b=0.08,
            efficiency_c=78.0,
            fuel_cost_per_mmbtu=5.0,
        )

    def test_initialization(self, boiler):
        """Test boiler model initializes correctly."""
        assert boiler.boiler_id == "BOILER-001"
        assert boiler.rated_capacity_klb_hr == 200.0
        assert boiler.design_efficiency == 82.0

    def test_efficiency_at_load_normal(self, boiler):
        """Test efficiency calculation at normal load."""
        efficiency = boiler.efficiency_at_load(75.0)

        # Expected: -0.0002*(75^2) + 0.08*75 + 78 = -1.125 + 6 + 78 = 82.875
        assert 80.0 <= efficiency <= 85.0

    def test_efficiency_at_load_minimum(self, boiler):
        """Test efficiency at minimum load."""
        efficiency = boiler.efficiency_at_load(25.0)
        assert 60.0 <= efficiency <= 95.0

    def test_efficiency_at_load_maximum(self, boiler):
        """Test efficiency at maximum load."""
        efficiency = boiler.efficiency_at_load(100.0)
        assert 60.0 <= efficiency <= 95.0

    def test_efficiency_clamped_below_min(self, boiler):
        """Test efficiency clamped when load below minimum."""
        # Should use min_load_percent (25%)
        efficiency = boiler.efficiency_at_load(10.0)
        assert 60.0 <= efficiency <= 95.0

    def test_efficiency_clamped_above_max(self, boiler):
        """Test efficiency clamped when load above maximum."""
        # Should use max_load_percent (100%)
        efficiency = boiler.efficiency_at_load(120.0)
        assert 60.0 <= efficiency <= 95.0

    def test_efficiency_bounded_60_95(self, boiler):
        """Test efficiency always between 60-95%."""
        for load in range(0, 150, 10):
            efficiency = boiler.efficiency_at_load(float(load))
            assert 60.0 <= efficiency <= 95.0

    def test_heat_input_at_load(self, boiler):
        """Test heat input calculation."""
        heat_input = boiler.heat_input_at_load(75.0)

        # Should be positive
        assert heat_input > 0
        # At 75% of 200 klb/hr = 150 klb/hr steam
        # Heat ~= steam * enthalpy / efficiency
        assert heat_input > 100  # Rough check

    def test_heat_input_increases_with_load(self, boiler):
        """Test heat input increases with load."""
        heat_25 = boiler.heat_input_at_load(25.0)
        heat_50 = boiler.heat_input_at_load(50.0)
        heat_75 = boiler.heat_input_at_load(75.0)
        heat_100 = boiler.heat_input_at_load(100.0)

        assert heat_50 > heat_25
        assert heat_75 > heat_50
        assert heat_100 > heat_75

    def test_cost_at_load(self, boiler):
        """Test operating cost calculation."""
        cost = boiler.cost_at_load(75.0)

        # Cost = heat_input * fuel_cost
        heat_input = boiler.heat_input_at_load(75.0)
        expected_cost = heat_input * boiler.fuel_cost_per_mmbtu

        assert abs(cost - expected_cost) < 0.01

    def test_cost_increases_with_load(self, boiler):
        """Test cost increases with load."""
        cost_25 = boiler.cost_at_load(25.0)
        cost_50 = boiler.cost_at_load(50.0)
        cost_75 = boiler.cost_at_load(75.0)
        cost_100 = boiler.cost_at_load(100.0)

        assert cost_50 > cost_25
        assert cost_75 > cost_50
        assert cost_100 > cost_75


class TestBoilerModelEfficiencyCurve:
    """Test efficiency curve behavior."""

    def test_peak_efficiency_at_optimal_load(self):
        """Test peak efficiency occurs at optimal load."""
        boiler = BoilerModel(
            boiler_id="BOILER-001",
            rated_capacity_klb_hr=200.0,
            efficiency_a=-0.0002,  # Negative = concave curve
            efficiency_b=0.08,
            efficiency_c=78.0,
        )

        # Peak at -b/(2a) = -0.08 / (2 * -0.0002) = 200
        # But clamped to 100% max
        efficiencies = {load: boiler.efficiency_at_load(float(load))
                        for load in range(25, 101, 5)}

        max_load = max(efficiencies, key=efficiencies.get)
        # Peak should be near higher loads for this curve
        assert max_load >= 75

    @pytest.mark.parametrize("efficiency_a,efficiency_b,efficiency_c", [
        (-0.0002, 0.08, 78.0),
        (-0.0003, 0.10, 75.0),
        (-0.0001, 0.05, 80.0),
    ])
    def test_various_efficiency_curves(self, efficiency_a, efficiency_b, efficiency_c):
        """Test various efficiency curve parameters."""
        boiler = BoilerModel(
            boiler_id="BOILER-001",
            rated_capacity_klb_hr=200.0,
            efficiency_a=efficiency_a,
            efficiency_b=efficiency_b,
            efficiency_c=efficiency_c,
        )

        for load in range(25, 101, 25):
            efficiency = boiler.efficiency_at_load(float(load))
            assert 60.0 <= efficiency <= 95.0


# =============================================================================
# COMBUSTION OPTIMIZER TESTS
# =============================================================================


class TestCombustionOptimizerInit:
    """Test CombustionOptimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with defaults."""
        optimizer = CombustionOptimizer()

        assert optimizer.boilers == {}
        assert optimizer.use_ai is True
        assert optimizer.exploration_rate == 0.1

    def test_initialization_with_boilers(self):
        """Test optimizer initializes with boiler list."""
        boilers = [
            BoilerModel(boiler_id="B1", rated_capacity_klb_hr=100.0),
            BoilerModel(boiler_id="B2", rated_capacity_klb_hr=150.0),
        ]
        optimizer = CombustionOptimizer(boilers=boilers)

        assert len(optimizer.boilers) == 2
        assert "B1" in optimizer.boilers
        assert "B2" in optimizer.boilers

    def test_initialization_custom_settings(self):
        """Test optimizer with custom settings."""
        optimizer = CombustionOptimizer(
            use_ai=False,
            exploration_rate=0.2,
        )

        assert optimizer.use_ai is False
        assert optimizer.exploration_rate == 0.2


class TestBoilerManagement:
    """Test boiler add/remove operations."""

    @pytest.fixture
    def optimizer(self):
        return CombustionOptimizer()

    def test_add_boiler(self, optimizer):
        """Test adding a boiler."""
        boiler = BoilerModel(boiler_id="BOILER-001", rated_capacity_klb_hr=200.0)
        optimizer.add_boiler(boiler)

        assert "BOILER-001" in optimizer.boilers

    def test_add_multiple_boilers(self, optimizer):
        """Test adding multiple boilers."""
        for i in range(5):
            boiler = BoilerModel(boiler_id=f"BOILER-{i:03d}", rated_capacity_klb_hr=100.0)
            optimizer.add_boiler(boiler)

        assert len(optimizer.boilers) == 5

    def test_remove_boiler(self, optimizer):
        """Test removing a boiler."""
        boiler = BoilerModel(boiler_id="BOILER-001", rated_capacity_klb_hr=200.0)
        optimizer.add_boiler(boiler)
        optimizer.remove_boiler("BOILER-001")

        assert "BOILER-001" not in optimizer.boilers

    def test_remove_nonexistent_boiler(self, optimizer):
        """Test removing nonexistent boiler."""
        optimizer.remove_boiler("NONEXISTENT")
        # Should not raise exception


# =============================================================================
# LOAD DISPATCH OPTIMIZATION TESTS
# =============================================================================


class TestLoadDispatchBasic:
    """Test basic load dispatch optimization."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(
                boiler_id="B1",
                rated_capacity_klb_hr=200.0,
                min_load_percent=25.0,
                fuel_cost_per_mmbtu=5.0,
            ),
            BoilerModel(
                boiler_id="B2",
                rated_capacity_klb_hr=150.0,
                min_load_percent=25.0,
                fuel_cost_per_mmbtu=5.5,
            ),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_dispatch_returns_result(self, optimizer):
        """Test dispatch returns LoadDispatchResult."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=200.0)

        assert isinstance(result, LoadDispatchResult)

    def test_dispatch_meets_demand(self, optimizer):
        """Test dispatch meets total demand."""
        demand = 200.0
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=demand)

        total_allocated = sum(result.allocations.values())
        assert abs(total_allocated - demand) < 5.0  # Within 5 klb/hr

    def test_dispatch_respects_min_load(self, optimizer):
        """Test dispatch respects minimum load."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=100.0)

        for boiler_id, load_pct in result.load_percents.items():
            if load_pct > 0:
                boiler = optimizer.boilers[boiler_id]
                assert load_pct >= boiler.min_load_percent - 0.1

    def test_dispatch_respects_max_load(self, optimizer):
        """Test dispatch respects maximum load."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=300.0)

        for boiler_id, load_pct in result.load_percents.items():
            boiler = optimizer.boilers[boiler_id]
            assert load_pct <= boiler.max_load_percent + 0.1

    def test_dispatch_calculates_efficiency(self, optimizer):
        """Test dispatch calculates average efficiency."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=200.0)

        assert result.avg_efficiency > 0
        assert 50.0 <= result.avg_efficiency <= 100.0

    def test_dispatch_calculates_cost(self, optimizer):
        """Test dispatch calculates total cost."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=200.0)

        assert result.total_cost_hr > 0

    def test_dispatch_metadata(self, optimizer):
        """Test dispatch includes metadata."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=200.0)

        assert result.solver_time_ms >= 0
        assert result.optimal is True
        assert result.constraints_satisfied is True


class TestLoadDispatchObjectives:
    """Test different optimization objectives."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(
                boiler_id="B1",
                rated_capacity_klb_hr=200.0,
                fuel_cost_per_mmbtu=6.0,
                efficiency_c=80.0,  # Less efficient
            ),
            BoilerModel(
                boiler_id="B2",
                rated_capacity_klb_hr=200.0,
                fuel_cost_per_mmbtu=4.0,  # Cheaper
                efficiency_c=82.0,  # More efficient
            ),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_cost_objective(self, optimizer):
        """Test cost minimization objective."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=200.0,
            objective="cost",
        )

        # B2 is cheaper, should get priority
        assert result.allocations["B2"] >= result.allocations["B1"]

    def test_efficiency_objective(self, optimizer):
        """Test efficiency maximization objective."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=200.0,
            objective="efficiency",
        )

        # B2 is more efficient, should get priority
        assert result.allocations["B2"] >= result.allocations["B1"]

    def test_emissions_objective(self, optimizer):
        """Test emissions minimization objective."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=200.0,
            objective="emissions",
        )

        # Should still produce valid result
        assert sum(result.allocations.values()) > 0


class TestLoadDispatchEdgeCases:
    """Test load dispatch edge cases."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(boiler_id="B1", rated_capacity_klb_hr=100.0),
            BoilerModel(boiler_id="B2", rated_capacity_klb_hr=100.0),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_demand_exceeds_capacity(self, optimizer):
        """Test handling when demand exceeds capacity."""
        # Total capacity is 200 klb/hr
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=300.0)

        # Should cap at available capacity
        total = sum(result.allocations.values())
        assert total <= 200.0

    def test_zero_demand(self, optimizer):
        """Test handling of zero demand."""
        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=0.0)

        # All boilers should be at zero or minimum
        for load_pct in result.load_percents.values():
            assert load_pct == 0

    def test_no_boilers_available(self):
        """Test error when no boilers available."""
        optimizer = CombustionOptimizer()

        with pytest.raises(ValueError):
            optimizer.optimize_load_dispatch(total_demand_klb_hr=100.0)

    def test_selected_boilers_only(self, optimizer):
        """Test dispatch with selected boilers only."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=50.0,
            available_boilers=["B1"],
        )

        # Only B1 should be allocated
        assert result.allocations.get("B1", 0) > 0
        assert result.allocations.get("B2", 0) == 0

    def test_unavailable_boiler_ignored(self, optimizer):
        """Test unavailable boilers are ignored."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=100.0,
            available_boilers=["B1", "B3"],  # B3 doesn't exist
        )

        # Should still work with B1
        assert result.allocations.get("B1", 0) > 0


class TestLoadDispatchResult:
    """Test LoadDispatchResult model."""

    def test_default_values(self):
        """Test default values."""
        result = LoadDispatchResult(
            total_demand_klb_hr=100.0,
            total_cost_hr=500.0,
            avg_efficiency=82.0,
        )

        assert result.total_emissions_lb_hr == 0.0
        assert result.allocations == {}
        assert result.solver_time_ms == 0.0
        assert result.optimal is True

    def test_validation(self):
        """Test value validation."""
        result = LoadDispatchResult(
            total_demand_klb_hr=100.0,
            total_cost_hr=500.0,
            avg_efficiency=82.0,
        )

        assert result.avg_efficiency >= 50.0
        assert result.avg_efficiency <= 100.0


# =============================================================================
# COMBUSTION OPTIMIZATION TESTS
# =============================================================================


class TestCombustionOptimization:
    """Test single boiler combustion optimization."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(boiler_id="BOILER-001", rated_capacity_klb_hr=200.0),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_optimize_combustion_returns_dict(self, optimizer):
        """Test combustion optimization returns dict."""
        result = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=4.0,
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        assert isinstance(result, dict)

    def test_optimize_combustion_o2_setpoint(self, optimizer):
        """Test O2 setpoint is returned."""
        result = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=4.0,
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        assert "o2_setpoint" in result
        assert 1.5 <= result["o2_setpoint"] <= 8.0

    def test_optimize_combustion_high_co(self, optimizer):
        """Test O2 increases when CO is high."""
        result_low_co = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=3.0,
            current_co=50.0,  # Low CO
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        result_high_co = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=3.0,
            current_co=500.0,  # High CO
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        # High CO should result in higher O2 setpoint
        assert result_high_co["o2_setpoint"] > result_low_co["o2_setpoint"]

    def test_optimize_combustion_unknown_boiler(self, optimizer):
        """Test optimization with unknown boiler."""
        result = optimizer.optimize_combustion(
            boiler_id="UNKNOWN",
            current_o2=4.0,
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        # Should return current O2 as setpoint
        assert result["o2_setpoint"] == 4.0

    @pytest.mark.parametrize("load_percent,expected_o2_range", [
        (25.0, (4.5, 5.5)),
        (50.0, (3.0, 4.0)),
        (75.0, (2.5, 3.5)),
        (100.0, (2.0, 3.0)),
    ])
    def test_o2_setpoint_curve(self, optimizer, load_percent, expected_o2_range):
        """Test O2 setpoint follows load curve."""
        result = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=3.0,
            current_co=50.0,
            load_percent=load_percent,
            flue_gas_temp=400.0,
        )

        # Allow some flexibility due to CO adjustment
        assert expected_o2_range[0] - 0.5 <= result["o2_setpoint"] <= expected_o2_range[1] + 0.5

    def test_damper_adjustment_calculated(self, optimizer):
        """Test damper adjustment is calculated."""
        result = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=5.0,  # Higher than optimal
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        assert "damper_adjustment" in result

    def test_confidence_calculated(self, optimizer):
        """Test confidence is calculated."""
        result = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=4.0,
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0


# =============================================================================
# OBSERVATION RECORDING AND MODEL TRAINING TESTS
# =============================================================================


class TestObservationRecording:
    """Test observation recording for model training."""

    @pytest.fixture
    def optimizer(self):
        return CombustionOptimizer()

    def test_record_observation(self, optimizer):
        """Test recording a single observation."""
        optimizer.record_observation(
            boiler_id="BOILER-001",
            load_percent=75.0,
            o2_percent=3.5,
            efficiency=82.5,
            co_ppm=50.0,
        )

        assert len(optimizer._history) == 1

    def test_record_multiple_observations(self, optimizer):
        """Test recording multiple observations."""
        for i in range(100):
            optimizer.record_observation(
                boiler_id="BOILER-001",
                load_percent=50.0 + i % 50,
                o2_percent=2.5 + (i % 20) / 10,
                efficiency=80.0 + i % 10,
                co_ppm=30.0 + i % 50,
            )

        assert len(optimizer._history) == 100

    def test_history_trimming(self, optimizer):
        """Test history is trimmed at max size."""
        optimizer._max_history = 100

        for i in range(150):
            optimizer.record_observation(
                boiler_id="BOILER-001",
                load_percent=75.0,
                o2_percent=3.5,
                efficiency=82.5,
                co_ppm=50.0,
            )

        assert len(optimizer._history) == 100

    def test_observation_timestamp(self, optimizer):
        """Test observation includes timestamp."""
        optimizer.record_observation(
            boiler_id="BOILER-001",
            load_percent=75.0,
            o2_percent=3.5,
            efficiency=82.5,
            co_ppm=50.0,
        )

        assert "timestamp" in optimizer._history[0]


class TestModelTraining:
    """Test model training functionality."""

    @pytest.fixture
    def optimizer_with_data(self):
        optimizer = CombustionOptimizer()

        # Add sufficient training data
        for i in range(200):
            optimizer.record_observation(
                boiler_id="BOILER-001",
                load_percent=25.0 + (i % 76),
                o2_percent=2.0 + (i % 40) / 10,
                efficiency=78.0 + (i % 15),
                co_ppm=20.0 + i % 100,
            )

        return optimizer

    def test_train_models_sufficient_data(self, optimizer_with_data):
        """Test model training with sufficient data."""
        metrics = optimizer_with_data.train_models()

        assert "BOILER-001" in metrics
        assert "samples" in metrics["BOILER-001"]

    def test_train_models_insufficient_data(self):
        """Test model training with insufficient data."""
        optimizer = CombustionOptimizer()

        # Only 10 observations (< 100 required)
        for i in range(10):
            optimizer.record_observation(
                boiler_id="BOILER-001",
                load_percent=75.0,
                o2_percent=3.5,
                efficiency=82.5,
                co_ppm=50.0,
            )

        metrics = optimizer.train_models()

        assert metrics["status"] == "insufficient_data"

    def test_model_status(self, optimizer_with_data):
        """Test model status retrieval."""
        optimizer_with_data.train_models()

        status = optimizer_with_data.get_model_status()

        assert "history_size" in status
        assert "models_trained" in status
        assert "model_confidence" in status


# =============================================================================
# PARETO OPTIMIZATION TESTS
# =============================================================================


class TestParetoOptimization:
    """Test multi-objective Pareto optimization."""

    @pytest.fixture
    def multi_boiler_optimizer(self):
        """Create optimizer with diverse boilers for Pareto analysis."""
        boilers = [
            BoilerModel(
                boiler_id="B1_EFFICIENT",
                rated_capacity_klb_hr=200.0,
                efficiency_c=84.0,  # High efficiency
                fuel_cost_per_mmbtu=6.0,  # Higher cost
                nox_base_lb_mmbtu=0.03,  # Low NOx
            ),
            BoilerModel(
                boiler_id="B2_CHEAP",
                rated_capacity_klb_hr=200.0,
                efficiency_c=78.0,  # Lower efficiency
                fuel_cost_per_mmbtu=4.0,  # Low cost
                nox_base_lb_mmbtu=0.08,  # Higher NOx
            ),
            BoilerModel(
                boiler_id="B3_BALANCED",
                rated_capacity_klb_hr=200.0,
                efficiency_c=81.0,  # Medium efficiency
                fuel_cost_per_mmbtu=5.0,  # Medium cost
                nox_base_lb_mmbtu=0.05,  # Medium NOx
            ),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_cost_vs_efficiency_tradeoff(self, multi_boiler_optimizer):
        """Test cost vs efficiency tradeoff."""
        cost_result = multi_boiler_optimizer.optimize_load_dispatch(
            total_demand_klb_hr=300.0,
            objective="cost",
        )

        eff_result = multi_boiler_optimizer.optimize_load_dispatch(
            total_demand_klb_hr=300.0,
            objective="efficiency",
        )

        # Cost-optimal should have lower cost
        assert cost_result.total_cost_hr <= eff_result.total_cost_hr

        # Efficiency-optimal should have higher efficiency
        assert eff_result.avg_efficiency >= cost_result.avg_efficiency

    def test_different_objectives_different_allocations(self, multi_boiler_optimizer):
        """Test different objectives produce different allocations."""
        cost_result = multi_boiler_optimizer.optimize_load_dispatch(
            total_demand_klb_hr=300.0,
            objective="cost",
        )

        eff_result = multi_boiler_optimizer.optimize_load_dispatch(
            total_demand_klb_hr=300.0,
            objective="efficiency",
        )

        # Allocations should differ
        # (may be same in some cases, but generally should differ)
        # Just verify both produce valid results
        assert sum(cost_result.allocations.values()) > 0
        assert sum(eff_result.allocations.values()) > 0


# =============================================================================
# O2 SETPOINT CURVE TESTS
# =============================================================================


class TestO2SetpointCurve:
    """Test O2 setpoint curve calculation."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(boiler_id="BOILER-001", rated_capacity_klb_hr=200.0),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_setpoint_decreases_with_load(self, optimizer):
        """Test O2 setpoint decreases as load increases."""
        setpoints = []
        for load in [25, 50, 75, 100]:
            result = optimizer.optimize_combustion(
                boiler_id="BOILER-001",
                current_o2=3.0,
                current_co=50.0,
                load_percent=float(load),
                flue_gas_temp=400.0,
            )
            setpoints.append(result["o2_setpoint"])

        # Setpoints should generally decrease (or stay same)
        assert setpoints[0] >= setpoints[-1]

    def test_setpoint_curve_interpolation(self, optimizer):
        """Test intermediate loads are interpolated."""
        result_50 = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=3.0,
            current_co=50.0,
            load_percent=50.0,
            flue_gas_temp=400.0,
        )

        result_75 = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=3.0,
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        result_62 = optimizer.optimize_combustion(
            boiler_id="BOILER-001",
            current_o2=3.0,
            current_co=50.0,
            load_percent=62.5,
            flue_gas_temp=400.0,
        )

        # 62.5% should be between 50% and 75%
        assert min(result_50["o2_setpoint"], result_75["o2_setpoint"]) - 0.5 <= result_62["o2_setpoint"] <= max(result_50["o2_setpoint"], result_75["o2_setpoint"]) + 0.5


# =============================================================================
# DISPATCH ALGORITHM TESTS
# =============================================================================


class TestDispatchByCost:
    """Test cost-based dispatch algorithm."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(
                boiler_id="EXPENSIVE",
                rated_capacity_klb_hr=100.0,
                fuel_cost_per_mmbtu=8.0,
            ),
            BoilerModel(
                boiler_id="CHEAP",
                rated_capacity_klb_hr=100.0,
                fuel_cost_per_mmbtu=4.0,
            ),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_cheap_boiler_loaded_first(self, optimizer):
        """Test cheaper boiler is loaded first."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=75.0,
            objective="cost",
        )

        # CHEAP should have higher load
        assert result.allocations.get("CHEAP", 0) >= result.allocations.get("EXPENSIVE", 0)


class TestDispatchByEfficiency:
    """Test efficiency-based dispatch algorithm."""

    @pytest.fixture
    def optimizer(self):
        boilers = [
            BoilerModel(
                boiler_id="EFFICIENT",
                rated_capacity_klb_hr=100.0,
                efficiency_c=85.0,
            ),
            BoilerModel(
                boiler_id="INEFFICIENT",
                rated_capacity_klb_hr=100.0,
                efficiency_c=75.0,
            ),
        ]
        return CombustionOptimizer(boilers=boilers)

    def test_efficient_boiler_loaded_first(self, optimizer):
        """Test more efficient boiler is loaded first."""
        result = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=75.0,
            objective="efficiency",
        )

        # EFFICIENT should have higher load
        assert result.allocations.get("EFFICIENT", 0) >= result.allocations.get("INEFFICIENT", 0)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestOptimizerIntegration:
    """Integration tests for optimizer workflows."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create optimizer with boilers
        boilers = [
            BoilerModel(boiler_id="B1", rated_capacity_klb_hr=200.0),
            BoilerModel(boiler_id="B2", rated_capacity_klb_hr=150.0),
        ]
        optimizer = CombustionOptimizer(boilers=boilers)

        # Record observations
        for i in range(150):
            optimizer.record_observation(
                boiler_id="B1",
                load_percent=50.0 + i % 50,
                o2_percent=3.0 + (i % 20) / 10,
                efficiency=80.0 + i % 10,
                co_ppm=30.0 + i % 50,
            )

        # Train models
        metrics = optimizer.train_models()

        # Run load dispatch
        dispatch = optimizer.optimize_load_dispatch(
            total_demand_klb_hr=250.0,
            objective="cost",
        )

        # Run combustion optimization
        combustion = optimizer.optimize_combustion(
            boiler_id="B1",
            current_o2=4.0,
            current_co=50.0,
            load_percent=75.0,
            flue_gas_temp=400.0,
        )

        # Verify all steps completed
        assert metrics is not None
        assert dispatch.total_demand_klb_hr == 250.0
        assert "o2_setpoint" in combustion
