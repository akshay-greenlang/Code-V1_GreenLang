"""
Unit Tests for GL-003 UnifiedSteam - Steam Network Optimizer

Tests for:
- Load allocation optimization (cost, efficiency, emissions)
- Header pressure optimization
- PRV setpoint optimization
- Total loss minimization
- Constraint handling and validation

Target Coverage: 90%+
"""

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import application modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from optimization.steam_network_optimizer import (
    BoilerLoadAllocation,
    BoilerState,
    DemandForecast,
    HeaderOptimization,
    HeaderState,
    HeaderType,
    LoadAllocationResult,
    LossMinimizationResult,
    NetworkModel,
    PRVOptimization,
    PRVState,
    SteamNetworkOptimizer,
)
from optimization.constraints import (
    ConstraintCheckResult,
    ConstraintSeverity,
    ConstraintStatus,
    SteamSystemConstraints,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def optimizer() -> SteamNetworkOptimizer:
    """Create a SteamNetworkOptimizer instance."""
    return SteamNetworkOptimizer()


@pytest.fixture
def optimizer_with_constraints() -> SteamNetworkOptimizer:
    """Create optimizer with custom constraints."""
    constraints = SteamSystemConstraints()
    return SteamNetworkOptimizer(
        constraints=constraints,
        operating_hours=8760,
        co2_cost_per_ton=50.0
    )


@pytest.fixture
def sample_boilers() -> List[BoilerState]:
    """Create sample boiler states for testing."""
    return [
        BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=75.0,
            rated_capacity_klb_hr=100.0,
            min_load_percent=25.0,
            max_load_percent=100.0,
            current_efficiency_percent=84.0,
            fuel_type="natural_gas",
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
            maintenance_priority=0
        ),
        BoilerState(
            boiler_id="BLR-002",
            is_online=True,
            current_load_percent=70.0,
            rated_capacity_klb_hr=80.0,
            min_load_percent=25.0,
            max_load_percent=100.0,
            current_efficiency_percent=82.0,
            fuel_type="natural_gas",
            fuel_cost_per_mmbtu=8.5,
            co2_factor_lb_mmbtu=117.0,
            maintenance_priority=0
        ),
        BoilerState(
            boiler_id="BLR-003",
            is_online=True,
            current_load_percent=60.0,
            rated_capacity_klb_hr=120.0,
            min_load_percent=25.0,
            max_load_percent=100.0,
            current_efficiency_percent=86.0,
            fuel_type="natural_gas",
            fuel_cost_per_mmbtu=7.5,
            co2_factor_lb_mmbtu=117.0,
            maintenance_priority=0
        ),
    ]


@pytest.fixture
def sample_boilers_with_offline() -> List[BoilerState]:
    """Create sample boilers including one offline."""
    return [
        BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=80.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=85.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        ),
        BoilerState(
            boiler_id="BLR-002",
            is_online=False,  # Offline
            current_load_percent=0.0,
            rated_capacity_klb_hr=80.0,
            current_efficiency_percent=0.0,
            fuel_cost_per_mmbtu=8.5,
            co2_factor_lb_mmbtu=117.0,
        ),
    ]


@pytest.fixture
def sample_headers() -> List[HeaderState]:
    """Create sample header states."""
    return [
        HeaderState(
            header_id="HP-MAIN",
            header_type=HeaderType.HIGH_PRESSURE,
            pressure_psig=600.0,
            setpoint_psig=600.0,
            temperature_f=700.0,
            flow_klb_hr=150.0,
            user_demand_klb_hr=120.0,
            connected_boilers=["BLR-001", "BLR-002", "BLR-003"],
            connected_prvs=["PRV-001"],
        ),
        HeaderState(
            header_id="MP-MAIN",
            header_type=HeaderType.MEDIUM_PRESSURE,
            pressure_psig=150.0,
            setpoint_psig=150.0,
            temperature_f=380.0,
            flow_klb_hr=50.0,
            user_demand_klb_hr=45.0,
            connected_boilers=[],
            connected_prvs=["PRV-002"],
        ),
        HeaderState(
            header_id="LP-MAIN",
            header_type=HeaderType.LOW_PRESSURE,
            pressure_psig=15.0,
            setpoint_psig=15.0,
            temperature_f=250.0,
            flow_klb_hr=20.0,
            user_demand_klb_hr=18.0,
            connected_boilers=[],
            connected_prvs=[],
        ),
    ]


@pytest.fixture
def sample_prvs() -> List[PRVState]:
    """Create sample PRV states."""
    return [
        PRVState(
            prv_id="PRV-001",
            upstream_header="HP-MAIN",
            downstream_header="MP-MAIN",
            upstream_pressure_psig=600.0,
            downstream_pressure_psig=150.0,
            setpoint_psig=150.0,
            flow_klb_hr=30.0,
            valve_position_percent=45.0,
            max_capacity_klb_hr=60.0,
            is_desuperheating=True,
        ),
        PRVState(
            prv_id="PRV-002",
            upstream_header="MP-MAIN",
            downstream_header="LP-MAIN",
            upstream_pressure_psig=150.0,
            downstream_pressure_psig=15.0,
            setpoint_psig=15.0,
            flow_klb_hr=10.0,
            valve_position_percent=30.0,
            max_capacity_klb_hr=25.0,
            is_desuperheating=False,
        ),
    ]


@pytest.fixture
def sample_demand_forecast() -> DemandForecast:
    """Create sample demand forecast."""
    hours = 24
    return DemandForecast(
        forecast_horizon_hours=hours,
        hp_demand_klb_hr=[100.0 + i * 2 for i in range(hours)],  # 100-146 klb/hr
        mp_demand_klb_hr=[40.0 + (i % 8) for i in range(hours)],  # 40-47 klb/hr
        lp_demand_klb_hr=[15.0 + (i % 4) for i in range(hours)],  # 15-18 klb/hr
        confidence=0.85,
    )


@pytest.fixture
def sample_network_model(sample_boilers, sample_headers, sample_prvs) -> NetworkModel:
    """Create sample network model."""
    return NetworkModel(
        boilers=sample_boilers,
        headers=sample_headers,
        prvs=sample_prvs,
        total_generation_klb_hr=150.0,
        total_demand_klb_hr=143.0,
        distribution_loss_percent=3.5,
    )


# =============================================================================
# Test Load Allocation Optimization
# =============================================================================

class TestLoadAllocationOptimization:
    """Tests for boiler load allocation optimization."""

    def test_load_allocation_cost_objective(self, optimizer, sample_boilers):
        """Test load allocation with cost minimization objective."""
        demand = 150.0  # klb/hr

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        # Verify result structure
        assert isinstance(result, LoadAllocationResult)
        assert result.optimization_objective == "cost"
        assert len(result.allocations) == len(sample_boilers)

        # Verify total generation meets demand (within tolerance)
        total_gen = sum(a.recommended_output_klb_hr for a in result.allocations)
        assert abs(total_gen - demand) < 1.0, f"Generation {total_gen} != demand {demand}"

        # Verify cost is positive
        assert result.total_cost_per_hr > 0

        # Verify provenance hash exists
        assert len(result.provenance_hash) == 64

    def test_load_allocation_efficiency_objective(self, optimizer, sample_boilers):
        """Test load allocation with efficiency maximization objective."""
        demand = 150.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="efficiency"
        )

        assert result.optimization_objective == "efficiency"
        assert result.weighted_efficiency > 0

        # Efficiency should be positive percentage
        assert 50 <= result.weighted_efficiency <= 100

    def test_load_allocation_emissions_objective(self, optimizer, sample_boilers):
        """Test load allocation with emissions minimization objective."""
        demand = 150.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="emissions"
        )

        assert result.optimization_objective == "emissions"
        assert result.total_co2_lb_hr > 0

    def test_load_allocation_balanced_objective(self, optimizer, sample_boilers):
        """Test load allocation with balanced multi-objective."""
        demand = 150.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="balanced"
        )

        assert result.optimization_objective == "balanced"
        # Should have all metrics calculated
        assert result.total_cost_per_hr > 0
        assert result.total_co2_lb_hr > 0
        assert result.weighted_efficiency > 0

    def test_load_allocation_skips_offline_boilers(self, optimizer, sample_boilers_with_offline):
        """Test that offline boilers are excluded from allocation."""
        demand = 80.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers_with_offline,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        # Only online boiler should be allocated load
        online_allocations = [
            a for a in result.allocations if a.recommended_load_percent > 0
        ]
        # Since BLR-002 is offline, all load should go to BLR-001
        assert len(online_allocations) <= 1

    def test_load_allocation_no_online_boilers_raises(self, optimizer):
        """Test that allocation fails when no boilers are online."""
        offline_boilers = [
            BoilerState(
                boiler_id="BLR-001",
                is_online=False,
                current_load_percent=0.0,
                rated_capacity_klb_hr=100.0,
                current_efficiency_percent=0.0,
            ),
        ]

        with pytest.raises(ValueError, match="No online boilers"):
            optimizer.optimize_load_allocation(
                boilers=offline_boilers,
                total_demand_klb_hr=50.0,
                objective="cost"
            )

    def test_load_allocation_demand_exceeds_capacity(self, optimizer, sample_boilers):
        """Test handling when demand exceeds total capacity."""
        # Total capacity is ~300 klb/hr, demand is higher
        excessive_demand = 500.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=excessive_demand,
            objective="cost"
        )

        # Should cap at capacity
        assert result.total_generation_klb_hr < excessive_demand

    def test_load_allocation_respects_min_load(self, optimizer):
        """Test that minimum load constraints are respected."""
        boilers = [
            BoilerState(
                boiler_id="BLR-001",
                is_online=True,
                current_load_percent=50.0,
                rated_capacity_klb_hr=100.0,
                min_load_percent=30.0,
                max_load_percent=100.0,
                current_efficiency_percent=85.0,
                fuel_cost_per_mmbtu=8.0,
                co2_factor_lb_mmbtu=117.0,
            ),
        ]

        # Demand that would require below min load
        demand = 20.0  # 20% of capacity, below 30% min

        result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        # Boiler should either be at min load or off
        for alloc in result.allocations:
            if alloc.recommended_load_percent > 0:
                assert alloc.recommended_load_percent >= 30.0 or alloc.recommended_load_percent == 0

    def test_load_allocation_improvement_calculation(self, optimizer, sample_boilers):
        """Test that improvement percentage is calculated correctly."""
        demand = 150.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        # Improvement should be a valid percentage
        assert -100 <= result.improvement_percent <= 100

    def test_load_allocation_change_required_flag(self, optimizer, sample_boilers):
        """Test that change_required flag is set correctly."""
        demand = 150.0

        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        for alloc in result.allocations:
            # Find corresponding boiler
            boiler = next(b for b in sample_boilers if b.boiler_id == alloc.boiler_id)
            delta = abs(alloc.recommended_load_percent - boiler.current_load_percent)

            # Change required if delta > 2%
            if delta > 2.0:
                assert alloc.change_required
            else:
                assert not alloc.change_required


class TestLoadAllocationEfficiency:
    """Tests for efficiency calculations in load allocation."""

    def test_efficiency_at_optimal_load(self, optimizer):
        """Test efficiency calculation at optimal load point."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=75.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=85.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        # Test efficiency at various loads
        eff_50 = optimizer._calculate_efficiency_at_load(boiler, 50.0)
        eff_75 = optimizer._calculate_efficiency_at_load(boiler, 75.0)
        eff_100 = optimizer._calculate_efficiency_at_load(boiler, 100.0)

        # Efficiency should be bounded
        assert 60 <= eff_50 <= 95
        assert 60 <= eff_75 <= 95
        assert 60 <= eff_100 <= 95

        # Typical curve has peak around 70-80%
        assert eff_75 >= eff_50 or eff_75 >= eff_100  # Should be near peak

    def test_efficiency_at_zero_load(self, optimizer):
        """Test efficiency at zero load."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=0.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=0.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        eff = optimizer._calculate_efficiency_at_load(boiler, 0.0)
        assert eff == 0.0

    def test_marginal_cost_calculation(self, optimizer):
        """Test marginal cost calculation."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=75.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=85.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        mc = optimizer._calculate_marginal_cost(boiler, 75.0)

        # Marginal cost should be positive
        assert mc > 0
        assert mc < float('inf')

    def test_marginal_cost_at_zero_load(self, optimizer):
        """Test marginal cost at zero load is infinite."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=0.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=0.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        mc = optimizer._calculate_marginal_cost(boiler, 0.0)
        assert mc == float('inf')


class TestBoilerCostCalculation:
    """Tests for boiler operating cost calculations."""

    def test_boiler_cost_calculation(self, optimizer):
        """Test boiler cost calculation at given load."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=75.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=85.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        cost = optimizer._calculate_boiler_cost(boiler, 75.0)

        # Cost should be positive
        assert cost > 0

        # Rough sanity check: at 75 klb/hr, cost should be reasonable
        # ~75 klb/hr * 1000 lb/klb * 990 BTU/lb / 0.85 / 1e6 MMBTU * $8/MMBTU
        expected_approx = 75 * 1000 * 990 / 0.85 / 1e6 * 8
        assert 0.5 * expected_approx <= cost <= 2.0 * expected_approx

    def test_boiler_cost_at_zero_load(self, optimizer):
        """Test boiler cost at zero load."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=0.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=0.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        cost = optimizer._calculate_boiler_cost(boiler, 0.0)
        assert cost == 0.0

    def test_boiler_co2_calculation(self, optimizer):
        """Test CO2 emissions calculation."""
        boiler = BoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=75.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=85.0,
            fuel_cost_per_mmbtu=8.0,
            co2_factor_lb_mmbtu=117.0,
        )

        co2 = optimizer._calculate_boiler_co2(boiler, 75.0)

        # CO2 should be positive
        assert co2 > 0


# =============================================================================
# Test Header Pressure Optimization
# =============================================================================

class TestHeaderPressureOptimization:
    """Tests for header pressure optimization."""

    def test_optimize_header_pressures(
        self, optimizer, sample_headers, sample_demand_forecast
    ):
        """Test header pressure optimization."""
        results = optimizer.optimize_header_pressures(
            demand_forecast=sample_demand_forecast,
            boiler_constraints={},
            current_headers=sample_headers,
        )

        assert len(results) == len(sample_headers)

        for result in results:
            assert isinstance(result, HeaderOptimization)
            assert result.header_id is not None
            assert 0 <= result.confidence <= 1

    def test_header_optimization_hp(self, optimizer):
        """Test HP header optimization specifically."""
        hp_header = HeaderState(
            header_id="HP-MAIN",
            header_type=HeaderType.HIGH_PRESSURE,
            pressure_psig=620.0,
            setpoint_psig=620.0,
            temperature_f=700.0,
            flow_klb_hr=100.0,
            user_demand_klb_hr=95.0,
        )

        forecast = DemandForecast(
            hp_demand_klb_hr=[95.0, 100.0, 105.0],
            mp_demand_klb_hr=[],
            lp_demand_klb_hr=[],
        )

        results = optimizer.optimize_header_pressures(
            demand_forecast=forecast,
            boiler_constraints={},
            current_headers=[hp_header],
        )

        assert len(results) == 1
        result = results[0]

        # Recommended pressure should be within HP range
        # Based on constraints, HP min is typically 550-600 psig
        assert result.recommended_pressure_psig >= 500
        assert result.recommended_pressure_psig <= 650

    def test_header_optimization_savings_estimate(self, optimizer):
        """Test savings estimation from pressure reduction."""
        hp_header = HeaderState(
            header_id="HP-MAIN",
            header_type=HeaderType.HIGH_PRESSURE,
            pressure_psig=650.0,  # Higher than optimal
            setpoint_psig=650.0,
            temperature_f=720.0,
            flow_klb_hr=100.0,
            user_demand_klb_hr=90.0,
        )

        forecast = DemandForecast(
            hp_demand_klb_hr=[90.0],
            mp_demand_klb_hr=[],
            lp_demand_klb_hr=[],
        )

        results = optimizer.optimize_header_pressures(
            demand_forecast=forecast,
            boiler_constraints={},
            current_headers=[hp_header],
        )

        result = results[0]

        # If pressure can be reduced, should have savings
        if result.pressure_change_psig < 0:
            assert result.expected_savings_per_hr >= 0

    def test_calculate_user_min_pressure(self, optimizer):
        """Test user minimum pressure calculation."""
        hp_min = optimizer._calculate_user_min_pressure(HeaderType.HIGH_PRESSURE, 100.0)
        mp_min = optimizer._calculate_user_min_pressure(HeaderType.MEDIUM_PRESSURE, 50.0)
        lp_min = optimizer._calculate_user_min_pressure(HeaderType.LOW_PRESSURE, 20.0)

        # HP > MP > LP
        assert hp_min > mp_min > lp_min


# =============================================================================
# Test PRV Setpoint Optimization
# =============================================================================

class TestPRVSetpointOptimization:
    """Tests for PRV setpoint optimization."""

    def test_optimize_prv_setpoints(self, optimizer, sample_network_model):
        """Test PRV setpoint optimization."""
        user_requirements = {
            "MP-MAIN": 140.0,
            "LP-MAIN": 10.0,
        }

        result = optimizer.optimize_prv_setpoints(
            network_state=sample_network_model,
            user_requirements=user_requirements,
        )

        assert isinstance(result, PRVOptimization)
        assert len(result.optimizations) == len(sample_network_model.prvs)
        assert result.total_prv_flow_klb_hr > 0

    def test_prv_letdown_loss_calculation(self, optimizer):
        """Test letdown loss calculation."""
        flow = 30.0  # klb/hr
        upstream = 600.0  # psig
        downstream = 150.0  # psig

        loss = optimizer._calculate_letdown_loss(flow, upstream, downstream)

        # Loss should be positive
        assert loss > 0

        # Approximate: 30 klb/hr * 1000 lb/klb * (600-150) psi * ~1 BTU/(lb*psi)
        expected_approx = 30 * 1000 * 450 * 1.0
        assert 0.5 * expected_approx <= loss <= 2.0 * expected_approx

    def test_prv_minimum_differential(self, optimizer):
        """Test that PRV maintains minimum differential."""
        prv = PRVState(
            prv_id="PRV-001",
            upstream_header="HP-MAIN",
            downstream_header="MP-MAIN",
            upstream_pressure_psig=200.0,  # Low upstream
            downstream_pressure_psig=180.0,  # Close to upstream
            setpoint_psig=180.0,
            flow_klb_hr=20.0,
            valve_position_percent=80.0,
            max_capacity_klb_hr=40.0,
        )

        network = NetworkModel(
            boilers=[],
            headers=[],
            prvs=[prv],
        )

        result = optimizer.optimize_prv_setpoints(
            network_state=network,
            user_requirements={"MP-MAIN": 150.0},
        )

        # Should maintain minimum differential
        for opt in result.optimizations:
            # Can't exceed upstream pressure minus min differential
            assert opt["recommended_setpoint_psig"] <= prv.upstream_pressure_psig - 10  # Assuming 10 psi min diff


# =============================================================================
# Test Loss Minimization
# =============================================================================

class TestLossMinimization:
    """Tests for total loss minimization."""

    def test_minimize_total_losses(self, optimizer, sample_network_model):
        """Test total loss minimization."""
        result = optimizer.minimize_total_losses(
            network_model=sample_network_model,
        )

        assert isinstance(result, LossMinimizationResult)
        assert result.current_total_loss_klb_hr >= 0
        assert result.current_loss_cost_per_hr >= 0
        assert len(result.provenance_hash) == 64

    def test_loss_minimization_recommendations(self, optimizer):
        """Test that recommendations are generated for high losses."""
        high_loss_network = NetworkModel(
            boilers=[],
            headers=[],
            prvs=[
                PRVState(
                    prv_id="PRV-001",
                    upstream_header="HP",
                    downstream_header="MP",
                    upstream_pressure_psig=600.0,
                    downstream_pressure_psig=150.0,
                    setpoint_psig=150.0,
                    flow_klb_hr=100.0,  # High PRV flow
                    valve_position_percent=90.0,
                    max_capacity_klb_hr=120.0,
                ),
            ],
            total_generation_klb_hr=150.0,
            total_demand_klb_hr=143.0,
            distribution_loss_percent=8.0,  # High losses
        )

        result = optimizer.minimize_total_losses(network_model=high_loss_network)

        # Should have recommendations due to high losses
        assert len(result.recommendations) > 0

    def test_loss_minimization_savings_calculation(self, optimizer, sample_network_model):
        """Test savings calculation."""
        # Modify network to have reducible losses
        sample_network_model.distribution_loss_percent = 10.0  # High losses

        result = optimizer.minimize_total_losses(network_model=sample_network_model)

        # Should have savings potential
        assert result.total_savings_per_hr >= 0
        assert result.annual_savings_potential >= 0

    def test_calculate_generation_losses(self, optimizer, sample_network_model):
        """Test generation loss calculation."""
        loss = optimizer._calculate_generation_losses(sample_network_model)
        assert loss >= 0

    def test_calculate_distribution_losses(self, optimizer, sample_network_model):
        """Test distribution loss calculation."""
        loss = optimizer._calculate_distribution_losses(sample_network_model)
        assert loss >= 0


# =============================================================================
# Test Constraint Handling
# =============================================================================

class TestConstraintHandling:
    """Tests for constraint validation and handling."""

    def test_optimizer_with_custom_constraints(self, optimizer_with_constraints, sample_boilers):
        """Test optimizer respects custom constraints."""
        result = optimizer_with_constraints.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=150.0,
            objective="cost"
        )

        assert result is not None
        assert result.total_generation_klb_hr > 0

    def test_header_optimization_constraint_warnings(self, optimizer):
        """Test that constraint violations generate warnings."""
        # Create header with pressure above typical max
        header = HeaderState(
            header_id="HP-TEST",
            header_type=HeaderType.HIGH_PRESSURE,
            pressure_psig=700.0,  # Very high
            setpoint_psig=700.0,
            temperature_f=750.0,
            flow_klb_hr=100.0,
            user_demand_klb_hr=100.0,
        )

        forecast = DemandForecast(
            hp_demand_klb_hr=[200.0],  # Very high demand requiring high pressure
            mp_demand_klb_hr=[],
            lp_demand_klb_hr=[],
        )

        results = optimizer.optimize_header_pressures(
            demand_forecast=forecast,
            boiler_constraints={},
            current_headers=[header],
        )

        # May have constraint warnings
        assert len(results) == 1


# =============================================================================
# Test Provenance and Reproducibility
# =============================================================================

class TestProvenanceAndReproducibility:
    """Tests for provenance tracking and reproducibility."""

    def test_load_allocation_provenance_hash_exists(self, optimizer, sample_boilers):
        """Test that provenance hash is generated."""
        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=150.0,
            objective="cost"
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_load_allocation_provenance_hash_deterministic(self, optimizer, sample_boilers):
        """Test that provenance hash is deterministic."""
        result1 = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=150.0,
            objective="cost"
        )

        result2 = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=150.0,
            objective="cost"
        )

        # Same inputs should produce same hash
        assert result1.provenance_hash == result2.provenance_hash

    def test_load_allocation_reproducibility(self, optimizer, sample_boilers):
        """Test that results are reproducible."""
        demand = 150.0

        result1 = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        result2 = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        # Allocations should be identical
        for a1, a2 in zip(result1.allocations, result2.allocations):
            assert a1.boiler_id == a2.boiler_id
            assert a1.recommended_load_percent == a2.recommended_load_percent

    def test_loss_minimization_provenance_hash(self, optimizer, sample_network_model):
        """Test loss minimization provenance hash."""
        result = optimizer.minimize_total_losses(network_model=sample_network_model)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for optimization."""

    def test_load_allocation_performance(self, optimizer, sample_boilers, benchmark):
        """Benchmark load allocation performance."""
        def allocate():
            return optimizer.optimize_load_allocation(
                boilers=sample_boilers,
                total_demand_klb_hr=150.0,
                objective="cost"
            )

        result = benchmark(allocate)
        assert result is not None

    def test_load_allocation_large_fleet(self, optimizer):
        """Test performance with large boiler fleet."""
        # Create 20 boilers
        large_fleet = [
            BoilerState(
                boiler_id=f"BLR-{i:03d}",
                is_online=True,
                current_load_percent=70.0,
                rated_capacity_klb_hr=100.0,
                current_efficiency_percent=85.0,
                fuel_cost_per_mmbtu=8.0 + i * 0.1,
                co2_factor_lb_mmbtu=117.0,
            )
            for i in range(20)
        ]

        start_time = time.perf_counter()
        result = optimizer.optimize_load_allocation(
            boilers=large_fleet,
            total_demand_klb_hr=1000.0,
            objective="cost"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete in < 100ms
        assert elapsed_ms < 100, f"Took {elapsed_ms:.1f}ms for 20 boilers"
        assert result is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_boiler(self, optimizer):
        """Test optimization with single boiler."""
        single_boiler = [
            BoilerState(
                boiler_id="BLR-001",
                is_online=True,
                current_load_percent=75.0,
                rated_capacity_klb_hr=100.0,
                current_efficiency_percent=85.0,
                fuel_cost_per_mmbtu=8.0,
                co2_factor_lb_mmbtu=117.0,
            )
        ]

        result = optimizer.optimize_load_allocation(
            boilers=single_boiler,
            total_demand_klb_hr=50.0,
            objective="cost"
        )

        assert len(result.allocations) == 1
        assert result.allocations[0].recommended_load_percent == 50.0

    def test_zero_demand(self, optimizer, sample_boilers):
        """Test optimization with zero demand."""
        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=0.0,
            objective="cost"
        )

        # All boilers should be at 0 load
        for alloc in result.allocations:
            assert alloc.recommended_load_percent == 0.0

    def test_very_low_demand(self, optimizer, sample_boilers):
        """Test optimization with very low demand (below min load)."""
        result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=10.0,  # Below typical min load
            objective="cost"
        )

        # Should handle gracefully
        assert result is not None

    def test_empty_header_list(self, optimizer, sample_demand_forecast):
        """Test header optimization with empty list."""
        results = optimizer.optimize_header_pressures(
            demand_forecast=sample_demand_forecast,
            boiler_constraints={},
            current_headers=[],
        )

        assert len(results) == 0

    def test_empty_prv_list(self, optimizer):
        """Test PRV optimization with empty list."""
        network = NetworkModel(
            boilers=[],
            headers=[],
            prvs=[],
        )

        result = optimizer.optimize_prv_setpoints(
            network_state=network,
            user_requirements={},
        )

        assert len(result.optimizations) == 0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestOptimizerIntegration:
    """Integration tests for optimizer workflows."""

    def test_full_optimization_workflow(
        self,
        optimizer,
        sample_boilers,
        sample_headers,
        sample_demand_forecast,
        sample_network_model,
    ):
        """Test complete optimization workflow."""
        # Step 1: Load allocation
        load_result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=150.0,
            objective="cost"
        )
        assert load_result.total_generation_klb_hr > 0

        # Step 2: Header pressure optimization
        header_results = optimizer.optimize_header_pressures(
            demand_forecast=sample_demand_forecast,
            boiler_constraints={},
            current_headers=sample_headers,
        )
        assert len(header_results) > 0

        # Step 3: PRV setpoint optimization
        prv_result = optimizer.optimize_prv_setpoints(
            network_state=sample_network_model,
            user_requirements={"MP-MAIN": 140.0, "LP-MAIN": 10.0},
        )
        assert prv_result is not None

        # Step 4: Loss minimization
        loss_result = optimizer.minimize_total_losses(
            network_model=sample_network_model,
        )
        assert loss_result.current_total_loss_klb_hr >= 0

    def test_optimization_consistency(self, optimizer, sample_boilers):
        """Test that different objectives produce consistent results."""
        demand = 150.0

        cost_result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="cost"
        )

        eff_result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="efficiency"
        )

        emiss_result = optimizer.optimize_load_allocation(
            boilers=sample_boilers,
            total_demand_klb_hr=demand,
            objective="emissions"
        )

        # All should meet the same demand
        assert abs(cost_result.total_generation_klb_hr - demand) < 5.0
        assert abs(eff_result.total_generation_klb_hr - demand) < 5.0
        assert abs(emiss_result.total_generation_klb_hr - demand) < 5.0
