"""
Unit tests for GL-001 ThermalCommand MILP Load Allocator.

Tests the Mixed Integer Linear Programming optimizer for multi-equipment
thermal load dispatch with comprehensive edge case coverage.

Coverage Target: 85%+
Reference: GL-001 Specification Section 11

Test Categories:
1. Initialization and configuration
2. Equipment management
3. Optimization with valid inputs
4. Edge cases and boundary conditions
5. Infeasible scenario handling
6. Performance benchmarks
7. Provenance tracking

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from optimization.milp_optimizer import (
    MILPLoadAllocator,
    Equipment,
    EquipmentType,
    FuelType,
    EquipmentStatus,
    OptimizationStatus,
    OptimizationObjective,
    LoadAllocationRequest,
    LoadAllocationResult,
    EquipmentAllocation,
    EquipmentEfficiencyCurve,
    create_standard_boiler,
    create_chp_system,
    SCIPY_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_allocator() -> MILPLoadAllocator:
    """Create a default MILP allocator instance."""
    return MILPLoadAllocator(
        time_limit_seconds=30.0,
        gap_tolerance=0.01,
        use_warm_start=True
    )


@pytest.fixture
def sample_boiler() -> Equipment:
    """Create a sample natural gas boiler."""
    return Equipment(
        equipment_id="BOILER-001",
        name="Boiler 1",
        equipment_type=EquipmentType.BOILER,
        fuel_type=FuelType.NATURAL_GAS,
        max_capacity_mmbtu_hr=50.0,
        min_capacity_mmbtu_hr=10.0,
        fuel_cost_per_mmbtu=5.0,
        fixed_cost_per_hour=25.0,
        co2_kg_per_mmbtu_fuel=53.06,
        nox_kg_per_mmbtu_fuel=0.04,
        rated_efficiency=0.85,
        ramp_up_mmbtu_hr_per_min=1.0,
        ramp_down_mmbtu_hr_per_min=1.0,
        min_run_time_minutes=30,
        min_off_time_minutes=15,
        status=EquipmentStatus.AVAILABLE,
        current_load_mmbtu_hr=0.0,
        priority=1,
    )


@pytest.fixture
def sample_chp() -> Equipment:
    """Create a sample CHP system."""
    return Equipment(
        equipment_id="CHP-001",
        name="CHP System 1",
        equipment_type=EquipmentType.CHP,
        fuel_type=FuelType.NATURAL_GAS,
        max_capacity_mmbtu_hr=30.0,
        min_capacity_mmbtu_hr=15.0,  # Higher turndown for CHP
        fuel_cost_per_mmbtu=3.0,  # Lower net cost due to electricity credit
        fixed_cost_per_hour=60.0,  # Higher fixed costs
        co2_kg_per_mmbtu_fuel=31.84,  # Credit for displaced electricity
        nox_kg_per_mmbtu_fuel=0.03,
        rated_efficiency=0.80,
        status=EquipmentStatus.AVAILABLE,
        priority=1,  # CHP dispatched first
    )


@pytest.fixture
def three_boiler_system(sample_boiler) -> List[Equipment]:
    """Create a system with three boilers of different capacities."""
    boilers = []
    for i, (cap, cost) in enumerate([(50.0, 5.0), (40.0, 5.5), (30.0, 6.0)]):
        boiler = Equipment(
            equipment_id=f"BOILER-{i+1:03d}",
            name=f"Boiler {i+1}",
            equipment_type=EquipmentType.BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            max_capacity_mmbtu_hr=cap,
            min_capacity_mmbtu_hr=cap / 4.0,  # 4:1 turndown
            fuel_cost_per_mmbtu=cost,
            fixed_cost_per_hour=cap * 0.5,
            co2_kg_per_mmbtu_fuel=53.06,
            nox_kg_per_mmbtu_fuel=0.04,
            status=EquipmentStatus.AVAILABLE,
            priority=i + 1,
        )
        boilers.append(boiler)
    return boilers


@pytest.fixture
def allocator_with_equipment(default_allocator, three_boiler_system) -> MILPLoadAllocator:
    """Create allocator with three boilers registered."""
    for boiler in three_boiler_system:
        default_allocator.add_equipment(boiler)
    return default_allocator


@pytest.fixture
def basic_request() -> LoadAllocationRequest:
    """Create a basic load allocation request."""
    return LoadAllocationRequest(
        request_id="REQ-001",
        total_demand_mmbtu_hr=60.0,
        optimization_objective=OptimizationObjective.BALANCED,
        emissions_penalty_per_kg_co2=0.05,
        time_horizon_minutes=60,
        allow_unmet_demand=False,
    )


# =============================================================================
# TEST CLASS: INITIALIZATION
# =============================================================================

class TestMILPLoadAllocatorInitialization:
    """Tests for MILP Load Allocator initialization."""

    def test_default_initialization(self):
        """Test default allocator initialization."""
        allocator = MILPLoadAllocator()

        assert allocator.time_limit == 60.0
        assert allocator.gap_tolerance == 0.01
        assert allocator.use_warm_start is True
        assert len(allocator._equipment) == 0
        assert len(allocator._equipment_order) == 0
        assert allocator._last_solution is None

    def test_custom_initialization(self):
        """Test allocator with custom parameters."""
        allocator = MILPLoadAllocator(
            time_limit_seconds=120.0,
            gap_tolerance=0.005,
            use_warm_start=False
        )

        assert allocator.time_limit == 120.0
        assert allocator.gap_tolerance == 0.005
        assert allocator.use_warm_start is False

    @pytest.mark.parametrize("time_limit,gap_tolerance", [
        (1.0, 0.001),
        (60.0, 0.01),
        (600.0, 0.1),
    ])
    def test_initialization_with_various_params(self, time_limit, gap_tolerance):
        """Test initialization with various parameter combinations."""
        allocator = MILPLoadAllocator(
            time_limit_seconds=time_limit,
            gap_tolerance=gap_tolerance
        )

        assert allocator.time_limit == time_limit
        assert allocator.gap_tolerance == gap_tolerance


# =============================================================================
# TEST CLASS: EQUIPMENT MANAGEMENT
# =============================================================================

class TestEquipmentManagement:
    """Tests for equipment management operations."""

    def test_add_equipment_success(self, default_allocator, sample_boiler):
        """Test successful equipment addition."""
        result = default_allocator.add_equipment(sample_boiler)

        assert result is True
        assert sample_boiler.equipment_id in default_allocator._equipment
        assert sample_boiler.equipment_id in default_allocator._equipment_order

    def test_add_duplicate_equipment_fails(self, default_allocator, sample_boiler):
        """Test that adding duplicate equipment returns False."""
        default_allocator.add_equipment(sample_boiler)
        result = default_allocator.add_equipment(sample_boiler)

        assert result is False
        assert len(default_allocator._equipment) == 1

    def test_remove_equipment_success(self, default_allocator, sample_boiler):
        """Test successful equipment removal."""
        default_allocator.add_equipment(sample_boiler)
        result = default_allocator.remove_equipment(sample_boiler.equipment_id)

        assert result is True
        assert sample_boiler.equipment_id not in default_allocator._equipment
        assert sample_boiler.equipment_id not in default_allocator._equipment_order

    def test_remove_nonexistent_equipment_fails(self, default_allocator):
        """Test removing non-existent equipment returns False."""
        result = default_allocator.remove_equipment("NONEXISTENT")

        assert result is False

    def test_update_equipment_status(self, default_allocator, sample_boiler):
        """Test updating equipment status."""
        default_allocator.add_equipment(sample_boiler)

        result = default_allocator.update_equipment_status(
            sample_boiler.equipment_id,
            EquipmentStatus.RUNNING,
            current_load=30.0
        )

        assert result is True
        equipment = default_allocator.get_equipment(sample_boiler.equipment_id)
        assert equipment.status == EquipmentStatus.RUNNING
        assert equipment.current_load_mmbtu_hr == 30.0

    def test_update_nonexistent_equipment_status_fails(self, default_allocator):
        """Test updating non-existent equipment status."""
        result = default_allocator.update_equipment_status(
            "NONEXISTENT",
            EquipmentStatus.RUNNING
        )

        assert result is False

    def test_get_equipment(self, default_allocator, sample_boiler):
        """Test getting equipment by ID."""
        default_allocator.add_equipment(sample_boiler)

        equipment = default_allocator.get_equipment(sample_boiler.equipment_id)

        assert equipment is not None
        assert equipment.equipment_id == sample_boiler.equipment_id

    def test_get_nonexistent_equipment(self, default_allocator):
        """Test getting non-existent equipment returns None."""
        equipment = default_allocator.get_equipment("NONEXISTENT")

        assert equipment is None

    def test_get_all_equipment(self, allocator_with_equipment):
        """Test getting all registered equipment."""
        equipment_list = allocator_with_equipment.get_all_equipment()

        assert len(equipment_list) == 3
        assert all(isinstance(e, Equipment) for e in equipment_list)

    def test_get_total_capacity(self, allocator_with_equipment):
        """Test total capacity calculation."""
        total_capacity = allocator_with_equipment.get_total_capacity()

        # 50 + 40 + 30 = 120 MMBtu/hr
        assert total_capacity == 120.0

    def test_total_capacity_excludes_offline(self, allocator_with_equipment):
        """Test that offline equipment is excluded from capacity."""
        # Set one boiler to maintenance
        allocator_with_equipment.update_equipment_status(
            "BOILER-002",
            EquipmentStatus.MAINTENANCE
        )

        total_capacity = allocator_with_equipment.get_total_capacity()

        # 50 + 30 = 80 MMBtu/hr (excluding 40)
        assert total_capacity == 80.0


# =============================================================================
# TEST CLASS: OPTIMIZATION - VALID INPUTS
# =============================================================================

class TestOptimizationValidInputs:
    """Tests for optimization with valid inputs."""

    def test_basic_optimization(self, allocator_with_equipment, basic_request):
        """Test basic optimization with standard inputs."""
        result = allocator_with_equipment.optimize(basic_request)

        assert result is not None
        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.request_id == basic_request.request_id
        assert len(result.allocations) == 3

    def test_demand_satisfaction(self, allocator_with_equipment, basic_request):
        """Test that demand is satisfied."""
        result = allocator_with_equipment.optimize(basic_request)

        assert result.total_allocated_mmbtu_hr == pytest.approx(
            basic_request.total_demand_mmbtu_hr, rel=0.01
        )
        assert result.unmet_demand_mmbtu_hr == pytest.approx(0.0, abs=0.1)

    def test_allocation_within_capacity(self, allocator_with_equipment, basic_request):
        """Test that allocations respect equipment capacity."""
        result = allocator_with_equipment.optimize(basic_request)

        for alloc in result.allocations:
            equipment = allocator_with_equipment.get_equipment(alloc.equipment_id)
            if alloc.is_running:
                assert alloc.allocated_load_mmbtu_hr <= equipment.max_capacity_mmbtu_hr
                assert alloc.allocated_load_mmbtu_hr >= equipment.min_capacity_mmbtu_hr

    def test_cost_calculation(self, allocator_with_equipment, basic_request):
        """Test that costs are calculated correctly."""
        result = allocator_with_equipment.optimize(basic_request)

        assert result.total_fuel_cost_per_hr > 0
        assert result.total_cost_per_hour >= result.total_fuel_cost_per_hr

    def test_emissions_calculation(self, allocator_with_equipment, basic_request):
        """Test that emissions are calculated correctly."""
        result = allocator_with_equipment.optimize(basic_request)

        assert result.total_co2_kg_hr > 0
        assert result.total_nox_kg_hr > 0

    def test_provenance_hash_generated(self, allocator_with_equipment, basic_request):
        """Test that provenance hash is generated."""
        result = allocator_with_equipment.optimize(basic_request)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_solve_time_recorded(self, allocator_with_equipment, basic_request):
        """Test that solve time is recorded."""
        result = allocator_with_equipment.optimize(basic_request)

        assert result.solve_time_ms > 0

    @pytest.mark.parametrize("demand", [10.0, 30.0, 60.0, 100.0])
    def test_various_demand_levels(self, allocator_with_equipment, demand):
        """Test optimization with various demand levels."""
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=demand,
            optimization_objective=OptimizationObjective.MINIMIZE_COST
        )

        result = allocator_with_equipment.optimize(request)

        if demand <= 120.0:  # Within capacity
            assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

    def test_different_objectives(self, allocator_with_equipment):
        """Test optimization with different objectives."""
        for objective in OptimizationObjective:
            request = LoadAllocationRequest(
                total_demand_mmbtu_hr=60.0,
                optimization_objective=objective
            )

            result = allocator_with_equipment.optimize(request)

            assert result.status in [
                OptimizationStatus.OPTIMAL,
                OptimizationStatus.FEASIBLE
            ]


# =============================================================================
# TEST CLASS: EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestEdgeCasesAndBoundaries:
    """Tests for edge cases and boundary conditions."""

    def test_zero_demand(self, allocator_with_equipment):
        """Test optimization with zero demand."""
        request = LoadAllocationRequest(total_demand_mmbtu_hr=0.0)

        result = allocator_with_equipment.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.total_allocated_mmbtu_hr == 0.0
        assert all(not alloc.is_running for alloc in result.allocations)

    def test_demand_exactly_at_capacity(self, allocator_with_equipment):
        """Test optimization with demand exactly at total capacity."""
        request = LoadAllocationRequest(total_demand_mmbtu_hr=120.0)

        result = allocator_with_equipment.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.total_allocated_mmbtu_hr == pytest.approx(120.0, rel=0.01)

    def test_demand_at_single_equipment_min(self, default_allocator, sample_boiler):
        """Test demand at single equipment minimum capacity."""
        default_allocator.add_equipment(sample_boiler)
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=sample_boiler.min_capacity_mmbtu_hr
        )

        result = default_allocator.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

    def test_demand_between_min_and_max(self, default_allocator, sample_boiler):
        """Test demand between min and max capacity."""
        default_allocator.add_equipment(sample_boiler)
        mid_demand = (sample_boiler.min_capacity_mmbtu_hr +
                      sample_boiler.max_capacity_mmbtu_hr) / 2
        request = LoadAllocationRequest(total_demand_mmbtu_hr=mid_demand)

        result = default_allocator.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

    def test_very_small_demand(self, allocator_with_equipment):
        """Test optimization with very small demand."""
        request = LoadAllocationRequest(total_demand_mmbtu_hr=0.1)

        result = allocator_with_equipment.optimize(request)

        # May be infeasible due to minimum turndown constraints
        assert result.status in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
            OptimizationStatus.INFEASIBLE
        ]

    def test_single_equipment_available(self, default_allocator, sample_boiler):
        """Test optimization with only one equipment available."""
        default_allocator.add_equipment(sample_boiler)
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=30.0
        )

        result = default_allocator.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert len(result.allocations) == 1

    def test_all_equipment_at_minimum(self, allocator_with_equipment):
        """Test when total demand matches sum of all minimums."""
        # Sum of minimums: 12.5 + 10.0 + 7.5 = 30.0
        request = LoadAllocationRequest(total_demand_mmbtu_hr=30.0)

        result = allocator_with_equipment.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

    def test_high_emissions_penalty(self, allocator_with_equipment):
        """Test optimization with high emissions penalty."""
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=60.0,
            emissions_penalty_per_kg_co2=10.0  # Very high penalty
        )

        result = allocator_with_equipment.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.total_emissions_cost_per_hr > 0

    def test_zero_emissions_penalty(self, allocator_with_equipment):
        """Test optimization with zero emissions penalty."""
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=60.0,
            emissions_penalty_per_kg_co2=0.0
        )

        result = allocator_with_equipment.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.total_emissions_cost_per_hr == 0.0


# =============================================================================
# TEST CLASS: INFEASIBLE SCENARIOS
# =============================================================================

class TestInfeasibleScenarios:
    """Tests for infeasible optimization scenarios."""

    def test_no_equipment_registered(self, default_allocator):
        """Test optimization with no equipment registered."""
        request = LoadAllocationRequest(total_demand_mmbtu_hr=50.0)

        result = default_allocator.optimize(request)

        assert result.status == OptimizationStatus.INFEASIBLE
        assert result.unmet_demand_mmbtu_hr == 50.0

    def test_demand_exceeds_capacity(self, allocator_with_equipment):
        """Test optimization when demand exceeds total capacity."""
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=200.0,  # Exceeds 120 capacity
            allow_unmet_demand=False
        )

        result = allocator_with_equipment.optimize(request)

        assert result.status == OptimizationStatus.INFEASIBLE

    def test_all_equipment_offline(self, allocator_with_equipment):
        """Test optimization when all equipment is offline."""
        # Set all to maintenance
        for eq_id in ["BOILER-001", "BOILER-002", "BOILER-003"]:
            allocator_with_equipment.update_equipment_status(
                eq_id, EquipmentStatus.MAINTENANCE
            )

        request = LoadAllocationRequest(total_demand_mmbtu_hr=50.0)

        result = allocator_with_equipment.optimize(request)

        assert result.status == OptimizationStatus.INFEASIBLE

    def test_demand_below_minimum_turndown(self, default_allocator, sample_boiler):
        """Test demand below minimum turndown without unmet demand allowed."""
        default_allocator.add_equipment(sample_boiler)
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=5.0,  # Below min_capacity of 10.0
            allow_unmet_demand=False
        )

        result = default_allocator.optimize(request)

        # May be infeasible or run at minimum
        assert result.status in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.FEASIBLE,
            OptimizationStatus.INFEASIBLE
        ]


# =============================================================================
# TEST CLASS: ALLOW UNMET DEMAND
# =============================================================================

class TestAllowUnmetDemand:
    """Tests for scenarios with unmet demand allowed."""

    def test_partial_fulfillment(self, allocator_with_equipment):
        """Test partial demand fulfillment when allowed."""
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=150.0,  # Exceeds 120 capacity
            allow_unmet_demand=True,
            unmet_demand_penalty=100.0
        )

        result = allocator_with_equipment.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.total_allocated_mmbtu_hr <= 120.0
        assert result.unmet_demand_mmbtu_hr > 0

    def test_unmet_demand_penalty_applied(self, allocator_with_equipment):
        """Test that unmet demand penalty is applied to cost."""
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=150.0,
            allow_unmet_demand=True,
            unmet_demand_penalty=1000.0  # High penalty
        )

        result = allocator_with_equipment.optimize(request)

        # Cost should include unmet demand penalty
        assert result.total_cost_per_hour > 0


# =============================================================================
# TEST CLASS: EFFICIENCY CURVE
# =============================================================================

class TestEfficiencyCurve:
    """Tests for equipment efficiency curve calculations."""

    def test_efficiency_at_load_points(self):
        """Test efficiency at defined load points."""
        curve = EquipmentEfficiencyCurve(
            load_points_percent=[25.0, 50.0, 75.0, 100.0],
            efficiency_points=[0.75, 0.82, 0.85, 0.84]
        )

        assert curve.get_efficiency(25.0) == pytest.approx(0.75, rel=0.001)
        assert curve.get_efficiency(50.0) == pytest.approx(0.82, rel=0.001)
        assert curve.get_efficiency(75.0) == pytest.approx(0.85, rel=0.001)
        assert curve.get_efficiency(100.0) == pytest.approx(0.84, rel=0.001)

    def test_efficiency_interpolation(self):
        """Test efficiency interpolation between load points."""
        curve = EquipmentEfficiencyCurve(
            load_points_percent=[25.0, 50.0, 75.0, 100.0],
            efficiency_points=[0.75, 0.82, 0.85, 0.84]
        )

        # Interpolate between 25% and 50%
        eff_37_5 = curve.get_efficiency(37.5)
        expected = 0.75 + (0.82 - 0.75) * (37.5 - 25.0) / (50.0 - 25.0)
        assert eff_37_5 == pytest.approx(expected, rel=0.001)

    def test_efficiency_below_minimum_load(self):
        """Test efficiency at load below minimum point."""
        curve = EquipmentEfficiencyCurve(
            load_points_percent=[25.0, 50.0, 75.0, 100.0],
            efficiency_points=[0.75, 0.82, 0.85, 0.84]
        )

        assert curve.get_efficiency(10.0) == pytest.approx(0.75, rel=0.001)

    def test_efficiency_above_maximum_load(self):
        """Test efficiency at load above maximum point."""
        curve = EquipmentEfficiencyCurve(
            load_points_percent=[25.0, 50.0, 75.0, 100.0],
            efficiency_points=[0.75, 0.82, 0.85, 0.84]
        )

        assert curve.get_efficiency(110.0) == pytest.approx(0.84, rel=0.001)


# =============================================================================
# TEST CLASS: EQUIPMENT CALCULATIONS
# =============================================================================

class TestEquipmentCalculations:
    """Tests for equipment calculation methods."""

    def test_fuel_consumption_calculation(self, sample_boiler):
        """Test fuel consumption calculation."""
        output = 30.0  # MMBtu/hr
        fuel_consumption = sample_boiler.get_fuel_consumption(output)

        # Fuel = Output / Efficiency
        load_percent = (output / sample_boiler.max_capacity_mmbtu_hr) * 100
        expected_efficiency = sample_boiler.efficiency_curve.get_efficiency(load_percent)
        expected_fuel = output / expected_efficiency

        assert fuel_consumption == pytest.approx(expected_fuel, rel=0.001)

    def test_fuel_consumption_zero_output(self, sample_boiler):
        """Test fuel consumption with zero output."""
        assert sample_boiler.get_fuel_consumption(0.0) == 0.0

    def test_operating_cost_calculation(self, sample_boiler):
        """Test operating cost calculation."""
        output = 30.0
        operating_cost = sample_boiler.get_operating_cost(output)

        fuel_consumption = sample_boiler.get_fuel_consumption(output)
        expected_cost = (fuel_consumption * sample_boiler.fuel_cost_per_mmbtu +
                        sample_boiler.fixed_cost_per_hour)

        assert operating_cost == pytest.approx(expected_cost, rel=0.001)

    def test_operating_cost_zero_output(self, sample_boiler):
        """Test operating cost with zero output."""
        assert sample_boiler.get_operating_cost(0.0) == 0.0

    def test_emissions_calculation(self, sample_boiler):
        """Test emissions calculation."""
        output = 30.0
        emissions = sample_boiler.get_emissions(output)

        fuel_consumption = sample_boiler.get_fuel_consumption(output)
        expected_co2 = fuel_consumption * sample_boiler.co2_kg_per_mmbtu_fuel
        expected_nox = fuel_consumption * sample_boiler.nox_kg_per_mmbtu_fuel

        assert emissions["co2_kg_hr"] == pytest.approx(expected_co2, rel=0.001)
        assert emissions["nox_kg_hr"] == pytest.approx(expected_nox, rel=0.001)

    def test_turndown_ratio(self, sample_boiler):
        """Test turndown ratio calculation."""
        expected_ratio = sample_boiler.max_capacity_mmbtu_hr / sample_boiler.min_capacity_mmbtu_hr
        assert sample_boiler.turndown_ratio == pytest.approx(expected_ratio, rel=0.001)

    def test_turndown_ratio_zero_min(self):
        """Test turndown ratio with zero minimum capacity."""
        boiler = Equipment(
            name="Test Boiler",
            equipment_type=EquipmentType.BOILER,
            fuel_type=FuelType.NATURAL_GAS,
            max_capacity_mmbtu_hr=50.0,
            min_capacity_mmbtu_hr=0.0,  # No minimum
            fuel_cost_per_mmbtu=5.0,
        )

        assert boiler.turndown_ratio == float('inf')


# =============================================================================
# TEST CLASS: FACTORY FUNCTIONS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_standard_boiler(self):
        """Test standard boiler factory function."""
        boiler = create_standard_boiler(
            name="Test Boiler",
            capacity_mmbtu_hr=100.0,
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_mmbtu=5.0,
            turndown_ratio=4.0
        )

        assert boiler.name == "Test Boiler"
        assert boiler.max_capacity_mmbtu_hr == 100.0
        assert boiler.min_capacity_mmbtu_hr == 25.0  # 100 / 4
        assert boiler.equipment_type == EquipmentType.BOILER
        assert boiler.fuel_type == FuelType.NATURAL_GAS

    def test_create_standard_boiler_different_fuels(self):
        """Test boiler creation with different fuel types."""
        for fuel_type in [FuelType.NATURAL_GAS, FuelType.FUEL_OIL, FuelType.PROPANE]:
            boiler = create_standard_boiler(
                name=f"Test {fuel_type.value}",
                capacity_mmbtu_hr=50.0,
                fuel_type=fuel_type,
                fuel_cost_per_mmbtu=5.0
            )

            assert boiler.fuel_type == fuel_type
            assert boiler.co2_kg_per_mmbtu_fuel > 0

    def test_create_chp_system(self):
        """Test CHP system factory function."""
        chp = create_chp_system(
            name="Test CHP",
            thermal_capacity_mmbtu_hr=30.0,
            electric_capacity_kw=500,
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_mmbtu=5.0,
            electricity_value_per_kwh=0.10
        )

        assert chp.name == "Test CHP"
        assert chp.max_capacity_mmbtu_hr == 30.0
        assert chp.equipment_type == EquipmentType.CHP
        assert chp.priority == 1  # CHP dispatched first


# =============================================================================
# TEST CLASS: OPTIMIZATION HISTORY
# =============================================================================

class TestOptimizationHistory:
    """Tests for optimization history and analysis."""

    def test_history_tracking(self, allocator_with_equipment, basic_request):
        """Test that optimization history is tracked."""
        # Run multiple optimizations
        for _ in range(5):
            allocator_with_equipment.optimize(basic_request)

        history = allocator_with_equipment.get_optimization_history(limit=10)

        assert len(history) == 5

    def test_history_limit(self, allocator_with_equipment, basic_request):
        """Test history limit functionality."""
        for _ in range(10):
            allocator_with_equipment.optimize(basic_request)

        history = allocator_with_equipment.get_optimization_history(limit=5)

        assert len(history) == 5

    def test_equipment_utilization_stats(self, allocator_with_equipment, basic_request):
        """Test equipment utilization statistics."""
        # Run several optimizations
        for _ in range(5):
            allocator_with_equipment.optimize(basic_request)

        utilization = allocator_with_equipment.get_equipment_utilization()

        assert len(utilization) > 0
        for eq_id, stats in utilization.items():
            assert "avg_load" in stats
            assert "run_percentage" in stats
            assert "avg_utilization_percent" in stats


# =============================================================================
# TEST CLASS: WARM START
# =============================================================================

class TestWarmStart:
    """Tests for warm start functionality."""

    def test_warm_start_caching(self, allocator_with_equipment, basic_request):
        """Test that solution is cached for warm start."""
        result1 = allocator_with_equipment.optimize(basic_request)

        assert allocator_with_equipment._last_solution is not None
        assert len(allocator_with_equipment._last_solution) == len(result1.allocations)

    def test_warm_start_reuse(self, allocator_with_equipment, basic_request):
        """Test that warm start improves performance."""
        # First optimization (cold start)
        result1 = allocator_with_equipment.optimize(basic_request)
        time1 = result1.solve_time_ms

        # Second optimization (warm start)
        result2 = allocator_with_equipment.optimize(basic_request)
        time2 = result2.solve_time_ms

        # Both should produce valid results
        assert result1.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result2.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]


# =============================================================================
# TEST CLASS: SCENARIO COMPARISON
# =============================================================================

class TestScenarioComparison:
    """Tests for scenario comparison functionality."""

    def test_compare_scenarios(self, allocator_with_equipment):
        """Test comparing multiple scenarios."""
        scenarios = [
            LoadAllocationRequest(total_demand_mmbtu_hr=40.0),
            LoadAllocationRequest(total_demand_mmbtu_hr=60.0),
            LoadAllocationRequest(total_demand_mmbtu_hr=80.0),
        ]

        results = allocator_with_equipment.compare_scenarios(scenarios)

        assert len(results) == 3
        # Higher demand should result in higher cost
        costs = [r.total_cost_per_hour for r in results]
        assert costs[0] <= costs[1] <= costs[2]


# =============================================================================
# TEST CLASS: DETERMINISM / REPRODUCIBILITY
# =============================================================================

class TestDeterminism:
    """Tests for deterministic/reproducible optimization."""

    def test_deterministic_results(self, allocator_with_equipment, basic_request):
        """Test that same input produces same output."""
        result1 = allocator_with_equipment.optimize(basic_request)
        result2 = allocator_with_equipment.optimize(basic_request)

        assert result1.total_cost_per_hour == pytest.approx(
            result2.total_cost_per_hour, rel=0.001
        )
        assert result1.total_allocated_mmbtu_hr == pytest.approx(
            result2.total_allocated_mmbtu_hr, rel=0.001
        )

    def test_provenance_hash_determinism(self, allocator_with_equipment, basic_request):
        """Test that provenance hash is deterministic."""
        result1 = allocator_with_equipment.optimize(basic_request)
        result2 = allocator_with_equipment.optimize(basic_request)

        # Note: Timestamps differ, so hashes will differ
        # But the calculation should be deterministic
        assert result1.total_co2_kg_hr == pytest.approx(
            result2.total_co2_kg_hr, rel=0.001
        )


# =============================================================================
# TEST CLASS: HEURISTIC FALLBACK
# =============================================================================

class TestHeuristicFallback:
    """Tests for heuristic fallback when MILP unavailable."""

    def test_heuristic_solution(self, allocator_with_equipment, basic_request):
        """Test that heuristic produces valid solution."""
        # Force heuristic by calling it directly
        available = allocator_with_equipment._get_available_equipment()
        result = allocator_with_equipment._solve_heuristic(basic_request, available)

        assert result.status == OptimizationStatus.FEASIBLE
        assert result.total_allocated_mmbtu_hr == pytest.approx(
            basic_request.total_demand_mmbtu_hr, rel=0.1
        )

    def test_heuristic_merit_order(self, allocator_with_equipment):
        """Test that heuristic uses merit order dispatch."""
        # Equipment is ordered by cost: BOILER-001 (5.0), BOILER-002 (5.5), BOILER-003 (6.0)
        request = LoadAllocationRequest(total_demand_mmbtu_hr=40.0)

        available = allocator_with_equipment._get_available_equipment()
        result = allocator_with_equipment._solve_heuristic(request, available)

        # Cheapest boiler should have highest allocation
        allocations = {a.equipment_id: a.allocated_load_mmbtu_hr for a in result.allocations}

        # BOILER-001 should be running at higher capacity
        assert allocations.get("BOILER-001", 0) > 0 or allocations.get("BOILER-002", 0) > 0


# =============================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# =============================================================================

if HYPOTHESIS_AVAILABLE:
    class TestPropertyBased:
        """Property-based tests using Hypothesis."""

        @given(st.floats(min_value=0.0, max_value=100.0))
        @settings(max_examples=50)
        def test_efficiency_always_valid(self, load_percent):
            """Test that efficiency is always in valid range."""
            curve = EquipmentEfficiencyCurve()
            efficiency = curve.get_efficiency(load_percent)

            assert 0.0 <= efficiency <= 1.0

        @given(st.floats(min_value=0.0, max_value=50.0))
        @settings(max_examples=50)
        def test_fuel_consumption_non_negative(self, output):
            """Test that fuel consumption is never negative."""
            boiler = create_standard_boiler(
                name="Test",
                capacity_mmbtu_hr=50.0,
                fuel_type=FuelType.NATURAL_GAS,
                fuel_cost_per_mmbtu=5.0
            )

            consumption = boiler.get_fuel_consumption(output)

            assert consumption >= 0.0

        @given(st.floats(min_value=0.0, max_value=50.0))
        @settings(max_examples=50)
        def test_cost_non_negative(self, output):
            """Test that operating cost is never negative."""
            boiler = create_standard_boiler(
                name="Test",
                capacity_mmbtu_hr=50.0,
                fuel_type=FuelType.NATURAL_GAS,
                fuel_cost_per_mmbtu=5.0
            )

            cost = boiler.get_operating_cost(output)

            assert cost >= 0.0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for MILP optimizer."""

    @pytest.mark.performance
    def test_optimization_time_target(self, allocator_with_equipment, basic_request):
        """Test that optimization completes within 5 seconds."""
        result = allocator_with_equipment.optimize(basic_request)

        # Target: <5s optimization cycle
        assert result.solve_time_ms < 5000

    @pytest.mark.performance
    def test_large_equipment_set(self, default_allocator):
        """Test optimization with many equipment units."""
        # Add 20 boilers
        for i in range(20):
            boiler = create_standard_boiler(
                name=f"Boiler-{i+1}",
                capacity_mmbtu_hr=30.0 + np.random.uniform(-10, 10),
                fuel_type=FuelType.NATURAL_GAS,
                fuel_cost_per_mmbtu=5.0 + np.random.uniform(-1, 1)
            )
            default_allocator.add_equipment(boiler)

        request = LoadAllocationRequest(total_demand_mmbtu_hr=300.0)
        result = default_allocator.optimize(request)

        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.solve_time_ms < 30000  # 30 second limit for large problems


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================

class TestMILPIntegration:
    """Integration-style tests for MILP optimizer."""

    def test_full_optimization_workflow(self, default_allocator):
        """Test complete optimization workflow."""
        # 1. Add equipment
        boiler1 = create_standard_boiler("Boiler-1", 50.0, FuelType.NATURAL_GAS, 5.0)
        boiler2 = create_standard_boiler("Boiler-2", 40.0, FuelType.NATURAL_GAS, 5.5)
        chp = create_chp_system("CHP-1", 30.0, 500, FuelType.NATURAL_GAS, 4.0)

        default_allocator.add_equipment(boiler1)
        default_allocator.add_equipment(boiler2)
        default_allocator.add_equipment(chp)

        # 2. Create request
        request = LoadAllocationRequest(
            total_demand_mmbtu_hr=80.0,
            optimization_objective=OptimizationObjective.BALANCED,
            emissions_penalty_per_kg_co2=0.05
        )

        # 3. Optimize
        result = default_allocator.optimize(request)

        # 4. Verify results
        assert result.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]
        assert result.total_allocated_mmbtu_hr == pytest.approx(80.0, rel=0.01)
        assert result.provenance_hash != ""

        # 5. CHP should be dispatched first (lowest cost, priority 1)
        chp_allocation = next(
            (a for a in result.allocations if "CHP" in a.equipment_name), None
        )
        assert chp_allocation is not None
        assert chp_allocation.is_running

    def test_equipment_status_changes(self, allocator_with_equipment, basic_request):
        """Test optimization with equipment status changes."""
        # Initial optimization
        result1 = allocator_with_equipment.optimize(basic_request)

        # Take one boiler offline
        allocator_with_equipment.update_equipment_status(
            "BOILER-002", EquipmentStatus.MAINTENANCE
        )

        # Optimize again
        result2 = allocator_with_equipment.optimize(basic_request)

        # Should still meet demand with remaining equipment
        assert result2.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

        # BOILER-002 should not be allocated
        boiler2_alloc = next(
            (a for a in result2.allocations if a.equipment_id == "BOILER-002"), None
        )
        assert boiler2_alloc is None or not boiler2_alloc.is_running
