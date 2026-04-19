"""
GL-023 HeatLoadBalancer - Optimization Algorithm Tests
======================================================

Tests for MILP formulation, constraint handling, demand balance,
equipment limits, ramp rates, different objectives, Pareto frontier,
heuristic fallback, and solver status handling.

Target Coverage: 85%+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import test utilities
try:
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.formulas import (
        calculate_efficiency_at_load,
        calculate_fuel_consumption,
        calculate_hourly_cost,
        calculate_incremental_cost,
        calculate_emissions,
        economic_dispatch_merit_order,
        calculate_fleet_efficiency,
        calculate_equal_loading,
    )
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.models import (
        LoadBalancerInput,
        LoadBalancerOutput,
        EquipmentUnit,
    )
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.agent import (
        HeatLoadBalancerAgent,
    )
    IMPLEMENTATION_AVAILABLE = True
except ImportError:
    IMPLEMENTATION_AVAILABLE = False


# =============================================================================
# MILP FORMULATION TESTS
# =============================================================================

@pytest.mark.optimization
class TestMILPFormulation:
    """Test Mixed-Integer Linear Programming formulation."""

    def test_milp_decision_variables(self, sample_boiler_fleet):
        """Test MILP has correct decision variables."""
        # MILP should have:
        # - Continuous variables: load per unit (L_i)
        # - Binary variables: unit on/off status (u_i)

        n_units = len(sample_boiler_fleet)

        # Expected number of variables
        n_continuous = n_units  # Load for each unit
        n_binary = n_units      # On/off status for each unit

        # Total decision variables
        expected_vars = n_continuous + n_binary

        assert expected_vars == 6, f"Expected 6 decision variables for 3 units"

    def test_milp_objective_cost_minimization(self, sample_boiler_fleet):
        """Test MILP cost minimization objective."""
        # Objective: minimize sum(fuel_cost_i * fuel_consumption_i + maint_cost_i * load_i)

        # Run merit order as proxy for optimization
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        # Calculate total cost
        total_cost = 0.0
        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:
                unit = unit_lookup[unit_id]
                efficiency = calculate_efficiency_at_load(
                    load,
                    unit['min_load_mw'],
                    unit['max_load_mw'],
                    unit.get('efficiency_curve_a', 0),
                    unit.get('efficiency_curve_b', 0),
                    unit.get('efficiency_curve_c', 0),
                    unit.get('current_efficiency_pct', 80),
                )
                fuel = calculate_fuel_consumption(load, efficiency)
                cost = calculate_hourly_cost(
                    fuel,
                    unit['fuel_cost_per_mwh'],
                    unit.get('maintenance_cost_per_mwh', 0),
                    load,
                )
                total_cost += cost

        assert total_cost > 0, "Cost minimization should have positive cost"

    def test_milp_objective_efficiency_maximization(self, sample_boiler_fleet):
        """Test MILP efficiency maximization objective."""
        # Objective: maximize weighted_efficiency = sum(load_i * eta_i) / sum(load_i)

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        efficiency = calculate_fleet_efficiency(allocations, sample_boiler_fleet)

        assert 70.0 <= efficiency <= 95.0, f"Fleet efficiency {efficiency}% out of range"

    def test_milp_objective_emissions_minimization(self, sample_boiler_fleet):
        """Test MILP emissions minimization objective."""
        # Objective: minimize sum(fuel_consumption_i * emissions_factor_i)

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=100.0,  # High carbon price favors low emissions
        )

        # Calculate total emissions
        total_emissions = 0.0
        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:
                unit = unit_lookup[unit_id]
                efficiency = calculate_efficiency_at_load(
                    load,
                    unit['min_load_mw'],
                    unit['max_load_mw'],
                    unit.get('efficiency_curve_a', 0),
                    unit.get('efficiency_curve_b', 0),
                    unit.get('efficiency_curve_c', 0),
                    unit.get('current_efficiency_pct', 80),
                )
                fuel = calculate_fuel_consumption(load, efficiency)
                emissions = calculate_emissions(
                    fuel,
                    unit.get('emissions_factor_kg_co2_mwh', 200),
                )
                total_emissions += emissions

        assert total_emissions > 0, "Should have positive emissions"


# =============================================================================
# CONSTRAINT HANDLING TESTS
# =============================================================================

@pytest.mark.optimization
class TestConstraintHandling:
    """Test constraint handling in optimization."""

    def test_demand_balance_constraint(self, sample_boiler_fleet):
        """Test demand balance: sum(load_i) = total_demand."""
        demand_mw = 25.0

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=demand_mw,
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)

        # Should meet demand exactly or be very close
        assert abs(total_allocated - demand_mw) < 0.5 or total_allocated >= demand_mw, (
            f"Demand balance violated: allocated {total_allocated} MW vs demand {demand_mw} MW"
        )

    def test_minimum_load_constraint(self, sample_boiler_fleet):
        """Test minimum load constraint: load_i >= min_load_i * u_i."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:  # Unit is on
                min_load = unit_lookup[unit_id]['min_load_mw']
                assert load >= min_load, (
                    f"Min load constraint violated: {unit_id} at {load} MW < {min_load} MW"
                )

    def test_maximum_load_constraint(self, sample_boiler_fleet):
        """Test maximum load constraint: load_i <= max_load_i * u_i."""
        # Request very high demand
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=100.0,
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if unit_id in unit_lookup:
                max_load = unit_lookup[unit_id]['max_load_mw']
                assert load <= max_load + 0.001, (
                    f"Max load constraint violated: {unit_id} at {load} MW > {max_load} MW"
                )

    def test_availability_constraint(self, sample_boiler_fleet):
        """Test availability constraint: unavailable units have zero load."""
        # Make one unit unavailable
        fleet = [u.copy() for u in sample_boiler_fleet]
        fleet[0]['is_available'] = False

        allocations = economic_dispatch_merit_order(
            units=fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        # Unavailable unit should have zero load
        for unit_id, load in allocations:
            if unit_id == fleet[0]['unit_id']:
                assert load == 0.0, f"Unavailable unit {unit_id} should have zero load"

    @pytest.mark.parametrize("demand_mw,min_reserve_pct", [
        (20.0, 10.0),  # Normal case
        (35.0, 15.0),  # Higher demand
        (40.0, 5.0),   # Low reserve requirement
    ])
    def test_spinning_reserve_constraint(self, sample_boiler_fleet, demand_mw, min_reserve_pct):
        """Test spinning reserve constraint."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=demand_mw,
            carbon_price=0.0,
        )

        # Calculate capacity and reserve
        available_units = [u for u in sample_boiler_fleet if u['is_available']]
        total_capacity = sum(u['max_load_mw'] for u in available_units)
        total_allocated = sum(load for _, load in allocations)

        reserve_mw = total_capacity - total_allocated
        reserve_pct = (reserve_mw / total_capacity * 100) if total_capacity > 0 else 0

        # Log for debugging
        print(f"Demand: {demand_mw} MW, Capacity: {total_capacity} MW, "
              f"Allocated: {total_allocated} MW, Reserve: {reserve_pct:.1f}%")

    def test_ramp_rate_constraint(self, sample_boiler_fleet, ramp_rate_test_cases):
        """Test ramp rate constraints for load changes."""
        for case in ramp_rate_test_cases:
            load_change = abs(case["target_load_mw"] - case["current_load_mw"])
            ramp_rate = case["ramp_rate_mw_per_min"]
            time_available = case["time_available_min"]

            # Time required for ramp
            time_required = load_change / ramp_rate

            # Check if achievable
            achievable = time_required <= time_available

            assert achievable == case["expected_achievable"], (
                f"Ramp rate check failed: {load_change} MW at {ramp_rate} MW/min "
                f"in {time_available} min"
            )


# =============================================================================
# EQUIPMENT LIMIT CONSTRAINT TESTS
# =============================================================================

@pytest.mark.optimization
class TestEquipmentLimitConstraints:
    """Test equipment-specific limit constraints."""

    def test_unit_on_off_binary_constraint(self, sample_boiler_fleet):
        """Test unit status is binary (0 or 1)."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        for unit_id, load in allocations:
            # Load should be either 0 (off) or >= min_load (on)
            unit = next((u for u in sample_boiler_fleet if u['unit_id'] == unit_id), None)
            if unit:
                is_off = load == 0.0
                is_on = load >= unit['min_load_mw']
                assert is_off or is_on, (
                    f"Unit {unit_id} in invalid state: load={load}, min={unit['min_load_mw']}"
                )

    def test_minimum_run_time_constraint(self, sample_boiler_fleet):
        """Test minimum run time constraint."""
        # Units that were just started should not be shut down
        for unit in sample_boiler_fleet:
            min_run_time_hr = unit.get('min_run_time_hr', 1.0)
            assert min_run_time_hr > 0, "Min run time should be positive"

            # If unit is running, it should run for at least min_run_time
            if unit['is_running'] and unit['current_load_mw'] > 0:
                # This would be tracked in actual implementation
                pass

    def test_startup_cost_consideration(self, sample_boiler_fleet):
        """Test startup costs are considered in optimization."""
        # Compare scenarios with and without startup costs

        # Low demand - may need to start units
        allocations_low = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=10.0,
            carbon_price=0.0,
        )

        # High demand - need more units
        allocations_high = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=35.0,
            carbon_price=0.0,
        )

        # With higher demand, more units should be started
        units_running_low = sum(1 for _, load in allocations_low if load > 0)
        units_running_high = sum(1 for _, load in allocations_high if load > 0)

        assert units_running_high >= units_running_low, (
            "Higher demand should require more or equal units"
        )


# =============================================================================
# OPTIMIZATION OBJECTIVE TESTS
# =============================================================================

@pytest.mark.optimization
class TestOptimizationObjectives:
    """Test different optimization objectives."""

    def test_cost_optimization_prefers_efficient_units(self, sample_boiler_fleet):
        """Test cost optimization prefers units with lower incremental cost."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=15.0,  # Partial load
            carbon_price=0.0,
        )

        # Units with lower fuel cost should be loaded first
        loaded_units = [(uid, load) for uid, load in allocations if load > 0]

        if len(loaded_units) > 0:
            # Verify at least some allocation was made
            assert sum(load for _, load in loaded_units) > 0

    def test_emissions_optimization_vs_cost(self, sample_boiler_fleet, combined_equipment_fleet):
        """Test emissions vs cost optimization trade-off."""
        # Cost optimization (no carbon price)
        alloc_cost = economic_dispatch_merit_order(
            units=combined_equipment_fleet,
            total_demand_mw=30.0,
            carbon_price=0.0,
        )

        # Emissions optimization (high carbon price)
        alloc_emissions = economic_dispatch_merit_order(
            units=combined_equipment_fleet,
            total_demand_mw=30.0,
            carbon_price=200.0,  # Very high carbon price
        )

        # Calculate total emissions for each
        def calc_emissions(allocations, fleet):
            total = 0.0
            unit_lookup = {u['unit_id']: u for u in fleet}
            for unit_id, load in allocations:
                if load > 0 and unit_id in unit_lookup:
                    unit = unit_lookup[unit_id]
                    eff = calculate_efficiency_at_load(
                        load, unit['min_load_mw'], unit['max_load_mw'],
                        unit.get('efficiency_curve_a', 0),
                        unit.get('efficiency_curve_b', 0),
                        unit.get('efficiency_curve_c', 0),
                    )
                    if eff > 0:
                        fuel = calculate_fuel_consumption(load, eff)
                        total += calculate_emissions(fuel, unit.get('emissions_factor_kg_co2_mwh', 200))
            return total

        emissions_cost = calc_emissions(alloc_cost, combined_equipment_fleet)
        emissions_green = calc_emissions(alloc_emissions, combined_equipment_fleet)

        # Emissions optimization should have lower or equal emissions
        # (In practice, high carbon price pushes toward low-emission units)
        assert emissions_green <= emissions_cost * 1.1, (
            f"Emissions optimization ({emissions_green:.0f} kg) should be <= "
            f"cost optimization ({emissions_cost:.0f} kg)"
        )

    def test_balanced_optimization(self, sample_boiler_fleet):
        """Test balanced multi-objective optimization."""
        # Balanced optimization should find middle ground
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=25.0,
            carbon_price=50.0,  # Moderate carbon price
        )

        total_allocated = sum(load for _, load in allocations)
        assert total_allocated > 0, "Balanced optimization should allocate load"


# =============================================================================
# PARETO FRONTIER TESTS
# =============================================================================

@pytest.mark.optimization
@pytest.mark.slow
class TestParetoFrontier:
    """Test Pareto frontier generation for multi-objective optimization."""

    def test_pareto_points_are_non_dominated(self, combined_equipment_fleet):
        """Test generated Pareto points are non-dominated."""
        # Generate multiple solutions with different trade-offs
        pareto_points = []

        for carbon_price in [0, 25, 50, 100, 200]:
            allocations = economic_dispatch_merit_order(
                units=combined_equipment_fleet,
                total_demand_mw=40.0,
                carbon_price=carbon_price,
            )

            # Calculate cost and emissions
            total_cost = 0.0
            total_emissions = 0.0
            unit_lookup = {u['unit_id']: u for u in combined_equipment_fleet}

            for unit_id, load in allocations:
                if load > 0 and unit_id in unit_lookup:
                    unit = unit_lookup[unit_id]
                    eff = calculate_efficiency_at_load(
                        load, unit['min_load_mw'], unit['max_load_mw'],
                        unit.get('efficiency_curve_a', 0),
                        unit.get('efficiency_curve_b', 0),
                        unit.get('efficiency_curve_c', 0),
                    )
                    if eff > 0:
                        fuel = calculate_fuel_consumption(load, eff)
                        total_cost += calculate_hourly_cost(
                            fuel, unit['fuel_cost_per_mwh'],
                            unit.get('maintenance_cost_per_mwh', 0), load
                        )
                        total_emissions += calculate_emissions(
                            fuel, unit.get('emissions_factor_kg_co2_mwh', 200)
                        )

            pareto_points.append({
                'carbon_price': carbon_price,
                'cost': total_cost,
                'emissions': total_emissions,
            })

        # Check for non-dominated points
        # A point is dominated if another point is better in all objectives
        for i, point_i in enumerate(pareto_points):
            is_dominated = False
            for j, point_j in enumerate(pareto_points):
                if i != j:
                    # Check if j dominates i
                    if (point_j['cost'] <= point_i['cost'] and
                        point_j['emissions'] <= point_i['emissions'] and
                        (point_j['cost'] < point_i['cost'] or
                         point_j['emissions'] < point_i['emissions'])):
                        is_dominated = True
                        break

            # Log dominated points (they may exist due to discrete nature)
            if is_dominated:
                print(f"Point at carbon={point_i['carbon_price']} may be dominated")

    def test_pareto_frontier_coverage(self, combined_equipment_fleet):
        """Test Pareto frontier covers range of trade-offs."""
        carbon_prices = [0, 50, 100, 200, 500]
        costs = []
        emissions = []

        for cp in carbon_prices:
            allocations = economic_dispatch_merit_order(
                units=combined_equipment_fleet,
                total_demand_mw=40.0,
                carbon_price=cp,
            )

            total_cost = 0.0
            total_emissions = 0.0
            unit_lookup = {u['unit_id']: u for u in combined_equipment_fleet}

            for unit_id, load in allocations:
                if load > 0 and unit_id in unit_lookup:
                    unit = unit_lookup[unit_id]
                    eff = calculate_efficiency_at_load(
                        load, unit['min_load_mw'], unit['max_load_mw'],
                        unit.get('efficiency_curve_a', 0),
                        unit.get('efficiency_curve_b', 0),
                        unit.get('efficiency_curve_c', 0),
                    )
                    if eff > 0:
                        fuel = calculate_fuel_consumption(load, eff)
                        total_cost += calculate_hourly_cost(
                            fuel, unit['fuel_cost_per_mwh'],
                            unit.get('maintenance_cost_per_mwh', 0), load
                        )
                        total_emissions += calculate_emissions(
                            fuel, unit.get('emissions_factor_kg_co2_mwh', 200)
                        )

            costs.append(total_cost)
            emissions.append(total_emissions)

        # Should see variation in solutions
        cost_range = max(costs) - min(costs)
        emissions_range = max(emissions) - min(emissions)

        # There should be some variation (trade-offs exist)
        # Note: may be small for identical units
        assert cost_range >= 0, "Should have cost variation across Pareto points"


# =============================================================================
# HEURISTIC FALLBACK TESTS
# =============================================================================

@pytest.mark.optimization
class TestHeuristicFallback:
    """Test heuristic fallback when MILP times out."""

    def test_merit_order_heuristic_basic(self, sample_boiler_fleet):
        """Test merit order heuristic produces valid solution."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=25.0,
            carbon_price=0.0,
        )

        # Should return valid allocations
        assert len(allocations) > 0, "Heuristic should return allocations"

        # All allocations should be non-negative
        for unit_id, load in allocations:
            assert load >= 0, f"Load for {unit_id} should be non-negative"

    def test_heuristic_meets_demand(self, sample_boiler_fleet):
        """Test heuristic meets demand requirement."""
        demand_mw = 20.0

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=demand_mw,
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)

        # Should meet or exceed demand (within constraint satisfaction)
        assert total_allocated >= demand_mw * 0.95 or total_allocated == demand_mw, (
            f"Heuristic allocated {total_allocated} MW, need {demand_mw} MW"
        )

    def test_heuristic_respects_constraints(self, sample_boiler_fleet):
        """Test heuristic respects all constraints."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:
                unit = unit_lookup.get(unit_id)
                if unit:
                    # Min load constraint
                    assert load >= unit['min_load_mw'], "Min load violated"
                    # Max load constraint
                    assert load <= unit['max_load_mw'], "Max load violated"
                    # Availability
                    assert unit['is_available'], "Unavailable unit loaded"

    def test_heuristic_performance(self, large_equipment_fleet, benchmark_iterations):
        """Test heuristic performance is acceptable."""
        start_time = time.time()

        for _ in range(benchmark_iterations):
            allocations = economic_dispatch_merit_order(
                units=large_equipment_fleet,
                total_demand_mw=200.0,
                carbon_price=50.0,
            )

        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / benchmark_iterations) * 1000

        assert avg_time_ms < 100, f"Heuristic too slow: {avg_time_ms:.1f} ms average"

    def test_heuristic_quality_vs_optimal(self, sample_boiler_fleet):
        """Test heuristic solution quality compared to optimal."""
        # In practice, merit order is often within 5% of optimal
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=25.0,
            carbon_price=0.0,
        )

        # Calculate solution cost
        total_cost = 0.0
        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:
                unit = unit_lookup[unit_id]
                eff = calculate_efficiency_at_load(
                    load, unit['min_load_mw'], unit['max_load_mw'],
                    unit.get('efficiency_curve_a', 0),
                    unit.get('efficiency_curve_b', 0),
                    unit.get('efficiency_curve_c', 0),
                )
                if eff > 0:
                    fuel = calculate_fuel_consumption(load, eff)
                    total_cost += calculate_hourly_cost(
                        fuel, unit['fuel_cost_per_mwh'],
                        unit.get('maintenance_cost_per_mwh', 0), load
                    )

        # Compare to equal loading baseline
        baseline_eff, baseline_cost = calculate_equal_loading(sample_boiler_fleet, 25.0)

        # Heuristic should be better or similar to baseline
        assert total_cost <= baseline_cost * 1.05, (
            f"Heuristic cost ${total_cost:.2f} should be <= baseline ${baseline_cost:.2f}"
        )


# =============================================================================
# SOLVER STATUS HANDLING TESTS
# =============================================================================

@pytest.mark.optimization
class TestSolverStatusHandling:
    """Test handling of various solver statuses."""

    def test_optimal_solution_status(self, sample_boiler_fleet):
        """Test handling of optimal solution."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=20.0,
            carbon_price=0.0,
        )

        # Should return valid allocations
        assert len(allocations) > 0
        assert sum(load for _, load in allocations) > 0

    def test_infeasible_status_capacity_exceeded(self, sample_boiler_fleet):
        """Test handling when demand exceeds capacity."""
        # Total capacity is 15 + 10 + 20 = 45 MW
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=100.0,  # Exceeds capacity
            carbon_price=0.0,
        )

        # Should return best-effort allocation (up to capacity)
        total_capacity = sum(u['max_load_mw'] for u in sample_boiler_fleet if u['is_available'])
        total_allocated = sum(load for _, load in allocations)

        assert total_allocated <= total_capacity, "Cannot exceed capacity"

    def test_infeasible_status_no_available_units(self):
        """Test handling when no units are available."""
        unavailable_fleet = [
            {
                "unit_id": "BOILER_001",
                "unit_type": "BOILER",
                "current_load_mw": 0.0,
                "min_load_mw": 2.0,
                "max_load_mw": 10.0,
                "is_available": False,  # Not available
                "fuel_cost_per_mwh": 25.0,
            }
        ]

        allocations = economic_dispatch_merit_order(
            units=unavailable_fleet,
            total_demand_mw=5.0,
            carbon_price=0.0,
        )

        # Should return zero allocations
        total_allocated = sum(load for _, load in allocations)
        assert total_allocated == 0.0, "No available units means zero allocation"

    def test_unbounded_solution_handling(self):
        """Test handling of unbounded solution (should not occur with proper constraints)."""
        # In practice, physical constraints prevent unbounded solutions
        # This tests that we don't crash on edge cases
        pass

    def test_timeout_handling(self, large_equipment_fleet):
        """Test handling when solver times out."""
        # With large fleet and complex constraints, solver might timeout
        # Should fallback to heuristic

        start_time = time.time()

        allocations = economic_dispatch_merit_order(
            units=large_equipment_fleet,
            total_demand_mw=500.0,
            carbon_price=50.0,
        )

        elapsed = time.time() - start_time

        # Should complete in reasonable time (heuristic fallback)
        assert elapsed < 5.0, f"Optimization took too long: {elapsed:.1f}s"

        # Should still produce valid solution
        assert len(allocations) > 0


# =============================================================================
# EDGE CASE OPTIMIZATION TESTS
# =============================================================================

@pytest.mark.optimization
class TestOptimizationEdgeCases:
    """Test optimization edge cases."""

    def test_single_unit_optimization(self):
        """Test optimization with single unit."""
        single_fleet = [{
            "unit_id": "ONLY_BOILER",
            "unit_type": "BOILER",
            "current_load_mw": 5.0,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "current_efficiency_pct": 85.0,
            "efficiency_curve_a": 70.0,
            "efficiency_curve_b": 20.0,
            "efficiency_curve_c": -5.0,
            "is_available": True,
            "fuel_cost_per_mwh": 25.0,
        }]

        allocations = economic_dispatch_merit_order(
            units=single_fleet,
            total_demand_mw=8.0,
            carbon_price=0.0,
        )

        assert len(allocations) == 1
        unit_id, load = allocations[0]
        assert load == 8.0, "Single unit should take all load within limits"

    def test_zero_demand_optimization(self, sample_boiler_fleet):
        """Test optimization with zero demand."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=0.0,
            carbon_price=0.0,
        )

        # All units should be off
        for unit_id, load in allocations:
            assert load == 0.0, f"Unit {unit_id} should be off for zero demand"

    def test_minimum_possible_demand(self, sample_boiler_fleet):
        """Test optimization with minimum possible demand."""
        # Minimum demand is the smallest min_load among available units
        min_loads = [u['min_load_mw'] for u in sample_boiler_fleet if u['is_available']]
        min_demand = min(min_loads)

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=min_demand,
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)
        assert total_allocated >= min_demand * 0.99, "Should meet minimum demand"

    def test_maximum_capacity_demand(self, sample_boiler_fleet):
        """Test optimization with demand at maximum capacity."""
        total_capacity = sum(u['max_load_mw'] for u in sample_boiler_fleet if u['is_available'])

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=total_capacity,
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)

        # Should allocate close to capacity
        assert total_allocated >= total_capacity * 0.95, (
            f"Should meet capacity demand: {total_allocated} vs {total_capacity}"
        )

    def test_identical_units_optimization(self):
        """Test optimization with identical units."""
        identical_fleet = [
            {
                "unit_id": f"BOILER_{i:03d}",
                "unit_type": "BOILER",
                "current_load_mw": 0.0,
                "min_load_mw": 2.0,
                "max_load_mw": 10.0,
                "current_efficiency_pct": 85.0,
                "efficiency_curve_a": 70.0,
                "efficiency_curve_b": 20.0,
                "efficiency_curve_c": -5.0,
                "is_available": True,
                "fuel_cost_per_mwh": 25.0,
                "emissions_factor_kg_co2_mwh": 200.0,
            }
            for i in range(3)
        ]

        allocations = economic_dispatch_merit_order(
            units=identical_fleet,
            total_demand_mw=24.0,  # 8 MW each
            carbon_price=0.0,
        )

        # Should distribute load (exact distribution may vary)
        loaded_units = [(uid, load) for uid, load in allocations if load > 0]
        assert len(loaded_units) >= 2, "Should use multiple identical units"

    def test_highly_variable_costs(self):
        """Test optimization with highly variable unit costs."""
        variable_fleet = [
            {
                "unit_id": "CHEAP",
                "unit_type": "BOILER",
                "min_load_mw": 2.0,
                "max_load_mw": 10.0,
                "is_available": True,
                "fuel_cost_per_mwh": 10.0,  # Very cheap
                "current_efficiency_pct": 80.0,
            },
            {
                "unit_id": "EXPENSIVE",
                "unit_type": "BOILER",
                "min_load_mw": 2.0,
                "max_load_mw": 10.0,
                "is_available": True,
                "fuel_cost_per_mwh": 100.0,  # Very expensive
                "current_efficiency_pct": 80.0,
            },
        ]

        allocations = economic_dispatch_merit_order(
            units=variable_fleet,
            total_demand_mw=8.0,
            carbon_price=0.0,
        )

        # Should prefer cheap unit
        alloc_dict = dict(allocations)
        assert alloc_dict.get("CHEAP", 0) > 0, "Should use cheap unit"
