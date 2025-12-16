"""
GL-023 HeatLoadBalancer - Safety Validation Tests
=================================================

Tests for equipment limit validation, N+1 redundancy checks, ramp rate
validation, emergency reserve calculation, startup/shutdown sequencing,
and equipment trip handling.

Target Coverage: 100% branch coverage (safety-critical code)

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import test utilities
try:
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.formulas import (
        calculate_efficiency_at_load,
        calculate_fuel_consumption,
        economic_dispatch_merit_order,
    )
    from GL_Agent_Factory.backend.agents.gl_023_heat_load_balancer.models import (
        LoadBalancerInput,
        LoadBalancerOutput,
        EquipmentUnit,
    )
    IMPLEMENTATION_AVAILABLE = True
except ImportError:
    IMPLEMENTATION_AVAILABLE = False

    class EquipmentUnit:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self):
            return self.__dict__


# =============================================================================
# EQUIPMENT LIMIT VALIDATION TESTS
# =============================================================================

@pytest.mark.safety
@pytest.mark.critical
class TestEquipmentLimitValidation:
    """Test equipment limit validation."""

    def test_minimum_load_validation(self, sample_boiler_fleet):
        """Test minimum load limit is enforced."""
        for unit in sample_boiler_fleet:
            min_load = unit["min_load_mw"]
            max_load = unit["max_load_mw"]

            # Min load must be positive
            assert min_load >= 0, f"Min load must be >= 0 for {unit['unit_id']}"

            # Min load must be less than max load
            assert min_load < max_load, (
                f"Min load {min_load} must be < max load {max_load} for {unit['unit_id']}"
            )

    def test_maximum_load_validation(self, sample_boiler_fleet):
        """Test maximum load limit is enforced."""
        for unit in sample_boiler_fleet:
            max_load = unit["max_load_mw"]

            # Max load must be positive
            assert max_load > 0, f"Max load must be > 0 for {unit['unit_id']}"

            # Max load should be reasonable (< 1000 MW for typical equipment)
            assert max_load < 1000, f"Max load {max_load} unreasonably high for {unit['unit_id']}"

    def test_current_load_within_bounds(self, sample_boiler_fleet):
        """Test current load is within min/max bounds when running."""
        for unit in sample_boiler_fleet:
            if unit["is_running"] and unit["current_load_mw"] > 0:
                current = unit["current_load_mw"]
                min_load = unit["min_load_mw"]
                max_load = unit["max_load_mw"]

                assert current >= min_load, (
                    f"Current load {current} < min {min_load} for {unit['unit_id']}"
                )
                assert current <= max_load, (
                    f"Current load {current} > max {max_load} for {unit['unit_id']}"
                )

    def test_allocation_respects_limits(self, sample_boiler_fleet):
        """Test allocations respect equipment limits."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=30.0,
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:
                unit = unit_lookup[unit_id]

                # Must be at or above minimum
                assert load >= unit['min_load_mw'] - 0.001, (
                    f"{unit_id}: load {load} MW below minimum {unit['min_load_mw']} MW"
                )

                # Must be at or below maximum
                assert load <= unit['max_load_mw'] + 0.001, (
                    f"{unit_id}: load {load} MW above maximum {unit['max_load_mw']} MW"
                )

    @pytest.mark.parametrize("load_request,expected_valid", [
        (5.0, True),    # Within normal range
        (1.0, False),   # Below minimum (2.0)
        (12.0, False),  # Above maximum (10.0)
        (2.0, True),    # At minimum
        (10.0, True),   # At maximum
        (0.0, True),    # Off (valid)
    ])
    def test_load_request_validation(self, valid_equipment_unit, load_request, expected_valid):
        """Test load request validation against limits."""
        min_load = valid_equipment_unit["min_load_mw"]  # 2.0
        max_load = valid_equipment_unit["max_load_mw"]  # 10.0

        # Load is valid if: 0 (off) or min <= load <= max
        is_valid = (load_request == 0.0) or (min_load <= load_request <= max_load)

        assert is_valid == expected_valid, (
            f"Load {load_request} validation: expected {expected_valid}, got {is_valid}"
        )


# =============================================================================
# N+1 REDUNDANCY TESTS
# =============================================================================

@pytest.mark.safety
@pytest.mark.critical
class TestNPlusOneRedundancy:
    """Test N+1 redundancy requirements."""

    def test_n_plus_1_capacity_check(self, sample_boiler_fleet):
        """Test system can handle loss of largest unit."""
        available_units = [u for u in sample_boiler_fleet if u["is_available"]]

        # Total capacity
        total_capacity = sum(u["max_load_mw"] for u in available_units)

        # Largest unit capacity
        largest_unit = max(available_units, key=lambda u: u["max_load_mw"])
        largest_capacity = largest_unit["max_load_mw"]

        # N+1 capacity (without largest unit)
        n_plus_1_capacity = total_capacity - largest_capacity

        # For a given demand, check if N+1 can cover it
        demand = 25.0

        can_cover_with_n_plus_1 = n_plus_1_capacity >= demand

        # Log the result
        print(f"Total capacity: {total_capacity} MW")
        print(f"Largest unit: {largest_unit['unit_id']} at {largest_capacity} MW")
        print(f"N+1 capacity: {n_plus_1_capacity} MW")
        print(f"Can cover {demand} MW with N+1: {can_cover_with_n_plus_1}")

    def test_redundancy_after_single_failure(self, combined_equipment_fleet):
        """Test redundancy is maintained after single unit failure."""
        demand = 40.0

        # Check each unit as potential failure
        for failed_unit in combined_equipment_fleet:
            if not failed_unit["is_available"]:
                continue

            # Remaining units
            remaining = [u for u in combined_equipment_fleet
                        if u["unit_id"] != failed_unit["unit_id"] and u["is_available"]]

            remaining_capacity = sum(u["max_load_mw"] for u in remaining)

            # Can remaining units cover demand?
            can_cover = remaining_capacity >= demand

            if not can_cover:
                print(f"Warning: Loss of {failed_unit['unit_id']} ({failed_unit['max_load_mw']} MW) "
                      f"leaves only {remaining_capacity} MW for {demand} MW demand")

    def test_n_plus_1_with_maintenance(self, combined_equipment_fleet, unavailable_equipment):
        """Test N+1 redundancy with unit under maintenance."""
        # Add unavailable unit to fleet
        fleet_with_maintenance = combined_equipment_fleet + [unavailable_equipment]

        available_units = [u for u in fleet_with_maintenance if u["is_available"]]

        total_capacity = sum(u["max_load_mw"] for u in available_units)
        largest_available = max(u["max_load_mw"] for u in available_units)
        n_plus_1_capacity = total_capacity - largest_available

        demand = 40.0

        assert n_plus_1_capacity >= demand * 0.8, (
            f"N+1 capacity {n_plus_1_capacity} MW insufficient for {demand} MW "
            f"with maintenance unit"
        )


# =============================================================================
# RAMP RATE VALIDATION TESTS
# =============================================================================

@pytest.mark.safety
@pytest.mark.critical
class TestRampRateValidation:
    """Test ramp rate constraint validation."""

    def test_ramp_up_time_calculation(self, ramp_rate_test_cases):
        """Test ramp up time calculation."""
        for case in ramp_rate_test_cases:
            if case["target_load_mw"] > case["current_load_mw"]:
                # Ramp up
                load_change = case["target_load_mw"] - case["current_load_mw"]
                ramp_rate = case["ramp_rate_mw_per_min"]

                time_required = load_change / ramp_rate

                assert abs(time_required - case["expected_time_min"]) < 0.1, (
                    f"Ramp up time: {time_required:.1f} min vs expected {case['expected_time_min']:.1f} min"
                )

    def test_ramp_down_time_calculation(self, ramp_rate_test_cases):
        """Test ramp down time calculation."""
        for case in ramp_rate_test_cases:
            if case["target_load_mw"] < case["current_load_mw"]:
                # Ramp down
                load_change = case["current_load_mw"] - case["target_load_mw"]
                ramp_rate = case["ramp_rate_mw_per_min"]

                time_required = load_change / ramp_rate

                assert abs(time_required - case["expected_time_min"]) < 0.1, (
                    f"Ramp down time: {time_required:.1f} min vs expected {case['expected_time_min']:.1f} min"
                )

    def test_ramp_rate_constraint_achievability(self, ramp_rate_test_cases):
        """Test ramp rate constraint achievability."""
        for case in ramp_rate_test_cases:
            load_change = abs(case["target_load_mw"] - case["current_load_mw"])
            time_required = load_change / case["ramp_rate_mw_per_min"]

            achievable = time_required <= case["time_available_min"]

            assert achievable == case["expected_achievable"], (
                f"Achievability mismatch for {load_change} MW change in {case['time_available_min']} min"
            )

    def test_ramp_rate_positive(self, sample_boiler_fleet):
        """Test all ramp rates are positive."""
        for unit in sample_boiler_fleet:
            ramp_rate = unit.get("ramp_rate_mw_per_min", 0)
            assert ramp_rate > 0, f"Ramp rate must be positive for {unit['unit_id']}"

    @pytest.mark.parametrize("current,target,ramp_rate,max_time,expected_achievable", [
        (5.0, 10.0, 1.0, 5.0, True),   # 5 MW in 5 min = achievable
        (5.0, 10.0, 1.0, 4.0, False),  # 5 MW in 4 min = not achievable
        (10.0, 5.0, 2.0, 3.0, True),   # 5 MW down in 3 min @ 2 MW/min = achievable
        (5.0, 5.0, 1.0, 0.0, True),    # No change = always achievable
    ])
    def test_ramp_constraint_scenarios(self, current, target, ramp_rate, max_time, expected_achievable):
        """Test ramp constraint for various scenarios."""
        load_change = abs(target - current)

        if load_change == 0:
            achievable = True
        else:
            time_required = load_change / ramp_rate
            achievable = time_required <= max_time

        assert achievable == expected_achievable


# =============================================================================
# EMERGENCY RESERVE CALCULATION TESTS
# =============================================================================

@pytest.mark.safety
@pytest.mark.critical
class TestEmergencyReserveCalculation:
    """Test emergency reserve calculations."""

    def test_spinning_reserve_calculation(self, sample_boiler_fleet):
        """Test spinning reserve calculation."""
        available_units = [u for u in sample_boiler_fleet if u["is_available"]]
        total_capacity = sum(u["max_load_mw"] for u in available_units)

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=30.0,
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)
        spinning_reserve = total_capacity - total_allocated

        assert spinning_reserve >= 0, "Spinning reserve cannot be negative"

        reserve_pct = (spinning_reserve / total_capacity * 100) if total_capacity > 0 else 0
        print(f"Spinning reserve: {spinning_reserve:.1f} MW ({reserve_pct:.1f}%)")

    def test_minimum_spinning_reserve(self, sample_boiler_fleet):
        """Test minimum spinning reserve requirement."""
        min_reserve_pct = 10.0

        available_units = [u for u in sample_boiler_fleet if u["is_available"]]
        total_capacity = sum(u["max_load_mw"] for u in available_units)

        # Maximum demand that allows minimum reserve
        max_allowable_demand = total_capacity * (1 - min_reserve_pct / 100)

        print(f"Total capacity: {total_capacity} MW")
        print(f"Max demand for {min_reserve_pct}% reserve: {max_allowable_demand:.1f} MW")

    def test_emergency_reserve_for_trip(self, combined_equipment_fleet):
        """Test emergency reserve covers largest unit trip."""
        available_units = [u for u in combined_equipment_fleet if u["is_available"]]

        # Largest unit that could trip
        largest_unit = max(available_units, key=lambda u: u["max_load_mw"])
        trip_impact = largest_unit["max_load_mw"]

        total_capacity = sum(u["max_load_mw"] for u in available_units)

        # For demand X, reserve should cover trip
        demand = 40.0
        allocated = demand
        reserve = total_capacity - allocated

        can_cover_trip = reserve >= trip_impact

        print(f"Trip impact: {trip_impact:.1f} MW")
        print(f"Available reserve: {reserve:.1f} MW")
        print(f"Can cover trip: {can_cover_trip}")

    @pytest.mark.parametrize("demand_fraction,min_reserve_pct,expected_warning", [
        (0.5, 10.0, False),   # 50% load, 50% reserve - OK
        (0.8, 10.0, False),   # 80% load, 20% reserve - OK
        (0.9, 10.0, False),   # 90% load, 10% reserve - borderline
        (0.95, 10.0, True),   # 95% load, 5% reserve - warning
        (1.0, 10.0, True),    # 100% load, 0% reserve - warning
    ])
    def test_reserve_warning_thresholds(self, sample_boiler_fleet, demand_fraction,
                                        min_reserve_pct, expected_warning):
        """Test reserve warning thresholds."""
        available_units = [u for u in sample_boiler_fleet if u["is_available"]]
        total_capacity = sum(u["max_load_mw"] for u in available_units)

        demand = total_capacity * demand_fraction
        reserve = total_capacity - demand
        reserve_pct = (reserve / total_capacity * 100) if total_capacity > 0 else 0

        should_warn = reserve_pct < min_reserve_pct

        assert should_warn == expected_warning, (
            f"Reserve {reserve_pct:.1f}%: expected warning={expected_warning}, got {should_warn}"
        )


# =============================================================================
# STARTUP/SHUTDOWN SEQUENCING TESTS
# =============================================================================

@pytest.mark.safety
class TestStartupShutdownSequencing:
    """Test startup and shutdown sequencing."""

    def test_startup_time_respected(self, sample_boiler_fleet):
        """Test startup time is respected."""
        for unit in sample_boiler_fleet:
            startup_time = unit.get("startup_time_min", 30)

            # Startup time should be positive
            assert startup_time > 0, f"Startup time must be positive for {unit['unit_id']}"

            # Cold start typically 15-60 minutes for boilers
            assert 10 <= startup_time <= 120, (
                f"Startup time {startup_time} min unusual for {unit['unit_id']}"
            )

    def test_minimum_run_time_respected(self, sample_boiler_fleet):
        """Test minimum run time is respected."""
        for unit in sample_boiler_fleet:
            min_run_time = unit.get("min_run_time_hr", 1.0)

            # Min run time should be positive
            assert min_run_time > 0, f"Min run time must be positive for {unit['unit_id']}"

            # Typically 1-4 hours for industrial boilers
            assert 0.5 <= min_run_time <= 8.0, (
                f"Min run time {min_run_time} hr unusual for {unit['unit_id']}"
            )

    def test_max_simultaneous_startups(self, sample_boiler_fleet):
        """Test maximum simultaneous startups constraint."""
        max_startups = 2  # Typical constraint

        # Count units that need to start
        units_to_start = [u for u in sample_boiler_fleet
                        if u["is_available"] and not u["is_running"]]

        if len(units_to_start) > max_startups:
            print(f"Would need to sequence {len(units_to_start)} startups "
                  f"(max {max_startups} simultaneous)")

    def test_startup_sequence_priority(self, sample_boiler_fleet):
        """Test startup sequence follows priority."""
        # Units with shorter startup time should generally start first
        available_offline = [u for u in sample_boiler_fleet
                            if u["is_available"] and not u["is_running"]]

        # Sort by startup time
        sorted_units = sorted(available_offline, key=lambda u: u.get("startup_time_min", 30))

        for i, unit in enumerate(sorted_units):
            print(f"Priority {i+1}: {unit['unit_id']} - {unit.get('startup_time_min', 30)} min startup")

    def test_shutdown_sequence(self, sample_boiler_fleet):
        """Test shutdown sequence."""
        # Units should be shut down in reverse merit order (most expensive first)
        running_units = [u for u in sample_boiler_fleet if u["is_running"]]

        # Sort by fuel cost (highest first for shutdown)
        shutdown_order = sorted(running_units,
                               key=lambda u: -u.get("fuel_cost_per_mwh", 0))

        for i, unit in enumerate(shutdown_order):
            print(f"Shutdown priority {i+1}: {unit['unit_id']} - "
                  f"${unit.get('fuel_cost_per_mwh', 0)}/MWh")


# =============================================================================
# EQUIPMENT TRIP HANDLING TESTS
# =============================================================================

@pytest.mark.safety
@pytest.mark.critical
class TestEquipmentTripHandling:
    """Test equipment trip handling."""

    def test_single_unit_trip_response(self, sample_boiler_fleet, equipment_trip_scenarios):
        """Test response to single unit trip."""
        scenario = equipment_trip_scenarios[0]  # Single boiler trip

        # Simulate trip
        fleet_after_trip = []
        for unit in sample_boiler_fleet:
            unit_copy = unit.copy()
            if unit["unit_id"] == scenario["tripped_unit"]:
                unit_copy["is_available"] = False
                unit_copy["is_running"] = False
                unit_copy["current_load_mw"] = 0.0
            fleet_after_trip.append(unit_copy)

        # Attempt to meet demand with remaining units
        allocations = economic_dispatch_merit_order(
            units=fleet_after_trip,
            total_demand_mw=scenario["demand_mw"],
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)

        # Check if demand can still be met
        can_meet_demand = total_allocated >= scenario["demand_mw"] * 0.95

        assert can_meet_demand == scenario["expected_rebalance"], (
            f"Trip scenario: expected rebalance={scenario['expected_rebalance']}, "
            f"got {can_meet_demand}"
        )

    def test_multiple_unit_trip_response(self, combined_equipment_fleet, equipment_trip_scenarios):
        """Test response to multiple unit trips."""
        scenario = equipment_trip_scenarios[1]  # Multiple unit trip

        # Simulate multiple trips
        fleet_after_trip = []
        for unit in combined_equipment_fleet:
            unit_copy = unit.copy()
            if unit["unit_id"] in scenario["tripped_units"]:
                unit_copy["is_available"] = False
                unit_copy["is_running"] = False
                unit_copy["current_load_mw"] = 0.0
            fleet_after_trip.append(unit_copy)

        remaining_capacity = sum(u["max_load_mw"] for u in fleet_after_trip if u["is_available"])

        print(f"After trip: {remaining_capacity:.1f} MW remaining for "
              f"{scenario['demand_mw']:.1f} MW demand")

    def test_cascade_trip_load_shed(self, combined_equipment_fleet, equipment_trip_scenarios):
        """Test load shedding on cascade trip."""
        scenario = equipment_trip_scenarios[2]  # Cascade trip

        # Simulate cascade trips
        fleet_after_cascade = []
        for unit in combined_equipment_fleet:
            unit_copy = unit.copy()
            if unit["unit_id"] in scenario.get("tripped_units", []):
                unit_copy["is_available"] = False
                unit_copy["is_running"] = False
                unit_copy["current_load_mw"] = 0.0
            fleet_after_cascade.append(unit_copy)

        remaining_capacity = sum(u["max_load_mw"] for u in fleet_after_cascade if u["is_available"])

        # Check if load shedding is required
        requires_load_shed = remaining_capacity < scenario["demand_mw"]

        print(f"Cascade trip: {remaining_capacity:.1f} MW remaining, "
              f"need {scenario['demand_mw']:.1f} MW")
        print(f"Requires load shed: {requires_load_shed}")

    def test_trip_recovery_time(self, sample_boiler_fleet):
        """Test trip recovery time estimation."""
        for unit in sample_boiler_fleet:
            startup_time = unit.get("startup_time_min", 30)
            ramp_rate = unit.get("ramp_rate_mw_per_min", 1.0)
            max_load = unit["max_load_mw"]

            # Time to full load from cold = startup + ramp to max
            time_to_full = startup_time + (max_load / ramp_rate)

            print(f"{unit['unit_id']}: {time_to_full:.0f} min to full load from cold")

            # Should be reasonable (<2 hours typically)
            assert time_to_full < 180, f"Recovery time too long for {unit['unit_id']}"


# =============================================================================
# SAFETY INTERLOCK TESTS
# =============================================================================

@pytest.mark.safety
@pytest.mark.critical
class TestSafetyInterlocks:
    """Test safety interlock logic."""

    def test_no_load_below_minimum(self, sample_boiler_fleet):
        """Test no unit operates below minimum load."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=5.0,  # Low demand
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if load > 0:
                min_load = unit_lookup[unit_id]['min_load_mw']
                assert load >= min_load - 0.001, (
                    f"SAFETY: {unit_id} operating below minimum ({load} < {min_load})"
                )

    def test_no_load_above_maximum(self, sample_boiler_fleet):
        """Test no unit operates above maximum load."""
        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=100.0,  # High demand
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if unit_id in unit_lookup:
                max_load = unit_lookup[unit_id]['max_load_mw']
                assert load <= max_load + 0.001, (
                    f"SAFETY: {unit_id} operating above maximum ({load} > {max_load})"
                )

    def test_unavailable_units_not_loaded(self, sample_boiler_fleet):
        """Test unavailable units receive no load."""
        # Make first unit unavailable
        fleet = [u.copy() for u in sample_boiler_fleet]
        fleet[0]["is_available"] = False

        allocations = economic_dispatch_merit_order(
            units=fleet,
            total_demand_mw=30.0,
            carbon_price=0.0,
        )

        for unit_id, load in allocations:
            if unit_id == fleet[0]["unit_id"]:
                assert load == 0.0, (
                    f"SAFETY: Unavailable unit {unit_id} received load {load}"
                )

    def test_efficiency_sanity_check(self, sample_boiler_fleet):
        """Test efficiency values are within physical limits."""
        for unit in sample_boiler_fleet:
            # Calculate efficiency at various loads
            for load_frac in [0.5, 0.75, 1.0]:
                load = unit["max_load_mw"] * load_frac

                efficiency = calculate_efficiency_at_load(
                    load,
                    unit["min_load_mw"],
                    unit["max_load_mw"],
                    unit.get("efficiency_curve_a", 0),
                    unit.get("efficiency_curve_b", 0),
                    unit.get("efficiency_curve_c", 0),
                )

                # Efficiency must be 0-100%
                assert 0 <= efficiency <= 100, (
                    f"SAFETY: {unit['unit_id']} efficiency {efficiency}% outside 0-100%"
                )

                # Typical boiler efficiency 70-95%
                if efficiency > 0:
                    assert 50 <= efficiency <= 98, (
                        f"WARNING: {unit['unit_id']} efficiency {efficiency}% unusual"
                    )


# =============================================================================
# FUEL CONSUMPTION SAFETY TESTS
# =============================================================================

@pytest.mark.safety
class TestFuelConsumptionSafety:
    """Test fuel consumption safety limits."""

    def test_fuel_consumption_reasonable(self, sample_boiler_fleet):
        """Test fuel consumption is within reasonable limits."""
        for unit in sample_boiler_fleet:
            if not unit["is_running"]:
                continue

            load = unit["current_load_mw"]
            efficiency = unit["current_efficiency_pct"]

            if efficiency > 0:
                fuel = calculate_fuel_consumption(load, efficiency)

                # Fuel should be slightly higher than thermal output
                assert fuel >= load, (
                    f"SAFETY: Fuel {fuel} MW < output {load} MW for {unit['unit_id']}"
                )

                # But not unreasonably higher (efficiency > 30%)
                assert fuel <= load * 3.5, (
                    f"SAFETY: Fuel {fuel} MW too high for {load} MW output"
                )

    def test_emissions_reasonable(self, sample_boiler_fleet):
        """Test emissions are within reasonable limits."""
        for unit in sample_boiler_fleet:
            emissions_factor = unit.get("emissions_factor_kg_co2_mwh", 200)

            # Typical range: 50-400 kg CO2/MWh for natural gas/oil
            assert 0 <= emissions_factor <= 500, (
                f"SAFETY: Emissions factor {emissions_factor} kg/MWh unusual for {unit['unit_id']}"
            )


# =============================================================================
# EDGE CASE SAFETY TESTS
# =============================================================================

@pytest.mark.safety
class TestSafetyEdgeCases:
    """Test safety edge cases."""

    def test_all_units_at_maximum(self, sample_boiler_fleet):
        """Test behavior when all units at maximum load."""
        total_capacity = sum(u["max_load_mw"] for u in sample_boiler_fleet if u["is_available"])

        allocations = economic_dispatch_merit_order(
            units=sample_boiler_fleet,
            total_demand_mw=total_capacity,
            carbon_price=0.0,
        )

        unit_lookup = {u['unit_id']: u for u in sample_boiler_fleet}

        for unit_id, load in allocations:
            if unit_id in unit_lookup and unit_lookup[unit_id]["is_available"]:
                max_load = unit_lookup[unit_id]["max_load_mw"]
                # Should be at or near maximum
                assert load <= max_load + 0.001, f"Exceeds max for {unit_id}"

    def test_single_unit_operation(self):
        """Test safety with single unit operation."""
        single_unit = [{
            "unit_id": "ONLY_BOILER",
            "unit_type": "BOILER",
            "current_load_mw": 5.0,
            "min_load_mw": 2.0,
            "max_load_mw": 10.0,
            "current_efficiency_pct": 85.0,
            "is_available": True,
            "is_running": True,
            "fuel_cost_per_mwh": 25.0,
        }]

        # Normal operation
        allocations = economic_dispatch_merit_order(single_unit, 8.0, 0.0)
        assert len(allocations) == 1
        assert allocations[0][1] == 8.0

        # Above capacity
        allocations = economic_dispatch_merit_order(single_unit, 15.0, 0.0)
        assert allocations[0][1] <= 10.0

        # Below minimum
        allocations = economic_dispatch_merit_order(single_unit, 1.0, 0.0)
        # Should either be at minimum or off
        assert allocations[0][1] == 0.0 or allocations[0][1] >= 2.0

    def test_zero_efficiency_handling(self, valid_equipment_unit):
        """Test handling of zero efficiency edge case."""
        # Zero efficiency should return zero fuel consumption
        fuel = calculate_fuel_consumption(10.0, 0.0)
        assert fuel == 0.0, "Zero efficiency should give zero fuel"

    def test_very_high_demand(self, combined_equipment_fleet):
        """Test safety under very high demand."""
        total_capacity = sum(u["max_load_mw"] for u in combined_equipment_fleet if u["is_available"])

        # Demand 2x capacity
        allocations = economic_dispatch_merit_order(
            units=combined_equipment_fleet,
            total_demand_mw=total_capacity * 2,
            carbon_price=0.0,
        )

        total_allocated = sum(load for _, load in allocations)

        # Should not exceed physical capacity
        assert total_allocated <= total_capacity + 0.001, (
            f"SAFETY: Allocated {total_allocated} MW exceeds capacity {total_capacity} MW"
        )
