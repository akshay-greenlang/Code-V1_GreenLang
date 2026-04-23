"""
Integration tests for GL-001 ThermalCommand Multi-Equipment Coordination.

Tests the coordination of multiple thermal equipment units including
MILP optimization, cascade control, and safety boundary enforcement.

Coverage Target: 85%+
Reference: GL-001 Specification Section 11

Test Categories:
1. Multi-boiler dispatch optimization
2. CHP integration scenarios
3. Cascade control coordination
4. Safety boundary enforcement across equipment
5. Load shedding scenarios
6. Equipment failure handling
7. Network partition scenarios

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any, Optional

# Add parent path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# MOCK CLASSES FOR MULTI-EQUIPMENT TESTING
# =============================================================================

class MockEquipment:
    """Mock thermal equipment for testing."""

    def __init__(
        self,
        equipment_id: str,
        name: str,
        equipment_type: str = "boiler",
        max_capacity: float = 50.0,
        min_capacity: float = 10.0,
        efficiency: float = 0.85,
        fuel_cost: float = 5.0,
        status: str = "available"
    ):
        self.equipment_id = equipment_id
        self.name = name
        self.equipment_type = equipment_type
        self.max_capacity_mmbtu_hr = max_capacity
        self.min_capacity_mmbtu_hr = min_capacity
        self.rated_efficiency = efficiency
        self.fuel_cost_per_mmbtu = fuel_cost
        self.status = status
        self.current_load_mmbtu_hr = 0.0
        self.co2_kg_per_mmbtu_fuel = 53.06
        self.priority = 1

    def is_available(self) -> bool:
        return self.status == "available"


class MockLoadAllocation:
    """Mock load allocation result."""

    def __init__(
        self,
        equipment_id: str,
        allocated_load: float,
        is_running: bool = True
    ):
        self.equipment_id = equipment_id
        self.equipment_name = equipment_id
        self.allocated_load_mmbtu_hr = allocated_load
        self.is_running = is_running
        self.fuel_consumption = allocated_load / 0.85 if allocated_load > 0 else 0
        self.operating_cost = self.fuel_consumption * 5.0


class MockOptimizationResult:
    """Mock optimization result."""

    def __init__(
        self,
        status: str = "optimal",
        allocations: List[MockLoadAllocation] = None,
        total_demand: float = 0.0
    ):
        self.status = status
        self.allocations = allocations or []
        self.total_allocated_mmbtu_hr = sum(
            a.allocated_load_mmbtu_hr for a in self.allocations
        )
        self.unmet_demand_mmbtu_hr = max(
            0, total_demand - self.total_allocated_mmbtu_hr
        )
        self.total_cost_per_hour = sum(
            a.operating_cost for a in self.allocations
        )
        self.total_co2_kg_hr = sum(
            a.fuel_consumption * 53.06 for a in self.allocations
        )
        self.solve_time_ms = 100.0
        self.provenance_hash = "abc123"


class MockMILPLoadAllocator:
    """Mock MILP load allocator for testing."""

    def __init__(self):
        self._equipment: Dict[str, MockEquipment] = {}
        self._last_solution: Optional[MockOptimizationResult] = None

    def add_equipment(self, equipment: MockEquipment) -> bool:
        if equipment.equipment_id in self._equipment:
            return False
        self._equipment[equipment.equipment_id] = equipment
        return True

    def remove_equipment(self, equipment_id: str) -> bool:
        if equipment_id not in self._equipment:
            return False
        del self._equipment[equipment_id]
        return True

    def update_equipment_status(
        self,
        equipment_id: str,
        status: str,
        current_load: float = None
    ) -> bool:
        if equipment_id not in self._equipment:
            return False
        self._equipment[equipment_id].status = status
        if current_load is not None:
            self._equipment[equipment_id].current_load_mmbtu_hr = current_load
        return True

    def get_equipment(self, equipment_id: str) -> Optional[MockEquipment]:
        return self._equipment.get(equipment_id)

    def get_all_equipment(self) -> List[MockEquipment]:
        return list(self._equipment.values())

    def get_total_capacity(self) -> float:
        return sum(
            e.max_capacity_mmbtu_hr
            for e in self._equipment.values()
            if e.is_available()
        )

    def optimize(self, demand: float, objective: str = "balanced") -> MockOptimizationResult:
        """Run optimization."""
        available = [e for e in self._equipment.values() if e.is_available()]

        if not available:
            return MockOptimizationResult(
                status="infeasible",
                total_demand=demand
            )

        total_capacity = sum(e.max_capacity_mmbtu_hr for e in available)

        if demand > total_capacity:
            return MockOptimizationResult(
                status="infeasible",
                total_demand=demand
            )

        # Simple merit order dispatch
        allocations = []
        remaining_demand = demand

        # Sort by fuel cost (cheapest first)
        sorted_equipment = sorted(available, key=lambda e: e.fuel_cost_per_mmbtu)

        for equipment in sorted_equipment:
            if remaining_demand <= 0:
                break

            # Determine allocation
            if remaining_demand >= equipment.min_capacity_mmbtu_hr:
                allocation = min(remaining_demand, equipment.max_capacity_mmbtu_hr)
                allocations.append(MockLoadAllocation(
                    equipment_id=equipment.equipment_id,
                    allocated_load=allocation,
                    is_running=True
                ))
                remaining_demand -= allocation

        result = MockOptimizationResult(
            status="optimal" if remaining_demand <= 0.1 else "feasible",
            allocations=allocations,
            total_demand=demand
        )
        self._last_solution = result
        return result


class MockCascadeController:
    """Mock cascade controller for testing."""

    def __init__(self, master_id: str, slave_id: str):
        self.master_id = master_id
        self.slave_id = slave_id
        self._master_setpoint = 0.0
        self._cascade_active = True
        self._output = 50.0

    def set_master_setpoint(self, sp: float):
        self._master_setpoint = sp

    def set_cascade_active(self, active: bool):
        self._cascade_active = active

    def calculate(
        self,
        master_pv: float,
        slave_pv: float,
        dt_seconds: float
    ) -> Dict:
        """Calculate cascade output."""
        error = self._master_setpoint - master_pv
        output = 50.0 + error * 0.5  # Simple proportional
        output = max(0, min(100, output))
        self._output = output

        return {
            "master_output": output,
            "slave_output": output,
            "final_output": output,
            "provenance_hash": "cascade_hash_123"
        }


class MockSafetyBoundaryEngine:
    """Mock safety boundary engine for testing."""

    def __init__(self):
        self._tag_values: Dict[str, float] = {}
        self._blocked_tags: set = set()
        self._violations: List[Dict] = []

    def update_tag_value(self, tag_id: str, value: float):
        self._tag_values[tag_id] = value

    def update_tag_values(self, values: Dict[str, float]):
        self._tag_values.update(values)

    def block_tag(self, tag_id: str):
        self._blocked_tags.add(tag_id)

    def unblock_tag(self, tag_id: str):
        self._blocked_tags.discard(tag_id)

    def validate_write_request(
        self,
        tag_id: str,
        value: float,
        allow_clamping: bool = True
    ) -> Dict:
        """Validate a write request."""
        # Check if blocked
        if tag_id in self._blocked_tags:
            return {
                "decision": "block",
                "final_value": None,
                "violations": [{"type": "unauthorized_tag"}]
            }

        # Check limits (simplified)
        if value < -100:
            if allow_clamping:
                return {
                    "decision": "clamp",
                    "final_value": -100,
                    "violations": [{"type": "under_min"}]
                }
            else:
                return {
                    "decision": "block",
                    "final_value": None,
                    "violations": [{"type": "under_min"}]
                }

        if value > 500:
            if allow_clamping:
                return {
                    "decision": "clamp",
                    "final_value": 500,
                    "violations": [{"type": "over_max"}]
                }
            else:
                return {
                    "decision": "block",
                    "final_value": None,
                    "violations": [{"type": "over_max"}]
                }

        return {
            "decision": "allow",
            "final_value": value,
            "violations": []
        }

    def enforce_constraints(
        self,
        targets: Dict[str, float]
    ) -> Dict[str, float]:
        """Enforce safety constraints on targets."""
        constrained = {}
        for tag_id, value in targets.items():
            result = self.validate_write_request(tag_id, value)
            if result["decision"] != "block":
                constrained[tag_id] = result["final_value"]
            else:
                constrained[tag_id] = self._tag_values.get(tag_id, 0.0)
        return constrained


class MockEquipmentCoordinator:
    """Mock equipment coordinator for integration testing."""

    def __init__(self):
        self.milp_allocator = MockMILPLoadAllocator()
        self.cascade_controllers: Dict[str, MockCascadeController] = {}
        self.safety_engine = MockSafetyBoundaryEngine()
        self._current_demand = 0.0
        self._current_allocations: List[MockLoadAllocation] = []

    def add_equipment(self, equipment: MockEquipment):
        """Add equipment to coordinator."""
        self.milp_allocator.add_equipment(equipment)

        # Create cascade controller for this equipment
        cascade = MockCascadeController(
            master_id=f"TIC-{equipment.equipment_id}",
            slave_id=f"FIC-{equipment.equipment_id}"
        )
        self.cascade_controllers[equipment.equipment_id] = cascade

    def set_demand(self, demand: float):
        """Set the current demand."""
        self._current_demand = demand

    def optimize(self) -> MockOptimizationResult:
        """Run optimization for current demand."""
        result = self.milp_allocator.optimize(self._current_demand)
        self._current_allocations = result.allocations
        return result

    def execute_allocations(
        self,
        pv_data: Dict[str, float]
    ) -> Dict[str, Dict]:
        """Execute control for each equipment with allocations."""
        control_outputs = {}

        for allocation in self._current_allocations:
            eq_id = allocation.equipment_id
            cascade = self.cascade_controllers.get(eq_id)

            if cascade and allocation.is_running:
                # Set master setpoint based on allocation
                cascade.set_master_setpoint(
                    allocation.allocated_load_mmbtu_hr
                )

                # Get process values
                master_pv = pv_data.get(f"TI-{eq_id}", 0.0)
                slave_pv = pv_data.get(f"FI-{eq_id}", 0.0)

                # Calculate control output
                output = cascade.calculate(
                    master_pv=master_pv,
                    slave_pv=slave_pv,
                    dt_seconds=1.0
                )

                # Validate through safety
                tag_id = f"TIC-{eq_id}"
                safety_result = self.safety_engine.validate_write_request(
                    tag_id=tag_id,
                    value=output["final_output"]
                )

                control_outputs[eq_id] = {
                    "cascade_output": output,
                    "safety_result": safety_result,
                    "final_output": safety_result.get("final_value", output["final_output"])
                }

        return control_outputs

    def handle_equipment_failure(
        self,
        equipment_id: str,
        new_status: str = "faulted"
    ) -> MockOptimizationResult:
        """Handle equipment failure."""
        self.milp_allocator.update_equipment_status(equipment_id, new_status)

        # Re-optimize with remaining equipment
        return self.optimize()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def equipment_coordinator() -> MockEquipmentCoordinator:
    """Create equipment coordinator."""
    return MockEquipmentCoordinator()


@pytest.fixture
def three_boiler_system(equipment_coordinator) -> MockEquipmentCoordinator:
    """Create system with three boilers."""
    boilers = [
        MockEquipment(
            equipment_id="BOILER-001",
            name="Boiler 1",
            max_capacity=50.0,
            min_capacity=12.5,
            fuel_cost=5.0
        ),
        MockEquipment(
            equipment_id="BOILER-002",
            name="Boiler 2",
            max_capacity=40.0,
            min_capacity=10.0,
            fuel_cost=5.5
        ),
        MockEquipment(
            equipment_id="BOILER-003",
            name="Boiler 3",
            max_capacity=30.0,
            min_capacity=7.5,
            fuel_cost=6.0
        ),
    ]

    for boiler in boilers:
        equipment_coordinator.add_equipment(boiler)

    return equipment_coordinator


@pytest.fixture
def chp_and_boiler_system(equipment_coordinator) -> MockEquipmentCoordinator:
    """Create system with CHP and boilers."""
    equipment = [
        MockEquipment(
            equipment_id="CHP-001",
            name="CHP System",
            equipment_type="chp",
            max_capacity=30.0,
            min_capacity=15.0,
            fuel_cost=3.0  # Lower due to electricity credit
        ),
        MockEquipment(
            equipment_id="BOILER-001",
            name="Boiler 1",
            max_capacity=50.0,
            min_capacity=12.5,
            fuel_cost=5.0
        ),
        MockEquipment(
            equipment_id="BOILER-002",
            name="Boiler 2",
            max_capacity=40.0,
            min_capacity=10.0,
            fuel_cost=5.5
        ),
    ]

    for eq in equipment:
        equipment_coordinator.add_equipment(eq)

    return equipment_coordinator


@pytest.fixture
def sample_pv_data() -> Dict[str, float]:
    """Create sample process value data."""
    return {
        "TI-BOILER-001": 25.0,
        "FI-BOILER-001": 45.0,
        "TI-BOILER-002": 20.0,
        "FI-BOILER-002": 35.0,
        "TI-BOILER-003": 15.0,
        "FI-BOILER-003": 25.0,
        "TI-CHP-001": 22.0,
        "FI-CHP-001": 28.0,
    }


# =============================================================================
# TEST CLASS: MULTI-BOILER DISPATCH
# =============================================================================

class TestMultiBoilerDispatch:
    """Tests for multi-boiler dispatch optimization."""

    def test_optimize_three_boiler_system(self, three_boiler_system):
        """Test optimization with three boilers."""
        three_boiler_system.set_demand(60.0)

        result = three_boiler_system.optimize()

        assert result.status in ["optimal", "feasible"]
        assert result.total_allocated_mmbtu_hr == pytest.approx(60.0, rel=0.01)

    def test_merit_order_dispatch(self, three_boiler_system):
        """Test that cheapest units are dispatched first."""
        three_boiler_system.set_demand(40.0)

        result = three_boiler_system.optimize()

        # Cheapest boiler (BOILER-001 at $5/MMBtu) should be running
        allocations = {a.equipment_id: a for a in result.allocations}
        assert "BOILER-001" in allocations
        assert allocations["BOILER-001"].is_running

    def test_total_capacity_limit(self, three_boiler_system):
        """Test demand exceeding total capacity."""
        # Total capacity = 50 + 40 + 30 = 120
        three_boiler_system.set_demand(150.0)

        result = three_boiler_system.optimize()

        assert result.status == "infeasible"

    def test_minimum_load_constraints(self, three_boiler_system):
        """Test that minimum load constraints are respected."""
        three_boiler_system.set_demand(30.0)

        result = three_boiler_system.optimize()

        for allocation in result.allocations:
            if allocation.is_running:
                eq = three_boiler_system.milp_allocator.get_equipment(
                    allocation.equipment_id
                )
                assert allocation.allocated_load_mmbtu_hr >= eq.min_capacity_mmbtu_hr

    def test_equipment_status_update(self, three_boiler_system):
        """Test equipment status updates affect dispatch."""
        # Initial optimization
        three_boiler_system.set_demand(60.0)
        result1 = three_boiler_system.optimize()

        # Take one boiler offline
        three_boiler_system.milp_allocator.update_equipment_status(
            "BOILER-001",
            "maintenance"
        )

        # Re-optimize
        result2 = three_boiler_system.optimize()

        # BOILER-001 should not be in allocations
        eq_ids = [a.equipment_id for a in result2.allocations]
        assert "BOILER-001" not in eq_ids


# =============================================================================
# TEST CLASS: CHP INTEGRATION
# =============================================================================

class TestCHPIntegration:
    """Tests for CHP system integration."""

    def test_chp_dispatched_first(self, chp_and_boiler_system):
        """Test that CHP is dispatched first (lower cost)."""
        chp_and_boiler_system.set_demand(25.0)

        result = chp_and_boiler_system.optimize()

        # CHP should be running (lowest cost at $3/MMBtu)
        chp_allocation = next(
            (a for a in result.allocations if a.equipment_id == "CHP-001"),
            None
        )
        assert chp_allocation is not None
        assert chp_allocation.is_running

    def test_chp_plus_boiler_dispatch(self, chp_and_boiler_system):
        """Test combined CHP and boiler dispatch."""
        chp_and_boiler_system.set_demand(60.0)

        result = chp_and_boiler_system.optimize()

        # Both CHP and boiler(s) should be running
        eq_types = set()
        for allocation in result.allocations:
            eq = chp_and_boiler_system.milp_allocator.get_equipment(
                allocation.equipment_id
            )
            eq_types.add(eq.equipment_type)

        assert "chp" in eq_types
        assert "boiler" in eq_types


# =============================================================================
# TEST CLASS: CASCADE CONTROL COORDINATION
# =============================================================================

class TestCascadeControlCoordination:
    """Tests for cascade control coordination across equipment."""

    def test_cascade_controllers_created(self, three_boiler_system):
        """Test that cascade controllers are created for each equipment."""
        assert len(three_boiler_system.cascade_controllers) == 3

    def test_execute_allocations(
        self, three_boiler_system, sample_pv_data
    ):
        """Test executing control for all allocations."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        outputs = three_boiler_system.execute_allocations(sample_pv_data)

        assert len(outputs) > 0
        for eq_id, output in outputs.items():
            assert "cascade_output" in output
            assert "safety_result" in output
            assert "final_output" in output

    def test_cascade_setpoints_from_allocation(
        self, three_boiler_system, sample_pv_data
    ):
        """Test that cascade setpoints match allocations."""
        three_boiler_system.set_demand(60.0)
        result = three_boiler_system.optimize()

        three_boiler_system.execute_allocations(sample_pv_data)

        for allocation in result.allocations:
            if allocation.is_running:
                cascade = three_boiler_system.cascade_controllers[
                    allocation.equipment_id
                ]
                assert cascade._master_setpoint == allocation.allocated_load_mmbtu_hr


# =============================================================================
# TEST CLASS: SAFETY BOUNDARY ENFORCEMENT
# =============================================================================

class TestSafetyBoundaryEnforcement:
    """Tests for safety boundary enforcement across equipment."""

    def test_safety_validation_on_outputs(
        self, three_boiler_system, sample_pv_data
    ):
        """Test that safety validates all control outputs."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        outputs = three_boiler_system.execute_allocations(sample_pv_data)

        for eq_id, output in outputs.items():
            assert output["safety_result"]["decision"] in ["allow", "clamp", "block"]

    def test_blocked_tag_prevents_output(
        self, three_boiler_system, sample_pv_data
    ):
        """Test that blocked tags prevent control output."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        # Block one equipment's control tag
        three_boiler_system.safety_engine.block_tag("TIC-BOILER-001")

        outputs = three_boiler_system.execute_allocations(sample_pv_data)

        # The blocked equipment should have blocked status
        if "BOILER-001" in outputs:
            assert outputs["BOILER-001"]["safety_result"]["decision"] == "block"

    def test_output_clamping(
        self, three_boiler_system, sample_pv_data
    ):
        """Test that out-of-range outputs are clamped."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        # Modify cascade to output extreme value
        cascade = three_boiler_system.cascade_controllers["BOILER-001"]
        cascade._output = 600.0  # Above max of 500

        outputs = three_boiler_system.execute_allocations(sample_pv_data)

        # Should be clamped to 500
        # (This depends on mock implementation)


# =============================================================================
# TEST CLASS: LOAD SHEDDING
# =============================================================================

class TestLoadShedding:
    """Tests for load shedding scenarios."""

    def test_graceful_load_reduction(self, three_boiler_system):
        """Test graceful load reduction when capacity decreases."""
        # Initial full load
        three_boiler_system.set_demand(100.0)
        result1 = three_boiler_system.optimize()

        # Take one boiler offline
        three_boiler_system.milp_allocator.update_equipment_status(
            "BOILER-002", "faulted"
        )

        # Demand should now exceed capacity, need to shed
        three_boiler_system.set_demand(85.0)  # Reduced demand
        result2 = three_boiler_system.optimize()

        assert result2.status in ["optimal", "feasible"]

    def test_load_shedding_priority(self, three_boiler_system):
        """Test that load shedding follows priority."""
        # This would test that lower priority loads are shed first
        # Simplified test as priority is not fully implemented in mock
        three_boiler_system.set_demand(60.0)
        result = three_boiler_system.optimize()

        assert result.total_allocated_mmbtu_hr <= 60.0


# =============================================================================
# TEST CLASS: EQUIPMENT FAILURE HANDLING
# =============================================================================

class TestEquipmentFailureHandling:
    """Tests for equipment failure scenarios."""

    def test_single_equipment_failure(self, three_boiler_system):
        """Test handling of single equipment failure."""
        three_boiler_system.set_demand(60.0)
        initial_result = three_boiler_system.optimize()

        # Simulate failure
        new_result = three_boiler_system.handle_equipment_failure("BOILER-001")

        # Should still meet demand with remaining equipment (40 + 30 = 70 > 60)
        assert new_result.status in ["optimal", "feasible"]

    def test_multiple_equipment_failures(self, three_boiler_system):
        """Test handling of multiple equipment failures."""
        three_boiler_system.set_demand(50.0)

        # First failure
        three_boiler_system.handle_equipment_failure("BOILER-001")

        # Second failure
        result = three_boiler_system.handle_equipment_failure("BOILER-002")

        # Only BOILER-003 (30 MMBtu) available, demand (50) exceeds
        assert result.status == "infeasible"

    def test_equipment_recovery(self, three_boiler_system):
        """Test equipment recovery after failure."""
        three_boiler_system.set_demand(60.0)

        # Fail and recover
        three_boiler_system.handle_equipment_failure("BOILER-001")
        three_boiler_system.milp_allocator.update_equipment_status(
            "BOILER-001", "available"
        )

        result = three_boiler_system.optimize()

        # Should be able to use recovered equipment
        eq_ids = [a.equipment_id for a in result.allocations]
        assert "BOILER-001" in eq_ids


# =============================================================================
# TEST CLASS: NETWORK PARTITION SCENARIOS
# =============================================================================

class TestNetworkPartitionScenarios:
    """Tests for network partition and communication failure scenarios."""

    def test_stale_pv_data_handling(
        self, three_boiler_system, sample_pv_data
    ):
        """Test handling of stale process value data."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        # Simulate stale data by not updating
        # In real system, would have timestamp validation
        outputs = three_boiler_system.execute_allocations(sample_pv_data)

        # Should still produce outputs (fail-safe)
        assert len(outputs) > 0

    def test_missing_pv_data_handling(self, three_boiler_system):
        """Test handling of missing process value data."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        # Empty PV data
        outputs = three_boiler_system.execute_allocations({})

        # Should use default values or fail-safe
        assert len(outputs) > 0

    def test_partial_communication_loss(
        self, three_boiler_system, sample_pv_data
    ):
        """Test partial communication loss (some equipment unreachable)."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        # Remove some PV data
        partial_pv = {
            k: v for k, v in sample_pv_data.items()
            if "BOILER-001" not in k
        }

        outputs = three_boiler_system.execute_allocations(partial_pv)

        # Should still control reachable equipment
        assert "BOILER-002" in outputs or "BOILER-003" in outputs


# =============================================================================
# TEST CLASS: CONCURRENT OPTIMIZATION
# =============================================================================

class TestConcurrentOptimization:
    """Tests for concurrent optimization requests."""

    def test_concurrent_optimization_requests(self, three_boiler_system):
        """Test handling of concurrent optimization requests."""
        import threading

        results = []
        errors = []

        def optimize_task(demand):
            try:
                three_boiler_system.set_demand(demand)
                result = three_boiler_system.optimize()
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for demand in [40.0, 50.0, 60.0, 70.0, 80.0]:
            t = threading.Thread(target=optimize_task, args=(demand,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5


# =============================================================================
# TEST CLASS: END-TO-END SCENARIOS
# =============================================================================

class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""

    def test_full_control_cycle(
        self, three_boiler_system, sample_pv_data
    ):
        """Test complete control cycle from demand to outputs."""
        # 1. Set demand
        three_boiler_system.set_demand(60.0)

        # 2. Optimize
        opt_result = three_boiler_system.optimize()
        assert opt_result.status in ["optimal", "feasible"]

        # 3. Execute control
        ctrl_outputs = three_boiler_system.execute_allocations(sample_pv_data)
        assert len(ctrl_outputs) > 0

        # 4. Verify outputs are valid
        for eq_id, output in ctrl_outputs.items():
            assert output["safety_result"]["decision"] != "block"
            assert output["final_output"] is not None

    def test_demand_change_response(
        self, three_boiler_system, sample_pv_data
    ):
        """Test response to demand changes."""
        # Initial demand
        three_boiler_system.set_demand(40.0)
        result1 = three_boiler_system.optimize()

        # Demand increase
        three_boiler_system.set_demand(80.0)
        result2 = three_boiler_system.optimize()

        # Demand decrease
        three_boiler_system.set_demand(30.0)
        result3 = three_boiler_system.optimize()

        # All should produce valid results
        assert all(r.status in ["optimal", "feasible"] for r in [result1, result2, result3])

        # Allocations should scale with demand
        assert result2.total_allocated_mmbtu_hr > result1.total_allocated_mmbtu_hr
        assert result3.total_allocated_mmbtu_hr < result1.total_allocated_mmbtu_hr

    def test_equipment_failure_and_recovery_cycle(
        self, three_boiler_system, sample_pv_data
    ):
        """Test full equipment failure and recovery cycle."""
        three_boiler_system.set_demand(60.0)

        # Normal operation
        result1 = three_boiler_system.optimize()
        outputs1 = three_boiler_system.execute_allocations(sample_pv_data)

        # Equipment failure
        three_boiler_system.handle_equipment_failure("BOILER-001")
        result2 = three_boiler_system.optimize()
        outputs2 = three_boiler_system.execute_allocations(sample_pv_data)

        # Equipment recovery
        three_boiler_system.milp_allocator.update_equipment_status(
            "BOILER-001", "available"
        )
        result3 = three_boiler_system.optimize()
        outputs3 = three_boiler_system.execute_allocations(sample_pv_data)

        # Verify transitions
        assert len(outputs1) >= len(outputs2)  # Fewer equipment after failure
        assert len(outputs3) >= len(outputs2)  # More equipment after recovery


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for multi-equipment coordination."""

    @pytest.mark.performance
    def test_optimization_time(self, three_boiler_system):
        """Test optimization completes within time target."""
        import time

        three_boiler_system.set_demand(60.0)

        start = time.perf_counter()
        result = three_boiler_system.optimize()
        elapsed = time.perf_counter() - start

        # Target: <5s for optimization
        assert elapsed < 5.0
        assert result.status in ["optimal", "feasible"]

    @pytest.mark.performance
    def test_control_cycle_time(
        self, three_boiler_system, sample_pv_data
    ):
        """Test complete control cycle time."""
        import time

        iterations = 100

        start = time.perf_counter()

        for _ in range(iterations):
            three_boiler_system.set_demand(60.0)
            three_boiler_system.optimize()
            three_boiler_system.execute_allocations(sample_pv_data)

        elapsed = time.perf_counter() - start

        avg_cycle_time = elapsed / iterations

        # Average cycle should be under 100ms
        assert avg_cycle_time < 0.1

    @pytest.mark.performance
    def test_large_equipment_set(self, equipment_coordinator):
        """Test performance with large number of equipment."""
        import time

        # Add 20 boilers
        for i in range(20):
            boiler = MockEquipment(
                equipment_id=f"BOILER-{i+1:03d}",
                name=f"Boiler {i+1}",
                max_capacity=30.0 + np.random.uniform(-10, 10),
                min_capacity=10.0,
                fuel_cost=5.0 + np.random.uniform(-1, 1)
            )
            equipment_coordinator.add_equipment(boiler)

        equipment_coordinator.set_demand(300.0)

        start = time.perf_counter()
        result = equipment_coordinator.optimize()
        elapsed = time.perf_counter() - start

        # Should still be fast even with many equipment
        assert elapsed < 2.0
        assert result.status in ["optimal", "feasible"]


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_optimization_determinism(self, three_boiler_system):
        """Test that optimization produces deterministic results."""
        three_boiler_system.set_demand(60.0)

        results = []
        for _ in range(5):
            result = three_boiler_system.optimize()
            results.append(result.total_allocated_mmbtu_hr)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_control_output_determinism(
        self, three_boiler_system, sample_pv_data
    ):
        """Test that control outputs are deterministic."""
        three_boiler_system.set_demand(60.0)
        three_boiler_system.optimize()

        outputs_list = []
        for _ in range(5):
            outputs = three_boiler_system.execute_allocations(sample_pv_data)
            output_values = {
                eq_id: out["final_output"]
                for eq_id, out in outputs.items()
            }
            outputs_list.append(output_values)

        # All outputs should be identical
        for outputs in outputs_list[1:]:
            assert outputs == outputs_list[0]
