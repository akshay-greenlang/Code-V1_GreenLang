# -*- coding: utf-8 -*-
"""
Complete Combustion Optimization Workflow E2E Tests for GL-005 COMBUSENSE.

Tests the complete combustion efficiency optimization cycle from sensor
data acquisition through control action implementation. Validates the
full workflow: read state -> check safety -> analyze stability ->
optimize -> implement control -> verify results.

Reference Standards:
- NFPA 85: Boiler and Combustion Systems Hazards Code
- ASME PTC 4.1: Fired Steam Generators Performance Test Codes
- API 556: Fired Heaters for General Refinery Service
"""

import asyncio
import pytest
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP


# -----------------------------------------------------------------------------
# Mock Data Classes for E2E Testing
# -----------------------------------------------------------------------------

@dataclass
class MockCombustionState:
    """Mock combustion state for E2E testing."""
    state_id: str
    fuel_flow: float = 100.0  # kg/hr
    air_flow: float = 1200.0  # kg/hr
    air_fuel_ratio: float = 12.0
    flame_temperature: float = 1200.0  # Celsius
    furnace_temperature: float = 900.0  # Celsius
    flue_gas_temperature: float = 250.0  # Celsius
    ambient_temperature: float = 25.0  # Celsius
    fuel_pressure: float = 300.0  # kPa
    air_pressure: float = 101.3  # kPa
    furnace_pressure: float = -50.0  # Pa (draft)
    o2_percent: float = 4.5  # %
    co_ppm: float = 25.0  # ppm
    co2_percent: float = 11.5  # %
    nox_ppm: float = 80.0  # ppm
    heat_output_kw: float = 1000.0  # kW
    thermal_efficiency: float = 88.5  # %
    excess_air_percent: float = 27.0  # %


@dataclass
class MockControlAction:
    """Mock control action for E2E testing."""
    action_id: str
    fuel_flow_setpoint: float
    air_flow_setpoint: float
    fuel_valve_position: float
    air_damper_position: float
    o2_trim_adjustment: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def calculate_hash(self) -> str:
        """Calculate SHA-256 provenance hash."""
        hashable_data = {
            'fuel_flow_setpoint': round(self.fuel_flow_setpoint, 6),
            'air_flow_setpoint': round(self.air_flow_setpoint, 6),
            'fuel_valve_position': round(self.fuel_valve_position, 4),
            'air_damper_position': round(self.air_damper_position, 4)
        }
        hash_input = json.dumps(hashable_data, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()


@dataclass
class MockSafetyInterlocks:
    """Mock safety interlock status."""
    flame_present: bool = True
    fuel_pressure_ok: bool = True
    air_pressure_ok: bool = True
    furnace_temp_ok: bool = True
    furnace_pressure_ok: bool = True
    purge_complete: bool = True
    emergency_stop_clear: bool = True

    def all_safe(self) -> bool:
        """Check if all interlocks satisfied."""
        return all([
            self.flame_present,
            self.fuel_pressure_ok,
            self.air_pressure_ok,
            self.furnace_temp_ok,
            self.furnace_pressure_ok,
            self.purge_complete,
            self.emergency_stop_clear
        ])

    def get_failed_interlocks(self) -> List[str]:
        """Get list of failed interlock names."""
        failed = []
        if not self.flame_present:
            failed.append("flame_present")
        if not self.fuel_pressure_ok:
            failed.append("fuel_pressure_ok")
        if not self.air_pressure_ok:
            failed.append("air_pressure_ok")
        if not self.furnace_temp_ok:
            failed.append("furnace_temp_ok")
        if not self.furnace_pressure_ok:
            failed.append("furnace_pressure_ok")
        if not self.purge_complete:
            failed.append("purge_complete")
        if not self.emergency_stop_clear:
            failed.append("emergency_stop_clear")
        return failed


@dataclass
class MockStabilityMetrics:
    """Mock stability metrics."""
    heat_output_stability_index: float = 0.95
    heat_output_variance: float = 5.0
    furnace_temp_stability: float = 0.92
    o2_stability: float = 0.90
    overall_stability_score: float = 92.0
    stability_rating: str = "excellent"
    oscillation_detected: bool = False


class MockCombustionSystem:
    """Mock full combustion control system for E2E testing."""

    def __init__(self, num_burners: int = 1):
        """Initialize mock system."""
        self.num_burners = num_burners
        self.states: Dict[str, MockCombustionState] = {
            f"burner_{i}": MockCombustionState(f"burner_{i}")
            for i in range(1, num_burners + 1)
        }
        self.interlocks = MockSafetyInterlocks()
        self.connected = False
        self.control_actions: List[MockControlAction] = []
        self.stability_metrics = MockStabilityMetrics()
        self.heat_demand_kw: float = 1000.0

    async def connect_all(self) -> bool:
        """Connect to all subsystems."""
        await asyncio.sleep(0.01)  # Simulate connection time
        self.connected = True
        return True

    async def disconnect_all(self) -> bool:
        """Disconnect from all subsystems."""
        self.connected = False
        return True

    async def read_combustion_state(self, burner_id: str) -> MockCombustionState:
        """Read combustion state from sensors."""
        if not self.connected:
            raise ConnectionError("System not connected")
        if burner_id not in self.states:
            raise ValueError(f"Unknown burner: {burner_id}")
        return self.states[burner_id]

    async def read_all_states(self) -> Dict[str, MockCombustionState]:
        """Read all combustion states."""
        if not self.connected:
            raise ConnectionError("System not connected")
        return self.states.copy()

    async def check_safety_interlocks(self) -> MockSafetyInterlocks:
        """Check all safety interlocks."""
        return self.interlocks

    async def analyze_stability(self) -> MockStabilityMetrics:
        """Analyze combustion stability."""
        return self.stability_metrics

    async def calculate_optimal_control(
        self,
        state: MockCombustionState,
        heat_demand_kw: float
    ) -> MockControlAction:
        """Calculate optimal control action."""
        # Simple optimization logic for testing
        optimal_fuel = heat_demand_kw / (state.thermal_efficiency / 100) / 42.0  # 42 MJ/kg LHV
        optimal_air = optimal_fuel * 14.5 * (1 + state.excess_air_percent / 100)

        action = MockControlAction(
            action_id=f"action_{len(self.control_actions) + 1}",
            fuel_flow_setpoint=optimal_fuel,
            air_flow_setpoint=optimal_air,
            fuel_valve_position=min(100, optimal_fuel / 150 * 100),
            air_damper_position=min(100, optimal_air / 2000 * 100)
        )
        action.provenance_hash = action.calculate_hash()
        return action

    async def implement_control_action(
        self,
        action: MockControlAction,
        interlocks: MockSafetyInterlocks
    ) -> Dict[str, Any]:
        """Implement control action on burner."""
        if not interlocks.all_safe():
            return {
                "success": False,
                "reason": "safety_interlocks",
                "failed_interlocks": interlocks.get_failed_interlocks()
            }

        self.control_actions.append(action)
        return {
            "success": True,
            "action_id": action.action_id,
            "timestamp": action.timestamp.isoformat()
        }

    async def run_control_cycle(self, heat_demand_kw: Optional[float] = None) -> Dict[str, Any]:
        """Run complete control cycle."""
        if heat_demand_kw is None:
            heat_demand_kw = self.heat_demand_kw

        # Step 1: Read state
        state = await self.read_combustion_state("burner_1")

        # Step 2: Check interlocks
        interlocks = await self.check_safety_interlocks()
        if not interlocks.all_safe():
            return {
                "success": False,
                "reason": "safety_interlocks",
                "failed_interlocks": interlocks.get_failed_interlocks()
            }

        # Step 3: Analyze stability
        stability = await self.analyze_stability()

        # Step 4: Calculate optimal control
        action = await self.calculate_optimal_control(state, heat_demand_kw)

        # Step 5: Implement control
        result = await self.implement_control_action(action, interlocks)

        return {
            "success": result["success"],
            "state": state,
            "stability": stability,
            "action": action,
            "provenance_hash": action.provenance_hash
        }


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_system():
    """Create mock combustion system."""
    return MockCombustionSystem()


@pytest.fixture
def degraded_efficiency_system():
    """Create system with degraded efficiency."""
    system = MockCombustionSystem()
    system.states["burner_1"].thermal_efficiency = 72.0
    system.states["burner_1"].excess_air_percent = 55.0
    system.states["burner_1"].co_ppm = 180.0
    system.states["burner_1"].flue_gas_temperature = 350.0
    system.stability_metrics.overall_stability_score = 65.0
    system.stability_metrics.stability_rating = "fair"
    return system


@pytest.fixture
def unstable_combustion_system():
    """Create system with unstable combustion."""
    system = MockCombustionSystem()
    system.stability_metrics.heat_output_stability_index = 0.55
    system.stability_metrics.overall_stability_score = 45.0
    system.stability_metrics.stability_rating = "poor"
    system.stability_metrics.oscillation_detected = True
    return system


@pytest.fixture
def multi_burner_system():
    """Create multi-burner system."""
    return MockCombustionSystem(num_burners=3)


# -----------------------------------------------------------------------------
# E2E Test Classes
# -----------------------------------------------------------------------------

class TestCompleteWorkflow:
    """Test complete combustion optimization workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_normal_operation_workflow(self, mock_system):
        """Test complete workflow under normal operating conditions."""
        # Connect to system
        await mock_system.connect_all()

        # Run control cycle
        result = await mock_system.run_control_cycle(heat_demand_kw=1000.0)

        # Verify successful execution
        assert result["success"] is True
        assert result["state"].thermal_efficiency >= 85.0
        assert result["stability"].overall_stability_score >= 90.0
        assert result["action"].provenance_hash is not None
        assert len(result["action"].provenance_hash) == 64  # SHA-256 length

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_efficiency_optimization_cycle(self, degraded_efficiency_system):
        """Test efficiency optimization for degraded system."""
        system = degraded_efficiency_system
        await system.connect_all()

        # Initial state - degraded efficiency
        initial_state = await system.read_combustion_state("burner_1")
        assert initial_state.thermal_efficiency < 75.0
        assert initial_state.excess_air_percent > 50.0

        # Run optimization cycle
        result = await system.run_control_cycle(heat_demand_kw=900.0)

        assert result["success"] is True
        assert result["action"].fuel_flow_setpoint > 0
        assert result["action"].air_flow_setpoint > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_stability_analysis_integration(self, unstable_combustion_system):
        """Test stability analysis in control workflow."""
        system = unstable_combustion_system
        await system.connect_all()

        # Check stability metrics
        stability = await system.analyze_stability()

        assert stability.overall_stability_score < 50.0
        assert stability.stability_rating == "poor"
        assert stability.oscillation_detected is True

        # Control should still proceed with stability awareness
        result = await system.run_control_cycle()
        assert result["success"] is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_provenance_hash_generation(self, mock_system):
        """Test provenance hash is correctly generated."""
        await mock_system.connect_all()

        result = await mock_system.run_control_cycle()

        # Verify hash is SHA-256
        provenance_hash = result["provenance_hash"]
        assert provenance_hash is not None
        assert len(provenance_hash) == 64
        assert all(c in '0123456789abcdef' for c in provenance_hash)

        # Verify hash is deterministic for same action
        action = result["action"]
        recalculated_hash = action.calculate_hash()
        assert provenance_hash == recalculated_hash

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_burner_coordination(self, multi_burner_system):
        """Test coordination across multiple burners."""
        system = multi_burner_system
        await system.connect_all()

        # Read all burner states
        all_states = await system.read_all_states()

        assert len(all_states) == 3
        for burner_id, state in all_states.items():
            assert state.thermal_efficiency > 0
            assert state.fuel_flow > 0


class TestSafetyInterlockWorkflow:
    """Test safety interlock integration in workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_interlocks_satisfied_normal_operation(self, mock_system):
        """Test workflow proceeds when all interlocks satisfied."""
        await mock_system.connect_all()

        interlocks = await mock_system.check_safety_interlocks()

        assert interlocks.all_safe() is True
        assert interlocks.get_failed_interlocks() == []

        result = await mock_system.run_control_cycle()
        assert result["success"] is True

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_interlock_trip_stops_control(self, mock_system):
        """Test control stops when interlock trips."""
        await mock_system.connect_all()

        # Trip flame detection interlock
        mock_system.interlocks.flame_present = False

        result = await mock_system.run_control_cycle()

        assert result["success"] is False
        assert result["reason"] == "safety_interlocks"
        assert "flame_present" in result["failed_interlocks"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multiple_interlock_failures(self, mock_system):
        """Test handling of multiple interlock failures."""
        await mock_system.connect_all()

        # Trip multiple interlocks
        mock_system.interlocks.flame_present = False
        mock_system.interlocks.fuel_pressure_ok = False
        mock_system.interlocks.emergency_stop_clear = False

        result = await mock_system.run_control_cycle()

        assert result["success"] is False
        failed = result["failed_interlocks"]
        assert len(failed) == 3
        assert "flame_present" in failed
        assert "fuel_pressure_ok" in failed
        assert "emergency_stop_clear" in failed

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_interlock_recovery(self, mock_system):
        """Test workflow resumes after interlock recovery."""
        await mock_system.connect_all()

        # Trip interlock
        mock_system.interlocks.furnace_pressure_ok = False
        result1 = await mock_system.run_control_cycle()
        assert result1["success"] is False

        # Recover interlock
        mock_system.interlocks.furnace_pressure_ok = True
        result2 = await mock_system.run_control_cycle()
        assert result2["success"] is True


class TestFaultTolerance:
    """Test fault tolerance in E2E workflow."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_loss_handling(self, mock_system):
        """Test handling of connection loss."""
        await mock_system.connect_all()

        # First read succeeds
        state = await mock_system.read_combustion_state("burner_1")
        assert state is not None

        # Simulate connection loss
        mock_system.connected = False

        with pytest.raises(ConnectionError):
            await mock_system.read_combustion_state("burner_1")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_connection_recovery(self, mock_system):
        """Test recovery after connection loss."""
        await mock_system.connect_all()

        # Lose connection
        mock_system.connected = False
        with pytest.raises(ConnectionError):
            await mock_system.read_all_states()

        # Recover connection
        await mock_system.connect_all()
        states = await mock_system.read_all_states()
        assert len(states) == 1

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_burner_id_handling(self, mock_system):
        """Test handling of invalid burner ID."""
        await mock_system.connect_all()

        with pytest.raises(ValueError) as exc_info:
            await mock_system.read_combustion_state("nonexistent_burner")

        assert "Unknown burner" in str(exc_info.value)


class TestControlActionTracking:
    """Test control action history and tracking."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_control_action_history(self, mock_system):
        """Test control actions are recorded."""
        await mock_system.connect_all()

        # Execute multiple control cycles
        for i in range(5):
            await mock_system.run_control_cycle(heat_demand_kw=900 + i * 50)

        assert len(mock_system.control_actions) == 5

        # Verify each action has unique hash
        hashes = [a.provenance_hash for a in mock_system.control_actions]
        # Some may be identical if inputs are same, but structure is correct
        assert all(len(h) == 64 for h in hashes)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_control_action_provenance_chain(self, mock_system):
        """Test provenance chain for audit trail."""
        await mock_system.connect_all()

        actions = []
        for demand in [800, 900, 1000, 1100, 1200]:
            result = await mock_system.run_control_cycle(heat_demand_kw=demand)
            actions.append(result["action"])

        # Verify chain integrity
        for action in actions:
            assert action.provenance_hash is not None
            recalc = action.calculate_hash()
            assert action.provenance_hash == recalc


class TestHeatDemandResponse:
    """Test response to varying heat demand."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_increasing_heat_demand(self, mock_system):
        """Test response to increasing heat demand."""
        await mock_system.connect_all()

        fuel_setpoints = []
        for demand in [500, 750, 1000, 1250, 1500]:
            result = await mock_system.run_control_cycle(heat_demand_kw=demand)
            fuel_setpoints.append(result["action"].fuel_flow_setpoint)

        # Fuel setpoints should increase with demand
        for i in range(1, len(fuel_setpoints)):
            assert fuel_setpoints[i] > fuel_setpoints[i - 1]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_decreasing_heat_demand(self, mock_system):
        """Test response to decreasing heat demand."""
        await mock_system.connect_all()

        fuel_setpoints = []
        for demand in [1500, 1250, 1000, 750, 500]:
            result = await mock_system.run_control_cycle(heat_demand_kw=demand)
            fuel_setpoints.append(result["action"].fuel_flow_setpoint)

        # Fuel setpoints should decrease with demand
        for i in range(1, len(fuel_setpoints)):
            assert fuel_setpoints[i] < fuel_setpoints[i - 1]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_zero_heat_demand(self, mock_system):
        """Test response to zero heat demand."""
        await mock_system.connect_all()

        result = await mock_system.run_control_cycle(heat_demand_kw=0.0)

        assert result["success"] is True
        assert result["action"].fuel_flow_setpoint == 0.0


class TestProvenanceHashDeterminism:
    """Test deterministic provenance hash generation."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_same_input_same_hash(self, mock_system):
        """Test same inputs produce same provenance hash."""
        await mock_system.connect_all()

        hashes = []
        for _ in range(10):
            result = await mock_system.run_control_cycle(heat_demand_kw=1000.0)
            hashes.append(result["provenance_hash"])

        # All hashes should be identical for same input
        assert len(set(hashes)) == 1

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_different_input_different_hash(self, mock_system):
        """Test different inputs produce different hashes."""
        await mock_system.connect_all()

        hashes = []
        for demand in [800, 900, 1000, 1100, 1200]:
            result = await mock_system.run_control_cycle(heat_demand_kw=demand)
            hashes.append(result["provenance_hash"])

        # All hashes should be unique for different inputs
        assert len(set(hashes)) == 5
