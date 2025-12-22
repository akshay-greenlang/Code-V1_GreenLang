"""
Simulation Tests: Digital Twin Surrogate Plant

Tests the digital twin simulation model including:
- Plant state simulation accuracy
- Dynamic response modeling
- Multi-equipment coordination
- Disturbance response
- State prediction accuracy

Reference: GL-001 Specification Section 11.2
Target Coverage: 85%+
"""

import pytest
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch


# =============================================================================
# Digital Twin Classes (Simulated Production Code)
# =============================================================================

@dataclass
class EquipmentState:
    """State of a single piece of equipment."""
    equipment_id: str
    equipment_type: str
    status: str  # 'running', 'standby', 'fault'
    setpoint: float
    process_variable: float
    output: float
    efficiency: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PlantState:
    """Overall plant state."""
    timestamp: datetime
    boilers: Dict[str, EquipmentState]
    pumps: Dict[str, EquipmentState]
    valves: Dict[str, EquipmentState]
    heat_demand: float
    heat_supply: float
    total_efficiency: float
    alarm_count: int = 0


class BoilerModel:
    """Simulates boiler thermal dynamics."""

    def __init__(self, boiler_id: str, capacity: float = 1000.0,
                 time_constant: float = 300.0, efficiency: float = 0.88):
        self.boiler_id = boiler_id
        self.capacity = capacity  # kW
        self.time_constant = time_constant  # seconds
        self.base_efficiency = efficiency
        self.current_output = 0.0
        self.current_temperature = 20.0
        self.setpoint = 0.0

    def step(self, dt: float, fuel_rate: float) -> Tuple[float, float]:
        """Advance boiler state by dt seconds.

        Returns: (heat_output_kw, temperature_c)
        """
        # First-order response model
        target_output = min(fuel_rate * self.base_efficiency, self.capacity)
        alpha = 1 - np.exp(-dt / self.time_constant)

        self.current_output += alpha * (target_output - self.current_output)

        # Temperature model (simplified)
        target_temp = 100 + (self.current_output / self.capacity) * 400
        self.current_temperature += alpha * (target_temp - self.current_temperature)

        return self.current_output, self.current_temperature

    def get_state(self) -> EquipmentState:
        """Get current equipment state."""
        return EquipmentState(
            equipment_id=self.boiler_id,
            equipment_type="boiler",
            status="running" if self.current_output > 0 else "standby",
            setpoint=self.setpoint,
            process_variable=self.current_temperature,
            output=self.current_output,
            efficiency=self.base_efficiency
        )


class PumpModel:
    """Simulates pump hydraulic dynamics."""

    def __init__(self, pump_id: str, max_flow: float = 500.0,
                 response_time: float = 10.0):
        self.pump_id = pump_id
        self.max_flow = max_flow  # m3/h
        self.response_time = response_time  # seconds
        self.current_flow = 0.0
        self.speed_setpoint = 0.0

    def step(self, dt: float, speed_percent: float) -> float:
        """Advance pump state by dt seconds.

        Returns: flow_rate_m3h
        """
        speed_percent = np.clip(speed_percent, 0, 100)
        target_flow = (speed_percent / 100) * self.max_flow

        alpha = 1 - np.exp(-dt / self.response_time)
        self.current_flow += alpha * (target_flow - self.current_flow)
        self.speed_setpoint = speed_percent

        return self.current_flow

    def get_state(self) -> EquipmentState:
        """Get current equipment state."""
        return EquipmentState(
            equipment_id=self.pump_id,
            equipment_type="pump",
            status="running" if self.current_flow > 0 else "standby",
            setpoint=self.speed_setpoint,
            process_variable=self.current_flow,
            output=self.current_flow,
            efficiency=0.75 if self.current_flow > 0 else 0.0
        )


class ValveModel:
    """Simulates control valve dynamics."""

    def __init__(self, valve_id: str, cv: float = 100.0,
                 stroke_time: float = 30.0):
        self.valve_id = valve_id
        self.cv = cv  # Flow coefficient
        self.stroke_time = stroke_time  # Full stroke time in seconds
        self.current_position = 0.0  # 0-100%
        self.setpoint = 0.0

    def step(self, dt: float, position_setpoint: float) -> float:
        """Advance valve state by dt seconds.

        Returns: valve_position_percent
        """
        position_setpoint = np.clip(position_setpoint, 0, 100)

        # Rate-limited response
        max_change = (dt / self.stroke_time) * 100
        change_needed = position_setpoint - self.current_position
        change = np.clip(change_needed, -max_change, max_change)

        self.current_position += change
        self.setpoint = position_setpoint

        return self.current_position

    def get_flow(self, pressure_drop: float) -> float:
        """Calculate flow through valve."""
        position_factor = (self.current_position / 100) ** 0.5
        return self.cv * position_factor * np.sqrt(max(0, pressure_drop))

    def get_state(self) -> EquipmentState:
        """Get current equipment state."""
        return EquipmentState(
            equipment_id=self.valve_id,
            equipment_type="valve",
            status="running",
            setpoint=self.setpoint,
            process_variable=self.current_position,
            output=self.current_position,
            efficiency=1.0
        )


class DigitalTwinPlant:
    """Digital twin model of the complete thermal plant."""

    def __init__(self):
        self.boilers: Dict[str, BoilerModel] = {}
        self.pumps: Dict[str, PumpModel] = {}
        self.valves: Dict[str, ValveModel] = {}
        self.current_time = datetime.now()
        self.heat_demand = 0.0
        self.ambient_temp = 20.0

    def add_boiler(self, boiler_id: str, **kwargs):
        """Add a boiler to the plant."""
        self.boilers[boiler_id] = BoilerModel(boiler_id, **kwargs)

    def add_pump(self, pump_id: str, **kwargs):
        """Add a pump to the plant."""
        self.pumps[pump_id] = PumpModel(pump_id, **kwargs)

    def add_valve(self, valve_id: str, **kwargs):
        """Add a valve to the plant."""
        self.valves[valve_id] = ValveModel(valve_id, **kwargs)

    def set_demand(self, demand_kw: float):
        """Set current heat demand."""
        self.heat_demand = max(0, demand_kw)

    def step(self, dt: float, commands: Dict[str, Dict[str, float]]) -> PlantState:
        """Advance plant state by dt seconds.

        Args:
            dt: Time step in seconds
            commands: Dict of equipment commands
                {'boilers': {'BOILER_001': fuel_rate, ...},
                 'pumps': {'PUMP_001': speed_percent, ...},
                 'valves': {'VALVE_001': position_percent, ...}}

        Returns:
            Current plant state
        """
        self.current_time += timedelta(seconds=dt)

        # Update boilers
        total_heat = 0.0
        boiler_commands = commands.get('boilers', {})
        for boiler_id, boiler in self.boilers.items():
            fuel_rate = boiler_commands.get(boiler_id, 0)
            heat_out, temp = boiler.step(dt, fuel_rate)
            total_heat += heat_out

        # Update pumps
        pump_commands = commands.get('pumps', {})
        for pump_id, pump in self.pumps.items():
            speed = pump_commands.get(pump_id, 0)
            pump.step(dt, speed)

        # Update valves
        valve_commands = commands.get('valves', {})
        for valve_id, valve in self.valves.items():
            position = valve_commands.get(valve_id, 0)
            valve.step(dt, position)

        # Calculate total efficiency
        running_boilers = [b for b in self.boilers.values() if b.current_output > 0]
        if running_boilers:
            total_efficiency = sum(b.base_efficiency for b in running_boilers) / len(running_boilers)
        else:
            total_efficiency = 0.0

        return PlantState(
            timestamp=self.current_time,
            boilers={bid: b.get_state() for bid, b in self.boilers.items()},
            pumps={pid: p.get_state() for pid, p in self.pumps.items()},
            valves={vid: v.get_state() for vid, v in self.valves.items()},
            heat_demand=self.heat_demand,
            heat_supply=total_heat,
            total_efficiency=total_efficiency
        )

    def get_state(self) -> PlantState:
        """Get current plant state without advancing time."""
        total_heat = sum(b.current_output for b in self.boilers.values())
        running_boilers = [b for b in self.boilers.values() if b.current_output > 0]
        total_efficiency = (
            sum(b.base_efficiency for b in running_boilers) / len(running_boilers)
            if running_boilers else 0.0
        )

        return PlantState(
            timestamp=self.current_time,
            boilers={bid: b.get_state() for bid, b in self.boilers.items()},
            pumps={pid: p.get_state() for pid, p in self.pumps.items()},
            valves={vid: v.get_state() for vid, v in self.valves.items()},
            heat_demand=self.heat_demand,
            heat_supply=total_heat,
            total_efficiency=total_efficiency
        )

    def run_simulation(self, duration: float, dt: float,
                       command_schedule: List[Tuple[float, Dict]]) -> List[PlantState]:
        """Run simulation for specified duration.

        Args:
            duration: Total simulation time in seconds
            dt: Time step in seconds
            command_schedule: List of (time, commands) tuples

        Returns:
            List of plant states at each time step
        """
        states = []
        current_time = 0.0
        schedule_idx = 0
        current_commands = {'boilers': {}, 'pumps': {}, 'valves': {}}

        while current_time < duration:
            # Check for command updates
            while (schedule_idx < len(command_schedule) and
                   command_schedule[schedule_idx][0] <= current_time):
                current_commands = command_schedule[schedule_idx][1]
                schedule_idx += 1

            state = self.step(dt, current_commands)
            states.append(state)
            current_time += dt

        return states


# =============================================================================
# Test Classes
# =============================================================================

class TestBoilerModel:
    """Test suite for boiler simulation model."""

    @pytest.fixture
    def boiler(self):
        """Create a test boiler."""
        return BoilerModel("BOILER_TEST", capacity=1000, time_constant=300)

    def test_boiler_initialization(self, boiler):
        """Test boiler initializes with correct defaults."""
        assert boiler.current_output == 0.0
        assert boiler.current_temperature == 20.0
        assert boiler.capacity == 1000

    def test_boiler_response_to_fuel(self, boiler):
        """Test boiler responds to fuel input."""
        initial_output = boiler.current_output

        # Step with fuel input
        heat_out, temp = boiler.step(dt=60, fuel_rate=500)

        assert heat_out > initial_output
        assert temp > 20.0

    def test_boiler_reaches_setpoint(self, boiler):
        """Test boiler reaches setpoint over time."""
        target_fuel = 800  # kW fuel rate
        expected_output = target_fuel * boiler.base_efficiency

        # Simulate for several time constants
        for _ in range(50):
            boiler.step(dt=60, fuel_rate=target_fuel)

        # Should be close to expected output
        assert pytest.approx(boiler.current_output, rel=0.05) == expected_output

    def test_boiler_respects_capacity(self, boiler):
        """Test boiler does not exceed capacity."""
        # Try to exceed capacity
        for _ in range(100):
            boiler.step(dt=60, fuel_rate=2000)

        assert boiler.current_output <= boiler.capacity

    def test_boiler_first_order_dynamics(self, boiler):
        """Test boiler exhibits first-order response."""
        outputs = []

        for i in range(100):
            heat_out, _ = boiler.step(dt=10, fuel_rate=500)
            outputs.append(heat_out)

        # Output should increase monotonically
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i-1]

        # Rate of change should decrease over time
        early_change = outputs[10] - outputs[5]
        late_change = outputs[90] - outputs[85]
        assert late_change < early_change

    def test_boiler_state_output(self, boiler):
        """Test boiler state output is correct."""
        boiler.step(dt=60, fuel_rate=500)
        state = boiler.get_state()

        assert state.equipment_id == "BOILER_TEST"
        assert state.equipment_type == "boiler"
        assert state.status == "running"
        assert state.output > 0


class TestPumpModel:
    """Test suite for pump simulation model."""

    @pytest.fixture
    def pump(self):
        """Create a test pump."""
        return PumpModel("PUMP_TEST", max_flow=500, response_time=10)

    def test_pump_initialization(self, pump):
        """Test pump initializes with correct defaults."""
        assert pump.current_flow == 0.0
        assert pump.max_flow == 500

    def test_pump_response_to_speed(self, pump):
        """Test pump responds to speed setpoint."""
        initial_flow = pump.current_flow

        flow = pump.step(dt=5, speed_percent=50)

        assert flow > initial_flow

    def test_pump_reaches_setpoint(self, pump):
        """Test pump reaches setpoint over time."""
        target_speed = 80  # 80%
        expected_flow = (80 / 100) * pump.max_flow

        # Simulate for several time constants
        for _ in range(20):
            pump.step(dt=5, speed_percent=target_speed)

        assert pytest.approx(pump.current_flow, rel=0.05) == expected_flow

    def test_pump_speed_limits(self, pump):
        """Test pump respects speed limits."""
        # Try invalid speed
        flow = pump.step(dt=5, speed_percent=150)

        # Should be clamped to 100%
        for _ in range(20):
            flow = pump.step(dt=5, speed_percent=150)

        assert flow <= pump.max_flow


class TestValveModel:
    """Test suite for valve simulation model."""

    @pytest.fixture
    def valve(self):
        """Create a test valve."""
        return ValveModel("VALVE_TEST", cv=100, stroke_time=30)

    def test_valve_initialization(self, valve):
        """Test valve initializes with correct defaults."""
        assert valve.current_position == 0.0
        assert valve.cv == 100

    def test_valve_opens_gradually(self, valve):
        """Test valve opens at rate-limited speed."""
        positions = []

        for _ in range(10):
            pos = valve.step(dt=1, position_setpoint=100)
            positions.append(pos)

        # Should increase gradually, not instantly
        assert positions[-1] < 100  # Not fully open after 10 seconds (30s stroke time)
        assert positions[-1] > 0  # But should have opened some

    def test_valve_reaches_setpoint(self, valve):
        """Test valve reaches setpoint over time."""
        target = 75.0

        # Simulate for full stroke time
        for _ in range(60):
            valve.step(dt=1, position_setpoint=target)

        assert pytest.approx(valve.current_position, rel=0.01) == target

    def test_valve_flow_calculation(self, valve):
        """Test valve flow calculation."""
        valve.current_position = 50  # 50% open

        flow = valve.get_flow(pressure_drop=4.0)

        # Flow = Cv * sqrt(position_factor) * sqrt(dP)
        # position_factor = (0.5)^0.5 = 0.707
        # flow = 100 * 0.707 * 2 = 141.4
        expected = 100 * np.sqrt(0.5) * np.sqrt(4.0)
        assert pytest.approx(flow, rel=0.01) == expected

    def test_valve_flow_zero_at_closed(self, valve):
        """Test valve has zero flow when closed."""
        valve.current_position = 0

        flow = valve.get_flow(pressure_drop=10.0)

        assert flow == 0


class TestDigitalTwinPlant:
    """Test suite for digital twin plant model."""

    @pytest.fixture
    def plant(self):
        """Create a test plant."""
        plant = DigitalTwinPlant()
        plant.add_boiler("BOILER_001", capacity=1000, efficiency=0.88)
        plant.add_boiler("BOILER_002", capacity=800, efficiency=0.86)
        plant.add_pump("PUMP_001", max_flow=500)
        plant.add_valve("VALVE_001", cv=100)
        return plant

    def test_plant_initialization(self, plant):
        """Test plant initializes correctly."""
        assert len(plant.boilers) == 2
        assert len(plant.pumps) == 1
        assert len(plant.valves) == 1

    def test_plant_step_advances_time(self, plant):
        """Test that stepping advances simulation time."""
        initial_time = plant.current_time

        plant.step(dt=60, commands={})

        assert plant.current_time > initial_time

    def test_plant_step_updates_equipment(self, plant):
        """Test that stepping updates equipment states."""
        commands = {
            'boilers': {'BOILER_001': 500, 'BOILER_002': 400},
            'pumps': {'PUMP_001': 75},
            'valves': {'VALVE_001': 50}
        }

        state = plant.step(dt=60, commands=commands)

        assert state.boilers['BOILER_001'].output > 0
        assert state.pumps['PUMP_001'].process_variable > 0
        assert state.valves['VALVE_001'].process_variable > 0

    def test_plant_heat_supply_calculation(self, plant):
        """Test plant calculates total heat supply."""
        commands = {
            'boilers': {'BOILER_001': 500, 'BOILER_002': 300}
        }

        # Simulate to steady state
        for _ in range(100):
            state = plant.step(dt=60, commands=commands)

        assert state.heat_supply > 0
        # Should be approximately sum of boiler outputs
        total_boiler_output = sum(b.output for b in state.boilers.values())
        assert pytest.approx(state.heat_supply, rel=0.01) == total_boiler_output

    def test_plant_run_simulation(self, plant):
        """Test plant simulation run."""
        schedule = [
            (0, {'boilers': {'BOILER_001': 500}}),
            (300, {'boilers': {'BOILER_001': 800}}),
            (600, {'boilers': {'BOILER_001': 300}})
        ]

        states = plant.run_simulation(duration=900, dt=30, command_schedule=schedule)

        assert len(states) == 30  # 900/30 = 30 steps
        # Heat supply should vary based on commands
        early_heat = states[5].heat_supply
        mid_heat = states[15].heat_supply
        late_heat = states[25].heat_supply

        # After command changes, outputs should differ
        assert mid_heat != early_heat or late_heat != mid_heat

    def test_plant_demand_tracking(self, plant):
        """Test plant demand setting."""
        plant.set_demand(1500)

        state = plant.get_state()

        assert state.heat_demand == 1500

    def test_plant_negative_demand_clamped(self, plant):
        """Test that negative demand is clamped to zero."""
        plant.set_demand(-100)

        assert plant.heat_demand == 0


class TestDisturbanceResponse:
    """Test plant response to disturbances."""

    @pytest.fixture
    def plant_with_controllers(self):
        """Create plant with simulated disturbances."""
        plant = DigitalTwinPlant()
        plant.add_boiler("BOILER_001", capacity=1000)
        plant.add_boiler("BOILER_002", capacity=800)
        return plant

    def test_demand_step_change_response(self, plant_with_controllers):
        """Test plant response to demand step change."""
        plant = plant_with_controllers

        # Initial steady state
        initial_commands = {'boilers': {'BOILER_001': 500, 'BOILER_002': 400}}
        for _ in range(50):
            plant.step(dt=60, initial_commands)

        initial_state = plant.get_state()

        # Step change in demand
        plant.set_demand(initial_state.heat_supply * 1.5)

        # New commands to meet demand
        new_commands = {'boilers': {'BOILER_001': 750, 'BOILER_002': 600}}
        for _ in range(50):
            state = plant.step(dt=60, new_commands)

        # Should have increased supply
        assert state.heat_supply > initial_state.heat_supply

    def test_equipment_trip_response(self, plant_with_controllers):
        """Test plant response to equipment trip."""
        plant = plant_with_controllers

        # Initial state with both boilers running
        commands = {'boilers': {'BOILER_001': 500, 'BOILER_002': 400}}
        for _ in range(50):
            plant.step(dt=60, commands)

        pre_trip_state = plant.get_state()

        # Simulate boiler trip (set fuel to 0)
        trip_commands = {'boilers': {'BOILER_001': 0, 'BOILER_002': 400}}
        for _ in range(50):
            state = plant.step(dt=60, trip_commands)

        # Heat supply should decrease
        assert state.heat_supply < pre_trip_state.heat_supply


class TestPredictionAccuracy:
    """Test prediction accuracy of digital twin."""

    def test_steady_state_prediction(self):
        """Test prediction accuracy at steady state."""
        plant = DigitalTwinPlant()
        plant.add_boiler("BOILER_001", capacity=1000, efficiency=0.88)

        fuel_rate = 800

        # Run to steady state
        for _ in range(100):
            plant.step(dt=60, commands={'boilers': {'BOILER_001': fuel_rate}})

        state = plant.get_state()
        expected_output = fuel_rate * 0.88

        # Should be within 5% of expected
        assert pytest.approx(state.heat_supply, rel=0.05) == expected_output

    def test_transient_response_shape(self):
        """Test that transient response has correct shape."""
        plant = DigitalTwinPlant()
        plant.add_boiler("BOILER_001", capacity=1000, time_constant=300)

        outputs = []
        commands = {'boilers': {'BOILER_001': 800}}

        for _ in range(60):  # 60 steps x 10s = 600s = 2 time constants
            state = plant.step(dt=10, commands=commands)
            outputs.append(state.heat_supply)

        # At 1 time constant (t=300s, step 30), should be at ~63%
        # At 2 time constants (t=600s, step 60), should be at ~86%
        final_output = outputs[-1]
        one_tc_output = outputs[29]

        # 63% of final
        assert pytest.approx(one_tc_output / final_output, rel=0.1) == 0.63
