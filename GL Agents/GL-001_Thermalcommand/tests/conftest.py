"""
GL-001 ThermalCommand Test Suite - Global Configuration and Fixtures

This module provides shared fixtures, configuration, and utilities
for the comprehensive test suite covering:
- Unit tests (schemas, converters, constraints, calculators)
- Simulation tests (digital twin, dispatch, MILP, PID)
- Integration tests (OPC-UA, Kafka, GraphQL/gRPC, CMMS)
- Safety tests (boundaries, SIS, alarms, E-stop)
- Performance tests (cycle time, throughput, memory, API response)
- Determinism tests (reproducibility, SHA-256 provenance)
- Acceptance tests (heat delivery, SLAs, explainability, operator workflows)

Target Coverage: 85%+
Reference: GL-001 Specification Section 11

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np


# =============================================================================
# Configuration Constants
# =============================================================================

# Test environment settings
TEST_SEED = 42  # For reproducibility
TEST_TIMEOUT = 30  # seconds
PERFORMANCE_TIMEOUT = 60  # seconds for performance tests

# Performance targets from specification
OPTIMIZATION_CYCLE_TIME_TARGET = 5.0  # seconds
DATA_PROCESSING_RATE_TARGET = 10000  # points per second
MEMORY_USAGE_TARGET = 512  # MB
API_RESPONSE_TIME_TARGET = 0.200  # seconds (200ms)

# Safety boundaries
TEMPERATURE_MIN = 0.0  # Celsius
TEMPERATURE_MAX = 1200.0  # Celsius
PRESSURE_MIN = 0.0  # bar
PRESSURE_MAX = 100.0  # bar
FLOW_RATE_MIN = 0.0  # m3/h
FLOW_RATE_MAX = 10000.0  # m3/h

# Thermal efficiency parameters
BOILER_EFFICIENCY_MIN = 0.70
BOILER_EFFICIENCY_MAX = 0.98
HEAT_RECOVERY_EFFICIENCY_MIN = 0.50
HEAT_RECOVERY_EFFICIENCY_MAX = 0.95


# =============================================================================
# Data Classes for Test Fixtures
# =============================================================================

@dataclass
class ThermalMeasurement:
    """Represents a single thermal measurement point."""
    timestamp: datetime
    temperature: float
    pressure: float
    flow_rate: float
    energy_input: float
    energy_output: float
    efficiency: float
    sensor_id: str
    quality_score: float = 1.0


@dataclass
class BoilerState:
    """Represents boiler operational state."""
    boiler_id: str
    status: str  # 'running', 'standby', 'maintenance', 'fault'
    temperature: float
    pressure: float
    fuel_rate: float
    steam_output: float
    efficiency: float
    runtime_hours: float
    last_maintenance: datetime


@dataclass
class HeatDemand:
    """Represents heat demand from a consumer."""
    consumer_id: str
    demand_type: str  # 'process', 'hvac', 'hot_water'
    required_temperature: float
    required_flow_rate: float
    priority: int  # 1-5, where 1 is highest priority
    tolerance_temp: float
    tolerance_flow: float


@dataclass
class OptimizationResult:
    """Represents MILP optimization output."""
    timestamp: datetime
    objective_value: float
    boiler_setpoints: Dict[str, float]
    valve_positions: Dict[str, float]
    pump_speeds: Dict[str, float]
    predicted_cost: float
    predicted_emissions: float
    solver_status: str
    solve_time: float
    provenance_hash: str


@dataclass
class SafetyEvent:
    """Represents a safety-related event."""
    timestamp: datetime
    event_type: str  # 'alarm', 'warning', 'trip', 'permissive_change'
    severity: str  # 'low', 'medium', 'high', 'critical'
    source: str
    message: str
    acknowledged: bool = False
    cleared: bool = False


# =============================================================================
# Seed Management for Determinism
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure reproducible randomness across all tests."""
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    yield


# =============================================================================
# Mock Data Generators
# =============================================================================

class TestDataGenerator:
    """Generate realistic test data for ThermalCommand testing."""

    def __init__(self, seed: int = TEST_SEED):
        """Initialize generator with seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.base_time = datetime(2025, 1, 1, 0, 0, 0)

    def generate_thermal_measurements(
        self,
        count: int = 100,
        sensor_id: str = "SENSOR_001"
    ) -> List[ThermalMeasurement]:
        """Generate realistic thermal measurement data."""
        measurements = []

        for i in range(count):
            timestamp = self.base_time + timedelta(seconds=i * 10)

            # Generate correlated thermal data
            base_temp = 450.0 + 50.0 * np.sin(2 * np.pi * i / 100)
            temperature = base_temp + np.random.normal(0, 2)

            base_pressure = 15.0 + 2.0 * np.sin(2 * np.pi * i / 100)
            pressure = base_pressure + np.random.normal(0, 0.1)

            base_flow = 500.0 + 50.0 * np.sin(2 * np.pi * i / 100)
            flow_rate = base_flow + np.random.normal(0, 5)

            energy_input = flow_rate * 4.18 * (temperature - 20) / 3600
            efficiency = 0.85 + np.random.normal(0, 0.02)
            energy_output = energy_input * efficiency

            measurements.append(ThermalMeasurement(
                timestamp=timestamp,
                temperature=max(0, temperature),
                pressure=max(0, pressure),
                flow_rate=max(0, flow_rate),
                energy_input=max(0, energy_input),
                energy_output=max(0, energy_output),
                efficiency=np.clip(efficiency, 0.5, 1.0),
                sensor_id=sensor_id,
                quality_score=np.random.uniform(0.95, 1.0)
            ))

        return measurements

    def generate_boiler_states(self, count: int = 5) -> List[BoilerState]:
        """Generate boiler state data."""
        statuses = ['running', 'running', 'running', 'standby', 'maintenance']
        boilers = []

        for i in range(count):
            status = statuses[i % len(statuses)]

            if status == 'running':
                temperature = 450.0 + np.random.normal(0, 10)
                pressure = 15.0 + np.random.normal(0, 0.5)
                fuel_rate = 100.0 + np.random.normal(0, 5)
                steam_output = 800.0 + np.random.normal(0, 20)
                efficiency = 0.88 + np.random.normal(0, 0.02)
            elif status == 'standby':
                temperature = 200.0 + np.random.normal(0, 5)
                pressure = 5.0 + np.random.normal(0, 0.2)
                fuel_rate = 10.0
                steam_output = 0.0
                efficiency = 0.0
            else:
                temperature = 20.0
                pressure = 0.0
                fuel_rate = 0.0
                steam_output = 0.0
                efficiency = 0.0

            boilers.append(BoilerState(
                boiler_id=f"BOILER_{i+1:03d}",
                status=status,
                temperature=max(0, temperature),
                pressure=max(0, pressure),
                fuel_rate=max(0, fuel_rate),
                steam_output=max(0, steam_output),
                efficiency=np.clip(efficiency, 0, 1),
                runtime_hours=np.random.uniform(0, 10000),
                last_maintenance=self.base_time - timedelta(days=np.random.randint(1, 365))
            ))

        return boilers

    def generate_heat_demands(self, count: int = 10) -> List[HeatDemand]:
        """Generate heat demand data."""
        demand_types = ['process', 'hvac', 'hot_water']
        demands = []

        for i in range(count):
            demand_type = demand_types[i % len(demand_types)]

            if demand_type == 'process':
                temp = 350.0 + np.random.uniform(-50, 50)
                flow = 200.0 + np.random.uniform(-50, 100)
                priority = np.random.randint(1, 3)
            elif demand_type == 'hvac':
                temp = 80.0 + np.random.uniform(-10, 10)
                flow = 50.0 + np.random.uniform(-10, 20)
                priority = np.random.randint(3, 5)
            else:
                temp = 60.0 + np.random.uniform(-5, 5)
                flow = 30.0 + np.random.uniform(-5, 10)
                priority = np.random.randint(2, 4)

            demands.append(HeatDemand(
                consumer_id=f"CONSUMER_{i+1:03d}",
                demand_type=demand_type,
                required_temperature=temp,
                required_flow_rate=flow,
                priority=priority,
                tolerance_temp=temp * 0.02,  # 2% tolerance
                tolerance_flow=flow * 0.05   # 5% tolerance
            ))

        return demands

    def generate_optimization_result(self) -> OptimizationResult:
        """Generate a sample optimization result."""
        timestamp = datetime.now()

        # Create deterministic hash for provenance
        data_for_hash = f"{timestamp.isoformat()}_optimization_result"
        provenance_hash = hashlib.sha256(data_for_hash.encode()).hexdigest()

        return OptimizationResult(
            timestamp=timestamp,
            objective_value=np.random.uniform(1000, 5000),
            boiler_setpoints={
                "BOILER_001": 85.0 + np.random.uniform(-5, 5),
                "BOILER_002": 80.0 + np.random.uniform(-5, 5),
                "BOILER_003": 75.0 + np.random.uniform(-5, 5)
            },
            valve_positions={
                "VALVE_001": np.random.uniform(0.3, 0.7),
                "VALVE_002": np.random.uniform(0.4, 0.8),
                "VALVE_003": np.random.uniform(0.5, 0.9)
            },
            pump_speeds={
                "PUMP_001": np.random.uniform(0.6, 0.9),
                "PUMP_002": np.random.uniform(0.5, 0.85)
            },
            predicted_cost=np.random.uniform(5000, 15000),
            predicted_emissions=np.random.uniform(100, 500),
            solver_status="optimal",
            solve_time=np.random.uniform(0.5, 3.0),
            provenance_hash=provenance_hash
        )

    def generate_safety_events(self, count: int = 20) -> List[SafetyEvent]:
        """Generate safety event data."""
        event_types = ['alarm', 'warning', 'trip', 'permissive_change']
        severities = ['low', 'medium', 'high', 'critical']
        sources = ['BOILER_001', 'BOILER_002', 'PUMP_001', 'VALVE_001', 'SENSOR_001']

        events = []
        for i in range(count):
            event_type = np.random.choice(event_types, p=[0.5, 0.3, 0.1, 0.1])
            severity = np.random.choice(severities, p=[0.4, 0.35, 0.2, 0.05])

            events.append(SafetyEvent(
                timestamp=self.base_time + timedelta(minutes=i * 5),
                event_type=event_type,
                severity=severity,
                source=np.random.choice(sources),
                message=f"Test {event_type} event {i+1}",
                acknowledged=np.random.random() > 0.3,
                cleared=np.random.random() > 0.5
            ))

        return events


# =============================================================================
# Global Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator for entire test session."""
    return TestDataGenerator(seed=TEST_SEED)


@pytest.fixture
def thermal_measurements(test_data_generator) -> List[ThermalMeasurement]:
    """Provide sample thermal measurements."""
    return test_data_generator.generate_thermal_measurements(count=100)


@pytest.fixture
def boiler_states(test_data_generator) -> List[BoilerState]:
    """Provide sample boiler states."""
    return test_data_generator.generate_boiler_states(count=5)


@pytest.fixture
def heat_demands(test_data_generator) -> List[HeatDemand]:
    """Provide sample heat demands."""
    return test_data_generator.generate_heat_demands(count=10)


@pytest.fixture
def optimization_result(test_data_generator) -> OptimizationResult:
    """Provide sample optimization result."""
    return test_data_generator.generate_optimization_result()


@pytest.fixture
def safety_events(test_data_generator) -> List[SafetyEvent]:
    """Provide sample safety events."""
    return test_data_generator.generate_safety_events(count=20)


# =============================================================================
# Mock Service Fixtures
# =============================================================================

@pytest.fixture
def mock_opc_ua_client():
    """Mock OPC-UA client for integration testing."""
    client = MagicMock()
    client.connect = MagicMock(return_value=True)
    client.disconnect = MagicMock(return_value=True)
    client.read_node = MagicMock(return_value={"value": 450.0, "quality": "Good"})
    client.write_node = MagicMock(return_value=True)
    client.subscribe = MagicMock(return_value="subscription_123")
    client.get_endpoints = MagicMock(return_value=[
        {"url": "opc.tcp://localhost:4840", "security_mode": "SignAndEncrypt"}
    ])
    return client


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for integration testing."""
    producer = MagicMock()
    producer.send = MagicMock(return_value=MagicMock(get=MagicMock(return_value=None)))
    producer.flush = MagicMock()
    producer.close = MagicMock()
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer for integration testing."""
    consumer = MagicMock()
    consumer.subscribe = MagicMock()
    consumer.poll = MagicMock(return_value={})
    consumer.commit = MagicMock()
    consumer.close = MagicMock()
    return consumer


@pytest.fixture
def mock_graphql_client():
    """Mock GraphQL client for integration testing."""
    client = MagicMock()
    client.execute = MagicMock(return_value={
        "data": {"thermalStatus": {"temperature": 450.0, "pressure": 15.0}}
    })
    return client


@pytest.fixture
def mock_grpc_stub():
    """Mock gRPC stub for integration testing."""
    stub = MagicMock()
    stub.GetThermalData = MagicMock(return_value=MagicMock(
        temperature=450.0,
        pressure=15.0,
        status="OK"
    ))
    stub.SetSetpoint = MagicMock(return_value=MagicMock(success=True))
    return stub


@pytest.fixture
def mock_cmms_client():
    """Mock CMMS client for integration testing."""
    client = MagicMock()
    client.create_work_order = MagicMock(return_value={"work_order_id": "WO-001"})
    client.get_maintenance_schedule = MagicMock(return_value=[
        {"asset_id": "BOILER_001", "scheduled_date": "2025-02-01", "type": "preventive"}
    ])
    client.update_asset_status = MagicMock(return_value=True)
    return client


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_mock_opc_ua_client():
    """Async mock OPC-UA client for async integration testing."""
    client = AsyncMock()
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock(return_value=True)
    client.read_node = AsyncMock(return_value={"value": 450.0, "quality": "Good"})
    client.write_node = AsyncMock(return_value=True)
    return client


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_timer():
    """Provide a context manager for measuring execution time."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time

    return Timer


@pytest.fixture
def memory_tracker():
    """Provide memory usage tracking."""
    import tracemalloc

    class MemoryTracker:
        def __init__(self):
            self.start_size = None
            self.peak_size = None
            self.end_size = None

        def __enter__(self):
            tracemalloc.start()
            self.start_size, _ = tracemalloc.get_traced_memory()
            return self

        def __exit__(self, *args):
            self.end_size, self.peak_size = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        @property
        def memory_used_mb(self) -> float:
            return (self.peak_size - self.start_size) / (1024 * 1024)

    return MemoryTracker


@pytest.fixture
def large_dataset(test_data_generator) -> List[ThermalMeasurement]:
    """Provide large dataset for performance testing."""
    return test_data_generator.generate_thermal_measurements(count=100000)


# =============================================================================
# Safety Testing Fixtures
# =============================================================================

@pytest.fixture
def boundary_conditions():
    """Provide boundary condition test cases."""
    return {
        "temperature": {
            "min": TEMPERATURE_MIN,
            "max": TEMPERATURE_MAX,
            "below_min": TEMPERATURE_MIN - 10,
            "above_max": TEMPERATURE_MAX + 10,
            "at_min": TEMPERATURE_MIN,
            "at_max": TEMPERATURE_MAX
        },
        "pressure": {
            "min": PRESSURE_MIN,
            "max": PRESSURE_MAX,
            "below_min": PRESSURE_MIN - 1,
            "above_max": PRESSURE_MAX + 10,
            "at_min": PRESSURE_MIN,
            "at_max": PRESSURE_MAX
        },
        "flow_rate": {
            "min": FLOW_RATE_MIN,
            "max": FLOW_RATE_MAX,
            "below_min": FLOW_RATE_MIN - 10,
            "above_max": FLOW_RATE_MAX + 100,
            "at_min": FLOW_RATE_MIN,
            "at_max": FLOW_RATE_MAX
        }
    }


@pytest.fixture
def sis_permissive_states():
    """Provide SIS permissive state test cases."""
    return [
        {"permissive": "boiler_ready", "conditions": ["fuel_available", "air_flow_ok", "flame_detected"]},
        {"permissive": "steam_pressure_ok", "conditions": ["pressure_below_max", "safety_valve_ok"]},
        {"permissive": "water_level_ok", "conditions": ["level_above_min", "level_below_max"]},
        {"permissive": "combustion_air_ok", "conditions": ["fan_running", "damper_open"]},
    ]


@pytest.fixture
def alarm_storm_scenario():
    """Provide alarm storm test scenario."""
    base_time = datetime.now()
    return [
        SafetyEvent(
            timestamp=base_time + timedelta(seconds=i * 0.5),
            event_type='alarm',
            severity='high' if i % 3 == 0 else 'medium',
            source=f"SENSOR_{(i % 5) + 1:03d}",
            message=f"Alarm {i+1}: Parameter out of range"
        )
        for i in range(100)  # 100 alarms in 50 seconds
    ]


# =============================================================================
# Determinism Testing Fixtures
# =============================================================================

@pytest.fixture
def determinism_inputs():
    """Provide identical inputs for determinism testing."""
    return {
        "thermal_input": {
            "temperatures": [450.0, 455.0, 448.0, 452.0, 450.5],
            "pressures": [15.0, 15.1, 14.9, 15.0, 15.05],
            "flow_rates": [500.0, 505.0, 498.0, 502.0, 500.5]
        },
        "optimization_input": {
            "demands": [1000.0, 1200.0, 800.0],
            "boiler_capacities": [500.0, 600.0, 400.0],
            "costs": [10.0, 12.0, 8.0]
        }
    }


@pytest.fixture
def provenance_calculator():
    """Provide provenance hash calculator."""
    class ProvenanceCalculator:
        @staticmethod
        def calculate_hash(data: Any) -> str:
            """Calculate SHA-256 hash for provenance tracking."""
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            return hashlib.sha256(data_str.encode()).hexdigest()

        @staticmethod
        def verify_hash(data: Any, expected_hash: str) -> bool:
            """Verify data matches expected provenance hash."""
            actual_hash = ProvenanceCalculator.calculate_hash(data)
            return actual_hash == expected_hash

    return ProvenanceCalculator


# =============================================================================
# Acceptance Testing Fixtures
# =============================================================================

@pytest.fixture
def sla_requirements():
    """Provide SLA requirements for acceptance testing."""
    return {
        "heat_delivery_tolerance": 0.02,  # 2% tolerance
        "data_availability": 0.999,  # 99.9% availability
        "optimization_cycle_time": 5.0,  # seconds
        "api_response_time": 0.200,  # seconds
        "calculation_reproducibility": 1.0,  # 100% reproducible
        "alarm_response_time": 1.0,  # seconds
        "report_generation_time": 10.0  # seconds
    }


@pytest.fixture
def operator_workflow_scenarios():
    """Provide operator workflow test scenarios."""
    return [
        {
            "name": "Start-up sequence",
            "steps": [
                "Check safety interlocks",
                "Enable fuel supply",
                "Start combustion air fan",
                "Ignite pilot flame",
                "Ramp up main burner",
                "Monitor temperature rise",
                "Transfer to automatic control"
            ]
        },
        {
            "name": "Load change",
            "steps": [
                "Receive demand change",
                "Calculate new setpoints",
                "Adjust fuel rate",
                "Monitor process response",
                "Verify stable operation"
            ]
        },
        {
            "name": "Emergency shutdown",
            "steps": [
                "Detect emergency condition",
                "Trip fuel supply",
                "Close isolation valves",
                "Start purge sequence",
                "Log all events",
                "Notify operators"
            ]
        }
    ]


# =============================================================================
# Utility Functions
# =============================================================================

def assert_within_tolerance(actual: float, expected: float, tolerance: float = 0.001):
    """Assert that actual value is within tolerance of expected."""
    assert abs(actual - expected) <= abs(expected * tolerance), \
        f"Expected {expected} +/- {tolerance*100}%, got {actual}"


def assert_provenance_valid(data: Dict, hash_value: str):
    """Assert that provenance hash is valid for given data."""
    calculated = hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert calculated == hash_value, f"Provenance mismatch: {calculated} != {hash_value}"


def assert_within_sla(metric_name: str, actual: float, target: float, unit: str = ""):
    """Assert that metric meets SLA target."""
    assert actual <= target, \
        f"SLA violation: {metric_name} = {actual}{unit}, target = {target}{unit}"


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "simulation: Simulation tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "safety: Safety tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "determinism: Determinism tests")
    config.addinivalue_line("markers", "acceptance: Acceptance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "requires_hardware: Tests requiring real hardware")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on path."""
    for item in items:
        if "test_unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_simulation" in str(item.fspath):
            item.add_marker(pytest.mark.simulation)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_safety" in str(item.fspath):
            item.add_marker(pytest.mark.safety)
        elif "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "test_determinism" in str(item.fspath):
            item.add_marker(pytest.mark.determinism)
        elif "test_acceptance" in str(item.fspath):
            item.add_marker(pytest.mark.acceptance)
