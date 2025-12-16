"""
GL-022 SuperheaterControlAgent - Pytest Configuration and Fixtures

This module provides:
- Shared pytest fixtures for all test modules
- Mock OPC-UA and Kafka connections
- Sample input data generators
- Test configuration helpers
- Performance benchmark fixtures

Usage:
    All fixtures are automatically available to test modules.
    Use pytest markers for selective test execution:
        pytest -m unit          # Run unit tests only
        pytest -m integration   # Run integration tests only
        pytest -m "not slow"    # Skip slow tests
"""

import hashlib
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add the agent module path for imports
AGENT_BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.parent
BACKEND_AGENT_PATH = AGENT_BASE_PATH / "GL-Agent-Factory" / "backend" / "agents"
sys.path.insert(0, str(AGENT_BASE_PATH))
sys.path.insert(0, str(BACKEND_AGENT_PATH))

# Try to import the actual agent components
try:
    from gl_022_superheater_control.agent import SuperheaterControlAgent
    from gl_022_superheater_control.models import (
        SuperheaterInput,
        SuperheaterOutput,
        SprayControlAction,
        ControlParameters,
    )
    from gl_022_superheater_control.formulas import (
        calculate_saturation_temperature,
        calculate_superheat,
        calculate_steam_enthalpy,
        calculate_spray_water_flow,
        calculate_valve_position,
        calculate_pid_parameters,
        calculate_spray_energy_loss,
        calculate_thermal_efficiency_impact,
        generate_calculation_hash,
        bar_to_psi,
        celsius_to_fahrenheit,
        kg_s_to_lb_hr,
    )
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "compliance: Regulatory compliance tests")
    config.addinivalue_line("markers", "safety: Safety-critical tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "asyncio: Async tests")


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on naming conventions."""
    for item in items:
        # Auto-mark tests based on module name
        if "test_safety" in item.nodeid:
            item.add_marker(pytest.mark.safety)
        if "test_calculators" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        if "test_controller" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# =============================================================================
# AGENT CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> Dict[str, Any]:
    """
    Default agent configuration for testing.

    Returns a configuration dict with standard PID parameters
    and safety thresholds.
    """
    return {
        "time_constant": 60.0,      # Process time constant (seconds)
        "dead_time": 10.0,          # Process dead time (seconds)
        "response_time": 120.0,     # Desired response time (seconds)
        "max_spray_rate": 10.0,     # Maximum spray flow rate change
        "safety_margin_c": 25.0,    # Safety margin for tube temps
    }


@pytest.fixture
def aggressive_pid_config() -> Dict[str, Any]:
    """Aggressive PID tuning configuration for fast response."""
    return {
        "time_constant": 30.0,
        "dead_time": 5.0,
        "response_time": 60.0,
    }


@pytest.fixture
def conservative_pid_config() -> Dict[str, Any]:
    """Conservative PID tuning configuration for stability."""
    return {
        "time_constant": 90.0,
        "dead_time": 15.0,
        "response_time": 180.0,
    }


@pytest.fixture
def agent(default_config) -> "SuperheaterControlAgent":
    """
    Create SuperheaterControlAgent instance with default config.

    Returns:
        Configured SuperheaterControlAgent ready for testing.
    """
    if not AGENT_AVAILABLE:
        pytest.skip("Agent module not available")
    return SuperheaterControlAgent(config=default_config)


@pytest.fixture
def agent_aggressive(aggressive_pid_config) -> "SuperheaterControlAgent":
    """Agent with aggressive PID tuning."""
    if not AGENT_AVAILABLE:
        pytest.skip("Agent module not available")
    return SuperheaterControlAgent(config=aggressive_pid_config)


@pytest.fixture
def agent_no_config() -> "SuperheaterControlAgent":
    """Agent with no configuration (uses defaults)."""
    if not AGENT_AVAILABLE:
        pytest.skip("Agent module not available")
    return SuperheaterControlAgent()


# =============================================================================
# INPUT DATA FIXTURES
# =============================================================================

@pytest.fixture
def valid_input_data() -> Dict[str, Any]:
    """
    Valid input data for standard operation scenario.

    Represents a typical superheater operating at:
    - 40 bar steam pressure
    - 450C outlet temperature (above 400C target)
    - Moderate spray flow
    """
    return {
        "inlet_steam_temp_c": 480.0,
        "outlet_steam_temp_c": 450.0,
        "target_steam_temp_c": 400.0,
        "steam_pressure_bar": 40.0,
        "steam_flow_kg_s": 20.0,
        "spray_water_temp_c": 100.0,
        "current_spray_flow_kg_s": 0.5,
        "max_spray_flow_kg_s": 5.0,
        "spray_valve_position_pct": 10.0,
        "burner_load_pct": 80.0,
        "flue_gas_temp_c": 350.0,
        "excess_air_pct": 15.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": 520.0,
        "equipment_id": "SH-001",
    }


@pytest.fixture
def valid_input(valid_input_data) -> "SuperheaterInput":
    """Create validated SuperheaterInput from test data."""
    if not AGENT_AVAILABLE:
        pytest.skip("Agent module not available")
    return SuperheaterInput(**valid_input_data)


@pytest.fixture
def input_at_target() -> Dict[str, Any]:
    """Input where temperature is at target (within tolerance)."""
    return {
        "inlet_steam_temp_c": 420.0,
        "outlet_steam_temp_c": 402.0,  # Within 5C tolerance of 400C
        "target_steam_temp_c": 400.0,
        "steam_pressure_bar": 40.0,
        "steam_flow_kg_s": 20.0,
        "spray_water_temp_c": 100.0,
        "current_spray_flow_kg_s": 0.5,
        "max_spray_flow_kg_s": 5.0,
        "spray_valve_position_pct": 10.0,
        "burner_load_pct": 80.0,
        "flue_gas_temp_c": 350.0,
        "excess_air_pct": 15.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": 520.0,
        "equipment_id": "SH-001",
    }


@pytest.fixture
def input_below_target() -> Dict[str, Any]:
    """Input where temperature is below target (needs less spray)."""
    return {
        "inlet_steam_temp_c": 380.0,
        "outlet_steam_temp_c": 390.0,  # Below 400C target
        "target_steam_temp_c": 400.0,
        "steam_pressure_bar": 40.0,
        "steam_flow_kg_s": 20.0,
        "spray_water_temp_c": 100.0,
        "current_spray_flow_kg_s": 2.0,  # High spray flow
        "max_spray_flow_kg_s": 5.0,
        "spray_valve_position_pct": 40.0,
        "burner_load_pct": 70.0,
        "flue_gas_temp_c": 320.0,
        "excess_air_pct": 15.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": 480.0,
        "equipment_id": "SH-001",
    }


@pytest.fixture
def input_high_pressure() -> Dict[str, Any]:
    """High pressure operation (100 bar)."""
    return {
        "inlet_steam_temp_c": 550.0,
        "outlet_steam_temp_c": 520.0,
        "target_steam_temp_c": 480.0,
        "steam_pressure_bar": 100.0,
        "steam_flow_kg_s": 50.0,
        "spray_water_temp_c": 150.0,
        "current_spray_flow_kg_s": 1.0,
        "max_spray_flow_kg_s": 10.0,
        "spray_valve_position_pct": 10.0,
        "burner_load_pct": 90.0,
        "flue_gas_temp_c": 400.0,
        "excess_air_pct": 12.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 30.0,
        "max_tube_metal_temp_c": 650.0,
        "current_tube_metal_temp_c": 580.0,
        "equipment_id": "SH-HP-001",
    }


@pytest.fixture
def input_low_pressure() -> Dict[str, Any]:
    """Low pressure operation (5 bar)."""
    return {
        "inlet_steam_temp_c": 200.0,
        "outlet_steam_temp_c": 180.0,
        "target_steam_temp_c": 160.0,
        "steam_pressure_bar": 5.0,
        "steam_flow_kg_s": 5.0,
        "spray_water_temp_c": 50.0,
        "current_spray_flow_kg_s": 0.1,
        "max_spray_flow_kg_s": 1.0,
        "spray_valve_position_pct": 10.0,
        "burner_load_pct": 50.0,
        "flue_gas_temp_c": 200.0,
        "excess_air_pct": 20.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 15.0,
        "max_tube_metal_temp_c": 500.0,
        "current_tube_metal_temp_c": 350.0,
        "equipment_id": "SH-LP-001",
    }


# =============================================================================
# SAFETY SCENARIO FIXTURES
# =============================================================================

@pytest.fixture
def input_critical_tube_temp() -> Dict[str, Any]:
    """Critical tube metal temperature scenario."""
    return {
        "inlet_steam_temp_c": 550.0,
        "outlet_steam_temp_c": 520.0,
        "target_steam_temp_c": 480.0,
        "steam_pressure_bar": 60.0,
        "steam_flow_kg_s": 30.0,
        "spray_water_temp_c": 120.0,
        "current_spray_flow_kg_s": 0.5,
        "max_spray_flow_kg_s": 8.0,
        "spray_valve_position_pct": 6.25,
        "burner_load_pct": 95.0,
        "flue_gas_temp_c": 450.0,
        "excess_air_pct": 10.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 25.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": 585.0,  # Only 15C margin (critical)
        "equipment_id": "SH-002",
    }


@pytest.fixture
def input_warning_tube_temp() -> Dict[str, Any]:
    """Warning level tube metal temperature."""
    return {
        "inlet_steam_temp_c": 520.0,
        "outlet_steam_temp_c": 490.0,
        "target_steam_temp_c": 460.0,
        "steam_pressure_bar": 50.0,
        "steam_flow_kg_s": 25.0,
        "spray_water_temp_c": 110.0,
        "current_spray_flow_kg_s": 0.8,
        "max_spray_flow_kg_s": 6.0,
        "spray_valve_position_pct": 13.3,
        "burner_load_pct": 85.0,
        "flue_gas_temp_c": 380.0,
        "excess_air_pct": 12.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": 565.0,  # 35C margin (warning)
        "equipment_id": "SH-003",
    }


@pytest.fixture
def input_low_superheat() -> Dict[str, Any]:
    """Low superheat condition (near saturation)."""
    return {
        "inlet_steam_temp_c": 280.0,
        "outlet_steam_temp_c": 265.0,  # Close to saturation at 40 bar
        "target_steam_temp_c": 260.0,
        "steam_pressure_bar": 40.0,
        "steam_flow_kg_s": 20.0,
        "spray_water_temp_c": 100.0,
        "current_spray_flow_kg_s": 2.0,
        "max_spray_flow_kg_s": 5.0,
        "spray_valve_position_pct": 40.0,
        "burner_load_pct": 60.0,
        "flue_gas_temp_c": 280.0,
        "excess_air_pct": 18.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": 400.0,
        "equipment_id": "SH-004",
    }


@pytest.fixture
def input_spray_capacity_exceeded() -> Dict[str, Any]:
    """Scenario where required spray exceeds capacity."""
    return {
        "inlet_steam_temp_c": 600.0,
        "outlet_steam_temp_c": 550.0,  # Very high - needs lots of spray
        "target_steam_temp_c": 400.0,  # 150C reduction needed
        "steam_pressure_bar": 60.0,
        "steam_flow_kg_s": 40.0,
        "spray_water_temp_c": 100.0,
        "current_spray_flow_kg_s": 3.0,
        "max_spray_flow_kg_s": 5.0,  # Limited capacity
        "spray_valve_position_pct": 60.0,
        "burner_load_pct": 100.0,
        "flue_gas_temp_c": 500.0,
        "excess_air_pct": 8.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 650.0,
        "current_tube_metal_temp_c": 600.0,
        "equipment_id": "SH-005",
    }


@pytest.fixture
def input_no_tube_temp_sensor() -> Dict[str, Any]:
    """Scenario without tube metal temperature sensor."""
    return {
        "inlet_steam_temp_c": 480.0,
        "outlet_steam_temp_c": 450.0,
        "target_steam_temp_c": 400.0,
        "steam_pressure_bar": 40.0,
        "steam_flow_kg_s": 20.0,
        "spray_water_temp_c": 100.0,
        "current_spray_flow_kg_s": 0.5,
        "max_spray_flow_kg_s": 5.0,
        "spray_valve_position_pct": 10.0,
        "burner_load_pct": 80.0,
        "flue_gas_temp_c": 350.0,
        "excess_air_pct": 15.0,
        "process_temp_tolerance_c": 5.0,
        "min_superheat_c": 20.0,
        "max_tube_metal_temp_c": 600.0,
        "current_tube_metal_temp_c": None,  # No sensor reading
        "equipment_id": "SH-006",
    }


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================

@pytest.fixture
def boundary_pressure_values() -> List[float]:
    """Boundary values for pressure (bar)."""
    return [1.0, 1.01, 10.0, 100.0, 199.9, 200.0]


@pytest.fixture
def boundary_temperature_values() -> List[float]:
    """Boundary values for temperature (C)."""
    return [100.0, 100.1, 250.0, 450.0, 699.9, 700.0]


@pytest.fixture
def invalid_pressure_values() -> List[float]:
    """Invalid pressure values for error testing."""
    return [-1.0, 0.0, 0.5, 201.0, 1000.0]


@pytest.fixture
def invalid_temperature_values() -> List[float]:
    """Invalid temperature values for error testing."""
    return [-50.0, 50.0, 99.9, 701.0, 1000.0]


# =============================================================================
# MOCK FIXTURES FOR INTEGRATION TESTING
# =============================================================================

@pytest.fixture
def mock_opcua_client() -> MagicMock:
    """
    Mock OPC-UA client for integration testing.

    Simulates reading sensor values and writing control outputs.
    """
    client = MagicMock()
    client.connect = MagicMock(return_value=True)
    client.disconnect = MagicMock(return_value=True)
    client.get_node = MagicMock()

    # Simulate reading values
    client.read_values = MagicMock(return_value={
        "ns=2;s=SH001.OutletTemp": 450.0,
        "ns=2;s=SH001.SteamPressure": 40.0,
        "ns=2;s=SH001.SprayFlow": 0.5,
        "ns=2;s=SH001.TubeMetal": 520.0,
    })

    # Simulate writing values
    client.write_value = MagicMock(return_value=True)

    return client


@pytest.fixture
def mock_kafka_producer() -> MagicMock:
    """
    Mock Kafka producer for event streaming tests.

    Simulates publishing control events to Kafka topics.
    """
    producer = MagicMock()
    producer.send = MagicMock(return_value=MagicMock(
        get=MagicMock(return_value=MagicMock(
            topic="superheater-control",
            partition=0,
            offset=12345
        ))
    ))
    producer.flush = MagicMock()
    producer.close = MagicMock()

    return producer


@pytest.fixture
def mock_kafka_consumer() -> MagicMock:
    """Mock Kafka consumer for event consumption tests."""
    consumer = MagicMock()
    consumer.subscribe = MagicMock()
    consumer.poll = MagicMock(return_value={})
    consumer.close = MagicMock()

    return consumer


@pytest.fixture
def mock_database_session() -> MagicMock:
    """Mock database session for persistence tests."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()
    session.query = MagicMock()

    return session


@pytest.fixture
def mock_historian() -> MagicMock:
    """Mock process historian for historical data tests."""
    historian = MagicMock()

    # Return mock historical data
    historian.read_tag_values = MagicMock(return_value=[
        {"timestamp": "2025-01-01T00:00:00Z", "value": 448.0},
        {"timestamp": "2025-01-01T00:01:00Z", "value": 449.0},
        {"timestamp": "2025-01-01T00:02:00Z", "value": 450.0},
        {"timestamp": "2025-01-01T00:03:00Z", "value": 451.0},
        {"timestamp": "2025-01-01T00:04:00Z", "value": 450.5},
    ])

    historian.write_tag_value = MagicMock(return_value=True)

    return historian


# =============================================================================
# PROVENANCE AND DETERMINISM FIXTURES
# =============================================================================

@pytest.fixture
def known_hash_inputs() -> Dict[str, Any]:
    """Input data with known provenance hash for determinism testing."""
    return {
        "steam_temp": 450.0,
        "target_temp": 400.0,
        "pressure": 40.0,
        "steam_flow": 20.0,
        "spray_water_temp": 100.0,
    }


@pytest.fixture
def known_hash_outputs() -> Dict[str, Any]:
    """Output data with known provenance hash for determinism testing."""
    return {
        "spray_flow": 1.5,
        "valve_position": 30.0,
        "t_sat": 250.35,
        "superheat": 199.65,
    }


@pytest.fixture
def expected_provenance_hash(known_hash_inputs, known_hash_outputs) -> str:
    """Pre-computed provenance hash for determinism verification."""
    data = {
        "inputs": known_hash_inputs,
        "outputs": known_hash_outputs,
        "formula_version": "1.0.0",
        "standard": "IAPWS-IF97"
    }
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# IAPWS-IF97 REFERENCE DATA FIXTURES
# =============================================================================

@pytest.fixture
def iapws_if97_reference_points() -> List[Dict[str, float]]:
    """
    Reference data points from IAPWS-IF97 steam tables.

    These are used to validate thermodynamic calculations against
    known standard values.
    """
    return [
        # (pressure_bar, saturation_temp_c) - approximate values
        {"pressure_bar": 1.0, "t_sat_c": 99.97, "tolerance": 1.0},
        {"pressure_bar": 5.0, "t_sat_c": 151.84, "tolerance": 2.0},
        {"pressure_bar": 10.0, "t_sat_c": 179.88, "tolerance": 2.0},
        {"pressure_bar": 20.0, "t_sat_c": 212.38, "tolerance": 3.0},
        {"pressure_bar": 40.0, "t_sat_c": 250.35, "tolerance": 3.0},
        {"pressure_bar": 60.0, "t_sat_c": 275.58, "tolerance": 4.0},
        {"pressure_bar": 100.0, "t_sat_c": 311.0, "tolerance": 5.0},
        {"pressure_bar": 150.0, "t_sat_c": 342.16, "tolerance": 5.0},
    ]


@pytest.fixture
def steam_enthalpy_reference() -> List[Dict[str, float]]:
    """Reference enthalpy values for validation."""
    return [
        # Approximate values from steam tables
        {"temp_c": 200, "pressure_bar": 10, "enthalpy_kj_kg": 2829, "tolerance": 50},
        {"temp_c": 300, "pressure_bar": 20, "enthalpy_kj_kg": 3025, "tolerance": 50},
        {"temp_c": 400, "pressure_bar": 40, "enthalpy_kj_kg": 3214, "tolerance": 50},
        {"temp_c": 500, "pressure_bar": 60, "enthalpy_kj_kg": 3423, "tolerance": 50},
    ]


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def large_batch_inputs(valid_input_data) -> List[Dict[str, Any]]:
    """Generate large batch of inputs for performance testing."""
    import random
    random.seed(42)  # Reproducibility

    inputs = []
    for i in range(1000):
        data = valid_input_data.copy()
        # Add some variation
        data["outlet_steam_temp_c"] = 400 + random.uniform(-50, 100)
        data["steam_pressure_bar"] = 20 + random.uniform(0, 80)
        data["steam_flow_kg_s"] = 10 + random.uniform(0, 40)
        data["equipment_id"] = f"SH-{i:04d}"
        inputs.append(data)

    return inputs


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for benchmarking."""
    return {
        "single_process_ms": 5.0,        # Single process < 5ms
        "batch_1000_s": 2.0,              # 1000 records < 2s
        "throughput_min": 500,            # > 500 records/second
        "memory_increase_mb": 100,        # < 100MB memory increase
    }


# =============================================================================
# HELPER FUNCTIONS (available to tests)
# =============================================================================

def generate_random_input(seed: int = None) -> Dict[str, Any]:
    """Generate random valid input data for fuzzing tests."""
    import random
    if seed is not None:
        random.seed(seed)

    return {
        "inlet_steam_temp_c": random.uniform(200, 600),
        "outlet_steam_temp_c": random.uniform(150, 550),
        "target_steam_temp_c": random.uniform(200, 500),
        "steam_pressure_bar": random.uniform(5, 150),
        "steam_flow_kg_s": random.uniform(1, 100),
        "spray_water_temp_c": random.uniform(20, 150),
        "current_spray_flow_kg_s": random.uniform(0, 5),
        "max_spray_flow_kg_s": random.uniform(5, 20),
        "spray_valve_position_pct": random.uniform(0, 100),
        "burner_load_pct": random.uniform(20, 100),
        "flue_gas_temp_c": random.uniform(150, 600),
        "excess_air_pct": random.uniform(5, 30),
        "process_temp_tolerance_c": random.uniform(2, 10),
        "min_superheat_c": random.uniform(10, 50),
        "max_tube_metal_temp_c": random.uniform(500, 700),
        "current_tube_metal_temp_c": random.uniform(400, 600),
        "equipment_id": f"SH-TEST-{random.randint(1, 9999):04d}",
    }


def assert_output_valid(output: Dict[str, Any]) -> None:
    """Assert that output contains all required fields with valid values."""
    # Required fields
    assert "spray_control" in output
    assert "control_parameters" in output
    assert "current_superheat_c" in output
    assert "saturation_temp_c" in output
    assert "safety_status" in output
    assert "calculation_hash" in output

    # Validate calculation hash format (SHA-256 = 64 hex chars)
    assert len(output["calculation_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in output["calculation_hash"])

    # Validate safety status
    assert output["safety_status"] in ["SAFE", "WARNING", "CRITICAL"]

    # Validate spray control
    spray = output["spray_control"]
    assert spray["target_spray_flow_kg_s"] >= 0
    assert 0 <= spray["valve_position_pct"] <= 100
    assert spray["action_type"] in ["INCREASE", "DECREASE", "MAINTAIN"]


def assert_deterministic(agent, input_data: Dict[str, Any], iterations: int = 5) -> None:
    """Assert that agent produces identical output for identical input."""
    results = [agent.run(input_data) for _ in range(iterations)]

    first_hash = results[0]["calculation_hash"]
    for result in results[1:]:
        assert result["calculation_hash"] == first_hash, "Calculation hash not deterministic"
        assert result["spray_control"] == results[0]["spray_control"], "Spray control not deterministic"


# Make helper functions available as fixtures
@pytest.fixture
def random_input_generator():
    """Fixture that returns the random input generator function."""
    return generate_random_input


@pytest.fixture
def output_validator():
    """Fixture that returns the output validation function."""
    return assert_output_valid


@pytest.fixture
def determinism_checker():
    """Fixture that returns the determinism check function."""
    return assert_deterministic
