"""
Unit Test Fixtures for GL-001 ThermalCommand.

Additional fixtures specific to unit tests.
These supplement the global fixtures in tests/conftest.py.

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock


# =============================================================================
# MILP Optimizer Fixtures
# =============================================================================

@pytest.fixture
def mock_milp_result():
    """Provide a mock MILP optimization result."""
    return {
        "status": "optimal",
        "objective_value": 5000.0,
        "allocations": [
            {"equipment_id": "BOILER-001", "load": 40.0, "is_running": True},
            {"equipment_id": "BOILER-002", "load": 30.0, "is_running": True},
        ],
        "total_cost": 450.0,
        "total_emissions": 150.0,
        "solve_time_ms": 250.0,
    }


@pytest.fixture
def equipment_factory():
    """Provide factory for creating test equipment."""
    def create_equipment(
        equipment_id: str = "TEST-001",
        equipment_type: str = "boiler",
        capacity: float = 50.0,
        efficiency: float = 0.85,
        fuel_cost: float = 5.0,
        status: str = "available"
    ):
        return {
            "equipment_id": equipment_id,
            "equipment_type": equipment_type,
            "max_capacity_mmbtu_hr": capacity,
            "min_capacity_mmbtu_hr": capacity / 4,
            "rated_efficiency": efficiency,
            "fuel_cost_per_mmbtu": fuel_cost,
            "status": status,
            "current_load_mmbtu_hr": 0.0,
            "co2_kg_per_mmbtu_fuel": 53.06 if equipment_type == "boiler" else 31.84,
        }

    return create_equipment


# =============================================================================
# PID Controller Fixtures
# =============================================================================

@pytest.fixture
def mock_pid_state():
    """Provide a mock PID controller state."""
    return {
        "integral": 0.0,
        "last_error": 0.0,
        "last_pv": 450.0,
        "last_output": 50.0,
        "derivative_filter": 0.0,
    }


@pytest.fixture
def pid_tuning_sets():
    """Provide various PID tuning parameter sets."""
    return {
        "conservative": {"kp": 1.0, "ki": 0.05, "kd": 0.2},
        "moderate": {"kp": 2.0, "ki": 0.1, "kd": 0.5},
        "aggressive": {"kp": 5.0, "ki": 0.5, "kd": 1.0},
        "pi_only": {"kp": 2.0, "ki": 0.2, "kd": 0.0},
        "p_only": {"kp": 3.0, "ki": 0.0, "kd": 0.0},
    }


# =============================================================================
# Safety Boundary Fixtures
# =============================================================================

@pytest.fixture
def mock_safety_violation():
    """Provide a mock safety boundary violation."""
    return {
        "violation_id": "VIO-001",
        "timestamp": datetime.now(timezone.utc),
        "policy_id": "TEMP_MAX_001",
        "tag_id": "TIC-101",
        "requested_value": 250.0,
        "boundary_value": 200.0,
        "violation_type": "over_max",
        "severity": "critical",
        "blocked": True,
    }


@pytest.fixture
def boundary_test_cases():
    """Provide boundary test cases for temperature limits."""
    return {
        "within_bounds": [
            {"tag": "TIC-101", "value": 100.0, "expected": "allow"},
            {"tag": "TIC-101", "value": 0.0, "expected": "allow"},
            {"tag": "TIC-101", "value": 199.0, "expected": "allow"},
        ],
        "at_boundary": [
            {"tag": "TIC-101", "value": -40.0, "expected": "allow"},
            {"tag": "TIC-101", "value": 200.0, "expected": "allow"},
        ],
        "outside_bounds": [
            {"tag": "TIC-101", "value": -50.0, "expected": "block_or_clamp"},
            {"tag": "TIC-101", "value": 250.0, "expected": "block_or_clamp"},
        ],
    }


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def extreme_temperature_values():
    """Provide extreme temperature values for edge case testing."""
    return {
        "cryogenic": -200.0,
        "freezing": 0.0,
        "ambient": 25.0,
        "process_low": 100.0,
        "process_normal": 450.0,
        "process_high": 800.0,
        "extreme_high": 1500.0,
        "limit_high": 1200.0,
    }


@pytest.fixture
def extreme_pressure_values():
    """Provide extreme pressure values for edge case testing."""
    return {
        "vacuum": -1.0,
        "atmospheric": 1.0,
        "low_pressure": 5.0,
        "normal_pressure": 15.0,
        "high_pressure": 50.0,
        "extreme_pressure": 150.0,
        "limit_pressure": 100.0,
    }


@pytest.fixture
def sensor_failure_scenario():
    """Provide sensor failure test scenario."""
    return {
        "primary_sensor": {
            "tag_id": "TI-101",
            "value": float('nan'),
            "quality": "Bad",
            "timestamp": datetime.now(timezone.utc),
        },
        "backup_sensor": {
            "tag_id": "TI-101-B",
            "value": 445.0,
            "quality": "Good",
            "timestamp": datetime.now(timezone.utc),
        },
        "expected_behavior": "use_backup",
    }


# =============================================================================
# SIL-3 Testing Fixtures
# =============================================================================

@pytest.fixture
def sil3_voting_scenario():
    """Provide SIL-3 2oo3 voting scenario."""
    return {
        "sensors": [
            {"id": "TI-101-A", "value": 445.0, "quality": "Good"},
            {"id": "TI-101-B", "value": 450.0, "quality": "Good"},
            {"id": "TI-101-C", "value": 448.0, "quality": "Good"},
        ],
        "voting_logic": "2oo3",
        "deviation_threshold": 5.0,
        "expected_result": 447.67,
    }


@pytest.fixture
def sil3_failure_scenarios():
    """Provide SIL-3 failure scenarios."""
    return [
        {
            "name": "single_sensor_failure",
            "sensors": [
                {"id": "TI-101-A", "value": 445.0, "quality": "Good"},
                {"id": "TI-101-B", "value": float('nan'), "quality": "Bad"},
                {"id": "TI-101-C", "value": 448.0, "quality": "Good"},
            ],
            "expected_result": "use_2_good_sensors",
        },
        {
            "name": "two_sensor_failure",
            "sensors": [
                {"id": "TI-101-A", "value": 445.0, "quality": "Good"},
                {"id": "TI-101-B", "value": float('nan'), "quality": "Bad"},
                {"id": "TI-101-C", "value": float('nan'), "quality": "Bad"},
            ],
            "expected_result": "safe_state",
        },
        {
            "name": "sensor_deviation",
            "sensors": [
                {"id": "TI-101-A", "value": 445.0, "quality": "Good"},
                {"id": "TI-101-B", "value": 500.0, "quality": "Good"},  # Deviation
                {"id": "TI-101-C", "value": 448.0, "quality": "Good"},
            ],
            "expected_result": "exclude_deviant",
        },
    ]


# =============================================================================
# Provenance Tracking Fixtures
# =============================================================================

@pytest.fixture
def provenance_test_data():
    """Provide test data for provenance hash verification."""
    return {
        "input": {
            "temperatures": [450.0, 455.0, 448.0],
            "pressures": [15.0, 15.1, 14.9],
            "timestamp": "2025-01-01T00:00:00Z",
        },
        "output": {
            "setpoints": {"TIC-101": 85.0, "TIC-102": 80.0},
            "allocations": {"BOILER-001": 40.0, "BOILER-002": 30.0},
        },
        "expected_hash_length": 64,  # SHA-256 hex
    }
