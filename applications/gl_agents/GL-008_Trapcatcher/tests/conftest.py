"""
Pytest Configuration and Shared Fixtures for GL-008 Trapcatcher Test Suite.
Target Coverage: 85%+
Author: GL-TestEngineer
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
from enum import Enum, auto

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "golden: marks tests as golden master tests")


class TrapType(Enum):
    THERMOSTATIC = auto()
    THERMODYNAMIC = auto()
    MECHANICAL_FLOAT = auto()
    MECHANICAL_BUCKET = auto()
    ORIFICE = auto()


class TrapFailureMode(Enum):
    NORMAL = auto()
    FAILED_OPEN = auto()
    FAILED_CLOSED = auto()
    BLOW_THROUGH = auto()
    PARTIAL_BLOCKAGE = auto()
    INTERNAL_EROSION = auto()
    UNKNOWN = auto()


class MaintenancePriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    NONE = auto()


class DiagnosticMethod(Enum):
    ACOUSTIC = auto()
    TEMPERATURE = auto()
    VISUAL = auto()
    COMBINED = auto()


@dataclass
class MockTrapData:
    trap_id: str
    trap_type: TrapType
    location: str
    inlet_pressure_mpa: float
    outlet_pressure_mpa: float
    inlet_temperature_k: float
    outlet_temperature_k: float
    ultrasonic_db: Optional[float] = None
    acoustic_frequency_hz: Optional[float] = None
    cycle_rate_per_min: Optional[float] = None
    orifice_diameter_mm: float = 10.0
    operating_hours: float = 0.0


@dataclass
class MockSensorReading:
    tag: str
    value: float
    unit: str
    quality_code: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uncertainty_percent: float = 2.0


STEAM_COST_PER_TON = 30.0
OPERATING_HOURS_PER_YEAR = 8000
ULTRASONIC_NORMAL_MAX_DB = 70
ULTRASONIC_WARNING_DB = 85
ULTRASONIC_FAILED_DB = 95


@pytest.fixture
def healthy_trap():
    return MockTrapData("TRAP-001", TrapType.THERMODYNAMIC, "Building A", 1.0, 0.1, 453.0, 440.0, 65, 2500.0, 10, 10.0, 5000.0)


@pytest.fixture
def failed_open_trap():
    return MockTrapData("TRAP-002", TrapType.THERMODYNAMIC, "Building B", 1.0, 0.1, 453.0, 451.0, 98, 8000.0, 0, 10.0, 12000.0)


@pytest.fixture
def failed_closed_trap():
    return MockTrapData("TRAP-003", TrapType.MECHANICAL_FLOAT, "Building C", 1.0, 0.1, 453.0, 400.0, 40, 500.0, 0, 15.0, 8000.0)


@pytest.fixture
def trap_fleet():
    return [
        MockTrapData("TRAP-H1", TrapType.THERMODYNAMIC, "Area 1", 1.0, 0.1, 453, 440, 65, 2500, 10, 10, 5000),
        MockTrapData("TRAP-H2", TrapType.MECHANICAL_FLOAT, "Area 1", 1.0, 0.1, 453, 438, 68, 2800, 8, 12, 3000),
        MockTrapData("TRAP-F1", TrapType.THERMODYNAMIC, "Area 2", 1.0, 0.1, 453, 451, 96, 8000, 0, 10, 12000),
        MockTrapData("TRAP-F2", TrapType.MECHANICAL_BUCKET, "Area 3", 1.0, 0.1, 453, 395, 45, 600, 0, 15, 9000),
    ]


@pytest.fixture
def mock_sensor_readings():
    base_time = datetime.now(timezone.utc)
    return {"trap.ultrasonic": MockSensorReading("trap.ultrasonic", 65.0, "dB", 0, base_time)}


@pytest.fixture
def mock_historian_connector():
    mock = AsyncMock()
    mock.connect.return_value = True
    return mock


@pytest.fixture
def agent_config():
    return {"agent_id": "GL-008", "agent_name": "Trapcatcher", "version": "1.0.0"}


@pytest.fixture
def napier_equation_params():
    return {"napier_constant": 0.0413, "discharge_coefficient": 0.62}


class AssertionHelpers:
    @staticmethod
    def assert_approx_equal(actual, expected, rel_tol=0.01):
        assert math.isclose(actual, expected, rel_tol=rel_tol)

    @staticmethod
    def assert_provenance_hash_valid(hash_value):
        assert len(hash_value) == 64


@pytest.fixture
def assertions():
    return AssertionHelpers()


@dataclass
class GoldenTestCase:
    test_id: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: float = 0.01


@pytest.fixture
def golden_test_cases():
    return [
        GoldenTestCase("GOLDEN_001", "Healthy trap", {"ultrasonic_db": 65.0}, {"failure_mode": "NORMAL"}),
        GoldenTestCase("GOLDEN_002", "Failed open", {"ultrasonic_db": 98.0}, {"failure_mode": "FAILED_OPEN"}),
    ]


@pytest.fixture(autouse=True)
def reset_random_seeds():
    np.random.seed(42)
    random.seed(42)
    yield
