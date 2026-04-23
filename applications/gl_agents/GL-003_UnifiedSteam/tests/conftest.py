"""
Pytest Configuration and Shared Fixtures for GL-003 UnifiedSteam Test Suite.

This module provides:
- Shared fixtures for steam system state and components
- Mock data generators for sensor readings
- Test configuration and markers
- Reusable assertion helpers
- Factory fixtures for creating test objects

Target Coverage: 90%+
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

# Import application modules
# Note: These imports assume the package structure - adjust paths as needed
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "compliance: marks tests as regulatory compliance tests")
    config.addinivalue_line("markers", "hypothesis: marks tests using hypothesis library")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Auto-mark integration tests
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark slow tests
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# =============================================================================
# IAPWS-IF97 Constants and Reference Data
# =============================================================================

IAPWS_IF97_CONSTANTS = {
    "R": 0.461526,  # kJ/(kg*K) specific gas constant
    "T_CRIT": 647.096,  # K critical temperature
    "P_CRIT": 22.064,  # MPa critical pressure
    "RHO_CRIT": 322.0,  # kg/m3 critical density
    "T_TRIPLE": 273.16,  # K triple point temperature
    "P_TRIPLE": 0.000611657,  # MPa triple point pressure
}

# IAPWS-IF97 Table 5 verification points for Region 1
REGION1_TABLE5_VERIFICATION = [
    # (P [MPa], T [K], v [m3/kg], h [kJ/kg], s [kJ/(kg*K)], u [kJ/kg])
    (3.0, 300.0, 0.00100215168e-2, 115.331273, 0.392294792, 112.324818),
    (80.0, 300.0, 0.000971180894e-2, 184.142828, 0.368563852, 106.448356),
    (80.0, 500.0, 0.00120241800e-2, 975.542239, 2.58041912, 971.934985),
]

# IAPWS-IF97 Table 15 verification points for Region 2
REGION2_TABLE15_VERIFICATION = [
    # (P [MPa], T [K], v [m3/kg], h [kJ/kg], s [kJ/(kg*K)], u [kJ/kg])
    (0.001, 300.0, 0.394913866e2, 2549.91145, 9.15546786, 2411.69160),
    (0.001, 500.0, 0.658861816e2, 2928.62360, 9.87571394, 2862.75261),
    (3.0, 300.0, 0.00394913866, 2549.91145, 5.85296786, 2411.69160),
]

# IAPWS-IF97 Table 33 saturation verification points
SATURATION_TABLE33_VERIFICATION = [
    # (T [K], P_sat [MPa])
    (300.0, 0.00353658941),
    (500.0, 2.63889776),
    (600.0, 12.3443146),
]


# =============================================================================
# Steam System State Fixtures
# =============================================================================

@dataclass
class MockBoilerState:
    """Mock boiler state for testing."""
    boiler_id: str
    is_online: bool = True
    current_load_percent: float = 75.0
    rated_capacity_klb_hr: float = 100.0
    min_load_percent: float = 25.0
    max_load_percent: float = 100.0
    current_efficiency_percent: float = 85.0
    fuel_type: str = "natural_gas"
    fuel_cost_per_mmbtu: float = 8.0
    co2_factor_lb_mmbtu: float = 117.0
    maintenance_priority: int = 0


@dataclass
class MockHeaderState:
    """Mock steam header state for testing."""
    header_id: str
    header_type: str = "hp"  # hp, mp, lp
    pressure_psig: float = 600.0
    setpoint_psig: float = 600.0
    temperature_f: float = 700.0
    flow_klb_hr: float = 80.0
    user_demand_klb_hr: float = 75.0
    connected_boilers: List[str] = field(default_factory=list)
    connected_prvs: List[str] = field(default_factory=list)


@dataclass
class MockPRVState:
    """Mock PRV state for testing."""
    prv_id: str
    upstream_header: str = "hp_header"
    downstream_header: str = "mp_header"
    upstream_pressure_psig: float = 600.0
    downstream_pressure_psig: float = 150.0
    setpoint_psig: float = 150.0
    flow_klb_hr: float = 20.0
    valve_position_percent: float = 50.0
    max_capacity_klb_hr: float = 50.0
    is_desuperheating: bool = False


@dataclass
class MockNetworkState:
    """Complete mock steam network state."""
    boilers: List[MockBoilerState] = field(default_factory=list)
    headers: List[MockHeaderState] = field(default_factory=list)
    prvs: List[MockPRVState] = field(default_factory=list)
    total_generation_klb_hr: float = 150.0
    total_demand_klb_hr: float = 140.0
    distribution_loss_percent: float = 3.0


@pytest.fixture
def mock_boiler_state() -> MockBoilerState:
    """Create a single mock boiler state."""
    return MockBoilerState(boiler_id="BLR-001")


@pytest.fixture
def mock_boiler_fleet() -> List[MockBoilerState]:
    """Create a fleet of mock boilers for load allocation testing."""
    return [
        MockBoilerState(
            boiler_id="BLR-001",
            is_online=True,
            current_load_percent=80.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=85.0,
            fuel_cost_per_mmbtu=8.0,
        ),
        MockBoilerState(
            boiler_id="BLR-002",
            is_online=True,
            current_load_percent=70.0,
            rated_capacity_klb_hr=80.0,
            current_efficiency_percent=83.0,
            fuel_cost_per_mmbtu=8.5,
        ),
        MockBoilerState(
            boiler_id="BLR-003",
            is_online=True,
            current_load_percent=60.0,
            rated_capacity_klb_hr=120.0,
            current_efficiency_percent=87.0,
            fuel_cost_per_mmbtu=7.5,
        ),
        MockBoilerState(
            boiler_id="BLR-004",
            is_online=False,  # Offline for maintenance
            current_load_percent=0.0,
            rated_capacity_klb_hr=100.0,
            current_efficiency_percent=0.0,
            fuel_cost_per_mmbtu=8.0,
        ),
    ]


@pytest.fixture
def mock_header_state() -> MockHeaderState:
    """Create a single mock header state."""
    return MockHeaderState(
        header_id="HP-MAIN",
        header_type="hp",
        pressure_psig=600.0,
        setpoint_psig=600.0,
        temperature_f=700.0,
        flow_klb_hr=100.0,
        user_demand_klb_hr=95.0,
        connected_boilers=["BLR-001", "BLR-002"],
    )


@pytest.fixture
def mock_network_state(mock_boiler_fleet) -> MockNetworkState:
    """Create a complete mock network state."""
    headers = [
        MockHeaderState(
            header_id="HP-MAIN",
            header_type="hp",
            pressure_psig=600.0,
            setpoint_psig=600.0,
            flow_klb_hr=150.0,
            user_demand_klb_hr=80.0,
            connected_boilers=["BLR-001", "BLR-002", "BLR-003"],
            connected_prvs=["PRV-001"],
        ),
        MockHeaderState(
            header_id="MP-MAIN",
            header_type="mp",
            pressure_psig=150.0,
            setpoint_psig=150.0,
            flow_klb_hr=50.0,
            user_demand_klb_hr=45.0,
            connected_prvs=["PRV-002"],
        ),
        MockHeaderState(
            header_id="LP-MAIN",
            header_type="lp",
            pressure_psig=15.0,
            setpoint_psig=15.0,
            flow_klb_hr=20.0,
            user_demand_klb_hr=18.0,
        ),
    ]

    prvs = [
        MockPRVState(
            prv_id="PRV-001",
            upstream_header="HP-MAIN",
            downstream_header="MP-MAIN",
            upstream_pressure_psig=600.0,
            downstream_pressure_psig=150.0,
            flow_klb_hr=30.0,
        ),
        MockPRVState(
            prv_id="PRV-002",
            upstream_header="MP-MAIN",
            downstream_header="LP-MAIN",
            upstream_pressure_psig=150.0,
            downstream_pressure_psig=15.0,
            flow_klb_hr=10.0,
        ),
    ]

    return MockNetworkState(
        boilers=mock_boiler_fleet,
        headers=headers,
        prvs=prvs,
        total_generation_klb_hr=150.0,
        total_demand_klb_hr=143.0,
        distribution_loss_percent=3.5,
    )


# =============================================================================
# Uncertainty and Measurement Fixtures
# =============================================================================

@dataclass
class MockUncertainValue:
    """Mock uncertain value for testing."""
    mean: float
    std: float
    lower_95: float = 0.0
    upper_95: float = 0.0
    unit: str = ""
    source_id: str = ""

    def __post_init__(self):
        if self.lower_95 == 0.0 and self.upper_95 == 0.0:
            z_95 = 1.96
            self.lower_95 = self.mean - z_95 * self.std
            self.upper_95 = self.mean + z_95 * self.std

    def relative_uncertainty(self) -> float:
        """Return uncertainty as percentage of mean value."""
        if abs(self.mean) < 1e-10:
            return float('inf') if self.std > 0 else 0.0
        return (self.std / abs(self.mean)) * 100.0


@dataclass
class MockSensorReading:
    """Mock sensor reading with quality information."""
    tag: str
    value: float
    unit: str
    quality_code: int = 0  # 0 = GOOD
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uncertainty_percent: float = 2.0

    @property
    def is_good_quality(self) -> bool:
        return (self.quality_code & 0xC0000000) == 0x00000000


@pytest.fixture
def mock_uncertain_temperature() -> MockUncertainValue:
    """Create a mock uncertain temperature measurement."""
    return MockUncertainValue(
        mean=350.0,  # K
        std=1.5,  # K
        unit="K",
        source_id="TT-001"
    )


@pytest.fixture
def mock_uncertain_pressure() -> MockUncertainValue:
    """Create a mock uncertain pressure measurement."""
    return MockUncertainValue(
        mean=10.0,  # MPa
        std=0.05,  # MPa
        unit="MPa",
        source_id="PT-001"
    )


@pytest.fixture
def mock_uncertain_flow() -> MockUncertainValue:
    """Create a mock uncertain flow measurement."""
    return MockUncertainValue(
        mean=50.0,  # klb/hr
        std=1.0,  # klb/hr
        unit="klb/hr",
        source_id="FT-001"
    )


@pytest.fixture
def mock_sensor_readings() -> Dict[str, MockSensorReading]:
    """Create a set of mock sensor readings."""
    base_time = datetime.now(timezone.utc)
    return {
        "header.pressure": MockSensorReading(
            tag="header.pressure",
            value=600.0,
            unit="psig",
            timestamp=base_time,
            uncertainty_percent=1.5
        ),
        "header.temperature": MockSensorReading(
            tag="header.temperature",
            value=700.0,
            unit="degF",
            timestamp=base_time,
            uncertainty_percent=1.0
        ),
        "header.flow": MockSensorReading(
            tag="header.flow",
            value=100.0,
            unit="klb/hr",
            timestamp=base_time,
            uncertainty_percent=2.0
        ),
        "feedwater.temperature": MockSensorReading(
            tag="feedwater.temperature",
            value=220.0,
            unit="degF",
            timestamp=base_time,
            uncertainty_percent=1.5
        ),
        "stack.temperature": MockSensorReading(
            tag="stack.temperature",
            value=350.0,
            unit="degF",
            timestamp=base_time,
            uncertainty_percent=2.0
        ),
    }


# =============================================================================
# Mock Data Generators
# =============================================================================

class SensorDataGenerator:
    """Generator for realistic sensor data with noise and anomalies."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_temperature_series(
        self,
        n_points: int = 100,
        base_value: float = 350.0,
        noise_std: float = 2.0,
        drift_rate: float = 0.0,
        anomaly_probability: float = 0.02,
        unit: str = "K"
    ) -> List[MockSensorReading]:
        """Generate a time series of temperature readings."""
        readings = []
        base_time = datetime.now(timezone.utc)

        for i in range(n_points):
            # Base value with drift
            value = base_value + drift_rate * i

            # Add noise
            value += self.rng.normal(0, noise_std)

            # Add occasional anomalies
            quality_code = 0  # GOOD
            if random.random() < anomaly_probability:
                value += self.rng.normal(0, noise_std * 10)  # Large spike
                quality_code = 0x40000000  # UNCERTAIN

            readings.append(MockSensorReading(
                tag=f"temperature_{i}",
                value=value,
                unit=unit,
                quality_code=quality_code,
                timestamp=base_time + timedelta(seconds=i * 60),
                uncertainty_percent=noise_std / base_value * 100
            ))

        return readings

    def generate_pressure_series(
        self,
        n_points: int = 100,
        base_value: float = 10.0,
        noise_std: float = 0.1,
        cycle_amplitude: float = 0.5,
        cycle_period: int = 50,
        unit: str = "MPa"
    ) -> List[MockSensorReading]:
        """Generate a time series of pressure readings with cyclic behavior."""
        readings = []
        base_time = datetime.now(timezone.utc)

        for i in range(n_points):
            # Base value with cyclic variation
            cycle_component = cycle_amplitude * math.sin(2 * math.pi * i / cycle_period)
            value = base_value + cycle_component

            # Add noise
            value += self.rng.normal(0, noise_std)

            readings.append(MockSensorReading(
                tag=f"pressure_{i}",
                value=value,
                unit=unit,
                timestamp=base_time + timedelta(seconds=i * 60),
                uncertainty_percent=noise_std / base_value * 100
            ))

        return readings

    def generate_batch_readings(
        self,
        tags: List[str],
        n_batches: int = 10,
        base_values: Optional[Dict[str, float]] = None,
        noise_levels: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, MockSensorReading]]:
        """Generate batches of sensor readings."""
        base_values = base_values or {}
        noise_levels = noise_levels or {}
        base_time = datetime.now(timezone.utc)

        batches = []
        for batch_idx in range(n_batches):
            batch = {}
            timestamp = base_time + timedelta(seconds=batch_idx * 60)

            for tag in tags:
                base_val = base_values.get(tag, 100.0)
                noise = noise_levels.get(tag, 1.0)

                batch[tag] = MockSensorReading(
                    tag=tag,
                    value=base_val + self.rng.normal(0, noise),
                    unit="",
                    timestamp=timestamp,
                    uncertainty_percent=noise / base_val * 100
                )

            batches.append(batch)

        return batches

    def generate_trap_feature_data(
        self,
        n_samples: int = 100,
        failure_fraction: float = 0.1
    ) -> List[Dict[str, float]]:
        """Generate trap feature data for ML model testing."""
        samples = []
        n_failures = int(n_samples * failure_fraction)

        for i in range(n_samples):
            is_failure = i < n_failures

            # Base features
            sample = {
                "temp_differential_f": self.rng.normal(15, 5) if not is_failure else self.rng.normal(35, 10),
                "outlet_temp_f": self.rng.normal(180, 10) if not is_failure else self.rng.normal(250, 20),
                "inlet_temp_f": self.rng.normal(350, 15),
                "superheat_f": self.rng.normal(20, 5),
                "subcooling_f": self.rng.normal(5, 2) if not is_failure else self.rng.normal(1, 0.5),
                "operating_hours": self.rng.uniform(1000, 8000) if not is_failure else self.rng.uniform(8000, 15000),
                "cycles_count": self.rng.uniform(500, 2000),
                "days_since_inspection": self.rng.uniform(10, 60) if not is_failure else self.rng.uniform(60, 180),
                "differential_pressure_psi": self.rng.normal(50, 10),
                "previous_failures": self.rng.integers(0, 2) if not is_failure else self.rng.integers(1, 5),
                "is_failure": 1.0 if is_failure else 0.0,
            }
            samples.append(sample)

        return samples


@pytest.fixture
def sensor_data_generator() -> SensorDataGenerator:
    """Create a sensor data generator with fixed seed."""
    return SensorDataGenerator(seed=42)


@pytest.fixture
def sample_temperature_series(sensor_data_generator) -> List[MockSensorReading]:
    """Generate sample temperature time series."""
    return sensor_data_generator.generate_temperature_series(n_points=50)


@pytest.fixture
def sample_pressure_series(sensor_data_generator) -> List[MockSensorReading]:
    """Generate sample pressure time series."""
    return sensor_data_generator.generate_pressure_series(n_points=50)


@pytest.fixture
def sample_trap_features(sensor_data_generator) -> List[Dict[str, float]]:
    """Generate sample trap feature data for ML testing."""
    return sensor_data_generator.generate_trap_feature_data(n_samples=100)


# =============================================================================
# Mock Classes and Factories
# =============================================================================

@pytest.fixture
def mock_historian_connector():
    """Create a mock historian connector."""
    mock = AsyncMock()
    mock.connect.return_value = True
    mock.query_historical.return_value = {
        "header.pressure": {
            "tag": "header.pressure",
            "values": [600.0, 601.0, 599.5, 600.5],
            "timestamps": [
                datetime.now(timezone.utc) - timedelta(minutes=i)
                for i in range(4)
            ],
            "quality_codes": [0, 0, 0, 0],
        }
    }
    return mock


@pytest.fixture
def mock_tag_mapper():
    """Create a mock tag mapper."""
    mock = MagicMock()
    mock.get_canonical_tag.side_effect = lambda raw_tag: raw_tag.replace("-", ".").lower()
    mock.get_sensor_metadata.return_value = {
        "sensor_type": "pressure",
        "unit": "psig",
        "range_min": 0.0,
        "range_max": 1000.0,
    }
    return mock


@pytest.fixture
def mock_sensor_transformer():
    """Create a mock sensor transformer."""
    mock = MagicMock()
    mock.transform_single.side_effect = lambda tag, value, ts: MockSensorReading(
        tag=tag,
        value=value,
        unit="",
        timestamp=ts or datetime.now(timezone.utc),
    )
    mock.transform_batch.return_value = {
        "values": {},
        "quality_score": 95.0,
    }
    return mock


# =============================================================================
# Assertion Helpers
# =============================================================================

class AssertionHelpers:
    """Collection of assertion helper functions."""

    @staticmethod
    def assert_approx_equal(actual: float, expected: float, rel_tol: float = 0.01, abs_tol: float = 1e-9):
        """Assert two values are approximately equal."""
        assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), \
            f"Values not approximately equal: {actual} vs {expected} (rel_tol={rel_tol})"

    @staticmethod
    def assert_in_range(value: float, min_val: float, max_val: float):
        """Assert a value is within a specified range."""
        assert min_val <= value <= max_val, \
            f"Value {value} not in range [{min_val}, {max_val}]"

    @staticmethod
    def assert_positive(value: float):
        """Assert a value is positive."""
        assert value > 0, f"Value {value} is not positive"

    @staticmethod
    def assert_non_negative(value: float):
        """Assert a value is non-negative."""
        assert value >= 0, f"Value {value} is negative"

    @staticmethod
    def assert_valid_probability(value: float):
        """Assert a value is a valid probability [0, 1]."""
        assert 0 <= value <= 1, f"Value {value} is not a valid probability"

    @staticmethod
    def assert_valid_percentage(value: float):
        """Assert a value is a valid percentage [0, 100]."""
        assert 0 <= value <= 100, f"Value {value} is not a valid percentage"

    @staticmethod
    def assert_provenance_hash_valid(hash_value: str):
        """Assert a provenance hash is valid SHA-256."""
        assert len(hash_value) == 64, f"Hash length {len(hash_value)} != 64"
        assert all(c in '0123456789abcdef' for c in hash_value.lower()), \
            f"Hash contains non-hex characters"

    @staticmethod
    def assert_deterministic(func: Callable, args: tuple, kwargs: dict = None, n_iterations: int = 5):
        """Assert a function produces deterministic results."""
        kwargs = kwargs or {}
        results = [func(*args, **kwargs) for _ in range(n_iterations)]
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result == first, f"Iteration {i} produced different result: {result} vs {first}"

    @staticmethod
    def assert_monotonic_increasing(values: List[float]):
        """Assert values are monotonically increasing."""
        for i in range(1, len(values)):
            assert values[i] >= values[i-1], \
                f"Values not monotonic: {values[i-1]} > {values[i]} at index {i}"

    @staticmethod
    def assert_monotonic_decreasing(values: List[float]):
        """Assert values are monotonically decreasing."""
        for i in range(1, len(values)):
            assert values[i] <= values[i-1], \
                f"Values not monotonic: {values[i-1]} < {values[i]} at index {i}"


@pytest.fixture
def assertions() -> AssertionHelpers:
    """Provide assertion helper functions."""
    return AssertionHelpers()


# =============================================================================
# IAPWS-IF97 Reference Fixtures
# =============================================================================

@pytest.fixture
def iapws_constants() -> Dict[str, float]:
    """Provide IAPWS-IF97 constants."""
    return IAPWS_IF97_CONSTANTS


@pytest.fixture
def region1_verification_data() -> List[Tuple]:
    """Provide Region 1 verification data from IAPWS-IF97 Table 5."""
    return REGION1_TABLE5_VERIFICATION


@pytest.fixture
def region2_verification_data() -> List[Tuple]:
    """Provide Region 2 verification data from IAPWS-IF97 Table 15."""
    return REGION2_TABLE15_VERIFICATION


@pytest.fixture
def saturation_verification_data() -> List[Tuple]:
    """Provide saturation verification data from IAPWS-IF97 Table 33."""
    return SATURATION_TABLE33_VERIFICATION


# =============================================================================
# Quality Gate Fixtures
# =============================================================================

@dataclass
class MockGateResult:
    """Mock quality gate result."""
    status: str  # "passed", "warning", "blocked", "requires_confirmation"
    recommendation_id: str
    uncertainty_level: float
    threshold: float
    confidence_level: float
    reason: str
    required_action: str = ""

    def is_blocked(self) -> bool:
        return self.status == "blocked"

    def requires_human_action(self) -> bool:
        return self.status in ["blocked", "requires_confirmation"]


@pytest.fixture
def mock_gate_thresholds() -> Dict[str, float]:
    """Provide default quality gate thresholds."""
    return {
        "passed_threshold": 5.0,
        "warning_threshold": 10.0,
        "blocked_threshold": 20.0,
        "confirmation_threshold": 15.0,
    }


@pytest.fixture
def mock_gate_result_passed() -> MockGateResult:
    """Create a mock passed gate result."""
    return MockGateResult(
        status="passed",
        recommendation_id="REC-001",
        uncertainty_level=3.0,
        threshold=20.0,
        confidence_level=0.95,
        reason="Uncertainty within acceptable limits"
    )


@pytest.fixture
def mock_gate_result_blocked() -> MockGateResult:
    """Create a mock blocked gate result."""
    return MockGateResult(
        status="blocked",
        recommendation_id="REC-002",
        uncertainty_level=25.0,
        threshold=20.0,
        confidence_level=0.75,
        reason="Uncertainty exceeds limit",
        required_action="Improve sensor calibration"
    )


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_threshold_ms() -> float:
    """Default performance threshold in milliseconds."""
    return 5.0


@pytest.fixture
def large_dataset_size() -> int:
    """Size for large dataset performance tests."""
    return 10000


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Hypothesis Configuration
# =============================================================================

# Import hypothesis settings if available
try:
    from hypothesis import settings, Verbosity, Phase

    # Configure hypothesis settings
    settings.register_profile(
        "ci",
        max_examples=100,
        deadline=None,
        suppress_health_check=[],
    )
    settings.register_profile(
        "dev",
        max_examples=10,
        deadline=None,
    )
    settings.register_profile(
        "full",
        max_examples=1000,
        deadline=None,
    )
except ImportError:
    pass


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    yield


@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory for file-based tests."""
    return tmp_path


# =============================================================================
# Integration Test Helpers
# =============================================================================

@pytest.fixture
def mock_api_response():
    """Create a factory for mock API responses."""
    def _create_response(status_code: int = 200, data: Dict = None):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = data or {}
        response.ok = status_code < 400
        return response
    return _create_response


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    mock = AsyncMock()
    mock.send.return_value = None
    mock.receive_json.return_value = {"type": "data", "values": {}}
    mock.close.return_value = None
    return mock
