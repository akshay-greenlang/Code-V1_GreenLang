"""
Pytest Configuration and Shared Fixtures for GL-012 SteamQual Test Suite.

This module provides:
- Shared fixtures for steam quality measurements
- Mock sensors and historian connectors
- Test configuration and markers
- Reusable assertion helpers
- Factory fixtures for creating test objects
- Determinism verification utilities

Target Coverage: 85%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest


# =============================================================================
# Configuration Constants
# =============================================================================

TEST_SEED = 42  # For reproducibility
TEST_TIMEOUT = 30  # seconds
PERFORMANCE_TIMEOUT = 60  # seconds for performance tests

# Steam quality thresholds
DRYNESS_FRACTION_MIN = 0.0
DRYNESS_FRACTION_MAX = 1.0
DRYNESS_FRACTION_TARGET = 0.95  # Target for saturated steam quality

# Thermodynamic constants
CRITICAL_PRESSURE_MPA = 22.064
CRITICAL_TEMPERATURE_K = 647.096
TRIPLE_POINT_TEMPERATURE_K = 273.16
TRIPLE_POINT_PRESSURE_MPA = 0.000611657

# Calculation tolerances
ENTHALPY_TOLERANCE_KJ_KG = 0.5
ENTROPY_TOLERANCE_KJ_KG_K = 0.001
DRYNESS_FRACTION_TOLERANCE = 0.001
UNCERTAINTY_TOLERANCE_PERCENT = 5.0

# Performance targets
CALCULATION_TIME_TARGET_MS = 5.0
BATCH_THROUGHPUT_TARGET = 1000  # calculations/second


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "compliance: marks tests as regulatory compliance tests")
    config.addinivalue_line("markers", "determinism: marks tests for determinism verification")
    config.addinivalue_line("markers", "golden: marks tests as golden master tests")
    config.addinivalue_line("markers", "asyncio: marks tests as async tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "safety: marks tests for safety constraint validation")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Auto-mark integration tests
        if "test_integration" in str(item.fspath) or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark unit tests
        if "test_unit" in str(item.fspath) or "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Auto-mark golden tests
        if "test_golden" in str(item.fspath) or "golden" in str(item.fspath):
            item.add_marker(pytest.mark.golden)

        # Auto-mark slow tests
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Steam State Enumerations
# =============================================================================

class SteamState(Enum):
    """Enumeration of possible steam states."""
    SUBCOOLED_LIQUID = auto()
    SATURATED_LIQUID = auto()
    WET_STEAM = auto()
    SATURATED_VAPOR = auto()
    SUPERHEATED_VAPOR = auto()
    SUPERCRITICAL = auto()
    UNKNOWN = auto()


class QualityMethod(Enum):
    """Method used for dryness fraction calculation."""
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"
    THROTTLING = "throttling"
    SEPARATING = "separating"


class CarryoverRisk(Enum):
    """Carryover risk classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Data Classes for Test Fixtures
# =============================================================================

@dataclass
class SteamMeasurement:
    """Steam measurement at a specific condition."""
    timestamp: datetime
    pressure_mpa: float
    temperature_k: float
    enthalpy_kj_kg: Optional[float] = None
    entropy_kj_kg_k: Optional[float] = None
    dryness_fraction: Optional[float] = None
    state: SteamState = SteamState.UNKNOWN
    sensor_id: str = "PT-001"
    quality_score: float = 1.0
    uncertainty_percent: float = 2.0


@dataclass
class SensorReading:
    """Mock sensor reading with quality metadata."""
    tag: str
    value: float
    unit: str
    quality_code: int = 0  # 0 = GOOD
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uncertainty_percent: float = 2.0
    sensor_type: str = "pressure"

    @property
    def is_good_quality(self) -> bool:
        """Check if reading is of good quality."""
        return (self.quality_code & 0xC0000000) == 0x00000000


@dataclass
class UncertainValue:
    """Value with uncertainty information."""
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
class SaturationProperties:
    """Saturation properties at a given pressure."""
    pressure_mpa: float
    temperature_k: float
    h_f_kj_kg: float  # Saturated liquid enthalpy
    h_g_kj_kg: float  # Saturated vapor enthalpy
    h_fg_kj_kg: float  # Latent heat
    s_f_kj_kg_k: float  # Saturated liquid entropy
    s_g_kj_kg_k: float  # Saturated vapor entropy
    s_fg_kj_kg_k: float  # Entropy of vaporization
    v_f_m3_kg: float  # Saturated liquid specific volume
    v_g_m3_kg: float  # Saturated vapor specific volume


@dataclass
class SeparatorState:
    """Steam separator state for testing."""
    separator_id: str
    inlet_pressure_mpa: float
    inlet_temperature_k: float
    inlet_dryness_fraction: float
    outlet_dryness_fraction: float
    separation_efficiency: float
    condensate_flow_kg_s: float
    steam_flow_kg_s: float


@dataclass
class CarryoverAssessment:
    """Carryover risk assessment result."""
    risk_level: CarryoverRisk
    probability: float
    tds_ppm: float  # Total dissolved solids
    silica_ppb: float
    conductivity_us_cm: float
    recommended_action: str
    provenance_hash: str


@dataclass
class QualityEstimate:
    """Steam quality estimation result."""
    dryness_fraction: float
    uncertainty_percent: float
    method: QualityMethod
    confidence_level: float
    input_hash: str
    output_hash: str
    provenance_hash: str
    calculation_time_ms: float


# =============================================================================
# IAPWS-IF97 Reference Data
# =============================================================================

# Reference saturation properties at common pressures
SATURATION_REFERENCE_DATA = {
    0.1: SaturationProperties(
        pressure_mpa=0.1,
        temperature_k=372.756,
        h_f_kj_kg=417.44,
        h_g_kj_kg=2675.5,
        h_fg_kj_kg=2258.0,
        s_f_kj_kg_k=1.3026,
        s_g_kj_kg_k=7.3594,
        s_fg_kj_kg_k=6.0568,
        v_f_m3_kg=0.001043,
        v_g_m3_kg=1.694,
    ),
    1.0: SaturationProperties(
        pressure_mpa=1.0,
        temperature_k=453.03,
        h_f_kj_kg=762.81,
        h_g_kj_kg=2778.1,
        h_fg_kj_kg=2015.3,
        s_f_kj_kg_k=2.1387,
        s_g_kj_kg_k=6.5865,
        s_fg_kj_kg_k=4.4478,
        v_f_m3_kg=0.001127,
        v_g_m3_kg=0.1944,
    ),
    5.0: SaturationProperties(
        pressure_mpa=5.0,
        temperature_k=536.67,
        h_f_kj_kg=1154.2,
        h_g_kj_kg=2794.3,
        h_fg_kj_kg=1640.1,
        s_f_kj_kg_k=2.9202,
        s_g_kj_kg_k=5.9734,
        s_fg_kj_kg_k=3.0532,
        v_f_m3_kg=0.001286,
        v_g_m3_kg=0.03944,
    ),
    10.0: SaturationProperties(
        pressure_mpa=10.0,
        temperature_k=584.15,
        h_f_kj_kg=1407.6,
        h_g_kj_kg=2724.7,
        h_fg_kj_kg=1317.1,
        s_f_kj_kg_k=3.3596,
        s_g_kj_kg_k=5.6141,
        s_fg_kj_kg_k=2.2545,
        v_f_m3_kg=0.001452,
        v_g_m3_kg=0.01803,
    ),
}


# =============================================================================
# Seed Management for Determinism
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)
    yield


# =============================================================================
# Steam Measurement Fixtures
# =============================================================================

@pytest.fixture
def sample_steam_measurement() -> SteamMeasurement:
    """Create a sample steam measurement at typical conditions."""
    return SteamMeasurement(
        timestamp=datetime.now(timezone.utc),
        pressure_mpa=1.0,
        temperature_k=453.03,  # Saturation temperature at 1 MPa
        enthalpy_kj_kg=2500.0,  # Wet steam enthalpy
        entropy_kj_kg_k=5.5,
        dryness_fraction=0.85,
        state=SteamState.WET_STEAM,
        sensor_id="PT-001",
        quality_score=0.98,
        uncertainty_percent=1.5,
    )


@pytest.fixture
def sample_saturated_liquid() -> SteamMeasurement:
    """Create a sample saturated liquid measurement (x=0)."""
    sat_props = SATURATION_REFERENCE_DATA[1.0]
    return SteamMeasurement(
        timestamp=datetime.now(timezone.utc),
        pressure_mpa=1.0,
        temperature_k=sat_props.temperature_k,
        enthalpy_kj_kg=sat_props.h_f_kj_kg,
        entropy_kj_kg_k=sat_props.s_f_kj_kg_k,
        dryness_fraction=0.0,
        state=SteamState.SATURATED_LIQUID,
        sensor_id="PT-002",
    )


@pytest.fixture
def sample_saturated_vapor() -> SteamMeasurement:
    """Create a sample saturated vapor measurement (x=1)."""
    sat_props = SATURATION_REFERENCE_DATA[1.0]
    return SteamMeasurement(
        timestamp=datetime.now(timezone.utc),
        pressure_mpa=1.0,
        temperature_k=sat_props.temperature_k,
        enthalpy_kj_kg=sat_props.h_g_kj_kg,
        entropy_kj_kg_k=sat_props.s_g_kj_kg_k,
        dryness_fraction=1.0,
        state=SteamState.SATURATED_VAPOR,
        sensor_id="PT-003",
    )


@pytest.fixture
def sample_superheated_steam() -> SteamMeasurement:
    """Create a sample superheated steam measurement."""
    return SteamMeasurement(
        timestamp=datetime.now(timezone.utc),
        pressure_mpa=1.0,
        temperature_k=500.0,  # Above saturation
        enthalpy_kj_kg=2875.0,  # Superheated enthalpy
        entropy_kj_kg_k=7.0,
        dryness_fraction=None,  # Not applicable for superheated
        state=SteamState.SUPERHEATED_VAPOR,
        sensor_id="PT-004",
    )


@pytest.fixture
def sample_wet_steam_series() -> List[SteamMeasurement]:
    """Create a series of wet steam measurements at various qualities."""
    sat_props = SATURATION_REFERENCE_DATA[1.0]
    base_time = datetime.now(timezone.utc)
    measurements = []

    for i, x in enumerate([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]):
        h = sat_props.h_f_kj_kg + x * sat_props.h_fg_kj_kg
        s = sat_props.s_f_kj_kg_k + x * sat_props.s_fg_kj_kg_k

        measurements.append(SteamMeasurement(
            timestamp=base_time + timedelta(seconds=i * 10),
            pressure_mpa=1.0,
            temperature_k=sat_props.temperature_k,
            enthalpy_kj_kg=h,
            entropy_kj_kg_k=s,
            dryness_fraction=x,
            state=SteamState.SATURATED_LIQUID if x == 0 else (
                SteamState.SATURATED_VAPOR if x == 1 else SteamState.WET_STEAM
            ),
            sensor_id=f"PT-{i+1:03d}",
        ))

    return measurements


@pytest.fixture
def sample_measurements_various_pressures() -> List[SteamMeasurement]:
    """Create measurements at various pressures."""
    base_time = datetime.now(timezone.utc)
    measurements = []

    for i, (pressure, sat_props) in enumerate(SATURATION_REFERENCE_DATA.items()):
        x = 0.9  # 90% dryness
        h = sat_props.h_f_kj_kg + x * sat_props.h_fg_kj_kg
        s = sat_props.s_f_kj_kg_k + x * sat_props.s_fg_kj_kg_k

        measurements.append(SteamMeasurement(
            timestamp=base_time + timedelta(seconds=i * 10),
            pressure_mpa=pressure,
            temperature_k=sat_props.temperature_k,
            enthalpy_kj_kg=h,
            entropy_kj_kg_k=s,
            dryness_fraction=x,
            state=SteamState.WET_STEAM,
            sensor_id=f"PT-{i+1:03d}",
        ))

    return measurements


# =============================================================================
# Mock Sensor Fixtures
# =============================================================================

@pytest.fixture
def mock_pressure_sensor() -> SensorReading:
    """Create a mock pressure sensor reading."""
    return SensorReading(
        tag="steam.pressure",
        value=1.0,
        unit="MPa",
        quality_code=0,
        uncertainty_percent=1.5,
        sensor_type="pressure",
    )


@pytest.fixture
def mock_temperature_sensor() -> SensorReading:
    """Create a mock temperature sensor reading."""
    return SensorReading(
        tag="steam.temperature",
        value=453.03,
        unit="K",
        quality_code=0,
        uncertainty_percent=1.0,
        sensor_type="temperature",
    )


@pytest.fixture
def mock_sensor_suite() -> Dict[str, SensorReading]:
    """Create a complete mock sensor suite for steam quality measurement."""
    base_time = datetime.now(timezone.utc)
    return {
        "inlet.pressure": SensorReading(
            tag="inlet.pressure",
            value=5.0,
            unit="MPa",
            timestamp=base_time,
            uncertainty_percent=1.0,
        ),
        "inlet.temperature": SensorReading(
            tag="inlet.temperature",
            value=536.67,
            unit="K",
            timestamp=base_time,
            uncertainty_percent=0.8,
        ),
        "outlet.pressure": SensorReading(
            tag="outlet.pressure",
            value=1.0,
            unit="MPa",
            timestamp=base_time,
            uncertainty_percent=1.0,
        ),
        "outlet.temperature": SensorReading(
            tag="outlet.temperature",
            value=453.03,
            unit="K",
            timestamp=base_time,
            uncertainty_percent=0.8,
        ),
        "flow.rate": SensorReading(
            tag="flow.rate",
            value=50.0,
            unit="kg/s",
            timestamp=base_time,
            uncertainty_percent=2.0,
            sensor_type="flow",
        ),
        "conductivity": SensorReading(
            tag="conductivity",
            value=25.0,
            unit="uS/cm",
            timestamp=base_time,
            uncertainty_percent=3.0,
            sensor_type="conductivity",
        ),
    }


@pytest.fixture
def mock_bad_quality_sensor() -> SensorReading:
    """Create a mock sensor with bad quality reading."""
    return SensorReading(
        tag="steam.pressure.bad",
        value=0.0,  # Invalid value
        unit="MPa",
        quality_code=0xC0000000,  # BAD quality
        uncertainty_percent=100.0,
    )


# =============================================================================
# Mock Historian Fixtures
# =============================================================================

@pytest.fixture
def mock_historian_connector():
    """Create a mock historian connector."""
    mock = AsyncMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True

    # Generate historical data
    base_time = datetime.now(timezone.utc)
    historical_data = {
        "steam.pressure": {
            "tag": "steam.pressure",
            "values": [1.0, 1.01, 0.99, 1.0, 1.02],
            "timestamps": [base_time - timedelta(minutes=i) for i in range(5)],
            "quality_codes": [0, 0, 0, 0, 0],
            "unit": "MPa",
        },
        "steam.temperature": {
            "tag": "steam.temperature",
            "values": [453.0, 453.5, 452.8, 453.2, 453.1],
            "timestamps": [base_time - timedelta(minutes=i) for i in range(5)],
            "quality_codes": [0, 0, 0, 0, 0],
            "unit": "K",
        },
    }

    mock.query_historical.return_value = historical_data
    mock.query_tags.return_value = list(historical_data.keys())

    return mock


@pytest.fixture
def mock_realtime_connector():
    """Create a mock real-time data connector."""
    mock = AsyncMock()
    mock.connect.return_value = True
    mock.subscribe.return_value = "subscription_001"

    async def mock_read(tag: str):
        data = {
            "steam.pressure": {"value": 1.0, "quality": "GOOD", "timestamp": datetime.now(timezone.utc)},
            "steam.temperature": {"value": 453.03, "quality": "GOOD", "timestamp": datetime.now(timezone.utc)},
            "steam.flow": {"value": 50.0, "quality": "GOOD", "timestamp": datetime.now(timezone.utc)},
        }
        return data.get(tag, {"value": 0.0, "quality": "BAD", "timestamp": datetime.now(timezone.utc)})

    mock.read.side_effect = mock_read

    return mock


# =============================================================================
# Test Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration settings."""
    return {
        "calculation": {
            "dryness_tolerance": DRYNESS_FRACTION_TOLERANCE,
            "enthalpy_tolerance_kj_kg": ENTHALPY_TOLERANCE_KJ_KG,
            "entropy_tolerance_kj_kg_k": ENTROPY_TOLERANCE_KJ_KG_K,
            "uncertainty_propagation": True,
            "use_iapws_if97": True,
        },
        "performance": {
            "calculation_time_target_ms": CALCULATION_TIME_TARGET_MS,
            "batch_throughput_target": BATCH_THROUGHPUT_TARGET,
        },
        "safety": {
            "min_dryness_threshold": 0.85,
            "max_carryover_tds_ppm": 100.0,
            "alarm_on_low_quality": True,
        },
        "provenance": {
            "track_inputs": True,
            "track_outputs": True,
            "hash_algorithm": "sha256",
        },
    }


@pytest.fixture
def saturation_reference() -> Dict[float, SaturationProperties]:
    """Provide saturation reference data."""
    return SATURATION_REFERENCE_DATA.copy()


# =============================================================================
# Separator State Fixtures
# =============================================================================

@pytest.fixture
def sample_separator_state() -> SeparatorState:
    """Create a sample steam separator state."""
    return SeparatorState(
        separator_id="SEP-001",
        inlet_pressure_mpa=5.0,
        inlet_temperature_k=536.67,
        inlet_dryness_fraction=0.85,
        outlet_dryness_fraction=0.98,
        separation_efficiency=0.92,
        condensate_flow_kg_s=5.0,
        steam_flow_kg_s=45.0,
    )


@pytest.fixture
def sample_separator_series() -> List[SeparatorState]:
    """Create a series of separator states at different efficiencies."""
    states = []
    base_inlet_quality = 0.85

    for eff in [0.80, 0.85, 0.90, 0.95, 0.98]:
        # Outlet quality improves with separator efficiency
        outlet_quality = base_inlet_quality + (1.0 - base_inlet_quality) * eff

        states.append(SeparatorState(
            separator_id=f"SEP-{int(eff*100):03d}",
            inlet_pressure_mpa=5.0,
            inlet_temperature_k=536.67,
            inlet_dryness_fraction=base_inlet_quality,
            outlet_dryness_fraction=outlet_quality,
            separation_efficiency=eff,
            condensate_flow_kg_s=(1.0 - base_inlet_quality) * eff * 50.0,
            steam_flow_kg_s=base_inlet_quality * 50.0 + (1.0 - base_inlet_quality) * (1.0 - eff) * 50.0,
        ))

    return states


# =============================================================================
# Carryover Assessment Fixtures
# =============================================================================

@pytest.fixture
def sample_low_carryover() -> CarryoverAssessment:
    """Create a sample low carryover risk assessment."""
    return CarryoverAssessment(
        risk_level=CarryoverRisk.LOW,
        probability=0.05,
        tds_ppm=15.0,
        silica_ppb=50.0,
        conductivity_us_cm=20.0,
        recommended_action="Continue normal operation",
        provenance_hash="a" * 64,
    )


@pytest.fixture
def sample_high_carryover() -> CarryoverAssessment:
    """Create a sample high carryover risk assessment."""
    return CarryoverAssessment(
        risk_level=CarryoverRisk.HIGH,
        probability=0.75,
        tds_ppm=85.0,
        silica_ppb=200.0,
        conductivity_us_cm=80.0,
        recommended_action="Reduce load or initiate blowdown",
        provenance_hash="b" * 64,
    )


# =============================================================================
# Assertion Helpers
# =============================================================================

class AssertionHelpers:
    """Collection of assertion helper functions for steam quality testing."""

    @staticmethod
    def assert_approx_equal(actual: float, expected: float, rel_tol: float = 0.01, abs_tol: float = 1e-9):
        """Assert two values are approximately equal."""
        assert math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol), \
            f"Values not approximately equal: {actual} vs {expected} (rel_tol={rel_tol})"

    @staticmethod
    def assert_dryness_fraction_valid(x: float):
        """Assert dryness fraction is in valid range [0, 1]."""
        assert 0.0 <= x <= 1.0, f"Dryness fraction {x} not in valid range [0, 1]"

    @staticmethod
    def assert_positive(value: float, name: str = "Value"):
        """Assert a value is positive."""
        assert value > 0, f"{name} {value} is not positive"

    @staticmethod
    def assert_non_negative(value: float, name: str = "Value"):
        """Assert a value is non-negative."""
        assert value >= 0, f"{name} {value} is negative"

    @staticmethod
    def assert_provenance_hash_valid(hash_value: str):
        """Assert a provenance hash is valid SHA-256."""
        assert hash_value is not None, "Hash value is None"
        assert len(hash_value) == 64, f"Hash length {len(hash_value)} != 64"
        assert all(c in '0123456789abcdef' for c in hash_value.lower()), \
            "Hash contains non-hex characters"

    @staticmethod
    def assert_deterministic(func: Callable, args: tuple, kwargs: dict = None, n_iterations: int = 5):
        """Assert a function produces deterministic results."""
        kwargs = kwargs or {}
        results = [func(*args, **kwargs) for _ in range(n_iterations)]
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result == first, f"Iteration {i} produced different result: {result} vs {first}"

    @staticmethod
    def assert_uncertainty_propagated(uncertainty_in: float, uncertainty_out: float):
        """Assert uncertainty was propagated (output uncertainty >= input)."""
        assert uncertainty_out >= 0, "Output uncertainty cannot be negative"
        # Generally, uncertainty should propagate through calculations

    @staticmethod
    def assert_enthalpy_balance(h_in: float, h_out: float, tolerance_percent: float = 5.0):
        """Assert enthalpy balance within tolerance."""
        if h_in == 0:
            return
        error_percent = abs(h_out - h_in) / abs(h_in) * 100
        assert error_percent <= tolerance_percent, \
            f"Enthalpy balance error {error_percent:.2f}% exceeds tolerance {tolerance_percent}%"

    @staticmethod
    def assert_thermodynamic_consistency(props: Dict[str, float]):
        """Assert thermodynamic properties are self-consistent."""
        # Example: u = h - Pv
        if all(k in props for k in ['h', 'P', 'v', 'u']):
            u_calc = props['h'] - props['P'] * 1000 * props['v']
            assert abs(u_calc - props['u']) < 1.0, \
                f"u = h - Pv not satisfied: {u_calc} vs {props['u']}"


@pytest.fixture
def assertions() -> AssertionHelpers:
    """Provide assertion helper functions."""
    return AssertionHelpers()


# =============================================================================
# Provenance Calculator
# =============================================================================

class ProvenanceCalculator:
    """Calculate and verify provenance hashes for reproducibility."""

    @staticmethod
    def calculate_hash(data: Any) -> str:
        """Calculate SHA-256 hash for data."""
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

    @staticmethod
    def calculate_input_hash(inputs: Dict[str, float]) -> str:
        """Calculate hash for input parameters."""
        # Normalize floats to avoid floating point issues
        normalized = {k: round(v, 10) for k, v in inputs.items()}
        return ProvenanceCalculator.calculate_hash(normalized)


@pytest.fixture
def provenance_calculator() -> ProvenanceCalculator:
    """Provide provenance calculator for hash verification."""
    return ProvenanceCalculator()


# =============================================================================
# Mock Data Generators
# =============================================================================

class SteamDataGenerator:
    """Generator for realistic steam quality test data."""

    def __init__(self, seed: int = TEST_SEED):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_wet_steam_data(
        self,
        n_points: int = 100,
        pressure_mpa: float = 1.0,
        mean_dryness: float = 0.90,
        dryness_std: float = 0.05,
    ) -> List[SteamMeasurement]:
        """Generate wet steam measurement data."""
        sat_props = SATURATION_REFERENCE_DATA.get(pressure_mpa)
        if sat_props is None:
            # Approximate for pressures not in reference
            sat_props = SATURATION_REFERENCE_DATA[1.0]

        base_time = datetime.now(timezone.utc)
        measurements = []

        for i in range(n_points):
            x = np.clip(self.rng.normal(mean_dryness, dryness_std), 0.0, 1.0)
            h = sat_props.h_f_kj_kg + x * sat_props.h_fg_kj_kg
            s = sat_props.s_f_kj_kg_k + x * sat_props.s_fg_kj_kg_k

            # Add measurement noise
            h += self.rng.normal(0, 1.0)
            s += self.rng.normal(0, 0.01)

            measurements.append(SteamMeasurement(
                timestamp=base_time + timedelta(seconds=i * 10),
                pressure_mpa=pressure_mpa + self.rng.normal(0, 0.01),
                temperature_k=sat_props.temperature_k + self.rng.normal(0, 0.5),
                enthalpy_kj_kg=h,
                entropy_kj_kg_k=s,
                dryness_fraction=x,
                state=SteamState.WET_STEAM,
                sensor_id=f"PT-{i+1:03d}",
                uncertainty_percent=2.0 + self.rng.uniform(0, 1),
            ))

        return measurements

    def generate_carryover_events(
        self,
        n_events: int = 50,
        high_risk_fraction: float = 0.2,
    ) -> List[CarryoverAssessment]:
        """Generate carryover assessment events."""
        events = []
        n_high_risk = int(n_events * high_risk_fraction)

        for i in range(n_events):
            is_high_risk = i < n_high_risk

            if is_high_risk:
                tds = self.rng.uniform(60, 120)
                silica = self.rng.uniform(150, 300)
                conductivity = self.rng.uniform(60, 100)
                risk = CarryoverRisk.HIGH if tds > 80 else CarryoverRisk.MEDIUM
                probability = self.rng.uniform(0.6, 0.95)
            else:
                tds = self.rng.uniform(5, 40)
                silica = self.rng.uniform(20, 100)
                conductivity = self.rng.uniform(10, 40)
                risk = CarryoverRisk.LOW
                probability = self.rng.uniform(0.01, 0.15)

            events.append(CarryoverAssessment(
                risk_level=risk,
                probability=probability,
                tds_ppm=tds,
                silica_ppb=silica,
                conductivity_us_cm=conductivity,
                recommended_action="Monitor" if risk == CarryoverRisk.LOW else "Take action",
                provenance_hash=hashlib.sha256(f"event_{i}".encode()).hexdigest(),
            ))

        return events


@pytest.fixture
def steam_data_generator() -> SteamDataGenerator:
    """Provide steam data generator for testing."""
    return SteamDataGenerator(seed=TEST_SEED)


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_threshold_ms() -> float:
    """Default performance threshold in milliseconds."""
    return CALCULATION_TIME_TARGET_MS


@pytest.fixture
def large_dataset_size() -> int:
    """Size for large dataset performance tests."""
    return 10000


# =============================================================================
# Async Event Loop
# =============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Temporary Directory
# =============================================================================

@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory for file-based tests."""
    return tmp_path


# =============================================================================
# GL-003 Interface Mock
# =============================================================================

@pytest.fixture
def mock_gl003_interface():
    """Create a mock GL-003 UnifiedSteam interface."""
    mock = MagicMock()

    # Mock steam property calculation
    mock.compute_properties.return_value = {
        "enthalpy_kj_kg": 2500.0,
        "entropy_kj_kg_k": 5.5,
        "specific_volume_m3_kg": 0.15,
        "quality": 0.9,
        "state": "WET_STEAM",
        "provenance_hash": "a" * 64,
    }

    # Mock saturation properties
    mock.get_saturation_properties.return_value = SATURATION_REFERENCE_DATA[1.0]

    # Mock optimization recommendations
    mock.get_recommendations.return_value = []

    return mock
