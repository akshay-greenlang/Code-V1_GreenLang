# -*- coding: utf-8 -*-
"""
Pytest Configuration and Shared Fixtures for GL-017 Condensync Test Suite.

Comprehensive fixtures for condenser optimization testing including:
- HEI calculator fixtures and reference data
- Vacuum optimization fixtures
- Fouling prediction fixtures
- Condenser state classification fixtures
- Mock connectors and integrations
- Performance measurement fixtures

Target Coverage: 85%+
Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser Performance Guidelines

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    markers = [
        "unit: Unit tests for individual components",
        "integration: Integration tests for system components",
        "performance: Performance benchmark tests",
        "golden: Golden master tests for determinism",
        "property: Property-based tests with Hypothesis",
        "slow: Slow-running tests",
        "hei: HEI standard calculation tests",
        "vacuum: Vacuum optimization tests",
        "fouling: Fouling prediction tests",
        "classifier: State classification tests",
        "api: API endpoint tests",
        "requires_network: Tests requiring network access",
        "requires_database: Tests requiring database connection",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on directory path."""
    for item in items:
        path_str = str(item.fspath)

        if "unit" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "integration" in path_str:
            item.add_marker(pytest.mark.integration)
        elif "golden" in path_str:
            item.add_marker(pytest.mark.golden)
        elif "property" in path_str:
            item.add_marker(pytest.mark.property)
        elif "chaos" in path_str:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

TEST_SEED = 42
DEFAULT_TIMEOUT = 30
PERFORMANCE_TIMEOUT = 60
COVERAGE_TARGET = 85.0

# HEI Standard Reference Values
HEI_REFERENCE_CONDITIONS = {
    "cw_inlet_temp_f": 70.0,      # 70F = 21.11C
    "cw_velocity_fps": 7.0,       # 7 ft/s = 2.134 m/s
    "tube_cleanliness": 0.85,     # 85% cleanliness factor
    "fouling_factor_hr_ft2_f_btu": 0.0005,  # HEI design fouling
}

# Condenser Operating Limits
OPERATING_LIMITS = {
    "vacuum_min_kpa_abs": 2.0,    # Minimum condenser pressure
    "vacuum_max_kpa_abs": 15.0,   # Maximum condenser pressure
    "cw_inlet_min_c": 5.0,        # Minimum CW inlet
    "cw_inlet_max_c": 40.0,       # Maximum CW inlet
    "cw_rise_min_c": 5.0,         # Minimum temperature rise
    "cw_rise_max_c": 20.0,        # Maximum temperature rise
    "cf_min": 0.60,               # Minimum cleanliness factor
    "cf_max": 1.0,                # Maximum cleanliness factor
    "ttd_min_c": 2.0,             # Minimum terminal temp difference
    "ttd_max_c": 15.0,            # Maximum terminal temp difference
}


# =============================================================================
# DOMAIN ENUMERATIONS
# =============================================================================

class TubeMaterial(str, Enum):
    """Condenser tube material types."""
    ADMIRALTY_BRASS = "admiralty_brass"
    COPPER_NICKEL_90_10 = "cu_ni_90_10"
    COPPER_NICKEL_70_30 = "cu_ni_70_30"
    TITANIUM_GRADE_2 = "titanium_grade_2"
    STAINLESS_304 = "ss_304"
    STAINLESS_316 = "ss_316"
    DUPLEX_2205 = "duplex_2205"


class FailureMode(str, Enum):
    """Condenser failure/degradation modes."""
    NORMAL = "normal"
    FOULING_BIOLOGICAL = "fouling_bio"
    FOULING_SCALE = "fouling_scale"
    FOULING_DEBRIS = "fouling_debris"
    AIR_LEAK_MINOR = "air_leak_minor"
    AIR_LEAK_MAJOR = "air_leak_major"
    TUBE_LEAK = "tube_leak"
    TUBE_PLUGGED = "tube_plugged"
    CW_PUMP_DEGRADED = "cw_pump"
    UNKNOWN = "unknown"


class FailureSeverity(str, Enum):
    """Severity classification for detected issues."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CleaningMethod(str, Enum):
    """Condenser tube cleaning methods."""
    ONLINE_BALL = "online_ball"
    ONLINE_BRUSH = "online_brush"
    OFFLINE_HYDROLANCE = "offline_hydro"
    OFFLINE_CHEMICAL = "offline_chem"
    NONE = "none"


class OperatingMode(str, Enum):
    """Condenser operating mode."""
    NORMAL = "normal"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    LOAD_FOLLOW = "load_follow"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class WaterSource(str, Enum):
    """Cooling water source type."""
    ONCE_THROUGH_OCEAN = "ocean"
    ONCE_THROUGH_RIVER = "river"
    COOLING_TOWER = "ct_mechanical"
    COOLING_POND = "pond"


# =============================================================================
# DATA CLASSES - INPUTS
# =============================================================================

@dataclass
class CondenserConfig:
    """Condenser design configuration."""
    condenser_id: str
    plant_name: str
    unit_capacity_mw: float
    tube_material: TubeMaterial
    tube_od_mm: float
    tube_thickness_mm: float
    tube_length_m: float
    num_tubes: int
    num_passes: int
    shell_diameter_m: float
    surface_area_m2: float
    design_pressure_kpa_abs: float
    design_cw_flow_m3_s: float
    design_cw_inlet_temp_c: float
    water_source: WaterSource = WaterSource.COOLING_TOWER

    @property
    def tube_id_mm(self) -> float:
        """Calculate tube inner diameter."""
        return self.tube_od_mm - 2 * self.tube_thickness_mm

    @property
    def flow_area_m2(self) -> float:
        """Calculate total flow area."""
        tube_id_m = self.tube_id_mm / 1000
        return (math.pi / 4) * (tube_id_m ** 2) * self.num_tubes / self.num_passes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condenser_id": self.condenser_id,
            "plant_name": self.plant_name,
            "unit_capacity_mw": self.unit_capacity_mw,
            "tube_material": self.tube_material.value,
            "surface_area_m2": self.surface_area_m2,
        }


@dataclass
class CondenserReading:
    """Real-time condenser sensor reading."""
    timestamp: datetime
    condenser_id: str

    # Cooling water temperatures
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float

    # Cooling water flow
    cw_flow_m3_s: float
    cw_velocity_m_s: float

    # Vacuum and steam
    vacuum_pressure_kpa_abs: float
    saturation_temp_c: float
    steam_flow_kg_s: float

    # Hotwell
    hotwell_temp_c: float
    hotwell_level_pct: float

    # Air ingress
    air_ingress_scfm: float
    dissolved_oxygen_ppb: float

    # Operational
    unit_load_mw: float
    operating_mode: OperatingMode = OperatingMode.NORMAL

    @property
    def cw_temp_rise_c(self) -> float:
        """Calculate cooling water temperature rise."""
        return self.cw_outlet_temp_c - self.cw_inlet_temp_c

    @property
    def terminal_temp_diff_c(self) -> float:
        """Calculate terminal temperature difference (TTD)."""
        return self.saturation_temp_c - self.cw_outlet_temp_c

    @property
    def approach_temp_c(self) -> float:
        """Calculate approach temperature."""
        return self.saturation_temp_c - self.cw_inlet_temp_c

    @property
    def subcooling_c(self) -> float:
        """Calculate condensate subcooling."""
        return self.saturation_temp_c - self.hotwell_temp_c

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "condenser_id": self.condenser_id,
            "cw_inlet_temp_c": self.cw_inlet_temp_c,
            "cw_outlet_temp_c": self.cw_outlet_temp_c,
            "cw_flow_m3_s": self.cw_flow_m3_s,
            "vacuum_pressure_kpa_abs": self.vacuum_pressure_kpa_abs,
            "unit_load_mw": self.unit_load_mw,
        }


@dataclass
class ThermalInput:
    """Input for thermal performance calculations."""
    cw_inlet_temp_c: float
    cw_outlet_temp_c: float
    cw_flow_kg_s: float
    steam_saturation_temp_c: float
    steam_flow_kg_s: float
    latent_heat_kj_kg: float = 2257.0  # At atmospheric pressure
    cw_cp_kj_kg_k: float = 4.186

    @property
    def heat_duty_kw(self) -> float:
        """Calculate actual heat duty from CW side."""
        return self.cw_flow_kg_s * self.cw_cp_kj_kg_k * (self.cw_outlet_temp_c - self.cw_inlet_temp_c)

    @property
    def ttd_c(self) -> float:
        """Terminal temperature difference."""
        return self.steam_saturation_temp_c - self.cw_outlet_temp_c

    @property
    def approach_c(self) -> float:
        """Approach temperature (steam to CW inlet)."""
        return self.steam_saturation_temp_c - self.cw_inlet_temp_c


@dataclass
class VacuumOptimizationInput:
    """Input for vacuum optimization calculations."""
    current_backpressure_kpa: float
    unit_load_mw: float
    cw_inlet_temp_c: float
    cw_flow_m3_s: float
    cw_pump_power_kw: float
    heat_rate_kj_kwh: float
    electricity_price_usd_mwh: float
    steam_cycle_efficiency: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_backpressure_kpa": self.current_backpressure_kpa,
            "unit_load_mw": self.unit_load_mw,
            "cw_inlet_temp_c": self.cw_inlet_temp_c,
            "cw_flow_m3_s": self.cw_flow_m3_s,
        }


@dataclass
class FoulingHistoryEntry:
    """Historical fouling data entry."""
    timestamp: datetime
    cleanliness_factor: float
    heat_duty_mw: float
    cw_inlet_temp_c: float
    data_quality_score: float = 1.0


# =============================================================================
# DATA CLASSES - OUTPUTS
# =============================================================================

@dataclass
class HEICalculationResult:
    """Result from HEI cleanliness factor calculation."""
    cleanliness_factor: float
    lmtd_c: float
    heat_duty_kw: float
    ua_actual_kw_k: float
    ua_clean_kw_k: float
    fouling_resistance_m2_k_kw: float
    correction_factors: Dict[str, float]
    provenance_hash: str
    calculation_timestamp: datetime
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cleanliness_factor": self.cleanliness_factor,
            "lmtd_c": self.lmtd_c,
            "heat_duty_kw": self.heat_duty_kw,
            "ua_actual_kw_k": self.ua_actual_kw_k,
            "ua_clean_kw_k": self.ua_clean_kw_k,
            "fouling_resistance_m2_k_kw": self.fouling_resistance_m2_k_kw,
            "correction_factors": self.correction_factors,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class VacuumOptimizationResult:
    """Result from vacuum/backpressure optimization."""
    optimal_backpressure_kpa: float
    current_backpressure_kpa: float
    mw_gain_potential: float
    annual_savings_usd: float
    cw_flow_recommendation_m3_s: float
    economic_optimum: bool
    sensitivity_analysis: Dict[str, float]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimal_backpressure_kpa": self.optimal_backpressure_kpa,
            "mw_gain_potential": self.mw_gain_potential,
            "annual_savings_usd": self.annual_savings_usd,
        }


@dataclass
class FoulingPredictionResult:
    """Result from fouling prediction calculation."""
    current_cf: float
    predicted_cf: float
    days_to_threshold: int
    cf_decay_rate_per_day: float
    recommended_cleaning_date: datetime
    cleaning_method: CleaningMethod
    cleaning_cost_usd: float
    lost_generation_cost_usd: float
    roi_cleaning: float
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_cf": self.current_cf,
            "predicted_cf": self.predicted_cf,
            "days_to_threshold": self.days_to_threshold,
            "cleaning_method": self.cleaning_method.value,
        }


@dataclass
class CondenserStateResult:
    """Result from condenser state classification."""
    condenser_id: str
    timestamp: datetime
    failure_mode: FailureMode
    severity: FailureSeverity
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    estimated_impact_mw: float
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condenser_id": self.condenser_id,
            "failure_mode": self.failure_mode.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
        }


@dataclass
class GoldenTestCase:
    """Golden test case with expected values."""
    test_id: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    tolerance: float = 0.01
    source: str = "HEI Standards"

    def verify(self, actual: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify actual output against expected."""
        for key, expected_value in self.expected_output.items():
            if key not in actual:
                return False, f"Missing key: {key}"
            actual_value = actual[key]

            if isinstance(expected_value, (int, float)):
                if expected_value == 0:
                    if abs(actual_value) > self.tolerance:
                        return False, f"{key}: expected 0, got {actual_value}"
                else:
                    rel_error = abs(actual_value - expected_value) / abs(expected_value)
                    if rel_error > self.tolerance:
                        return False, f"{key}: expected {expected_value}, got {actual_value}, error={rel_error:.4%}"
            elif actual_value != expected_value:
                return False, f"{key}: expected {expected_value}, got {actual_value}"

        return True, "All values match"


# =============================================================================
# MOCK CLASSES
# =============================================================================

class MockSensorReading:
    """Mock sensor reading for testing."""
    def __init__(
        self,
        tag: str,
        value: float,
        unit: str,
        quality_code: int = 0,
        timestamp: datetime = None,
        uncertainty_percent: float = 2.0
    ):
        self.tag = tag
        self.value = value
        self.unit = unit
        self.quality_code = quality_code
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.uncertainty_percent = uncertainty_percent


class MockHistorianConnector:
    """Mock historian connector for integration testing."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._connected = False
        self._data_cache: Dict[str, List[Tuple[datetime, float]]] = {}

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def read_tag(self, tag: str) -> MockSensorReading:
        """Read current value of a tag."""
        if not self._connected:
            raise ConnectionError("Not connected")
        return MockSensorReading(tag, random.uniform(0, 100), "unit")

    async def read_historical(
        self,
        tag: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> List[Tuple[datetime, float]]:
        """Read historical data for a tag."""
        if not self._connected:
            raise ConnectionError("Not connected")

        data = []
        current = start_time
        while current <= end_time:
            value = random.uniform(0, 100)
            data.append((current, value))
            current += timedelta(seconds=interval_seconds)
        return data


class MockCondenserSensorConnector:
    """Mock condenser sensor connector for testing."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._connected = False
        self._read_count = 0
        self._error_count = 0

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def read_condenser_data(
        self,
        condenser_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Read condenser data bundle."""
        if not self._connected:
            raise ConnectionError("Not connected")

        results = {}
        timestamp = datetime.now(timezone.utc)

        for cond_id in condenser_ids:
            # Generate realistic values
            random.seed(hash(cond_id) + int(time.time() / 60))
            cw_inlet = 25.0 + random.uniform(-3, 3)
            cw_outlet = cw_inlet + random.uniform(8, 15)
            vacuum_mbar = 40.0 + random.uniform(-5, 15)

            results[cond_id] = {
                "condenser_id": cond_id,
                "timestamp": timestamp.isoformat(),
                "cw_inlet_temp_c": round(cw_inlet, 2),
                "cw_outlet_temp_c": round(cw_outlet, 2),
                "vacuum_pressure_mbar_a": round(vacuum_mbar, 2),
                "data_quality": "good",
            }

        self._read_count += 1
        return results


class MockCMMSConnector:
    """Mock CMMS connector for work order testing."""

    def __init__(self):
        self._work_orders: List[Dict[str, Any]] = []
        self._next_id = 1

    async def create_work_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a work order."""
        wo = {
            "work_order_id": f"WO-{self._next_id:06d}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "CREATED",
            **data
        }
        self._work_orders.append(wo)
        self._next_id += 1
        return wo

    async def get_work_order(self, wo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a work order."""
        return next((wo for wo in self._work_orders if wo["work_order_id"] == wo_id), None)


# =============================================================================
# HELPER CLASSES
# =============================================================================

class AssertionHelpers:
    """Collection of assertion helper methods."""

    @staticmethod
    def assert_approx_equal(actual: float, expected: float, rel_tol: float = 0.01, msg: str = ""):
        """Assert two values are approximately equal."""
        if expected == 0:
            assert abs(actual) < rel_tol, f"{msg}: expected 0, got {actual}"
        else:
            rel_error = abs(actual - expected) / abs(expected)
            assert rel_error <= rel_tol, f"{msg}: expected {expected}, got {actual}, error={rel_error:.4%}"

    @staticmethod
    def assert_provenance_hash_valid(hash_value: str):
        """Assert provenance hash is valid SHA-256."""
        assert isinstance(hash_value, str), f"Hash must be string, got {type(hash_value)}"
        assert len(hash_value) == 64, f"SHA-256 hash must be 64 chars, got {len(hash_value)}"
        try:
            int(hash_value, 16)
        except ValueError:
            raise AssertionError(f"Invalid hex string: {hash_value}")

    @staticmethod
    def assert_in_range(value: float, min_val: float, max_val: float, param_name: str = "value"):
        """Assert value is within range."""
        assert min_val <= value <= max_val, f"{param_name}={value} outside range [{min_val}, {max_val}]"

    @staticmethod
    def assert_cf_valid(cf: float):
        """Assert cleanliness factor is physically valid."""
        AssertionHelpers.assert_in_range(cf, 0.0, 1.5, "cleanliness_factor")

    @staticmethod
    def assert_positive(value: float, param_name: str = "value"):
        """Assert value is positive."""
        assert value > 0, f"{param_name} must be positive, got {value}"

    @staticmethod
    def assert_non_negative(value: float, param_name: str = "value"):
        """Assert value is non-negative."""
        assert value >= 0, f"{param_name} must be non-negative, got {value}"


class ProvenanceCalculator:
    """Provenance hash calculation utilities."""

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute SHA-256 hash for data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @staticmethod
    def verify_hash(data: Any, expected_hash: str) -> bool:
        """Verify data matches expected hash."""
        actual = ProvenanceCalculator.compute_hash(data)
        return actual == expected_hash


class PerformanceTimer:
    """Performance timing utility."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None
        self.measurements: List[float] = []

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.measurements.append(self.elapsed)

    @property
    def average(self) -> float:
        """Get average of all measurements."""
        return sum(self.measurements) / len(self.measurements) if self.measurements else 0.0

    @property
    def percentile_95(self) -> float:
        """Get 95th percentile of measurements."""
        if not self.measurements:
            return 0.0
        sorted_times = sorted(self.measurements)
        idx = int(0.95 * len(sorted_times))
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def assert_within_target(self, target_seconds: float) -> None:
        """Assert elapsed time is within target."""
        if self.elapsed is None:
            raise RuntimeError("Timer not stopped")
        assert self.elapsed <= target_seconds, (
            f"Execution time {self.elapsed:.3f}s exceeds target {target_seconds}s"
        )


class ThroughputMeasurer:
    """Throughput measurement utility."""

    def __init__(self):
        self.item_count = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    def add_items(self, count: int):
        """Add processed items to counter."""
        self.item_count += count

    @property
    def items_per_second(self) -> float:
        """Calculate throughput in items/second."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        duration = self.end_time - self.start_time
        return self.item_count / duration if duration > 0 else 0.0

    def assert_meets_target(self, target_per_second: float) -> None:
        """Assert throughput meets target."""
        assert self.items_per_second >= target_per_second, (
            f"Throughput {self.items_per_second:.0f}/s below target {target_per_second}/s"
        )


class DeterminismChecker:
    """Utility to verify deterministic behavior."""

    def __init__(self):
        self.results: List[Any] = []

    def run_multiple(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 10
    ) -> List[Any]:
        """Run function multiple times and collect results."""
        kwargs = kwargs or {}
        self.results = [func(*args, **kwargs) for _ in range(iterations)]
        return self.results

    def check_identical_results(self) -> Tuple[bool, str]:
        """Check if all results are identical."""
        if not self.results:
            return False, "No results to check"

        first = self.results[0]
        for i, result in enumerate(self.results[1:], 2):
            if result != first:
                return False, f"Result {i} differs from result 1"

        return True, f"All {len(self.results)} results are identical"

    def check_identical_hashes(
        self,
        hash_func: Callable[[Any], str]
    ) -> Tuple[bool, str]:
        """Check if all result hashes are identical."""
        if not self.results:
            return False, "No results to check"

        hashes = [hash_func(r) for r in self.results]
        first_hash = hashes[0]

        for i, h in enumerate(hashes[1:], 2):
            if h != first_hash:
                return False, f"Hash {i} differs from hash 1"

        return True, f"All {len(hashes)} hashes are identical: {first_hash[:16]}..."


# =============================================================================
# FIXTURES - SEED MANAGEMENT
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Ensure reproducible randomness across all tests."""
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)
    yield


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Provide the test seed for reproducibility."""
    return TEST_SEED


# =============================================================================
# FIXTURES - AGENT CONFIGURATION
# =============================================================================

@pytest.fixture
def agent_config() -> Dict[str, Any]:
    """Provide default agent test configuration."""
    return {
        "agent_id": "GL-017",
        "agent_name": "Condensync",
        "version": "1.0.0",
        "environment": "test",
        "track_provenance": True,
        "enable_caching": False,
        "log_level": "DEBUG",
    }


@pytest.fixture
def hei_config() -> Dict[str, Any]:
    """HEI calculator configuration."""
    return {
        "standard_version": "12th_edition",
        "design_cleanliness": 0.85,
        "min_cleanliness_threshold": 0.75,
        "fouling_factor_design_m2_k_kw": 0.000088,
        "enable_corrections": True,
    }


@pytest.fixture
def vacuum_optimizer_config() -> Dict[str, Any]:
    """Vacuum optimizer configuration."""
    return {
        "heat_rate_sensitivity_kj_kwh_per_kpa": 45.0,
        "min_backpressure_kpa": 2.5,
        "max_backpressure_kpa": 12.0,
        "optimization_interval_hours": 1,
        "economic_optimization_enabled": True,
    }


@pytest.fixture
def fouling_predictor_config() -> Dict[str, Any]:
    """Fouling predictor configuration."""
    return {
        "cf_threshold_warning": 0.80,
        "cf_threshold_critical": 0.75,
        "prediction_horizon_days": 30,
        "history_window_days": 90,
        "decay_model": "exponential",
    }


# =============================================================================
# FIXTURES - CONDENSER DESIGN DATA
# =============================================================================

@pytest.fixture
def sample_condenser_config() -> CondenserConfig:
    """Sample condenser design configuration."""
    return CondenserConfig(
        condenser_id="COND-001",
        plant_name="Test Power Plant",
        unit_capacity_mw=500.0,
        tube_material=TubeMaterial.TITANIUM_GRADE_2,
        tube_od_mm=25.4,
        tube_thickness_mm=0.889,  # 22 BWG
        tube_length_m=12.0,
        num_tubes=18000,
        num_passes=2,
        shell_diameter_m=5.5,
        surface_area_m2=25000.0,
        design_pressure_kpa_abs=5.0,
        design_cw_flow_m3_s=15.0,
        design_cw_inlet_temp_c=25.0,
        water_source=WaterSource.COOLING_TOWER,
    )


@pytest.fixture
def large_condenser_config() -> CondenserConfig:
    """Large power plant condenser configuration."""
    return CondenserConfig(
        condenser_id="COND-002",
        plant_name="Large Power Plant",
        unit_capacity_mw=1000.0,
        tube_material=TubeMaterial.STAINLESS_316,
        tube_od_mm=28.575,
        tube_thickness_mm=1.245,
        tube_length_m=15.0,
        num_tubes=30000,
        num_passes=2,
        shell_diameter_m=7.0,
        surface_area_m2=45000.0,
        design_pressure_kpa_abs=4.5,
        design_cw_flow_m3_s=30.0,
        design_cw_inlet_temp_c=22.0,
        water_source=WaterSource.ONCE_THROUGH_OCEAN,
    )


@pytest.fixture
def small_condenser_config() -> CondenserConfig:
    """Small industrial condenser configuration."""
    return CondenserConfig(
        condenser_id="COND-003",
        plant_name="Small Industrial Plant",
        unit_capacity_mw=100.0,
        tube_material=TubeMaterial.ADMIRALTY_BRASS,
        tube_od_mm=19.05,
        tube_thickness_mm=1.245,
        tube_length_m=6.0,
        num_tubes=5000,
        num_passes=2,
        shell_diameter_m=3.0,
        surface_area_m2=6000.0,
        design_pressure_kpa_abs=7.0,
        design_cw_flow_m3_s=3.0,
        design_cw_inlet_temp_c=28.0,
        water_source=WaterSource.COOLING_TOWER,
    )


# =============================================================================
# FIXTURES - CONDENSER OPERATING DATA
# =============================================================================

@pytest.fixture
def healthy_condenser_reading() -> CondenserReading:
    """Reading from a healthy condenser."""
    return CondenserReading(
        timestamp=datetime.now(timezone.utc),
        condenser_id="COND-001",
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.5,
        cw_flow_m3_s=15.0,
        cw_velocity_m_s=2.1,
        vacuum_pressure_kpa_abs=5.0,
        saturation_temp_c=32.9,
        steam_flow_kg_s=180.0,
        hotwell_temp_c=32.5,
        hotwell_level_pct=50.0,
        air_ingress_scfm=2.0,
        dissolved_oxygen_ppb=5.0,
        unit_load_mw=480.0,
        operating_mode=OperatingMode.NORMAL,
    )


@pytest.fixture
def fouled_condenser_reading() -> CondenserReading:
    """Reading from a fouled condenser."""
    return CondenserReading(
        timestamp=datetime.now(timezone.utc),
        condenser_id="COND-001",
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=37.5,
        cw_flow_m3_s=14.5,
        cw_velocity_m_s=2.0,
        vacuum_pressure_kpa_abs=7.5,  # Elevated due to fouling
        saturation_temp_c=40.3,
        steam_flow_kg_s=175.0,
        hotwell_temp_c=39.8,
        hotwell_level_pct=52.0,
        air_ingress_scfm=3.5,
        dissolved_oxygen_ppb=12.0,
        unit_load_mw=450.0,
        operating_mode=OperatingMode.NORMAL,
    )


@pytest.fixture
def air_leak_condenser_reading() -> CondenserReading:
    """Reading from a condenser with air in-leakage."""
    return CondenserReading(
        timestamp=datetime.now(timezone.utc),
        condenser_id="COND-001",
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=36.0,
        cw_flow_m3_s=15.0,
        cw_velocity_m_s=2.1,
        vacuum_pressure_kpa_abs=8.0,
        saturation_temp_c=41.5,
        steam_flow_kg_s=178.0,
        hotwell_temp_c=38.0,  # Elevated subcooling
        hotwell_level_pct=48.0,
        air_ingress_scfm=12.0,  # High air ingress
        dissolved_oxygen_ppb=45.0,  # Elevated DO
        unit_load_mw=460.0,
        operating_mode=OperatingMode.NORMAL,
    )


@pytest.fixture
def condenser_fleet() -> List[CondenserReading]:
    """Fleet of condenser readings for batch testing."""
    base_time = datetime.now(timezone.utc)
    return [
        # Healthy
        CondenserReading(
            timestamp=base_time,
            condenser_id="COND-001",
            cw_inlet_temp_c=25.0, cw_outlet_temp_c=35.5, cw_flow_m3_s=15.0,
            cw_velocity_m_s=2.1, vacuum_pressure_kpa_abs=5.0, saturation_temp_c=32.9,
            steam_flow_kg_s=180.0, hotwell_temp_c=32.5, hotwell_level_pct=50.0,
            air_ingress_scfm=2.0, dissolved_oxygen_ppb=5.0, unit_load_mw=480.0,
        ),
        # Moderately fouled
        CondenserReading(
            timestamp=base_time,
            condenser_id="COND-002",
            cw_inlet_temp_c=26.0, cw_outlet_temp_c=37.0, cw_flow_m3_s=14.0,
            cw_velocity_m_s=1.9, vacuum_pressure_kpa_abs=6.5, saturation_temp_c=37.6,
            steam_flow_kg_s=175.0, hotwell_temp_c=37.0, hotwell_level_pct=52.0,
            air_ingress_scfm=3.0, dissolved_oxygen_ppb=8.0, unit_load_mw=460.0,
        ),
        # Severely fouled
        CondenserReading(
            timestamp=base_time,
            condenser_id="COND-003",
            cw_inlet_temp_c=24.0, cw_outlet_temp_c=38.0, cw_flow_m3_s=13.0,
            cw_velocity_m_s=1.8, vacuum_pressure_kpa_abs=9.0, saturation_temp_c=43.8,
            steam_flow_kg_s=165.0, hotwell_temp_c=43.0, hotwell_level_pct=55.0,
            air_ingress_scfm=4.0, dissolved_oxygen_ppb=15.0, unit_load_mw=420.0,
        ),
        # Air leak
        CondenserReading(
            timestamp=base_time,
            condenser_id="COND-004",
            cw_inlet_temp_c=25.0, cw_outlet_temp_c=36.0, cw_flow_m3_s=15.0,
            cw_velocity_m_s=2.1, vacuum_pressure_kpa_abs=7.5, saturation_temp_c=40.3,
            steam_flow_kg_s=170.0, hotwell_temp_c=36.0, hotwell_level_pct=45.0,
            air_ingress_scfm=15.0, dissolved_oxygen_ppb=60.0, unit_load_mw=450.0,
        ),
    ]


# =============================================================================
# FIXTURES - THERMAL CALCULATION INPUTS
# =============================================================================

@pytest.fixture
def thermal_input_baseline() -> ThermalInput:
    """Baseline thermal calculation input."""
    return ThermalInput(
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_kg_s=15000.0,  # ~15 m3/s @ 1000 kg/m3
        steam_saturation_temp_c=38.0,
        steam_flow_kg_s=180.0,
        latent_heat_kj_kg=2400.0,
    )


@pytest.fixture
def thermal_input_high_load() -> ThermalInput:
    """High load thermal calculation input."""
    return ThermalInput(
        cw_inlet_temp_c=22.0,
        cw_outlet_temp_c=34.0,
        cw_flow_kg_s=18000.0,
        steam_saturation_temp_c=36.0,
        steam_flow_kg_s=220.0,
        latent_heat_kj_kg=2400.0,
    )


@pytest.fixture
def thermal_input_summer() -> ThermalInput:
    """Summer conditions thermal input (high CW inlet)."""
    return ThermalInput(
        cw_inlet_temp_c=32.0,
        cw_outlet_temp_c=42.0,
        cw_flow_kg_s=15000.0,
        steam_saturation_temp_c=45.0,
        steam_flow_kg_s=170.0,
        latent_heat_kj_kg=2400.0,
    )


@pytest.fixture
def thermal_inputs_parametric() -> List[ThermalInput]:
    """Parametric thermal inputs for sensitivity testing."""
    inputs = []
    for cw_inlet in [20.0, 25.0, 30.0, 35.0]:
        for cw_rise in [8.0, 10.0, 12.0]:
            inputs.append(ThermalInput(
                cw_inlet_temp_c=cw_inlet,
                cw_outlet_temp_c=cw_inlet + cw_rise,
                cw_flow_kg_s=15000.0,
                steam_saturation_temp_c=cw_inlet + cw_rise + 3.0,
                steam_flow_kg_s=180.0,
            ))
    return inputs


# =============================================================================
# FIXTURES - VACUUM OPTIMIZATION INPUTS
# =============================================================================

@pytest.fixture
def vacuum_optimization_input() -> VacuumOptimizationInput:
    """Standard vacuum optimization input."""
    return VacuumOptimizationInput(
        current_backpressure_kpa=6.0,
        unit_load_mw=500.0,
        cw_inlet_temp_c=25.0,
        cw_flow_m3_s=15.0,
        cw_pump_power_kw=2000.0,
        heat_rate_kj_kwh=9500.0,
        electricity_price_usd_mwh=50.0,
    )


@pytest.fixture
def vacuum_optimization_high_price() -> VacuumOptimizationInput:
    """Vacuum optimization with high electricity price."""
    return VacuumOptimizationInput(
        current_backpressure_kpa=6.5,
        unit_load_mw=500.0,
        cw_inlet_temp_c=25.0,
        cw_flow_m3_s=15.0,
        cw_pump_power_kw=2000.0,
        heat_rate_kj_kwh=9500.0,
        electricity_price_usd_mwh=100.0,
    )


# =============================================================================
# FIXTURES - FOULING HISTORY DATA
# =============================================================================

@pytest.fixture
def fouling_history_clean() -> List[FoulingHistoryEntry]:
    """Fouling history showing clean condenser."""
    base_time = datetime.now(timezone.utc)
    entries = []
    for i in range(30):
        entries.append(FoulingHistoryEntry(
            timestamp=base_time - timedelta(days=i),
            cleanliness_factor=0.92 - 0.001 * i,  # Slow degradation
            heat_duty_mw=350.0 + random.uniform(-10, 10),
            cw_inlet_temp_c=25.0 + random.uniform(-2, 2),
            data_quality_score=0.95,
        ))
    return entries


@pytest.fixture
def fouling_history_rapid_decline() -> List[FoulingHistoryEntry]:
    """Fouling history showing rapid degradation."""
    base_time = datetime.now(timezone.utc)
    entries = []
    for i in range(30):
        entries.append(FoulingHistoryEntry(
            timestamp=base_time - timedelta(days=i),
            cleanliness_factor=0.90 - 0.01 * i,  # Rapid degradation
            heat_duty_mw=350.0 - 2.0 * i,
            cw_inlet_temp_c=25.0 + random.uniform(-2, 2),
            data_quality_score=0.90,
        ))
    return entries


@pytest.fixture
def fouling_history_post_cleaning() -> List[FoulingHistoryEntry]:
    """Fouling history showing recovery after cleaning."""
    base_time = datetime.now(timezone.utc)
    entries = []
    for i in range(30):
        # Cleaning happened 15 days ago
        if i >= 15:
            cf = 0.75 - 0.005 * (i - 15)  # Pre-cleaning decline
        else:
            cf = 0.92 - 0.002 * i  # Post-cleaning with slow decline

        entries.append(FoulingHistoryEntry(
            timestamp=base_time - timedelta(days=i),
            cleanliness_factor=cf,
            heat_duty_mw=350.0,
            cw_inlet_temp_c=25.0,
            data_quality_score=0.95,
        ))
    return entries


# =============================================================================
# FIXTURES - GOLDEN TEST CASES
# =============================================================================

@pytest.fixture
def golden_test_cases() -> List[GoldenTestCase]:
    """Collection of golden test cases with known values."""
    return [
        GoldenTestCase(
            test_id="HEI_CF_001",
            description="Clean condenser at design conditions",
            input_data={
                "cw_inlet_temp_c": 21.1,
                "cw_outlet_temp_c": 31.1,
                "cw_flow_kg_s": 15000.0,
                "steam_saturation_temp_c": 34.0,
                "design_ua_kw_k": 5000.0,
            },
            expected_output={
                "cleanliness_factor": 0.85,
                "lmtd_c": 6.08,  # Calculated LMTD
            },
            tolerance=0.05,
            source="HEI Standards 12th Edition"
        ),
        GoldenTestCase(
            test_id="LMTD_001",
            description="LMTD calculation - equal temperature difference",
            input_data={
                "ttd_c": 5.0,
                "approach_c": 5.0,
            },
            expected_output={
                "lmtd_c": 5.0,  # Equal temps -> LMTD = TTD = approach
            },
            tolerance=0.001,
            source="Heat Transfer Fundamentals"
        ),
        GoldenTestCase(
            test_id="LMTD_002",
            description="LMTD calculation - typical condenser",
            input_data={
                "ttd_c": 3.0,
                "approach_c": 13.0,
            },
            expected_output={
                "lmtd_c": 6.80,  # (13-3)/ln(13/3)
            },
            tolerance=0.02,
            source="Heat Transfer Fundamentals"
        ),
        GoldenTestCase(
            test_id="VACUUM_001",
            description="Saturation temperature at 5 kPa",
            input_data={
                "pressure_kpa_abs": 5.0,
            },
            expected_output={
                "saturation_temp_c": 32.88,  # From steam tables
            },
            tolerance=0.01,
            source="IAPWS-IF97"
        ),
        GoldenTestCase(
            test_id="HEAT_DUTY_001",
            description="Heat duty from CW side",
            input_data={
                "cw_flow_kg_s": 15000.0,
                "cw_cp_kj_kg_k": 4.186,
                "cw_temp_rise_c": 10.0,
            },
            expected_output={
                "heat_duty_kw": 627900.0,  # 15000 * 4.186 * 10
            },
            tolerance=0.001,
            source="First Law of Thermodynamics"
        ),
    ]


@pytest.fixture
def hei_correction_factors() -> Dict[str, Any]:
    """HEI correction factor reference data."""
    return {
        "inlet_water_temp_correction": {
            # Temperature (F) -> Correction Factor
            60: 1.04,
            70: 1.00,
            80: 0.96,
            90: 0.93,
            100: 0.90,
        },
        "tube_material_correction": {
            TubeMaterial.ADMIRALTY_BRASS: 1.00,
            TubeMaterial.COPPER_NICKEL_90_10: 0.95,
            TubeMaterial.COPPER_NICKEL_70_30: 0.92,
            TubeMaterial.TITANIUM_GRADE_2: 0.88,
            TubeMaterial.STAINLESS_304: 0.85,
            TubeMaterial.STAINLESS_316: 0.85,
        },
        "velocity_correction": {
            # Velocity (ft/s) -> Correction Factor
            5.0: 0.90,
            6.0: 0.95,
            7.0: 1.00,
            8.0: 1.04,
            9.0: 1.07,
        },
    }


# =============================================================================
# FIXTURES - MOCK CONNECTORS
# =============================================================================

@pytest.fixture
def mock_historian_connector():
    """Mock historian connector for testing."""
    connector = MockHistorianConnector()
    return connector


@pytest.fixture
def mock_condenser_sensor_connector():
    """Mock condenser sensor connector for testing."""
    connector = MockCondenserSensorConnector()
    return connector


@pytest.fixture
def mock_cmms_connector():
    """Mock CMMS connector for testing."""
    return MockCMMSConnector()


@pytest.fixture
def mock_opcua_client():
    """Mock OPC-UA client for integration testing."""
    client = MagicMock()
    client.connect = MagicMock(return_value=True)
    client.disconnect = MagicMock(return_value=True)
    client.is_connected = MagicMock(return_value=True)

    def read_node(node_id: str):
        base_values = {
            "ns=2;s=CW_INLET_TEMP": 25.0,
            "ns=2;s=CW_OUTLET_TEMP": 35.0,
            "ns=2;s=CW_FLOW": 15.0,
            "ns=2;s=VACUUM_PRESSURE": 5.0,
        }
        value = base_values.get(node_id, random.uniform(0, 100))
        return MagicMock(
            Value=MagicMock(Value=value + random.gauss(0, value * 0.01)),
            StatusCode=MagicMock(is_good=lambda: True),
            SourceTimestamp=datetime.now(timezone.utc),
        )

    client.get_node = MagicMock(side_effect=lambda nid: MagicMock(read_value=lambda: read_node(nid)))
    return client


@pytest.fixture
async def async_mock_historian_connector():
    """Async mock historian connector."""
    connector = AsyncMock()
    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock(return_value=True)
    connector.is_connected = AsyncMock(return_value=True)

    async def read_tag(tag: str):
        return MockSensorReading(tag, random.uniform(0, 100), "unit")

    connector.read_tag = AsyncMock(side_effect=read_tag)
    return connector


# =============================================================================
# FIXTURES - HELPER INSTANCES
# =============================================================================

@pytest.fixture
def assertions() -> AssertionHelpers:
    """Provide assertion helper instance."""
    return AssertionHelpers()


@pytest.fixture
def provenance_calculator() -> ProvenanceCalculator:
    """Provide provenance calculator instance."""
    return ProvenanceCalculator()


@pytest.fixture
def performance_timer() -> type:
    """Provide PerformanceTimer class."""
    return PerformanceTimer


@pytest.fixture
def throughput_measurer() -> type:
    """Provide ThroughputMeasurer class."""
    return ThroughputMeasurer


@pytest.fixture
def determinism_checker() -> DeterminismChecker:
    """Provide determinism checker instance."""
    return DeterminismChecker()


# =============================================================================
# FIXTURES - TEST DATA GENERATORS
# =============================================================================

@pytest.fixture
def condenser_data_generator():
    """Factory for generating condenser test data."""

    class CondenserDataGenerator:
        def __init__(self, seed: int = TEST_SEED):
            self.rng = np.random.default_rng(seed)

        def generate_reading(
            self,
            condenser_id: str = "COND-001",
            cf_target: float = 0.85,
            timestamp: datetime = None,
        ) -> CondenserReading:
            """Generate a condenser reading with target CF."""
            timestamp = timestamp or datetime.now(timezone.utc)

            # Base values
            cw_inlet = 25.0 + self.rng.uniform(-5, 5)
            cw_rise = 10.0 / cf_target  # Higher rise for lower CF
            cw_outlet = cw_inlet + cw_rise

            # Vacuum depends on CF and CW temps
            base_vacuum = 5.0
            vacuum_penalty = (1.0 - cf_target) * 5.0
            vacuum = base_vacuum + vacuum_penalty + self.rng.uniform(-0.5, 0.5)

            # Calculate saturation temp from vacuum
            sat_temp = 32.9 + (vacuum - 5.0) * 2.0

            return CondenserReading(
                timestamp=timestamp,
                condenser_id=condenser_id,
                cw_inlet_temp_c=round(cw_inlet, 2),
                cw_outlet_temp_c=round(cw_outlet, 2),
                cw_flow_m3_s=15.0 + self.rng.uniform(-1, 1),
                cw_velocity_m_s=2.1,
                vacuum_pressure_kpa_abs=round(vacuum, 2),
                saturation_temp_c=round(sat_temp, 2),
                steam_flow_kg_s=180.0,
                hotwell_temp_c=round(sat_temp - 0.5, 2),
                hotwell_level_pct=50.0,
                air_ingress_scfm=2.0,
                dissolved_oxygen_ppb=5.0,
                unit_load_mw=500.0,
                operating_mode=OperatingMode.NORMAL,
            )

        def generate_time_series(
            self,
            condenser_id: str,
            start_time: datetime,
            num_points: int,
            interval_minutes: int = 60,
            cf_initial: float = 0.92,
            cf_decay_per_day: float = 0.002,
        ) -> List[CondenserReading]:
            """Generate time series of readings with degradation."""
            readings = []
            for i in range(num_points):
                timestamp = start_time + timedelta(minutes=i * interval_minutes)
                days_elapsed = (i * interval_minutes) / (24 * 60)
                cf = max(0.60, cf_initial - cf_decay_per_day * days_elapsed)
                readings.append(self.generate_reading(condenser_id, cf, timestamp))
            return readings

    return CondenserDataGenerator()


@pytest.fixture
def boundary_conditions() -> Dict[str, Dict[str, float]]:
    """Boundary condition values for edge case testing."""
    return {
        "temperature_c": {
            "min": 0.0,
            "max": 100.0,
            "below_min": -10.0,
            "above_max": 120.0,
            "at_min": 0.0,
            "at_max": 100.0,
            "typical": 25.0,
        },
        "pressure_kpa": {
            "min": 1.0,
            "max": 20.0,
            "below_min": 0.5,
            "above_max": 25.0,
            "at_min": 1.0,
            "at_max": 20.0,
            "typical": 5.0,
        },
        "cleanliness_factor": {
            "min": 0.0,
            "max": 1.0,
            "below_min": -0.1,
            "above_max": 1.5,
            "at_min": 0.0,
            "at_max": 1.0,
            "typical": 0.85,
        },
        "flow_rate_m3_s": {
            "min": 0.0,
            "max": 100.0,
            "below_min": -1.0,
            "above_max": 150.0,
            "at_min": 0.0,
            "at_max": 100.0,
            "typical": 15.0,
        },
    }


# =============================================================================
# FIXTURES - SLA REQUIREMENTS
# =============================================================================

@pytest.fixture
def sla_requirements() -> Dict[str, Any]:
    """SLA requirements for acceptance testing."""
    return {
        "calculation_time_ms": 50,        # Max time for CF calculation
        "optimization_time_ms": 500,      # Max time for vacuum optimization
        "prediction_time_ms": 200,        # Max time for fouling prediction
        "classification_time_ms": 100,    # Max time for state classification
        "batch_throughput_per_second": 1000,  # Min readings/second
        "api_response_time_ms": 200,      # Max API response time
        "memory_usage_mb": 512,           # Max memory usage
        "coverage_target_percent": COVERAGE_TARGET,
        "determinism_requirement": 1.0,   # 100% reproducible
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_lmtd(ttd: float, approach: float) -> float:
    """Calculate Log Mean Temperature Difference."""
    if ttd <= 0 or approach <= 0:
        return 0.0
    if abs(ttd - approach) < 0.001:
        return ttd
    return (ttd - approach) / math.log(ttd / approach)


def saturation_temp_from_pressure(pressure_kpa: float) -> float:
    """Calculate saturation temperature from pressure (simplified)."""
    if pressure_kpa <= 0:
        return 25.0
    # Simplified Antoine equation approximation
    log_p = math.log10(pressure_kpa * 7.50062)  # Convert to mmHg
    t_sat = (1730.63 / (8.07131 - log_p)) - 233.426
    return max(20.0, min(t_sat, 100.0))


def pressure_from_saturation_temp(temp_c: float) -> float:
    """Calculate pressure from saturation temperature (simplified)."""
    if temp_c < 20 or temp_c > 100:
        return 5.0
    # Inverse Antoine equation
    log_p = 8.07131 - (1730.63 / (temp_c + 233.426))
    p_mmhg = 10 ** log_p
    return p_mmhg / 7.50062  # Convert to kPa


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TubeMaterial",
    "FailureMode",
    "FailureSeverity",
    "CleaningMethod",
    "OperatingMode",
    "WaterSource",
    # Data classes - Input
    "CondenserConfig",
    "CondenserReading",
    "ThermalInput",
    "VacuumOptimizationInput",
    "FoulingHistoryEntry",
    # Data classes - Output
    "HEICalculationResult",
    "VacuumOptimizationResult",
    "FoulingPredictionResult",
    "CondenserStateResult",
    "GoldenTestCase",
    # Mock classes
    "MockSensorReading",
    "MockHistorianConnector",
    "MockCondenserSensorConnector",
    "MockCMMSConnector",
    # Helper classes
    "AssertionHelpers",
    "ProvenanceCalculator",
    "PerformanceTimer",
    "ThroughputMeasurer",
    "DeterminismChecker",
    # Utility functions
    "calculate_lmtd",
    "saturation_temp_from_pressure",
    "pressure_from_saturation_temp",
    # Constants
    "TEST_SEED",
    "HEI_REFERENCE_CONDITIONS",
    "OPERATING_LIMITS",
    "COVERAGE_TARGET",
]
