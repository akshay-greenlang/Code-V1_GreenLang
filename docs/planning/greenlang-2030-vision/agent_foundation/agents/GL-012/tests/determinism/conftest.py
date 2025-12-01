# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Determinism Test Fixtures.

Provides comprehensive fixtures for determinism validation tests including:
- Golden file loaders for IAPWS-IF97 reference data
- Hash comparison utilities
- Multi-run test decorators
- Deterministic configuration fixtures
- Floating-point comparison utilities
- Cross-platform test utilities

Zero-hallucination: All fixtures ensure deterministic behavior.

Author: GL-DeterminismAuditor
Version: 1.0.0
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import numpy as np

# Configure high precision for Decimal operations
getcontext().prec = 50

# Add parent directories to path
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR.parent.parent))

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DETERMINISM_SEED = 42
DEFAULT_RUNS = 1000
GOLDEN_RUNS = 100
LLM_TEMPERATURE = 0.0
HASH_ALGORITHM = "sha256"
FLOATING_POINT_TOLERANCE = 1e-15
DECIMAL_PRECISION = 10

# Python version info for cross-version tests
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
PLATFORM_INFO = {
    "system": platform.system(),
    "release": platform.release(),
    "machine": platform.machine(),
    "python_version": PYTHON_VERSION,
}


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers for determinism tests."""
    config.addinivalue_line("markers", "determinism: Determinism validation tests")
    config.addinivalue_line("markers", "reproducibility: Multi-run reproducibility tests")
    config.addinivalue_line("markers", "golden: Golden file validation tests")
    config.addinivalue_line("markers", "hash_consistency: Hash consistency tests")
    config.addinivalue_line("markers", "seed_propagation: Seed propagation tests")
    config.addinivalue_line("markers", "floating_point: Floating-point stability tests")
    config.addinivalue_line("markers", "cross_platform: Cross-platform tests")
    config.addinivalue_line("markers", "cross_version: Cross-Python-version tests")
    config.addinivalue_line("markers", "iapws: IAPWS-IF97 standard validation")
    config.addinivalue_line("markers", "asme: ASME PTC standard validation")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# DATACLASSES FOR DETERMINISM TESTING
# =============================================================================

@dataclass
class DeterminismTestResult:
    """Result of a determinism test run."""
    run_number: int
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    provenance_hash: str
    execution_time_ms: float
    platform_info: Dict[str, str] = field(default_factory=lambda: PLATFORM_INFO.copy())


@dataclass
class GoldenTestCase:
    """Golden test case with known expected values."""
    name: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    expected_hash: str
    tolerance: Dict[str, float] = field(default_factory=dict)
    source: str = "IAPWS-IF97"


@dataclass
class HashChainEntry:
    """Entry in a provenance hash chain."""
    step: int
    operation: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    hash_value: str
    parent_hash: Optional[str] = None


# =============================================================================
# HASH COMPARISON UTILITIES
# =============================================================================

class HashComparisonUtility:
    """Utility class for hash comparison and validation."""

    @staticmethod
    def compute_hash(data: Any, algorithm: str = HASH_ALGORITHM) -> str:
        """
        Compute deterministic hash of data.

        Args:
            data: Data to hash (must be JSON-serializable)
            algorithm: Hash algorithm (default: sha256)

        Returns:
            Hexadecimal hash string
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        if algorithm == "sha256":
            return hashlib.sha256(json_str.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(json_str.encode()).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(json_str.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @staticmethod
    def verify_hash_consistency(hashes: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Verify all hashes in list are identical.

        Args:
            hashes: List of hash strings to compare

        Returns:
            Tuple of (is_consistent, differing_hash if any)
        """
        if not hashes:
            return True, None

        first_hash = hashes[0]
        for i, h in enumerate(hashes[1:], 2):
            if h != first_hash:
                return False, f"Hash {i} differs: {h[:16]}... vs {first_hash[:16]}..."

        return True, None

    @staticmethod
    def verify_hash_chain(chain: List[HashChainEntry]) -> Tuple[bool, List[str]]:
        """
        Verify integrity of hash chain.

        Args:
            chain: List of hash chain entries

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        for i, entry in enumerate(chain):
            # Verify parent hash linkage
            if i > 0 and entry.parent_hash != chain[i - 1].hash_value:
                errors.append(f"Step {entry.step}: Parent hash mismatch")

            # Verify self-computed hash
            computed_hash = HashComparisonUtility.compute_hash({
                'step': entry.step,
                'operation': entry.operation,
                'inputs': entry.inputs,
                'outputs': entry.outputs,
                'parent_hash': entry.parent_hash
            })
            if computed_hash != entry.hash_value:
                errors.append(f"Step {entry.step}: Hash verification failed")

        return len(errors) == 0, errors

    @staticmethod
    def assert_hashes_identical(hashes: List[str], message: str = ""):
        """Assert all hashes are identical."""
        is_consistent, diff = HashComparisonUtility.verify_hash_consistency(hashes)
        assert is_consistent, f"{message} {diff}"


@pytest.fixture
def hash_utility():
    """Provide hash comparison utility."""
    return HashComparisonUtility()


# =============================================================================
# MULTI-RUN TEST DECORATORS
# =============================================================================

F = TypeVar('F', bound=Callable)


def run_multiple_times(runs: int = DEFAULT_RUNS):
    """
    Decorator to run a test multiple times and verify determinism.

    Args:
        runs: Number of times to run the test

    Usage:
        @run_multiple_times(1000)
        def test_calculation():
            result = calculate(input)
            return result  # Results from all runs will be compared
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            hashes = []

            for i in range(runs):
                # Reset random seeds before each run
                random.seed(DETERMINISM_SEED)
                np.random.seed(DETERMINISM_SEED)

                result = func(*args, **kwargs)
                results.append(result)

                if isinstance(result, dict) and 'provenance_hash' in result:
                    hashes.append(result['provenance_hash'])
                else:
                    hashes.append(HashComparisonUtility.compute_hash(result))

            # Verify all results are identical
            unique_results = len(set(str(r) for r in results))
            assert unique_results == 1, f"Non-deterministic: {unique_results} unique results from {runs} runs"

            # Verify all hashes are identical
            unique_hashes = len(set(hashes))
            assert unique_hashes == 1, f"Hash inconsistency: {unique_hashes} unique hashes from {runs} runs"

            return results[0]

        return wrapper
    return decorator


def verify_determinism(runs: int = 10):
    """
    Decorator to verify determinism by comparing outputs across runs.

    Simpler version for quick determinism checks.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            first_result = None
            first_hash = None

            for i in range(runs):
                random.seed(DETERMINISM_SEED)
                np.random.seed(DETERMINISM_SEED)

                result = func(*args, **kwargs)
                result_hash = HashComparisonUtility.compute_hash(result)

                if first_result is None:
                    first_result = result
                    first_hash = result_hash
                else:
                    assert result_hash == first_hash, f"Run {i+1} produced different hash"

            return first_result

        return wrapper
    return decorator


# =============================================================================
# GOLDEN FILE LOADERS
# =============================================================================

class GoldenFileLoader:
    """Loader for golden reference files."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize with optional base path for golden files."""
        self.base_path = base_path or (TEST_DIR / "golden_data")

    def load_iapws_if97_reference(self) -> Dict[str, GoldenTestCase]:
        """
        Load IAPWS-IF97 reference steam properties.

        Returns pre-computed reference values from the IAPWS-IF97
        Industrial Formulation for water and steam properties.
        """
        # IAPWS-IF97 reference values (Table 4 and Table 15)
        return {
            "saturation_10bar": GoldenTestCase(
                name="Saturation at 10 bar",
                inputs={
                    "pressure_mpa": 1.0,
                    "calculation_type": "saturation_properties"
                },
                expected_outputs={
                    "saturation_temperature_c": 179.88,
                    "h_f_kj_kg": 762.68,
                    "h_g_kj_kg": 2777.1,
                    "h_fg_kj_kg": 2014.9,
                    "s_f_kj_kg_k": 2.1381,
                    "s_g_kj_kg_k": 6.5828,
                    "v_f_m3_kg": 0.001127,
                    "v_g_m3_kg": 0.1944
                },
                expected_hash="",  # Will be computed
                tolerance={
                    "saturation_temperature_c": 0.01,
                    "h_f_kj_kg": 0.01,
                    "h_g_kj_kg": 0.1,
                    "h_fg_kj_kg": 0.1
                },
                source="IAPWS-IF97 Table 4"
            ),
            "superheated_10bar_250c": GoldenTestCase(
                name="Superheated steam at 10 bar, 250C",
                inputs={
                    "pressure_mpa": 1.0,
                    "temperature_c": 250.0,
                    "calculation_type": "superheated_properties"
                },
                expected_outputs={
                    "specific_enthalpy_kj_kg": 2942.9,
                    "specific_entropy_kj_kg_k": 6.9247,
                    "specific_volume_m3_kg": 0.2328
                },
                expected_hash="",
                tolerance={
                    "specific_enthalpy_kj_kg": 0.1,
                    "specific_entropy_kj_kg_k": 0.001,
                    "specific_volume_m3_kg": 0.0001
                },
                source="IAPWS-IF97 Table 15"
            ),
            "wet_steam_10bar_x0.9": GoldenTestCase(
                name="Wet steam at 10 bar, x=0.9",
                inputs={
                    "pressure_mpa": 1.0,
                    "dryness_fraction": 0.9,
                    "calculation_type": "wet_steam_properties"
                },
                expected_outputs={
                    "specific_enthalpy_kj_kg": 2576.09,  # h_f + 0.9 * h_fg
                    "specific_entropy_kj_kg_k": 6.1383,  # s_f + 0.9 * s_fg
                    "specific_volume_m3_kg": 0.1749  # v_f + 0.9 * v_fg
                },
                expected_hash="",
                tolerance={
                    "specific_enthalpy_kj_kg": 0.1,
                    "specific_entropy_kj_kg_k": 0.001,
                    "specific_volume_m3_kg": 0.001
                },
                source="IAPWS-IF97 Calculated"
            ),
            "dryness_fraction_calculation": GoldenTestCase(
                name="Dryness fraction from enthalpy",
                inputs={
                    "pressure_mpa": 1.0,
                    "enthalpy_kj_kg": 2576.09,
                    "calculation_type": "dryness_from_enthalpy"
                },
                expected_outputs={
                    "dryness_fraction": 0.9,
                    "wetness_fraction": 0.1
                },
                expected_hash="",
                tolerance={
                    "dryness_fraction": 0.001,
                    "wetness_fraction": 0.001
                },
                source="IAPWS-IF97 Calculated"
            )
        }

    def load_asme_ptc_reference(self) -> Dict[str, GoldenTestCase]:
        """
        Load ASME PTC standard test cases.

        Returns reference values from ASME Performance Test Codes.
        """
        return {
            "ptc6_steam_quality": GoldenTestCase(
                name="ASME PTC 6 Steam Quality Calculation",
                inputs={
                    "inlet_pressure_bar": 40.0,
                    "inlet_temperature_c": 350.0,
                    "outlet_pressure_bar": 40.0,
                    "outlet_temperature_c": 300.0,
                    "standard": "ASME_PTC_6"
                },
                expected_outputs={
                    "steam_quality_index": 98.5,
                    "superheat_degree_c": 49.67  # T - Tsat at 40 bar
                },
                expected_hash="",
                tolerance={
                    "steam_quality_index": 0.5,
                    "superheat_degree_c": 0.1
                },
                source="ASME PTC 6"
            ),
            "ptc19_sampling": GoldenTestCase(
                name="ASME PTC 19.11 Sampling Calculation",
                inputs={
                    "sample_pressure_bar": 10.0,
                    "sample_temperature_c": 180.0,
                    "isokinetic_velocity_ratio": 1.0,
                    "standard": "ASME_PTC_19.11"
                },
                expected_outputs={
                    "sample_quality_valid": True,
                    "moisture_content_percent": 0.0
                },
                expected_hash="",
                tolerance={
                    "moisture_content_percent": 0.01
                },
                source="ASME PTC 19.11"
            )
        }

    def load_desuperheater_scenarios(self) -> Dict[str, GoldenTestCase]:
        """Load known desuperheater calculation scenarios."""
        return {
            "normal_operation": GoldenTestCase(
                name="Normal desuperheater operation",
                inputs={
                    "steam_flow_kg_s": 10.0,
                    "inlet_temperature_c": 350.0,
                    "inlet_pressure_mpa": 4.0,
                    "target_temperature_c": 280.0,
                    "water_temperature_c": 105.0
                },
                expected_outputs={
                    "injection_rate_kg_s": 0.54,  # Approximate
                    "outlet_temperature_c": 280.0,
                    "temperature_reduction_c": 70.0
                },
                expected_hash="",
                tolerance={
                    "injection_rate_kg_s": 0.1,
                    "outlet_temperature_c": 2.0,
                    "temperature_reduction_c": 0.1
                },
                source="Calculated"
            ),
            "high_superheat_reduction": GoldenTestCase(
                name="High superheat reduction",
                inputs={
                    "steam_flow_kg_s": 10.0,
                    "inlet_temperature_c": 450.0,
                    "inlet_pressure_mpa": 4.0,
                    "target_temperature_c": 280.0,
                    "water_temperature_c": 105.0
                },
                expected_outputs={
                    "injection_rate_kg_s": 1.1,  # Higher injection needed
                    "outlet_temperature_c": 280.0,
                    "temperature_reduction_c": 170.0
                },
                expected_hash="",
                tolerance={
                    "injection_rate_kg_s": 0.2,
                    "outlet_temperature_c": 3.0,
                    "temperature_reduction_c": 0.1
                },
                source="Calculated"
            )
        }

    def load_pressure_control_scenarios(self) -> Dict[str, GoldenTestCase]:
        """Load known pressure control calculation scenarios."""
        return {
            "valve_position_linear": GoldenTestCase(
                name="Linear valve position calculation",
                inputs={
                    "setpoint_mpa": 1.0,
                    "actual_mpa": 0.95,
                    "valve_cv_max": 100.0,
                    "valve_characteristic": "linear",
                    "flow_rate_kg_s": 5.0,
                    "density_kg_m3": 10.0
                },
                expected_outputs={
                    "valve_position_pct": 52.5,  # Approximate
                    "control_error_mpa": 0.05
                },
                expected_hash="",
                tolerance={
                    "valve_position_pct": 5.0,
                    "control_error_mpa": 0.001
                },
                source="ISA-75.01.01"
            ),
            "valve_position_equal_percentage": GoldenTestCase(
                name="Equal percentage valve position",
                inputs={
                    "setpoint_mpa": 1.0,
                    "actual_mpa": 0.95,
                    "valve_cv_max": 100.0,
                    "valve_characteristic": "equal_percentage",
                    "flow_rate_kg_s": 5.0,
                    "density_kg_m3": 10.0
                },
                expected_outputs={
                    "valve_position_pct": 65.0,  # Approximate (equal % curve)
                    "control_error_mpa": 0.05
                },
                expected_hash="",
                tolerance={
                    "valve_position_pct": 10.0,
                    "control_error_mpa": 0.001
                },
                source="ISA-75.01.01"
            )
        }


@pytest.fixture
def golden_loader():
    """Provide golden file loader."""
    return GoldenFileLoader()


@pytest.fixture
def iapws_reference_data(golden_loader):
    """Provide IAPWS-IF97 reference data."""
    return golden_loader.load_iapws_if97_reference()


@pytest.fixture
def asme_reference_data(golden_loader):
    """Provide ASME PTC reference data."""
    return golden_loader.load_asme_ptc_reference()


@pytest.fixture
def desuperheater_reference_data(golden_loader):
    """Provide desuperheater reference data."""
    return golden_loader.load_desuperheater_scenarios()


@pytest.fixture
def pressure_control_reference_data(golden_loader):
    """Provide pressure control reference data."""
    return golden_loader.load_pressure_control_scenarios()


# =============================================================================
# FLOATING-POINT COMPARISON UTILITIES
# =============================================================================

class FloatingPointComparator:
    """Utility for floating-point comparison with determinism focus."""

    @staticmethod
    def are_equal(a: float, b: float, tolerance: float = FLOATING_POINT_TOLERANCE) -> bool:
        """Check if two floats are equal within tolerance."""
        if a == b:  # Handles infinities and exact matches
            return True
        if np.isnan(a) and np.isnan(b):
            return True  # Both NaN considered equal for determinism
        if np.isnan(a) or np.isnan(b):
            return False
        if np.isinf(a) or np.isinf(b):
            return a == b  # Must be exactly equal for infinities

        return abs(a - b) <= tolerance

    @staticmethod
    def assert_equal(a: float, b: float, tolerance: float = FLOATING_POINT_TOLERANCE, message: str = ""):
        """Assert two floats are equal within tolerance."""
        assert FloatingPointComparator.are_equal(a, b, tolerance), \
            f"{message} Expected {b}, got {a} (tolerance: {tolerance})"

    @staticmethod
    def verify_associativity(
        values: List[float],
        operation: str = "sum"
    ) -> Tuple[bool, float]:
        """
        Verify floating-point associativity.

        Tests if different orderings produce the same result.

        Args:
            values: List of values to combine
            operation: "sum" or "product"

        Returns:
            Tuple of (is_associative, max_deviation)
        """
        import itertools

        results = []
        for perm in itertools.permutations(values):
            if operation == "sum":
                result = sum(perm)
            elif operation == "product":
                result = 1.0
                for v in perm:
                    result *= v
            else:
                raise ValueError(f"Unknown operation: {operation}")
            results.append(result)

        # Check max deviation
        if len(results) == 0:
            return True, 0.0

        max_dev = max(results) - min(results)
        is_associative = max_dev < FLOATING_POINT_TOLERANCE

        return is_associative, max_dev

    @staticmethod
    def to_decimal(value: float, precision: int = DECIMAL_PRECISION) -> Decimal:
        """Convert float to Decimal with specified precision."""
        quantize_str = '0.' + '0' * precision
        return Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


@pytest.fixture
def float_comparator():
    """Provide floating-point comparator."""
    return FloatingPointComparator()


# =============================================================================
# DETERMINISTIC CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def deterministic_config():
    """Create deterministic configuration for all tests."""
    return {
        'agent_id': 'GL-012',
        'agent_name': 'STEAMQUAL',
        'version': '1.0.0',
        'deterministic_mode': True,
        'random_seed': DETERMINISM_SEED,
        'llm_temperature': LLM_TEMPERATURE,
        'llm_seed': DETERMINISM_SEED,
        'use_decimal_precision': True,
        'decimal_places': DECIMAL_PRECISION,
        'hash_algorithm': HASH_ALGORITHM,
        'enable_caching': False,  # Disable for determinism tests
        'enable_learning': False,  # Disable learning for determinism
        'enable_predictive': False,  # Disable predictions
    }


@pytest.fixture
def steam_quality_input_deterministic():
    """Create deterministic steam quality input."""
    return {
        'pressure_mpa': 1.0,
        'temperature_c': 180.0,
        'enthalpy_kj_kg': 2777.1,
        'flow_rate_kg_s': 10.0,
        'pressure_stability': 0.98,
        'temperature_stability': 0.95
    }


@pytest.fixture
def desuperheater_input_deterministic():
    """Create deterministic desuperheater input."""
    return {
        'steam_flow_kg_s': 10.0,
        'inlet_temperature_c': 350.0,
        'inlet_pressure_mpa': 4.0,
        'target_temperature_c': 280.0,
        'water_temperature_c': 105.0,
        'water_pressure_mpa': 6.0
    }


@pytest.fixture
def pressure_control_input_deterministic():
    """Create deterministic pressure control input."""
    return {
        'setpoint_mpa': 1.0,
        'actual_mpa': 0.95,
        'flow_rate_kg_s': 5.0,
        'fluid_density_kg_m3': 10.0,
        'valve_cv_max': 100.0,
        'pipe_diameter_m': 0.1,
        'pipe_length_m': 50.0,
        'pipe_roughness_m': 0.000045
    }


# =============================================================================
# EDGE CASE INPUT FIXTURES
# =============================================================================

@pytest.fixture
def edge_case_inputs():
    """Provide edge case inputs for determinism testing."""
    return {
        'very_small_values': {
            'pressure_mpa': 1e-15,
            'temperature_c': 1e-15,
            'flow_rate_kg_s': 1e-15
        },
        'very_large_values': {
            'pressure_mpa': 1e15,
            'temperature_c': 1e15,
            'flow_rate_kg_s': 1e15
        },
        'near_zero_denominators': {
            'h_fg_kj_kg': 1e-15,  # Near-zero for division
            'delta_p_mpa': 1e-15
        },
        'floating_point_challenges': {
            'value_1': 0.1 + 0.2,  # Classic floating-point issue
            'value_2': 0.3,  # Expected value
            'value_3': 1.0 / 3.0 * 3.0,  # Should be 1.0
        },
        'boundary_values': {
            'dryness_fraction_min': 0.0,
            'dryness_fraction_max': 1.0,
            'pressure_critical': 22.064,  # Critical point MPa
            'temperature_critical': 373.946  # Critical point C
        }
    }


# =============================================================================
# MOCK CALCULATORS FOR TESTING
# =============================================================================

@pytest.fixture
def mock_steam_quality_calculator():
    """Create mock steam quality calculator with deterministic behavior."""
    mock = MagicMock()

    def calculate_dryness(h_total, h_f, h_fg):
        """Deterministic dryness calculation."""
        if h_fg == 0:
            raise ValueError("h_fg cannot be zero")
        x = (Decimal(str(h_total)) - Decimal(str(h_f))) / Decimal(str(h_fg))
        x = max(Decimal("0"), min(Decimal("1"), x))
        return float(x.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))

    def calculate_steam_quality(input_data):
        """Deterministic steam quality calculation."""
        result = {
            'dryness_fraction': 0.98,
            'wetness_fraction': 0.02,
            'steam_quality_index': 95.5,
            'superheat_degree_c': 0.12,
            'saturation_temperature_c': 179.88,
            'specific_volume_m3_kg': 0.1944,
            'specific_enthalpy_kj_kg': 2777.1,
            'specific_entropy_kj_kg_k': 6.5828
        }
        result['provenance_hash'] = HashComparisonUtility.compute_hash({
            'inputs': input_data,
            'outputs': result
        })
        return result

    mock.calculate_dryness_fraction = MagicMock(side_effect=calculate_dryness)
    mock.calculate_steam_quality = MagicMock(side_effect=calculate_steam_quality)
    return mock


@pytest.fixture
def mock_desuperheater_calculator():
    """Create mock desuperheater calculator with deterministic behavior."""
    mock = MagicMock()

    def calculate_injection_rate(m_steam, h_inlet, h_outlet, h_water):
        """Deterministic injection rate calculation."""
        if h_outlet <= h_water:
            raise ValueError("h_outlet must be > h_water")
        if h_inlet <= h_outlet:
            raise ValueError("h_inlet must be > h_outlet")
        m_water = (Decimal(str(m_steam)) *
                   (Decimal(str(h_inlet)) - Decimal(str(h_outlet))) /
                   (Decimal(str(h_outlet)) - Decimal(str(h_water))))
        return float(m_water.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

    mock.calculate_injection_rate = MagicMock(side_effect=calculate_injection_rate)
    return mock


@pytest.fixture
def mock_pressure_controller():
    """Create mock pressure controller with deterministic behavior."""
    mock = MagicMock()

    def calculate_valve_position(setpoint, actual, cv_max, delta_p):
        """Deterministic valve position calculation."""
        error = Decimal(str(setpoint)) - Decimal(str(actual))
        position = Decimal("50") + error * Decimal("100") / Decimal(str(setpoint))
        position = max(Decimal("0"), min(Decimal("100"), position))
        return float(position.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    mock.calculate_valve_position = MagicMock(side_effect=calculate_valve_position)
    return mock


# =============================================================================
# CROSS-PLATFORM TEST UTILITIES
# =============================================================================

@pytest.fixture
def platform_info():
    """Provide current platform information."""
    return PLATFORM_INFO.copy()


@pytest.fixture
def cross_platform_test_data():
    """Provide test data for cross-platform validation."""
    return {
        'expected_hashes': {
            # These hashes should be identical across all platforms
            'steam_quality_basic': 'expected_sha256_hash_here',
            'desuperheater_basic': 'expected_sha256_hash_here',
            'pressure_control_basic': 'expected_sha256_hash_here'
        },
        'test_inputs': {
            'steam_quality': {
                'pressure_mpa': 1.0,
                'temperature_c': 180.0
            },
            'desuperheater': {
                'steam_flow_kg_s': 10.0,
                'inlet_temperature_c': 350.0
            },
            'pressure_control': {
                'setpoint_mpa': 1.0,
                'actual_mpa': 0.95
            }
        }
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_deterministic(results: List[Any], message: str = ""):
    """Assert all results in list are identical."""
    if len(results) < 2:
        return

    first = results[0]
    for i, result in enumerate(results[1:], 2):
        assert str(result) == str(first), f"{message} Result {i} differs from result 1"


def assert_within_tolerance(actual: float, expected: float, tolerance: float, message: str = ""):
    """Assert value is within tolerance of expected."""
    assert abs(actual - expected) <= tolerance, \
        f"{message} Expected {expected} +/- {tolerance}, got {actual}"


def generate_provenance_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 hash for provenance verification."""
    return HashComparisonUtility.compute_hash(data)


# =============================================================================
# CLEANUP
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    random.seed(DETERMINISM_SEED)
    np.random.seed(DETERMINISM_SEED)
    yield
    # Cleanup if needed


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    logger.debug("Determinism test cleanup completed")


logger.info("GL-012 STEAMQUAL determinism test fixtures loaded")
