# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Determinism Test Fixtures.

Provides comprehensive fixtures for determinism validation tests including:
- Fuel composition fixtures (natural gas, fuel oil, biogas blends)
- Golden file loaders for ISO/ASTM reference data
- Hash comparison utilities
- Multi-run test decorators
- Deterministic configuration fixtures
- Floating-point comparison utilities

Zero-hallucination: All fixtures ensure deterministic behavior.

Author: GL-DeterminismAuditor
Version: 1.0.0
"""

import asyncio
import functools
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pytest

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
FLOATING_POINT_TOLERANCE = 1e-10
DECIMAL_PRECISION = 10


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers for determinism tests."""
    config.addinivalue_line("markers", "determinism: Determinism validation tests")
    config.addinivalue_line("markers", "reproducibility: Multi-run reproducibility tests")
    config.addinivalue_line("markers", "golden: Golden file validation tests")
    config.addinivalue_line("markers", "hash_consistency: Hash consistency tests")
    config.addinivalue_line("markers", "heating_value: Calorific value determinism tests")
    config.addinivalue_line("markers", "stoichiometry: Combustion stoichiometry tests")
    config.addinivalue_line("markers", "emission_factor: Emission factor determinism tests")
    config.addinivalue_line("markers", "blending: Fuel blending determinism tests")
    config.addinivalue_line("markers", "iso_6976: ISO 6976 standard validation")
    config.addinivalue_line("markers", "ipcc: IPCC emission factor validation")


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
class FuelDeterminismTestResult:
    """Result of a fuel determinism test run."""
    run_number: int
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    provenance_hash: str
    execution_time_ms: float
    heating_value_hhv: float
    heating_value_lhv: float
    emission_factor_co2: float


@dataclass
class GoldenFuelTestCase:
    """Golden test case with known expected fuel values."""
    name: str
    fuel_type: str
    composition: Dict[str, float]
    expected_hhv_mj_kg: float
    expected_lhv_mj_kg: float
    expected_co2_kg_gj: float
    tolerance: Dict[str, float] = field(default_factory=dict)
    source: str = "ISO_6976"


# =============================================================================
# HASH COMPARISON UTILITIES
# =============================================================================

class FuelHashUtility:
    """Utility class for fuel calculation hash comparison."""

    @staticmethod
    def compute_hash(data: Any, algorithm: str = HASH_ALGORITHM) -> str:
        """Compute deterministic hash of fuel data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    @staticmethod
    def verify_hash_consistency(hashes: List[str]) -> Tuple[bool, Optional[str]]:
        """Verify all hashes are identical."""
        if not hashes:
            return True, None
        first_hash = hashes[0]
        for i, h in enumerate(hashes[1:], 2):
            if h != first_hash:
                return False, f"Hash {i} differs: {h[:16]}... vs {first_hash[:16]}..."
        return True, None

    @staticmethod
    def assert_hashes_identical(hashes: List[str], message: str = ""):
        """Assert all hashes are identical."""
        is_consistent, diff = FuelHashUtility.verify_hash_consistency(hashes)
        assert is_consistent, f"{message} {diff}"


@pytest.fixture
def hash_utility():
    """Provide fuel hash comparison utility."""
    return FuelHashUtility()


# =============================================================================
# FUEL COMPOSITION FIXTURES
# =============================================================================

@pytest.fixture
def natural_gas_composition():
    """Standard natural gas composition (pipeline quality)."""
    return {
        'methane': 95.0,
        'ethane': 2.5,
        'propane': 0.5,
        'n_butane': 0.1,
        'nitrogen': 1.5,
        'carbon_dioxide': 0.4
    }


@pytest.fixture
def biogas_composition():
    """Standard biogas composition from anaerobic digestion."""
    return {
        'methane': 60.0,
        'carbon_dioxide': 35.0,
        'nitrogen': 3.0,
        'hydrogen_sulfide': 0.5,
        'oxygen': 1.0,
        'water_vapor': 0.5
    }


@pytest.fixture
def fuel_oil_composition():
    """Fuel oil No. 2 typical composition (ultimate analysis)."""
    return {
        'carbon': 87.0,
        'hydrogen': 12.5,
        'sulfur': 0.3,
        'nitrogen': 0.01,
        'oxygen': 0.1,
        'ash': 0.01,
        'density_kg_m3': 850.0
    }


@pytest.fixture
def coal_ultimate_analysis():
    """Bituminous coal ultimate analysis (dry basis)."""
    return {
        'carbon': 75.0,
        'hydrogen': 5.0,
        'oxygen': 8.0,
        'nitrogen': 1.5,
        'sulfur': 2.0,
        'ash': 8.5,
        'moisture': 8.0
    }


@pytest.fixture
def biomass_wood_pellet_composition():
    """Wood pellet composition (ENplus A1 grade)."""
    return {
        'carbon': 50.0,
        'hydrogen': 6.2,
        'oxygen': 43.0,
        'nitrogen': 0.3,
        'sulfur': 0.02,
        'ash': 0.5,
        'moisture': 8.0
    }


@pytest.fixture
def hydrogen_fuel_composition():
    """Green hydrogen fuel composition."""
    return {
        'hydrogen': 99.97,
        'nitrogen': 0.02,
        'oxygen': 0.01
    }


# =============================================================================
# GOLDEN TEST CASE FIXTURES
# =============================================================================

@pytest.fixture
def iso_6976_golden_cases() -> List[GoldenFuelTestCase]:
    """ISO 6976:2016 reference cases for natural gas calorific value."""
    return [
        GoldenFuelTestCase(
            name="Pure methane",
            fuel_type="natural_gas",
            composition={'methane': 100.0},
            expected_hhv_mj_kg=55.53,
            expected_lhv_mj_kg=50.01,
            expected_co2_kg_gj=56.1,
            tolerance={'hhv': 0.1, 'lhv': 0.1, 'co2': 0.5},
            source="ISO_6976_Table_4"
        ),
        GoldenFuelTestCase(
            name="Pipeline natural gas",
            fuel_type="natural_gas",
            composition={'methane': 95.0, 'ethane': 2.5, 'propane': 0.5, 'nitrogen': 2.0},
            expected_hhv_mj_kg=52.8,
            expected_lhv_mj_kg=47.6,
            expected_co2_kg_gj=56.1,
            tolerance={'hhv': 0.5, 'lhv': 0.5, 'co2': 1.0},
            source="ISO_6976_Calculated"
        ),
        GoldenFuelTestCase(
            name="Rich natural gas",
            fuel_type="natural_gas",
            composition={'methane': 85.0, 'ethane': 8.0, 'propane': 4.0, 'n_butane': 2.0, 'nitrogen': 1.0},
            expected_hhv_mj_kg=51.2,
            expected_lhv_mj_kg=46.5,
            expected_co2_kg_gj=58.0,
            tolerance={'hhv': 0.5, 'lhv': 0.5, 'co2': 1.0},
            source="ISO_6976_Calculated"
        )
    ]


@pytest.fixture
def ipcc_emission_factor_golden_cases() -> List[GoldenFuelTestCase]:
    """IPCC 2006 emission factor reference cases."""
    return [
        GoldenFuelTestCase(
            name="Natural gas IPCC default",
            fuel_type="natural_gas",
            composition={},
            expected_hhv_mj_kg=50.0,
            expected_lhv_mj_kg=45.0,
            expected_co2_kg_gj=56.1,
            tolerance={'co2': 0.1},
            source="IPCC_2006_Vol2_Table_2.2"
        ),
        GoldenFuelTestCase(
            name="Bituminous coal IPCC default",
            fuel_type="coal",
            composition={},
            expected_hhv_mj_kg=26.0,
            expected_lhv_mj_kg=25.0,
            expected_co2_kg_gj=94.6,
            tolerance={'co2': 0.1},
            source="IPCC_2006_Vol2_Table_2.2"
        ),
        GoldenFuelTestCase(
            name="Fuel oil IPCC default",
            fuel_type="fuel_oil",
            composition={},
            expected_hhv_mj_kg=43.0,
            expected_lhv_mj_kg=40.4,
            expected_co2_kg_gj=77.4,
            tolerance={'co2': 0.1},
            source="IPCC_2006_Vol2_Table_2.2"
        )
    ]


# =============================================================================
# DETERMINISTIC CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def deterministic_config():
    """Create deterministic configuration for all tests."""
    return {
        'agent_id': 'GL-011',
        'agent_name': 'FUELCRAFT',
        'version': '1.0.0',
        'deterministic_mode': True,
        'random_seed': DETERMINISM_SEED,
        'llm_temperature': LLM_TEMPERATURE,
        'llm_seed': DETERMINISM_SEED,
        'use_decimal_precision': True,
        'decimal_places': DECIMAL_PRECISION,
        'hash_algorithm': HASH_ALGORITHM,
        'enable_caching': False,
        'enable_learning': False,
        'enable_predictive': False,
    }


@pytest.fixture
def calorific_value_input_deterministic(natural_gas_composition):
    """Create deterministic calorific value calculation input."""
    return {
        'fuel_type': 'natural_gas',
        'composition': natural_gas_composition,
        'temperature_c': 15.0,
        'pressure_kpa': 101.325,
        'moisture_percent': 0.0
    }


@pytest.fixture
def emission_factor_input_deterministic():
    """Create deterministic emission factor lookup input."""
    return {
        'fuel_type': 'natural_gas',
        'combustion_technology': 'boiler',
        'emission_control': 'uncontrolled',
        'region': 'default'
    }


@pytest.fixture
def fuel_blending_input_deterministic():
    """Create deterministic fuel blending input."""
    return {
        'available_fuels': ['coal', 'biomass'],
        'target_heating_value_mj_kg': 22.0,
        'max_moisture_percent': 15.0,
        'max_ash_percent': 12.0,
        'max_sulfur_percent': 1.5,
        'optimization_objective': 'balanced'
    }


# =============================================================================
# MULTI-RUN TEST DECORATORS
# =============================================================================

F = TypeVar('F', bound=Callable)


def run_fuel_calculation_multiple_times(runs: int = DEFAULT_RUNS):
    """Decorator to run fuel calculation multiple times and verify determinism."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            hashes = []
            for _ in range(runs):
                random.seed(DETERMINISM_SEED)
                result = func(*args, **kwargs)
                results.append(result)
                if isinstance(result, dict) and 'provenance_hash' in result:
                    hashes.append(result['provenance_hash'])
                else:
                    hashes.append(FuelHashUtility.compute_hash(result))
            unique_results = len(set(str(r) for r in results))
            assert unique_results == 1, f"Non-deterministic: {unique_results} unique results from {runs} runs"
            unique_hashes = len(set(hashes))
            assert unique_hashes == 1, f"Hash inconsistency: {unique_hashes} unique hashes from {runs} runs"
            return results[0]
        return wrapper
    return decorator


# =============================================================================
# FLOATING-POINT COMPARISON UTILITIES
# =============================================================================

class FuelFloatComparator:
    """Utility for floating-point comparison in fuel calculations."""

    @staticmethod
    def are_equal(a: float, b: float, tolerance: float = FLOATING_POINT_TOLERANCE) -> bool:
        """Check if two floats are equal within tolerance."""
        if a == b:
            return True
        return abs(a - b) <= tolerance

    @staticmethod
    def to_decimal(value: float, precision: int = DECIMAL_PRECISION) -> Decimal:
        """Convert float to Decimal with specified precision."""
        quantize_str = '0.' + '0' * precision
        return Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


@pytest.fixture
def float_comparator():
    """Provide floating-point comparator."""
    return FuelFloatComparator()


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
    return FuelHashUtility.compute_hash(data)


# =============================================================================
# CLEANUP
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    random.seed(DETERMINISM_SEED)
    yield


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    logger.debug("Determinism test cleanup completed")


logger.info("GL-011 FUELCRAFT determinism test fixtures loaded")
