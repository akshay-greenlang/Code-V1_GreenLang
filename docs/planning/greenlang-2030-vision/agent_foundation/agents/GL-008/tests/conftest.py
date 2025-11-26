# -*- coding: utf-8 -*-
"""
Shared test fixtures for GL-008 SteamTrapInspector test suite.

This module provides common fixtures, mock data generators, and test
configurations used across all test files.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    TrapInspectorConfig,
    TrapType,
    FailureMode,
    InspectionMethod,
    AcousticConfig,
    ThermalConfig,
    SteamTrapConfig
)
# Import without relative imports - tools.py uses relative imports internally
# We'll import after adding parent to path
try:
    from tools import (
        SteamTrapTools,
        AcousticAnalysisResult,
        ThermalAnalysisResult
    )
except ImportError:
    # Fallback: tools.py may not be importable in test environment
    # Tests will need to mock these
    SteamTrapTools = None
    AcousticAnalysisResult = None
    ThermalAnalysisResult = None


# ============================================================================
# CORE FIXTURES
# ============================================================================

@pytest.fixture
def base_config():
    """Fixture providing base test configuration."""
    return TrapInspectorConfig(
        agent_id="GL-008-TEST",
        enable_llm_classification=False,
        cache_ttl_seconds=60,
        max_concurrent_inspections=5,
        llm_temperature=0.0,
        llm_seed=42
    )


@pytest.fixture
def tools():
    """Fixture providing SteamTrapTools instance."""
    return SteamTrapTools()


@pytest.fixture
def acoustic_config():
    """Fixture providing acoustic analysis configuration."""
    return AcousticConfig(
        frequency_range_hz=(20000, 100000),
        sampling_rate_hz=250000,
        fft_window_size=2048,
        overlap_ratio=0.5,
        noise_floor_db=30.0,
        detection_threshold_db=45.0
    )


@pytest.fixture
def thermal_config():
    """Fixture providing thermal imaging configuration."""
    return ThermalConfig(
        image_resolution=(640, 480),
        temperature_range_c=(0, 250),
        emissivity=0.95,
        delta_t_threshold_c=10.0
    )


# ============================================================================
# ACOUSTIC SIGNAL GENERATORS
# ============================================================================

class AcousticSignalGenerator:
    """Generate synthetic acoustic signals for testing."""

    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        np.random.seed(seed)

    def generate_normal_signal(
        self,
        duration: float = 1.0,
        sampling_rate: int = 250000,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Generate normal operation acoustic signature."""
        n_samples = int(sampling_rate * duration)
        # Low amplitude white noise
        signal = np.random.randn(n_samples) * noise_level
        return signal

    def generate_failed_open_signal(
        self,
        duration: float = 1.0,
        sampling_rate: int = 250000,
        frequency: float = 30000.0,
        amplitude: float = 2.0
    ) -> np.ndarray:
        """Generate failed open acoustic signature (high freq, high amplitude)."""
        n_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_samples)
        # High frequency steam leak signature
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        # Add noise
        signal += np.random.randn(n_samples) * 0.2
        return signal

    def generate_failed_closed_signal(
        self,
        duration: float = 1.0,
        sampling_rate: int = 250000
    ) -> np.ndarray:
        """Generate failed closed signature (very low signal)."""
        n_samples = int(sampling_rate * duration)
        # Minimal signal - trap not passing condensate
        signal = np.random.randn(n_samples) * 0.02
        return signal

    def generate_leaking_signal(
        self,
        duration: float = 1.0,
        sampling_rate: int = 250000,
        leak_frequency: float = 25000.0
    ) -> np.ndarray:
        """Generate leaking trap signature (intermittent high frequency)."""
        n_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, n_samples)
        # Intermittent leak pattern
        signal = np.sin(2 * np.pi * leak_frequency * t) * np.abs(np.sin(2 * np.pi * 5 * t))
        signal += np.random.randn(n_samples) * 0.15
        return signal

    def generate_saturated_signal(
        self,
        duration: float = 1.0,
        sampling_rate: int = 250000
    ) -> np.ndarray:
        """Generate saturated signal (clipping)."""
        n_samples = int(sampling_rate * duration)
        signal = np.random.randn(n_samples) * 5.0
        # Clip to simulate saturation
        signal = np.clip(signal, -3.0, 3.0)
        return signal


@pytest.fixture
def signal_generator():
    """Fixture providing acoustic signal generator."""
    return AcousticSignalGenerator()


@pytest.fixture
def normal_acoustic_signal(signal_generator):
    """Generate normal operation acoustic signal."""
    return signal_generator.generate_normal_signal()


@pytest.fixture
def failed_open_signal(signal_generator):
    """Generate failed open acoustic signal."""
    return signal_generator.generate_failed_open_signal()


@pytest.fixture
def failed_closed_signal(signal_generator):
    """Generate failed closed acoustic signal."""
    return signal_generator.generate_failed_closed_signal()


# ============================================================================
# THERMAL DATA GENERATORS
# ============================================================================

class ThermalDataGenerator:
    """Generate synthetic thermal imaging data for testing."""

    def generate_normal_thermal(self) -> Dict[str, Any]:
        """Generate normal thermal signature."""
        return {
            'trap_id': 'TRAP-NORMAL-THERMAL',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,  # ΔT = 20°C (normal)
            'ambient_temp_c': 20.0
        }

    def generate_failed_open_thermal(self) -> Dict[str, Any]:
        """Generate failed open thermal signature (minimal ΔT)."""
        return {
            'trap_id': 'TRAP-FAILED-OPEN-THERMAL',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 148.0,  # ΔT = 2°C (steam bypass)
            'ambient_temp_c': 20.0
        }

    def generate_failed_closed_thermal(self) -> Dict[str, Any]:
        """Generate failed closed thermal signature (large ΔT)."""
        return {
            'trap_id': 'TRAP-FAILED-CLOSED-THERMAL',
            'temperature_upstream_c': 180.0,
            'temperature_downstream_c': 70.0,  # ΔT = 110°C (condensate backup)
            'ambient_temp_c': 20.0
        }

    def generate_cold_environment_thermal(self) -> Dict[str, Any]:
        """Generate thermal data in cold environment."""
        return {
            'trap_id': 'TRAP-COLD-ENV',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': -20.0  # Cold environment
        }

    def generate_hot_environment_thermal(self) -> Dict[str, Any]:
        """Generate thermal data in hot environment."""
        return {
            'trap_id': 'TRAP-HOT-ENV',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,
            'ambient_temp_c': 50.0  # Hot environment
        }


@pytest.fixture
def thermal_generator():
    """Fixture providing thermal data generator."""
    return ThermalDataGenerator()


@pytest.fixture
def normal_thermal_data(thermal_generator):
    """Generate normal thermal data."""
    return thermal_generator.generate_normal_thermal()


@pytest.fixture
def failed_open_thermal_data(thermal_generator):
    """Generate failed open thermal data."""
    return thermal_generator.generate_failed_open_thermal()


@pytest.fixture
def failed_closed_thermal_data(thermal_generator):
    """Generate failed closed thermal data."""
    return thermal_generator.generate_failed_closed_thermal()


# ============================================================================
# TRAP CONFIGURATION GENERATORS
# ============================================================================

@pytest.fixture
def trap_configs():
    """Generate various trap configurations for testing."""
    return {
        'thermodynamic': SteamTrapConfig(
            trap_id='TRAP-THERMO-001',
            trap_type=TrapType.THERMODYNAMIC,
            location='Building A - Level 1',
            process_criticality=8,
            steam_pressure_psig=100.0,
            expected_condensate_load_lb_hr=1000.0
        ),
        'thermostatic': SteamTrapConfig(
            trap_id='TRAP-THERMOSTAT-001',
            trap_type=TrapType.THERMOSTATIC,
            location='Building B - Level 2',
            process_criticality=6,
            steam_pressure_psig=75.0,
            expected_condensate_load_lb_hr=800.0
        ),
        'float': SteamTrapConfig(
            trap_id='TRAP-FLOAT-001',
            trap_type=TrapType.FLOAT_AND_THERMOSTATIC,
            location='Building C - Level 3',
            process_criticality=9,
            steam_pressure_psig=125.0,
            expected_condensate_load_lb_hr=1500.0
        ),
        'inverted_bucket': SteamTrapConfig(
            trap_id='TRAP-BUCKET-001',
            trap_type=TrapType.INVERTED_BUCKET,
            location='Building D - Level 1',
            process_criticality=7,
            steam_pressure_psig=90.0,
            expected_condensate_load_lb_hr=900.0
        )
    }


# ============================================================================
# ENERGY LOSS TEST DATA
# ============================================================================

@pytest.fixture
def energy_loss_test_data():
    """Known values for energy loss calculation validation."""
    return {
        'napier_equation_reference': {
            # W = 24.24 * P * D² * C
            # For P=100, D=0.125, C=0.7: W ≈ 26.51 lb/hr
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'discharge_coefficient': 0.7,
            'expected_steam_loss_lb_hr': 26.5125
        },
        'steam_table_reference': {
            # At 100 psig (114.7 psia), saturated steam:
            # Temperature: 338°F (170°C)
            # Enthalpy: 1187.5 BTU/lb
            'pressure_psig': 100.0,
            'expected_temperature_f': 338.0,
            'expected_enthalpy_btu_lb': 1187.5
        },
        'cost_scenarios': [
            {'steam_cost_usd_per_1000lb': 5.0, 'annual_loss_lb': 100000, 'expected_cost_usd': 500.0},
            {'steam_cost_usd_per_1000lb': 8.5, 'annual_loss_lb': 200000, 'expected_cost_usd': 1700.0},
            {'steam_cost_usd_per_1000lb': 12.0, 'annual_loss_lb': 150000, 'expected_cost_usd': 1800.0}
        ]
    }


# ============================================================================
# FLEET TEST DATA
# ============================================================================

@pytest.fixture
def test_fleet():
    """Generate test fleet for prioritization testing."""
    return [
        {
            'trap_id': 'TRAP-FLEET-001',
            'failure_mode': FailureMode.FAILED_OPEN,
            'energy_loss_usd_yr': 15000,
            'process_criticality': 9,
            'current_age_years': 10,
            'health_score': 25
        },
        {
            'trap_id': 'TRAP-FLEET-002',
            'failure_mode': FailureMode.LEAKING,
            'energy_loss_usd_yr': 3000,
            'process_criticality': 5,
            'current_age_years': 3,
            'health_score': 60
        },
        {
            'trap_id': 'TRAP-FLEET-003',
            'failure_mode': FailureMode.FAILED_CLOSED,
            'energy_loss_usd_yr': 8000,
            'process_criticality': 8,
            'current_age_years': 7,
            'health_score': 35
        },
        {
            'trap_id': 'TRAP-FLEET-004',
            'failure_mode': FailureMode.NORMAL,
            'energy_loss_usd_yr': 0,
            'process_criticality': 6,
            'current_age_years': 2,
            'health_score': 95
        },
        {
            'trap_id': 'TRAP-FLEET-005',
            'failure_mode': FailureMode.WORN_SEAT,
            'energy_loss_usd_yr': 5000,
            'process_criticality': 7,
            'current_age_years': 12,
            'health_score': 45
        }
    ]


# ============================================================================
# RUL PREDICTION TEST DATA
# ============================================================================

@pytest.fixture
def rul_test_data():
    """Test data for RUL prediction validation."""
    return {
        'weibull_parameters': {
            # Shape parameter (beta) = 2.5
            # Scale parameter (eta) = 2000 days
            'beta': 2.5,
            'eta': 2000,
            'expected_mean_life': 1770  # Approximate for beta=2.5
        },
        'historical_failures': {
            'trap_type_a': [1800, 2000, 2200, 1900, 2100],  # MTBF ≈ 2000 days
            'trap_type_b': [1200, 1400, 1300, 1500, 1250],  # MTBF ≈ 1330 days
            'trap_type_c': [2500, 2600, 2400, 2700, 2550]   # MTBF ≈ 2550 days
        },
        'degradation_scenarios': [
            {'age_days': 500, 'health_score': 90, 'expected_rul_days': 1500},
            {'age_days': 1000, 'health_score': 70, 'expected_rul_days': 800},
            {'age_days': 1500, 'health_score': 50, 'expected_rul_days': 400}
        ]
    }


# ============================================================================
# MOCK ANALYSIS RESULTS
# ============================================================================

@pytest.fixture
def mock_acoustic_result():
    """Create mock acoustic analysis result."""
    return AcousticAnalysisResult(
        trap_id='TRAP-MOCK-001',
        failure_probability=0.75,
        failure_mode=FailureMode.FAILED_OPEN,
        confidence_score=0.85,
        acoustic_signature={'fft_peaks': [30000, 45000]},
        anomaly_detected=True,
        signal_strength_db=65.0,
        frequency_peak_hz=32000,
        spectral_features={'bandwidth_hz': 5000, 'centroid_hz': 30000},
        timestamp=datetime.now().isoformat(),
        provenance_hash='a' * 64
    )


@pytest.fixture
def mock_thermal_result():
    """Create mock thermal analysis result."""
    return ThermalAnalysisResult(
        trap_id='TRAP-MOCK-001',
        trap_health_score=40.0,
        temperature_upstream_c=150.0,
        temperature_downstream_c=148.0,
        temperature_differential_c=2.0,
        anomalies_detected=['Minimal temperature differential'],
        hot_spots=[],
        cold_spots=[],
        thermal_signature={'pattern': 'uniform'},
        condensate_pooling_detected=False,
        timestamp=datetime.now().isoformat(),
        provenance_hash='b' * 64
    )


# ============================================================================
# PERFORMANCE TEST HELPERS
# ============================================================================

@pytest.fixture
def performance_test_config():
    """Configuration for performance benchmarks."""
    return {
        'max_execution_time_ms': 100,  # Max time for single analysis
        'batch_size': 1000,  # Records per batch
        'target_throughput': 100,  # Records/second
        'memory_limit_mb': 500  # Max memory increase
    }


# ============================================================================
# DETERMINISM TEST HELPERS
# ============================================================================

@pytest.fixture
def determinism_test_iterations():
    """Number of iterations for determinism testing."""
    return 10


@pytest.fixture
def provenance_validator():
    """Validator for provenance hash checking."""
    class ProvenanceValidator:
        @staticmethod
        def validate_hash(hash_str: str) -> bool:
            """Validate provenance hash format (SHA-256)."""
            return isinstance(hash_str, str) and len(hash_str) == 64 and all(c in '0123456789abcdef' for c in hash_str)

        @staticmethod
        def compare_hashes(hash1: str, hash2: str) -> bool:
            """Compare two provenance hashes for equality."""
            return hash1 == hash2

    return ProvenanceValidator()


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "edge_case: mark test as edge case validation"
    )
    config.addinivalue_line(
        "markers", "determinism: mark test as determinism validation"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "validation: mark test as calculation validation"
    )
