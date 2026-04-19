"""
GL-013 PredictiveMaintenance Test Suite - conftest.py

Comprehensive pytest fixtures for the GL-013 PREDICTMAINT agent.
Provides mock data, test helpers, and shared configurations.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
import hashlib
import uuid
import math


# =============================================================================
# Test Configuration
# =============================================================================

TEST_CONFIG = {
    "coverage_target": 0.85,
    "performance_threshold_ms": 100,
    "async_timeout_seconds": 30,
    "monte_carlo_iterations": 1000,
    "random_seed": 42,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SensorReading:
    """Represents a single sensor reading with metadata."""
    sensor_id: str
    sensor_type: str
    timestamp: datetime
    value: float
    unit: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetData:
    """Represents an industrial asset with its properties."""
    asset_id: str
    asset_type: str
    manufacturer: str
    model: str
    installation_date: datetime
    operating_hours: float
    rated_power_kw: float
    rated_speed_rpm: float
    bearing_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RULPrediction:
    """Represents a Remaining Useful Life prediction result."""
    prediction_id: str
    equipment_id: str
    timestamp: datetime
    rul_hours_mean: float
    rul_hours_p10: float
    rul_hours_p50: float
    rul_hours_p90: float
    confidence_score: float
    failure_mode: str
    recommended_action: str
    urgency: str
    provenance_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Sensor Reading Fixtures
# =============================================================================

@pytest.fixture
def sample_vibration_readings() -> List[SensorReading]:
    """Generate sample vibration sensor readings."""
    np.random.seed(TEST_CONFIG["random_seed"])
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)
    readings = []

    for i in range(100):
        reading = SensorReading(
            sensor_id="VIB-001",
            sensor_type="vibration_acceleration",
            timestamp=base_time + timedelta(seconds=i * 0.1),
            value=float(np.random.normal(0.35, 0.05)),
            unit="g",
            quality_score=0.98,
            metadata={"axis": "radial", "location": "drive_end"}
        )
        readings.append(reading)

    return readings


@pytest.fixture
def sample_temperature_readings() -> List[SensorReading]:
    """Generate sample temperature sensor readings."""
    np.random.seed(TEST_CONFIG["random_seed"])
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)
    readings = []

    for i in range(60):
        reading = SensorReading(
            sensor_id="TEMP-001",
            sensor_type="temperature",
            timestamp=base_time + timedelta(minutes=i),
            value=float(np.random.normal(65.0, 2.0)),
            unit="celsius",
            quality_score=0.99,
            metadata={"location": "bearing_housing"}
        )
        readings.append(reading)

    return readings


@pytest.fixture
def sample_current_readings() -> List[SensorReading]:
    """Generate sample motor current readings."""
    np.random.seed(TEST_CONFIG["random_seed"])
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)
    readings = []

    for i in range(100):
        reading = SensorReading(
            sensor_id="CURR-001",
            sensor_type="current",
            timestamp=base_time + timedelta(seconds=i * 0.1),
            value=float(np.random.normal(15.5, 0.3)),
            unit="amperes",
            quality_score=0.97,
            metadata={"phase": "A"}
        )
        readings.append(reading)

    return readings


@pytest.fixture
def sample_pressure_readings() -> List[SensorReading]:
    """Generate sample pressure sensor readings."""
    np.random.seed(TEST_CONFIG["random_seed"])
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)
    readings = []

    for i in range(30):
        reading = SensorReading(
            sensor_id="PRES-001",
            sensor_type="pressure",
            timestamp=base_time + timedelta(minutes=i * 2),
            value=float(np.random.normal(4.5, 0.1)),
            unit="bar",
            quality_score=0.96,
            metadata={"location": "discharge"}
        )
        readings.append(reading)

    return readings


# =============================================================================
# Asset Fixtures
# =============================================================================

@pytest.fixture
def sample_motor_asset() -> AssetData:
    """Create a sample motor asset."""
    return AssetData(
        asset_id="MOT-001",
        asset_type="motor_ac_induction",
        manufacturer="ABB",
        model="M3BP-315-SMC-4",
        installation_date=datetime(2020, 6, 15, tzinfo=timezone.utc),
        operating_hours=18500.0,
        rated_power_kw=160.0,
        rated_speed_rpm=1480.0,
        bearing_type="6316-2Z",
        metadata={
            "voltage_v": 400,
            "frequency_hz": 50,
            "poles": 4,
            "efficiency_class": "IE3"
        }
    )


@pytest.fixture
def sample_pump_asset() -> AssetData:
    """Create a sample pump asset."""
    return AssetData(
        asset_id="PMP-001",
        asset_type="centrifugal_pump",
        manufacturer="Grundfos",
        model="NB-100-250",
        installation_date=datetime(2019, 3, 20, tzinfo=timezone.utc),
        operating_hours=25000.0,
        rated_power_kw=45.0,
        rated_speed_rpm=2960.0,
        bearing_type="6205-2RS",
        metadata={
            "flow_rate_m3h": 100,
            "head_m": 50,
            "impeller_diameter_mm": 220
        }
    )


# =============================================================================
# Weibull Parameters Fixtures
# =============================================================================

@pytest.fixture
def sample_weibull_params() -> Dict[str, Dict[str, Any]]:
    """Provide Weibull distribution parameters for various equipment types."""
    return {
        "motor_ac_induction_large": {
            "beta": Decimal("2.5"),
            "eta": Decimal("87600"),
            "gamma": Decimal("0"),
            "description": "Large AC induction motor (>100kW)"
        },
        "motor_ac_induction_small": {
            "beta": Decimal("2.2"),
            "eta": Decimal("43800"),
            "gamma": Decimal("0"),
            "description": "Small AC induction motor (<100kW)"
        },
        "bearing_6205": {
            "beta": Decimal("1.5"),
            "eta": Decimal("35000"),
            "gamma": Decimal("1000"),
            "description": "6205 series ball bearing"
        },
        "pump_centrifugal": {
            "beta": Decimal("2.0"),
            "eta": Decimal("52560"),
            "gamma": Decimal("500"),
            "description": "Centrifugal pump"
        }
    }


@pytest.fixture
def sample_failure_data() -> Dict[str, Any]:
    """Provide sample failure time data for Weibull fitting."""
    np.random.seed(TEST_CONFIG["random_seed"])

    beta, eta = 2.5, 50000
    failure_times = eta * np.random.weibull(beta, 50)

    censoring_time = 40000
    censored = failure_times > censoring_time
    observed_times = np.where(censored, censoring_time, failure_times)

    return {
        "failure_times": observed_times.tolist(),
        "censored": censored.tolist(),
        "n_failures": int(np.sum(~censored)),
        "n_censored": int(np.sum(censored)),
        "true_beta": beta,
        "true_eta": eta
    }


# =============================================================================
# RUL Prediction Fixtures
# =============================================================================

@pytest.fixture
def sample_rul_prediction() -> RULPrediction:
    """Create a sample RUL prediction result."""
    prediction_id = str(uuid.uuid4())
    data_for_hash = f"{prediction_id}:MOT-001:3500:0.87"
    provenance_hash = hashlib.sha256(data_for_hash.encode()).hexdigest()

    return RULPrediction(
        prediction_id=prediction_id,
        equipment_id="MOT-001",
        timestamp=datetime.now(timezone.utc),
        rul_hours_mean=3500.0,
        rul_hours_p10=2100.0,
        rul_hours_p50=3400.0,
        rul_hours_p90=5200.0,
        confidence_score=0.87,
        failure_mode="bearing_wear",
        recommended_action="Schedule bearing replacement within 3 months",
        urgency="medium",
        provenance_hash=provenance_hash,
        metadata={
            "model_version": "1.2.0",
            "features_used": ["vibration_rms", "temperature_trend", "current_imbalance"]
        }
    )


# =============================================================================
# CMMS Integration Fixtures
# =============================================================================

@pytest.fixture
def mock_cmms_connector():
    """Create a mock CMMS connector with async methods."""
    connector = Mock()

    connector.health_check = AsyncMock(return_value={
        "status": "healthy",
        "latency_ms": 45,
        "version": "2.1.0"
    })

    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock(return_value=True)

    work_order_cache = {}

    async def create_work_order(request):
        key = f"{request[chr(39)+'equipment_id'+chr(39)]}:{request[chr(39)+'prediction_id'+chr(39)]}"
        if key not in work_order_cache:
            work_order_cache[key] = {
                "work_order_id": f"WO-{datetime.now().strftime(chr(39)+'%Y'+chr(39))}-{len(work_order_cache) + 1:03d}",
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        return work_order_cache[key]

    connector.create_work_order = AsyncMock(side_effect=create_work_order)

    connector.get_work_order = AsyncMock(return_value={
        "work_order_id": "WO-2024-001",
        "equipment_id": "MOT-001",
        "status": "pending",
        "priority": "medium",
        "description": "Predictive maintenance - bearing replacement",
        "created_at": "2024-01-15T10:30:00Z"
    })

    connector.list_work_orders = AsyncMock(return_value=[
        {"work_order_id": "WO-2024-001", "status": "pending", "equipment_id": "MOT-001"},
        {"work_order_id": "WO-2024-002", "status": "approved", "equipment_id": "PMP-001"},
        {"work_order_id": "WO-2024-003", "status": "completed", "equipment_id": "MOT-002"}
    ])

    connector.get_equipment = AsyncMock(return_value={
        "equipment_id": "MOT-001",
        "equipment_name": "Main Drive Motor",
        "status": "operational",
        "location": "Building A - Line 1",
        "criticality": "high"
    })

    return connector


@pytest.fixture
def mock_work_order_request() -> Dict[str, Any]:
    """Create a sample work order request."""
    return {
        "equipment_id": "MOT-001",
        "prediction_id": str(uuid.uuid4()),
        "priority": "medium",
        "confidence_score": 0.87,
        "remaining_useful_life_hours": 3500,
        "failure_mode": "bearing_wear",
        "recommended_action": "Replace drive-end bearing",
        "scheduled_start": datetime.now(timezone.utc) + timedelta(days=30),
        "due_date": datetime.now(timezone.utc) + timedelta(days=60),
        "estimated_duration_hours": 4,
        "parts_required": ["6316-2Z bearing", "Bearing grease"]
    }


# =============================================================================
# Data Quality Fixtures
# =============================================================================

@pytest.fixture
def sample_good_quality_data() -> Dict[str, Any]:
    """Provide sample data that passes all quality checks."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "values": [0.35, 0.37, 0.34, 0.36, 0.35, 0.38, 0.33, 0.35, 0.36, 0.34],
        "unit": "g",
        "sample_rate_hz": 10000,
        "quality_flags": {
            "sensor_healthy": True,
            "timestamp_valid": True,
            "values_in_range": True,
            "no_data_gaps": True
        }
    }


@pytest.fixture
def sample_bad_quality_data() -> Dict[str, Any]:
    """Provide sample data that fails quality checks."""
    return {
        "timestamp": "invalid-timestamp-format",
        "values": [0.35, None, 0.34, float("nan"), 0.35, None, 0.33, 0.35, 999.99, 0.34],
        "unit": "invalid_unit",
        "sample_rate_hz": -100,
        "quality_flags": {
            "sensor_healthy": False,
            "timestamp_valid": False,
            "values_in_range": False,
            "no_data_gaps": False
        }
    }


# =============================================================================
# Explainability Fixtures
# =============================================================================

@pytest.fixture
def sample_shap_values() -> Dict[str, Any]:
    """Provide sample SHAP values for explainability testing."""
    feature_names = [
        "vibration_rms",
        "temperature_mean",
        "current_imbalance",
        "operating_hours",
        "bearing_age_factor"
    ]
    shap_values = [-450.0, -280.0, 120.0, -180.0, 90.0]
    feature_values = [0.45, 72.5, 0.03, 18500, 0.65]
    base_value = 5000.0
    prediction = base_value + sum(shap_values)

    return {
        "feature_names": feature_names,
        "shap_values": shap_values,
        "feature_values": feature_values,
        "base_value": base_value,
        "prediction": prediction
    }


@pytest.fixture
def sample_lime_explanation() -> Dict[str, Any]:
    """Provide sample LIME explanation for local interpretability."""
    return {
        "prediction_id": str(uuid.uuid4()),
        "local_model_score": 0.92,
        "feature_weights": [
            ("vibration_rms > 0.4", -0.35),
            ("temperature_mean > 70", -0.22),
            ("operating_hours > 15000", -0.18),
            ("current_imbalance < 0.05", 0.08),
            ("bearing_age_factor > 0.5", -0.12)
        ],
        "intercept": 4500.0,
        "num_features": 5,
        "neighborhood_size": 5000
    }


@pytest.fixture
def sample_causal_graph() -> Dict[str, Any]:
    """Provide sample causal graph for failure mode analysis."""
    return {
        "nodes": [
            {"id": "vibration", "type": "symptom", "label": "Increased Vibration"},
            {"id": "temperature", "type": "symptom", "label": "Elevated Temperature"},
            {"id": "bearing_wear", "type": "cause", "label": "Bearing Wear"},
            {"id": "misalignment", "type": "cause", "label": "Shaft Misalignment"},
            {"id": "lubrication", "type": "cause", "label": "Inadequate Lubrication"},
            {"id": "failure", "type": "outcome", "label": "Equipment Failure"}
        ],
        "edges": [
            {"source": "bearing_wear", "target": "vibration", "weight": 0.85},
            {"source": "bearing_wear", "target": "temperature", "weight": 0.72},
            {"source": "misalignment", "target": "vibration", "weight": 0.78},
            {"source": "lubrication", "target": "bearing_wear", "weight": 0.65},
            {"source": "lubrication", "target": "temperature", "weight": 0.58},
            {"source": "vibration", "target": "failure", "weight": 0.45},
            {"source": "temperature", "target": "failure", "weight": 0.38}
        ]
    }


# =============================================================================
# Bearing and Vibration Analysis Fixtures
# =============================================================================

@pytest.fixture
def sample_bearing_6205() -> Dict[str, Any]:
    """Provide sample 6205 bearing geometry parameters."""
    return {
        "bearing_type": "6205-2RS",
        "inner_diameter_mm": 25.0,
        "outer_diameter_mm": 52.0,
        "pitch_diameter_mm": 38.5,
        "ball_diameter_mm": 7.94,
        "num_rolling_elements": 9,
        "contact_angle_deg": 0.0
    }


@pytest.fixture
def sample_bearing_frequencies_6205(sample_bearing_6205) -> Dict[str, float]:
    """Calculate bearing fault frequencies for 6205 bearing at 1480 RPM."""
    shaft_speed_hz = 1480 / 60
    n = sample_bearing_6205["num_rolling_elements"]
    bd = sample_bearing_6205["ball_diameter_mm"]
    pd = sample_bearing_6205["pitch_diameter_mm"]
    theta = math.radians(sample_bearing_6205["contact_angle_deg"])

    ftf = (shaft_speed_hz / 2) * (1 - (bd / pd) * math.cos(theta))
    bpfo = (n / 2) * shaft_speed_hz * (1 - (bd / pd) * math.cos(theta))
    bpfi = (n / 2) * shaft_speed_hz * (1 + (bd / pd) * math.cos(theta))
    bsf = (pd / (2 * bd)) * shaft_speed_hz * (1 - ((bd / pd) * math.cos(theta)) ** 2)

    return {
        "shaft_frequency_hz": shaft_speed_hz,
        "ftf_hz": ftf,
        "bpfo_hz": bpfo,
        "bpfi_hz": bpfi,
        "bsf_hz": bsf
    }


@pytest.fixture
def sample_vibration_limits_class_ii() -> Dict[str, Dict[str, float]]:
    """Provide ISO 10816-3 vibration limits for Class II machines."""
    return {
        "zone_A": {"max_velocity_mm_s": 2.8},
        "zone_B": {"max_velocity_mm_s": 7.1},
        "zone_C": {"max_velocity_mm_s": 11.2},
        "zone_D": {"min_velocity_mm_s": 11.2}
    }


@pytest.fixture
def sample_vibration_fft_data() -> Dict[str, Any]:
    """Provide sample FFT analysis results."""
    return {
        "shaft_frequency_hz": 24.67,
        "peak_1x": 0.35,
        "peak_2x": 0.12,
        "peak_3x": 0.05,
        "bpfo_amplitude": 0.08,
        "bpfi_amplitude": 0.04,
        "overall_rms": 0.42,
        "crest_factor": 3.8,
        "kurtosis": 3.2
    }


# =============================================================================
# Mock Calculators and Analyzers
# =============================================================================

@pytest.fixture
def mock_weibull_calculator():
    """Create a mock Weibull calculator."""
    calculator = Mock()

    calculator.fit = Mock(return_value={
        "beta": 2.5,
        "eta": 50000,
        "gamma": 0,
        "log_likelihood": -245.6,
        "aic": 495.2
    })

    calculator.survival_probability = Mock(side_effect=lambda t, beta, eta, gamma=0:
        math.exp(-((t - gamma) / eta) ** beta) if t > gamma else 1.0
    )

    calculator.hazard_rate = Mock(side_effect=lambda t, beta, eta, gamma=0:
        (beta / eta) * ((t - gamma) / eta) ** (beta - 1) if t > gamma else 0.0
    )

    calculator.percentile = Mock(side_effect=lambda p, beta, eta, gamma=0:
        gamma + eta * (-math.log(1 - p)) ** (1 / beta)
    )

    return calculator


@pytest.fixture
def mock_rul_calculator():
    """Create a mock RUL calculator."""
    calculator = Mock()

    def calculate_rul():
        return {
            "rul_mean_hours": 3500.0,
            "rul_p10_hours": 2100.0,
            "rul_p50_hours": 3400.0,
            "rul_p90_hours": 5200.0,
            "confidence_score": 0.87,
            "failure_mode": "bearing_wear",
            "recommended_action": "Schedule bearing replacement"
        }

    calculator.calculate_rul = Mock(side_effect=calculate_rul)

    return calculator


@pytest.fixture
def mock_vibration_analyzer():
    """Create a mock vibration analyzer."""
    analyzer = Mock()

    analyzer.compute_fft = Mock(return_value={
        "frequencies": np.linspace(0, 1000, 1024).tolist(),
        "amplitudes": (np.random.random(1024) * 0.1).tolist(),
        "peak_frequency": 24.67,
        "peak_amplitude": 0.35
    })

    analyzer.extract_features = Mock(return_value={
        "rms": 0.42,
        "peak": 1.2,
        "crest_factor": 2.86,
        "kurtosis": 3.2,
        "skewness": 0.1
    })

    analyzer.detect_bearing_faults = Mock(return_value={
        "bpfo_detected": False,
        "bpfi_detected": False,
        "bsf_detected": False,
        "ftf_detected": False,
        "overall_severity": "normal"
    })

    return analyzer


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_timer():
    """Create a performance timer context manager."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = None

        def __enter__(self):
            self.start_time = datetime.now()
            return self

        def __exit__(self, *args):
            self.end_time = datetime.now()
            self.elapsed_ms = (self.end_time - self.start_time).total_seconds() * 1000

        def assert_under(self, max_ms: float):
            assert self.elapsed_ms is not None, "Timer not used in context"
            assert self.elapsed_ms < max_ms, f"Elapsed {self.elapsed_ms:.2f}ms exceeds {max_ms}ms"

    return Timer


# =============================================================================
# Pytest Hooks and Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "compliance: marks tests as compliance tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(TEST_CONFIG["random_seed"])
