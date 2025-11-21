# -*- coding: utf-8 -*-
"""
Shared fixtures and test utilities for GL-006 unit tests.

Provides mock objects, test data generators, and common fixtures
for testing the HeatRecoveryMaximizer agent components.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import random
import numpy as np
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_random


# Test configuration
TEST_SEED = 42
TEST_TIMEOUT = 5.0  # seconds
PERFORMANCE_TARGET_MS = 100.0  # milliseconds


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducibility."""
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)


@pytest.fixture
def mock_config():
    """Create mock agent configuration."""
    return {
        "name": "GL-006-HeatRecoveryMaximizer",
        "version": "1.0.0",
        "environment": "test",
        "min_temperature_approach": 10.0,  # °C
        "max_pressure_drop": 50.0,  # kPa
        "min_heat_recovery": 0.65,  # 65%
        "fouling_threshold": 0.85,  # 85% of design
        "optimization_interval": 3600,  # seconds
        "thermal_camera_resolution": (640, 480),
        "anomaly_detection_threshold": 3.0,  # standard deviations
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "heat_recovery_test"
        },
        "mqtt": {
            "broker": "localhost",
            "port": 1883,
            "topics": {
                "temperature": "sensors/temperature/+",
                "pressure": "sensors/pressure/+",
                "flow": "sensors/flow/+"
            }
        }
    }


@pytest.fixture
def mock_heat_exchanger_data():
    """Generate mock heat exchanger performance data."""
    return {
        "exchanger_id": "HX-001",
        "type": "shell_and_tube",
        "hot_inlet_temp": 150.0,  # °C
        "hot_outlet_temp": 90.0,
        "cold_inlet_temp": 30.0,
        "cold_outlet_temp": 80.0,
        "hot_flow_rate": 10.5,  # kg/s
        "cold_flow_rate": 12.0,
        "pressure_drop_hot": 25.0,  # kPa
        "pressure_drop_cold": 30.0,
        "effectiveness": 0.72,
        "fouling_factor": 0.0002,  # m²·K/W
        "design_duty": 2500.0,  # kW
        "actual_duty": 2100.0,
        "timestamp": DeterministicClock.now().isoformat()
    }


@pytest.fixture
def mock_economizer_data():
    """Generate mock economizer performance data."""
    return {
        "economizer_id": "ECO-001",
        "type": "air_to_air",
        "exhaust_temp": 180.0,  # °C
        "supply_temp": 20.0,
        "exhaust_flow": 5000.0,  # m³/h
        "supply_flow": 4800.0,
        "effectiveness": 0.68,
        "pressure_drop": 150.0,  # Pa
        "bypass_damper_position": 25.0,  # %
        "frost_protection_active": False,
        "timestamp": DeterministicClock.now().isoformat()
    }


@pytest.fixture
def mock_stream_data():
    """Generate mock process stream data for pinch analysis."""
    hot_streams = [
        {
            "stream_id": "H1",
            "name": "Reactor Outlet",
            "supply_temp": 180.0,
            "target_temp": 60.0,
            "heat_capacity_rate": 10.0,  # kW/K
            "heat_load": 1200.0  # kW
        },
        {
            "stream_id": "H2",
            "name": "Distillation Overhead",
            "supply_temp": 150.0,
            "target_temp": 40.0,
            "heat_capacity_rate": 8.0,
            "heat_load": 880.0
        },
        {
            "stream_id": "H3",
            "name": "Compressor Discharge",
            "supply_temp": 120.0,
            "target_temp": 35.0,
            "heat_capacity_rate": 6.0,
            "heat_load": 510.0
        }
    ]

    cold_streams = [
        {
            "stream_id": "C1",
            "name": "Feed Preheater",
            "supply_temp": 20.0,
            "target_temp": 135.0,
            "heat_capacity_rate": 7.5,
            "heat_load": 862.5
        },
        {
            "stream_id": "C2",
            "name": "Reboiler Feed",
            "supply_temp": 80.0,
            "target_temp": 140.0,
            "heat_capacity_rate": 12.0,
            "heat_load": 720.0
        }
    ]

    return {
        "hot_streams": hot_streams,
        "cold_streams": cold_streams,
        "utilities": {
            "hot_utility_temp": 250.0,
            "cold_utility_temp": 10.0,
            "hot_utility_cost": 120.0,  # $/MWh
            "cold_utility_cost": 20.0
        }
    }


@pytest.fixture
def mock_thermal_image_data():
    """Generate mock thermal imaging data."""
    # Create synthetic thermal image (640x480)
    base_temp = 50.0
    image = np.random.normal(base_temp, 5.0, (480, 640))

    # Add some hot spots
    hot_spots = [
        {"x": 100, "y": 100, "radius": 20, "temp": 85.0},
        {"x": 400, "y": 300, "radius": 15, "temp": 75.0},
        {"x": 550, "y": 200, "radius": 25, "temp": 90.0}
    ]

    for spot in hot_spots:
        y, x = np.ogrid[:480, :640]
        mask = (x - spot["x"])**2 + (y - spot["y"])**2 <= spot["radius"]**2
        image[mask] = spot["temp"]

    return {
        "image": image,
        "timestamp": DeterministicClock.now().isoformat(),
        "camera_id": "FLIR-001",
        "location": "Heat Exchanger Bank A",
        "emissivity": 0.95,
        "ambient_temp": 25.0,
        "hot_spots": hot_spots
    }


@pytest.fixture
def mock_pi_historian():
    """Mock PI Historian interface."""
    mock = Mock()
    mock.get_tag_value = Mock(return_value={"value": 75.5, "timestamp": DeterministicClock.now()})
    mock.get_tag_history = Mock(return_value=[
        {"value": 75.0 + i * 0.1, "timestamp": DeterministicClock.now() - timedelta(minutes=i)}
        for i in range(60)
    ])
    mock.write_tag_value = Mock(return_value=True)
    return mock


@pytest.fixture
def mock_mqtt_client():
    """Mock MQTT client for sensor data."""
    client = Mock()
    client.connect = Mock(return_value=(0, None))
    client.subscribe = Mock(return_value=(0, 1))
    client.publish = Mock(return_value=(0, 2))
    client.on_message = None
    client.loop_start = Mock()
    client.loop_stop = Mock()
    return client


@pytest.fixture
def mock_database():
    """Mock database connection."""
    db = Mock()
    db.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[])))
    db.commit = Mock()
    db.rollback = Mock()
    return db


@pytest.fixture
def performance_benchmark():
    """Fixture for performance testing."""
    import time

    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()
            return (self.end_time - self.start_time) * 1000  # Convert to ms

        def assert_under_target(self, target_ms=PERFORMANCE_TARGET_MS):
            duration_ms = self.stop()
            assert duration_ms < target_ms, f"Performance target failed: {duration_ms:.2f}ms > {target_ms}ms"

    return PerformanceBenchmark()


@pytest.fixture
def validation_helpers():
    """Helper functions for validation testing."""
    class ValidationHelpers:
        @staticmethod
        def is_valid_temperature(temp: float, min_val: float = -273.15, max_val: float = 1000.0) -> bool:
            """Validate temperature is within reasonable bounds."""
            return min_val <= temp <= max_val

        @staticmethod
        def is_valid_effectiveness(eff: float) -> bool:
            """Validate heat exchanger effectiveness."""
            return 0.0 <= eff <= 1.0

        @staticmethod
        def is_valid_pressure_drop(dp: float, max_dp: float = 100.0) -> bool:
            """Validate pressure drop is acceptable."""
            return 0.0 <= dp <= max_dp

        @staticmethod
        def validate_energy_balance(hot_duty: float, cold_duty: float, tolerance: float = 0.05) -> bool:
            """Validate energy balance within tolerance."""
            if hot_duty == 0:
                return False
            error = abs((hot_duty - cold_duty) / hot_duty)
            return error <= tolerance

    return ValidationHelpers()


@pytest.fixture
def async_mock_factory():
    """Factory for creating async mocks."""
    def create_async_mock(return_value=None, side_effect=None):
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock

    return create_async_mock


# Test data generators
class TestDataGenerator:
    """Generate realistic test data for heat recovery systems."""

    @staticmethod
    def generate_temperature_series(base_temp: float, duration_hours: int = 24,
                                  noise_level: float = 2.0) -> List[Dict[str, Any]]:
        """Generate time series temperature data."""
        data = []
        for i in range(duration_hours * 60):  # Minute resolution
            timestamp = DeterministicClock.now() - timedelta(minutes=i)
            temp = base_temp + noise_level * np.sin(i * 0.1) + np.random.normal(0, noise_level * 0.3)
            data.append({
                "timestamp": timestamp.isoformat(),
                "value": round(temp, 2),
                "quality": "Good" if deterministic_random().random() > 0.05 else "Uncertain"
            })
        return data

    @staticmethod
    def generate_exchanger_network(num_exchangers: int = 5) -> List[Dict[str, Any]]:
        """Generate a network of heat exchangers."""
        types = ["shell_and_tube", "plate", "spiral", "air_cooled"]
        exchangers = []

        for i in range(num_exchangers):
            exchangers.append({
                "id": f"HX-{i+1:03d}",
                "type": deterministic_random().choice(types),
                "design_duty": random.uniform(500, 5000),  # kW
                "area": random.uniform(10, 200),  # m²
                "hot_side_fluid": deterministic_random().choice(["steam", "oil", "water", "glycol"]),
                "cold_side_fluid": deterministic_random().choice(["water", "air", "process_fluid"]),
                "age_years": random.uniform(0, 15),
                "last_cleaned": DeterministicClock.now() - timedelta(days=deterministic_random().randint(30, 365))
            })

        return exchangers


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()