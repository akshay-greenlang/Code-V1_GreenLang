"""
Integration Test Fixtures and Configuration

Shared fixtures for all integration tests including mock servers,
test data generators, and common utilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List

from mock_servers import (
    MockOPCUAServer,
    MockModbusServer,
    MockSAPServer,
    MockOracleAPIServer,
    MockMQTTBroker,
    start_all_mock_servers,
    stop_all_mock_servers
)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


# Session-scoped fixtures for mock servers
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def mock_servers():
    """Start all mock servers for testing session."""
    servers = await start_all_mock_servers()
    yield servers
    await stop_all_mock_servers(servers)


# Test data generators
@pytest.fixture
def sample_scada_data():
    """Generate sample SCADA data."""
    def _generate():
        return {
            'BOILER.STEAM.PRESSURE': random.uniform(95, 105),
            'BOILER.STEAM.TEMPERATURE': random.uniform(480, 500),
            'BOILER.STEAM.FLOW': random.uniform(190, 210),
            'BOILER.EFFICIENCY': random.uniform(88, 92),
            'BOILER.O2.CONTENT': random.uniform(3, 4),
            'BOILER.DRUM.LEVEL': random.uniform(-50, 50),
            'BOILER.FUEL.VALVE.POSITION': random.uniform(40, 60)
        }
    return _generate


@pytest.fixture
def sample_fuel_quality():
    """Generate sample fuel quality data."""
    def _generate(fuel_type='natural_gas'):
        if fuel_type == 'natural_gas':
            return {
                'heating_value': random.uniform(45, 52),
                'wobbe_index': random.uniform(48, 53),
                'methane_content': random.uniform(85, 95),
                'sulfur_content': random.uniform(0.001, 0.01)
            }
        elif fuel_type == 'fuel_oil':
            return {
                'heating_value': random.uniform(41, 44),
                'density': random.uniform(840, 860),
                'viscosity': random.uniform(3, 5),
                'sulfur_content': random.uniform(0.1, 0.4),
                'water_content': random.uniform(0.01, 0.08)
            }
        else:
            return {}
    return _generate


@pytest.fixture
def sample_emissions_data():
    """Generate sample emissions data."""
    def _generate():
        return {
            'co2': random.uniform(11, 14),
            'nox': random.uniform(80, 120),
            'so2': random.uniform(40, 60),
            'pm': random.uniform(5, 15),
            'o2': random.uniform(3, 5),
            'co': random.uniform(10, 30)
        }
    return _generate


@pytest.fixture
def sample_load_forecast():
    """Generate sample load forecast."""
    def _generate(hours=24, base_load=100):
        forecast = []
        for i in range(hours):
            # Simulate daily load pattern
            hour_of_day = i % 24
            if 0 <= hour_of_day < 6:  # Night
                load = base_load * 0.6
            elif 6 <= hour_of_day < 18:  # Day
                load = base_load * 1.0
            else:  # Evening
                load = base_load * 0.8

            # Add some variation
            load += random.uniform(-10, 10)
            forecast.append(max(0, load))

        return forecast
    return _generate


@pytest.fixture
def sample_optimization_constraints():
    """Generate sample optimization constraints."""
    return {
        'min_efficiency': 85.0,
        'max_emissions': {
            'nox': 150.0,
            'so2': 100.0,
            'pm': 20.0
        },
        'max_fuel_cost': 1500.0,
        'min_load': 50.0,
        'max_load': 150.0
    }


# Helper utilities
class TestDataGenerator:
    """Generate realistic test data for various scenarios."""

    @staticmethod
    def generate_time_series(
        duration_hours: int,
        interval_seconds: int,
        base_value: float,
        variation: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Generate time series data."""
        data = []
        start_time = datetime.utcnow() - timedelta(hours=duration_hours)

        total_points = (duration_hours * 3600) // interval_seconds

        for i in range(total_points):
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            value = base_value * (1 + random.uniform(-variation, variation))

            data.append({
                'timestamp': timestamp,
                'value': value
            })

        return data

    @staticmethod
    def generate_boiler_operating_scenario(scenario_type: str) -> Dict[str, Any]:
        """Generate complete boiler operating scenario."""
        scenarios = {
            'normal': {
                'load': 100.0,
                'efficiency': 90.0,
                'steam_pressure': 100.0,
                'steam_temperature': 490.0,
                'o2_content': 3.5,
                'fuel_type': 'natural_gas',
                'nox_emissions': 95.0,
                'so2_emissions': 45.0
            },
            'high_load': {
                'load': 140.0,
                'efficiency': 88.0,
                'steam_pressure': 110.0,
                'steam_temperature': 510.0,
                'o2_content': 3.0,
                'fuel_type': 'natural_gas',
                'nox_emissions': 130.0,
                'so2_emissions': 50.0
            },
            'low_efficiency': {
                'load': 100.0,
                'efficiency': 82.0,
                'steam_pressure': 98.0,
                'steam_temperature': 485.0,
                'o2_content': 5.5,
                'fuel_type': 'natural_gas',
                'nox_emissions': 85.0,
                'so2_emissions': 40.0
            },
            'high_emissions': {
                'load': 120.0,
                'efficiency': 87.0,
                'steam_pressure': 105.0,
                'steam_temperature': 500.0,
                'o2_content': 2.8,
                'fuel_type': 'fuel_oil',
                'nox_emissions': 165.0,
                'so2_emissions': 95.0
            }
        }

        return scenarios.get(scenario_type, scenarios['normal'])


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# Assertion helpers
class IntegrationTestAssertions:
    """Custom assertions for integration tests."""

    @staticmethod
    def assert_scada_connection_healthy(scada_connector):
        """Assert SCADA connection is healthy."""
        assert scada_connector.connected is True
        assert scada_connector.connection is not None

    @staticmethod
    def assert_fuel_system_operational(fuel_connector):
        """Assert fuel system is operational."""
        assert fuel_connector.connected is True
        assert len(fuel_connector.fuel_tanks) > 0
        assert len(fuel_connector.flow_meters) > 0

    @staticmethod
    def assert_emissions_within_limits(emissions_data, limits):
        """Assert emissions are within regulatory limits."""
        for pollutant, value in emissions_data.items():
            if pollutant in limits:
                assert value <= limits[pollutant], \
                    f"{pollutant} {value} exceeds limit {limits[pollutant]}"

    @staticmethod
    def assert_optimization_results_valid(optimization_results):
        """Assert optimization results are valid."""
        assert 'schedule' in optimization_results
        assert 'total_cost' in optimization_results
        assert optimization_results['total_cost'] > 0


@pytest.fixture
def integration_assertions():
    """Provide integration test assertions."""
    return IntegrationTestAssertions()


# Performance measurement
@pytest.fixture
def performance_monitor():
    """Monitor performance of integration tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = {}

        def start(self):
            self.start_time = datetime.utcnow()

        def record(self, metric_name: str, value: float):
            self.metrics[metric_name] = value

        def stop(self):
            if self.start_time:
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                self.metrics['total_duration'] = elapsed
                return elapsed
            return 0

        def get_report(self):
            return self.metrics

    return PerformanceMonitor()
