"""
Integration Test Fixtures and Configuration for GL-003 SteamSystemAnalyzer

Shared fixtures for all integration tests including mock servers,
test data generators, database connections, and common utilities.

Test Infrastructure:
- Mock SCADA/DCS servers (OPC UA, Modbus)
- Mock steam meters (Modbus, HART)
- Mock pressure sensors (4-20mA, analog)
- PostgreSQL database
- Redis cache
- MQTT broker

Author: GreenLang Test Engineering Team
"""

import pytest
import asyncio
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from mock_servers import (
    MockOPCUAServer,
    MockModbusServer,
    MockSteamMeterServer,
    MockPressureSensorServer,
    MockTemperatureSensorServer,
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
    config.addinivalue_line(
        "markers", "scada: mark test as SCADA integration test"
    )
    config.addinivalue_line(
        "markers", "steam_meter: mark test as steam meter integration test"
    )
    config.addinivalue_line(
        "markers", "pressure_sensor: mark test as pressure sensor test"
    )


# Session-scoped fixtures for async testing
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Mock Server Fixtures
@pytest.fixture(scope="session")
async def mock_servers():
    """Start all mock servers for testing session."""
    servers = await start_all_mock_servers()
    yield servers
    await stop_all_mock_servers(servers)


@pytest.fixture
async def mock_opcua_server():
    """Provide mock OPC UA server."""
    server = MockOPCUAServer(host="localhost", port=4840)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def mock_modbus_server():
    """Provide mock Modbus server."""
    server = MockModbusServer(host="localhost", port=502)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def mock_steam_meter():
    """Provide mock steam meter."""
    meter = MockSteamMeterServer(host="localhost", port=5020, meter_id="SM-001")
    await meter.start()
    yield meter
    await meter.stop()


@pytest.fixture
async def mock_pressure_sensor():
    """Provide mock pressure sensor."""
    sensor = MockPressureSensorServer(host="localhost", port=5030, sensor_id="PS-001")
    await sensor.start()
    yield sensor
    await sensor.stop()


@pytest.fixture
async def mock_temperature_sensor():
    """Provide mock temperature sensor."""
    sensor = MockTemperatureSensorServer(host="localhost", port=5040, sensor_id="TS-001")
    await sensor.start()
    yield sensor
    await sensor.stop()


@pytest.fixture
async def mock_mqtt_broker():
    """Provide mock MQTT broker."""
    broker = MockMQTTBroker(host="localhost", port=1883)
    await broker.start()
    yield broker
    await broker.stop()


# Database Fixtures
@pytest.fixture(scope="session")
def postgres_connection_string():
    """PostgreSQL connection string for tests."""
    return os.getenv(
        "TEST_POSTGRES_URL",
        "postgresql://gl003_user:test_password@localhost:5432/gl003_test"
    )


@pytest.fixture(scope="session")
async def postgres_pool(postgres_connection_string):
    """Create PostgreSQL connection pool."""
    import asyncpg

    pool = await asyncpg.create_pool(
        postgres_connection_string,
        min_size=2,
        max_size=10
    )

    # Create test tables
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS steam_measurements (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                pressure NUMERIC(10, 3),
                temperature NUMERIC(10, 3),
                flow_rate NUMERIC(10, 3),
                quality VARCHAR(20),
                source VARCHAR(100)
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pressure_measurements (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                sensor_id VARCHAR(50),
                pressure_value NUMERIC(10, 3),
                pressure_type VARCHAR(20),
                quality VARCHAR(20)
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS leak_detections (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                location VARCHAR(200),
                severity VARCHAR(20),
                estimated_loss NUMERIC(10, 3),
                status VARCHAR(20)
            )
        """)

    yield pool

    # Cleanup
    await pool.close()


@pytest.fixture
async def clean_database(postgres_pool):
    """Clean database before each test."""
    async with postgres_pool.acquire() as conn:
        await conn.execute("DELETE FROM steam_measurements")
        await conn.execute("DELETE FROM pressure_measurements")
        await conn.execute("DELETE FROM leak_detections")
    yield


# Redis Fixtures
@pytest.fixture(scope="session")
def redis_connection_string():
    """Redis connection string for tests."""
    return os.getenv("TEST_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture(scope="session")
async def redis_client(redis_connection_string):
    """Create Redis client."""
    import aioredis

    redis = await aioredis.create_redis_pool(redis_connection_string)
    yield redis
    redis.close()
    await redis.wait_closed()


@pytest.fixture
async def clean_redis(redis_client):
    """Clean Redis cache before each test."""
    await redis_client.flushdb()
    yield


# Test Data Generator Fixtures
@pytest.fixture
def sample_steam_data():
    """Generate sample steam system data."""
    def _generate(pressure=100.0, temperature=200.0, flow_rate=50.0):
        return {
            'timestamp': datetime.utcnow(),
            'header_pressure': pressure + random.uniform(-5, 5),
            'header_temperature': temperature + random.uniform(-10, 10),
            'flow_rate': flow_rate + random.uniform(-5, 5),
            'condensate_return': flow_rate * 0.85 + random.uniform(-2, 2),
            'steam_quality': random.uniform(0.95, 0.99)
        }
    return _generate


@pytest.fixture
def sample_pressure_readings():
    """Generate sample pressure sensor readings."""
    def _generate(num_sensors=5, base_pressure=100.0):
        readings = []
        for i in range(num_sensors):
            readings.append({
                'sensor_id': f"PS-{i+1:03d}",
                'timestamp': datetime.utcnow(),
                'pressure': base_pressure + random.uniform(-10, 10),
                'pressure_type': random.choice(['absolute', 'gauge', 'differential']),
                'quality': 'GOOD',
                'unit': 'bar'
            })
        return readings
    return _generate


@pytest.fixture
def sample_steam_meter_data():
    """Generate sample steam meter data."""
    def _generate(meter_id="SM-001"):
        return {
            'meter_id': meter_id,
            'timestamp': datetime.utcnow(),
            'volumetric_flow': random.uniform(1000, 5000),  # m3/hr
            'mass_flow': random.uniform(800, 4000),  # kg/hr
            'totalizer': random.uniform(100000, 500000),  # Total m3
            'pressure': random.uniform(8, 12),  # bar
            'temperature': random.uniform(180, 220),  # Celsius
            'quality': 'GOOD',
            'density': random.uniform(4.5, 5.5),  # kg/m3
            'energy_flow': random.uniform(2000, 10000)  # kW
        }
    return _generate


@pytest.fixture
def sample_leak_scenario():
    """Generate sample leak detection scenario."""
    def _generate(severity='medium'):
        severities = {
            'low': {'loss_rate': random.uniform(1, 5), 'priority': 3},
            'medium': {'loss_rate': random.uniform(5, 15), 'priority': 2},
            'high': {'loss_rate': random.uniform(15, 50), 'priority': 1},
            'critical': {'loss_rate': random.uniform(50, 200), 'priority': 0}
        }

        scenario = severities.get(severity, severities['medium'])

        return {
            'location': f"Section-{random.randint(1, 10)}-Valve-{random.randint(1, 20)}",
            'timestamp': datetime.utcnow(),
            'severity': severity,
            'estimated_loss_kg_hr': scenario['loss_rate'],
            'priority': scenario['priority'],
            'confidence': random.uniform(0.75, 0.99),
            'detection_method': random.choice(['pressure_drop', 'thermal_imaging', 'acoustic']),
            'recommended_action': 'Inspect and repair'
        }
    return _generate


@pytest.fixture
def sample_steam_trap_data():
    """Generate sample steam trap data."""
    def _generate(num_traps=20):
        traps = []
        statuses = ['GOOD', 'DEGRADED', 'FAILED', 'BLOCKED']

        for i in range(num_traps):
            status = random.choice(statuses)

            trap_data = {
                'trap_id': f"ST-{i+1:03d}",
                'location': f"Line-{random.randint(1, 5)}-Segment-{random.randint(1, 10)}",
                'timestamp': datetime.utcnow(),
                'status': status,
                'temperature_upstream': random.uniform(180, 220),
                'temperature_downstream': random.uniform(60, 100) if status == 'GOOD' else random.uniform(150, 200),
                'pressure_drop': random.uniform(0.5, 2.0) if status == 'GOOD' else random.uniform(0.1, 0.3),
                'last_maintenance': datetime.utcnow() - timedelta(days=random.randint(30, 365)),
                'estimated_loss_kg_hr': 0 if status == 'GOOD' else random.uniform(1, 20)
            }

            traps.append(trap_data)

        return traps
    return _generate


@pytest.fixture
def sample_efficiency_data():
    """Generate sample steam system efficiency data."""
    def _generate():
        return {
            'timestamp': datetime.utcnow(),
            'overall_efficiency': random.uniform(0.75, 0.92),
            'distribution_efficiency': random.uniform(0.85, 0.95),
            'condensate_recovery_rate': random.uniform(0.70, 0.90),
            'heat_loss_kw': random.uniform(50, 200),
            'steam_quality': random.uniform(0.95, 0.99),
            'system_pressure': random.uniform(8, 12),
            'system_temperature': random.uniform(170, 210),
            'estimated_savings_potential_kwh': random.uniform(100, 1000)
        }
    return _generate


@pytest.fixture
def sample_time_series_data():
    """Generate time series data for testing."""
    def _generate(duration_hours=24, interval_minutes=5):
        data_points = []
        start_time = datetime.utcnow() - timedelta(hours=duration_hours)
        num_points = (duration_hours * 60) // interval_minutes

        for i in range(num_points):
            timestamp = start_time + timedelta(minutes=i * interval_minutes)

            # Simulate daily pattern
            hour_of_day = timestamp.hour
            if 0 <= hour_of_day < 6:  # Night
                load_factor = 0.6
            elif 6 <= hour_of_day < 8:  # Morning ramp
                load_factor = 0.6 + (hour_of_day - 6) * 0.2
            elif 8 <= hour_of_day < 18:  # Day
                load_factor = 1.0
            elif 18 <= hour_of_day < 22:  # Evening
                load_factor = 0.8
            else:  # Night
                load_factor = 0.7

            data_point = {
                'timestamp': timestamp,
                'pressure': 100.0 * load_factor + random.uniform(-5, 5),
                'temperature': 200.0 + random.uniform(-10, 10),
                'flow_rate': 50.0 * load_factor + random.uniform(-5, 5),
                'efficiency': 0.85 + random.uniform(-0.05, 0.05)
            }

            data_points.append(data_point)

        return data_points
    return _generate


# Test Scenario Fixtures
@pytest.fixture
def normal_operation_scenario():
    """Normal steam system operation scenario."""
    return {
        'name': 'Normal Operation',
        'description': 'Steam system running at normal conditions',
        'pressure': 10.0,  # bar
        'temperature': 184.0,  # Celsius (saturated at 10 bar)
        'flow_rate': 50.0,  # tonnes/hr
        'condensate_recovery': 0.85,
        'expected_efficiency': 0.88,
        'expected_leaks': 0,
        'expected_trap_failures': 0
    }


@pytest.fixture
def low_pressure_scenario():
    """Low pressure operation scenario."""
    return {
        'name': 'Low Pressure Operation',
        'description': 'Steam system pressure below optimal',
        'pressure': 6.0,  # bar
        'temperature': 158.0,  # Celsius (saturated at 6 bar)
        'flow_rate': 35.0,  # tonnes/hr
        'condensate_recovery': 0.75,
        'expected_efficiency': 0.75,
        'expected_leaks': 1,
        'expected_trap_failures': 2
    }


@pytest.fixture
def high_load_scenario():
    """High load operation scenario."""
    return {
        'name': 'High Load Operation',
        'description': 'Steam system at maximum capacity',
        'pressure': 12.0,  # bar
        'temperature': 188.0,  # Celsius (saturated at 12 bar)
        'flow_rate': 80.0,  # tonnes/hr
        'condensate_recovery': 0.90,
        'expected_efficiency': 0.90,
        'expected_leaks': 0,
        'expected_trap_failures': 0
    }


@pytest.fixture
def leak_detection_scenario():
    """Leak detection scenario."""
    return {
        'name': 'Leak Detection',
        'description': 'Multiple leaks in steam distribution',
        'pressure': 9.0,  # bar (reduced due to leaks)
        'temperature': 175.0,  # Celsius
        'flow_rate': 55.0,  # tonnes/hr
        'condensate_recovery': 0.65,  # Low due to leaks
        'expected_efficiency': 0.70,
        'expected_leaks': 3,
        'leak_locations': ['Valve-12', 'Flange-45', 'Trap-ST-008'],
        'estimated_total_loss_kg_hr': 150.0
    }


# Helper Utilities
class TestDataGenerator:
    """Generate realistic test data for various scenarios."""

    @staticmethod
    def generate_modbus_register_map() -> Dict[int, Any]:
        """Generate Modbus register map for steam system."""
        return {
            # Steam header (40001-40020)
            40001: 10.5,   # Pressure (bar * 10)
            40002: 1840,   # Temperature (°C * 10)
            40003: 500,    # Flow rate (t/hr * 10)
            40004: 96,     # Steam quality (% * 10)
            40005: 425,    # Condensate return (t/hr * 10)

            # Pressure sensors (40021-40040)
            40021: 105,    # PS-001 (bar * 10)
            40022: 103,    # PS-002 (bar * 10)
            40023: 98,     # PS-003 (bar * 10)
            40024: 101,    # PS-004 (bar * 10)
            40025: 102,    # PS-005 (bar * 10)

            # Temperature sensors (40041-40060)
            40041: 1850,   # TS-001 (°C * 10)
            40042: 1830,   # TS-002 (°C * 10)
            40043: 1810,   # TS-003 (°C * 10)
            40044: 1840,   # TS-004 (°C * 10)
            40045: 1820,   # TS-005 (°C * 10)

            # Steam meters (40061-40080)
            40061: 3500,   # SM-001 volumetric (m3/hr)
            40062: 2800,   # SM-001 mass (kg/hr)
            40063: 7500,   # SM-002 volumetric (m3/hr)
            40064: 6000,   # SM-002 mass (kg/hr)

            # Steam trap status (40081-40100)
            40081: 1,      # ST-001 (1=OK, 2=WARN, 3=FAIL)
            40082: 1,      # ST-002
            40083: 2,      # ST-003 (degraded)
            40084: 1,      # ST-004
            40085: 3,      # ST-005 (failed)
        }

    @staticmethod
    def generate_opc_ua_node_tree() -> Dict[str, Any]:
        """Generate OPC UA node tree for steam system."""
        return {
            'Root': {
                'SteamSystem': {
                    'MainHeader': {
                        'Pressure': {'value': 10.5, 'unit': 'bar', 'quality': 'Good'},
                        'Temperature': {'value': 184.0, 'unit': 'degC', 'quality': 'Good'},
                        'FlowRate': {'value': 50.0, 'unit': 't/hr', 'quality': 'Good'},
                        'Quality': {'value': 0.96, 'unit': '%', 'quality': 'Good'}
                    },
                    'Distribution': {
                        'Line1': {
                            'Pressure': {'value': 10.2, 'unit': 'bar', 'quality': 'Good'},
                            'Temperature': {'value': 182.0, 'unit': 'degC', 'quality': 'Good'},
                            'FlowRate': {'value': 20.0, 'unit': 't/hr', 'quality': 'Good'}
                        },
                        'Line2': {
                            'Pressure': {'value': 10.1, 'unit': 'bar', 'quality': 'Good'},
                            'Temperature': {'value': 181.0, 'unit': 'degC', 'quality': 'Good'},
                            'FlowRate': {'value': 30.0, 'unit': 't/hr', 'quality': 'Good'}
                        }
                    },
                    'Condensate': {
                        'ReturnFlow': {'value': 42.5, 'unit': 't/hr', 'quality': 'Good'},
                        'Temperature': {'value': 90.0, 'unit': 'degC', 'quality': 'Good'},
                        'TankLevel': {'value': 75.0, 'unit': '%', 'quality': 'Good'}
                    }
                }
            }
        }


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# Assertion Helpers
class IntegrationTestAssertions:
    """Custom assertions for integration tests."""

    @staticmethod
    def assert_scada_connection_healthy(scada_connector):
        """Assert SCADA connection is healthy."""
        assert scada_connector.is_connected is True
        assert scada_connector.connection is not None

    @staticmethod
    def assert_steam_meter_operational(meter_connector):
        """Assert steam meter is operational."""
        assert meter_connector.is_connected is True
        assert meter_connector.meter_id is not None

    @staticmethod
    def assert_pressure_within_range(pressure: float, min_val: float, max_val: float):
        """Assert pressure is within safe operating range."""
        assert min_val <= pressure <= max_val, \
            f"Pressure {pressure} bar outside range [{min_val}, {max_val}]"

    @staticmethod
    def assert_leak_detection_valid(leak_data: Dict[str, Any]):
        """Assert leak detection data is valid."""
        assert 'location' in leak_data
        assert 'severity' in leak_data
        assert 'estimated_loss_kg_hr' in leak_data
        assert leak_data['severity'] in ['low', 'medium', 'high', 'critical']
        assert leak_data['estimated_loss_kg_hr'] > 0

    @staticmethod
    def assert_efficiency_acceptable(efficiency: float, min_efficiency: float = 0.75):
        """Assert steam system efficiency is acceptable."""
        assert efficiency >= min_efficiency, \
            f"Efficiency {efficiency:.2%} below minimum {min_efficiency:.2%}"

    @staticmethod
    def assert_steam_quality_valid(quality: float):
        """Assert steam quality is valid (0-1)."""
        assert 0.0 <= quality <= 1.0, f"Steam quality {quality} out of range [0, 1]"
        assert quality >= 0.90, f"Steam quality {quality} below acceptable threshold 0.90"


@pytest.fixture
def integration_assertions():
    """Provide integration test assertions."""
    return IntegrationTestAssertions()


# Performance Monitoring
@pytest.fixture
def performance_monitor():
    """Monitor performance of integration tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time: Optional[datetime] = None
            self.metrics: Dict[str, Any] = {}
            self.checkpoints: List[Dict[str, Any]] = []

        def start(self):
            """Start performance monitoring."""
            self.start_time = datetime.utcnow()

        def checkpoint(self, name: str):
            """Record a checkpoint."""
            if self.start_time:
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                self.checkpoints.append({
                    'name': name,
                    'elapsed_seconds': elapsed
                })

        def record(self, metric_name: str, value: Any):
            """Record a metric."""
            self.metrics[metric_name] = value

        def stop(self) -> float:
            """Stop monitoring and return total duration."""
            if self.start_time:
                elapsed = (datetime.utcnow() - self.start_time).total_seconds()
                self.metrics['total_duration_seconds'] = elapsed
                return elapsed
            return 0.0

        def get_report(self) -> Dict[str, Any]:
            """Get performance report."""
            return {
                'metrics': self.metrics,
                'checkpoints': self.checkpoints
            }

    return PerformanceMonitor()


# Environment Configuration
@pytest.fixture(scope="session")
def test_environment():
    """Test environment configuration."""
    return {
        'SCADA_HOST': os.getenv('SCADA_HOST', 'localhost'),
        'SCADA_PORT': int(os.getenv('SCADA_PORT', '4840')),
        'MODBUS_HOST': os.getenv('MODBUS_HOST', 'localhost'),
        'MODBUS_PORT': int(os.getenv('MODBUS_PORT', '502')),
        'MQTT_HOST': os.getenv('MQTT_HOST', 'localhost'),
        'MQTT_PORT': int(os.getenv('MQTT_PORT', '1883')),
        'POSTGRES_HOST': os.getenv('POSTGRES_HOST', 'localhost'),
        'POSTGRES_PORT': int(os.getenv('POSTGRES_PORT', '5432')),
        'REDIS_HOST': os.getenv('REDIS_HOST', 'localhost'),
        'REDIS_PORT': int(os.getenv('REDIS_PORT', '6379')),
        'TEST_TIMEOUT': int(os.getenv('TEST_TIMEOUT', '300')),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
    }
