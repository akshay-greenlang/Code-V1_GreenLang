# -*- coding: utf-8 -*-
"""
Integration Test Fixtures and Configuration for GL-001

Comprehensive fixtures for all integration tests including:
- Mock servers (SCADA, ERP, sub-agents)
- Test databases (PostgreSQL, Redis)
- Message brokers (MQTT)
- Multi-plant test data generators
- Performance monitoring utilities
"""

import pytest
import asyncio
import os
import random
import psycopg
import redis.asyncio as redis
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from mock_servers import (
from greenlang.determinism import deterministic_random
    MockOPCUAServer,
    MockModbusServer,
    MockSAPServer,
    MockOracleAPIServer,
    MockMQTTBroker,
    MockSubAgent,
    MockMultiPlantCoordinator,
    start_all_mock_servers,
    stop_all_mock_servers
)


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    markers = [
        "e2e: End-to-end integration tests",
        "slow: Slow-running tests (>30s)",
        "integration: General integration tests",
        "scada: SCADA integration tests",
        "erp: ERP integration tests",
        "coordination: Agent coordination tests",
        "multi_plant: Multi-plant orchestration tests",
        "performance: Performance and load tests",
        "compliance: Compliance validation tests",
        "docker: Tests requiring Docker infrastructure",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--docker",
        action="store_true",
        default=False,
        help="Run tests with Docker infrastructure"
    )
    parser.addoption(
        "--load-test",
        action="store_true",
        default=False,
        help="Run performance load tests"
    )
    parser.addoption(
        "--multi-plant-count",
        action="store",
        default=3,
        type=int,
        help="Number of plants for multi-plant tests"
    )


# ==============================================================================
# EVENT LOOP FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for Windows compatibility."""
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ==============================================================================
# MOCK SERVER FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
async def mock_servers():
    """Start all mock servers for testing session."""
    servers = await start_all_mock_servers()
    yield servers
    await stop_all_mock_servers(servers)


@pytest.fixture(scope="function")
async def mock_opcua_server():
    """Mock OPC UA server for SCADA testing."""
    server = MockOPCUAServer(host="localhost", port=4840)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(scope="function")
async def mock_modbus_server():
    """Mock Modbus TCP server for fuel/emissions systems."""
    server = MockModbusServer(host="localhost", port=502)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(scope="function")
async def mock_sap_server():
    """Mock SAP RFC server for ERP integration."""
    server = MockSAPServer(host="localhost", port=3300)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(scope="function")
async def mock_oracle_server():
    """Mock Oracle REST API server for ERP integration."""
    server = MockOracleAPIServer(host="localhost", port=8080)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(scope="function")
async def mock_mqtt_broker():
    """Mock MQTT broker for agent messaging."""
    broker = MockMQTTBroker(host="localhost", port=1883)
    await broker.start()
    yield broker
    await broker.stop()


# ==============================================================================
# SUB-AGENT MOCK FIXTURES
# ==============================================================================

@pytest.fixture(scope="function")
async def mock_gl002_agent():
    """Mock GL-002 Boiler Efficiency Agent."""
    agent = MockSubAgent(
        agent_id="GL-002",
        agent_type="boiler_efficiency",
        port=5002
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture(scope="function")
async def mock_gl003_agent():
    """Mock GL-003 Steam Distribution Agent."""
    agent = MockSubAgent(
        agent_id="GL-003",
        agent_type="steam_distribution",
        port=5003
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture(scope="function")
async def mock_gl004_agent():
    """Mock GL-004 Heat Recovery Agent."""
    agent = MockSubAgent(
        agent_id="GL-004",
        agent_type="heat_recovery",
        port=5004
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture(scope="function")
async def mock_gl005_agent():
    """Mock GL-005 Emissions Monitoring Agent."""
    agent = MockSubAgent(
        agent_id="GL-005",
        agent_type="emissions_monitoring",
        port=5005
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.fixture(scope="function")
async def all_sub_agents(mock_gl002_agent, mock_gl003_agent, mock_gl004_agent, mock_gl005_agent):
    """All sub-agents for coordinated testing."""
    return {
        "GL-002": mock_gl002_agent,
        "GL-003": mock_gl003_agent,
        "GL-004": mock_gl004_agent,
        "GL-005": mock_gl005_agent,
    }


# ==============================================================================
# MULTI-PLANT FIXTURES
# ==============================================================================

@pytest.fixture(scope="function")
async def mock_multi_plant_coordinator(request):
    """Mock multi-plant coordinator for orchestration testing."""
    plant_count = request.config.getoption("--multi-plant-count")
    coordinator = MockMultiPlantCoordinator(plant_count=plant_count)
    await coordinator.start()
    yield coordinator
    await coordinator.stop()


@pytest.fixture(scope="function")
def multi_plant_configs(request):
    """Generate configurations for multiple plants."""
    plant_count = request.config.getoption("--multi-plant-count")

    configs = []
    for i in range(plant_count):
        config = {
            'plant_id': f"PLANT-{i+1:03d}",
            'plant_name': f"Industrial Plant {i+1}",
            'location': {
                'country': deterministic_random().choice(['US', 'DE', 'CN', 'JP']),
                'region': f"Region-{i+1}",
                'coordinates': {
                    'lat': random.uniform(-90, 90),
                    'lon': random.uniform(-180, 180)
                }
            },
            'capacity_mw': random.uniform(50, 500),
            'fuel_types': deterministic_random().choice([
                ['natural_gas', 'biomass'],
                ['coal', 'natural_gas'],
                ['fuel_oil', 'natural_gas']
            ]),
            'scada': {
                'opcua_endpoint': f"opc.tcp://localhost:{4840 + i}",
                'modbus_host': 'localhost',
                'modbus_port': 502 + i
            },
            'erp': {
                'system': deterministic_random().choice(['SAP', 'Oracle']),
                'endpoint': f"http://localhost:{8080 + i}"
            }
        }
        configs.append(config)

    return configs


# ==============================================================================
# DATABASE FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
async def postgres_connection():
    """PostgreSQL database connection for integration testing."""
    host = os.getenv('TEST_POSTGRES_HOST', 'localhost')
    port = int(os.getenv('TEST_POSTGRES_PORT', 5432))
    database = os.getenv('TEST_POSTGRES_DB', 'gl001_test')
    user = os.getenv('TEST_POSTGRES_USER', 'postgres')
    password = os.getenv('TEST_POSTGRES_PASSWORD', 'postgres')

    conn = await psycopg.AsyncConnection.connect(
        f"host={host} port={port} dbname={database} user={user} password={password}"
    )

    # Create test tables
    await _create_test_tables(conn)

    yield conn

    await conn.close()


async def _create_test_tables(conn):
    """Create test database tables."""
    async with conn.cursor() as cur:
        # Plant data table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS plant_data (
                id SERIAL PRIMARY KEY,
                plant_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                fuel_input_mw FLOAT,
                useful_heat_mw FLOAT,
                efficiency_percent FLOAT,
                emissions_kg_hr FLOAT,
                data JSONB
            )
        """)

        # Orchestration results table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS orchestration_results (
                id SERIAL PRIMARY KEY,
                orchestration_id VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                plant_count INT,
                execution_time_ms FLOAT,
                kpi_dashboard JSONB,
                provenance_hash VARCHAR(64)
            )
        """)

        await conn.commit()


@pytest.fixture(scope="session")
async def redis_connection():
    """Redis connection for caching tests."""
    host = os.getenv('TEST_REDIS_HOST', 'localhost')
    port = int(os.getenv('TEST_REDIS_PORT', 6379))

    client = await redis.from_url(f"redis://{host}:{port}/0")

    # Flush test database
    await client.flushdb()

    yield client

    await client.close()


# ==============================================================================
# TEST DATA GENERATORS
# ==============================================================================

@pytest.fixture
def sample_plant_data():
    """Generate sample plant operating data."""
    def _generate(plant_id="PLANT-001"):
        return {
            'plant_id': plant_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'inlet_temp_c': random.uniform(480, 520),
            'outlet_temp_c': random.uniform(140, 160),
            'ambient_temp_c': random.uniform(20, 30),
            'fuel_input_mw': random.uniform(90, 110),
            'useful_heat_mw': random.uniform(80, 95),
            'heat_recovery_mw': random.uniform(5, 15),
            'steam_pressure_bar': random.uniform(95, 105),
            'steam_temperature_c': random.uniform(485, 495),
            'steam_flow_kg_s': random.uniform(45, 55)
        }
    return _generate


@pytest.fixture
def sample_sensor_feeds():
    """Generate sample sensor feed data."""
    def _generate(plant_id="PLANT-001", tag_count=50):
        feeds = {
            'plant_id': plant_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tags': {}
        }

        # Generate realistic sensor tags
        tag_prefixes = ['BOILER', 'STEAM', 'HEAT', 'FUEL', 'EMISSIONS']
        measurements = ['TEMP', 'PRESSURE', 'FLOW', 'LEVEL', 'EFFICIENCY']

        for i in range(tag_count):
            prefix = deterministic_random().choice(tag_prefixes)
            measurement = deterministic_random().choice(measurements)
            tag_name = f"{prefix}.{measurement}.{i:03d}"

            # Generate value based on measurement type
            if measurement == 'TEMP':
                value = random.uniform(50, 500)
            elif measurement == 'PRESSURE':
                value = random.uniform(0, 150)
            elif measurement == 'FLOW':
                value = random.uniform(0, 200)
            elif measurement == 'LEVEL':
                value = random.uniform(0, 100)
            else:  # EFFICIENCY
                value = random.uniform(80, 95)

            feeds['tags'][tag_name] = {
                'value': value,
                'quality': 'GOOD',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        return feeds
    return _generate


@pytest.fixture
def sample_emissions_data():
    """Generate sample emissions data."""
    def _generate(plant_id="PLANT-001"):
        return {
            'plant_id': plant_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'co2_kg_hr': random.uniform(8000, 12000),
            'nox_mg_nm3': random.uniform(80, 120),
            'so2_mg_nm3': random.uniform(40, 60),
            'pm_mg_nm3': random.uniform(5, 15),
            'o2_percent': random.uniform(3, 5),
            'co_ppm': random.uniform(10, 30),
            'fuel_consumption_kg_hr': random.uniform(9000, 11000),
            'stack_flow_nm3_hr': random.uniform(50000, 70000)
        }
    return _generate


@pytest.fixture
def sample_optimization_constraints():
    """Generate sample optimization constraints."""
    return {
        'min_efficiency_percent': 85.0,
        'max_emissions': {
            'nox_mg_nm3': 150.0,
            'so2_mg_nm3': 100.0,
            'pm_mg_nm3': 20.0
        },
        'min_load_percent': 50.0,
        'max_load_percent': 110.0,
        'max_fuel_cost_usd_hr': 2000.0,
        'min_steam_pressure_bar': 90.0,
        'max_steam_pressure_bar': 120.0,
        'energy_balance_tolerance_percent': 2.0
    }


@pytest.fixture
def sample_erp_data():
    """Generate sample ERP data."""
    def _generate(plant_id="PLANT-001"):
        return {
            'plant_id': plant_id,
            'material_costs': {
                'natural_gas': {
                    'price_per_m3': 0.35,
                    'currency': 'USD',
                    'contract': 'CONT-2025-001'
                },
                'fuel_oil': {
                    'price_per_kg': 0.75,
                    'currency': 'USD',
                    'contract': 'CONT-2025-002'
                }
            },
            'production_schedule': {
                'planned_output_mw': random.uniform(80, 100),
                'duration_hours': 24,
                'priority': 'HIGH'
            },
            'maintenance_windows': [
                {
                    'start': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                    'end': (datetime.now(timezone.utc) + timedelta(days=7, hours=8)).isoformat(),
                    'type': 'PREVENTIVE'
                }
            ],
            'budget_allocation': {
                'fuel_budget_usd': 50000.0,
                'maintenance_budget_usd': 10000.0,
                'emissions_credits_usd': 5000.0
            }
        }
    return _generate


# ==============================================================================
# TEST DATA GENERATOR CLASS
# ==============================================================================

class TestDataGenerator:
    """Generate realistic test data for various scenarios."""

    @staticmethod
    def generate_time_series(
        duration_hours: int,
        interval_seconds: int,
        base_value: float,
        variation: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Generate time series data with realistic patterns."""
        data = []
        start_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)

        total_points = (duration_hours * 3600) // interval_seconds

        for i in range(total_points):
            timestamp = start_time + timedelta(seconds=i * interval_seconds)

            # Add daily pattern
            hour_of_day = timestamp.hour
            daily_factor = 1.0
            if 0 <= hour_of_day < 6:  # Night
                daily_factor = 0.7
            elif 6 <= hour_of_day < 18:  # Day
                daily_factor = 1.0
            else:  # Evening
                daily_factor = 0.85

            # Add random variation
            random_factor = 1 + random.uniform(-variation, variation)

            value = base_value * daily_factor * random_factor

            data.append({
                'timestamp': timestamp.isoformat(),
                'value': value
            })

        return data

    @staticmethod
    def generate_multi_plant_scenario(plant_count: int, scenario_type: str) -> List[Dict[str, Any]]:
        """Generate multi-plant operating scenario."""
        scenarios = {
            'normal': {
                'load_range': (80, 100),
                'efficiency_range': (88, 92),
                'emissions_factor': 1.0
            },
            'peak_demand': {
                'load_range': (100, 120),
                'efficiency_range': (85, 90),
                'emissions_factor': 1.2
            },
            'maintenance': {
                'load_range': (50, 70),
                'efficiency_range': (80, 85),
                'emissions_factor': 0.8
            },
            'emergency': {
                'load_range': (40, 60),
                'efficiency_range': (75, 80),
                'emissions_factor': 1.5
            }
        }

        scenario = scenarios.get(scenario_type, scenarios['normal'])
        plants = []

        for i in range(plant_count):
            plant_data = {
                'plant_id': f"PLANT-{i+1:03d}",
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'load_percent': random.uniform(*scenario['load_range']),
                'efficiency_percent': random.uniform(*scenario['efficiency_range']),
                'fuel_input_mw': random.uniform(90, 110),
                'emissions_kg_hr': random.uniform(8000, 12000) * scenario['emissions_factor'],
                'status': 'OPERATIONAL' if scenario_type != 'emergency' else 'DEGRADED'
            }
            plants.append(plant_data)

        return plants

    @staticmethod
    def generate_coordination_commands(agent_ids: List[str]) -> Dict[str, Any]:
        """Generate agent coordination commands."""
        commands = {}

        for agent_id in agent_ids:
            commands[agent_id] = {
                'command_type': deterministic_random().choice(['optimize', 'monitor', 'report']),
                'priority': deterministic_random().choice(['HIGH', 'MEDIUM', 'LOW']),
                'parameters': {
                    'target_efficiency': random.uniform(85, 92),
                    'emissions_limit': random.uniform(100, 150),
                    'time_horizon_hours': deterministic_random().randint(1, 24)
                },
                'timeout_seconds': 30
            }

        return commands


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================

class IntegrationTestAssertions:
    """Custom assertions for integration tests."""

    @staticmethod
    def assert_orchestration_result_valid(result: Dict[str, Any]):
        """Assert orchestration result is valid and complete."""
        required_fields = [
            'agent_id',
            'timestamp',
            'execution_time_ms',
            'thermal_efficiency',
            'heat_distribution',
            'energy_balance',
            'emissions_compliance',
            'kpi_dashboard',
            'provenance_hash'
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        assert result['execution_time_ms'] > 0
        assert len(result['provenance_hash']) == 64  # SHA-256 hash

    @staticmethod
    def assert_scada_connection_healthy(scada_status: Dict[str, Any]):
        """Assert SCADA connection is healthy."""
        assert scada_status.get('connected') is True
        assert scada_status.get('tag_count', 0) > 0
        assert 'last_update' in scada_status

    @staticmethod
    def assert_erp_integration_successful(erp_status: Dict[str, Any]):
        """Assert ERP integration is successful."""
        assert erp_status.get('connected') is True
        assert 'system_type' in erp_status
        assert erp_status.get('system_type') in ['SAP', 'Oracle']

    @staticmethod
    def assert_agent_coordination_successful(coordination_result: Dict[str, Any]):
        """Assert agent coordination is successful."""
        assert 'task_assignments' in coordination_result
        assert 'coordination_status' in coordination_result
        assert coordination_result['coordination_status'] == 'SUCCESS'

    @staticmethod
    def assert_multi_plant_optimization_valid(optimization_result: Dict[str, Any], plant_count: int):
        """Assert multi-plant optimization result is valid."""
        assert 'plants' in optimization_result
        assert len(optimization_result['plants']) == plant_count
        assert 'total_efficiency' in optimization_result
        assert 'total_emissions' in optimization_result

    @staticmethod
    def assert_emissions_compliance(emissions_data: Dict[str, Any], limits: Dict[str, float]):
        """Assert emissions are within regulatory limits."""
        for pollutant, limit in limits.items():
            actual = emissions_data.get(pollutant, 0)
            assert actual <= limit, f"{pollutant} {actual} exceeds limit {limit}"

    @staticmethod
    def assert_performance_target_met(execution_time_ms: float, target_ms: float):
        """Assert performance target is met."""
        assert execution_time_ms <= target_ms, \
            f"Execution time {execution_time_ms}ms exceeds target {target_ms}ms"

    @staticmethod
    def assert_energy_balance_valid(energy_balance: Dict[str, Any], tolerance_percent: float = 2.0):
        """Assert energy balance is within tolerance."""
        balance_error = energy_balance.get('balance_error_percent', 100)
        assert balance_error <= tolerance_percent, \
            f"Energy balance error {balance_error}% exceeds tolerance {tolerance_percent}%"


@pytest.fixture
def integration_assertions():
    """Provide integration test assertions."""
    return IntegrationTestAssertions()


# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

class PerformanceMonitor:
    """Monitor performance metrics during integration tests."""

    def __init__(self):
        self.start_time = None
        self.metrics = {}
        self.events = []

    def start(self):
        """Start performance monitoring."""
        self.start_time = datetime.now(timezone.utc)
        self.metrics = {}
        self.events = []

    def record_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance event."""
        event = {
            'name': event_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_ms': self.elapsed_ms(),
            'metadata': metadata or {}
        }
        self.events.append(event)

    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0
        elapsed = datetime.now(timezone.utc) - self.start_time
        return elapsed.total_seconds() * 1000

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return performance report."""
        total_duration_ms = self.elapsed_ms()

        # Calculate metric statistics
        metric_stats = {}
        for metric_name, values in self.metrics.items():
            if values:
                metric_stats[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }

        return {
            'total_duration_ms': total_duration_ms,
            'event_count': len(self.events),
            'events': self.events,
            'metrics': metric_stats
        }

    def get_report(self) -> Dict[str, Any]:
        """Get current performance report without stopping."""
        return self.stop()


@pytest.fixture
def performance_monitor():
    """Provide performance monitor instance."""
    return PerformanceMonitor()


# ==============================================================================
# CLEANUP FIXTURES
# ==============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test(postgres_connection, redis_connection):
    """Cleanup test data after each test."""
    yield

    # Clean up database
    async with postgres_connection.cursor() as cur:
        await cur.execute("DELETE FROM plant_data WHERE timestamp < NOW() - INTERVAL '1 hour'")
        await cur.execute("DELETE FROM orchestration_results WHERE timestamp < NOW() - INTERVAL '1 hour'")
        await postgres_connection.commit()

    # Clean up cache
    await redis_connection.flushdb()
