"""
Pytest fixtures and configuration for GL-004 integration tests.

Provides mock equipment, database setup, and common test utilities
for comprehensive integration testing of BurnerOptimizationAgent.
"""

import pytest
import asyncio
import psycopg2
import redis
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
import json
import random
import threading
import time
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.server.async_io import StartAsyncTcpServer, StopAsyncTcpServer
from pymodbus.device import ModbusDeviceIdentification
import paho.mqtt.client as mqtt
from flask import Flask, jsonify
import requests_mock
from contextlib import contextmanager


# Test Configuration
TEST_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'gl004_test',
    'user': 'test_user',
    'password': 'test_password'
}

TEST_REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 1,
    'decode_responses': True
}

TEST_MQTT_CONFIG = {
    'broker': 'localhost',
    'port': 1883,
    'client_id': 'gl004_test_client',
    'topics': ['emissions/co', 'emissions/nox', 'emissions/o2']
}

TEST_MODBUS_CONFIG = {
    'host': 'localhost',
    'port': 5502,
    'unit_id': 1
}


# Fixture: Database Connection
@pytest.fixture(scope='session')
def db_connection():
    """Create test database connection."""
    conn = psycopg2.connect(**TEST_DB_CONFIG)
    conn.autocommit = False

    yield conn

    conn.rollback()
    conn.close()


@pytest.fixture(scope='function')
def db_cursor(db_connection):
    """Create database cursor for test."""
    cursor = db_connection.cursor()

    # Create test tables
    cursor.execute("""
        CREATE TEMP TABLE IF NOT EXISTS burner_states (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            fuel_flow DECIMAL(10,2),
            air_flow DECIMAL(10,2),
            o2_level DECIMAL(5,2),
            temperature DECIMAL(10,2),
            co_emissions DECIMAL(10,2),
            nox_emissions DECIMAL(10,2),
            efficiency DECIMAL(5,2),
            provenance_hash VARCHAR(64)
        )
    """)

    cursor.execute("""
        CREATE TEMP TABLE IF NOT EXISTS optimization_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            original_efficiency DECIMAL(5,2),
            optimized_efficiency DECIMAL(5,2),
            adjustments JSONB,
            status VARCHAR(50)
        )
    """)

    db_connection.commit()

    yield cursor

    cursor.close()
    db_connection.rollback()


# Fixture: Redis Cache
@pytest.fixture(scope='function')
def redis_cache():
    """Create test Redis connection."""
    client = redis.Redis(**TEST_REDIS_CONFIG)

    # Clear test database
    client.flushdb()

    yield client

    # Cleanup
    client.flushdb()
    client.close()


# Fixture: Mock Modbus Server
@pytest.fixture(scope='function')
async def mock_modbus_server():
    """Create mock Modbus server for burner controller."""
    # Initialize data store with realistic values
    store = ModbusSlaveContext(
        # Discrete inputs (read-only binary sensors)
        di=ModbusSequentialDataBlock(0, [1, 0, 1, 0, 1, 0, 1, 0]),
        # Coils (read/write binary controls)
        co=ModbusSequentialDataBlock(0, [0] * 100),
        # Holding registers (read/write analog values)
        hr=ModbusSequentialDataBlock(0, [
            # Fuel flow: 1000 (10.00 kg/h)
            1000,
            # Air flow: 12000 (120.00 m³/h)
            12000,
            # O2 level: 350 (3.50%)
            350,
            # Temperature: 8500 (850.0°C)
            8500,
            # CO emissions: 50 (50 ppm)
            50,
            # NOx emissions: 120 (120 ppm)
            120,
            # Efficiency: 8750 (87.50%)
            8750,
            # Setpoint fuel flow: 1000
            1000,
            # Setpoint air flow: 12000
            12000,
            # Safety interlock: 1 (enabled)
            1,
        ] + [0] * 90),  # Padding to 100 registers
        # Input registers (read-only analog values)
        ir=ModbusSequentialDataBlock(0, [0] * 100)
    )

    context = ModbusServerContext(slaves=store, single=True)

    # Start server in background
    server_task = asyncio.create_task(
        StartAsyncTcpServer(
            context=context,
            address=(TEST_MODBUS_CONFIG['host'], TEST_MODBUS_CONFIG['port'])
        )
    )

    # Wait for server to start
    await asyncio.sleep(0.5)

    yield context

    # Stop server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


# Fixture: Mock MQTT Broker
@pytest.fixture(scope='function')
def mock_mqtt_broker():
    """Create mock MQTT broker for emissions monitoring."""
    class MockMQTTBroker:
        def __init__(self):
            self.messages = {}
            self.subscribers = {}
            self.running = True
            self.client = mqtt.Client()

        def publish(self, topic: str, payload: Dict[str, Any]):
            """Publish message to topic."""
            self.messages[topic] = json.dumps(payload)

            # Notify subscribers
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    callback(topic, payload)

        def subscribe(self, topic: str, callback):
            """Subscribe to topic."""
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)

        def get_latest(self, topic: str) -> Dict[str, Any]:
            """Get latest message from topic."""
            if topic in self.messages:
                return json.loads(self.messages[topic])
            return {}

        def simulate_emissions(self):
            """Simulate realistic emissions data."""
            base_co = 50
            base_nox = 120
            base_o2 = 3.5

            while self.running:
                # Add realistic variations
                co = base_co + random.gauss(0, 5)
                nox = base_nox + random.gauss(0, 10)
                o2 = base_o2 + random.gauss(0, 0.2)

                self.publish('emissions/co', {
                    'value': max(0, co),
                    'unit': 'ppm',
                    'timestamp': datetime.now().isoformat()
                })

                self.publish('emissions/nox', {
                    'value': max(0, nox),
                    'unit': 'ppm',
                    'timestamp': datetime.now().isoformat()
                })

                self.publish('emissions/o2', {
                    'value': max(0, o2),
                    'unit': '%',
                    'timestamp': datetime.now().isoformat()
                })

                time.sleep(1)

        def stop(self):
            """Stop the broker."""
            self.running = False

    broker = MockMQTTBroker()

    # Start emissions simulation in background
    sim_thread = threading.Thread(target=broker.simulate_emissions, daemon=True)
    sim_thread.start()

    yield broker

    broker.stop()


# Fixture: Mock HTTP API Server
@pytest.fixture(scope='function')
def mock_http_api():
    """Create mock HTTP API for flame scanner."""
    app = Flask(__name__)

    @app.route('/api/flame/status', methods=['GET'])
    def get_flame_status():
        """Get flame scanner status."""
        return jsonify({
            'status': 'stable',
            'intensity': 85.5,
            'stability': 92.3,
            'color_temp': 1850,
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/api/flame/history', methods=['GET'])
    def get_flame_history():
        """Get flame history data."""
        history = []
        base_time = datetime.now()

        for i in range(60):
            history.append({
                'timestamp': (base_time - timedelta(seconds=i)).isoformat(),
                'intensity': 85 + random.gauss(0, 2),
                'stability': 92 + random.gauss(0, 1)
            })

        return jsonify(history)

    # Run server in thread
    import threading
    from werkzeug.serving import make_server

    server = make_server('localhost', 5003, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield app

    server.shutdown()


# Fixture: Burner State Generator
@pytest.fixture
def burner_state_generator():
    """Generate realistic burner state data."""
    def generate_state(
        fuel_flow: float = 10.0,
        air_flow: float = 120.0,
        o2_level: float = 3.5,
        temperature: float = 850.0
    ) -> Dict[str, Any]:
        """Generate single burner state."""
        # Calculate efficiency based on O2 level
        optimal_o2 = 3.0
        efficiency = 90.0 - abs(o2_level - optimal_o2) * 5

        # Calculate emissions based on O2 level
        co = 50 * (1 + (o2_level - optimal_o2) * 0.2)
        nox = 120 * (1 - (o2_level - optimal_o2) * 0.1)

        return {
            'timestamp': datetime.now(),
            'fuel_flow': fuel_flow,
            'air_flow': air_flow,
            'o2_level': o2_level,
            'temperature': temperature,
            'co_emissions': max(0, co),
            'nox_emissions': max(0, nox),
            'efficiency': min(100, max(0, efficiency)),
            'flame_intensity': 85.0 + random.gauss(0, 2),
            'flame_stability': 92.0 + random.gauss(0, 1)
        }

    return generate_state


# Fixture: Test Data Loader
@pytest.fixture
def test_data_loader():
    """Load test data sets."""
    def load_scenario(scenario_name: str) -> List[Dict[str, Any]]:
        """Load specific test scenario data."""
        scenarios = {
            'normal_operation': [
                {'fuel_flow': 10.0, 'air_flow': 120.0, 'o2_level': 3.0},
                {'fuel_flow': 10.5, 'air_flow': 125.0, 'o2_level': 3.2},
                {'fuel_flow': 9.8, 'air_flow': 118.0, 'o2_level': 2.9},
            ],
            'high_emissions': [
                {'fuel_flow': 12.0, 'air_flow': 100.0, 'o2_level': 1.5},
                {'fuel_flow': 11.5, 'air_flow': 105.0, 'o2_level': 1.8},
                {'fuel_flow': 12.5, 'air_flow': 95.0, 'o2_level': 1.2},
            ],
            'low_efficiency': [
                {'fuel_flow': 10.0, 'air_flow': 150.0, 'o2_level': 5.5},
                {'fuel_flow': 10.5, 'air_flow': 155.0, 'o2_level': 5.8},
                {'fuel_flow': 9.5, 'air_flow': 145.0, 'o2_level': 5.2},
            ],
            'unstable_flame': [
                {'fuel_flow': 8.0, 'air_flow': 140.0, 'o2_level': 4.5},
                {'fuel_flow': 7.5, 'air_flow': 145.0, 'o2_level': 4.8},
                {'fuel_flow': 8.5, 'air_flow': 135.0, 'o2_level': 4.2},
            ]
        }

        return scenarios.get(scenario_name, [])

    return load_scenario


# Fixture: Performance Monitor
@pytest.fixture
def performance_monitor():
    """Monitor test performance metrics."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}

        @contextmanager
        def measure(self, operation: str):
            """Measure operation performance."""
            start_time = time.perf_counter()

            yield

            elapsed = (time.perf_counter() - start_time) * 1000  # ms

            if operation not in self.metrics:
                self.metrics[operation] = []

            self.metrics[operation].append(elapsed)

        def get_stats(self, operation: str) -> Dict[str, float]:
            """Get performance statistics."""
            if operation not in self.metrics:
                return {}

            values = self.metrics[operation]
            return {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }

    return PerformanceMonitor()


# Fixture: Async Test Helper
@pytest.fixture
def async_helper():
    """Helper for async test operations."""
    class AsyncHelper:
        @staticmethod
        async def wait_for_condition(condition, timeout: float = 5.0, interval: float = 0.1):
            """Wait for condition to become true."""
            start = time.time()
            while time.time() - start < timeout:
                if await condition() if asyncio.iscoroutinefunction(condition) else condition():
                    return True
                await asyncio.sleep(interval)
            return False

        @staticmethod
        async def gather_with_timeout(tasks: List, timeout: float = 10.0):
            """Gather tasks with timeout."""
            return await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=timeout
            )

    return AsyncHelper()


# Session-wide fixtures
@pytest.fixture(scope='session')
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()