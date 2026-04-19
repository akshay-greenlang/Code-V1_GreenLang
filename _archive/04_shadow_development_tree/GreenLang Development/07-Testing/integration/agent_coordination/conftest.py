# -*- coding: utf-8 -*-
"""
Shared fixtures for agent coordination integration tests.

This module provides reusable pytest fixtures for testing inter-agent
coordination scenarios across GL-001 through GL-010 agents.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock

import pytest


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment for agent coordination tests."""
    # Set ephemeral signing mode for tests
    os.environ['GL_SIGNING_MODE'] = 'ephemeral'

    # Set test mode flag
    os.environ['GL_TEST_MODE'] = 'true'

    yield

    # Cleanup
    os.environ.pop('GL_TEST_MODE', None)


# ============================================================================
# Mock Agent Factory
# ============================================================================

class MockAgentFactory:
    """Factory for creating mock agents with standard interfaces."""

    @staticmethod
    def create_mock_agent(agent_id: str, agent_name: str, version: str = "1.0.0"):
        """
        Create a mock agent with standard configuration.

        Args:
            agent_id: Agent identifier (e.g., 'GL-001')
            agent_name: Agent name (e.g., 'ProcessHeatOrchestrator')
            version: Agent version

        Returns:
            Mock agent instance
        """
        mock_agent = MagicMock()
        mock_agent.config = MagicMock()
        mock_agent.config.agent_id = agent_id
        mock_agent.config.agent_name = agent_name
        mock_agent.config.version = version

        mock_agent.state = 'ready'
        mock_agent.performance_metrics = {
            'executions': 0,
            'avg_execution_time_ms': 0,
            'success_rate': 100.0
        }

        # Add standard async execute method
        async def mock_execute(input_data):
            return {
                'agent_id': agent_id,
                'timestamp': '2025-12-01T00:00:00Z',
                'execution_time_ms': 42.0,
                'success': True,
                'data': {}
            }

        mock_agent.execute = mock_execute

        return mock_agent


@pytest.fixture
def mock_agent_factory():
    """Provide mock agent factory."""
    return MockAgentFactory()


# ============================================================================
# Agent Message Bus
# ============================================================================

class MockMessageBus:
    """Mock message bus for agent communication testing."""

    def __init__(self):
        self.messages = []
        self.subscriptions = {}

    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message to topic."""
        self.messages.append({
            'topic': topic,
            'message': message,
            'timestamp': '2025-12-01T00:00:00Z'
        })

        # Notify subscribers
        if topic in self.subscriptions:
            for callback in self.subscriptions[topic]:
                await callback(message)

    async def subscribe(self, topic: str, callback):
        """Subscribe to topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)

    def get_messages(self, topic: str = None) -> List[Dict[str, Any]]:
        """Get messages for topic."""
        if topic is None:
            return self.messages
        return [m for m in self.messages if m['topic'] == topic]

    def clear(self):
        """Clear all messages."""
        self.messages = []


@pytest.fixture
def mock_message_bus():
    """Provide mock message bus."""
    return MockMessageBus()


# ============================================================================
# Coordination Test Helpers
# ============================================================================

class CoordinationTestHelpers:
    """Helper functions for coordination testing."""

    @staticmethod
    def assert_coordination_success(result: Dict[str, Any]):
        """Assert coordination was successful."""
        assert result is not None
        assert 'status' in result
        assert result['status'] in ['success', 'partial_success']

    @staticmethod
    def assert_message_format_valid(message: Dict[str, Any]):
        """Assert message follows standard format."""
        required_fields = ['agent_id', 'timestamp']
        for field in required_fields:
            assert field in message, f"Missing required field: {field}"

    @staticmethod
    def assert_provenance_tracked(result: Dict[str, Any]):
        """Assert provenance hash is tracked."""
        assert 'provenance_hash' in result or 'hash' in result
        hash_value = result.get('provenance_hash') or result.get('hash')
        assert hash_value is not None
        assert len(str(hash_value)) > 0

    @staticmethod
    async def measure_coordination_latency(coordination_func, *args, **kwargs):
        """Measure coordination latency."""
        import time

        start_time = time.perf_counter()
        result = await coordination_func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return result, latency_ms


@pytest.fixture
def coordination_helpers():
    """Provide coordination test helpers."""
    return CoordinationTestHelpers()


# ============================================================================
# Sample Data Generators
# ============================================================================

class SampleDataGenerator:
    """Generate sample data for coordination tests."""

    @staticmethod
    def generate_sensor_data(num_sensors: int = 10) -> Dict[str, float]:
        """Generate sample sensor readings."""
        return {
            f'sensor_{i}': 100.0 + (i * 10.0)
            for i in range(num_sensors)
        }

    @staticmethod
    def generate_thermal_data() -> Dict[str, Any]:
        """Generate sample thermal system data."""
        return {
            'heat_input_mw': 10.0,
            'useful_heat_output_mw': 8.5,
            'inlet_temperature_k': 500.0,
            'outlet_temperature_k': 380.0,
            'mass_flow_rate_kg_s': 15.0,
            'pressure_bar': 40.0
        }

    @staticmethod
    def generate_emissions_data() -> Dict[str, Any]:
        """Generate sample emissions data."""
        return {
            'NOx_emissions_ppm': 140,
            'SOx_emissions_ppm': 180,
            'CO2_emissions_kg_mwh': 450,
            'CO_emissions_ppm': 60,
            'particulate_matter_mg_nm3': 15
        }

    @staticmethod
    def generate_steam_data() -> Dict[str, Any]:
        """Generate sample steam system data."""
        return {
            'pressure_readings': [
                {
                    'location': f'point_{i}',
                    'pressure_bar': 40.0 - (i * 2.0),
                    'target_pressure_bar': 40.0,
                    'temperature_c': 450 - (i * 10)
                }
                for i in range(5)
            ],
            'flow_rates_kg_hr': {
                'header_main': 45000,
                'branch_1': 18000,
                'branch_2': 22000
            }
        }


@pytest.fixture
def sample_data_generator():
    """Provide sample data generator."""
    return SampleDataGenerator()


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for coordination tests."""
    return {
        'max_coordination_latency_ms': 200,
        'min_throughput_per_second': 10,
        'max_memory_increase_mb': 100,
        'min_success_rate_percent': 95.0
    }


@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        'num_concurrent_requests': 50,
        'num_sequential_cycles': 30,
        'timeout_seconds': 10
    }


# ============================================================================
# Async Test Helpers
# ============================================================================

@pytest.fixture
def anyio_backend():
    """Async backend for tests."""
    return "asyncio"


@pytest.fixture
async def async_test_timeout():
    """Timeout for async tests."""
    return 5.0  # 5 seconds


# ============================================================================
# Mock SCADA/ERP Data
# ============================================================================

@pytest.fixture
def mock_scada_data():
    """Mock SCADA system data."""
    return {
        'tags': {
            'TEMP_001': 450.5,
            'PRESS_001': 40.2,
            'FLOW_001': 12500.0,
            'LEVEL_001': 75.3
        },
        'timestamp': '2025-12-01T00:00:00Z',
        'quality': 'good'
    }


@pytest.fixture
def mock_erp_data():
    """Mock ERP system data."""
    return {
        'production_schedule': {
            'shift_1': {'target_output': 50000, 'priority': 'high'},
            'shift_2': {'target_output': 45000, 'priority': 'normal'}
        },
        'inventory': {
            'fuel_natural_gas_kg': 100000,
            'water_treatment_chemicals_kg': 500
        },
        'costs': {
            'fuel_usd_per_kg': 0.50,
            'electricity_usd_per_kwh': 0.12,
            'steam_usd_per_kg': 0.015
        }
    }


# ============================================================================
# Validation Helpers
# ============================================================================

class ValidationHelpers:
    """Helper functions for validation."""

    @staticmethod
    def validate_efficiency_range(efficiency: float):
        """Validate efficiency is in valid range."""
        assert 0 <= efficiency <= 100, f"Efficiency {efficiency} out of range [0, 100]"

    @staticmethod
    def validate_temperature_kelvin(temp_k: float):
        """Validate temperature is above absolute zero."""
        assert temp_k > 0, f"Temperature {temp_k}K must be positive"

    @staticmethod
    def validate_pressure_positive(pressure_bar: float):
        """Validate pressure is positive."""
        assert pressure_bar > 0, f"Pressure {pressure_bar} bar must be positive"

    @staticmethod
    def validate_emissions_limits(emissions: Dict[str, float], limits: Dict[str, float]):
        """Validate emissions are within limits."""
        for pollutant, limit in limits.items():
            pollutant_key = pollutant.replace('_limit', '')
            if pollutant_key in emissions:
                actual = emissions[pollutant_key]
                assert actual <= limit, f"{pollutant_key}: {actual} exceeds limit {limit}"


@pytest.fixture
def validation_helpers():
    """Provide validation helpers."""
    return ValidationHelpers()


# ============================================================================
# Test Data Fixtures Directory
# ============================================================================

@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def ensure_test_data_dir(test_data_dir):
    """Ensure test data directory exists."""
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir


# ============================================================================
# Mock External Dependencies
# ============================================================================

@pytest.fixture
def mock_external_api():
    """Mock external API client."""
    mock_api = MagicMock()

    async def mock_get(endpoint: str):
        return {'status': 'success', 'data': {}}

    async def mock_post(endpoint: str, data: Dict[str, Any]):
        return {'status': 'success', 'id': 'mock-123'}

    mock_api.get = mock_get
    mock_api.post = mock_post

    return mock_api


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_async_tasks():
    """Cleanup async tasks after each test."""
    yield

    # Cancel any remaining tasks
    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    for task in tasks:
        task.cancel()

    # Wait for cancellation
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
