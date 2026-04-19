"""
Integration Test Fixtures for GL-001 ThermalCommand.

Additional fixtures specific to integration tests.
These supplement the global fixtures in tests/conftest.py.

Author: GreenLang QA Team
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, MagicMock


# =============================================================================
# Async Test Configuration
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_timeout():
    """Provide async timeout configuration."""
    return {
        "short": 1.0,
        "medium": 5.0,
        "long": 30.0,
    }


# =============================================================================
# Mock External System Fixtures
# =============================================================================

@pytest.fixture
def mock_opc_ua_connection():
    """Provide mock OPC-UA connection."""
    connection = AsyncMock()
    connection.connect = AsyncMock(return_value=True)
    connection.disconnect = AsyncMock(return_value=True)
    connection.read_node = AsyncMock(return_value={"value": 450.0, "quality": "Good"})
    connection.write_node = AsyncMock(return_value=True)
    connection.is_connected = True
    return connection


@pytest.fixture
def mock_mqtt_client():
    """Provide mock MQTT client."""
    client = MagicMock()
    client.connect = MagicMock(return_value=True)
    client.disconnect = MagicMock(return_value=True)
    client.publish = MagicMock(return_value=None)
    client.subscribe = MagicMock(return_value=None)
    return client


@pytest.fixture
def mock_kafka_client():
    """Provide mock Kafka client."""
    client = MagicMock()
    client.produce = MagicMock(return_value=None)
    client.consume = MagicMock(return_value=[])
    client.flush = MagicMock(return_value=None)
    return client


# =============================================================================
# Network Partition Fixtures
# =============================================================================

@pytest.fixture
def network_partition_scenario():
    """Provide network partition test scenario."""
    return {
        "reachable_equipment": ["BOILER-001", "BOILER-002"],
        "unreachable_equipment": ["BOILER-003", "CHP-001"],
        "partition_duration_seconds": 30,
        "expected_behavior": "operate_with_available",
    }


@pytest.fixture
def stale_data_scenario():
    """Provide stale data handling scenario."""
    return {
        "max_age_seconds": 10.0,
        "stale_tags": [
            {
                "tag_id": "TI-101",
                "value": 450.0,
                "timestamp": datetime.now(timezone.utc),
                "age_seconds": 5.0,  # Fresh
            },
            {
                "tag_id": "TI-102",
                "value": 445.0,
                "timestamp": datetime.now(timezone.utc),
                "age_seconds": 15.0,  # Stale
            },
        ],
    }


# =============================================================================
# Multi-Equipment Coordination Fixtures
# =============================================================================

@pytest.fixture
def equipment_fleet():
    """Provide a fleet of equipment for coordination tests."""
    return [
        {"equipment_id": "BOILER-001", "type": "boiler", "capacity": 50.0, "cost": 5.0},
        {"equipment_id": "BOILER-002", "type": "boiler", "capacity": 40.0, "cost": 5.5},
        {"equipment_id": "BOILER-003", "type": "boiler", "capacity": 30.0, "cost": 6.0},
        {"equipment_id": "CHP-001", "type": "chp", "capacity": 30.0, "cost": 3.0},
    ]


@pytest.fixture
def demand_scenarios():
    """Provide various demand scenarios for testing."""
    return {
        "low_demand": 30.0,
        "normal_demand": 80.0,
        "high_demand": 120.0,
        "exceeds_capacity": 200.0,
    }


# =============================================================================
# Orchestrator Fixtures
# =============================================================================

@pytest.fixture
def orchestrator_config():
    """Provide orchestrator configuration."""
    return {
        "name": "TestOrchestrator",
        "orchestrator_id": "ORCH-TEST-001",
        "version": "1.0.0",
        "safety_level": "SIL_2",
        "heartbeat_interval_ms": 1000,
    }


@pytest.fixture
def workflow_specs():
    """Provide various workflow specifications."""
    return {
        "simple": {
            "workflow_id": "WF-SIMPLE",
            "name": "Simple Workflow",
            "tasks": [{"task_id": "T1", "action": "read"}],
            "required_agents": [],
        },
        "optimization": {
            "workflow_id": "WF-OPT",
            "name": "Optimization Workflow",
            "tasks": [
                {"task_id": "T1", "action": "collect_data"},
                {"task_id": "T2", "action": "optimize"},
                {"task_id": "T3", "action": "deploy"},
            ],
            "required_agents": ["GL-002"],
        },
        "safety_check": {
            "workflow_id": "WF-SAFETY",
            "name": "Safety Check Workflow",
            "tasks": [
                {"task_id": "T1", "action": "check_boundaries"},
                {"task_id": "T2", "action": "validate_interlocks"},
            ],
            "required_agents": [],
        },
    }


# =============================================================================
# Event Handling Fixtures
# =============================================================================

@pytest.fixture
def event_scenarios():
    """Provide event handling scenarios."""
    return {
        "safety_alarm": {
            "event_type": "SAFETY_ALARM",
            "source": "TI-101",
            "priority": "HIGH",
            "payload": {"temperature": 550.0, "limit": 500.0},
        },
        "equipment_fault": {
            "event_type": "EQUIPMENT_FAULT",
            "source": "BOILER-001",
            "priority": "CRITICAL",
            "payload": {"fault_code": "E001", "description": "High temperature"},
        },
        "optimization_complete": {
            "event_type": "OPTIMIZATION_COMPLETE",
            "source": "MILP-001",
            "priority": "NORMAL",
            "payload": {"job_id": "OPT-001", "status": "success"},
        },
    }


# =============================================================================
# Concurrent Operation Fixtures
# =============================================================================

@pytest.fixture
def concurrent_requests():
    """Provide concurrent operation scenarios."""
    return {
        "num_concurrent": 10,
        "requests": [
            {"request_id": f"REQ-{i:03d}", "demand": 50.0 + i * 5}
            for i in range(10)
        ],
    }
