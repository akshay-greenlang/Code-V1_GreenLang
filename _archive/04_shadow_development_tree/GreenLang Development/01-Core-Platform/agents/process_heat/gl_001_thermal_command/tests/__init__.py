"""
GL-001 ThermalCommand Orchestrator - Test Suite

Comprehensive test suite for the ThermalCommand Orchestrator agent,
achieving 85%+ code coverage across all modules.

Test Categories:
    - Unit Tests: Individual function/method testing
    - Integration Tests: Module interaction testing
    - Performance Tests: Throughput and latency validation
    - Compliance Tests: Regulatory requirement validation

Test Coverage Targets:
    - config.py: 90%+
    - schemas.py: 90%+
    - orchestrator.py: 85%+
    - handlers.py: 85%+
    - coordinators.py: 85%+
    - load_allocation.py: 90%+
    - cascade_control.py: 90%+
    - sis_integration.py: 90%+
    - cmms_integration.py: 85%+

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Test markers
UNIT_TEST = pytest.mark.unit
INTEGRATION_TEST = pytest.mark.integration
PERFORMANCE_TEST = pytest.mark.performance
COMPLIANCE_TEST = pytest.mark.compliance


# =============================================================================
# SHARED TEST UTILITIES
# =============================================================================

class TestDataFactory:
    """Factory for generating test data."""

    @staticmethod
    def create_equipment_data(
        equipment_id: str = "BLR-001",
        equipment_type: str = "boiler",
        temperature: float = 450.0,
        pressure: float = 150.0
    ) -> Dict[str, Any]:
        """Create test equipment data."""
        return {
            "equipment_id": equipment_id,
            "equipment_type": equipment_type,
            "measurements": {
                "temperature": temperature,
                "pressure": pressure,
                "flow_rate": 100.0,
            },
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def create_thermal_load_request(
        total_demand_mw: float = 50.0,
        equipment_ids: Optional[List[str]] = None,
        optimization_mode: str = "efficiency"
    ) -> Dict[str, Any]:
        """Create test thermal load request."""
        return {
            "total_demand_mw": total_demand_mw,
            "equipment_ids": equipment_ids or ["BLR-001", "BLR-002", "BLR-003"],
            "optimization_mode": optimization_mode,
            "constraints": {
                "min_efficiency": 0.80,
                "max_emissions": 100.0,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def create_safety_event(
        event_type: str = "HIGH_TEMPERATURE",
        severity: str = "WARNING",
        equipment_id: str = "BLR-001",
        value: float = 550.0,
        threshold: float = 500.0
    ) -> Dict[str, Any]:
        """Create test safety event."""
        return {
            "event_type": event_type,
            "severity": severity,
            "equipment_id": equipment_id,
            "value": value,
            "threshold": threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def create_workflow_spec(
        workflow_type: str = "OPTIMIZATION",
        name: str = "Test Workflow",
        priority: str = "NORMAL"
    ) -> Dict[str, Any]:
        """Create test workflow specification."""
        return {
            "workflow_type": workflow_type,
            "name": name,
            "priority": priority,
            "timeout_s": 300.0,
            "parameters": {},
        }


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self):
        self._registered_agents = {}
        self._workflows = {}
        self._safety_state = "NORMAL"
        self._esd_triggered = False

    def register_agent(self, registration):
        self._registered_agents[registration.agent_id] = registration
        return True

    def deregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._registered_agents:
            del self._registered_agents[agent_id]
            return True
        return False

    def get_agent_status(self, agent_id: str):
        return self._registered_agents.get(agent_id)


class AsyncContextManager:
    """Async context manager helper for testing."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_provenance_hash(hash_value: str, length: int = 64) -> None:
    """Assert provenance hash is valid SHA-256."""
    assert hash_value is not None, "Provenance hash is None"
    assert len(hash_value) == length, f"Expected hash length {length}, got {len(hash_value)}"
    assert all(c in '0123456789abcdef' for c in hash_value.lower()), "Invalid hash characters"


def assert_timestamp_recent(timestamp: datetime, max_age_seconds: float = 60.0) -> None:
    """Assert timestamp is recent."""
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = (now - timestamp).total_seconds()
    assert age < max_age_seconds, f"Timestamp too old: {age}s > {max_age_seconds}s"


def assert_within_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> None:
    """Assert value is within range."""
    assert min_val <= value <= max_val, f"{name} {value} not in range [{min_val}, {max_val}]"


def assert_valid_output(output: Any) -> None:
    """Assert control output is valid."""
    assert output is not None, "Output is None"
    assert hasattr(output, 'output'), "Output missing 'output' attribute"
    assert 0.0 <= output.output <= 100.0, f"Output {output.output} not in range [0, 100]"


# =============================================================================
# PYTEST FIXTURES (Shared)
# =============================================================================

@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory()


@pytest.fixture
def mock_orchestrator():
    """Provide mock orchestrator."""
    return MockOrchestrator()


@pytest.fixture
def sample_timestamp():
    """Provide sample timestamp."""
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_equipment_list():
    """Provide sample equipment list."""
    return ["BLR-001", "BLR-002", "BLR-003", "HTR-001", "HTR-002"]
