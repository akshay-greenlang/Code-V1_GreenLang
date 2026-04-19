"""
GreenLang Framework - Tool Discovery Service Tests

Comprehensive test suite for the tool discovery service including:
- Tool registration and unregistration
- Capability matching
- Health monitoring
- Circuit breaker patterns
- Metrics collection
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Import modules under test
import sys
from pathlib import Path

# Add parent paths for imports
_framework_path = Path(__file__).parent.parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    MCPTool,
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    SecurityLevel,
    ExecutionMode,
    ToolCallRequest,
    ToolCallResponse,
)
from tools.tool_discovery import (
    # Enums
    ToolHealthStatus,
    CircuitState,
    CapabilityType,
    # Data models
    ToolCapability,
    ToolMetrics,
    ToolHealthCheck,
    CircuitBreaker,
    ToolRegistration,
    # Main service
    ToolDiscoveryService,
    # Decorators and builders
    discoverable_tool,
    CapabilityBuilder,
    capability,
    # Global functions
    get_discovery_service,
    register_tool_globally,
)


# =============================================================================
# MOCK TOOL FOR TESTING
# =============================================================================

class MockCalculatorTool(MCPTool):
    """Mock calculator tool for testing."""

    def __init__(self, should_fail: bool = False):
        definition = ToolDefinition(
            name="mock_calculator",
            description="A mock calculator for testing",
            parameters=[
                ToolParameter(
                    name="value",
                    type="number",
                    description="Input value",
                    required=True,
                ),
            ],
            category=ToolCategory.CALCULATOR,
            security_level=SecurityLevel.READ_ONLY,
        )
        super().__init__(definition)
        self.should_fail = should_fail
        self.call_count = 0

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute the mock calculation."""
        self.call_count += 1
        if self.should_fail:
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error="Simulated failure",
            )
        return ToolCallResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            success=True,
            result={"value": request.arguments.get("value", 0) * 2},
        )


class MockConnectorTool(MCPTool):
    """Mock connector tool for testing."""

    def __init__(self):
        definition = ToolDefinition(
            name="mock_connector",
            description="A mock connector for testing data integration",
            parameters=[
                ToolParameter(
                    name="endpoint",
                    type="string",
                    description="Endpoint URL",
                    required=True,
                ),
            ],
            category=ToolCategory.CONNECTOR,
            security_level=SecurityLevel.CONTROLLED_WRITE,
        )
        super().__init__(definition)

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute the mock connection."""
        return ToolCallResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            success=True,
            result={"connected": True},
        )


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def discovery_service():
    """Create a fresh discovery service for each test."""
    return ToolDiscoveryService()


@pytest.fixture
def mock_tool():
    """Create a mock calculator tool."""
    return MockCalculatorTool()


@pytest.fixture
def mock_connector():
    """Create a mock connector tool."""
    return MockConnectorTool()


@pytest.fixture
def failing_tool():
    """Create a tool that always fails."""
    return MockCalculatorTool(should_fail=True)


# =============================================================================
# CAPABILITY MODEL TESTS
# =============================================================================

class TestToolCapability:
    """Test ToolCapability matching."""

    def test_exact_name_match(self):
        """Test matching by exact capability name."""
        cap = ToolCapability(
            name="calculate_efficiency",
            capability_type=CapabilityType.CALCULATION,
            description="Calculate boiler efficiency",
            input_types=["number"],
            output_types=["number"],
            keywords=["efficiency", "boiler", "combustion"],
        )

        # Exact name match should score high
        score = cap.matches("calculate_efficiency")
        assert score >= 0.8

    def test_keyword_match(self):
        """Test matching by keywords."""
        cap = ToolCapability(
            name="calculate_efficiency",
            capability_type=CapabilityType.CALCULATION,
            description="Calculate boiler efficiency",
            input_types=["number"],
            output_types=["number"],
            keywords=["efficiency", "boiler", "combustion"],
        )

        score = cap.matches("I need to calculate boiler efficiency")
        assert score > 0.5

    def test_no_match(self):
        """Test no match returns zero."""
        cap = ToolCapability(
            name="calculate_efficiency",
            capability_type=CapabilityType.CALCULATION,
            description="Calculate boiler efficiency",
            input_types=["number"],
            output_types=["number"],
            keywords=["efficiency", "boiler", "combustion"],
        )

        score = cap.matches("what is the weather today")
        assert score == 0.0


# =============================================================================
# CAPABILITY BUILDER TESTS
# =============================================================================

class TestCapabilityBuilder:
    """Test capability builder pattern."""

    def test_build_capability(self):
        """Test building a capability using builder."""
        cap = (
            capability("steam_calculation")
            .with_type(CapabilityType.CALCULATION)
            .with_description("Calculate steam properties")
            .with_inputs("pressure", "temperature")
            .with_outputs("enthalpy", "entropy")
            .with_keywords("steam", "thermodynamic", "IAPWS")
            .with_confidence(0.95)
            .build()
        )

        assert cap.name == "steam_calculation"
        assert cap.capability_type == CapabilityType.CALCULATION
        assert "pressure" in cap.input_types
        assert "steam" in cap.keywords
        assert cap.confidence_score == 0.95


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestToolMetrics:
    """Test metrics collection."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = ToolMetrics(tool_name="test_tool")

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.success_rate == 1.0  # No failures yet
        assert metrics.error_rate == 0.0

    def test_record_successful_call(self):
        """Test recording successful call."""
        metrics = ToolMetrics(tool_name="test_tool")

        metrics.record_call(success=True, execution_time_ms=50.0)

        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.avg_execution_time_ms == 50.0
        assert metrics.success_rate == 1.0

    def test_record_failed_call(self):
        """Test recording failed call."""
        metrics = ToolMetrics(tool_name="test_tool")

        metrics.record_call(success=False, execution_time_ms=100.0, error="Test error")

        assert metrics.total_calls == 1
        assert metrics.failed_calls == 1
        assert metrics.error_rate == 1.0
        assert metrics.last_error == "Test error"

    def test_multiple_calls(self):
        """Test metrics with multiple calls."""
        metrics = ToolMetrics(tool_name="test_tool")

        metrics.record_call(success=True, execution_time_ms=40.0)
        metrics.record_call(success=True, execution_time_ms=60.0)
        metrics.record_call(success=False, execution_time_ms=100.0, error="Timeout")

        assert metrics.total_calls == 3
        assert metrics.successful_calls == 2
        assert metrics.failed_calls == 1
        assert metrics.success_rate == pytest.approx(0.667, rel=0.01)
        assert metrics.avg_execution_time_ms == 50.0  # (40+60)/2


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker(tool_name="test", failure_threshold=3)

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_trips_after_threshold(self):
        """Test circuit trips after failure threshold."""
        cb = CircuitBreaker(tool_name="test", failure_threshold=3)

        cb.record_failure()
        assert cb.can_execute() is True

        cb.record_failure()
        assert cb.can_execute() is True

        cb.record_failure()  # Threshold reached
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_counter(self):
        """Test success resets failure counter."""
        cb = CircuitBreaker(tool_name="test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Reset

        cb.record_failure()
        cb.record_failure()
        # Still closed because success reset counter
        assert cb.state == CircuitState.CLOSED

    def test_recovery_after_timeout(self):
        """Test circuit recovers after timeout."""
        cb = CircuitBreaker(
            tool_name="test",
            failure_threshold=2,
            recovery_timeout_seconds=1  # Short timeout for testing
        )

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

        # Wait for recovery timeout
        time.sleep(1.1)

        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """Test successful calls in half-open state close circuit."""
        cb = CircuitBreaker(
            tool_name="test",
            failure_threshold=2,
            recovery_timeout_seconds=0,  # Immediate recovery
            half_open_max_calls=2,
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()

        # Force to half-open
        cb._transition_to(CircuitState.HALF_OPEN)

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_opens(self):
        """Test failure in half-open state re-opens circuit."""
        cb = CircuitBreaker(
            tool_name="test",
            failure_threshold=2,
            recovery_timeout_seconds=0,
        )

        # Trip and go to half-open
        cb.record_failure()
        cb.record_failure()
        cb._transition_to(CircuitState.HALF_OPEN)

        # Failure in half-open
        cb.record_failure()

        assert cb.state == CircuitState.OPEN


# =============================================================================
# DISCOVERY SERVICE TESTS
# =============================================================================

class TestToolDiscoveryService:
    """Test tool discovery service."""

    def test_register_tool(self, discovery_service, mock_tool):
        """Test tool registration."""
        discovery_service.register_tool(mock_tool)

        registration = discovery_service.get_tool("mock_calculator")
        assert registration is not None
        assert registration.definition.name == "mock_calculator"

    def test_unregister_tool(self, discovery_service, mock_tool):
        """Test tool unregistration."""
        discovery_service.register_tool(mock_tool)
        result = discovery_service.unregister_tool("mock_calculator")

        assert result is True
        assert discovery_service.get_tool("mock_calculator") is None

    def test_unregister_nonexistent(self, discovery_service):
        """Test unregistering non-existent tool."""
        result = discovery_service.unregister_tool("nonexistent")
        assert result is False

    def test_list_tools(self, discovery_service, mock_tool, mock_connector):
        """Test listing all tools."""
        discovery_service.register_tool(mock_tool)
        discovery_service.register_tool(mock_connector)

        tools = discovery_service.list_tools()
        assert len(tools) == 2

    def test_register_with_capabilities(self, discovery_service, mock_tool):
        """Test registration with custom capabilities."""
        caps = [
            ToolCapability(
                name="calculation",
                capability_type=CapabilityType.CALCULATION,
                description="Perform calculations",
                input_types=["number"],
                output_types=["number"],
                keywords=["calculate", "compute", "math"],
            )
        ]

        discovery_service.register_tool(mock_tool, capabilities=caps)

        registration = discovery_service.get_tool("mock_calculator")
        assert len(registration.capabilities) == 1

    def test_find_by_capability(self, discovery_service, mock_tool, mock_connector):
        """Test finding tools by capability."""
        calc_caps = [
            ToolCapability(
                name="calculation",
                capability_type=CapabilityType.CALCULATION,
                description="Perform calculations",
                input_types=["number"],
                output_types=["number"],
                keywords=["calculate", "compute"],
            )
        ]
        conn_caps = [
            ToolCapability(
                name="integration",
                capability_type=CapabilityType.INTEGRATION,
                description="Connect to systems",
                input_types=["string"],
                output_types=["object"],
                keywords=["connect", "integrate"],
            )
        ]

        discovery_service.register_tool(mock_tool, capabilities=calc_caps)
        discovery_service.register_tool(mock_connector, capabilities=conn_caps)

        results = discovery_service.find_tools_by_capability("calculate something")
        assert len(results) > 0
        assert results[0][0].definition.name == "mock_calculator"

    def test_find_by_category(self, discovery_service, mock_tool, mock_connector):
        """Test finding tools by category."""
        discovery_service.register_tool(mock_tool)
        discovery_service.register_tool(mock_connector)

        calculators = discovery_service.find_tools_by_category(ToolCategory.CALCULATOR)
        connectors = discovery_service.find_tools_by_category(ToolCategory.CONNECTOR)

        assert len(calculators) == 1
        assert len(connectors) == 1

    def test_find_by_tag(self, discovery_service, mock_tool):
        """Test finding tools by tag."""
        discovery_service.register_tool(mock_tool, tags=["thermal", "efficiency"])

        thermal_tools = discovery_service.find_tools_by_tag("thermal")
        assert len(thermal_tools) == 1

    def test_invoke_with_routing(self, discovery_service, mock_tool):
        """Test invoking tool through discovery service."""
        discovery_service.register_tool(mock_tool)

        response = discovery_service.invoke_with_routing(
            "mock_calculator",
            {"value": 21},
            caller_agent_id="test"
        )

        assert response.success is True
        assert response.result["value"] == 42

    def test_invoke_updates_metrics(self, discovery_service, mock_tool):
        """Test that invocation updates metrics."""
        discovery_service.register_tool(mock_tool)

        discovery_service.invoke_with_routing("mock_calculator", {"value": 10})
        discovery_service.invoke_with_routing("mock_calculator", {"value": 20})

        metrics = discovery_service.get_all_metrics()
        assert metrics["mock_calculator"]["total_calls"] == 2
        assert metrics["mock_calculator"]["successful_calls"] == 2

    def test_invoke_with_circuit_breaker(self, discovery_service, failing_tool):
        """Test circuit breaker protection during invocation."""
        discovery_service.register_tool(failing_tool)

        # Make calls until circuit trips
        for _ in range(5):
            discovery_service.invoke_with_routing("mock_calculator", {"value": 1})

        registration = discovery_service.get_tool("mock_calculator")
        # Circuit should be open after failures
        assert registration.circuit_breaker.state == CircuitState.OPEN

        # Next call should fail due to circuit breaker
        response = discovery_service.invoke_with_routing("mock_calculator", {"value": 1})
        assert response.success is False
        assert "Circuit breaker" in response.error

    def test_health_check(self, discovery_service, mock_tool):
        """Test health checking."""
        discovery_service.register_tool(mock_tool)

        health = discovery_service.check_tool_health("mock_calculator")

        assert health.tool_name == "mock_calculator"
        assert health.status in [ToolHealthStatus.HEALTHY, ToolHealthStatus.UNKNOWN]

    def test_get_all_health(self, discovery_service, mock_tool, mock_connector):
        """Test getting health for all tools."""
        discovery_service.register_tool(mock_tool)
        discovery_service.register_tool(mock_connector)

        health = discovery_service.get_all_health()
        assert "mock_calculator" in health
        assert "mock_connector" in health

    def test_service_status(self, discovery_service, mock_tool):
        """Test getting overall service status."""
        discovery_service.register_tool(mock_tool)
        discovery_service.invoke_with_routing("mock_calculator", {"value": 5})

        status = discovery_service.get_service_status()

        assert status["total_tools"] == 1
        assert status["total_calls"] == 1


# =============================================================================
# FALLBACK TESTS
# =============================================================================

class TestFallbackRouting:
    """Test fallback routing functionality."""

    def test_fallback_on_failure(self, discovery_service):
        """Test fallback tool is used on primary failure."""
        failing = MockCalculatorTool(should_fail=True)
        failing.definition = ToolDefinition(
            name="primary_tool",
            description="Primary tool",
            parameters=[],
            category=ToolCategory.CALCULATOR,
        )

        backup = MockCalculatorTool(should_fail=False)
        backup.definition = ToolDefinition(
            name="backup_tool",
            description="Backup tool",
            parameters=[ToolParameter(name="value", type="number", description="Value")],
            category=ToolCategory.CALCULATOR,
        )

        discovery_service.register_tool(failing)
        discovery_service.register_tool(backup)

        # Primary fails, but we don't have fallback specified in this call
        response = discovery_service.invoke_with_routing(
            "primary_tool",
            {},
        )
        assert response.success is False

    def test_fallback_with_circuit_open(self, discovery_service):
        """Test fallback when circuit breaker is open."""
        primary = MockCalculatorTool(should_fail=True)
        primary.definition = ToolDefinition(
            name="primary",
            description="Primary",
            parameters=[],
            category=ToolCategory.CALCULATOR,
        )

        fallback = MockCalculatorTool(should_fail=False)
        fallback.definition = ToolDefinition(
            name="fallback",
            description="Fallback",
            parameters=[ToolParameter(name="value", type="number", description="Val")],
            category=ToolCategory.CALCULATOR,
        )

        discovery_service.register_tool(primary)
        discovery_service.register_tool(fallback)

        # Trip the circuit
        for _ in range(5):
            discovery_service.invoke_with_routing("primary", {})

        # Now invoke with fallback
        response = discovery_service.invoke_with_routing(
            "primary",
            {"value": 10},
            fallback_tool="fallback"
        )

        assert response.success is True
        assert response.result["value"] == 20


# =============================================================================
# SERVICE LIFECYCLE TESTS
# =============================================================================

class TestServiceLifecycle:
    """Test service start/stop lifecycle."""

    def test_start_stop(self, discovery_service):
        """Test starting and stopping the service."""
        discovery_service.start()
        assert discovery_service._running is True

        discovery_service.stop()
        assert discovery_service._running is False

    def test_double_start(self, discovery_service):
        """Test starting twice doesn't create issues."""
        discovery_service.start()
        discovery_service.start()  # Should be no-op

        assert discovery_service._running is True
        discovery_service.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
