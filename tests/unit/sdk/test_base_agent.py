# -*- coding: utf-8 -*-
"""
Comprehensive tests for GreenLang BaseAgent framework.

Tests cover:
- Agent initialization and configuration
- Lifecycle management (init, validate, execute, cleanup)
- Metrics collection (StatsTracker)
- Pre/post execution hooks
- Error handling and recovery
- Resource loading and caching
- Configuration validation
- Concurrent execution
- Edge cases and boundary conditions
"""

import pytest
import time
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from greenlang.agents.base import (
from greenlang.determinism import DeterministicClock
    BaseAgent,
    AgentConfig,
    AgentResult,
    AgentMetrics,
    StatsTracker,
)


# Test Agent Implementations
class SimpleAgent(BaseAgent):
    """Simple agent for testing basic functionality."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute simple doubling operation."""
        value = input_data.get("value", 0)
        return AgentResult(
            success=True,
            data={"result": value * 2}
        )


class ValidationAgent(BaseAgent):
    """Agent with custom validation logic."""

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that input contains required fields."""
        return "required_field" in input_data and input_data["required_field"] > 0

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute with validated input."""
        return AgentResult(
            success=True,
            data={"validated": True}
        )


class PreprocessAgent(BaseAgent):
    """Agent with preprocessing logic."""

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add preprocessing timestamp."""
        input_data["preprocessed_at"] = DeterministicClock.now().isoformat()
        input_data["value"] = input_data.get("value", 0) + 10
        return input_data

    def postprocess(self, result: AgentResult) -> AgentResult:
        """Add postprocessing flag."""
        result.data["postprocessed"] = True
        return result

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute and return processed value."""
        return AgentResult(
            success=True,
            data={"value": input_data.get("value", 0)}
        )


class FailingAgent(BaseAgent):
    """Agent that fails execution."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Always raise an exception."""
        raise ValueError("Intentional test failure")


class ResourceLoadingAgent(BaseAgent):
    """Agent that loads resources."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Load and return resource content."""
        resource_path = input_data.get("resource_path")
        if resource_path:
            content = self.load_resource(resource_path)
            return AgentResult(success=True, data={"content": content})
        return AgentResult(success=False, error="No resource path provided")


class CleanupAgent(BaseAgent):
    """Agent with cleanup logic."""

    def __init__(self, config=None):
        super().__init__(config)
        self.cleanup_called = False
        self.cleanup_data = []

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute and track data."""
        self.cleanup_data.append(input_data)
        return AgentResult(success=True, data={"executed": True})

    def cleanup(self):
        """Mark cleanup as called."""
        self.cleanup_called = True


class MetricsAgent(BaseAgent):
    """Agent for testing metrics collection."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute with custom metrics."""
        self.stats.increment("custom_counter", 5)
        self.stats.add_time("custom_timer", 123.45)

        return AgentResult(
            success=True,
            data={"processed": True}
        )


# Test Classes

@pytest.mark.unit
class TestAgentConfig:
    """Test AgentConfig model."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = AgentConfig(
            name="TestAgent",
            description="A test agent"
        )

        assert config.name == "TestAgent"
        assert config.description == "A test agent"
        assert config.version == "0.0.1"
        assert config.enabled is True
        assert config.enable_metrics is True
        assert config.enable_provenance is True
        assert config.parameters == {}
        assert config.resource_paths == []
        assert config.log_level == "INFO"

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = AgentConfig(
            name="CustomAgent",
            description="Custom test agent",
            version="1.2.3",
            enabled=False,
            enable_metrics=False,
            enable_provenance=False,
            parameters={"key": "value"},
            resource_paths=["/path/to/resource"],
            log_level="DEBUG"
        )

        assert config.name == "CustomAgent"
        assert config.version == "1.2.3"
        assert config.enabled is False
        assert config.enable_metrics is False
        assert config.parameters == {"key": "value"}
        assert config.resource_paths == ["/path/to/resource"]
        assert config.log_level == "DEBUG"

    def test_config_validation(self):
        """Test config validation via Pydantic."""
        # Should not raise for valid config
        config = AgentConfig(name="Test", description="Test")
        assert config.name == "Test"

    def test_config_serialization(self):
        """Test config can be serialized to dict."""
        config = AgentConfig(
            name="Test",
            description="Test agent",
            parameters={"param1": 100}
        )

        config_dict = config.dict()
        assert config_dict["name"] == "Test"
        assert config_dict["parameters"]["param1"] == 100


@pytest.mark.unit
class TestAgentMetrics:
    """Test AgentMetrics model."""

    def test_metrics_defaults(self):
        """Test metrics with default values."""
        metrics = AgentMetrics()

        assert metrics.execution_time_ms == 0.0
        assert metrics.input_size == 0
        assert metrics.output_size == 0
        assert metrics.records_processed == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.custom_metrics == {}

    def test_metrics_with_values(self):
        """Test metrics with custom values."""
        metrics = AgentMetrics(
            execution_time_ms=150.5,
            input_size=1024,
            output_size=2048,
            records_processed=100,
            cache_hits=25,
            cache_misses=75,
            custom_metrics={"metric1": 42.0}
        )

        assert metrics.execution_time_ms == 150.5
        assert metrics.input_size == 1024
        assert metrics.output_size == 2048
        assert metrics.records_processed == 100
        assert metrics.cache_hits == 25
        assert metrics.cache_misses == 75
        assert metrics.custom_metrics["metric1"] == 42.0


@pytest.mark.unit
class TestAgentResult:
    """Test AgentResult model."""

    def test_result_success(self):
        """Test successful result."""
        result = AgentResult(
            success=True,
            data={"value": 42}
        )

        assert result.success is True
        assert result.data == {"value": 42}
        assert result.error is None
        assert result.metadata == {}
        assert result.metrics is None
        assert result.provenance_id is None

    def test_result_failure(self):
        """Test failed result."""
        result = AgentResult(
            success=False,
            error="Something went wrong"
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data == {}

    def test_result_with_metrics(self):
        """Test result with metrics."""
        metrics = AgentMetrics(execution_time_ms=100.0)
        result = AgentResult(
            success=True,
            data={"result": "value"},
            metrics=metrics
        )

        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.execution_time_ms == 100.0

    def test_result_with_timestamp(self):
        """Test result with timestamp."""
        now = DeterministicClock.now()
        result = AgentResult(
            success=True,
            data={},
            timestamp=now
        )

        assert result.timestamp == now


@pytest.mark.unit
class TestStatsTracker:
    """Test StatsTracker functionality."""

    def test_tracker_initialization(self):
        """Test tracker starts with zero values."""
        tracker = StatsTracker()

        assert tracker.executions == 0
        assert tracker.successes == 0
        assert tracker.failures == 0
        assert tracker.total_time_ms == 0.0

    def test_record_successful_execution(self):
        """Test recording a successful execution."""
        tracker = StatsTracker()
        tracker.record_execution(success=True, duration_ms=100.5)

        assert tracker.executions == 1
        assert tracker.successes == 1
        assert tracker.failures == 0
        assert tracker.total_time_ms == 100.5

    def test_record_failed_execution(self):
        """Test recording a failed execution."""
        tracker = StatsTracker()
        tracker.record_execution(success=False, duration_ms=50.0)

        assert tracker.executions == 1
        assert tracker.successes == 0
        assert tracker.failures == 1
        assert tracker.total_time_ms == 50.0

    def test_record_multiple_executions(self):
        """Test recording multiple executions."""
        tracker = StatsTracker()
        tracker.record_execution(success=True, duration_ms=100.0)
        tracker.record_execution(success=True, duration_ms=200.0)
        tracker.record_execution(success=False, duration_ms=150.0)

        assert tracker.executions == 3
        assert tracker.successes == 2
        assert tracker.failures == 1
        assert tracker.total_time_ms == 450.0

    def test_increment_counter(self):
        """Test incrementing custom counters."""
        tracker = StatsTracker()
        tracker.increment("records_processed")
        tracker.increment("records_processed", 5)
        tracker.increment("errors")

        assert tracker.custom_counters["records_processed"] == 6
        assert tracker.custom_counters["errors"] == 1

    def test_add_time(self):
        """Test adding time to custom timers."""
        tracker = StatsTracker()
        tracker.add_time("db_query", 50.5)
        tracker.add_time("db_query", 25.0)
        tracker.add_time("api_call", 100.0)

        assert tracker.custom_timers["db_query"] == 75.5
        assert tracker.custom_timers["api_call"] == 100.0

    def test_get_stats_empty(self):
        """Test getting stats with no executions."""
        tracker = StatsTracker()
        stats = tracker.get_stats()

        assert stats["executions"] == 0
        assert stats["success_rate"] == 0
        assert stats["avg_time_ms"] == 0

    def test_get_stats_with_data(self):
        """Test getting stats with execution data."""
        tracker = StatsTracker()
        tracker.record_execution(True, 100.0)
        tracker.record_execution(True, 200.0)
        tracker.record_execution(False, 150.0)
        tracker.increment("custom", 5)
        tracker.add_time("timer", 75.0)

        stats = tracker.get_stats()

        assert stats["executions"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["success_rate"] == 66.67
        assert stats["total_time_ms"] == 450.0
        assert stats["avg_time_ms"] == 150.0
        assert stats["custom_counters"]["custom"] == 5
        assert stats["custom_timers"]["timer"] == 75.0


@pytest.mark.unit
class TestBaseAgentInitialization:
    """Test BaseAgent initialization."""

    def test_agent_initialization_defaults(self):
        """Test agent initializes with default config."""
        agent = SimpleAgent()

        assert agent.config is not None
        assert agent.config.name == "SimpleAgent"
        assert agent.logger is not None
        assert agent.stats is not None
        assert isinstance(agent.stats, StatsTracker)
        assert agent._resources == {}

    def test_agent_initialization_custom_config(self):
        """Test agent initializes with custom config."""
        config = AgentConfig(
            name="CustomAgent",
            description="Custom description",
            version="2.0.0",
            parameters={"param1": "value1"}
        )
        agent = SimpleAgent(config)

        assert agent.config.name == "CustomAgent"
        assert agent.config.description == "Custom description"
        assert agent.config.version == "2.0.0"
        assert agent.config.parameters["param1"] == "value1"

    def test_agent_logger_initialization(self):
        """Test agent logger is properly initialized."""
        agent = SimpleAgent()

        assert agent.logger is not None
        assert agent.logger.name == "SimpleAgent"

    def test_agent_initialize_hook_called(self):
        """Test initialize hook is called during construction."""
        class InitAgent(BaseAgent):
            def __init__(self, config=None):
                self.init_called = False
                super().__init__(config)

            def initialize(self):
                self.init_called = True

            def execute(self, input_data):
                return AgentResult(success=True, data={})

        agent = InitAgent()
        assert agent.init_called is True


@pytest.mark.unit
class TestBaseAgentValidation:
    """Test agent input validation."""

    def test_validation_default_passes(self):
        """Test default validation passes for any input."""
        agent = SimpleAgent()
        assert agent.validate_input({"any": "data"}) is True
        assert agent.validate_input({}) is True

    def test_validation_custom_logic(self):
        """Test custom validation logic."""
        agent = ValidationAgent()

        # Valid input
        assert agent.validate_input({"required_field": 10}) is True

        # Invalid inputs
        assert agent.validate_input({}) is False
        assert agent.validate_input({"required_field": 0}) is False
        assert agent.validate_input({"required_field": -5}) is False

    def test_run_with_validation_failure(self):
        """Test run fails when validation fails."""
        agent = ValidationAgent()
        result = agent.run({"invalid": "data"})

        assert result.success is False
        assert result.error == "Input validation failed"
        assert result.timestamp is not None


@pytest.mark.unit
class TestBaseAgentExecution:
    """Test agent execution."""

    def test_execute_success(self):
        """Test successful execution."""
        agent = SimpleAgent()
        result = agent.execute({"value": 5})

        assert result.success is True
        assert result.data == {"result": 10}

    def test_run_lifecycle(self):
        """Test complete run lifecycle."""
        agent = SimpleAgent()
        result = agent.run({"value": 7})

        assert result.success is True
        assert result.data == {"result": 14}
        assert result.timestamp is not None

    def test_run_with_metrics(self):
        """Test run collects metrics."""
        config = AgentConfig(
            name="TestAgent",
            description="Test",
            enable_metrics=True
        )
        agent = SimpleAgent(config)
        result = agent.run({"value": 5})

        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.execution_time_ms > 0
        assert result.metrics.input_size > 0
        assert result.metrics.output_size > 0

    def test_run_without_metrics(self):
        """Test run without metrics collection."""
        config = AgentConfig(
            name="TestAgent",
            description="Test",
            enable_metrics=False
        )
        agent = SimpleAgent(config)
        result = agent.run({"value": 5})

        assert result.success is True
        assert result.metrics is None

    def test_run_disabled_agent(self):
        """Test running disabled agent."""
        config = AgentConfig(
            name="TestAgent",
            description="Test",
            enabled=False
        )
        agent = SimpleAgent(config)
        result = agent.run({"value": 5})

        assert result.success is False
        assert "disabled" in result.error


@pytest.mark.unit
class TestBaseAgentPrePostProcessing:
    """Test agent pre/post processing."""

    def test_preprocess_modifies_input(self):
        """Test preprocessing modifies input data."""
        agent = PreprocessAgent()
        result = agent.run({"value": 5})

        assert result.success is True
        # Value was incremented by 10 in preprocessing
        assert result.data["value"] == 15

    def test_postprocess_modifies_result(self):
        """Test postprocessing modifies result."""
        agent = PreprocessAgent()
        result = agent.run({"value": 5})

        assert result.success is True
        assert result.data.get("postprocessed") is True

    def test_preprocess_adds_metadata(self):
        """Test preprocessing can add metadata."""
        agent = PreprocessAgent()

        # Verify preprocess is called
        processed = agent.preprocess({"value": 10})
        assert "preprocessed_at" in processed


@pytest.mark.unit
class TestBaseAgentHooks:
    """Test agent execution hooks."""

    def test_pre_execute_hook(self):
        """Test pre-execution hooks are called."""
        agent = SimpleAgent()
        hook_called = []

        def pre_hook(agent_instance, input_data):
            hook_called.append(("pre", input_data))

        agent.add_pre_hook(pre_hook)
        result = agent.run({"value": 5})

        assert result.success is True
        assert len(hook_called) == 1
        assert hook_called[0][0] == "pre"
        assert hook_called[0][1]["value"] == 5

    def test_post_execute_hook(self):
        """Test post-execution hooks are called."""
        agent = SimpleAgent()
        hook_called = []

        def post_hook(agent_instance, result):
            hook_called.append(("post", result.data))

        agent.add_post_hook(post_hook)
        result = agent.run({"value": 5})

        assert result.success is True
        assert len(hook_called) == 1
        assert hook_called[0][0] == "post"
        assert hook_called[0][1]["result"] == 10

    def test_multiple_hooks(self):
        """Test multiple hooks are called in order."""
        agent = SimpleAgent()
        execution_order = []

        def pre_hook1(agent_instance, input_data):
            execution_order.append("pre1")

        def pre_hook2(agent_instance, input_data):
            execution_order.append("pre2")

        def post_hook1(agent_instance, result):
            execution_order.append("post1")

        def post_hook2(agent_instance, result):
            execution_order.append("post2")

        agent.add_pre_hook(pre_hook1)
        agent.add_pre_hook(pre_hook2)
        agent.add_post_hook(post_hook1)
        agent.add_post_hook(post_hook2)

        result = agent.run({"value": 5})

        assert result.success is True
        assert execution_order == ["pre1", "pre2", "post1", "post2"]


@pytest.mark.unit
class TestBaseAgentErrorHandling:
    """Test agent error handling."""

    def test_execution_exception_handling(self):
        """Test exceptions during execution are caught."""
        agent = FailingAgent()
        result = agent.run({"any": "data"})

        assert result.success is False
        assert result.error is not None
        assert "Intentional test failure" in result.error
        assert result.timestamp is not None

    def test_error_recorded_in_stats(self):
        """Test errors are recorded in stats."""
        agent = FailingAgent()
        result = agent.run({"any": "data"})

        stats = agent.get_stats()
        assert stats["executions"] == 1
        assert stats["failures"] == 1
        assert stats["successes"] == 0

    def test_cleanup_called_on_error(self):
        """Test cleanup is called even when execution fails."""
        class FailingCleanupAgent(CleanupAgent):
            def execute(self, input_data):
                raise ValueError("Test error")

        agent = FailingCleanupAgent()
        result = agent.run({"any": "data"})

        assert result.success is False
        assert agent.cleanup_called is True


@pytest.mark.unit
class TestBaseAgentResourceLoading:
    """Test agent resource loading."""

    def test_load_resource_success(self):
        """Test loading a resource file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name

        try:
            agent = ResourceLoadingAgent()
            result = agent.run({"resource_path": temp_path})

            assert result.success is True
            assert result.data["content"] == "test content"
        finally:
            Path(temp_path).unlink()

    def test_load_resource_cached(self):
        """Test resource loading uses cache."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("cached content")
            temp_path = f.name

        try:
            agent = ResourceLoadingAgent()

            # First load
            content1 = agent.load_resource(temp_path)
            # Second load (should use cache)
            content2 = agent.load_resource(temp_path)

            assert content1 == content2
            assert content1 == "cached content"
            assert temp_path in agent._resources
        finally:
            Path(temp_path).unlink()

    def test_load_resource_not_found(self):
        """Test loading non-existent resource raises error."""
        agent = ResourceLoadingAgent()

        with pytest.raises(FileNotFoundError):
            agent.load_resource("/nonexistent/path/to/file.txt")


@pytest.mark.unit
class TestBaseAgentCleanup:
    """Test agent cleanup."""

    def test_cleanup_called_after_execution(self):
        """Test cleanup is called after successful execution."""
        agent = CleanupAgent()
        result = agent.run({"test": "data"})

        assert result.success is True
        assert agent.cleanup_called is True

    def test_cleanup_called_after_error(self):
        """Test cleanup is called even after errors."""
        class FailingCleanupAgent(CleanupAgent):
            def execute(self, input_data):
                raise RuntimeError("Test error")

        agent = FailingCleanupAgent()
        result = agent.run({"test": "data"})

        assert result.success is False
        assert agent.cleanup_called is True

    def test_cleanup_error_logged(self):
        """Test cleanup errors are logged but don't crash."""
        class BadCleanupAgent(BaseAgent):
            def execute(self, input_data):
                return AgentResult(success=True, data={})

            def cleanup(self):
                raise RuntimeError("Cleanup failed")

        agent = BadCleanupAgent()
        # Should not raise despite cleanup error
        result = agent.run({"test": "data"})
        assert result.success is True


@pytest.mark.unit
class TestBaseAgentMetrics:
    """Test agent metrics collection."""

    def test_stats_tracking(self):
        """Test stats are tracked across executions."""
        agent = SimpleAgent()

        # Run multiple times
        agent.run({"value": 1})
        agent.run({"value": 2})
        agent.run({"value": 3})

        stats = agent.get_stats()
        assert stats["executions"] == 3
        assert stats["successes"] == 3
        assert stats["failures"] == 0

    def test_custom_metrics(self):
        """Test custom metrics collection."""
        agent = MetricsAgent()
        result = agent.run({})

        stats = agent.get_stats()
        assert stats["custom_counters"]["custom_counter"] == 5
        assert stats["custom_timers"]["custom_timer"] == 123.45

    def test_reset_stats(self):
        """Test resetting stats."""
        agent = SimpleAgent()
        agent.run({"value": 1})
        agent.run({"value": 2})

        assert agent.stats.executions == 2

        agent.reset_stats()
        assert agent.stats.executions == 0
        assert agent.stats.successes == 0
        assert agent.stats.failures == 0


@pytest.mark.unit
class TestBaseAgentRepresentation:
    """Test agent string representation."""

    def test_repr(self):
        """Test agent __repr__ method."""
        config = AgentConfig(
            name="TestAgent",
            description="Test",
            version="1.0.0"
        )
        agent = SimpleAgent(config)
        agent.run({"value": 1})

        repr_str = repr(agent)
        assert "TestAgent" in repr_str
        assert "1.0.0" in repr_str
        assert "executions=1" in repr_str


@pytest.mark.unit
class TestBaseAgentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input_data(self):
        """Test handling empty input data."""
        agent = SimpleAgent()
        result = agent.run({})

        assert result.success is True
        assert result.data == {"result": 0}

    def test_none_input_handling(self):
        """Test handling None as input (validation should catch)."""
        class StrictAgent(BaseAgent):
            def validate_input(self, input_data):
                return input_data is not None and isinstance(input_data, dict)

            def execute(self, input_data):
                return AgentResult(success=True, data={})

        agent = StrictAgent()
        result = agent.run(None)

        assert result.success is False

    def test_large_input_data(self):
        """Test handling large input data."""
        agent = SimpleAgent()
        large_input = {"value": 1000000, "extra_data": "x" * 10000}
        result = agent.run(large_input)

        assert result.success is True
        assert result.metrics.input_size > 10000

    def test_concurrent_executions(self):
        """Test multiple concurrent executions."""
        import threading

        agent = SimpleAgent()
        results = []
        errors = []

        def run_agent(value):
            try:
                result = agent.run({"value": value})
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_agent, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(errors) == 0
        assert agent.stats.executions == 10

    def test_execution_time_tracking(self):
        """Test execution time is tracked correctly."""
        class SlowAgent(BaseAgent):
            def execute(self, input_data):
                time.sleep(0.1)  # Sleep for 100ms
                return AgentResult(success=True, data={})

        agent = SlowAgent()
        result = agent.run({})

        assert result.success is True
        assert result.metrics.execution_time_ms >= 100
