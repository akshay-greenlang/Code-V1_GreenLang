"""
Comprehensive tests for BaseAgent class.
Tests lifecycle management, metrics collection, hooks, and resource loading.
"""

import pytest
from greenlang.agents.base import (
    BaseAgent, AgentConfig, AgentResult, AgentMetrics, StatsTracker
)
from typing import Dict, Any


class TestAgent(BaseAgent):
    """Simple test agent for testing base functionality."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Simple execution that returns input doubled."""
        value = input_data.get("value", 0)
        return AgentResult(
            success=True,
            data={"result": value * 2}
        )


class FailingAgent(BaseAgent):
    """Agent that always fails."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        raise ValueError("Intentional test failure")


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig(
            name="TestAgent",
            description="Test agent"
        )
        assert config.name == "TestAgent"
        assert config.version == "0.0.1"
        assert config.enabled is True
        assert config.enable_metrics is True
        assert config.enable_provenance is True
        assert config.log_level == "INFO"

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            name="CustomAgent",
            description="Custom test agent",
            version="2.0.0",
            enabled=False,
            parameters={"key": "value"}
        )
        assert config.version == "2.0.0"
        assert config.enabled is False
        assert config.parameters["key"] == "value"


class TestAgentMetrics:
    """Test AgentMetrics model."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = AgentMetrics()
        assert metrics.execution_time_ms == 0.0
        assert metrics.input_size == 0
        assert metrics.output_size == 0
        assert metrics.records_processed == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_custom_metrics(self):
        """Test custom metric values."""
        metrics = AgentMetrics(
            execution_time_ms=123.45,
            input_size=1000,
            output_size=500,
            custom_metrics={"my_metric": 42.0}
        )
        assert metrics.execution_time_ms == 123.45
        assert metrics.custom_metrics["my_metric"] == 42.0


class TestStatsTracker:
    """Test StatsTracker functionality."""

    def test_record_execution(self):
        """Test recording executions."""
        tracker = StatsTracker()

        tracker.record_execution(success=True, duration_ms=100.0)
        tracker.record_execution(success=True, duration_ms=200.0)
        tracker.record_execution(success=False, duration_ms=50.0)

        assert tracker.executions == 3
        assert tracker.successes == 2
        assert tracker.failures == 1
        assert tracker.total_time_ms == 350.0

    def test_increment_counter(self):
        """Test incrementing custom counters."""
        tracker = StatsTracker()

        tracker.increment("records_processed", 10)
        tracker.increment("records_processed", 5)
        tracker.increment("errors", 1)

        assert tracker.custom_counters["records_processed"] == 15
        assert tracker.custom_counters["errors"] == 1

    def test_add_time(self):
        """Test adding time to custom timers."""
        tracker = StatsTracker()

        tracker.add_time("database_query", 50.0)
        tracker.add_time("database_query", 75.0)
        tracker.add_time("api_call", 100.0)

        assert tracker.custom_timers["database_query"] == 125.0
        assert tracker.custom_timers["api_call"] == 100.0

    def test_get_stats(self):
        """Test getting statistics."""
        tracker = StatsTracker()

        tracker.record_execution(success=True, duration_ms=100.0)
        tracker.record_execution(success=True, duration_ms=200.0)
        tracker.increment("records", 50)

        stats = tracker.get_stats()

        assert stats["executions"] == 2
        assert stats["successes"] == 2
        assert stats["failures"] == 0
        assert stats["success_rate"] == 100.0
        assert stats["avg_time_ms"] == 150.0
        assert stats["custom_counters"]["records"] == 50


class TestBaseAgent:
    """Test BaseAgent lifecycle and functionality."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = TestAgent()

        assert agent.config.name == "TestAgent"
        assert isinstance(agent.stats, StatsTracker)
        assert agent._resources == {}
        assert agent._pre_execute_hooks == []
        assert agent._post_execute_hooks == []

    def test_agent_with_custom_config(self):
        """Test agent with custom configuration."""
        config = AgentConfig(
            name="CustomAgent",
            description="Custom agent",
            version="2.0.0"
        )
        agent = TestAgent(config=config)

        assert agent.config.name == "CustomAgent"
        assert agent.config.version == "2.0.0"

    def test_successful_execution(self):
        """Test successful agent execution."""
        agent = TestAgent()
        input_data = {"value": 10}

        result = agent.run(input_data)

        assert result.success is True
        assert result.data["result"] == 20
        assert result.timestamp is not None
        assert result.metrics is not None
        assert result.metrics.execution_time_ms > 0

    def test_failed_execution(self):
        """Test failed agent execution."""
        agent = FailingAgent()
        input_data = {"value": 10}

        result = agent.run(input_data)

        assert result.success is False
        assert "Intentional test failure" in result.error
        assert result.timestamp is not None

    def test_disabled_agent(self):
        """Test that disabled agent doesn't execute."""
        config = AgentConfig(
            name="DisabledAgent",
            description="Disabled test agent",
            enabled=False
        )
        agent = TestAgent(config=config)

        result = agent.run({"value": 10})

        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_metrics_collection(self):
        """Test that metrics are collected during execution."""
        config = AgentConfig(
            name="MetricsAgent",
            description="Agent with metrics",
            enable_metrics=True
        )
        agent = TestAgent(config=config)

        result = agent.run({"value": 10, "extra": "data"})

        assert result.metrics is not None
        assert result.metrics.execution_time_ms > 0
        assert result.metrics.input_size > 0
        assert result.metrics.output_size > 0

    def test_metrics_disabled(self):
        """Test execution with metrics disabled."""
        config = AgentConfig(
            name="NoMetricsAgent",
            description="Agent without metrics",
            enable_metrics=False
        )
        agent = TestAgent(config=config)

        result = agent.run({"value": 10})

        assert result.success is True
        # Metrics should still be None when disabled

    def test_pre_execute_hooks(self):
        """Test pre-execution hooks."""
        agent = TestAgent()
        hook_called = []

        def pre_hook(agent_instance, input_data):
            hook_called.append("pre")

        agent.add_pre_hook(pre_hook)
        agent.run({"value": 10})

        assert "pre" in hook_called

    def test_post_execute_hooks(self):
        """Test post-execution hooks."""
        agent = TestAgent()
        hook_called = []

        def post_hook(agent_instance, result):
            hook_called.append("post")

        agent.add_post_hook(post_hook)
        agent.run({"value": 10})

        assert "post" in hook_called

    def test_stats_tracking(self):
        """Test that stats are tracked across executions."""
        agent = TestAgent()

        agent.run({"value": 10})
        agent.run({"value": 20})

        stats = agent.get_stats()

        assert stats["executions"] == 2
        assert stats["successes"] == 2
        assert stats["failures"] == 0

    def test_reset_stats(self):
        """Test resetting statistics."""
        agent = TestAgent()

        agent.run({"value": 10})
        agent.reset_stats()

        stats = agent.get_stats()
        assert stats["executions"] == 0

    def test_validate_input(self):
        """Test input validation."""
        agent = TestAgent()

        # Default validate_input returns True
        assert agent.validate_input({"value": 10}) is True

    def test_preprocess(self):
        """Test preprocessing."""
        agent = TestAgent()
        input_data = {"value": 10}

        # Default preprocess returns input unchanged
        processed = agent.preprocess(input_data)
        assert processed == input_data

    def test_postprocess(self):
        """Test postprocessing."""
        agent = TestAgent()
        result = AgentResult(success=True, data={"test": "value"})

        # Default postprocess returns result unchanged
        processed = agent.postprocess(result)
        assert processed == result

    def test_cleanup(self):
        """Test cleanup is called."""
        agent = TestAgent()
        cleanup_called = []

        original_cleanup = agent.cleanup
        def mock_cleanup():
            cleanup_called.append(True)
            original_cleanup()

        agent.cleanup = mock_cleanup
        agent.run({"value": 10})

        assert len(cleanup_called) > 0

    def test_repr(self):
        """Test string representation."""
        agent = TestAgent()
        agent.run({"value": 10})

        repr_str = repr(agent)
        assert "TestAgent" in repr_str
        assert "executions=" in repr_str


class TestResourceLoading:
    """Test resource loading functionality."""

    def test_load_resource(self, temp_dir):
        """Test loading a resource file."""
        # Create a test resource
        resource_path = temp_dir / "test_resource.txt"
        resource_path.write_text("Test resource content")

        agent = TestAgent()
        content = agent.load_resource(str(resource_path))

        assert content == "Test resource content"

    def test_resource_caching(self, temp_dir):
        """Test that resources are cached."""
        resource_path = temp_dir / "cached_resource.txt"
        resource_path.write_text("Cached content")

        agent = TestAgent()

        # Load twice
        content1 = agent.load_resource(str(resource_path))
        content2 = agent.load_resource(str(resource_path))

        assert content1 == content2
        assert str(resource_path) in agent._resources

    def test_missing_resource(self):
        """Test loading non-existent resource."""
        agent = TestAgent()

        with pytest.raises(FileNotFoundError):
            agent.load_resource("/nonexistent/resource.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
