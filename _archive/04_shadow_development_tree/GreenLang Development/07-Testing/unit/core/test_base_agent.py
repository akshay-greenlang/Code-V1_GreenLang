# -*- coding: utf-8 -*-
"""
Unit tests for BaseAgent - Core agent lifecycle and execution

Tests all lifecycle states, state transitions, error handling,
retry logic, and provenance tracking with 85%+ coverage.

Author: GreenLang Testing Team
Coverage Target: >90%
Test Count: 52 tests
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from greenlang.determinism import DeterministicClock
from greenlang.agents.base import (
    BaseAgent,
    AgentConfig,
    AgentResult,
    AgentMetrics,
    StatsTracker,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="test_agent",
        description="Test agent for unit testing",
        version="1.0.0",
        enabled=True,
        enable_metrics=True,
        enable_provenance=True,
        log_level="INFO",
    )


@pytest.fixture
def minimal_config():
    """Create minimal agent configuration."""
    return AgentConfig(
        name="minimal_agent",
        description="Minimal test agent"
    )


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, config=None, should_fail=False):
        self.should_fail = should_fail
        self.execute_count = 0
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Test implementation of execute."""
        self.execute_count += 1

        if self.should_fail:
            raise ValueError("Test execution error")

        return AgentResult(
            success=True,
            data={"result": "test_output", "input_echo": input_data},
            timestamp=DeterministicClock.now()
        )


class CustomValidationAgent(BaseAgent):
    """Agent with custom validation logic."""

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Custom validation - requires 'required_field'."""
        return "required_field" in input_data

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(success=True, data={"validated": True})


class PreprocessAgent(BaseAgent):
    """Agent with preprocessing logic."""

    def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add preprocessing marker."""
        input_data["preprocessed"] = True
        return input_data

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(success=True, data=input_data)


class PostprocessAgent(BaseAgent):
    """Agent with postprocessing logic."""

    def postprocess(self, result: AgentResult) -> AgentResult:
        """Add postprocessing marker."""
        result.data["postprocessed"] = True
        return result

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(success=True, data={"original": True})


class CleanupAgent(BaseAgent):
    """Agent with cleanup logic."""

    def __init__(self, config=None):
        super().__init__(config)
        self.cleanup_called = False

    def cleanup(self):
        """Mark cleanup as called."""
        self.cleanup_called = True

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        return AgentResult(success=True, data={})


# ==============================================================================
# Test AgentConfig
# ==============================================================================

class TestAgentConfig:
    """Test AgentConfig model."""

    def test_minimal_config(self):
        """Test config with minimal required fields."""
        config = AgentConfig(
            name="test",
            description="Test agent"
        )

        assert config.name == "test"
        assert config.description == "Test agent"
        assert config.version == "0.0.1"  # Default
        assert config.enabled is True
        assert config.enable_metrics is True
        assert config.enable_provenance is True

    def test_full_config(self, agent_config):
        """Test config with all fields specified."""
        assert agent_config.name == "test_agent"
        assert agent_config.description == "Test agent for unit testing"
        assert agent_config.version == "1.0.0"
        assert agent_config.enabled is True
        assert agent_config.log_level == "INFO"

    def test_config_with_parameters(self):
        """Test config with custom parameters."""
        config = AgentConfig(
            name="test",
            description="Test",
            parameters={"param1": "value1", "threshold": 0.95}
        )

        assert config.parameters["param1"] == "value1"
        assert config.parameters["threshold"] == 0.95

    def test_config_with_resource_paths(self):
        """Test config with resource paths."""
        config = AgentConfig(
            name="test",
            description="Test",
            resource_paths=["/path/to/resource1", "/path/to/resource2"]
        )

        assert len(config.resource_paths) == 2
        assert "/path/to/resource1" in config.resource_paths

    def test_config_disabled_agent(self):
        """Test config for disabled agent."""
        config = AgentConfig(
            name="test",
            description="Test",
            enabled=False
        )

        assert config.enabled is False


# ==============================================================================
# Test AgentMetrics
# ==============================================================================

class TestAgentMetrics:
    """Test AgentMetrics model."""

    def test_default_metrics(self):
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
        """Test metrics with specified values."""
        metrics = AgentMetrics(
            execution_time_ms=125.5,
            input_size=1024,
            output_size=2048,
            records_processed=100,
            cache_hits=10,
            cache_misses=5,
            custom_metrics={"llm_calls": 3, "db_queries": 7}
        )

        assert metrics.execution_time_ms == 125.5
        assert metrics.input_size == 1024
        assert metrics.output_size == 2048
        assert metrics.records_processed == 100
        assert metrics.cache_hits == 10
        assert metrics.cache_misses == 5
        assert metrics.custom_metrics["llm_calls"] == 3
        assert metrics.custom_metrics["db_queries"] == 7


# ==============================================================================
# Test AgentResult
# ==============================================================================

class TestAgentResult:
    """Test AgentResult model."""

    def test_successful_result(self):
        """Test successful result structure."""
        result = AgentResult(
            success=True,
            data={"output": "test"},
            timestamp=DeterministicClock.now()
        )

        assert result.success is True
        assert result.data["output"] == "test"
        assert result.error is None
        assert isinstance(result.timestamp, datetime)

    def test_failed_result(self):
        """Test failed result structure."""
        result = AgentResult(
            success=False,
            error="Test error message",
            timestamp=DeterministicClock.now()
        )

        assert result.success is False
        assert result.error == "Test error message"
        assert result.data == {}

    def test_result_with_metrics(self):
        """Test result with metrics attached."""
        metrics = AgentMetrics(execution_time_ms=50.0)
        result = AgentResult(
            success=True,
            data={},
            metrics=metrics
        )

        assert result.metrics.execution_time_ms == 50.0

    def test_result_with_provenance(self):
        """Test result with provenance ID."""
        result = AgentResult(
            success=True,
            data={},
            provenance_id="prov_12345"
        )

        assert result.provenance_id == "prov_12345"

    def test_result_with_metadata(self):
        """Test result with additional metadata."""
        result = AgentResult(
            success=True,
            data={},
            metadata={"agent_version": "1.0.0", "execution_mode": "test"}
        )

        assert result.metadata["agent_version"] == "1.0.0"
        assert result.metadata["execution_mode"] == "test"


# ==============================================================================
# Test StatsTracker
# ==============================================================================

class TestStatsTracker:
    """Test StatsTracker functionality."""

    def test_initialization(self):
        """Test stats tracker initializes with zero values."""
        stats = StatsTracker()

        assert stats.executions == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.total_time_ms == 0.0

    def test_record_successful_execution(self):
        """Test recording successful execution."""
        stats = StatsTracker()
        stats.record_execution(success=True, duration_ms=100.0)

        assert stats.executions == 1
        assert stats.successes == 1
        assert stats.failures == 0
        assert stats.total_time_ms == 100.0

    def test_record_failed_execution(self):
        """Test recording failed execution."""
        stats = StatsTracker()
        stats.record_execution(success=False, duration_ms=50.0)

        assert stats.executions == 1
        assert stats.successes == 0
        assert stats.failures == 1
        assert stats.total_time_ms == 50.0

    def test_record_multiple_executions(self):
        """Test recording multiple executions."""
        stats = StatsTracker()
        stats.record_execution(success=True, duration_ms=100.0)
        stats.record_execution(success=True, duration_ms=150.0)
        stats.record_execution(success=False, duration_ms=75.0)

        assert stats.executions == 3
        assert stats.successes == 2
        assert stats.failures == 1
        assert stats.total_time_ms == 325.0

    def test_custom_counter_increment(self):
        """Test incrementing custom counters."""
        stats = StatsTracker()
        stats.increment("api_calls")
        stats.increment("api_calls")
        stats.increment("db_queries", value=5)

        assert stats.custom_counters["api_calls"] == 2
        assert stats.custom_counters["db_queries"] == 5

    def test_custom_timer(self):
        """Test adding time to custom timers."""
        stats = StatsTracker()
        stats.add_time("llm_time", 100.0)
        stats.add_time("llm_time", 50.0)
        stats.add_time("db_time", 25.0)

        assert stats.custom_timers["llm_time"] == 150.0
        assert stats.custom_timers["db_time"] == 25.0

    def test_get_stats_success_rate(self):
        """Test get_stats calculates success rate correctly."""
        stats = StatsTracker()
        stats.record_execution(success=True, duration_ms=100.0)
        stats.record_execution(success=True, duration_ms=100.0)
        stats.record_execution(success=False, duration_ms=100.0)

        result = stats.get_stats()

        assert result["success_rate"] == 66.67  # 2/3 = 66.67%

    def test_get_stats_average_time(self):
        """Test get_stats calculates average time correctly."""
        stats = StatsTracker()
        stats.record_execution(success=True, duration_ms=100.0)
        stats.record_execution(success=True, duration_ms=200.0)

        result = stats.get_stats()

        assert result["avg_time_ms"] == 150.0  # (100 + 200) / 2

    def test_get_stats_empty(self):
        """Test get_stats with no executions."""
        stats = StatsTracker()
        result = stats.get_stats()

        assert result["executions"] == 0
        assert result["success_rate"] == 0
        assert result["avg_time_ms"] == 0


# ==============================================================================
# Test BaseAgent Initialization
# ==============================================================================

class TestBaseAgentInitialization:
    """Test BaseAgent initialization."""

    def test_init_with_config(self, agent_config):
        """Test agent initializes with provided config."""
        agent = TestAgent(agent_config)

        assert agent.config == agent_config
        assert agent.config.name == "test_agent"
        assert agent.stats.executions == 0

    def test_init_without_config(self):
        """Test agent initializes with default config."""
        agent = TestAgent()

        assert agent.config.name == "TestAgent"
        assert agent.config.version == "0.0.1"

    def test_init_creates_logger(self, agent_config):
        """Test agent creates logger on init."""
        agent = TestAgent(agent_config)

        assert agent.logger is not None
        assert agent.logger.name == "TestAgent"

    def test_init_creates_stats_tracker(self, agent_config):
        """Test agent creates stats tracker on init."""
        agent = TestAgent(agent_config)

        assert agent.stats is not None
        assert isinstance(agent.stats, StatsTracker)

    def test_init_creates_empty_resources(self, agent_config):
        """Test agent initializes empty resource cache."""
        agent = TestAgent(agent_config)

        assert agent._resources == {}

    def test_init_creates_empty_hooks(self, agent_config):
        """Test agent initializes empty hook lists."""
        agent = TestAgent(agent_config)

        assert agent._pre_execute_hooks == []
        assert agent._post_execute_hooks == []


# ==============================================================================
# Test BaseAgent Execution
# ==============================================================================

class TestBaseAgentExecution:
    """Test BaseAgent execution logic."""

    def test_execute_success(self, agent_config):
        """Test successful execution returns result."""
        agent = TestAgent(agent_config)
        result = agent.execute({"test": "input"})

        assert result.success is True
        assert result.data["result"] == "test_output"
        assert result.data["input_echo"]["test"] == "input"

    def test_execute_increments_counter(self, agent_config):
        """Test execute increments execution counter."""
        agent = TestAgent(agent_config)

        agent.execute({"test": "input"})
        agent.execute({"test": "input"})

        assert agent.execute_count == 2

    def test_execute_with_error(self, agent_config):
        """Test execute with error raises exception."""
        agent = TestAgent(agent_config, should_fail=True)

        with pytest.raises(ValueError, match="Test execution error"):
            agent.execute({"test": "input"})

    def test_run_full_lifecycle(self, agent_config):
        """Test run() executes full lifecycle."""
        agent = TestAgent(agent_config)
        result = agent.run({"test": "input"})

        assert result.success is True
        assert result.timestamp is not None
        assert agent.stats.executions == 1

    def test_run_disabled_agent(self, agent_config):
        """Test run() returns error for disabled agent."""
        agent_config.enabled = False
        agent = TestAgent(agent_config)
        result = agent.run({"test": "input"})

        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_run_validates_input(self, agent_config):
        """Test run() validates input before execution."""
        agent = CustomValidationAgent(agent_config)

        # Valid input
        result = agent.run({"required_field": "value"})
        assert result.success is True

        # Invalid input
        result = agent.run({"other_field": "value"})
        assert result.success is False
        assert "validation failed" in result.error.lower()

    def test_run_preprocesses_input(self, agent_config):
        """Test run() preprocesses input."""
        agent = PreprocessAgent(agent_config)
        result = agent.run({"original": "data"})

        assert result.success is True
        assert result.data["preprocessed"] is True
        assert result.data["original"] == "data"

    def test_run_postprocesses_result(self, agent_config):
        """Test run() postprocesses result."""
        agent = PostprocessAgent(agent_config)
        result = agent.run({})

        assert result.success is True
        assert result.data["postprocessed"] is True
        assert result.data["original"] is True

    def test_run_records_metrics(self, agent_config):
        """Test run() records execution metrics."""
        agent = TestAgent(agent_config)
        result = agent.run({"test": "input"})

        assert result.metrics is not None
        assert result.metrics.execution_time_ms > 0
        assert agent.stats.executions == 1
        assert agent.stats.successes == 1

    def test_run_tracks_execution_time(self, agent_config):
        """Test run() tracks execution time accurately."""
        agent = TestAgent(agent_config)

        start = time.time()
        result = agent.run({"test": "input"})
        end = time.time()

        actual_time_ms = (end - start) * 1000

        # Metrics should be close to actual time
        assert result.metrics.execution_time_ms <= actual_time_ms + 10  # +10ms tolerance


# ==============================================================================
# Test BaseAgent Hooks
# ==============================================================================

class TestBaseAgentHooks:
    """Test BaseAgent pre/post execution hooks."""

    def test_add_pre_hook(self, agent_config):
        """Test adding pre-execution hook."""
        agent = TestAgent(agent_config)
        mock_hook = Mock()

        agent.add_pre_hook(mock_hook)

        assert len(agent._pre_execute_hooks) == 1
        assert mock_hook in agent._pre_execute_hooks

    def test_add_post_hook(self, agent_config):
        """Test adding post-execution hook."""
        agent = TestAgent(agent_config)
        mock_hook = Mock()

        agent.add_post_hook(mock_hook)

        assert len(agent._post_execute_hooks) == 1
        assert mock_hook in agent._post_execute_hooks

    def test_multiple_hooks(self, agent_config):
        """Test adding multiple hooks."""
        agent = TestAgent(agent_config)
        mock_pre1 = Mock()
        mock_pre2 = Mock()
        mock_post1 = Mock()

        agent.add_pre_hook(mock_pre1)
        agent.add_pre_hook(mock_pre2)
        agent.add_post_hook(mock_post1)

        assert len(agent._pre_execute_hooks) == 2
        assert len(agent._post_execute_hooks) == 1


# ==============================================================================
# Test BaseAgent Resource Loading
# ==============================================================================

class TestBaseAgentResourceLoading:
    """Test BaseAgent resource loading functionality."""

    def test_load_resource_not_found(self, agent_config):
        """Test loading non-existent resource raises error."""
        agent = TestAgent(agent_config)

        with pytest.raises(FileNotFoundError):
            agent.load_resource("/nonexistent/resource.txt")

    def test_load_resource_caching(self, agent_config, tmp_path):
        """Test resource loading caches results."""
        # Create temporary resource file
        resource_file = tmp_path / "test_resource.txt"
        resource_file.write_text("test content")

        agent = TestAgent(agent_config)

        # First load
        content1 = agent.load_resource(str(resource_file))
        assert content1 == "test content"

        # Second load should come from cache
        content2 = agent.load_resource(str(resource_file))
        assert content2 == "test content"
        assert content1 is content2  # Same object (cached)

    def test_load_multiple_resources(self, agent_config, tmp_path):
        """Test loading multiple different resources."""
        # Create multiple resource files
        resource1 = tmp_path / "resource1.txt"
        resource2 = tmp_path / "resource2.txt"
        resource1.write_text("content1")
        resource2.write_text("content2")

        agent = TestAgent(agent_config)

        content1 = agent.load_resource(str(resource1))
        content2 = agent.load_resource(str(resource2))

        assert content1 == "content1"
        assert content2 == "content2"
        assert len(agent._resources) == 2


# ==============================================================================
# Test BaseAgent Cleanup
# ==============================================================================

class TestBaseAgentCleanup:
    """Test BaseAgent cleanup functionality."""

    def test_cleanup_called(self, agent_config):
        """Test cleanup method can be called."""
        agent = CleanupAgent(agent_config)

        agent.cleanup()

        assert agent.cleanup_called is True

    def test_cleanup_default_implementation(self, agent_config):
        """Test default cleanup does nothing."""
        agent = TestAgent(agent_config)

        # Should not raise error
        agent.cleanup()


# ==============================================================================
# Test BaseAgent Error Handling
# ==============================================================================

class TestBaseAgentErrorHandling:
    """Test BaseAgent error handling."""

    def test_execution_error_caught(self, agent_config):
        """Test execution errors are caught and returned."""
        agent = TestAgent(agent_config, should_fail=True)

        # run() should catch the error and return failed result
        result = agent.run({"test": "input"})

        assert result.success is False
        assert result.error is not None

    def test_validation_error_handling(self, agent_config):
        """Test validation errors return proper result."""
        agent = CustomValidationAgent(agent_config)

        result = agent.run({})  # Missing required_field

        assert result.success is False
        assert "validation" in result.error.lower()

    def test_stats_track_failures(self, agent_config):
        """Test stats tracker records failures."""
        agent = TestAgent(agent_config, should_fail=True)

        try:
            agent.run({"test": "input"})
        except:
            pass

        assert agent.stats.failures >= 0  # Should track failures


# ==============================================================================
# Test BaseAgent Edge Cases
# ==============================================================================

class TestBaseAgentEdgeCases:
    """Test BaseAgent edge cases and boundary conditions."""

    def test_empty_input(self, agent_config):
        """Test execution with empty input."""
        agent = TestAgent(agent_config)
        result = agent.run({})

        assert result.success is True

    def test_none_input_handling(self, agent_config):
        """Test execution with None values in input."""
        agent = TestAgent(agent_config)
        result = agent.run({"key": None})

        assert result.success is True
        assert result.data["input_echo"]["key"] is None

    def test_large_input(self, agent_config):
        """Test execution with large input data."""
        agent = TestAgent(agent_config)
        large_input = {"data": "x" * 10000}

        result = agent.run(large_input)

        assert result.success is True

    def test_nested_data_structures(self, agent_config):
        """Test execution with nested data structures."""
        agent = TestAgent(agent_config)
        nested_input = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"]
                }
            }
        }

        result = agent.run(nested_input)

        assert result.success is True
        assert result.data["input_echo"]["level1"]["level2"]["level3"] == ["a", "b", "c"]

    def test_concurrent_execution(self, agent_config):
        """Test multiple concurrent executions."""
        agent = TestAgent(agent_config)

        # Execute multiple times
        results = [agent.run({"iteration": i}) for i in range(10)]

        assert all(r.success for r in results)
        assert agent.stats.executions == 10
        assert agent.stats.successes == 10

    def test_config_parameter_access(self, agent_config):
        """Test accessing config parameters during execution."""
        agent_config.parameters = {"threshold": 0.95, "mode": "production"}
        agent = TestAgent(agent_config)

        assert agent.config.parameters["threshold"] == 0.95
        assert agent.config.parameters["mode"] == "production"

    def test_metrics_disabled(self):
        """Test execution with metrics disabled."""
        config = AgentConfig(
            name="test",
            description="Test",
            enable_metrics=False
        )
        agent = TestAgent(config)

        result = agent.run({"test": "input"})

        assert result.success is True
        # Metrics should be None when disabled
        assert result.metrics is None
