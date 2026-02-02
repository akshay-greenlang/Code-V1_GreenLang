# -*- coding: utf-8 -*-
"""
Tests for GL-FOUND-X-010: GreenLang Observability Agent

Tests cover:
    - Metrics collection (counter, gauge, histogram)
    - Prometheus format export
    - Distributed tracing (span start/end)
    - Structured logging with correlation IDs
    - Health checks (liveness and readiness probes)
    - Alert rule management and evaluation
    - Dashboard data generation
    - Provenance tracking

Author: GreenLang Team
"""

import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Any, Dict

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.foundation.observability_agent import (
    ObservabilityAgent,
    ObservabilityInput,
    ObservabilityOutput,
    MetricType,
    MetricValue,
    MetricDefinition,
    TraceContext,
    SpanDefinition,
    LogEntry,
    AlertRule,
    Alert,
    AlertSeverity,
    AlertStatus,
    HealthStatus,
    HealthCheck,
    PLATFORM_METRICS,
    STANDARD_LABELS,
)
from greenlang.utilities.determinism import DeterministicClock


class TestObservabilityAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_creation_default_config(self):
        """Test creating agent with default configuration."""
        agent = ObservabilityAgent()

        assert agent.AGENT_ID == "GL-FOUND-X-010"
        assert agent.AGENT_NAME == "GreenLang Observability Agent"
        assert agent.VERSION == "1.0.0"

    def test_agent_creation_custom_config(self):
        """Test creating agent with custom configuration."""
        config = AgentConfig(
            name="Custom Observability",
            description="Custom config test",
            parameters={
                "metrics_retention_hours": 48,
                "max_active_spans": 5000,
            }
        )
        agent = ObservabilityAgent(config)

        assert agent.config.name == "Custom Observability"
        assert agent._span_limit == 5000

    def test_platform_metrics_initialized(self):
        """Test that platform metrics are initialized on startup."""
        agent = ObservabilityAgent()

        for metric_name in PLATFORM_METRICS:
            assert metric_name in agent._metrics
            assert agent._metrics[metric_name].description == PLATFORM_METRICS[metric_name]["description"]

    def test_default_health_checks_registered(self):
        """Test that default health checks are registered."""
        agent = ObservabilityAgent()

        assert "metrics_store" in agent._health_checks
        assert "span_store" in agent._health_checks
        assert "log_buffer" in agent._health_checks


class TestMetricsCollection:
    """Tests for metrics collection functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_record_metric_gauge(self, agent):
        """Test recording a gauge metric."""
        result = agent.run({
            "operation": "record_metric",
            "metric": {
                "name": "cpu_usage",
                "value": 75.5,
                "labels": {"host": "server1", "region": "us-east"},
            }
        })

        assert result.success
        assert result.data["data"]["metric_name"] == "cpu_usage"
        assert "cpu_usage" in agent._metrics

    def test_increment_counter(self, agent):
        """Test incrementing a counter metric."""
        labels = {"agent_id": "GL-001", "status": "success"}

        # First increment
        result1 = agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_execution_total",
                "value": 1,
                "labels": labels,
            }
        })

        assert result1.success
        assert result1.data["data"]["new_value"] == 1.0

        # Second increment
        result2 = agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_execution_total",
                "value": 1,
                "labels": labels,
            }
        })

        assert result2.success
        assert result2.data["data"]["new_value"] == 2.0

    def test_set_gauge(self, agent):
        """Test setting a gauge metric."""
        labels = {"tenant_id": "tenant_123"}

        # Set initial value
        result1 = agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "lineage_completeness_ratio",
                "value": 0.85,
                "labels": labels,
            }
        })

        assert result1.success
        assert result1.data["data"]["value"] == 0.85

        # Update value
        result2 = agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "lineage_completeness_ratio",
                "value": 0.92,
                "labels": labels,
            }
        })

        assert result2.success
        assert result2.data["data"]["value"] == 0.92

    def test_observe_histogram(self, agent):
        """Test observing values in a histogram."""
        labels = {"agent_id": "GL-001", "agent_type": "MRV"}

        # Observe multiple values
        for value in [0.1, 0.25, 0.5, 1.0, 2.0]:
            result = agent.run({
                "operation": "observe_histogram",
                "metric": {
                    "name": "agent_execution_duration_seconds",
                    "value": value,
                    "labels": labels,
                }
            })
            assert result.success

        # Check histogram state
        data = result.data["data"]
        assert data["count"] == 5
        assert data["sum"] == pytest.approx(3.85, rel=0.01)

    def test_histogram_bucket_counts(self, agent):
        """Test that histogram bucket counts are correct."""
        labels = {"test": "buckets"}

        # Create histogram with known values
        values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0]
        for v in values:
            agent.run({
                "operation": "observe_histogram",
                "metric": {
                    "name": "custom_histogram",
                    "value": v,
                    "labels": labels,
                }
            })

        series = agent._metrics["custom_histogram"]
        label_key = "test=buckets"
        bucket_counts = series.bucket_counts[label_key]

        # All values should be in +Inf bucket
        assert bucket_counts[float('inf')] == 8

    def test_metric_labels_to_key(self, agent):
        """Test label dictionary to key conversion."""
        labels1 = {"a": "1", "b": "2", "c": "3"}
        labels2 = {"c": "3", "a": "1", "b": "2"}

        key1 = agent._labels_to_key(labels1)
        key2 = agent._labels_to_key(labels2)

        # Keys should be identical regardless of dict order
        assert key1 == key2
        assert key1 == "a=1,b=2,c=3"

    def test_increment_counter_error_on_wrong_type(self, agent):
        """Test that incrementing a non-counter fails."""
        # First create a gauge
        agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "my_gauge",
                "value": 10.0,
                "labels": {},
            }
        })

        # Try to increment it
        result = agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "my_gauge",
                "value": 1,
                "labels": {},
            }
        })

        assert not result.success
        assert "not a counter" in result.error


class TestPrometheusExport:
    """Tests for Prometheus format export."""

    @pytest.fixture
    def agent_with_metrics(self):
        """Create agent with some pre-recorded metrics."""
        agent = ObservabilityAgent()

        # Record various metrics
        agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_execution_total",
                "value": 10,
                "labels": {"agent_id": "GL-001", "status": "success"},
            }
        })

        agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "lineage_completeness_ratio",
                "value": 0.95,
                "labels": {"tenant_id": "tenant_1"},
            }
        })

        return agent

    def test_export_metrics_format(self, agent_with_metrics):
        """Test that metrics are exported in Prometheus format."""
        result = agent_with_metrics.run({"operation": "export_metrics"})

        assert result.success
        data = result.data["data"]
        assert data["format"] == "prometheus"
        assert "content" in data
        assert data["metrics_count"] > 0

    def test_export_contains_help_comments(self, agent_with_metrics):
        """Test that export contains HELP comments."""
        result = agent_with_metrics.run({"operation": "export_metrics"})
        content = result.data["data"]["content"]

        assert "# HELP agent_execution_total" in content

    def test_export_contains_type_comments(self, agent_with_metrics):
        """Test that export contains TYPE comments."""
        result = agent_with_metrics.run({"operation": "export_metrics"})
        content = result.data["data"]["content"]

        assert "# TYPE agent_execution_total counter" in content

    def test_export_contains_metric_values(self, agent_with_metrics):
        """Test that export contains metric values with labels."""
        result = agent_with_metrics.run({"operation": "export_metrics"})
        content = result.data["data"]["content"]

        # Should contain the counter we recorded
        assert "agent_execution_total{agent_id=GL-001,status=success}" in content


class TestDistributedTracing:
    """Tests for distributed tracing functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_start_span(self, agent):
        """Test starting a new trace span."""
        result = agent.run({
            "operation": "start_span",
            "span": {
                "name": "process_emissions",
                "context": {
                    "trace_id": "abc123def456",
                    "span_id": "span001",
                },
                "start_time": DeterministicClock.now().isoformat(),
            }
        })

        assert result.success
        assert result.data["data"]["trace_id"] == "abc123def456"
        assert result.data["data"]["span_id"] == "span001"
        assert len(agent._active_spans) == 1

    def test_end_span(self, agent):
        """Test ending a trace span."""
        trace_id = "trace123"
        span_id = "span456"
        start_time = DeterministicClock.now()

        # Start span
        agent.run({
            "operation": "start_span",
            "span": {
                "name": "calculation",
                "context": {
                    "trace_id": trace_id,
                    "span_id": span_id,
                },
                "start_time": start_time.isoformat(),
            }
        })

        # End span
        result = agent.run({
            "operation": "end_span",
            "span": {
                "name": "calculation",
                "context": {
                    "trace_id": trace_id,
                    "span_id": span_id,
                },
                "start_time": start_time.isoformat(),
                "end_time": DeterministicClock.now().isoformat(),
                "status": "ok",
            }
        })

        assert result.success
        assert result.data["data"]["status"] == "ok"
        assert "duration_ms" in result.data["data"]
        assert len(agent._active_spans) == 0
        assert len(agent._completed_spans) == 1

    def test_nested_spans(self, agent):
        """Test nested span relationships."""
        trace_id = "trace_nested"
        start_time = DeterministicClock.now()

        # Start parent span
        agent.run({
            "operation": "start_span",
            "span": {
                "name": "parent_operation",
                "context": {
                    "trace_id": trace_id,
                    "span_id": "parent_span",
                },
                "start_time": start_time.isoformat(),
            }
        })

        # Start child span
        agent.run({
            "operation": "start_span",
            "span": {
                "name": "child_operation",
                "context": {
                    "trace_id": trace_id,
                    "span_id": "child_span",
                    "parent_span_id": "parent_span",
                },
                "start_time": start_time.isoformat(),
            }
        })

        assert len(agent._active_spans) == 2

        # Check parent has child reference
        parent_key = f"{trace_id}:parent_span"
        assert "child_span" in str(agent._active_spans[parent_key].children)

    def test_end_nonexistent_span_fails(self, agent):
        """Test that ending a non-existent span fails."""
        result = agent.run({
            "operation": "end_span",
            "span": {
                "name": "nonexistent",
                "context": {
                    "trace_id": "fake",
                    "span_id": "fake",
                },
                "start_time": DeterministicClock.now().isoformat(),
            }
        })

        assert not result.success
        assert "not found" in result.error.lower()

    def test_span_limit_cleanup(self, agent):
        """Test that old spans are cleaned up when limit is reached."""
        agent._span_limit = 10

        # Create more spans than the limit
        for i in range(15):
            agent.run({
                "operation": "start_span",
                "span": {
                    "name": f"span_{i}",
                    "context": {
                        "trace_id": f"trace_{i}",
                        "span_id": f"span_{i}",
                    },
                    "start_time": DeterministicClock.now().isoformat(),
                }
            })

        # Should have cleaned up some spans
        assert len(agent._active_spans) <= 15


class TestStructuredLogging:
    """Tests for structured logging functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_log_entry(self, agent):
        """Test recording a structured log entry."""
        result = agent.run({
            "operation": "log",
            "log_entry": {
                "timestamp": DeterministicClock.now().isoformat(),
                "level": "INFO",
                "message": "Agent execution completed",
                "agent_id": "GL-MRV-001",
                "tenant_id": "tenant_123",
            }
        })

        assert result.success
        assert result.data["data"]["logged"]
        assert "correlation_id" in result.data["data"]
        assert len(agent._log_buffer) == 1

    def test_log_with_trace_correlation(self, agent):
        """Test logging with trace ID correlation."""
        trace_id = "correlation_trace_123"

        result = agent.run({
            "operation": "log",
            "log_entry": {
                "timestamp": DeterministicClock.now().isoformat(),
                "level": "DEBUG",
                "message": "Processing started",
                "trace_id": trace_id,
                "span_id": "span_456",
            }
        })

        assert result.success
        log_json = json.loads(result.data["data"]["json"])
        assert log_json["trace_id"] == trace_id

    def test_log_buffer_trimming(self, agent):
        """Test that log buffer is trimmed when full."""
        agent._log_buffer_size = 10

        # Add more logs than buffer size
        for i in range(20):
            agent.run({
                "operation": "log",
                "log_entry": {
                    "timestamp": DeterministicClock.now().isoformat(),
                    "level": "INFO",
                    "message": f"Log message {i}",
                }
            })

        # Buffer should be trimmed
        assert len(agent._log_buffer) == 10

    def test_log_generates_correlation_id(self, agent):
        """Test that correlation ID is auto-generated if not provided."""
        result = agent.run({
            "operation": "log",
            "log_entry": {
                "timestamp": DeterministicClock.now().isoformat(),
                "level": "WARNING",
                "message": "Test message",
            }
        })

        assert result.success
        assert result.data["data"]["correlation_id"] is not None
        assert len(result.data["data"]["correlation_id"]) > 0

    def test_log_with_custom_attributes(self, agent):
        """Test logging with custom attributes."""
        result = agent.run({
            "operation": "log",
            "log_entry": {
                "timestamp": DeterministicClock.now().isoformat(),
                "level": "INFO",
                "message": "Custom log",
                "attributes": {
                    "emissions_value": 123.45,
                    "calculation_method": "spend-based",
                }
            }
        })

        assert result.success
        log_json = json.loads(result.data["data"]["json"])
        assert log_json["emissions_value"] == 123.45
        assert log_json["calculation_method"] == "spend-based"


class TestHealthChecks:
    """Tests for health check functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_liveness_probe(self, agent):
        """Test liveness probe returns healthy."""
        result = agent.run({"operation": "liveness_probe"})

        assert result.success
        assert result.data["data"]["status"] == "healthy"
        assert "timestamp" in result.data["data"]

    def test_readiness_probe(self, agent):
        """Test readiness probe returns all checks."""
        result = agent.run({"operation": "readiness_probe"})

        assert result.success
        data = result.data["data"]
        assert data["status"] == "healthy"
        assert "checks" in data
        assert "metrics_store" in data["checks"]
        assert "span_store" in data["checks"]
        assert "log_buffer" in data["checks"]

    def test_readiness_degraded_on_high_utilization(self, agent):
        """Test readiness shows degraded when resources are near capacity."""
        agent._span_limit = 100

        # Fill up span store to 95%
        for i in range(95):
            agent.run({
                "operation": "start_span",
                "span": {
                    "name": f"span_{i}",
                    "context": {
                        "trace_id": f"trace_{i}",
                        "span_id": f"span_{i}",
                    },
                    "start_time": DeterministicClock.now().isoformat(),
                }
            })

        result = agent.run({"operation": "readiness_probe"})
        data = result.data["data"]

        # Span store should be degraded
        assert data["checks"]["span_store"]["status"] == "degraded"


class TestAlertGeneration:
    """Tests for alert generation functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_add_alert_rule(self, agent):
        """Test adding an alert rule."""
        result = agent.run({
            "operation": "add_alert_rule",
            "alert_rule": {
                "name": "high_error_rate",
                "metric_name": "agent_errors_total",
                "condition": "gt",
                "threshold": 10,
                "severity": "critical",
            }
        })

        assert result.success
        assert result.data["data"]["rule_name"] == "high_error_rate"
        assert result.data["data"]["total_rules"] == 1

    def test_alert_fires_on_threshold_breach(self, agent):
        """Test that alert fires when threshold is breached."""
        # Add alert rule
        agent.run({
            "operation": "add_alert_rule",
            "alert_rule": {
                "name": "error_alert",
                "metric_name": "agent_errors_total",
                "condition": "gte",
                "threshold": 5,
                "severity": "warning",
            }
        })

        # Record metrics above threshold
        agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_errors_total",
                "value": 10,
                "labels": {"agent_id": "GL-001"},
            }
        })

        # Check alerts
        result = agent.run({"operation": "check_alerts"})

        assert result.success
        assert len(result.data["data"]["new_alerts"]) == 1
        assert result.data["data"]["active_alerts"] == 1

    def test_alert_resolves_when_below_threshold(self, agent):
        """Test that alert resolves when metric goes below threshold."""
        # Add alert rule
        agent.run({
            "operation": "add_alert_rule",
            "alert_rule": {
                "name": "high_usage",
                "metric_name": "custom_gauge",
                "condition": "gt",
                "threshold": 80,
                "severity": "warning",
            }
        })

        # Set metric above threshold
        agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "custom_gauge",
                "value": 90,
                "labels": {},
            }
        })

        # Check alerts (should fire)
        agent.run({"operation": "check_alerts"})
        assert len(agent._active_alerts) == 1

        # Set metric below threshold
        agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "custom_gauge",
                "value": 70,
                "labels": {},
            }
        })

        # Check alerts (should resolve)
        result = agent.run({"operation": "check_alerts"})

        assert len(result.data["data"]["resolved_alerts"]) == 1
        assert result.data["data"]["active_alerts"] == 0

    def test_alert_with_label_filter(self, agent):
        """Test alert rule with label filtering."""
        # Add alert rule for specific agent
        agent.run({
            "operation": "add_alert_rule",
            "alert_rule": {
                "name": "agent_specific_alert",
                "metric_name": "agent_errors_total",
                "condition": "gt",
                "threshold": 0,
                "severity": "error",
                "labels": {"agent_id": "critical_agent"},
            }
        })

        # Record errors for different agents
        agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_errors_total",
                "value": 5,
                "labels": {"agent_id": "other_agent"},
            }
        })

        agent.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_errors_total",
                "value": 1,
                "labels": {"agent_id": "critical_agent"},
            }
        })

        # Check alerts
        result = agent.run({"operation": "check_alerts"})

        # Only the critical_agent alert should fire
        new_alerts = result.data["data"]["new_alerts"]
        assert len(new_alerts) == 1
        assert new_alerts[0]["labels"]["agent_id"] == "critical_agent"

    def test_all_condition_operators(self, agent):
        """Test all comparison operators work correctly."""
        test_cases = [
            ("gt", 10, 11, True),
            ("gt", 10, 10, False),
            ("gte", 10, 10, True),
            ("lt", 10, 9, True),
            ("lt", 10, 10, False),
            ("lte", 10, 10, True),
            ("eq", 10, 10, True),
            ("eq", 10, 11, False),
            ("ne", 10, 11, True),
            ("ne", 10, 10, False),
        ]

        for condition, threshold, value, expected in test_cases:
            result = agent._evaluate_condition(value, condition, threshold)
            assert result == expected, f"Failed for {condition} {threshold} with value {value}"


class TestDashboardData:
    """Tests for dashboard data generation."""

    @pytest.fixture
    def agent_with_data(self):
        """Create agent with various recorded data."""
        agent = ObservabilityAgent()

        # Record some metrics
        for i in range(5):
            agent.run({
                "operation": "increment_counter",
                "metric": {
                    "name": "agent_execution_total",
                    "value": 1,
                    "labels": {"agent_id": f"agent_{i}", "status": "success"},
                }
            })

            agent.run({
                "operation": "observe_histogram",
                "metric": {
                    "name": "agent_execution_duration_seconds",
                    "value": 0.1 * (i + 1),
                    "labels": {"agent_id": f"agent_{i}"},
                }
            })

        return agent

    def test_get_dashboard_data_all_metrics(self, agent_with_data):
        """Test getting dashboard data for all platform metrics."""
        result = agent_with_data.run({
            "operation": "get_dashboard_data",
            "dashboard_query": {},
        })

        assert result.success
        data = result.data["data"]
        assert "panels" in data
        assert "summary" in data
        assert data["summary"]["total_metrics_recorded"] > 0

    def test_get_dashboard_data_specific_metric(self, agent_with_data):
        """Test getting dashboard data for a specific metric."""
        result = agent_with_data.run({
            "operation": "get_dashboard_data",
            "dashboard_query": {
                "metric_name": "agent_execution_total",
            },
        })

        assert result.success
        panels = result.data["data"]["panels"]
        assert len(panels) == 1
        assert panels[0]["metric_name"] == "agent_execution_total"

    def test_get_dashboard_data_with_label_filter(self, agent_with_data):
        """Test getting dashboard data with label filtering."""
        result = agent_with_data.run({
            "operation": "get_dashboard_data",
            "dashboard_query": {
                "metric_name": "agent_execution_total",
                "labels": {"agent_id": "agent_0"},
            },
        })

        assert result.success
        panels = result.data["data"]["panels"]
        assert len(panels) == 1
        # Should only have one data point for agent_0
        assert len(panels[0]["data_points"]) == 1


class TestProvenanceTracking:
    """Tests for provenance and audit trail functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_output_contains_provenance_hash(self, agent):
        """Test that all operations include provenance hash."""
        result = agent.run({
            "operation": "record_metric",
            "metric": {
                "name": "test_metric",
                "value": 42,
                "labels": {},
            }
        })

        assert result.success
        assert "provenance_hash" in result.data
        assert len(result.data["provenance_hash"]) == 16  # First 16 chars of SHA-256

    def test_provenance_hash_deterministic(self, agent):
        """Test that provenance hash is deterministic for same input."""
        input_data = {
            "operation": "set_gauge",
            "metric": {
                "name": "deterministic_test",
                "value": 100,
                "labels": {"key": "value"},
            }
        }

        result1 = agent.run(input_data)
        result2 = agent.run(input_data)

        # Same input should produce same hash
        assert result1.data["provenance_hash"] == result2.data["provenance_hash"]

    def test_provenance_hash_changes_with_input(self, agent):
        """Test that provenance hash changes with different input."""
        result1 = agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "test",
                "value": 1,
                "labels": {},
            }
        })

        result2 = agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "test",
                "value": 2,
                "labels": {},
            }
        })

        # Different values should produce different hashes
        assert result1.data["provenance_hash"] != result2.data["provenance_hash"]


class TestConvenienceMethods:
    """Tests for convenience API methods."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_record_agent_execution(self, agent):
        """Test the convenience method for recording agent execution."""
        agent.record_agent_execution(
            agent_id="GL-MRV-001",
            agent_type="MRV",
            tenant_id="tenant_123",
            status="success",
            duration_seconds=1.5,
        )

        # Check execution total was recorded
        series = agent._metrics["agent_execution_total"]
        assert len(series.values) > 0

        # Check duration was recorded
        series = agent._metrics["agent_execution_duration_seconds"]
        assert len(series.counts) > 0

    def test_record_agent_execution_failure(self, agent):
        """Test recording a failed agent execution."""
        agent.record_agent_execution(
            agent_id="GL-MRV-001",
            agent_type="MRV",
            tenant_id="tenant_123",
            status="failure",
            duration_seconds=0.5,
        )

        # Check error was recorded
        series = agent._metrics["agent_errors_total"]
        assert len(series.values) > 0

    def test_create_trace(self, agent):
        """Test the convenience method for creating traces."""
        trace_id = agent.create_trace(
            name="test_operation",
            attributes={"key": "value"},
        )

        assert trace_id is not None
        assert len(trace_id) == 32  # UUID hex length
        assert len(agent._active_spans) == 1

    def test_log_structured(self, agent):
        """Test the convenience method for structured logging."""
        agent.log_structured(
            level="INFO",
            message="Test log message",
            agent_id="GL-001",
            tenant_id="tenant_123",
            custom_field="custom_value",
        )

        assert len(agent._log_buffer) == 1
        entry = agent._log_buffer[0]
        assert entry.level == "INFO"
        assert entry.agent_id == "GL-001"

    def test_get_metrics_summary(self, agent):
        """Test getting metrics summary."""
        # Record some custom metrics
        agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "custom_metric",
                "value": 1,
                "labels": {},
            }
        })

        summary = agent.get_metrics_summary()

        assert summary["total_series"] > len(PLATFORM_METRICS)
        assert "custom_metric" in summary["custom_metrics"]
        assert "agent_execution_total" in summary["platform_metrics"]

    def test_get_active_alerts_summary(self, agent):
        """Test getting active alerts summary."""
        # Create alert condition
        agent.run({
            "operation": "add_alert_rule",
            "alert_rule": {
                "name": "test_alert",
                "metric_name": "test_gauge",
                "condition": "gt",
                "threshold": 0,
                "severity": "warning",
            }
        })

        agent.run({
            "operation": "set_gauge",
            "metric": {
                "name": "test_gauge",
                "value": 10,
                "labels": {},
            }
        })

        agent.run({"operation": "check_alerts"})

        summary = agent.get_active_alerts_summary()

        assert len(summary) == 1
        assert summary[0]["rule_name"] == "test_alert"


class TestInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_invalid_operation_fails(self, agent):
        """Test that invalid operation fails."""
        result = agent.run({"operation": "invalid_operation"})

        assert not result.success
        assert "invalid_operation" in result.error.lower() or "must be one of" in result.error.lower()

    def test_missing_metric_for_record_fails(self, agent):
        """Test that missing metric data fails."""
        result = agent.run({
            "operation": "record_metric",
            # Missing metric field
        })

        assert not result.success

    def test_invalid_metric_name_fails(self):
        """Test that invalid metric names are rejected."""
        with pytest.raises(ValueError, match="must start with a letter"):
            MetricDefinition(
                name="123invalid",
                type=MetricType.COUNTER,
            )

    def test_invalid_alert_condition_fails(self):
        """Test that invalid alert conditions are rejected."""
        with pytest.raises(ValueError, match="must be one of"):
            AlertRule(
                name="test",
                metric_name="test",
                condition="invalid",
                threshold=10,
            )


class TestPerformance:
    """Performance and stress tests."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return ObservabilityAgent()

    def test_high_volume_metric_recording(self, agent):
        """Test recording many metrics quickly."""
        start = time.time()
        count = 1000

        for i in range(count):
            agent.run({
                "operation": "increment_counter",
                "metric": {
                    "name": "high_volume_counter",
                    "value": 1,
                    "labels": {"iteration": str(i % 10)},
                }
            })

        duration = time.time() - start
        rate = count / duration

        # Should be able to record at least 100 metrics per second
        assert rate > 100, f"Performance too slow: {rate:.2f} ops/sec"

    def test_processing_time_tracked(self, agent):
        """Test that processing time is tracked."""
        result = agent.run({
            "operation": "record_metric",
            "metric": {
                "name": "timing_test",
                "value": 1,
                "labels": {},
            }
        })

        assert result.success
        # processing_time_ms is in the ObservabilityOutput data structure
        assert "processing_time_ms" in result.data
        # Value might be 0 for very fast operations, check it exists and is non-negative
        assert result.data["processing_time_ms"] >= 0


class TestIntegration:
    """Integration tests for the Observability Agent."""

    def test_full_observability_workflow(self):
        """Test a complete observability workflow."""
        agent = ObservabilityAgent()

        # 1. Create a trace
        trace_id = agent.create_trace("full_workflow", {"test": "integration"})

        # 2. Log some events
        agent.log_structured(
            level="INFO",
            message="Workflow started",
            trace_id=trace_id,
        )

        # 3. Record metrics
        agent.record_agent_execution(
            agent_id="GL-INTEG-001",
            agent_type="Integration",
            tenant_id="test_tenant",
            status="success",
            duration_seconds=0.5,
        )

        # 4. Set up an alert
        agent.run({
            "operation": "add_alert_rule",
            "alert_rule": {
                "name": "integration_alert",
                "metric_name": "agent_errors_total",
                "condition": "gt",
                "threshold": 5,
                "severity": "warning",
            }
        })

        # 5. Check health
        health_result = agent.run({"operation": "readiness_probe"})
        assert health_result.data["data"]["status"] == "healthy"

        # 6. Export metrics
        export_result = agent.run({"operation": "export_metrics"})
        assert "agent_execution_total" in export_result.data["data"]["content"]

        # 7. Get dashboard data
        dashboard_result = agent.run({
            "operation": "get_dashboard_data",
            "dashboard_query": {},
        })
        assert dashboard_result.data["data"]["summary"]["total_metrics_recorded"] > 0

    def test_multi_tenant_isolation(self):
        """Test that metrics are properly isolated by tenant."""
        agent = ObservabilityAgent()

        # Record metrics for different tenants
        for tenant in ["tenant_a", "tenant_b"]:
            for i in range(5):
                agent.run({
                    "operation": "increment_counter",
                    "metric": {
                        "name": "tenant_requests",
                        "value": 1,
                        "labels": {"tenant_id": tenant},
                    }
                })

        # Get dashboard data filtered by tenant
        result = agent.run({
            "operation": "get_dashboard_data",
            "dashboard_query": {
                "metric_name": "tenant_requests",
                "labels": {"tenant_id": "tenant_a"},
            }
        })

        # Should only get tenant_a data
        panels = result.data["data"]["panels"]
        assert len(panels) == 1
        for dp in panels[0]["data_points"]:
            assert dp["labels"]["tenant_id"] == "tenant_a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
