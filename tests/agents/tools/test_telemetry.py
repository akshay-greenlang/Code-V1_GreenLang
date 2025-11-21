# -*- coding: utf-8 -*-
"""
Tests for GreenLang Tool Telemetry System
==========================================

Comprehensive tests for telemetry collection, metrics aggregation,
and export functionality.

Author: GreenLang Framework Team
Date: October 2025
"""

import json
import pytest
import threading
import time
from datetime import datetime

from greenlang.agents.tools.telemetry import (
from greenlang.determinism import DeterministicClock
    TelemetryCollector,
    ToolMetrics,
    get_telemetry,
    reset_global_telemetry,
)


class TestToolMetrics:
    """Test ToolMetrics dataclass."""

    def test_tool_metrics_creation(self):
        """Test creating ToolMetrics."""
        metrics = ToolMetrics(
            tool_name="test_tool",
            total_calls=100,
            successful_calls=95,
            failed_calls=5,
            avg_execution_time_ms=45.5
        )

        assert metrics.tool_name == "test_tool"
        assert metrics.total_calls == 100
        assert metrics.successful_calls == 95
        assert metrics.failed_calls == 5
        assert metrics.avg_execution_time_ms == 45.5

    def test_tool_metrics_to_dict(self):
        """Test converting ToolMetrics to dict."""
        metrics = ToolMetrics(
            tool_name="test_tool",
            total_calls=10,
            last_called=datetime(2025, 10, 1, 12, 0, 0)
        )

        result = metrics.to_dict()
        assert result["tool_name"] == "test_tool"
        assert result["total_calls"] == 10
        assert result["last_called"] == "2025-10-01T12:00:00"


class TestTelemetryCollector:
    """Test TelemetryCollector class."""

    def setup_method(self):
        """Set up test collector."""
        self.collector = TelemetryCollector(enable_real_time=True)

    def test_initialization(self):
        """Test telemetry collector initialization."""
        assert self.collector.enable_real_time is True
        assert len(self.collector._tool_data) == 0

    def test_record_successful_execution(self):
        """Test recording successful execution."""
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=50.0,
            success=True
        )

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.avg_execution_time_ms == 50.0

    def test_record_failed_execution(self):
        """Test recording failed execution."""
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=25.0,
            success=False,
            error_type="ValueError"
        )

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 1
        assert metrics.error_counts_by_type["ValueError"] == 1

    def test_multiple_executions(self):
        """Test recording multiple executions."""
        # Record 10 executions
        for i in range(10):
            self.collector.record_execution(
                tool_name="test_tool",
                execution_time_ms=10.0 + i,
                success=i < 8  # 8 successes, 2 failures
            )

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.total_calls == 10
        assert metrics.successful_calls == 8
        assert metrics.failed_calls == 2
        assert metrics.avg_execution_time_ms == 14.5  # Average of 10-19

    def test_percentile_calculations(self):
        """Test percentile calculations."""
        # Record executions with known distribution
        times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for t in times:
            self.collector.record_execution(
                tool_name="test_tool",
                execution_time_ms=float(t),
                success=True
            )

        metrics = self.collector.get_tool_metrics("test_tool")

        # Check percentiles (approximate)
        assert 50 <= metrics.p50_execution_time_ms <= 60  # Median
        assert 90 <= metrics.p95_execution_time_ms <= 100  # 95th percentile
        assert 95 <= metrics.p99_execution_time_ms <= 100  # 99th percentile

    def test_error_type_tracking(self):
        """Test tracking different error types."""
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=10.0,
            success=False,
            error_type="ValueError"
        )
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=15.0,
            success=False,
            error_type="ValueError"
        )
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=20.0,
            success=False,
            error_type="TypeError"
        )

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.error_counts_by_type["ValueError"] == 2
        assert metrics.error_counts_by_type["TypeError"] == 1

    def test_rate_limit_tracking(self):
        """Test tracking rate limit hits."""
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=10.0,
            success=False,
            rate_limited=True
        )
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=10.0,
            success=False,
            rate_limited=True
        )

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.rate_limit_hits == 2

    def test_validation_failure_tracking(self):
        """Test tracking validation failures."""
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=5.0,
            success=False,
            validation_failed=True
        )
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=5.0,
            success=False,
            validation_failed=True
        )
        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=5.0,
            success=False,
            validation_failed=True
        )

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.validation_failures == 3

    def test_last_called_timestamp(self):
        """Test last_called timestamp is updated."""
        before = DeterministicClock.now()

        self.collector.record_execution(
            tool_name="test_tool",
            execution_time_ms=10.0,
            success=True
        )

        after = DeterministicClock.now()

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.last_called is not None
        assert before <= metrics.last_called <= after

    def test_get_all_metrics(self):
        """Test getting metrics for all tools."""
        # Record for multiple tools
        self.collector.record_execution("tool_a", 10.0, True)
        self.collector.record_execution("tool_b", 20.0, True)
        self.collector.record_execution("tool_c", 30.0, False)

        all_metrics = self.collector.get_all_metrics()

        assert len(all_metrics) == 3
        assert "tool_a" in all_metrics
        assert "tool_b" in all_metrics
        assert "tool_c" in all_metrics
        assert all_metrics["tool_a"].total_calls == 1
        assert all_metrics["tool_b"].total_calls == 1
        assert all_metrics["tool_c"].total_calls == 1

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        # Record for multiple tools
        for i in range(10):
            self.collector.record_execution("tool_a", 10.0, True)
        for i in range(5):
            self.collector.record_execution("tool_b", 20.0, False)
        for i in range(3):
            self.collector.record_execution("tool_c", 50.0, True, rate_limited=True)

        summary = self.collector.get_summary_stats()

        assert summary["total_tools"] == 3
        assert summary["total_executions"] == 18
        assert summary["total_successes"] == 13
        assert summary["total_failures"] == 5
        assert summary["total_rate_limit_hits"] == 3
        assert 70 < summary["overall_success_rate"] < 75  # ~72%
        assert summary["most_used_tool"]["name"] == "tool_a"
        assert summary["most_used_tool"]["calls"] == 10

    def test_reset_specific_tool(self):
        """Test resetting metrics for specific tool."""
        self.collector.record_execution("tool_a", 10.0, True)
        self.collector.record_execution("tool_b", 20.0, True)

        # Reset tool_a
        self.collector.reset_metrics("tool_a")

        all_metrics = self.collector.get_all_metrics()
        assert "tool_a" not in all_metrics
        assert "tool_b" in all_metrics

    def test_reset_all_metrics(self):
        """Test resetting all metrics."""
        self.collector.record_execution("tool_a", 10.0, True)
        self.collector.record_execution("tool_b", 20.0, True)

        # Reset all
        self.collector.reset_metrics()

        all_metrics = self.collector.get_all_metrics()
        assert len(all_metrics) == 0

    def test_unknown_tool_metrics(self):
        """Test getting metrics for unknown tool returns empty metrics."""
        metrics = self.collector.get_tool_metrics("unknown_tool")

        assert metrics.tool_name == "unknown_tool"
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0

    def test_thread_safety(self):
        """Test thread-safe metric recording."""
        def record_metrics():
            for i in range(100):
                self.collector.record_execution(
                    tool_name="concurrent_tool",
                    execution_time_ms=float(i),
                    success=True
                )

        # Create 10 threads
        threads = [threading.Thread(target=record_metrics) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have 1000 total calls (10 threads * 100 calls)
        metrics = self.collector.get_tool_metrics("concurrent_tool")
        assert metrics.total_calls == 1000
        assert metrics.successful_calls == 1000

    def test_execution_time_trimming(self):
        """Test that execution times are trimmed when too large."""
        # Record more than 10000 executions
        for i in range(11000):
            self.collector.record_execution(
                tool_name="test_tool",
                execution_time_ms=float(i),
                success=True
            )

        # Check that execution_times list is trimmed
        data = self.collector._tool_data["test_tool"]
        assert len(data["execution_times"]) == 10000


class TestTelemetryExport:
    """Test telemetry export functionality."""

    def setup_method(self):
        """Set up test collector with sample data."""
        self.collector = TelemetryCollector()

        # Add sample data
        for i in range(10):
            self.collector.record_execution("tool_a", 10.0 + i, True)
        for i in range(5):
            self.collector.record_execution("tool_b", 20.0, False, error_type="ValueError")

    def test_export_json(self):
        """Test JSON export."""
        result = self.collector.export_metrics(format="json")

        assert isinstance(result, dict)
        assert "summary" in result
        assert "tools" in result
        assert "exported_at" in result

        assert result["summary"]["total_tools"] == 2
        assert "tool_a" in result["tools"]
        assert "tool_b" in result["tools"]

    def test_export_prometheus(self):
        """Test Prometheus export."""
        result = self.collector.export_metrics(format="prometheus")

        assert isinstance(result, str)
        assert "# HELP tool_calls_total" in result
        assert "# TYPE tool_calls_total counter" in result
        assert 'tool="tool_a"' in result
        assert 'tool="tool_b"' in result
        assert "tool_execution_time_milliseconds" in result
        assert "tool_rate_limit_hits" in result
        assert "tool_validation_failures" in result

    def test_export_csv(self):
        """Test CSV export."""
        result = self.collector.export_metrics(format="csv")

        assert isinstance(result, str)
        lines = result.split("\n")

        # Check header
        assert "tool_name" in lines[0]
        assert "total_calls" in lines[0]
        assert "successful_calls" in lines[0]

        # Check data rows
        assert len(lines) >= 3  # Header + 2 tools

    def test_export_invalid_format(self):
        """Test export with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.collector.export_metrics(format="invalid")

    def test_json_export_structure(self):
        """Test JSON export has correct structure."""
        result = self.collector.export_metrics(format="json")

        # Check summary structure
        summary = result["summary"]
        assert "total_tools" in summary
        assert "total_executions" in summary
        assert "overall_success_rate" in summary
        assert "most_used_tool" in summary

        # Check tool metrics structure
        tool_a = result["tools"]["tool_a"]
        assert "total_calls" in tool_a
        assert "successful_calls" in tool_a
        assert "avg_execution_time_ms" in tool_a
        assert "p50_execution_time_ms" in tool_a

    def test_prometheus_metric_names(self):
        """Test Prometheus metric names are sanitized."""
        # Record for tool with special chars
        self.collector.record_execution("tool-with.special-chars", 10.0, True)

        result = self.collector.export_metrics(format="prometheus")

        # Should be sanitized to tool_with_special_chars
        assert 'tool="tool_with_special_chars"' in result


class TestGlobalTelemetry:
    """Test global telemetry singleton."""

    def teardown_method(self):
        """Reset global telemetry after each test."""
        reset_global_telemetry()

    def test_get_telemetry_singleton(self):
        """Test get_telemetry returns singleton."""
        tel1 = get_telemetry()
        tel2 = get_telemetry()

        assert tel1 is tel2

    def test_reset_global_telemetry(self):
        """Test resetting global telemetry."""
        tel1 = get_telemetry()
        tel1.record_execution("test_tool", 10.0, True)

        reset_global_telemetry()

        tel2 = get_telemetry()
        assert tel1 is not tel2

        # New instance should be empty
        metrics = tel2.get_all_metrics()
        assert len(metrics) == 0


class TestTelemetryPercentiles:
    """Test percentile calculation edge cases."""

    def setup_method(self):
        """Set up test collector."""
        self.collector = TelemetryCollector()

    def test_percentiles_with_single_value(self):
        """Test percentiles with single value."""
        self.collector.record_execution("test_tool", 50.0, True)

        metrics = self.collector.get_tool_metrics("test_tool")
        assert metrics.p50_execution_time_ms == 50.0
        assert metrics.p95_execution_time_ms == 50.0
        assert metrics.p99_execution_time_ms == 50.0

    def test_percentiles_with_empty_data(self):
        """Test percentiles with no data."""
        metrics = self.collector.get_tool_metrics("unknown_tool")
        assert metrics.p50_execution_time_ms == 0.0
        assert metrics.p95_execution_time_ms == 0.0
        assert metrics.p99_execution_time_ms == 0.0

    def test_percentiles_with_two_values(self):
        """Test percentiles with two values."""
        self.collector.record_execution("test_tool", 10.0, True)
        self.collector.record_execution("test_tool", 90.0, True)

        metrics = self.collector.get_tool_metrics("test_tool")
        # Both values should be at high percentiles
        assert metrics.p50_execution_time_ms in [10.0, 90.0]
        assert metrics.p95_execution_time_ms == 90.0
        assert metrics.p99_execution_time_ms == 90.0


class TestTelemetryIntegration:
    """Integration tests for telemetry with tools."""

    def setup_method(self):
        """Set up test environment."""
        reset_global_telemetry()

    def test_telemetry_with_real_tool(self):
        """Test telemetry integration with actual tool."""
        from greenlang.agents.tools.financial import FinancialMetricsTool

        tool = FinancialMetricsTool()
        telemetry = get_telemetry()

        # Execute tool
        result = tool(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=10
        )

        assert result.success

        # Check telemetry was recorded
        metrics = telemetry.get_tool_metrics("calculate_financial_metrics")
        assert metrics.total_calls >= 1
        assert metrics.successful_calls >= 1
        assert metrics.avg_execution_time_ms > 0
