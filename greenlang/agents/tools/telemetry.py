"""
GreenLang Tool Telemetry System
================================

Thread-safe telemetry collector for tracking tool execution metrics including:
- Call counts (total, successful, failed)
- Execution time percentiles (p50, p95, p99)
- Rate limit hits
- Validation failures
- Error tracking by type
- Export to JSON, Prometheus, CSV formats

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# Tool Metrics Data Classes
# ==============================================================================

@dataclass
class ToolMetrics:
    """Metrics for a single tool."""

    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    p50_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0
    rate_limit_hits: int = 0
    validation_failures: int = 0
    last_called: Optional[datetime] = None
    error_counts_by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        if self.last_called:
            data["last_called"] = self.last_called.isoformat()
        return data


# ==============================================================================
# Telemetry Collector
# ==============================================================================

class TelemetryCollector:
    """
    Thread-safe telemetry collector for tool execution metrics.

    Features:
    - Thread-safe metric collection
    - Real-time percentile calculations
    - Error type tracking
    - Rate limit and validation tracking
    - Multiple export formats (JSON, Prometheus, CSV)

    Example:
        >>> collector = TelemetryCollector()
        >>> collector.record_execution(
        ...     tool_name="calculate_npv",
        ...     execution_time_ms=45.2,
        ...     success=True
        ... )
        >>> metrics = collector.get_tool_metrics("calculate_npv")
        >>> print(metrics.avg_execution_time_ms)
        45.2
    """

    def __init__(self, enable_real_time: bool = True):
        """
        Initialize telemetry collector.

        Args:
            enable_real_time: Enable real-time metric updates (default: True)
        """
        self.enable_real_time = enable_real_time
        self._lock = threading.RLock()

        # Storage for metrics
        self._tool_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "execution_times": [],
            "total_execution_time_ms": 0.0,
            "rate_limit_hits": 0,
            "validation_failures": 0,
            "error_counts": defaultdict(int),
            "last_called": None,
        })

        logger.info("TelemetryCollector initialized (real_time=%s)", enable_real_time)

    def record_execution(
        self,
        tool_name: str,
        execution_time_ms: float,
        success: bool,
        error_type: Optional[str] = None,
        user_id: Optional[str] = None,
        rate_limited: bool = False,
        validation_failed: bool = False,
    ) -> None:
        """
        Record a tool execution.

        Args:
            tool_name: Name of the tool that was executed
            execution_time_ms: Execution time in milliseconds
            success: Whether execution was successful
            error_type: Type of error if failed (e.g., "ValueError")
            user_id: User ID for per-user tracking (optional)
            rate_limited: Whether execution hit rate limit
            validation_failed: Whether execution failed validation
        """
        with self._lock:
            data = self._tool_data[tool_name]

            # Update call counts
            data["total_calls"] += 1
            if success:
                data["successful_calls"] += 1
            else:
                data["failed_calls"] += 1

            # Update execution times
            data["execution_times"].append(execution_time_ms)
            data["total_execution_time_ms"] += execution_time_ms

            # Update rate limit tracking
            if rate_limited:
                data["rate_limit_hits"] += 1

            # Update validation tracking
            if validation_failed:
                data["validation_failures"] += 1

            # Update error tracking
            if error_type:
                data["error_counts"][error_type] += 1

            # Update last called timestamp
            data["last_called"] = datetime.now()

            # Trim execution times if too large (keep last 10000)
            if len(data["execution_times"]) > 10000:
                data["execution_times"] = data["execution_times"][-10000:]

    def get_tool_metrics(self, tool_name: str) -> ToolMetrics:
        """
        Get metrics for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolMetrics object with aggregated metrics
        """
        with self._lock:
            if tool_name not in self._tool_data:
                # Return empty metrics for unknown tools
                return ToolMetrics(tool_name=tool_name)

            data = self._tool_data[tool_name]
            execution_times = data["execution_times"]

            # Calculate percentiles
            p50, p95, p99 = self._calculate_percentiles(execution_times)

            # Calculate average
            avg_time = (
                data["total_execution_time_ms"] / data["total_calls"]
                if data["total_calls"] > 0
                else 0.0
            )

            return ToolMetrics(
                tool_name=tool_name,
                total_calls=data["total_calls"],
                successful_calls=data["successful_calls"],
                failed_calls=data["failed_calls"],
                total_execution_time_ms=data["total_execution_time_ms"],
                avg_execution_time_ms=avg_time,
                p50_execution_time_ms=p50,
                p95_execution_time_ms=p95,
                p99_execution_time_ms=p99,
                rate_limit_hits=data["rate_limit_hits"],
                validation_failures=data["validation_failures"],
                last_called=data["last_called"],
                error_counts_by_type=dict(data["error_counts"])
            )

    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        """
        Get metrics for all tools.

        Returns:
            Dictionary mapping tool names to ToolMetrics objects
        """
        with self._lock:
            return {
                tool_name: self.get_tool_metrics(tool_name)
                for tool_name in self._tool_data.keys()
            }

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all tools.

        Returns:
            Dictionary with aggregate statistics:
                - total_tools: Number of unique tools used
                - total_executions: Total calls across all tools
                - total_successes: Total successful calls
                - total_failures: Total failed calls
                - overall_success_rate: Success rate across all tools
                - total_rate_limit_hits: Total rate limit violations
                - total_validation_failures: Total validation failures
                - most_used_tool: Tool with most calls
                - slowest_tool: Tool with highest avg execution time
                - fastest_tool: Tool with lowest avg execution time
        """
        with self._lock:
            all_metrics = self.get_all_metrics()

            if not all_metrics:
                return {
                    "total_tools": 0,
                    "total_executions": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                    "overall_success_rate": 0.0,
                    "total_rate_limit_hits": 0,
                    "total_validation_failures": 0,
                }

            # Aggregate statistics
            total_executions = sum(m.total_calls for m in all_metrics.values())
            total_successes = sum(m.successful_calls for m in all_metrics.values())
            total_failures = sum(m.failed_calls for m in all_metrics.values())
            total_rate_limit_hits = sum(m.rate_limit_hits for m in all_metrics.values())
            total_validation_failures = sum(m.validation_failures for m in all_metrics.values())

            success_rate = (
                total_successes / total_executions * 100
                if total_executions > 0
                else 0.0
            )

            # Find extremes
            most_used = max(all_metrics.values(), key=lambda m: m.total_calls)
            slowest = max(all_metrics.values(), key=lambda m: m.avg_execution_time_ms)
            fastest = min(all_metrics.values(), key=lambda m: m.avg_execution_time_ms or float('inf'))

            return {
                "total_tools": len(all_metrics),
                "total_executions": total_executions,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "overall_success_rate": round(success_rate, 2),
                "total_rate_limit_hits": total_rate_limit_hits,
                "total_validation_failures": total_validation_failures,
                "most_used_tool": {
                    "name": most_used.tool_name,
                    "calls": most_used.total_calls
                },
                "slowest_tool": {
                    "name": slowest.tool_name,
                    "avg_time_ms": round(slowest.avg_execution_time_ms, 2)
                },
                "fastest_tool": {
                    "name": fastest.tool_name,
                    "avg_time_ms": round(fastest.avg_execution_time_ms, 2)
                }
            }

    def reset_metrics(self, tool_name: Optional[str] = None) -> None:
        """
        Reset metrics (for testing or rotation).

        Args:
            tool_name: Reset specific tool (None = reset all)
        """
        with self._lock:
            if tool_name:
                if tool_name in self._tool_data:
                    del self._tool_data[tool_name]
                    logger.info("Reset metrics for tool: %s", tool_name)
            else:
                self._tool_data.clear()
                logger.info("Reset all metrics")

    def export_metrics(self, format: str = "json") -> Union[str, Dict]:
        """
        Export metrics in various formats.

        Args:
            format: Export format ("json", "prometheus", "csv")

        Returns:
            Formatted metrics as string or dict

        Raises:
            ValueError: If format is not supported
        """
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        elif format == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self) -> Dict[str, Any]:
        """
        Export metrics as JSON.

        Returns:
            Dictionary with all metrics and summary
        """
        all_metrics = self.get_all_metrics()
        summary = self.get_summary_stats()

        return {
            "summary": summary,
            "tools": {
                name: metrics.to_dict()
                for name, metrics in all_metrics.items()
            },
            "exported_at": datetime.now().isoformat()
        }

    def _export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        all_metrics = self.get_all_metrics()
        lines = []

        # Add HELP and TYPE comments
        lines.append("# HELP tool_calls_total Total number of tool calls")
        lines.append("# TYPE tool_calls_total counter")

        for name, metrics in all_metrics.items():
            # Sanitize tool name for Prometheus (replace invalid chars)
            safe_name = name.replace("-", "_").replace(".", "_")

            # Total calls
            lines.append(f'tool_calls_total{{tool="{safe_name}"}} {metrics.total_calls}')

        lines.append("")
        lines.append("# HELP tool_calls_successful Successful tool calls")
        lines.append("# TYPE tool_calls_successful counter")

        for name, metrics in all_metrics.items():
            safe_name = name.replace("-", "_").replace(".", "_")
            lines.append(f'tool_calls_successful{{tool="{safe_name}"}} {metrics.successful_calls}')

        lines.append("")
        lines.append("# HELP tool_calls_failed Failed tool calls")
        lines.append("# TYPE tool_calls_failed counter")

        for name, metrics in all_metrics.items():
            safe_name = name.replace("-", "_").replace(".", "_")
            lines.append(f'tool_calls_failed{{tool="{safe_name}"}} {metrics.failed_calls}')

        lines.append("")
        lines.append("# HELP tool_execution_time_milliseconds Tool execution time")
        lines.append("# TYPE tool_execution_time_milliseconds summary")

        for name, metrics in all_metrics.items():
            safe_name = name.replace("-", "_").replace(".", "_")
            lines.append(f'tool_execution_time_milliseconds{{tool="{safe_name}",quantile="0.5"}} {metrics.p50_execution_time_ms}')
            lines.append(f'tool_execution_time_milliseconds{{tool="{safe_name}",quantile="0.95"}} {metrics.p95_execution_time_ms}')
            lines.append(f'tool_execution_time_milliseconds{{tool="{safe_name}",quantile="0.99"}} {metrics.p99_execution_time_ms}')

        lines.append("")
        lines.append("# HELP tool_rate_limit_hits Rate limit violations")
        lines.append("# TYPE tool_rate_limit_hits counter")

        for name, metrics in all_metrics.items():
            safe_name = name.replace("-", "_").replace(".", "_")
            lines.append(f'tool_rate_limit_hits{{tool="{safe_name}"}} {metrics.rate_limit_hits}')

        lines.append("")
        lines.append("# HELP tool_validation_failures Validation failures")
        lines.append("# TYPE tool_validation_failures counter")

        for name, metrics in all_metrics.items():
            safe_name = name.replace("-", "_").replace(".", "_")
            lines.append(f'tool_validation_failures{{tool="{safe_name}"}} {metrics.validation_failures}')

        return "\n".join(lines)

    def _export_csv(self) -> str:
        """
        Export metrics as CSV.

        Returns:
            CSV-formatted metrics string
        """
        all_metrics = self.get_all_metrics()

        lines = [
            "tool_name,total_calls,successful_calls,failed_calls,"
            "avg_execution_time_ms,p50_execution_time_ms,p95_execution_time_ms,p99_execution_time_ms,"
            "rate_limit_hits,validation_failures,last_called"
        ]

        for name, metrics in all_metrics.items():
            last_called = metrics.last_called.isoformat() if metrics.last_called else ""
            lines.append(
                f"{name},{metrics.total_calls},{metrics.successful_calls},{metrics.failed_calls},"
                f"{metrics.avg_execution_time_ms:.2f},{metrics.p50_execution_time_ms:.2f},"
                f"{metrics.p95_execution_time_ms:.2f},{metrics.p99_execution_time_ms:.2f},"
                f"{metrics.rate_limit_hits},{metrics.validation_failures},{last_called}"
            )

        return "\n".join(lines)

    @staticmethod
    def _calculate_percentiles(values: List[float]) -> tuple[float, float, float]:
        """
        Calculate p50, p95, p99 percentiles.

        Args:
            values: List of values

        Returns:
            Tuple of (p50, p95, p99)
        """
        if not values:
            return (0.0, 0.0, 0.0)

        sorted_values = sorted(values)
        n = len(sorted_values)

        def get_percentile(p: float) -> float:
            """Get percentile value."""
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return sorted_values[idx]

        p50 = get_percentile(0.50)
        p95 = get_percentile(0.95)
        p99 = get_percentile(0.99)

        return (p50, p95, p99)


# ==============================================================================
# Global Telemetry Singleton
# ==============================================================================

_global_telemetry: Optional[TelemetryCollector] = None
_telemetry_lock = threading.Lock()


def get_telemetry() -> TelemetryCollector:
    """
    Get global telemetry collector (singleton).

    Returns:
        Global TelemetryCollector instance
    """
    global _global_telemetry

    if _global_telemetry is None:
        with _telemetry_lock:
            if _global_telemetry is None:
                _global_telemetry = TelemetryCollector(enable_real_time=True)

    return _global_telemetry


def reset_global_telemetry() -> None:
    """Reset global telemetry instance (for testing)."""
    global _global_telemetry
    with _telemetry_lock:
        _global_telemetry = None
