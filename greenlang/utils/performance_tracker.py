"""
Performance Tracker Utility

Performance monitoring and benchmarking for GreenLang agents.
Tracks execution times, memory usage, and other metrics.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceMetric:
    """Data class for performance metrics."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    memory_delta: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """Performance tracking utility for monitoring agent execution."""

    def __init__(self, agent_id: str):
        """Initialize performance tracker for an agent.

        Args:
            agent_id: The ID of the agent being tracked
        """
        self.agent_id = agent_id
        self.metrics: List[PerformanceMetric] = []
        self.active_metrics: Dict[str, PerformanceMetric] = {}
        self.summary_stats: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")

        # Performance thresholds
        self.slow_execution_threshold = 1.0  # seconds
        self.high_memory_threshold = 100  # MB

        # Initialize process monitor
        self.process = psutil.Process()

    @contextmanager
    def track(self, operation_name: str, **metadata):
        """Context manager for tracking performance of an operation.

        Args:
            operation_name: Name of the operation being tracked
            **metadata: Additional metadata to store with the metric

        Yields:
            PerformanceMetric: The metric being tracked
        """
        metric = self.start_tracking(operation_name, **metadata)
        try:
            yield metric
        finally:
            self.stop_tracking(operation_name)

    def start_tracking(self, operation_name: str, **metadata) -> PerformanceMetric:
        """Start tracking performance for an operation.

        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata

        Returns:
            PerformanceMetric: The metric being tracked
        """
        # Get current memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        metric = PerformanceMetric(
            name=operation_name,
            start_time=time.perf_counter(),
            memory_start=memory_mb,
            metadata=metadata,
        )

        self.active_metrics[operation_name] = metric
        self.logger.debug(f"Started tracking: {operation_name}")

        return metric

    def stop_tracking(self, operation_name: str) -> Optional[PerformanceMetric]:
        """Stop tracking performance for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Optional[PerformanceMetric]: The completed metric or None if not found
        """
        if operation_name not in self.active_metrics:
            self.logger.warning(f"No active tracking for: {operation_name}")
            return None

        metric = self.active_metrics.pop(operation_name)

        # Calculate duration
        metric.end_time = time.perf_counter()
        metric.duration = metric.end_time - metric.start_time

        # Get final memory usage
        memory_info = self.process.memory_info()
        metric.memory_end = memory_info.rss / 1024 / 1024
        metric.memory_delta = metric.memory_end - metric.memory_start

        # Store metric
        self.metrics.append(metric)
        self.summary_stats[operation_name].append(metric.duration)

        # Check for performance issues
        if metric.duration > self.slow_execution_threshold:
            self.logger.warning(
                f"Slow execution detected: {operation_name} took {metric.duration:.3f}s"
            )

        if abs(metric.memory_delta) > self.high_memory_threshold:
            self.logger.warning(
                f"High memory usage: {operation_name} used {metric.memory_delta:.1f}MB"
            )

        self.logger.debug(
            f"Stopped tracking: {operation_name} (duration: {metric.duration:.3f}s)"
        )

        return metric

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all tracked operations.

        Returns:
            Dict containing summary statistics
        """
        summary = {
            "agent_id": self.agent_id,
            "total_operations": len(self.metrics),
            "operations": {},
        }

        for operation_name, durations in self.summary_stats.items():
            if durations:
                summary["operations"][operation_name] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "average_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                }

        # Overall statistics
        all_durations = [m.duration for m in self.metrics if m.duration]
        if all_durations:
            summary["overall"] = {
                "total_time": sum(all_durations),
                "average_time": sum(all_durations) / len(all_durations),
                "min_time": min(all_durations),
                "max_time": max(all_durations),
            }

        # Memory statistics
        memory_deltas = [
            m.memory_delta for m in self.metrics if m.memory_delta is not None
        ]
        if memory_deltas:
            summary["memory"] = {
                "average_delta_mb": sum(memory_deltas) / len(memory_deltas),
                "max_delta_mb": max(memory_deltas),
                "total_allocated_mb": sum(d for d in memory_deltas if d > 0),
            }

        return summary

    def get_slow_operations(
        self, threshold: Optional[float] = None
    ) -> List[PerformanceMetric]:
        """Get operations that exceeded the execution time threshold.

        Args:
            threshold: Custom threshold in seconds (uses default if not provided)

        Returns:
            List of slow operations
        """
        threshold = threshold or self.slow_execution_threshold
        return [m for m in self.metrics if m.duration and m.duration > threshold]

    def get_memory_intensive_operations(
        self, threshold: Optional[float] = None
    ) -> List[PerformanceMetric]:
        """Get operations that used significant memory.

        Args:
            threshold: Custom threshold in MB (uses default if not provided)

        Returns:
            List of memory-intensive operations
        """
        threshold = threshold or self.high_memory_threshold
        return [
            m
            for m in self.metrics
            if m.memory_delta and abs(m.memory_delta) > threshold
        ]

    def benchmark(self, operation_name: str, iterations: int = 10) -> Dict[str, float]:
        """Benchmark an operation by running it multiple times.

        Args:
            operation_name: Name of the operation to benchmark
            iterations: Number of iterations to run

        Returns:
            Dict containing benchmark results
        """
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            # The actual operation would be performed here
            # For now, we'll just simulate with a small delay
            time.sleep(0.001)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "operation": operation_name,
            "iterations": iterations,
            "total_time": sum(times),
            "average_time": sum(times) / iterations,
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": self._calculate_std_dev(times),
        }

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values.

        Args:
            values: List of numeric values

        Returns:
            float: Standard deviation
        """
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics to a file.

        Args:
            format: Export format (json, csv)

        Returns:
            str: Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_{self.agent_id}_{timestamp}.{format}"

        if format == "json":
            import json

            data = {
                "agent_id": self.agent_id,
                "metrics": [
                    {
                        "name": m.name,
                        "duration": m.duration,
                        "memory_delta": m.memory_delta,
                        "metadata": m.metadata,
                    }
                    for m in self.metrics
                ],
                "summary": self.get_summary(),
            }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            import csv

            with open(filename, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["name", "duration", "memory_delta", "timestamp"]
                )
                writer.writeheader()

                for m in self.metrics:
                    writer.writerow(
                        {
                            "name": m.name,
                            "duration": m.duration,
                            "memory_delta": m.memory_delta,
                            "timestamp": m.start_time,
                        }
                    )

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported metrics to {filename}")
        return filename

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.active_metrics.clear()
        self.summary_stats.clear()
        self.logger.info("Performance tracker reset")

    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of the process.

        Returns:
            Dict containing memory usage information in MB
        """
        memory_info = self.process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
        }

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage.

        Returns:
            float: CPU usage percentage
        """
        return self.process.cpu_percent(interval=0.1)
