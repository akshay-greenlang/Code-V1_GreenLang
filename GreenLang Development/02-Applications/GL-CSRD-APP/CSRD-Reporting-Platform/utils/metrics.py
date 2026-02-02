# -*- coding: utf-8 -*-
"""
Performance Metrics and Monitoring

Provides Prometheus-compatible metrics for monitoring CSRD pipeline performance.

Author: GreenLang AI Team
Date: 2025-10-18
"""

from typing import Dict, Any, Optional
import time
from datetime import datetime
from pathlib import Path
import json
from greenlang.determinism import DeterministicClock


class PerformanceMonitor:
    """
    Monitor pipeline performance metrics

    In production, this would integrate with Prometheus.
    For now, we collect metrics in memory and save to file.
    """

    def __init__(self):
        """Initialize performance monitor"""
        self.metrics = {
            'counters': {},
            'histograms': {},
            'gauges': {}
        }
        self.start_time = time.time()

    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """
        Record a counter metric

        Args:
            name: Metric name
            value: Value to add (default 1.0)
            labels: Optional labels for the metric
        """
        key = self._make_key(name, labels)

        if key not in self.metrics['counters']:
            self.metrics['counters'][key] = 0

        self.metrics['counters'][key] += value

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram metric (for tracking distributions)

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for the metric
        """
        key = self._make_key(name, labels)

        if key not in self.metrics['histograms']:
            self.metrics['histograms'][key] = []

        self.metrics['histograms'][key].append({
            'value': value,
            'timestamp': time.time()
        })

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric (for tracking current values)

        Args:
            name: Metric name
            value: Current value
            labels: Optional labels for the metric
        """
        key = self._make_key(name, labels)

        self.metrics['gauges'][key] = {
            'value': value,
            'timestamp': time.time()
        }

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create metric key from name and labels"""
        if labels:
            label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics

        Returns:
            Dictionary with metric summaries
        """
        summary = {
            'collection_time': DeterministicClock.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'counters': self.metrics['counters'].copy(),
            'gauges': {
                key: data['value']
                for key, data in self.metrics['gauges'].items()
            },
            'histograms': {}
        }

        # Calculate histogram statistics
        for key, values in self.metrics['histograms'].items():
            if values:
                vals = [v['value'] for v in values]
                summary['histograms'][key] = {
                    'count': len(vals),
                    'sum': sum(vals),
                    'mean': sum(vals) / len(vals),
                    'min': min(vals),
                    'max': max(vals)
                }

        return summary

    def save_to_file(self, output_path: str):
        """
        Save metrics to JSON file

        Args:
            output_path: Path to save metrics
        """
        summary = self.get_summary()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'counters': {},
            'histograms': {},
            'gauges': {}
        }
        self.start_time = time.time()


class Timer:
    """Context manager for timing operations"""

    def __init__(self, monitor: PerformanceMonitor, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Initialize timer

        Args:
            monitor: PerformanceMonitor instance
            metric_name: Name of the metric to record
            labels: Optional labels for the metric
        """
        self.monitor = monitor
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record duration"""
        duration = time.time() - self.start_time
        self.monitor.record_histogram(self.metric_name, duration, self.labels)


# Global metrics instance
_global_monitor = None


def setup_metrics() -> PerformanceMonitor:
    """
    Setup global metrics monitor

    Returns:
        Global PerformanceMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()

    return _global_monitor


def get_metrics() -> PerformanceMonitor:
    """
    Get global metrics monitor

    Returns:
        Global PerformanceMonitor instance
    """
    global _global_monitor

    if _global_monitor is None:
        return setup_metrics()

    return _global_monitor


# Example usage
if __name__ == "__main__":
    # Setup metrics
    monitor = setup_metrics()

    # Record some metrics
    monitor.record_counter('intake_records_processed', value=1000)
    monitor.record_counter('calculations_completed', value=547)

    # Time an operation
    with Timer(monitor, 'calculation_duration_seconds'):
        time.sleep(0.1)  # Simulate work

    # Set gauges
    monitor.set_gauge('active_pipelines', 3)
    monitor.set_gauge('data_quality_score', 92.5)

    # Get summary
    summary = monitor.get_summary()

    print("Metrics Summary:")
    print(f"Uptime: {summary['uptime_seconds']:.2f}s")
    print(f"Counters: {summary['counters']}")
    print(f"Gauges: {summary['gauges']}")
    print(f"Histograms: {summary['histograms']}")

    # Save to file
    monitor.save_to_file('output/metrics.json')
    print("\nMetrics saved to output/metrics.json")
