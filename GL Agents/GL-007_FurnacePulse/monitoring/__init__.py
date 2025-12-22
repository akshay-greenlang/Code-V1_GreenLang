"""
FurnacePulse Monitoring Module

This module provides monitoring capabilities for the FurnacePulse furnace
monitoring system, including metrics collection, data quality monitoring,
and performance tracking.

Components:
    - MetricsCollector: Prometheus metrics for alert volumes, pipeline health, model performance
    - DataQualityMonitor: Signal quality tracking, drift detection, completeness metrics

Example:
    >>> from monitoring import MetricsCollector, DataQualityMonitor
    >>> metrics = MetricsCollector(config)
    >>> dq_monitor = DataQualityMonitor(config)
    >>> quality = dq_monitor.assess_signal_quality(sensor_readings)
    >>> metrics.record_data_quality(quality)
"""

from monitoring.metrics_collector import (
    MetricsCollector,
    MetricsCollectorConfig,
    AlertMetrics,
    PipelineMetrics,
    ModelMetrics,
)
from monitoring.data_quality_monitor import (
    DataQualityMonitor,
    DataQualityMonitorConfig,
    SignalQuality,
    SignalQualityFlag,
    DriftResult,
)

__all__ = [
    # Metrics Collector
    "MetricsCollector",
    "MetricsCollectorConfig",
    "AlertMetrics",
    "PipelineMetrics",
    "ModelMetrics",
    # Data Quality Monitor
    "DataQualityMonitor",
    "DataQualityMonitorConfig",
    "SignalQuality",
    "SignalQualityFlag",
    "DriftResult",
]

__version__ = "1.0.0"
