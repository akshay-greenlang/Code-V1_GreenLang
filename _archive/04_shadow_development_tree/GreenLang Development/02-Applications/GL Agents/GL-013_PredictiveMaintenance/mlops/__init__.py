# -*- coding: utf-8 -*-
"""
GL-013 PredictiveMaintenance - MLOps Module

Provides model registry, monitoring, and drift detection.
"""

__version__ = "1.0.0"

from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelArtifact,
    ModelStatus,
    ModelType,
)

from .drift_detection import (
    DriftDetector,
    DriftConfig,
    DriftResult,
    DriftType,
    DriftSeverity,
)

from .monitoring import (
    ModelMonitor,
    MonitoringConfig,
    MetricValue,
    MetricType,
    Alert,
    AlertLevel,
)

__all__ = [
    "__version__",
    "ModelRegistry",
    "ModelVersion",
    "ModelArtifact",
    "ModelStatus",
    "ModelType",
    "DriftDetector",
    "DriftConfig",
    "DriftResult",
    "DriftType",
    "DriftSeverity",
    "ModelMonitor",
    "MonitoringConfig",
    "MetricValue",
    "MetricType",
    "Alert",
    "AlertLevel",
]
