# -*- coding: utf-8 -*-
"""
GreenLang Monitoring
===================

Production monitoring and observability for GreenLang v0.2.0.

This module provides:
- Prometheus metrics integration (50+ standard metrics per agent)
- StandardAgentMetrics with 71 baseline metrics
- Agent-specific metric extensions
- Health check endpoints
- Performance monitoring
- Resource usage tracking
- Custom metrics collection
- Metrics validation tools
"""

from .metrics import (
    MetricsCollector,
    PrometheusExporter,
    CustomMetric,
    MetricType,
    setup_metrics,
)
from .health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    create_health_app,
)

# Import standard metrics (if available)
try:
    from .standard_metrics import (
        StandardAgentMetrics,
        track_with_metrics,
        PROMETHEUS_AVAILABLE,
    )
    _HAS_STANDARD_METRICS = True
except ImportError:
    _HAS_STANDARD_METRICS = False
    StandardAgentMetrics = None
    track_with_metrics = None
    PROMETHEUS_AVAILABLE = False

# Import agent extensions (if available)
try:
    from .agent_extensions import (
        ProcessHeatMetrics,
        BoilerOptimizerMetrics,
        HeatRecoveryMetrics,
    )
    _HAS_AGENT_EXTENSIONS = True
except ImportError:
    _HAS_AGENT_EXTENSIONS = False
    ProcessHeatMetrics = None
    BoilerOptimizerMetrics = None
    HeatRecoveryMetrics = None

__all__ = [
    # Core metrics
    "MetricsCollector",
    "PrometheusExporter",
    "CustomMetric",
    "MetricType",
    "setup_metrics",
    # Health checks
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "create_health_app",
    # Standard metrics (71 baseline)
    "StandardAgentMetrics",
    "track_with_metrics",
    "PROMETHEUS_AVAILABLE",
    # Agent extensions
    "ProcessHeatMetrics",
    "BoilerOptimizerMetrics",
    "HeatRecoveryMetrics",
]

__version__ = "1.0.0"
__standard_metrics_count__ = 71
__min_required_metrics__ = 50