# -*- coding: utf-8 -*-
"""
GreenLang Monitoring & Observability Layer
==========================================

Consolidated module containing monitoring, telemetry, observability, and sandbox.
Production monitoring and observability for GreenLang v2.0.0.

This module provides:
- Prometheus metrics integration (50+ standard metrics per agent)
- StandardAgentMetrics with 71 baseline metrics
- Agent-specific metric extensions
- Health check endpoints
- Performance monitoring
- Resource usage tracking
- Custom metrics collection
- Metrics validation tools
- Telemetry and logging (from telemetry/)
- Observability features (from observability/)
- Sandbox monitoring (from sandbox/)
- PushGateway SDK for batch job metrics (OBS-001 Phase 3)

Sub-modules:
- monitoring.metrics: Core metrics collection (base directory)
- monitoring.telemetry: Logging, metrics, health tracking
- monitoring.observability: Observability infrastructure
- monitoring.sandbox: Sandbox capabilities, OS isolation
- monitoring.pushgateway: PushGateway SDK for batch jobs
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

# Import PushGateway SDK (if available)
try:
    from .pushgateway import (
        BatchJobMetrics,
        PushGatewayConfig,
        PushGatewayError,
        get_pushgateway_client,
        clear_pushgateway_clients,
        create_batch_job_metrics,
        PROMETHEUS_CLIENT_AVAILABLE as PUSHGATEWAY_AVAILABLE,
    )
    _HAS_PUSHGATEWAY = True
except ImportError:
    _HAS_PUSHGATEWAY = False
    BatchJobMetrics = None
    PushGatewayConfig = None
    PushGatewayError = None
    get_pushgateway_client = None
    clear_pushgateway_clients = None
    create_batch_job_metrics = None
    PUSHGATEWAY_AVAILABLE = False

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
    # PushGateway SDK (OBS-001 Phase 3)
    "BatchJobMetrics",
    "PushGatewayConfig",
    "PushGatewayError",
    "get_pushgateway_client",
    "clear_pushgateway_clients",
    "create_batch_job_metrics",
    "PUSHGATEWAY_AVAILABLE",
]

__version__ = "2.1.0"
__standard_metrics_count__ = 71
__min_required_metrics__ = 50