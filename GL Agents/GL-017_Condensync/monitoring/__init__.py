# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Monitoring Module
======================================

Comprehensive monitoring infrastructure for condenser optimization.
Provides Prometheus metrics, health checks, and metrics export capabilities.

Components:
- metrics.py: Prometheus metrics (KPIs, latencies, recommendations, alerts)
- health.py: Health checks (liveness, readiness, dependencies)
- metrics_exporter.py: Metrics export (Prometheus endpoint, JSON, Grafana)

Example:
    >>> from monitoring import (
    ...     CondenserMetrics,
    ...     HealthCheckManager,
    ...     MetricsExporter,
    ... )
    >>>
    >>> # Initialize metrics
    >>> metrics = CondenserMetrics()
    >>> metrics.initialize("1.0.0", "production")
    >>>
    >>> # Record condenser KPIs
    >>> metrics.record_condenser_kpi(
    ...     condenser_id="COND-001",
    ...     cleanliness_factor=0.85,
    ...     ttd=3.5,
    ... )
    >>>
    >>> # Initialize health checks
    >>> health = HealthCheckManager(version="1.0.0")
    >>> health.register_dependency_check("opc_ua", check_opc_ua)
    >>>
    >>> # Start metrics server
    >>> exporter = MetricsExporter()
    >>> exporter.start_server(port=9090)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Technologies"

# Metrics
from .metrics import (
    # Main class
    CondenserMetrics,
    get_metrics_instance,
    initialize_metrics,
    # Data classes
    CondenserKPI,
    RecommendationMetrics,
    DataQualityMetrics,
    AlertMetrics,
    # Enums
    CondenserType,
    CalculationType,
    RecommendationType,
    RecommendationPriority,
    AlertSeverity,
    DataQualityDimension,
    DependencyType,
    # Prometheus metrics
    CLEANLINESS_FACTOR_GAUGE,
    TERMINAL_TEMP_DIFFERENCE_GAUGE,
    VACUUM_PRESSURE_GAUGE,
    HEAT_DUTY_GAUGE,
    CALCULATION_LATENCY_HISTOGRAM,
    RECOMMENDATIONS_GENERATED_COUNTER,
    DATA_QUALITY_SCORE_GAUGE,
    ALERTS_RAISED_COUNTER,
    API_REQUESTS_COUNTER,
)

# Health checks
from .health import (
    # Main class
    HealthCheckManager,
    # Data classes
    HealthCheckResult,
    ComponentHealth,
    DataFreshnessConfig,
    DependencyConfig,
    # Enums
    HealthStatus,
    ProbeType,
    CheckCategory,
    # Standard checks
    check_opc_ua_health,
    check_kafka_health,
    check_cmms_health,
    check_database_health,
    check_redis_health,
    check_pi_server_health,
    # Resource checks
    check_memory_health,
    check_cpu_health,
    check_disk_health,
)

# Metrics exporter
from .metrics_exporter import (
    MetricsExporter,
    MetricsHTTPHandler,
    get_exporter_instance,
)


__all__ = [
    # Version
    "__version__",
    # Metrics
    "CondenserMetrics",
    "get_metrics_instance",
    "initialize_metrics",
    "CondenserKPI",
    "RecommendationMetrics",
    "DataQualityMetrics",
    "AlertMetrics",
    "CondenserType",
    "CalculationType",
    "RecommendationType",
    "RecommendationPriority",
    "AlertSeverity",
    "DataQualityDimension",
    "DependencyType",
    "CLEANLINESS_FACTOR_GAUGE",
    "TERMINAL_TEMP_DIFFERENCE_GAUGE",
    "VACUUM_PRESSURE_GAUGE",
    "HEAT_DUTY_GAUGE",
    "CALCULATION_LATENCY_HISTOGRAM",
    "RECOMMENDATIONS_GENERATED_COUNTER",
    "DATA_QUALITY_SCORE_GAUGE",
    "ALERTS_RAISED_COUNTER",
    "API_REQUESTS_COUNTER",
    # Health checks
    "HealthCheckManager",
    "HealthCheckResult",
    "ComponentHealth",
    "DataFreshnessConfig",
    "DependencyConfig",
    "HealthStatus",
    "ProbeType",
    "CheckCategory",
    "check_opc_ua_health",
    "check_kafka_health",
    "check_cmms_health",
    "check_database_health",
    "check_redis_health",
    "check_pi_server_health",
    "check_memory_health",
    "check_cpu_health",
    "check_disk_health",
    # Metrics exporter
    "MetricsExporter",
    "MetricsHTTPHandler",
    "get_exporter_instance",
]
