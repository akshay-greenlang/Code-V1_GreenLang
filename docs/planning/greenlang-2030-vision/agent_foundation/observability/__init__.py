# -*- coding: utf-8 -*-
"""
GreenLang Agent Foundation - Observability Infrastructure
=========================================================

Comprehensive monitoring, logging, tracing, and debugging capabilities
for production-grade agent systems.

Components:
- Structured JSON logging with OpenTelemetry
- Distributed tracing (Jaeger, DataDog, New Relic)
- Prometheus metrics collection
- Performance monitoring and SLA tracking
- Grafana dashboards for all stakeholders
- Advanced debugging and troubleshooting tools

Author: GL-DevOpsEngineer
Version: 1.0.0
"""

from .logging import (
    StructuredLogger,
    LogLevel,
    LogContext,
    setup_logging
)

from .tracing import (
    TracingManager,
    SpanContext,
    trace_agent,
    trace_llm_call,
    setup_tracing
)

from .metrics import (
    MetricsCollector,
    MetricType,
    register_metric,
    record_metric,
    setup_metrics
)

from .performance_monitor import (
    PerformanceMonitor,
    SLATracker,
    LatencyTracker,
    ErrorRateMonitor
)

from .dashboards import (
    DashboardGenerator,
    DashboardType,
    generate_dashboard,
    export_grafana_config
)

from .debugging import (
    DebugTools,
    HealthChecker,
    Profiler,
    LogAnalyzer,
    troubleshoot_issue
)

__all__ = [
    # Logging
    'StructuredLogger',
    'LogLevel',
    'LogContext',
    'setup_logging',

    # Tracing
    'TracingManager',
    'SpanContext',
    'trace_agent',
    'trace_llm_call',
    'setup_tracing',

    # Metrics
    'MetricsCollector',
    'MetricType',
    'register_metric',
    'record_metric',
    'setup_metrics',

    # Performance
    'PerformanceMonitor',
    'SLATracker',
    'LatencyTracker',
    'ErrorRateMonitor',

    # Dashboards
    'DashboardGenerator',
    'DashboardType',
    'generate_dashboard',
    'export_grafana_config',

    # Debugging
    'DebugTools',
    'HealthChecker',
    'Profiler',
    'LogAnalyzer',
    'troubleshoot_issue'
]

__version__ = '1.0.0'