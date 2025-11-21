# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - Backend Module

Production-grade monitoring and observability components.

Components:
- health: Health check endpoints (liveness, readiness, basic)
- logging_config: Structured JSON logging with correlation IDs
- metrics: Prometheus metrics for monitoring
"""

from .health import CBAMHealthChecker, create_health_endpoints
from .logging_config import (
    LoggingConfig,
    StructuredLogger,
    CorrelationContext,
    configure_development_logging,
    configure_production_logging,
)
from .metrics import CBAMMetrics, create_metrics_endpoint, get_metrics

__all__ = [
    # Health checks
    "CBAMHealthChecker",
    "create_health_endpoints",
    # Logging
    "LoggingConfig",
    "StructuredLogger",
    "CorrelationContext",
    "configure_development_logging",
    "configure_production_logging",
    # Metrics
    "CBAMMetrics",
    "create_metrics_endpoint",
    "get_metrics",
]

__version__ = "1.0.0"
