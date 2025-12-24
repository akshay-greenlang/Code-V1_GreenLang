"""
GL-008 TRAPCATCHER - Monitoring Module

Prometheus metrics, health checks, and observability for steam trap monitoring.
"""

from .metrics import (
    MetricsExporter,
    TRAP_DIAGNOSES_TOTAL,
    TRAP_FAILURES_DETECTED,
    ENERGY_LOSS_KW,
    CO2_EMISSIONS_KG,
    DIAGNOSIS_LATENCY,
    FLEET_HEALTH_SCORE,
)
from .health import HealthChecker, HealthStatus

__all__ = [
    "MetricsExporter",
    "HealthChecker",
    "HealthStatus",
    "TRAP_DIAGNOSES_TOTAL",
    "TRAP_FAILURES_DETECTED",
    "ENERGY_LOSS_KW",
    "CO2_EMISSIONS_KG",
    "DIAGNOSIS_LATENCY",
    "FLEET_HEALTH_SCORE",
]
