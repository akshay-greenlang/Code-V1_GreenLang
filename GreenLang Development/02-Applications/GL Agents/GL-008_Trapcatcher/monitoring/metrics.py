"""
GL-008 TRAPCATCHER - Prometheus Metrics Exporter

Exposes operational metrics for Kubernetes monitoring and alerting.
Implements RED (Rate, Errors, Duration) and USE (Utilization, Saturation, Errors)
metrics patterns for comprehensive observability.

Standards:
- OpenMetrics specification
- Prometheus best practices
- Kubernetes observability patterns
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

# Counter: Total number of trap diagnoses performed
TRAP_DIAGNOSES_TOTAL = Counter(
    "trapcatcher_diagnoses_total",
    "Total number of steam trap diagnoses performed",
    labelnames=["trap_type", "condition", "severity"],
)

# Counter: Failed traps detected
TRAP_FAILURES_DETECTED = Counter(
    "trapcatcher_failures_detected_total",
    "Total number of failed steam traps detected",
    labelnames=["trap_type", "failure_mode"],
)

# Gauge: Current energy loss in kW
ENERGY_LOSS_KW = Gauge(
    "trapcatcher_energy_loss_kw",
    "Current estimated energy loss from failed traps (kW)",
    labelnames=["facility", "area"],
)

# Gauge: CO2 emissions in kg
CO2_EMISSIONS_KG = Gauge(
    "trapcatcher_co2_emissions_kg",
    "Estimated CO2 emissions from steam losses (kg)",
    labelnames=["facility", "area"],
)

# Histogram: Diagnosis latency
DIAGNOSIS_LATENCY = Histogram(
    "trapcatcher_diagnosis_duration_seconds",
    "Time taken to diagnose a steam trap",
    labelnames=["trap_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Gauge: Fleet health score (0-100)
FLEET_HEALTH_SCORE = Gauge(
    "trapcatcher_fleet_health_score",
    "Overall fleet health score (0-100)",
    labelnames=["facility"],
)

# Counter: API requests
API_REQUESTS_TOTAL = Counter(
    "trapcatcher_api_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status"],
)

# Histogram: API latency
API_LATENCY = Histogram(
    "trapcatcher_api_latency_seconds",
    "API request latency",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Gauge: Active connections
ACTIVE_CONNECTIONS = Gauge(
    "trapcatcher_active_connections",
    "Number of active connections",
    labelnames=["connection_type"],
)

# Counter: Errors
ERRORS_TOTAL = Counter(
    "trapcatcher_errors_total",
    "Total errors encountered",
    labelnames=["error_type", "component"],
)

# Info: Agent metadata
AGENT_INFO = Info(
    "trapcatcher_agent",
    "Agent metadata",
)


# =============================================================================
# METRICS EXPORTER CLASS
# =============================================================================

class TrapConditionLabel(str, Enum):
    """Labels for trap condition metrics."""
    NORMAL = "normal"
    LEAKING = "leaking"
    BLOCKED = "blocked"
    BLOWTHROUGH = "blowthrough"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"


class TrapTypeLabel(str, Enum):
    """Labels for trap type metrics."""
    THERMODYNAMIC = "thermodynamic"
    THERMOSTATIC = "thermostatic"
    MECHANICAL_FLOAT = "mechanical_float"
    MECHANICAL_BUCKET = "mechanical_bucket"
    ORIFICE = "orifice"


@dataclass
class DiagnosisMetrics:
    """Metrics for a single diagnosis."""
    trap_id: str
    trap_type: str
    condition: str
    severity: str
    energy_loss_kw: float
    co2_kg: float
    diagnosis_time_seconds: float
    is_failure: bool
    failure_mode: Optional[str] = None


class MetricsExporter:
    """
    Prometheus metrics exporter for GL-008 TRAPCATCHER.

    Provides methods to record operational metrics and expose
    them via HTTP endpoint for Prometheus scraping.

    Usage:
        exporter = MetricsExporter()
        exporter.set_agent_info("1.0.0", "production")

        # Record a diagnosis
        with exporter.measure_diagnosis("thermodynamic") as timer:
            result = classifier.classify(input_data)
        exporter.record_diagnosis(metrics)

        # Get metrics for HTTP response
        content = exporter.get_metrics()
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics exporter.

        Args:
            registry: Optional custom registry for testing
        """
        self.registry = registry or REGISTRY
        self._initialized = False

    def set_agent_info(
        self,
        version: str,
        environment: str,
        instance_id: Optional[str] = None,
    ) -> None:
        """
        Set agent metadata info metric.

        Args:
            version: Agent version string
            environment: Deployment environment (dev, staging, production)
            instance_id: Optional instance identifier
        """
        AGENT_INFO.info({
            "version": version,
            "environment": environment,
            "instance_id": instance_id or "default",
            "agent_id": "GL-008",
            "agent_name": "TRAPCATCHER",
        })
        self._initialized = True

    @contextmanager
    def measure_diagnosis(
        self, trap_type: str
    ) -> Generator[None, None, None]:
        """
        Context manager to measure diagnosis duration.

        Args:
            trap_type: Type of trap being diagnosed

        Yields:
            None - timing happens automatically

        Example:
            with exporter.measure_diagnosis("thermodynamic"):
                result = classifier.classify(input_data)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            DIAGNOSIS_LATENCY.labels(trap_type=trap_type).observe(duration)

    def record_diagnosis(self, metrics: DiagnosisMetrics) -> None:
        """
        Record metrics from a completed diagnosis.

        Args:
            metrics: Diagnosis metrics to record
        """
        # Increment diagnosis counter
        TRAP_DIAGNOSES_TOTAL.labels(
            trap_type=metrics.trap_type,
            condition=metrics.condition,
            severity=metrics.severity,
        ).inc()

        # Record failure if detected
        if metrics.is_failure and metrics.failure_mode:
            TRAP_FAILURES_DETECTED.labels(
                trap_type=metrics.trap_type,
                failure_mode=metrics.failure_mode,
            ).inc()

        logger.debug(
            f"Recorded diagnosis metrics for trap {metrics.trap_id}: "
            f"condition={metrics.condition}, energy_loss={metrics.energy_loss_kw}kW"
        )

    def update_fleet_metrics(
        self,
        facility: str,
        area: str,
        total_energy_loss_kw: float,
        total_co2_kg: float,
        health_score: float,
    ) -> None:
        """
        Update fleet-level aggregate metrics.

        Args:
            facility: Facility identifier
            area: Area within facility
            total_energy_loss_kw: Total energy loss from all failed traps
            total_co2_kg: Total CO2 emissions
            health_score: Fleet health score (0-100)
        """
        ENERGY_LOSS_KW.labels(
            facility=facility,
            area=area,
        ).set(total_energy_loss_kw)

        CO2_EMISSIONS_KG.labels(
            facility=facility,
            area=area,
        ).set(total_co2_kg)

        FLEET_HEALTH_SCORE.labels(facility=facility).set(health_score)

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
    ) -> None:
        """
        Record API request metrics.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            status: HTTP status code
            duration: Request duration in seconds
        """
        API_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=str(status),
        ).inc()

        API_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

    def record_error(
        self,
        error_type: str,
        component: str,
    ) -> None:
        """
        Record an error occurrence.

        Args:
            error_type: Type of error (validation, calculation, integration)
            component: Component where error occurred
        """
        ERRORS_TOTAL.labels(
            error_type=error_type,
            component=component,
        ).inc()

    def set_active_connections(
        self,
        connection_type: str,
        count: int,
    ) -> None:
        """
        Set number of active connections.

        Args:
            connection_type: Type of connection (opcua, modbus, mqtt, http)
            count: Number of active connections
        """
        ACTIVE_CONNECTIONS.labels(connection_type=connection_type).set(count)

    def get_metrics(self) -> bytes:
        """
        Generate Prometheus metrics output.

        Returns:
            Prometheus-formatted metrics as bytes
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """
        Get content type for metrics response.

        Returns:
            Prometheus content type string
        """
        return CONTENT_TYPE_LATEST


# =============================================================================
# FASTAPI MIDDLEWARE
# =============================================================================

def create_metrics_middleware(exporter: MetricsExporter) -> Callable:
    """
    Create FastAPI middleware for automatic request metrics.

    Args:
        exporter: MetricsExporter instance

    Returns:
        Middleware function
    """
    async def metrics_middleware(request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        exporter.record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration,
        )

        return response

    return metrics_middleware
