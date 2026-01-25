# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Prometheus Metrics Exporter

Production-grade Prometheus metrics exporter with comprehensive observability
for steam trap diagnostics. Implements RED (Rate, Errors, Duration) and
USE (Utilization, Saturation, Errors) metrics patterns for Kubernetes
environments.

Key Metrics:
- Trap health metrics (classification counts, failure rates)
- Steam loss metrics (energy loss kW, CO2 emissions)
- Calculation latency histograms (per calculator type)
- Error counters (by type and component)
- Fleet-level aggregations

Standards Compliance:
- OpenMetrics specification
- Prometheus best practices for naming and labeling
- Kubernetes observability patterns
- GreenLang Global AI Standards v2.0

Example:
    >>> from monitoring.metrics_exporter import PrometheusMetricsExporter
    >>> exporter = PrometheusMetricsExporter()
    >>> exporter.initialize("1.0.0", "production")
    >>>
    >>> with exporter.measure_diagnosis("thermodynamic", "multimodal") as timer:
    ...     result = classifier.classify(input_data)
    >>>
    >>> exporter.record_steam_loss(trap_id="ST-001", loss_kw=8.5, co2_kg=4.2)

Author: GL-BackendDeveloper
Date: December 2025
Version: 2.0.0
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    REGISTRY,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    generate_latest,
    start_http_server,
)

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC DEFINITIONS - Following Prometheus Naming Conventions
# =============================================================================

# -----------------------------------------------------------------------------
# DIAGNOSIS METRICS
# -----------------------------------------------------------------------------

# Histogram: Diagnosis latency (primary observability metric)
DIAGNOSIS_LATENCY_HISTOGRAM = Histogram(
    name="trapcatcher_diagnosis_latency_seconds",
    documentation="Time taken to diagnose a steam trap in seconds",
    labelnames=["trap_type", "modality", "result"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0),
)

# Counter: Total traps analyzed
TRAPS_ANALYZED_COUNTER = Counter(
    name="trapcatcher_traps_analyzed_total",
    documentation="Total number of steam traps analyzed",
    labelnames=["trap_type", "condition", "facility"],
)

# Gauge: Classification accuracy rate (updated periodically)
ACCURACY_RATE_GAUGE = Gauge(
    name="trapcatcher_accuracy_rate",
    documentation="Current classification accuracy rate (0-1 scale)",
    labelnames=["trap_type", "modality"],
)

# -----------------------------------------------------------------------------
# STEAM LOSS METRICS
# -----------------------------------------------------------------------------

# Counter: Energy loss detected (kWh)
ENERGY_LOSS_COUNTER = Counter(
    name="trapcatcher_energy_loss_kwh_total",
    documentation="Cumulative energy loss detected in kWh",
    labelnames=["facility", "severity"],
)

# Gauge: Current energy loss rate (kW)
ENERGY_LOSS_RATE_GAUGE = Gauge(
    name="trapcatcher_energy_loss_kw",
    documentation="Current energy loss rate in kW",
    labelnames=["facility", "area"],
)

# Counter: Steam loss mass (kg)
STEAM_LOSS_COUNTER = Counter(
    name="trapcatcher_steam_loss_kg_total",
    documentation="Cumulative steam loss in kilograms",
    labelnames=["facility", "trap_type"],
)

# Gauge: Current steam loss rate (kg/hr)
STEAM_LOSS_RATE_GAUGE = Gauge(
    name="trapcatcher_steam_loss_kg_hr",
    documentation="Current steam loss rate in kg/hr",
    labelnames=["facility", "trap_id"],
)

# Counter: CO2 emissions detected (kg)
CO2_EMISSIONS_COUNTER = Counter(
    name="trapcatcher_co2_emissions_kg_total",
    documentation="Cumulative CO2 emissions from steam losses in kg",
    labelnames=["facility", "trap_type"],
)

# Gauge: Annual projected CO2 emissions (tons)
CO2_EMISSIONS_PROJECTED_GAUGE = Gauge(
    name="trapcatcher_co2_emissions_tons_year_projected",
    documentation="Projected annual CO2 emissions in metric tons",
    labelnames=["facility"],
)

# Counter: Cost of steam losses (USD)
STEAM_LOSS_COST_COUNTER = Counter(
    name="trapcatcher_steam_loss_cost_usd_total",
    documentation="Cumulative cost of steam losses in USD",
    labelnames=["facility", "severity"],
)

# -----------------------------------------------------------------------------
# TRAP HEALTH METRICS
# -----------------------------------------------------------------------------

# Gauge: Fleet health score (0-100)
FLEET_HEALTH_GAUGE = Gauge(
    name="trapcatcher_fleet_health_score",
    documentation="Overall fleet health score (0-100)",
    labelnames=["facility"],
)

# Counter: Failed traps detected
FAILED_TRAPS_COUNTER = Counter(
    name="trapcatcher_failed_traps_total",
    documentation="Total number of failed steam traps detected",
    labelnames=["trap_type", "failure_mode", "severity"],
)

# Gauge: Active trap count by condition
TRAP_CONDITION_GAUGE = Gauge(
    name="trapcatcher_traps_by_condition",
    documentation="Number of traps in each condition state",
    labelnames=["facility", "condition"],
)

# Gauge: Trap failure rate percentage
TRAP_FAILURE_RATE_GAUGE = Gauge(
    name="trapcatcher_trap_failure_rate_percent",
    documentation="Percentage of traps in failed state",
    labelnames=["facility", "trap_type"],
)

# -----------------------------------------------------------------------------
# CALCULATION LATENCY METRICS
# -----------------------------------------------------------------------------

# Histogram: Energy loss calculation latency
ENERGY_CALC_LATENCY_HISTOGRAM = Histogram(
    name="trapcatcher_energy_calculation_latency_seconds",
    documentation="Time taken to calculate energy loss in seconds",
    labelnames=["calculator_type", "trap_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Histogram: Bounds validation latency
VALIDATION_LATENCY_HISTOGRAM = Histogram(
    name="trapcatcher_validation_latency_seconds",
    documentation="Time taken for bounds validation in seconds",
    labelnames=["validator_type"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05),
)

# Summary: General calculation duration
CALCULATION_DURATION_SUMMARY = Summary(
    name="trapcatcher_calculation_duration_seconds",
    documentation="Summary of calculation durations",
    labelnames=["calculation_type"],
)

# -----------------------------------------------------------------------------
# API METRICS (RED Pattern)
# -----------------------------------------------------------------------------

# Counter: API requests
API_REQUESTS_COUNTER = Counter(
    name="trapcatcher_api_requests_total",
    documentation="Total API requests by endpoint",
    labelnames=["method", "endpoint", "status_code"],
)

# Histogram: API latency
API_LATENCY_HISTOGRAM = Histogram(
    name="trapcatcher_api_latency_seconds",
    documentation="API request latency in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# -----------------------------------------------------------------------------
# ERROR METRICS
# -----------------------------------------------------------------------------

# Counter: Errors by type
ERRORS_COUNTER = Counter(
    name="trapcatcher_errors_total",
    documentation="Total errors by type and component",
    labelnames=["error_type", "component", "severity"],
)

# Counter: Validation failures
VALIDATION_FAILURES_COUNTER = Counter(
    name="trapcatcher_validation_failures_total",
    documentation="Total validation failures by type",
    labelnames=["validator_type", "parameter", "severity"],
)

# -----------------------------------------------------------------------------
# CONNECTION METRICS (USE Pattern)
# -----------------------------------------------------------------------------

# Gauge: Active connections
ACTIVE_CONNECTIONS_GAUGE = Gauge(
    name="trapcatcher_active_connections",
    documentation="Number of active connections by type",
    labelnames=["connection_type", "protocol"],
)

# Gauge: Sensor data freshness
SENSOR_DATA_AGE_GAUGE = Gauge(
    name="trapcatcher_sensor_data_age_seconds",
    documentation="Age of last sensor data in seconds",
    labelnames=["sensor_type", "trap_id"],
)

# -----------------------------------------------------------------------------
# COMPONENT HEALTH
# -----------------------------------------------------------------------------

# Gauge: Component health status
COMPONENT_HEALTH_GAUGE = Gauge(
    name="trapcatcher_component_health",
    documentation="Component health status (1=healthy, 0=unhealthy)",
    labelnames=["component", "instance"],
)

# Summary: Validation latency (legacy - for backwards compatibility)
VALIDATION_LATENCY_SUMMARY = Summary(
    name="trapcatcher_validation_duration_seconds",
    documentation="Time spent on input validation",
    labelnames=["validator_type"],
)

# Info: Agent metadata
AGENT_INFO_METRIC = Info(
    name="trapcatcher_agent",
    documentation="Agent metadata and version information",
)


# =============================================================================
# DATA CLASSES
# =============================================================================

class TrapConditionLabel(str, Enum):
    """Labels for trap condition metrics."""
    NORMAL = "normal"
    LEAKING = "leaking"
    BLOCKED = "blocked"
    BLOWTHROUGH = "blowthrough"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    INTERMITTENT = "intermittent"
    COLD = "cold"
    UNKNOWN = "unknown"


class TrapTypeLabel(str, Enum):
    """Labels for trap type metrics."""
    THERMODYNAMIC = "thermodynamic"
    THERMOSTATIC = "thermostatic"
    MECHANICAL_FLOAT = "mechanical_float"
    MECHANICAL_BUCKET = "mechanical_bucket"
    ORIFICE = "orifice"
    UNKNOWN = "unknown"


class ModalityLabel(str, Enum):
    """Labels for diagnostic modality."""
    ACOUSTIC = "acoustic"
    THERMAL = "thermal"
    CONTEXTUAL = "contextual"
    MULTIMODAL = "multimodal"


class SeverityLabel(str, Enum):
    """Labels for severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CalculatorTypeLabel(str, Enum):
    """Labels for calculator types."""
    ENERGY_LOSS = "energy_loss"
    STEAM_LOSS = "steam_loss"
    ACOUSTIC = "acoustic"
    TEMPERATURE = "temperature"
    ROI = "roi"
    CARBON = "carbon"


@dataclass(frozen=True)
class DiagnosisMetrics:
    """
    Immutable metrics from a single diagnosis.

    Attributes:
        trap_id: Unique trap identifier
        trap_type: Type of steam trap
        condition: Diagnosed condition
        severity: Severity level
        confidence: Classification confidence (0-1)
        energy_loss_kw: Energy loss in kilowatts
        co2_kg: CO2 emissions in kilograms
        diagnosis_duration_seconds: Time taken for diagnosis
        is_failure: Whether trap is in failed state
        failure_mode: Type of failure if applicable
        facility: Facility identifier
        modality: Diagnostic modality used
    """
    trap_id: str
    trap_type: str
    condition: str
    severity: str
    confidence: float
    energy_loss_kw: float
    co2_kg: float
    diagnosis_duration_seconds: float
    is_failure: bool
    failure_mode: Optional[str] = None
    facility: str = "default"
    modality: str = "multimodal"


@dataclass(frozen=True)
class SteamLossMetrics:
    """
    Immutable steam loss metrics for a trap.

    Attributes:
        trap_id: Unique trap identifier
        trap_type: Type of steam trap
        steam_loss_kg_hr: Steam loss rate in kg/hr
        energy_loss_kw: Energy loss in kW
        co2_kg_hr: CO2 emission rate in kg/hr
        annual_cost_usd: Projected annual cost
        facility: Facility identifier
        severity: Loss severity level
    """
    trap_id: str
    trap_type: str
    steam_loss_kg_hr: float
    energy_loss_kw: float
    co2_kg_hr: float
    annual_cost_usd: float
    facility: str = "default"
    severity: str = "medium"


@dataclass
class AccuracyWindow:
    """
    Sliding window for accuracy calculation.

    Attributes:
        correct_predictions: Number of correct predictions
        total_predictions: Total number of predictions
        window_start: Start time of current window
        window_size_seconds: Window size in seconds
    """
    correct_predictions: int = 0
    total_predictions: int = 0
    window_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    window_size_seconds: int = 3600  # 1 hour window

    def add_prediction(self, is_correct: bool) -> None:
        """Add a prediction to the window."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self.window_start).total_seconds()

        # Reset window if expired
        if elapsed > self.window_size_seconds:
            self.correct_predictions = 0
            self.total_predictions = 0
            self.window_start = now

        self.total_predictions += 1
        if is_correct:
            self.correct_predictions += 1

    @property
    def accuracy(self) -> float:
        """Calculate current accuracy rate."""
        if self.total_predictions == 0:
            return 1.0  # Default to 100% if no predictions
        return self.correct_predictions / self.total_predictions


# =============================================================================
# PROMETHEUS METRICS EXPORTER CLASS
# =============================================================================

class PrometheusMetricsExporter:
    """
    Production-grade Prometheus metrics exporter for GL-008 TRAPCATCHER.

    Provides comprehensive observability through:
    - RED metrics (Rate, Errors, Duration) for API monitoring
    - USE metrics (Utilization, Saturation, Errors) for resource monitoring
    - Business metrics (diagnoses, energy loss, fleet health)
    - Steam loss metrics (kg/hr, kW, CO2, cost)
    - Calculation latency histograms (by calculator type)
    - SHAP-compatible explainability metrics

    Thread-safe implementation with optional background HTTP server.

    Example:
        >>> exporter = PrometheusMetricsExporter()
        >>> exporter.initialize("1.0.0", "production")
        >>> exporter.start_http_server(port=9090)
        >>>
        >>> # Record a diagnosis
        >>> with exporter.measure_diagnosis("thermodynamic", "multimodal") as timer:
        ...     result = classifier.classify(input_data)
        >>>
        >>> exporter.record_diagnosis(DiagnosisMetrics(...))
        >>>
        >>> # Record steam loss
        >>> exporter.record_steam_loss(
        ...     trap_id="ST-001",
        ...     steam_loss_kg_hr=5.2,
        ...     energy_loss_kw=8.5,
        ...     co2_kg_hr=4.2
        ... )
        >>>
        >>> # Get metrics for HTTP response
        >>> content = exporter.get_metrics()

    Attributes:
        registry: Prometheus CollectorRegistry
        version: Agent version string
        environment: Deployment environment
        _initialized: Whether exporter is initialized
        _http_server: Background HTTP server instance
        _accuracy_windows: Accuracy tracking per modality
    """

    VERSION = "2.0.0"

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics exporter.

        Args:
            registry: Optional custom registry for testing isolation.
                      Uses default REGISTRY if not provided.
        """
        self.registry = registry or REGISTRY
        self.version: str = "1.0.0"
        self.environment: str = "development"
        self._initialized: bool = False
        self._http_server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._accuracy_windows: Dict[str, AccuracyWindow] = {}
        self._lock = threading.Lock()

        # Track cumulative metrics for fleet aggregation
        self._fleet_metrics: Dict[str, Dict[str, float]] = {}

        logger.debug("PrometheusMetricsExporter created")

    def initialize(
        self,
        version: str,
        environment: str,
        instance_id: Optional[str] = None,
    ) -> None:
        """
        Initialize exporter with agent metadata.

        Sets up the agent info metric with version and environment details.
        Must be called before recording any metrics.

        Args:
            version: Agent version string (e.g., "1.0.0")
            environment: Deployment environment (dev, staging, production)
            instance_id: Optional unique instance identifier
        """
        self.version = version
        self.environment = environment

        AGENT_INFO_METRIC.info({
            "agent_id": "GL-008",
            "agent_name": "TRAPCATCHER",
            "version": version,
            "environment": environment,
            "instance_id": instance_id or os.getenv("HOSTNAME", "default"),
            "exporter_version": self.VERSION,
        })

        self._initialized = True
        logger.info(
            f"PrometheusMetricsExporter v{self.VERSION} initialized: "
            f"version={version}, environment={environment}"
        )

    def start_http_server(
        self,
        port: int = 9090,
        address: str = "0.0.0.0",
    ) -> None:
        """
        Start background HTTP server for /metrics endpoint.

        Creates a dedicated thread to serve Prometheus metrics.
        Thread is daemonized and will terminate with main process.

        Args:
            port: Port to listen on (default: 9090)
            address: Address to bind to (default: 0.0.0.0)

        Raises:
            RuntimeError: If server is already running
        """
        if self._http_server is not None:
            raise RuntimeError("HTTP server is already running")

        try:
            start_http_server(port=port, addr=address, registry=self.registry)
            logger.info(f"Prometheus metrics server started on {address}:{port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def stop_http_server(self) -> None:
        """
        Stop the background HTTP server.

        Gracefully shuts down the metrics server if running.
        Safe to call even if server is not running.
        """
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
            logger.info("Prometheus metrics server stopped")

    # =========================================================================
    # DIAGNOSIS METRICS
    # =========================================================================

    @contextmanager
    def measure_diagnosis(
        self,
        trap_type: str,
        modality: str = "multimodal",
    ) -> Generator[None, None, None]:
        """
        Context manager to measure diagnosis duration.

        Automatically records diagnosis latency to histogram.
        Use with 'with' statement for accurate timing.

        Args:
            trap_type: Type of trap being diagnosed
            modality: Diagnostic modality used

        Yields:
            None - timing is automatic

        Example:
            >>> with exporter.measure_diagnosis("thermodynamic", "acoustic"):
            ...     result = classifier.classify(input_data)
        """
        start_time = time.perf_counter()
        result = "success"
        try:
            yield
        except Exception:
            result = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            DIAGNOSIS_LATENCY_HISTOGRAM.labels(
                trap_type=trap_type,
                modality=modality,
                result=result,
            ).observe(duration)

            logger.debug(
                f"Diagnosis completed: trap_type={trap_type}, "
                f"modality={modality}, duration={duration:.3f}s"
            )

    def record_diagnosis(self, metrics: DiagnosisMetrics) -> None:
        """
        Record metrics from a completed diagnosis.

        Updates all relevant counters and gauges for the diagnosis.
        Thread-safe operation.

        Args:
            metrics: DiagnosisMetrics from completed diagnosis
        """
        with self._lock:
            # Increment traps analyzed counter
            TRAPS_ANALYZED_COUNTER.labels(
                trap_type=metrics.trap_type,
                condition=metrics.condition,
                facility=metrics.facility,
            ).inc()

            # Record energy loss
            if metrics.energy_loss_kw > 0:
                ENERGY_LOSS_COUNTER.labels(
                    facility=metrics.facility,
                    severity=metrics.severity,
                ).inc(metrics.energy_loss_kw)

                ENERGY_LOSS_RATE_GAUGE.labels(
                    facility=metrics.facility,
                    area="main",
                ).set(metrics.energy_loss_kw)

            # Record CO2 emissions
            if metrics.co2_kg > 0:
                CO2_EMISSIONS_COUNTER.labels(
                    facility=metrics.facility,
                    trap_type=metrics.trap_type,
                ).inc(metrics.co2_kg)

            # Record failures
            if metrics.is_failure and metrics.failure_mode:
                FAILED_TRAPS_COUNTER.labels(
                    trap_type=metrics.trap_type,
                    failure_mode=metrics.failure_mode,
                    severity=metrics.severity,
                ).inc()

            logger.debug(
                f"Recorded diagnosis metrics: trap={metrics.trap_id}, "
                f"condition={metrics.condition}, energy_loss={metrics.energy_loss_kw}kW"
            )

    # =========================================================================
    # STEAM LOSS METRICS
    # =========================================================================

    def record_steam_loss(
        self,
        trap_id: str,
        steam_loss_kg_hr: float,
        energy_loss_kw: float,
        co2_kg_hr: float,
        annual_cost_usd: float = 0.0,
        trap_type: str = "unknown",
        facility: str = "default",
        severity: str = "medium",
    ) -> None:
        """
        Record steam loss metrics for a trap.

        Updates all steam loss related counters and gauges.

        Args:
            trap_id: Unique trap identifier
            steam_loss_kg_hr: Steam loss rate in kg/hr
            energy_loss_kw: Energy loss in kW
            co2_kg_hr: CO2 emission rate in kg/hr
            annual_cost_usd: Projected annual cost
            trap_type: Type of steam trap
            facility: Facility identifier
            severity: Loss severity level
        """
        with self._lock:
            # Steam loss rate gauge
            STEAM_LOSS_RATE_GAUGE.labels(
                facility=facility,
                trap_id=trap_id,
            ).set(steam_loss_kg_hr)

            # Steam loss counter (hourly increment)
            STEAM_LOSS_COUNTER.labels(
                facility=facility,
                trap_type=trap_type,
            ).inc(steam_loss_kg_hr)

            # Energy loss rate
            ENERGY_LOSS_RATE_GAUGE.labels(
                facility=facility,
                area=trap_id,
            ).set(energy_loss_kw)

            # CO2 emissions
            CO2_EMISSIONS_COUNTER.labels(
                facility=facility,
                trap_type=trap_type,
            ).inc(co2_kg_hr)

            # Cost tracking
            if annual_cost_usd > 0:
                # Convert annual to hourly
                hourly_cost = annual_cost_usd / 8760
                STEAM_LOSS_COST_COUNTER.labels(
                    facility=facility,
                    severity=severity,
                ).inc(hourly_cost)

            logger.debug(
                f"Recorded steam loss: trap={trap_id}, "
                f"loss={steam_loss_kg_hr}kg/hr, energy={energy_loss_kw}kW"
            )

    def record_steam_loss_metrics(self, metrics: SteamLossMetrics) -> None:
        """
        Record steam loss metrics from dataclass.

        Convenience wrapper for record_steam_loss.

        Args:
            metrics: SteamLossMetrics instance
        """
        self.record_steam_loss(
            trap_id=metrics.trap_id,
            steam_loss_kg_hr=metrics.steam_loss_kg_hr,
            energy_loss_kw=metrics.energy_loss_kw,
            co2_kg_hr=metrics.co2_kg_hr,
            annual_cost_usd=metrics.annual_cost_usd,
            trap_type=metrics.trap_type,
            facility=metrics.facility,
            severity=metrics.severity,
        )

    # =========================================================================
    # CALCULATION LATENCY METRICS
    # =========================================================================

    @contextmanager
    def measure_calculation(
        self,
        calculator_type: str,
        trap_type: str = "unknown",
    ) -> Generator[None, None, None]:
        """
        Context manager to measure calculation duration.

        Records latency to appropriate histogram based on calculator type.

        Args:
            calculator_type: Type of calculator (energy_loss, steam_loss, etc.)
            trap_type: Type of trap being calculated

        Yields:
            None - timing is automatic

        Example:
            >>> with exporter.measure_calculation("energy_loss", "thermodynamic"):
            ...     result = calculator.calculate(input_data)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time

            ENERGY_CALC_LATENCY_HISTOGRAM.labels(
                calculator_type=calculator_type,
                trap_type=trap_type,
            ).observe(duration)

            CALCULATION_DURATION_SUMMARY.labels(
                calculation_type=calculator_type,
            ).observe(duration)

            logger.debug(
                f"Calculation completed: type={calculator_type}, "
                f"trap_type={trap_type}, duration={duration:.4f}s"
            )

    @contextmanager
    def measure_validation(
        self,
        validator_type: str,
    ) -> Generator[None, None, None]:
        """
        Context manager to measure validation duration.

        Args:
            validator_type: Type of validator (bounds, schema, business)

        Yields:
            None - timing is automatic
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time

            VALIDATION_LATENCY_HISTOGRAM.labels(
                validator_type=validator_type,
            ).observe(duration)

            VALIDATION_LATENCY_SUMMARY.labels(
                validator_type=validator_type,
            ).observe(duration)

    def record_calculation_latency(
        self,
        calculator_type: str,
        duration_seconds: float,
        trap_type: str = "unknown",
    ) -> None:
        """
        Record calculation latency manually.

        Use when context manager is not suitable.

        Args:
            calculator_type: Type of calculator
            duration_seconds: Duration in seconds
            trap_type: Type of trap
        """
        ENERGY_CALC_LATENCY_HISTOGRAM.labels(
            calculator_type=calculator_type,
            trap_type=trap_type,
        ).observe(duration_seconds)

        CALCULATION_DURATION_SUMMARY.labels(
            calculation_type=calculator_type,
        ).observe(duration_seconds)

    # =========================================================================
    # TRAP HEALTH METRICS
    # =========================================================================

    def record_accuracy(
        self,
        trap_type: str,
        modality: str,
        is_correct: bool,
    ) -> None:
        """
        Record prediction accuracy for accuracy rate gauge.

        Maintains sliding window accuracy per modality.
        Updates accuracy rate gauge automatically.

        Args:
            trap_type: Type of trap
            modality: Diagnostic modality
            is_correct: Whether prediction was correct
        """
        key = f"{trap_type}_{modality}"

        with self._lock:
            if key not in self._accuracy_windows:
                self._accuracy_windows[key] = AccuracyWindow()

            self._accuracy_windows[key].add_prediction(is_correct)
            accuracy = self._accuracy_windows[key].accuracy

            ACCURACY_RATE_GAUGE.labels(
                trap_type=trap_type,
                modality=modality,
            ).set(accuracy)

    def update_fleet_health(
        self,
        facility: str,
        health_score: float,
        total_energy_loss_kw: float,
        total_co2_kg: float,
        trap_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Update fleet-level health metrics.

        Updates all fleet-wide aggregate gauges.

        Args:
            facility: Facility identifier
            health_score: Fleet health score (0-100)
            total_energy_loss_kw: Total energy loss across fleet
            total_co2_kg: Total CO2 emissions
            trap_counts: Optional dict of condition -> count
        """
        FLEET_HEALTH_GAUGE.labels(facility=facility).set(health_score)

        ENERGY_LOSS_RATE_GAUGE.labels(
            facility=facility,
            area="fleet_total",
        ).set(total_energy_loss_kw)

        # Calculate projected annual CO2
        annual_co2_tons = (total_co2_kg * 8760) / 1000
        CO2_EMISSIONS_PROJECTED_GAUGE.labels(
            facility=facility
        ).set(annual_co2_tons)

        # Update trap condition counts if provided
        if trap_counts:
            for condition, count in trap_counts.items():
                TRAP_CONDITION_GAUGE.labels(
                    facility=facility,
                    condition=condition,
                ).set(count)

        # Store for aggregation
        with self._lock:
            self._fleet_metrics[facility] = {
                "health_score": health_score,
                "energy_loss_kw": total_energy_loss_kw,
                "co2_kg": total_co2_kg,
            }

        logger.debug(
            f"Updated fleet health: facility={facility}, "
            f"score={health_score:.1f}, energy_loss={total_energy_loss_kw:.1f}kW"
        )

    def update_trap_failure_rate(
        self,
        facility: str,
        trap_type: str,
        failure_rate_percent: float,
    ) -> None:
        """
        Update trap failure rate percentage.

        Args:
            facility: Facility identifier
            trap_type: Type of trap
            failure_rate_percent: Failure rate as percentage
        """
        TRAP_FAILURE_RATE_GAUGE.labels(
            facility=facility,
            trap_type=trap_type,
        ).set(failure_rate_percent)

    # =========================================================================
    # API METRICS (RED Pattern)
    # =========================================================================

    @contextmanager
    def measure_api_request(
        self,
        method: str,
        endpoint: str,
    ) -> Generator[None, None, None]:
        """
        Context manager to measure API request duration.

        Records both latency histogram and request counter.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path

        Yields:
            None - timing is automatic
        """
        start_time = time.perf_counter()
        status_code = "200"
        try:
            yield
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start_time

            API_LATENCY_HISTOGRAM.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            API_REQUESTS_COUNTER.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """
        Record API request metrics manually.

        Use when context manager is not suitable.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration_seconds: Request duration in seconds
        """
        API_REQUESTS_COUNTER.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        API_LATENCY_HISTOGRAM.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration_seconds)

    # =========================================================================
    # ERROR METRICS
    # =========================================================================

    def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = "medium",
    ) -> None:
        """
        Record an error occurrence.

        Increments error counter with appropriate labels.

        Args:
            error_type: Type of error (validation, calculation, integration)
            component: Component where error occurred
            severity: Error severity level
        """
        ERRORS_COUNTER.labels(
            error_type=error_type,
            component=component,
            severity=severity,
        ).inc()

        logger.warning(
            f"Error recorded: type={error_type}, "
            f"component={component}, severity={severity}"
        )

    def record_validation_failure(
        self,
        validator_type: str,
        parameter: str,
        severity: str = "error",
    ) -> None:
        """
        Record a validation failure.

        Args:
            validator_type: Type of validator that failed
            parameter: Parameter that failed validation
            severity: Failure severity
        """
        VALIDATION_FAILURES_COUNTER.labels(
            validator_type=validator_type,
            parameter=parameter,
            severity=severity,
        ).inc()

    # =========================================================================
    # CONNECTION METRICS (USE Pattern)
    # =========================================================================

    def set_active_connections(
        self,
        connection_type: str,
        protocol: str,
        count: int,
    ) -> None:
        """
        Set number of active connections.

        Args:
            connection_type: Type of connection (sensor, database, cache)
            protocol: Protocol used (opcua, modbus, mqtt, http)
            count: Number of active connections
        """
        ACTIVE_CONNECTIONS_GAUGE.labels(
            connection_type=connection_type,
            protocol=protocol,
        ).set(count)

    def update_sensor_freshness(
        self,
        sensor_type: str,
        trap_id: str,
        age_seconds: float,
    ) -> None:
        """
        Update sensor data age metric.

        Args:
            sensor_type: Type of sensor (acoustic, thermal)
            trap_id: Trap identifier
            age_seconds: Age of last reading in seconds
        """
        SENSOR_DATA_AGE_GAUGE.labels(
            sensor_type=sensor_type,
            trap_id=trap_id,
        ).set(age_seconds)

    # =========================================================================
    # COMPONENT HEALTH
    # =========================================================================

    def set_component_health(
        self,
        component: str,
        instance: str,
        is_healthy: bool,
    ) -> None:
        """
        Set component health status.

        Args:
            component: Component name (classifier, calculator, explainer)
            instance: Instance identifier
            is_healthy: Whether component is healthy
        """
        COMPONENT_HEALTH_GAUGE.labels(
            component=component,
            instance=instance,
        ).set(1 if is_healthy else 0)

    # =========================================================================
    # METRICS OUTPUT
    # =========================================================================

    def get_metrics(self) -> bytes:
        """
        Generate Prometheus metrics output.

        Returns metrics in OpenMetrics format suitable for
        HTTP response to /metrics endpoint.

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

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get human-readable metrics summary.

        Returns dictionary with current metric values
        for logging and debugging purposes.

        Returns:
            Dictionary with metric summaries
        """
        with self._lock:
            return {
                "initialized": self._initialized,
                "version": self.version,
                "exporter_version": self.VERSION,
                "environment": self.environment,
                "accuracy_windows": {
                    k: {"accuracy": v.accuracy, "total": v.total_predictions}
                    for k, v in self._accuracy_windows.items()
                },
                "fleet_metrics": self._fleet_metrics.copy(),
            }


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

def create_metrics_endpoint(exporter: PrometheusMetricsExporter):
    """
    Create FastAPI endpoint for Prometheus metrics.

    Returns a route handler that can be added to FastAPI app.

    Args:
        exporter: PrometheusMetricsExporter instance

    Returns:
        Async function for /metrics endpoint

    Example:
        >>> app = FastAPI()
        >>> exporter = PrometheusMetricsExporter()
        >>>
        >>> @app.get("/metrics")
        >>> async def metrics():
        ...     return await create_metrics_endpoint(exporter)()
    """
    async def metrics_endpoint():
        from fastapi.responses import Response
        return Response(
            content=exporter.get_metrics(),
            media_type=exporter.get_content_type(),
        )
    return metrics_endpoint


def create_metrics_middleware(exporter: PrometheusMetricsExporter):
    """
    Create FastAPI middleware for automatic request metrics.

    Wraps all requests with timing and counting metrics.

    Args:
        exporter: PrometheusMetricsExporter instance

    Returns:
        Middleware function for FastAPI

    Example:
        >>> app = FastAPI()
        >>> exporter = PrometheusMetricsExporter()
        >>> app.middleware("http")(create_metrics_middleware(exporter))
    """
    async def metrics_middleware(request, call_next):
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            exporter.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration_seconds=duration,
            )

        return response

    return metrics_middleware


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Global exporter instance for convenience
_global_exporter: Optional[PrometheusMetricsExporter] = None


def get_metrics_exporter() -> PrometheusMetricsExporter:
    """
    Get or create global metrics exporter instance.

    Thread-safe singleton pattern.

    Returns:
        PrometheusMetricsExporter instance
    """
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = PrometheusMetricsExporter()
    return _global_exporter


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main exporter
    "PrometheusMetricsExporter",
    "get_metrics_exporter",

    # Data classes
    "DiagnosisMetrics",
    "SteamLossMetrics",
    "AccuracyWindow",

    # Enums
    "TrapConditionLabel",
    "TrapTypeLabel",
    "ModalityLabel",
    "SeverityLabel",
    "CalculatorTypeLabel",

    # FastAPI integration
    "create_metrics_endpoint",
    "create_metrics_middleware",

    # Metrics (for direct access if needed)
    "DIAGNOSIS_LATENCY_HISTOGRAM",
    "TRAPS_ANALYZED_COUNTER",
    "ACCURACY_RATE_GAUGE",
    "ENERGY_LOSS_COUNTER",
    "STEAM_LOSS_COUNTER",
    "STEAM_LOSS_RATE_GAUGE",
    "CO2_EMISSIONS_COUNTER",
    "FLEET_HEALTH_GAUGE",
    "FAILED_TRAPS_COUNTER",
    "API_REQUESTS_COUNTER",
    "API_LATENCY_HISTOGRAM",
    "ENERGY_CALC_LATENCY_HISTOGRAM",
    "VALIDATION_LATENCY_HISTOGRAM",
    "ERRORS_COUNTER",
]
