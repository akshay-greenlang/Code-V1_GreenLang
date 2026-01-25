# -*- coding: utf-8 -*-
"""
Prometheus Metrics Module for GL-015 INSULSCAN.

This module provides comprehensive Prometheus metrics collection for:
- Request metrics (latency, count, active requests)
- Inspection metrics (inspections, thermal images, hotspots, anomalies)
- Heat loss metrics (total heat loss, energy cost, carbon emissions)
- Degradation metrics (equipment condition, degradation rate, repairs)
- Integration metrics (camera connectivity, CMMS work orders)

Implements production-grade metric registration, label management,
histogram bucket configuration, and metric export endpoints.

Example:
    >>> from monitoring.metrics import MetricsCollector, get_metrics_collector
    >>> collector = get_metrics_collector()
    >>> collector.record_request_latency("/api/v1/inspect", "POST", 0.125)
    >>> collector.increment_inspections_completed("FACILITY-001", "thermal_scan")
    >>> collector.set_total_heat_loss("FACILITY-001", 15000.0)

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
import threading
import time
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC TYPE ENUMERATION
# =============================================================================

class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# =============================================================================
# HISTOGRAM BUCKET CONFIGURATIONS
# =============================================================================

# Request latency buckets (in seconds)
REQUEST_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, float("inf")
)

# Image processing duration buckets (in seconds)
IMAGE_PROCESSING_DURATION_BUCKETS = (
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
    300.0, 600.0, float("inf")
)

# Heat loss buckets (in watts per meter)
HEAT_LOSS_BUCKETS = (
    10, 25, 50, 100, 200, 500, 1000, 2500, 5000, 10000,
    25000, 50000, 100000, float("inf")
)

# Temperature buckets (in Celsius)
TEMPERATURE_BUCKETS = (
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    125, 150, 200, 300, float("inf")
)

# Connector latency buckets (in seconds)
CONNECTOR_LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    30.0, 60.0, 120.0, float("inf")
)

# Request/response size buckets (in bytes)
SIZE_BUCKETS = (
    100, 500, 1000, 5000, 10000, 50000, 100000, 500000,
    1000000, 5000000, 10000000, float("inf")
)


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class MetricDefinition:
    """
    Definition of a Prometheus metric.

    Attributes:
        name: Metric name (must follow Prometheus naming conventions)
        metric_type: Type of metric (counter, gauge, histogram, summary)
        description: Human-readable description
        labels: List of label names
        buckets: Histogram bucket boundaries (for histogram type)
        unit: Unit of measurement (optional)
    """
    name: str
    metric_type: MetricType
    description: str
    labels: tuple = field(default_factory=tuple)
    buckets: Optional[tuple] = None
    unit: str = ""

    def validate(self) -> bool:
        """Validate metric definition."""
        # Validate name format (Prometheus naming convention)
        if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$', self.name):
            raise ValueError(f"Invalid metric name: {self.name}")

        # Validate labels
        for label in self.labels:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
                raise ValueError(f"Invalid label name: {label}")

        # Validate buckets for histogram
        if self.metric_type == MetricType.HISTOGRAM:
            if self.buckets is None:
                raise ValueError(f"Histogram {self.name} requires buckets")
            if not all(isinstance(b, (int, float)) for b in self.buckets):
                raise ValueError(f"Buckets must be numeric for {self.name}")

        return True


# -----------------------------------------------------------------------------
# REQUEST METRICS
# -----------------------------------------------------------------------------

REQUEST_LATENCY_HISTOGRAM = MetricDefinition(
    name="gl015_request_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Request latency in seconds by endpoint and method",
    labels=("endpoint", "method"),
    buckets=REQUEST_LATENCY_BUCKETS,
    unit="seconds"
)

REQUEST_COUNT_TOTAL = MetricDefinition(
    name="gl015_request_count_total",
    metric_type=MetricType.COUNTER,
    description="Total number of requests by endpoint and status",
    labels=("endpoint", "status", "method"),
    unit="requests"
)

ACTIVE_REQUESTS_GAUGE = MetricDefinition(
    name="gl015_active_requests",
    metric_type=MetricType.GAUGE,
    description="Number of currently active requests",
    labels=("endpoint",),
    unit="requests"
)

REQUEST_SIZE_HISTOGRAM = MetricDefinition(
    name="gl015_request_size_bytes",
    metric_type=MetricType.HISTOGRAM,
    description="Request body size in bytes",
    labels=("endpoint", "method"),
    buckets=SIZE_BUCKETS,
    unit="bytes"
)

RESPONSE_SIZE_HISTOGRAM = MetricDefinition(
    name="gl015_response_size_bytes",
    metric_type=MetricType.HISTOGRAM,
    description="Response body size in bytes",
    labels=("endpoint", "method"),
    buckets=SIZE_BUCKETS,
    unit="bytes"
)


# -----------------------------------------------------------------------------
# INSPECTION METRICS
# -----------------------------------------------------------------------------

INSPECTIONS_COMPLETED_TOTAL = MetricDefinition(
    name="gl015_inspections_completed_total",
    metric_type=MetricType.COUNTER,
    description="Total number of inspections completed by facility and type",
    labels=("facility_id", "inspection_type", "status"),
    unit="inspections"
)

THERMAL_IMAGES_PROCESSED_TOTAL = MetricDefinition(
    name="gl015_thermal_images_processed_total",
    metric_type=MetricType.COUNTER,
    description="Total number of thermal images processed",
    labels=("facility_id", "camera_type", "resolution"),
    unit="images"
)

IMAGE_PROCESSING_DURATION_HISTOGRAM = MetricDefinition(
    name="gl015_image_processing_duration_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Image processing duration in seconds by type",
    labels=("processing_type", "image_format"),
    buckets=IMAGE_PROCESSING_DURATION_BUCKETS,
    unit="seconds"
)

HOTSPOTS_DETECTED_TOTAL = MetricDefinition(
    name="gl015_hotspots_detected_total",
    metric_type=MetricType.COUNTER,
    description="Total number of thermal hotspots detected by severity",
    labels=("facility_id", "severity", "zone"),
    unit="hotspots"
)

ANOMALIES_CLASSIFIED_TOTAL = MetricDefinition(
    name="gl015_anomalies_classified_total",
    metric_type=MetricType.COUNTER,
    description="Total number of anomalies classified by type",
    labels=("facility_id", "anomaly_type", "confidence_level"),
    unit="anomalies"
)

INSPECTION_COVERAGE_GAUGE = MetricDefinition(
    name="gl015_inspection_coverage_percent",
    metric_type=MetricType.GAUGE,
    description="Percentage of facility equipment inspected",
    labels=("facility_id", "equipment_type"),
    unit="percent"
)

SURFACE_AREA_INSPECTED_GAUGE = MetricDefinition(
    name="gl015_surface_area_inspected_m2",
    metric_type=MetricType.GAUGE,
    description="Total surface area inspected in square meters",
    labels=("facility_id", "zone"),
    unit="m2"
)


# -----------------------------------------------------------------------------
# HEAT LOSS METRICS
# -----------------------------------------------------------------------------

TOTAL_HEAT_LOSS_WATTS_GAUGE = MetricDefinition(
    name="gl015_total_heat_loss_watts",
    metric_type=MetricType.GAUGE,
    description="Total heat loss in watts by facility and zone",
    labels=("facility_id", "zone", "equipment_type"),
    unit="W"
)

HEAT_LOSS_PER_METER_GAUGE = MetricDefinition(
    name="gl015_heat_loss_per_meter_wm",
    metric_type=MetricType.GAUGE,
    description="Heat loss per meter of insulation in W/m",
    labels=("facility_id", "insulation_type", "condition"),
    unit="W/m"
)

ENERGY_COST_DOLLARS_GAUGE = MetricDefinition(
    name="gl015_energy_cost_dollars",
    metric_type=MetricType.GAUGE,
    description="Energy cost due to heat loss in dollars",
    labels=("facility_id", "period", "tariff_type"),
    unit="USD"
)

CARBON_EMISSIONS_KG_GAUGE = MetricDefinition(
    name="gl015_carbon_emissions_kg",
    metric_type=MetricType.GAUGE,
    description="Carbon emissions due to heat loss in kg CO2e",
    labels=("facility_id", "scope", "fuel_type"),
    unit="kg"
)

HEAT_LOSS_HISTOGRAM = MetricDefinition(
    name="gl015_heat_loss_distribution_watts",
    metric_type=MetricType.HISTOGRAM,
    description="Distribution of heat loss measurements",
    labels=("facility_id", "equipment_type"),
    buckets=HEAT_LOSS_BUCKETS,
    unit="W"
)

AMBIENT_TEMPERATURE_GAUGE = MetricDefinition(
    name="gl015_ambient_temperature_celsius",
    metric_type=MetricType.GAUGE,
    description="Ambient temperature during inspection",
    labels=("facility_id", "zone"),
    unit="celsius"
)

SURFACE_TEMPERATURE_HISTOGRAM = MetricDefinition(
    name="gl015_surface_temperature_distribution_celsius",
    metric_type=MetricType.HISTOGRAM,
    description="Distribution of surface temperature measurements",
    labels=("facility_id", "equipment_type"),
    buckets=TEMPERATURE_BUCKETS,
    unit="celsius"
)


# -----------------------------------------------------------------------------
# DEGRADATION METRICS
# -----------------------------------------------------------------------------

EQUIPMENT_BY_CONDITION_GAUGE = MetricDefinition(
    name="gl015_equipment_by_condition",
    metric_type=MetricType.GAUGE,
    description="Count of equipment by condition severity",
    labels=("facility_id", "condition_severity", "equipment_type"),
    unit="count"
)

AVERAGE_DEGRADATION_RATE_GAUGE = MetricDefinition(
    name="gl015_average_degradation_rate_percent_per_year",
    metric_type=MetricType.GAUGE,
    description="Average insulation degradation rate in percent per year",
    labels=("facility_id", "insulation_type", "age_bracket"),
    unit="percent/year"
)

INSULATION_THICKNESS_GAUGE = MetricDefinition(
    name="gl015_insulation_thickness_mm",
    metric_type=MetricType.GAUGE,
    description="Measured insulation thickness in millimeters",
    labels=("facility_id", "equipment_id", "insulation_type"),
    unit="mm"
)

INSULATION_EFFICIENCY_GAUGE = MetricDefinition(
    name="gl015_insulation_efficiency_percent",
    metric_type=MetricType.GAUGE,
    description="Insulation efficiency as percentage of design specification",
    labels=("facility_id", "equipment_id", "insulation_type"),
    unit="percent"
)

MOISTURE_CONTENT_GAUGE = MetricDefinition(
    name="gl015_moisture_content_percent",
    metric_type=MetricType.GAUGE,
    description="Measured moisture content in insulation",
    labels=("facility_id", "equipment_id", "zone"),
    unit="percent"
)

REPAIRS_PRIORITIZED_TOTAL = MetricDefinition(
    name="gl015_repairs_prioritized_total",
    metric_type=MetricType.COUNTER,
    description="Total number of repairs prioritized by urgency",
    labels=("facility_id", "urgency", "repair_type"),
    unit="repairs"
)

REPAIR_COST_ESTIMATE_GAUGE = MetricDefinition(
    name="gl015_repair_cost_estimate_dollars",
    metric_type=MetricType.GAUGE,
    description="Estimated repair cost in dollars",
    labels=("facility_id", "repair_type", "urgency"),
    unit="USD"
)

TIME_TO_FAILURE_DAYS_GAUGE = MetricDefinition(
    name="gl015_time_to_failure_days",
    metric_type=MetricType.GAUGE,
    description="Estimated days until insulation failure",
    labels=("facility_id", "equipment_id", "failure_mode"),
    unit="days"
)


# -----------------------------------------------------------------------------
# INTEGRATION METRICS
# -----------------------------------------------------------------------------

CAMERA_CONNECTION_STATUS_GAUGE = MetricDefinition(
    name="gl015_camera_connection_status",
    metric_type=MetricType.GAUGE,
    description="Thermal camera connection status (1=connected, 0=disconnected)",
    labels=("camera_id", "camera_type", "facility_id"),
    unit="status"
)

CAMERA_LATENCY_HISTOGRAM = MetricDefinition(
    name="gl015_camera_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Thermal camera operation latency in seconds",
    labels=("camera_id", "operation"),
    buckets=CONNECTOR_LATENCY_BUCKETS,
    unit="seconds"
)

CMMS_WORK_ORDERS_CREATED_TOTAL = MetricDefinition(
    name="gl015_cmms_work_orders_created_total",
    metric_type=MetricType.COUNTER,
    description="Total work orders created in CMMS",
    labels=("facility_id", "work_order_type", "priority"),
    unit="work_orders"
)

CMMS_LATENCY_HISTOGRAM = MetricDefinition(
    name="gl015_cmms_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="CMMS operation latency in seconds",
    labels=("cmms_type", "operation"),
    buckets=CONNECTOR_LATENCY_BUCKETS,
    unit="seconds"
)

CONNECTOR_ERRORS_TOTAL = MetricDefinition(
    name="gl015_connector_errors_total",
    metric_type=MetricType.COUNTER,
    description="Total connector errors by system and error type",
    labels=("system", "error_type", "operation"),
    unit="errors"
)

DATA_POINTS_PROCESSED_TOTAL = MetricDefinition(
    name="gl015_data_points_processed_total",
    metric_type=MetricType.COUNTER,
    description="Total data points processed from external systems",
    labels=("source", "data_type"),
    unit="points"
)


# -----------------------------------------------------------------------------
# CALCULATOR METRICS
# -----------------------------------------------------------------------------

CALCULATION_DURATION_HISTOGRAM = MetricDefinition(
    name="gl015_calculation_duration_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Calculation duration in seconds by calculator type",
    labels=("calculator_type", "calculation_mode"),
    buckets=IMAGE_PROCESSING_DURATION_BUCKETS,
    unit="seconds"
)

CALCULATION_COUNT_TOTAL = MetricDefinition(
    name="gl015_calculation_count_total",
    metric_type=MetricType.COUNTER,
    description="Total number of calculations by type and status",
    labels=("calculator_type", "status"),
    unit="calculations"
)

CACHE_HIT_RATIO_GAUGE = MetricDefinition(
    name="gl015_cache_hit_ratio",
    metric_type=MetricType.GAUGE,
    description="Cache hit ratio for calculation results (0-1)",
    labels=("cache_type",),
    unit="ratio"
)


# =============================================================================
# METRIC VALUE STORAGE
# =============================================================================

@dataclass
class MetricValue:
    """
    Storage for a single metric value with labels.

    Attributes:
        value: Current metric value
        labels: Dictionary of label values
        timestamp: Last update timestamp
    """
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HistogramValue:
    """
    Storage for histogram metric values.

    Attributes:
        buckets: Dictionary mapping bucket boundary to count
        sum_value: Sum of all observed values
        count: Total number of observations
        labels: Dictionary of label values
        timestamp: Last update timestamp
    """
    buckets: Dict[float, int]
    sum_value: float
    count: int
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# METRICS COLLECTOR CLASS
# =============================================================================

class MetricsCollector:
    """
    Production-grade Prometheus metrics collector for GL-015 INSULSCAN.

    Provides thread-safe metric registration, recording, and export
    for all insulation scanning and thermal imaging metrics.

    Attributes:
        namespace: Metric namespace prefix
        subsystem: Metric subsystem identifier
        registry: Dictionary of registered metrics

    Example:
        >>> collector = MetricsCollector()
        >>> collector.register_metric(REQUEST_LATENCY_HISTOGRAM)
        >>> collector.record_histogram("gl015_request_latency_seconds",
        ...                           {"endpoint": "/api/v1/inspect", "method": "POST"},
        ...                           0.125)
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "MetricsCollector":
        """Singleton pattern for global metrics collector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        namespace: str = "gl015",
        subsystem: str = "insulscan",
        auto_register: bool = True
    ):
        """
        Initialize the MetricsCollector.

        Args:
            namespace: Metric namespace prefix
            subsystem: Metric subsystem identifier
            auto_register: Whether to auto-register default metrics
        """
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.namespace = namespace
        self.subsystem = subsystem

        # Metric registries (thread-safe)
        self._metrics_lock = threading.RLock()
        self._definitions: Dict[str, MetricDefinition] = {}
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, HistogramValue]] = {}

        # Cache statistics
        self._cache_hits: Dict[str, int] = {}
        self._cache_misses: Dict[str, int] = {}

        # Start time for uptime calculation
        self._start_time = datetime.now(timezone.utc)

        # Auto-register default metrics
        if auto_register:
            self._register_default_metrics()

        self._initialized = True
        logger.info(f"MetricsCollector initialized: namespace={namespace}")

    def _register_default_metrics(self) -> None:
        """Register all default GL-015 metrics."""
        default_metrics = [
            # Request metrics
            REQUEST_LATENCY_HISTOGRAM,
            REQUEST_COUNT_TOTAL,
            ACTIVE_REQUESTS_GAUGE,
            REQUEST_SIZE_HISTOGRAM,
            RESPONSE_SIZE_HISTOGRAM,
            # Inspection metrics
            INSPECTIONS_COMPLETED_TOTAL,
            THERMAL_IMAGES_PROCESSED_TOTAL,
            IMAGE_PROCESSING_DURATION_HISTOGRAM,
            HOTSPOTS_DETECTED_TOTAL,
            ANOMALIES_CLASSIFIED_TOTAL,
            INSPECTION_COVERAGE_GAUGE,
            SURFACE_AREA_INSPECTED_GAUGE,
            # Heat loss metrics
            TOTAL_HEAT_LOSS_WATTS_GAUGE,
            HEAT_LOSS_PER_METER_GAUGE,
            ENERGY_COST_DOLLARS_GAUGE,
            CARBON_EMISSIONS_KG_GAUGE,
            HEAT_LOSS_HISTOGRAM,
            AMBIENT_TEMPERATURE_GAUGE,
            SURFACE_TEMPERATURE_HISTOGRAM,
            # Degradation metrics
            EQUIPMENT_BY_CONDITION_GAUGE,
            AVERAGE_DEGRADATION_RATE_GAUGE,
            INSULATION_THICKNESS_GAUGE,
            INSULATION_EFFICIENCY_GAUGE,
            MOISTURE_CONTENT_GAUGE,
            REPAIRS_PRIORITIZED_TOTAL,
            REPAIR_COST_ESTIMATE_GAUGE,
            TIME_TO_FAILURE_DAYS_GAUGE,
            # Integration metrics
            CAMERA_CONNECTION_STATUS_GAUGE,
            CAMERA_LATENCY_HISTOGRAM,
            CMMS_WORK_ORDERS_CREATED_TOTAL,
            CMMS_LATENCY_HISTOGRAM,
            CONNECTOR_ERRORS_TOTAL,
            DATA_POINTS_PROCESSED_TOTAL,
            # Calculator metrics
            CALCULATION_DURATION_HISTOGRAM,
            CALCULATION_COUNT_TOTAL,
            CACHE_HIT_RATIO_GAUGE,
        ]

        for metric in default_metrics:
            self.register_metric(metric)

        logger.debug(f"Registered {len(default_metrics)} default metrics")

    def register_metric(self, definition: MetricDefinition) -> None:
        """
        Register a metric definition.

        Args:
            definition: MetricDefinition to register

        Raises:
            ValueError: If metric name already registered or invalid
        """
        definition.validate()

        with self._metrics_lock:
            if definition.name in self._definitions:
                logger.debug(f"Metric already registered: {definition.name}")
                return

            self._definitions[definition.name] = definition

            # Initialize storage based on type
            if definition.metric_type == MetricType.COUNTER:
                self._counters[definition.name] = {}
            elif definition.metric_type == MetricType.GAUGE:
                self._gauges[definition.name] = {}
            elif definition.metric_type == MetricType.HISTOGRAM:
                self._histograms[definition.name] = {}

            logger.debug(f"Registered metric: {definition.name}")

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dictionary to a hashable key."""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    # -------------------------------------------------------------------------
    # COUNTER OPERATIONS
    # -------------------------------------------------------------------------

    def increment_counter(
        self,
        name: str,
        labels: Dict[str, str],
        value: float = 1.0
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            labels: Label values as dictionary
            value: Value to increment by (default 1.0)
        """
        with self._metrics_lock:
            if name not in self._counters:
                logger.warning(f"Counter not registered: {name}")
                return

            key = self._labels_to_key(labels)
            if key not in self._counters[name]:
                self._counters[name][key] = 0.0

            self._counters[name][key] += value

    def get_counter(self, name: str, labels: Dict[str, str]) -> float:
        """
        Get current counter value.

        Args:
            name: Metric name
            labels: Label values

        Returns:
            Current counter value
        """
        with self._metrics_lock:
            if name not in self._counters:
                return 0.0

            key = self._labels_to_key(labels)
            return self._counters[name].get(key, 0.0)

    # -------------------------------------------------------------------------
    # GAUGE OPERATIONS
    # -------------------------------------------------------------------------

    def set_gauge(
        self,
        name: str,
        labels: Dict[str, str],
        value: float
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            labels: Label values as dictionary
            value: Value to set
        """
        with self._metrics_lock:
            if name not in self._gauges:
                logger.warning(f"Gauge not registered: {name}")
                return

            key = self._labels_to_key(labels)
            self._gauges[name][key] = value

    def increment_gauge(
        self,
        name: str,
        labels: Dict[str, str],
        value: float = 1.0
    ) -> None:
        """
        Increment a gauge metric.

        Args:
            name: Metric name
            labels: Label values
            value: Value to increment by
        """
        with self._metrics_lock:
            if name not in self._gauges:
                logger.warning(f"Gauge not registered: {name}")
                return

            key = self._labels_to_key(labels)
            current = self._gauges[name].get(key, 0.0)
            self._gauges[name][key] = current + value

    def decrement_gauge(
        self,
        name: str,
        labels: Dict[str, str],
        value: float = 1.0
    ) -> None:
        """
        Decrement a gauge metric.

        Args:
            name: Metric name
            labels: Label values
            value: Value to decrement by
        """
        self.increment_gauge(name, labels, -value)

    def get_gauge(self, name: str, labels: Dict[str, str]) -> float:
        """
        Get current gauge value.

        Args:
            name: Metric name
            labels: Label values

        Returns:
            Current gauge value
        """
        with self._metrics_lock:
            if name not in self._gauges:
                return 0.0

            key = self._labels_to_key(labels)
            return self._gauges[name].get(key, 0.0)

    # -------------------------------------------------------------------------
    # HISTOGRAM OPERATIONS
    # -------------------------------------------------------------------------

    def record_histogram(
        self,
        name: str,
        labels: Dict[str, str],
        value: float
    ) -> None:
        """
        Record a value in a histogram metric.

        Args:
            name: Metric name
            labels: Label values as dictionary
            value: Value to record
        """
        with self._metrics_lock:
            if name not in self._definitions:
                logger.warning(f"Histogram not registered: {name}")
                return

            definition = self._definitions[name]
            if definition.metric_type != MetricType.HISTOGRAM:
                logger.warning(f"Metric {name} is not a histogram")
                return

            key = self._labels_to_key(labels)

            if key not in self._histograms[name]:
                # Initialize histogram with empty buckets
                self._histograms[name][key] = HistogramValue(
                    buckets={b: 0 for b in definition.buckets},
                    sum_value=0.0,
                    count=0,
                    labels=labels
                )

            histogram = self._histograms[name][key]

            # Update buckets (cumulative)
            for bucket in definition.buckets:
                if value <= bucket:
                    histogram.buckets[bucket] += 1

            histogram.sum_value += value
            histogram.count += 1
            histogram.timestamp = datetime.now(timezone.utc)

    def get_histogram(
        self,
        name: str,
        labels: Dict[str, str]
    ) -> Optional[HistogramValue]:
        """
        Get histogram data.

        Args:
            name: Metric name
            labels: Label values

        Returns:
            HistogramValue or None if not found
        """
        with self._metrics_lock:
            if name not in self._histograms:
                return None

            key = self._labels_to_key(labels)
            return self._histograms[name].get(key)

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS - REQUEST METRICS
    # -------------------------------------------------------------------------

    def record_request_latency(
        self,
        endpoint: str,
        method: str,
        latency_seconds: float
    ) -> None:
        """
        Record API request latency.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            latency_seconds: Request latency in seconds
        """
        self.record_histogram(
            REQUEST_LATENCY_HISTOGRAM.name,
            {"endpoint": endpoint, "method": method},
            latency_seconds
        )

    def increment_request_count(
        self,
        endpoint: str,
        status: str,
        method: str
    ) -> None:
        """
        Increment request counter.

        Args:
            endpoint: API endpoint path
            status: Response status (success, error, etc.)
            method: HTTP method
        """
        self.increment_counter(
            REQUEST_COUNT_TOTAL.name,
            {"endpoint": endpoint, "status": status, "method": method}
        )

    def set_active_requests(self, endpoint: str, count: int) -> None:
        """
        Set active request count for an endpoint.

        Args:
            endpoint: API endpoint path
            count: Number of active requests
        """
        self.set_gauge(
            ACTIVE_REQUESTS_GAUGE.name,
            {"endpoint": endpoint},
            float(count)
        )

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS - INSPECTION METRICS
    # -------------------------------------------------------------------------

    def increment_inspections_completed(
        self,
        facility_id: str,
        inspection_type: str,
        status: str = "success"
    ) -> None:
        """
        Increment inspections completed counter.

        Args:
            facility_id: Facility identifier
            inspection_type: Type of inspection (thermal_scan, visual, etc.)
            status: Inspection status (success, partial, failed)
        """
        self.increment_counter(
            INSPECTIONS_COMPLETED_TOTAL.name,
            {"facility_id": facility_id, "inspection_type": inspection_type, "status": status}
        )

    def increment_thermal_images_processed(
        self,
        facility_id: str,
        camera_type: str,
        resolution: str,
        count: int = 1
    ) -> None:
        """
        Increment thermal images processed counter.

        Args:
            facility_id: Facility identifier
            camera_type: Type of thermal camera
            resolution: Image resolution (e.g., "640x480")
            count: Number of images processed
        """
        self.increment_counter(
            THERMAL_IMAGES_PROCESSED_TOTAL.name,
            {"facility_id": facility_id, "camera_type": camera_type, "resolution": resolution},
            float(count)
        )

    def record_image_processing_duration(
        self,
        processing_type: str,
        image_format: str,
        duration_seconds: float
    ) -> None:
        """
        Record image processing duration.

        Args:
            processing_type: Type of processing (analysis, enhancement, etc.)
            image_format: Image format (FLIR, TIFF, etc.)
            duration_seconds: Processing duration in seconds
        """
        self.record_histogram(
            IMAGE_PROCESSING_DURATION_HISTOGRAM.name,
            {"processing_type": processing_type, "image_format": image_format},
            duration_seconds
        )

    def increment_hotspots_detected(
        self,
        facility_id: str,
        severity: str,
        zone: str,
        count: int = 1
    ) -> None:
        """
        Increment hotspots detected counter.

        Args:
            facility_id: Facility identifier
            severity: Severity level (low, medium, high, critical)
            zone: Facility zone where hotspot was detected
            count: Number of hotspots
        """
        self.increment_counter(
            HOTSPOTS_DETECTED_TOTAL.name,
            {"facility_id": facility_id, "severity": severity, "zone": zone},
            float(count)
        )

    def increment_anomalies_classified(
        self,
        facility_id: str,
        anomaly_type: str,
        confidence_level: str,
        count: int = 1
    ) -> None:
        """
        Increment anomalies classified counter.

        Args:
            facility_id: Facility identifier
            anomaly_type: Type of anomaly (missing_insulation, damaged, wet, etc.)
            confidence_level: Classification confidence (high, medium, low)
            count: Number of anomalies
        """
        self.increment_counter(
            ANOMALIES_CLASSIFIED_TOTAL.name,
            {"facility_id": facility_id, "anomaly_type": anomaly_type, "confidence_level": confidence_level},
            float(count)
        )

    def set_inspection_coverage(
        self,
        facility_id: str,
        equipment_type: str,
        coverage_percent: float
    ) -> None:
        """
        Set inspection coverage percentage.

        Args:
            facility_id: Facility identifier
            equipment_type: Type of equipment
            coverage_percent: Coverage percentage (0-100)
        """
        self.set_gauge(
            INSPECTION_COVERAGE_GAUGE.name,
            {"facility_id": facility_id, "equipment_type": equipment_type},
            coverage_percent
        )

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS - HEAT LOSS METRICS
    # -------------------------------------------------------------------------

    def set_total_heat_loss(
        self,
        facility_id: str,
        zone: str,
        equipment_type: str,
        heat_loss_watts: float
    ) -> None:
        """
        Set total heat loss for a facility zone.

        Args:
            facility_id: Facility identifier
            zone: Facility zone
            equipment_type: Type of equipment
            heat_loss_watts: Total heat loss in watts
        """
        self.set_gauge(
            TOTAL_HEAT_LOSS_WATTS_GAUGE.name,
            {"facility_id": facility_id, "zone": zone, "equipment_type": equipment_type},
            heat_loss_watts
        )

    def set_heat_loss_per_meter(
        self,
        facility_id: str,
        insulation_type: str,
        condition: str,
        heat_loss_wm: float
    ) -> None:
        """
        Set heat loss per meter.

        Args:
            facility_id: Facility identifier
            insulation_type: Type of insulation
            condition: Insulation condition
            heat_loss_wm: Heat loss in W/m
        """
        self.set_gauge(
            HEAT_LOSS_PER_METER_GAUGE.name,
            {"facility_id": facility_id, "insulation_type": insulation_type, "condition": condition},
            heat_loss_wm
        )

    def set_energy_cost(
        self,
        facility_id: str,
        period: str,
        tariff_type: str,
        cost_dollars: float
    ) -> None:
        """
        Set energy cost due to heat loss.

        Args:
            facility_id: Facility identifier
            period: Cost period (daily, monthly, annual)
            tariff_type: Energy tariff type
            cost_dollars: Cost in dollars
        """
        self.set_gauge(
            ENERGY_COST_DOLLARS_GAUGE.name,
            {"facility_id": facility_id, "period": period, "tariff_type": tariff_type},
            cost_dollars
        )

    def set_carbon_emissions(
        self,
        facility_id: str,
        scope: str,
        fuel_type: str,
        emissions_kg: float
    ) -> None:
        """
        Set carbon emissions due to heat loss.

        Args:
            facility_id: Facility identifier
            scope: Emission scope (scope1, scope2)
            fuel_type: Fuel type (natural_gas, oil, electric)
            emissions_kg: Emissions in kg CO2e
        """
        self.set_gauge(
            CARBON_EMISSIONS_KG_GAUGE.name,
            {"facility_id": facility_id, "scope": scope, "fuel_type": fuel_type},
            emissions_kg
        )

    def record_heat_loss(
        self,
        facility_id: str,
        equipment_type: str,
        heat_loss_watts: float
    ) -> None:
        """
        Record individual heat loss measurement in histogram.

        Args:
            facility_id: Facility identifier
            equipment_type: Type of equipment
            heat_loss_watts: Heat loss in watts
        """
        self.record_histogram(
            HEAT_LOSS_HISTOGRAM.name,
            {"facility_id": facility_id, "equipment_type": equipment_type},
            heat_loss_watts
        )

    def set_ambient_temperature(
        self,
        facility_id: str,
        zone: str,
        temperature_celsius: float
    ) -> None:
        """
        Set ambient temperature.

        Args:
            facility_id: Facility identifier
            zone: Facility zone
            temperature_celsius: Temperature in Celsius
        """
        self.set_gauge(
            AMBIENT_TEMPERATURE_GAUGE.name,
            {"facility_id": facility_id, "zone": zone},
            temperature_celsius
        )

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS - DEGRADATION METRICS
    # -------------------------------------------------------------------------

    def set_equipment_by_condition(
        self,
        facility_id: str,
        condition_severity: str,
        equipment_type: str,
        count: int
    ) -> None:
        """
        Set count of equipment by condition severity.

        Args:
            facility_id: Facility identifier
            condition_severity: Severity level (good, fair, poor, critical)
            equipment_type: Type of equipment
            count: Number of equipment items
        """
        self.set_gauge(
            EQUIPMENT_BY_CONDITION_GAUGE.name,
            {"facility_id": facility_id, "condition_severity": condition_severity, "equipment_type": equipment_type},
            float(count)
        )

    def set_average_degradation_rate(
        self,
        facility_id: str,
        insulation_type: str,
        age_bracket: str,
        rate_percent_per_year: float
    ) -> None:
        """
        Set average degradation rate.

        Args:
            facility_id: Facility identifier
            insulation_type: Type of insulation
            age_bracket: Age bracket (0-5yr, 5-10yr, 10-20yr, 20+yr)
            rate_percent_per_year: Degradation rate in percent per year
        """
        self.set_gauge(
            AVERAGE_DEGRADATION_RATE_GAUGE.name,
            {"facility_id": facility_id, "insulation_type": insulation_type, "age_bracket": age_bracket},
            rate_percent_per_year
        )

    def set_moisture_content(
        self,
        facility_id: str,
        equipment_id: str,
        zone: str,
        moisture_percent: float
    ) -> None:
        """
        Set moisture content measurement.

        Args:
            facility_id: Facility identifier
            equipment_id: Equipment identifier
            zone: Facility zone
            moisture_percent: Moisture content percentage
        """
        self.set_gauge(
            MOISTURE_CONTENT_GAUGE.name,
            {"facility_id": facility_id, "equipment_id": equipment_id, "zone": zone},
            moisture_percent
        )

    def increment_repairs_prioritized(
        self,
        facility_id: str,
        urgency: str,
        repair_type: str,
        count: int = 1
    ) -> None:
        """
        Increment repairs prioritized counter.

        Args:
            facility_id: Facility identifier
            urgency: Urgency level (routine, planned, urgent, emergency)
            repair_type: Type of repair
            count: Number of repairs
        """
        self.increment_counter(
            REPAIRS_PRIORITIZED_TOTAL.name,
            {"facility_id": facility_id, "urgency": urgency, "repair_type": repair_type},
            float(count)
        )

    def set_time_to_failure(
        self,
        facility_id: str,
        equipment_id: str,
        failure_mode: str,
        days: float
    ) -> None:
        """
        Set estimated time to failure.

        Args:
            facility_id: Facility identifier
            equipment_id: Equipment identifier
            failure_mode: Failure mode
            days: Days until failure
        """
        self.set_gauge(
            TIME_TO_FAILURE_DAYS_GAUGE.name,
            {"facility_id": facility_id, "equipment_id": equipment_id, "failure_mode": failure_mode},
            days
        )

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS - INTEGRATION METRICS
    # -------------------------------------------------------------------------

    def set_camera_connection_status(
        self,
        camera_id: str,
        camera_type: str,
        facility_id: str,
        is_connected: bool
    ) -> None:
        """
        Set thermal camera connection status.

        Args:
            camera_id: Camera identifier
            camera_type: Camera type (FLIR, InfraTec, etc.)
            facility_id: Facility identifier
            is_connected: True if connected
        """
        self.set_gauge(
            CAMERA_CONNECTION_STATUS_GAUGE.name,
            {"camera_id": camera_id, "camera_type": camera_type, "facility_id": facility_id},
            1.0 if is_connected else 0.0
        )

    def record_camera_latency(
        self,
        camera_id: str,
        operation: str,
        latency_seconds: float
    ) -> None:
        """
        Record camera operation latency.

        Args:
            camera_id: Camera identifier
            operation: Operation type (connect, capture, transfer)
            latency_seconds: Latency in seconds
        """
        self.record_histogram(
            CAMERA_LATENCY_HISTOGRAM.name,
            {"camera_id": camera_id, "operation": operation},
            latency_seconds
        )

    def increment_cmms_work_orders(
        self,
        facility_id: str,
        work_order_type: str,
        priority: str,
        count: int = 1
    ) -> None:
        """
        Increment CMMS work orders created counter.

        Args:
            facility_id: Facility identifier
            work_order_type: Type of work order
            priority: Work order priority
            count: Number of work orders
        """
        self.increment_counter(
            CMMS_WORK_ORDERS_CREATED_TOTAL.name,
            {"facility_id": facility_id, "work_order_type": work_order_type, "priority": priority},
            float(count)
        )

    def record_cmms_latency(
        self,
        cmms_type: str,
        operation: str,
        latency_seconds: float
    ) -> None:
        """
        Record CMMS operation latency.

        Args:
            cmms_type: CMMS type (sap_pm, maximo, etc.)
            operation: Operation type (create, update, query)
            latency_seconds: Latency in seconds
        """
        self.record_histogram(
            CMMS_LATENCY_HISTOGRAM.name,
            {"cmms_type": cmms_type, "operation": operation},
            latency_seconds
        )

    def increment_connector_errors(
        self,
        system: str,
        error_type: str,
        operation: str
    ) -> None:
        """
        Increment connector error counter.

        Args:
            system: External system name
            error_type: Type of error
            operation: Operation that failed
        """
        self.increment_counter(
            CONNECTOR_ERRORS_TOTAL.name,
            {"system": system, "error_type": error_type, "operation": operation}
        )

    def increment_data_points_processed(
        self,
        source: str,
        data_type: str,
        count: int = 1
    ) -> None:
        """
        Increment data points processed counter.

        Args:
            source: Data source identifier
            data_type: Type of data processed
            count: Number of data points
        """
        self.increment_counter(
            DATA_POINTS_PROCESSED_TOTAL.name,
            {"source": source, "data_type": data_type},
            float(count)
        )

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS - CALCULATOR METRICS
    # -------------------------------------------------------------------------

    def record_calculation_duration(
        self,
        calculator_type: str,
        calculation_mode: str,
        duration_seconds: float
    ) -> None:
        """
        Record calculation duration.

        Args:
            calculator_type: Type of calculator (heat_loss, degradation, etc.)
            calculation_mode: Calculation mode (single, batch, streaming)
            duration_seconds: Calculation duration in seconds
        """
        self.record_histogram(
            CALCULATION_DURATION_HISTOGRAM.name,
            {"calculator_type": calculator_type, "calculation_mode": calculation_mode},
            duration_seconds
        )

    def increment_calculation_count(
        self,
        calculator_type: str,
        status: str
    ) -> None:
        """
        Increment calculation counter.

        Args:
            calculator_type: Type of calculator
            status: Calculation status (success, error)
        """
        self.increment_counter(
            CALCULATION_COUNT_TOTAL.name,
            {"calculator_type": calculator_type, "status": status}
        )

    def set_cache_hit_ratio(self, cache_type: str, ratio: float) -> None:
        """
        Set cache hit ratio.

        Args:
            cache_type: Type of cache (calculation, lookup, etc.)
            ratio: Hit ratio between 0 and 1
        """
        self.set_gauge(
            CACHE_HIT_RATIO_GAUGE.name,
            {"cache_type": cache_type},
            ratio
        )

    def update_cache_stats(
        self,
        cache_type: str,
        hit: bool
    ) -> None:
        """
        Update cache hit/miss statistics and recalculate ratio.

        Args:
            cache_type: Type of cache
            hit: True if cache hit, False if miss
        """
        with self._metrics_lock:
            if cache_type not in self._cache_hits:
                self._cache_hits[cache_type] = 0
                self._cache_misses[cache_type] = 0

            if hit:
                self._cache_hits[cache_type] += 1
            else:
                self._cache_misses[cache_type] += 1

            total = self._cache_hits[cache_type] + self._cache_misses[cache_type]
            if total > 0:
                ratio = self._cache_hits[cache_type] / total
                self.set_cache_hit_ratio(cache_type, ratio)

    # -------------------------------------------------------------------------
    # METRIC EXPORT
    # -------------------------------------------------------------------------

    def export_prometheus_format(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            String containing metrics in Prometheus exposition format
        """
        lines: List[str] = []

        with self._metrics_lock:
            for name, definition in self._definitions.items():
                # Add HELP and TYPE headers
                lines.append(f"# HELP {name} {definition.description}")
                lines.append(f"# TYPE {name} {definition.metric_type.value}")

                if definition.metric_type == MetricType.COUNTER:
                    self._export_counter(lines, name, definition)
                elif definition.metric_type == MetricType.GAUGE:
                    self._export_gauge(lines, name, definition)
                elif definition.metric_type == MetricType.HISTOGRAM:
                    self._export_histogram(lines, name, definition)

                lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _export_counter(
        self,
        lines: List[str],
        name: str,
        definition: MetricDefinition
    ) -> None:
        """Export counter metrics."""
        if name not in self._counters:
            return

        for key, value in self._counters[name].items():
            labels_str = self._key_to_labels_str(key)
            if labels_str:
                lines.append(f"{name}{{{labels_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

    def _export_gauge(
        self,
        lines: List[str],
        name: str,
        definition: MetricDefinition
    ) -> None:
        """Export gauge metrics."""
        if name not in self._gauges:
            return

        for key, value in self._gauges[name].items():
            labels_str = self._key_to_labels_str(key)
            if labels_str:
                lines.append(f"{name}{{{labels_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

    def _export_histogram(
        self,
        lines: List[str],
        name: str,
        definition: MetricDefinition
    ) -> None:
        """Export histogram metrics."""
        if name not in self._histograms:
            return

        for key, histogram in self._histograms[name].items():
            labels_str = self._key_to_labels_str(key)

            # Export bucket values
            for bucket, count in sorted(histogram.buckets.items()):
                bucket_str = "+Inf" if bucket == float("inf") else str(bucket)
                if labels_str:
                    lines.append(
                        f'{name}_bucket{{{labels_str},le="{bucket_str}"}} {count}'
                    )
                else:
                    lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')

            # Export sum and count
            if labels_str:
                lines.append(f"{name}_sum{{{labels_str}}} {histogram.sum_value}")
                lines.append(f"{name}_count{{{labels_str}}} {histogram.count}")
            else:
                lines.append(f"{name}_sum {histogram.sum_value}")
                lines.append(f"{name}_count {histogram.count}")

    def _key_to_labels_str(self, key: str) -> str:
        """Convert label key back to Prometheus format."""
        if not key:
            return ""

        parts = key.split("|")
        label_pairs = []
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                label_pairs.append(f'{k}="{v}"')

        return ",".join(label_pairs)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary of all metrics organized by type
        """
        with self._metrics_lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        key: {
                            "buckets": dict(h.buckets),
                            "sum": h.sum_value,
                            "count": h.count,
                            "labels": h.labels
                        }
                        for key, h in histograms.items()
                    }
                    for name, histograms in self._histograms.items()
                },
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self._start_time
                ).total_seconds()
            }

    def reset_all(self) -> None:
        """Reset all metric values (for testing purposes)."""
        with self._metrics_lock:
            for name in self._counters:
                self._counters[name] = {}
            for name in self._gauges:
                self._gauges[name] = {}
            for name in self._histograms:
                self._histograms[name] = {}
            self._cache_hits = {}
            self._cache_misses = {}

        logger.info("All metrics reset")


# =============================================================================
# DECORATORS
# =============================================================================

def timed_operation(
    metric_name: str,
    labels_func: Optional[Callable[..., Dict[str, str]]] = None
) -> Callable:
    """
    Decorator to time function execution and record to histogram.

    Args:
        metric_name: Histogram metric name to record to
        labels_func: Function to extract labels from arguments

    Returns:
        Decorated function

    Example:
        >>> @timed_operation("gl015_calculation_duration_seconds",
        ...                  lambda self, x: {"calculator_type": "heat_loss"})
        ... def calculate_heat_loss(self, data):
        ...     # calculation logic
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                labels = labels_func(*args, **kwargs) if labels_func else {}
                collector.record_histogram(metric_name, labels, duration)

        return wrapper
    return decorator


def count_calls(
    metric_name: str,
    labels_func: Optional[Callable[..., Dict[str, str]]] = None,
    include_status: bool = True
) -> Callable:
    """
    Decorator to count function calls.

    Args:
        metric_name: Counter metric name
        labels_func: Function to extract labels
        include_status: Whether to include success/error status

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            labels = labels_func(*args, **kwargs) if labels_func else {}

            try:
                result = func(*args, **kwargs)
                if include_status:
                    labels["status"] = "success"
                collector.increment_counter(metric_name, labels)
                return result
            except Exception as e:
                if include_status:
                    labels["status"] = "error"
                collector.increment_counter(metric_name, labels)
                raise

        return wrapper
    return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        MetricsCollector singleton instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def create_metrics_endpoint() -> Callable[[], str]:
    """
    Create a metrics endpoint handler.

    Returns:
        Callable that returns Prometheus metrics string

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> metrics_handler = create_metrics_endpoint()
        >>> @app.get("/metrics")
        ... def metrics():
        ...     return Response(metrics_handler(), media_type="text/plain")
    """
    def handler() -> str:
        collector = get_metrics_collector()
        return collector.export_prometheus_format()

    return handler


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (for testing)."""
    global _global_collector
    _global_collector = None


__all__ = [
    # Main Classes
    "MetricsCollector",
    "MetricDefinition",
    "MetricValue",
    "HistogramValue",

    # Enumerations
    "MetricType",

    # Request Metrics
    "REQUEST_LATENCY_HISTOGRAM",
    "REQUEST_COUNT_TOTAL",
    "ACTIVE_REQUESTS_GAUGE",
    "REQUEST_SIZE_HISTOGRAM",
    "RESPONSE_SIZE_HISTOGRAM",

    # Inspection Metrics
    "INSPECTIONS_COMPLETED_TOTAL",
    "THERMAL_IMAGES_PROCESSED_TOTAL",
    "IMAGE_PROCESSING_DURATION_HISTOGRAM",
    "HOTSPOTS_DETECTED_TOTAL",
    "ANOMALIES_CLASSIFIED_TOTAL",
    "INSPECTION_COVERAGE_GAUGE",
    "SURFACE_AREA_INSPECTED_GAUGE",

    # Heat Loss Metrics
    "TOTAL_HEAT_LOSS_WATTS_GAUGE",
    "HEAT_LOSS_PER_METER_GAUGE",
    "ENERGY_COST_DOLLARS_GAUGE",
    "CARBON_EMISSIONS_KG_GAUGE",
    "HEAT_LOSS_HISTOGRAM",
    "AMBIENT_TEMPERATURE_GAUGE",
    "SURFACE_TEMPERATURE_HISTOGRAM",

    # Degradation Metrics
    "EQUIPMENT_BY_CONDITION_GAUGE",
    "AVERAGE_DEGRADATION_RATE_GAUGE",
    "INSULATION_THICKNESS_GAUGE",
    "INSULATION_EFFICIENCY_GAUGE",
    "MOISTURE_CONTENT_GAUGE",
    "REPAIRS_PRIORITIZED_TOTAL",
    "REPAIR_COST_ESTIMATE_GAUGE",
    "TIME_TO_FAILURE_DAYS_GAUGE",

    # Integration Metrics
    "CAMERA_CONNECTION_STATUS_GAUGE",
    "CAMERA_LATENCY_HISTOGRAM",
    "CMMS_WORK_ORDERS_CREATED_TOTAL",
    "CMMS_LATENCY_HISTOGRAM",
    "CONNECTOR_ERRORS_TOTAL",
    "DATA_POINTS_PROCESSED_TOTAL",

    # Calculator Metrics
    "CALCULATION_DURATION_HISTOGRAM",
    "CALCULATION_COUNT_TOTAL",
    "CACHE_HIT_RATIO_GAUGE",

    # Bucket Configurations
    "REQUEST_LATENCY_BUCKETS",
    "IMAGE_PROCESSING_DURATION_BUCKETS",
    "HEAT_LOSS_BUCKETS",
    "TEMPERATURE_BUCKETS",
    "CONNECTOR_LATENCY_BUCKETS",
    "SIZE_BUCKETS",

    # Decorators
    "timed_operation",
    "count_calls",

    # Utility Functions
    "get_metrics_collector",
    "create_metrics_endpoint",
    "reset_metrics_collector",
]
