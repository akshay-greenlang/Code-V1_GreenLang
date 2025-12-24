"""
GL-012 STEAMQUAL SteamQualityController - Metrics Collection

This module provides Prometheus-compatible metrics collection for steam quality
monitoring and control, including dryness fraction, carryover risk, separator
efficiency, quality events, and calculation duration tracking.

Key Metrics:
    - steam_quality_dryness_fraction (gauge): Current steam dryness fraction (0.0-1.0)
    - steam_quality_carryover_risk (gauge): Carryover risk score (0.0-1.0)
    - steam_quality_separator_efficiency (gauge): Separator efficiency percentage
    - steam_quality_events_total (counter): Total quality events by type
    - steam_quality_calculation_duration_seconds (histogram): Calculation latencies

Example:
    >>> collector = SteamQualityMetricsCollector(namespace="steamqual")
    >>> collector.record_dryness_fraction(0.97, separator_id="SEP-001")
    >>> collector.record_carryover_risk(0.15, separator_id="SEP-001")
    >>> prometheus_output = collector.to_prometheus()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import hashlib
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class MetricValue:
    """Single metric value with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: str = "gauge"  # gauge, counter, histogram, summary
    help_text: str = ""

    def to_prometheus_line(self) -> str:
        """Convert to Prometheus text format line."""
        label_str = ""
        if self.labels:
            parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
            label_str = "{" + ",".join(parts) + "}"
        return f"{self.name}{label_str} {self.value}"


@dataclass
class SteamQualityMetrics:
    """Steam quality operating metrics."""
    separator_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Primary quality metrics
    dryness_fraction: float = 1.0  # 0.0-1.0 (1.0 = 100% dry steam)
    moisture_content_percent: float = 0.0  # 0-100%
    carryover_risk: float = 0.0  # 0.0-1.0 risk score

    # Separator performance
    separator_efficiency_percent: float = 95.0
    separator_pressure_drop_bar: float = 0.0
    separator_level_percent: float = 50.0  # Drain pot level
    drain_rate_kg_h: float = 0.0

    # Operating conditions
    inlet_pressure_bar: float = 0.0
    outlet_pressure_bar: float = 0.0
    inlet_temperature_c: float = 0.0
    outlet_temperature_c: float = 0.0
    steam_flow_kg_h: float = 0.0
    steam_velocity_m_s: float = 0.0

    # Condensate tracking
    condensate_load_kg_h: float = 0.0
    flash_steam_percent: float = 0.0

    # Risk indicators
    water_hammer_risk: float = 0.0  # 0.0-1.0
    flooding_risk: float = 0.0  # 0.0-1.0

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data = (
            f"{self.separator_id}:{self.timestamp.isoformat()}:"
            f"{self.dryness_fraction}:{self.carryover_risk}:"
            f"{self.separator_efficiency_percent}"
        )
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class CalculationMetrics:
    """Metrics for calculation/inference operations."""
    operation: str
    duration_seconds: float
    success: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    separator_id: Optional[str] = None
    error_message: Optional[str] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None


@dataclass
class MetricsSummary:
    """Summary of steam quality metrics over a time window."""
    time_window_start: datetime
    time_window_end: datetime
    system_id: str

    # Dryness fraction statistics
    avg_dryness_fraction: float = 0.0
    min_dryness_fraction: float = 0.0
    max_dryness_fraction: float = 0.0

    # Carryover risk statistics
    avg_carryover_risk: float = 0.0
    max_carryover_risk: float = 0.0
    carryover_events_count: int = 0

    # Separator efficiency statistics
    avg_separator_efficiency: float = 0.0
    min_separator_efficiency: float = 0.0

    # Event counts
    total_events: int = 0
    low_dryness_events: int = 0
    high_moisture_events: int = 0
    separator_flooding_events: int = 0
    water_hammer_risk_events: int = 0

    # Calculation performance
    total_calculations: int = 0
    avg_calculation_time_s: float = 0.0
    p50_calculation_time_s: float = 0.0
    p95_calculation_time_s: float = 0.0
    p99_calculation_time_s: float = 0.0
    calculation_error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "time_window_start": self.time_window_start.isoformat(),
            "time_window_end": self.time_window_end.isoformat(),
            "system_id": self.system_id,
            "dryness_fraction": {
                "avg": self.avg_dryness_fraction,
                "min": self.min_dryness_fraction,
                "max": self.max_dryness_fraction,
            },
            "carryover_risk": {
                "avg": self.avg_carryover_risk,
                "max": self.max_carryover_risk,
                "events_count": self.carryover_events_count,
            },
            "separator_efficiency": {
                "avg": self.avg_separator_efficiency,
                "min": self.min_separator_efficiency,
            },
            "events": {
                "total": self.total_events,
                "low_dryness": self.low_dryness_events,
                "high_moisture": self.high_moisture_events,
                "separator_flooding": self.separator_flooding_events,
                "water_hammer_risk": self.water_hammer_risk_events,
            },
            "calculations": {
                "total": self.total_calculations,
                "avg_time_s": self.avg_calculation_time_s,
                "p50_time_s": self.p50_calculation_time_s,
                "p95_time_s": self.p95_calculation_time_s,
                "p99_time_s": self.p99_calculation_time_s,
                "error_rate": self.calculation_error_rate,
            },
        }


# =============================================================================
# Main Metrics Collector
# =============================================================================

class SteamQualityMetricsCollector:
    """
    Prometheus-compatible metrics collector for steam quality monitoring.

    This class provides comprehensive metrics collection for steam quality
    control including dryness fraction, carryover risk, separator efficiency,
    quality events, and calculation duration tracking with Prometheus text
    format export.

    Attributes:
        namespace: Metric namespace prefix (default: "steamqual")
        system_id: System identifier for labeling

    Example:
        >>> collector = SteamQualityMetricsCollector(namespace="steamqual")
        >>> collector.record_dryness_fraction(0.97, separator_id="SEP-001")
        >>> collector.record_calculation_duration("quality_assessment", 0.025)
        >>> print(collector.to_prometheus())
    """

    # Default histogram buckets for calculation duration (seconds)
    DEFAULT_DURATION_BUCKETS = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ]

    # Thresholds for quality metrics
    DRYNESS_THRESHOLD_LOW = 0.95  # Below this triggers LOW_DRYNESS alert
    CARRYOVER_RISK_THRESHOLD = 0.3  # Above this triggers CARRYOVER_RISK alert
    SEPARATOR_EFFICIENCY_MIN = 90.0  # Minimum acceptable efficiency

    def __init__(
        self,
        namespace: str = "steamqual",
        system_id: str = "default",
    ) -> None:
        """
        Initialize SteamQualityMetricsCollector.

        Args:
            namespace: Metric namespace prefix
            system_id: System identifier for labeling
        """
        self.namespace = namespace
        self.system_id = system_id
        self._start_time = time.time()
        self._lock = threading.Lock()

        # Metric storage
        self._gauges: Dict[str, MetricValue] = {}
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

        # Time-series data for summaries
        self._dryness_readings: List[Tuple[datetime, str, float]] = []
        self._carryover_readings: List[Tuple[datetime, str, float]] = []
        self._efficiency_readings: List[Tuple[datetime, str, float]] = []
        self._calculation_times: List[CalculationMetrics] = []
        self._quality_events: List[Tuple[datetime, str, str]] = []

        # Data retention settings
        self._max_data_points = 10000
        self._max_data_age_hours = 24

        # Register standard metrics
        self._register_standard_metrics()

        logger.info(
            "SteamQualityMetricsCollector initialized: namespace=%s, system_id=%s",
            namespace,
            system_id,
        )

    def _register_standard_metrics(self) -> None:
        """Register standard application metrics."""
        self._set_gauge(
            "info",
            1.0,
            labels={"version": "1.0.0", "agent": "GL-012", "codename": "STEAMQUAL"},
            help_text="Application info",
        )

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels to string key for storage."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def _set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        help_text: str = "",
    ) -> None:
        """Set gauge metric value."""
        with self._lock:
            full_name = f"{self.namespace}_{name}"
            labels = labels or {}
            labels["system_id"] = self.system_id
            label_key = self._labels_to_key(labels)

            self._gauges[f"{full_name}{label_key}"] = MetricValue(
                name=full_name,
                value=value,
                labels=labels,
                metric_type="gauge",
                help_text=help_text,
            )

    def _increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment counter metric."""
        with self._lock:
            full_name = f"{self.namespace}_{name}"
            labels = labels or {}
            labels["system_id"] = self.system_id
            label_key = self._labels_to_key(labels)
            key = f"{full_name}{label_key}"

            if key not in self._counters:
                self._counters[key] = 0.0
            self._counters[key] += value

            self._gauges[key] = MetricValue(
                name=full_name,
                value=self._counters[key],
                labels=labels,
                metric_type="counter",
            )

    def _observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add observation to histogram."""
        with self._lock:
            full_name = f"{self.namespace}_{name}"
            labels = labels or {}
            labels["system_id"] = self.system_id
            label_key = self._labels_to_key(labels)
            key = f"{full_name}{label_key}"

            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

            # Keep only recent observations
            if len(self._histograms[key]) > self._max_data_points:
                self._histograms[key] = self._histograms[key][-self._max_data_points:]

    def _cleanup_old_data(self) -> None:
        """Remove data older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_data_age_hours)

        with self._lock:
            self._dryness_readings = [
                x for x in self._dryness_readings if x[0] > cutoff
            ]
            self._carryover_readings = [
                x for x in self._carryover_readings if x[0] > cutoff
            ]
            self._efficiency_readings = [
                x for x in self._efficiency_readings if x[0] > cutoff
            ]
            self._calculation_times = [
                x for x in self._calculation_times if x.timestamp > cutoff
            ]
            self._quality_events = [
                x for x in self._quality_events if x[0] > cutoff
            ]

    # =========================================================================
    # Primary Steam Quality Metrics
    # =========================================================================

    def record_dryness_fraction(
        self,
        dryness_fraction: float,
        separator_id: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record steam dryness fraction measurement.

        The dryness fraction (x) represents the mass fraction of dry steam
        in a wet steam mixture. Value of 1.0 indicates completely dry steam.

        Args:
            dryness_fraction: Dryness fraction (0.0-1.0)
            separator_id: Separator/equipment identifier
            labels: Optional additional labels
        """
        now = datetime.now(timezone.utc)

        # Validate range
        dryness_fraction = max(0.0, min(1.0, dryness_fraction))

        # Store time-series data
        with self._lock:
            self._dryness_readings.append((now, separator_id, dryness_fraction))

            # Trim if needed
            if len(self._dryness_readings) > self._max_data_points:
                self._dryness_readings = self._dryness_readings[-self._max_data_points:]

        # Update gauge
        metric_labels = {"separator_id": separator_id}
        if labels:
            metric_labels.update(labels)

        self._set_gauge(
            "dryness_fraction",
            dryness_fraction,
            labels=metric_labels,
            help_text="Steam dryness fraction (0.0-1.0, 1.0 = 100% dry)",
        )

        # Calculate moisture content (inverse of dryness)
        moisture_content = 1.0 - dryness_fraction
        self._set_gauge(
            "moisture_content",
            moisture_content,
            labels=metric_labels,
            help_text="Steam moisture content fraction (0.0-1.0)",
        )

        logger.debug(
            "Recorded dryness fraction: separator=%s, x=%.4f",
            separator_id,
            dryness_fraction,
        )

    def record_carryover_risk(
        self,
        carryover_risk: float,
        separator_id: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record carryover risk score.

        Carryover risk indicates the probability of liquid carryover from
        steam separators into downstream equipment.

        Args:
            carryover_risk: Risk score (0.0-1.0, higher = more risk)
            separator_id: Separator/equipment identifier
            labels: Optional additional labels
        """
        now = datetime.now(timezone.utc)

        # Validate range
        carryover_risk = max(0.0, min(1.0, carryover_risk))

        # Store time-series data
        with self._lock:
            self._carryover_readings.append((now, separator_id, carryover_risk))

            if len(self._carryover_readings) > self._max_data_points:
                self._carryover_readings = self._carryover_readings[-self._max_data_points:]

        # Update gauge
        metric_labels = {"separator_id": separator_id}
        if labels:
            metric_labels.update(labels)

        self._set_gauge(
            "carryover_risk",
            carryover_risk,
            labels=metric_labels,
            help_text="Carryover risk score (0.0-1.0, higher = more risk)",
        )

        logger.debug(
            "Recorded carryover risk: separator=%s, risk=%.4f",
            separator_id,
            carryover_risk,
        )

    def record_separator_efficiency(
        self,
        efficiency_percent: float,
        separator_id: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record separator efficiency measurement.

        Separator efficiency indicates how effectively the separator removes
        liquid droplets from the steam flow.

        Args:
            efficiency_percent: Efficiency percentage (0-100)
            separator_id: Separator/equipment identifier
            labels: Optional additional labels
        """
        now = datetime.now(timezone.utc)

        # Validate range
        efficiency_percent = max(0.0, min(100.0, efficiency_percent))

        # Store time-series data
        with self._lock:
            self._efficiency_readings.append((now, separator_id, efficiency_percent))

            if len(self._efficiency_readings) > self._max_data_points:
                self._efficiency_readings = self._efficiency_readings[-self._max_data_points:]

        # Update gauge
        metric_labels = {"separator_id": separator_id}
        if labels:
            metric_labels.update(labels)

        self._set_gauge(
            "separator_efficiency",
            efficiency_percent,
            labels=metric_labels,
            help_text="Separator efficiency percentage (0-100)",
        )

        logger.debug(
            "Recorded separator efficiency: separator=%s, efficiency=%.2f%%",
            separator_id,
            efficiency_percent,
        )

    # =========================================================================
    # Event Counter
    # =========================================================================

    def record_quality_event(
        self,
        event_type: str,
        separator_id: str,
        severity: str = "warning",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a steam quality event.

        Event types:
            - LOW_DRYNESS: Dryness fraction below threshold
            - HIGH_MOISTURE: Moisture carryover detected
            - CARRYOVER_RISK: Carryover risk above threshold
            - SEPARATOR_FLOODING: Separator drain issues
            - WATER_HAMMER_RISK: Condensate accumulation risk
            - DATA_QUALITY_DEGRADED: Sensor/data issues

        Args:
            event_type: Type of quality event
            separator_id: Separator/equipment identifier
            severity: Event severity (info, warning, critical)
            labels: Optional additional labels
        """
        now = datetime.now(timezone.utc)

        # Store event
        with self._lock:
            self._quality_events.append((now, separator_id, event_type))

            if len(self._quality_events) > self._max_data_points:
                self._quality_events = self._quality_events[-self._max_data_points:]

        # Increment counter
        metric_labels = {
            "separator_id": separator_id,
            "event_type": event_type,
            "severity": severity,
        }
        if labels:
            metric_labels.update(labels)

        self._increment_counter(
            "events_total",
            labels=metric_labels,
        )

        logger.info(
            "Quality event recorded: type=%s, separator=%s, severity=%s",
            event_type,
            separator_id,
            severity,
        )

    # =========================================================================
    # Calculation Duration Histogram
    # =========================================================================

    def record_calculation_duration(
        self,
        operation: str,
        duration_seconds: float,
        separator_id: Optional[str] = None,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record calculation/inference duration.

        Args:
            operation: Operation name (e.g., "quality_assessment", "carryover_prediction")
            duration_seconds: Duration in seconds
            separator_id: Optional separator identifier
            success: Whether calculation succeeded
            labels: Optional additional labels
        """
        now = datetime.now(timezone.utc)

        # Store calculation metrics
        calc_metrics = CalculationMetrics(
            operation=operation,
            duration_seconds=duration_seconds,
            success=success,
            timestamp=now,
            separator_id=separator_id,
        )

        with self._lock:
            self._calculation_times.append(calc_metrics)

            if len(self._calculation_times) > self._max_data_points:
                self._calculation_times = self._calculation_times[-self._max_data_points:]

        # Update histogram
        metric_labels = {"operation": operation}
        if separator_id:
            metric_labels["separator_id"] = separator_id
        if labels:
            metric_labels.update(labels)

        self._observe_histogram(
            "calculation_duration_seconds",
            duration_seconds,
            labels=metric_labels,
        )

        # Increment counter
        status = "success" if success else "error"
        self._increment_counter(
            "calculations_total",
            labels={"operation": operation, "status": status},
        )

        logger.debug(
            "Recorded calculation: operation=%s, duration=%.4fs, success=%s",
            operation,
            duration_seconds,
            success,
        )

    # =========================================================================
    # Bulk Recording
    # =========================================================================

    def record_steam_quality_metrics(
        self,
        metrics: SteamQualityMetrics,
    ) -> None:
        """
        Record comprehensive steam quality metrics.

        Args:
            metrics: SteamQualityMetrics instance with all measurements
        """
        separator_id = metrics.separator_id
        labels = {"separator_id": separator_id}

        # Primary quality metrics
        self.record_dryness_fraction(metrics.dryness_fraction, separator_id)
        self.record_carryover_risk(metrics.carryover_risk, separator_id)
        self.record_separator_efficiency(
            metrics.separator_efficiency_percent, separator_id
        )

        # Additional gauges
        self._set_gauge(
            "separator_pressure_drop_bar",
            metrics.separator_pressure_drop_bar,
            labels=labels,
        )
        self._set_gauge(
            "separator_level_percent",
            metrics.separator_level_percent,
            labels=labels,
        )
        self._set_gauge(
            "drain_rate_kg_h",
            metrics.drain_rate_kg_h,
            labels=labels,
        )
        self._set_gauge(
            "inlet_pressure_bar",
            metrics.inlet_pressure_bar,
            labels=labels,
        )
        self._set_gauge(
            "outlet_pressure_bar",
            metrics.outlet_pressure_bar,
            labels=labels,
        )
        self._set_gauge(
            "steam_flow_kg_h",
            metrics.steam_flow_kg_h,
            labels=labels,
        )
        self._set_gauge(
            "steam_velocity_m_s",
            metrics.steam_velocity_m_s,
            labels=labels,
        )
        self._set_gauge(
            "condensate_load_kg_h",
            metrics.condensate_load_kg_h,
            labels=labels,
        )
        self._set_gauge(
            "water_hammer_risk",
            metrics.water_hammer_risk,
            labels=labels,
        )
        self._set_gauge(
            "flooding_risk",
            metrics.flooding_risk,
            labels=labels,
        )

        logger.debug(
            "Recorded steam quality metrics: separator=%s, dryness=%.4f",
            separator_id,
            metrics.dryness_fraction,
        )

    # =========================================================================
    # Summary and Export
    # =========================================================================

    def get_metrics_summary(
        self,
        time_window: Optional[timedelta] = None,
    ) -> MetricsSummary:
        """
        Get metrics summary over a time window.

        Args:
            time_window: Time window to summarize (default: last hour)

        Returns:
            MetricsSummary with aggregated metrics
        """
        self._cleanup_old_data()

        if time_window is None:
            time_window = timedelta(hours=1)

        now = datetime.now(timezone.utc)
        cutoff = now - time_window

        with self._lock:
            # Filter dryness readings
            recent_dryness = [
                x[2] for x in self._dryness_readings if x[0] > cutoff
            ]

            # Filter carryover readings
            recent_carryover = [
                x[2] for x in self._carryover_readings if x[0] > cutoff
            ]

            # Filter efficiency readings
            recent_efficiency = [
                x[2] for x in self._efficiency_readings if x[0] > cutoff
            ]

            # Filter calculations
            recent_calcs = [
                x for x in self._calculation_times if x.timestamp > cutoff
            ]
            calc_durations = [x.duration_seconds for x in recent_calcs]
            calc_errors = [x for x in recent_calcs if not x.success]

            # Filter events
            recent_events = [
                x for x in self._quality_events if x[0] > cutoff
            ]

        # Calculate statistics
        avg_dryness = sum(recent_dryness) / len(recent_dryness) if recent_dryness else 0.0
        min_dryness = min(recent_dryness) if recent_dryness else 0.0
        max_dryness = max(recent_dryness) if recent_dryness else 0.0

        avg_carryover = sum(recent_carryover) / len(recent_carryover) if recent_carryover else 0.0
        max_carryover = max(recent_carryover) if recent_carryover else 0.0

        avg_efficiency = sum(recent_efficiency) / len(recent_efficiency) if recent_efficiency else 0.0
        min_efficiency = min(recent_efficiency) if recent_efficiency else 0.0

        # Calculate percentiles for calculation times
        p50, p95, p99 = 0.0, 0.0, 0.0
        if calc_durations:
            sorted_durations = sorted(calc_durations)
            n = len(sorted_durations)
            p50 = sorted_durations[int(n * 0.5)] if n > 0 else 0.0
            p95 = sorted_durations[int(n * 0.95)] if n > 0 else 0.0
            p99 = sorted_durations[int(n * 0.99)] if n > 0 else 0.0

        # Count events by type
        low_dryness_events = len([e for e in recent_events if e[2] == "LOW_DRYNESS"])
        high_moisture_events = len([e for e in recent_events if e[2] == "HIGH_MOISTURE"])
        separator_flooding_events = len([e for e in recent_events if e[2] == "SEPARATOR_FLOODING"])
        water_hammer_events = len([e for e in recent_events if e[2] == "WATER_HAMMER_RISK"])
        carryover_events = len([e for e in recent_events if e[2] == "CARRYOVER_RISK"])

        return MetricsSummary(
            time_window_start=cutoff,
            time_window_end=now,
            system_id=self.system_id,
            avg_dryness_fraction=avg_dryness,
            min_dryness_fraction=min_dryness,
            max_dryness_fraction=max_dryness,
            avg_carryover_risk=avg_carryover,
            max_carryover_risk=max_carryover,
            carryover_events_count=carryover_events,
            avg_separator_efficiency=avg_efficiency,
            min_separator_efficiency=min_efficiency,
            total_events=len(recent_events),
            low_dryness_events=low_dryness_events,
            high_moisture_events=high_moisture_events,
            separator_flooding_events=separator_flooding_events,
            water_hammer_risk_events=water_hammer_events,
            total_calculations=len(recent_calcs),
            avg_calculation_time_s=(
                sum(calc_durations) / len(calc_durations) if calc_durations else 0.0
            ),
            p50_calculation_time_s=p50,
            p95_calculation_time_s=p95,
            p99_calculation_time_s=p99,
            calculation_error_rate=(
                len(calc_errors) / len(recent_calcs) if recent_calcs else 0.0
            ),
        )

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-compatible text format string
        """
        lines = []

        # Add uptime
        uptime = time.time() - self._start_time
        lines.append(f"# TYPE {self.namespace}_uptime_seconds gauge")
        lines.append(f"# HELP {self.namespace}_uptime_seconds Agent uptime in seconds")
        lines.append(f"{self.namespace}_uptime_seconds {uptime:.2f}")
        lines.append("")

        with self._lock:
            # Group gauges by metric name
            metrics_by_name: Dict[str, List[MetricValue]] = {}
            for metric in self._gauges.values():
                if metric.name not in metrics_by_name:
                    metrics_by_name[metric.name] = []
                metrics_by_name[metric.name].append(metric)

            # Output each metric
            for name, metrics in sorted(metrics_by_name.items()):
                metric_type = metrics[0].metric_type
                help_text = metrics[0].help_text

                lines.append(f"# TYPE {name} {metric_type}")
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")

                for m in metrics:
                    lines.append(m.to_prometheus_line())

                lines.append("")

            # Output histograms
            for key, values in sorted(self._histograms.items()):
                if not values:
                    continue

                name = key.split("{")[0]
                labels_part = key[len(name):] if "{" in key else ""

                lines.append(f"# TYPE {name} histogram")

                # Calculate bucket counts
                sorted_values = sorted(values)
                for bucket in self.DEFAULT_DURATION_BUCKETS:
                    count = len([v for v in sorted_values if v <= bucket])
                    bucket_labels = labels_part.rstrip("}") if labels_part else ""
                    if bucket_labels:
                        bucket_labels += f',le="{bucket}"' + "}"
                    else:
                        bucket_labels = f'{{le="{bucket}"}}'
                    lines.append(f"{name}_bucket{bucket_labels} {count}")

                # Add +Inf bucket
                inf_labels = labels_part.rstrip("}") if labels_part else ""
                if inf_labels:
                    inf_labels += ',le="+Inf"' + "}"
                else:
                    inf_labels = '{le="+Inf"}'
                lines.append(f"{name}_bucket{inf_labels} {len(values)}")

                # Add sum and count
                lines.append(f"{name}_sum{labels_part} {sum(values):.6f}")
                lines.append(f"{name}_count{labels_part} {len(values)}")
                lines.append("")

        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            return {
                key: {
                    "name": m.name,
                    "value": m.value,
                    "labels": m.labels,
                    "type": m.metric_type,
                    "timestamp": m.timestamp.isoformat(),
                }
                for key, m in self._gauges.items()
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._gauges.clear()
            self._counters.clear()
            self._histograms.clear()
            self._dryness_readings.clear()
            self._carryover_readings.clear()
            self._efficiency_readings.clear()
            self._calculation_times.clear()
            self._quality_events.clear()

        self._register_standard_metrics()
        logger.info("Metrics reset")
