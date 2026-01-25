"""
GL-015 INSULSCAN - Prometheus Metrics Collection

This module provides Prometheus-compatible metrics for monitoring insulation
scanning and thermal assessment operations, including analysis performance,
heat loss detection, condition scoring, hot spot detection, and repair
recommendations.

Metrics Categories:
    - Analysis metrics: Total analyses, duration, success rate
    - Heat loss metrics: Current heat loss by asset and surface type
    - Condition metrics: Condition score by asset and insulation type
    - Detection metrics: Hot spots detected, severity distribution
    - Recommendation metrics: Repair recommendations generated
    - Energy metrics: Projected energy savings
    - System metrics: Circuit breaker state, data quality

Prometheus Naming Conventions:
    - All metrics prefixed with 'insulscan_'
    - Counter metrics end with '_total'
    - Histogram metrics end with '_seconds' or '_bytes'
    - Gauge metrics use descriptive names
    - Labels follow snake_case naming

Example:
    >>> collector = InsulscanMetricsCollector()
    >>> collector.record_analysis(
    ...     asset_id="PIPE-001",
    ...     surface_type="pipe",
    ...     duration_seconds=1.5,
    ...     success=True
    ... )
    >>> print(collector.get_metrics_prometheus())
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import threading
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class SurfaceType(Enum):
    """Types of insulated surfaces."""
    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    VALVE = "valve"
    FLANGE = "flange"
    DUCT = "duct"
    EQUIPMENT = "equipment"
    OTHER = "other"


class InsulationType(Enum):
    """Types of insulation materials."""
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    FIBERGLASS = "fiberglass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    POLYURETHANE = "polyurethane"
    ELASTOMERIC = "elastomeric"
    CERAMIC_FIBER = "ceramic_fiber"
    UNKNOWN = "unknown"


class HotSpotSeverity(Enum):
    """Severity levels for detected hot spots."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = 0  # Normal operation
    HALF_OPEN = 1  # Testing recovery
    OPEN = 2  # Blocking requests


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class MetricValue:
    """A single metric value with labels and metadata."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: MetricType = MetricType.GAUGE
    help_text: str = ""
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "type": self.metric_type.value,
            "help": self.help_text,
            "unit": self.unit,
        }


@dataclass
class HistogramBucket:
    """Histogram bucket for latency/duration metrics."""
    le: float  # less than or equal
    count: int = 0


@dataclass
class AnalysisMetrics:
    """Metrics for a single insulation analysis."""
    asset_id: str
    surface_type: SurfaceType
    insulation_type: InsulationType
    duration_seconds: float
    success: bool
    heat_loss_watts: float = 0.0
    condition_score: float = 0.0
    hot_spots_detected: int = 0
    hot_spot_severities: Dict[HotSpotSeverity, int] = field(default_factory=dict)
    repair_recommended: bool = False
    projected_savings_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "asset_id": self.asset_id,
            "surface_type": self.surface_type.value,
            "insulation_type": self.insulation_type.value,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "heat_loss_watts": self.heat_loss_watts,
            "condition_score": self.condition_score,
            "hot_spots_detected": self.hot_spots_detected,
            "hot_spot_severities": {k.value: v for k, v in self.hot_spot_severities.items()},
            "repair_recommended": self.repair_recommended,
            "projected_savings_usd": self.projected_savings_usd,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetricsSummary:
    """Summary of metrics over a time window."""
    time_window_start: datetime
    time_window_end: datetime
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    avg_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    total_heat_loss_watts: float = 0.0
    avg_condition_score: float = 0.0
    min_condition_score: float = 1.0
    total_hot_spots: int = 0
    critical_hot_spots: int = 0
    total_repair_recommendations: int = 0
    total_projected_savings_usd: float = 0.0
    assets_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "time_window": {
                "start": self.time_window_start.isoformat(),
                "end": self.time_window_end.isoformat(),
            },
            "analyses": {
                "total": self.total_analyses,
                "successful": self.successful_analyses,
                "failed": self.failed_analyses,
                "success_rate": (
                    self.successful_analyses / self.total_analyses
                    if self.total_analyses > 0 else 0.0
                ),
            },
            "performance": {
                "avg_duration_seconds": self.avg_duration_seconds,
                "max_duration_seconds": self.max_duration_seconds,
            },
            "heat_loss": {
                "total_watts": self.total_heat_loss_watts,
            },
            "condition": {
                "avg_score": self.avg_condition_score,
                "min_score": self.min_condition_score,
            },
            "hot_spots": {
                "total": self.total_hot_spots,
                "critical": self.critical_hot_spots,
            },
            "recommendations": {
                "repair_count": self.total_repair_recommendations,
                "projected_savings_usd": self.total_projected_savings_usd,
            },
            "assets_analyzed": self.assets_analyzed,
        }


# =============================================================================
# Metrics Definitions
# =============================================================================

METRICS_DEFINITIONS = {
    # Counter metrics
    "insulscan_analyses_total": {
        "type": MetricType.COUNTER,
        "help": "Total number of insulation analyses performed",
        "labels": ["asset_id", "surface_type", "insulation_type", "status"],
    },
    "insulscan_hot_spots_detected": {
        "type": MetricType.COUNTER,
        "help": "Total number of hot spots detected during analyses",
        "labels": ["asset_id", "surface_type", "severity"],
    },
    "insulscan_repair_recommendations_total": {
        "type": MetricType.COUNTER,
        "help": "Total number of repair recommendations generated",
        "labels": ["asset_id", "surface_type", "insulation_type", "priority"],
    },
    "insulscan_errors_total": {
        "type": MetricType.COUNTER,
        "help": "Total number of errors during analysis operations",
        "labels": ["asset_id", "error_type"],
    },

    # Histogram metrics
    "insulscan_analysis_duration_seconds": {
        "type": MetricType.HISTOGRAM,
        "help": "Duration of insulation analysis operations in seconds",
        "labels": ["asset_id", "surface_type", "insulation_type"],
        "buckets": [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    },

    # Gauge metrics
    "insulscan_heat_loss_watts": {
        "type": MetricType.GAUGE,
        "help": "Current heat loss in watts for an asset",
        "labels": ["asset_id", "surface_type", "insulation_type"],
    },
    "insulscan_condition_score": {
        "type": MetricType.GAUGE,
        "help": "Current insulation condition score (0-1, 1 is best)",
        "labels": ["asset_id", "surface_type", "insulation_type"],
    },
    "insulscan_energy_savings_usd": {
        "type": MetricType.GAUGE,
        "help": "Projected annual energy savings in USD if repairs are made",
        "labels": ["asset_id", "surface_type"],
    },
    "insulscan_circuit_breaker_state": {
        "type": MetricType.GAUGE,
        "help": "Current state of circuit breaker (0=closed, 1=half-open, 2=open)",
        "labels": ["service"],
    },
    "insulscan_active_assets": {
        "type": MetricType.GAUGE,
        "help": "Number of assets currently being monitored",
        "labels": [],
    },
    "insulscan_data_quality_score": {
        "type": MetricType.GAUGE,
        "help": "Data quality score for analysis inputs (0-1)",
        "labels": ["asset_id"],
    },
}


# =============================================================================
# Main Metrics Collector Class
# =============================================================================

class InsulscanMetricsCollector:
    """
    Prometheus-compatible metrics collector for GL-015 INSULSCAN.

    This collector manages all metrics related to insulation scanning and
    thermal assessment operations, providing both real-time gauges and
    cumulative counters for comprehensive monitoring.

    Attributes:
        namespace: Metrics namespace prefix
        agent_id: Agent identifier for labeling
        agent_name: Agent codename for labeling

    Example:
        >>> collector = InsulscanMetricsCollector()
        >>> collector.record_analysis(
        ...     asset_id="PIPE-001",
        ...     surface_type=SurfaceType.PIPE,
        ...     duration_seconds=1.5,
        ...     success=True
        ... )
        >>> prometheus_metrics = collector.get_metrics_prometheus()
    """

    AGENT_ID = "GL-015"
    AGENT_NAME = "INSULSCAN"

    # Default histogram buckets for duration (in seconds)
    DEFAULT_DURATION_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]

    def __init__(
        self,
        namespace: str = "insulscan",
        enable_provenance: bool = True,
    ) -> None:
        """
        Initialize InsulscanMetricsCollector.

        Args:
            namespace: Metrics namespace prefix
            enable_provenance: Whether to compute provenance hashes
        """
        self.namespace = namespace
        self._enable_provenance = enable_provenance
        self._lock = threading.Lock()

        # Metric storage
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, List[HistogramBucket]]] = {}
        self._histogram_sums: Dict[str, Dict[str, float]] = {}
        self._histogram_counts: Dict[str, Dict[str, int]] = {}

        # Historical data for summary calculations
        self._analysis_history: List[AnalysisMetrics] = []
        self._max_history_size = 10000

        # Initialize standard metrics
        self._init_standard_metrics()

        logger.info(
            "InsulscanMetricsCollector initialized: namespace=%s, provenance=%s",
            namespace,
            enable_provenance,
        )

    def _init_standard_metrics(self) -> None:
        """Initialize standard metric containers."""
        # Counters
        self._counters = {
            "insulscan_analyses_total": {},
            "insulscan_hot_spots_detected": {},
            "insulscan_repair_recommendations_total": {},
            "insulscan_errors_total": {},
        }

        # Gauges
        self._gauges = {
            "insulscan_heat_loss_watts": {},
            "insulscan_condition_score": {},
            "insulscan_energy_savings_usd": {},
            "insulscan_circuit_breaker_state": {},
            "insulscan_active_assets": {},
            "insulscan_data_quality_score": {},
        }

        # Histograms
        for metric in ["insulscan_analysis_duration_seconds"]:
            self._histograms[metric] = {}
            self._histogram_sums[metric] = {}
            self._histogram_counts[metric] = {}

    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from labels dictionary."""
        if not labels:
            return ""
        sorted_items = sorted(labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_items)

    def _compute_provenance(self, data: str) -> str:
        """Compute SHA-256 provenance hash."""
        if not self._enable_provenance:
            return ""
        return hashlib.sha256(data.encode()).hexdigest()

    # =========================================================================
    # Counter Methods
    # =========================================================================

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter metric name
            value: Value to increment by (default: 1.0)
            labels: Optional label dictionary
        """
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._counters:
                self._counters[name] = {}

            if key not in self._counters[name]:
                self._counters[name][key] = 0.0

            self._counters[name][key] += value

    def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Get current counter value.

        Args:
            name: Counter metric name
            labels: Optional label dictionary

        Returns:
            Current counter value
        """
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._counters:
                return 0.0
            return self._counters[name].get(key, 0.0)

    # =========================================================================
    # Gauge Methods
    # =========================================================================

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Gauge metric name
            value: Value to set
            labels: Optional label dictionary
        """
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = {}

            self._gauges[name][key] = value

    def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """
        Get current gauge value.

        Args:
            name: Gauge metric name
            labels: Optional label dictionary

        Returns:
            Current gauge value or None if not set
        """
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._gauges:
                return None
            return self._gauges[name].get(key)

    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a gauge metric.

        Args:
            name: Gauge metric name
            value: Value to increment by
            labels: Optional label dictionary
        """
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = {}

            current = self._gauges[name].get(key, 0.0)
            self._gauges[name][key] = current + value

    def dec_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Decrement a gauge metric.

        Args:
            name: Gauge metric name
            value: Value to decrement by
            labels: Optional label dictionary
        """
        self.inc_gauge(name, -value, labels)

    # =========================================================================
    # Histogram Methods
    # =========================================================================

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        """
        Observe a value for a histogram metric.

        Args:
            name: Histogram metric name
            value: Observed value
            labels: Optional label dictionary
            buckets: Custom bucket boundaries (default: DEFAULT_DURATION_BUCKETS)
        """
        labels = labels or {}
        key = self._labels_key(labels)
        buckets = buckets or self.DEFAULT_DURATION_BUCKETS

        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = {}
                self._histogram_sums[name] = {}
                self._histogram_counts[name] = {}

            if key not in self._histograms[name]:
                self._histograms[name][key] = [
                    HistogramBucket(le=b) for b in buckets
                ] + [HistogramBucket(le=float("inf"))]
                self._histogram_sums[name][key] = 0.0
                self._histogram_counts[name][key] = 0

            # Update buckets
            for bucket in self._histograms[name][key]:
                if value <= bucket.le:
                    bucket.count += 1

            self._histogram_sums[name][key] += value
            self._histogram_counts[name][key] += 1

    # =========================================================================
    # High-Level Recording Methods
    # =========================================================================

    def record_analysis(
        self,
        asset_id: str,
        surface_type: SurfaceType,
        insulation_type: InsulationType,
        duration_seconds: float,
        success: bool,
        heat_loss_watts: float = 0.0,
        condition_score: float = 0.0,
        hot_spots: Optional[Dict[HotSpotSeverity, int]] = None,
        repair_recommended: bool = False,
        projected_savings_usd: float = 0.0,
    ) -> AnalysisMetrics:
        """
        Record a complete insulation analysis.

        Args:
            asset_id: Asset identifier
            surface_type: Type of insulated surface
            insulation_type: Type of insulation material
            duration_seconds: Analysis duration in seconds
            success: Whether analysis completed successfully
            heat_loss_watts: Calculated heat loss in watts
            condition_score: Insulation condition score (0-1)
            hot_spots: Dictionary of hot spot severities and counts
            repair_recommended: Whether repair is recommended
            projected_savings_usd: Projected annual savings if repaired

        Returns:
            AnalysisMetrics record
        """
        hot_spots = hot_spots or {}
        total_hot_spots = sum(hot_spots.values())
        status = "success" if success else "failure"

        # Create analysis record
        analysis = AnalysisMetrics(
            asset_id=asset_id,
            surface_type=surface_type,
            insulation_type=insulation_type,
            duration_seconds=duration_seconds,
            success=success,
            heat_loss_watts=heat_loss_watts,
            condition_score=condition_score,
            hot_spots_detected=total_hot_spots,
            hot_spot_severities=hot_spots,
            repair_recommended=repair_recommended,
            projected_savings_usd=projected_savings_usd,
        )

        # Base labels for this analysis
        base_labels = {
            "asset_id": asset_id,
            "surface_type": surface_type.value,
            "insulation_type": insulation_type.value,
        }

        # Record counter: analyses total
        self.inc_counter(
            "insulscan_analyses_total",
            labels={**base_labels, "status": status},
        )

        # Record histogram: duration
        self.observe_histogram(
            "insulscan_analysis_duration_seconds",
            duration_seconds,
            labels=base_labels,
        )

        if success:
            # Record gauge: heat loss
            self.set_gauge(
                "insulscan_heat_loss_watts",
                heat_loss_watts,
                labels=base_labels,
            )

            # Record gauge: condition score
            self.set_gauge(
                "insulscan_condition_score",
                condition_score,
                labels=base_labels,
            )

            # Record gauge: projected savings
            self.set_gauge(
                "insulscan_energy_savings_usd",
                projected_savings_usd,
                labels={"asset_id": asset_id, "surface_type": surface_type.value},
            )

            # Record counters: hot spots by severity
            for severity, count in hot_spots.items():
                if count > 0:
                    self.inc_counter(
                        "insulscan_hot_spots_detected",
                        count,
                        labels={
                            "asset_id": asset_id,
                            "surface_type": surface_type.value,
                            "severity": severity.value,
                        },
                    )

            # Record counter: repair recommendations
            if repair_recommended:
                priority = self._calculate_repair_priority(condition_score, total_hot_spots)
                self.inc_counter(
                    "insulscan_repair_recommendations_total",
                    labels={**base_labels, "priority": priority},
                )
        else:
            # Record error
            self.inc_counter(
                "insulscan_errors_total",
                labels={"asset_id": asset_id, "error_type": "analysis_failure"},
            )

        # Store in history
        with self._lock:
            self._analysis_history.append(analysis)
            if len(self._analysis_history) > self._max_history_size:
                self._analysis_history = self._analysis_history[-self._max_history_size:]

        logger.debug(
            "Recorded analysis: asset=%s, type=%s, duration=%.2fs, success=%s",
            asset_id,
            surface_type.value,
            duration_seconds,
            success,
        )

        return analysis

    def _calculate_repair_priority(
        self,
        condition_score: float,
        hot_spot_count: int,
    ) -> str:
        """Calculate repair priority based on condition and hot spots."""
        if condition_score < 0.3 or hot_spot_count >= 5:
            return "critical"
        elif condition_score < 0.5 or hot_spot_count >= 3:
            return "high"
        elif condition_score < 0.7 or hot_spot_count >= 1:
            return "medium"
        else:
            return "low"

    def record_hot_spot(
        self,
        asset_id: str,
        surface_type: SurfaceType,
        severity: HotSpotSeverity,
        temperature_delta_c: float = 0.0,
    ) -> None:
        """
        Record a detected hot spot.

        Args:
            asset_id: Asset identifier
            surface_type: Type of insulated surface
            severity: Hot spot severity level
            temperature_delta_c: Temperature difference from expected
        """
        self.inc_counter(
            "insulscan_hot_spots_detected",
            labels={
                "asset_id": asset_id,
                "surface_type": surface_type.value,
                "severity": severity.value,
            },
        )

        logger.debug(
            "Recorded hot spot: asset=%s, severity=%s, delta=%.1fC",
            asset_id,
            severity.value,
            temperature_delta_c,
        )

    def record_repair_recommendation(
        self,
        asset_id: str,
        surface_type: SurfaceType,
        insulation_type: InsulationType,
        priority: str,
        estimated_cost_usd: float = 0.0,
        projected_savings_usd: float = 0.0,
    ) -> None:
        """
        Record a repair recommendation.

        Args:
            asset_id: Asset identifier
            surface_type: Type of insulated surface
            insulation_type: Type of insulation material
            priority: Recommendation priority (low, medium, high, critical)
            estimated_cost_usd: Estimated repair cost
            projected_savings_usd: Projected annual savings
        """
        self.inc_counter(
            "insulscan_repair_recommendations_total",
            labels={
                "asset_id": asset_id,
                "surface_type": surface_type.value,
                "insulation_type": insulation_type.value,
                "priority": priority,
            },
        )

        # Update projected savings gauge
        self.set_gauge(
            "insulscan_energy_savings_usd",
            projected_savings_usd,
            labels={"asset_id": asset_id, "surface_type": surface_type.value},
        )

        logger.info(
            "Recorded repair recommendation: asset=%s, priority=%s, savings=$%.2f",
            asset_id,
            priority,
            projected_savings_usd,
        )

    def set_circuit_breaker_state(
        self,
        service: str,
        state: CircuitBreakerState,
    ) -> None:
        """
        Set circuit breaker state for a service.

        Args:
            service: Service name
            state: Circuit breaker state
        """
        self.set_gauge(
            "insulscan_circuit_breaker_state",
            state.value,
            labels={"service": service},
        )

        logger.info(
            "Circuit breaker state updated: service=%s, state=%s",
            service,
            state.name,
        )

    def set_data_quality_score(
        self,
        asset_id: str,
        score: float,
    ) -> None:
        """
        Set data quality score for an asset.

        Args:
            asset_id: Asset identifier
            score: Quality score (0-1)
        """
        self.set_gauge(
            "insulscan_data_quality_score",
            score,
            labels={"asset_id": asset_id},
        )

    def set_active_assets_count(self, count: int) -> None:
        """
        Set the count of active monitored assets.

        Args:
            count: Number of active assets
        """
        self.set_gauge("insulscan_active_assets", float(count))

    def record_error(
        self,
        asset_id: str,
        error_type: str,
    ) -> None:
        """
        Record an error during analysis.

        Args:
            asset_id: Asset identifier
            error_type: Type of error
        """
        self.inc_counter(
            "insulscan_errors_total",
            labels={"asset_id": asset_id, "error_type": error_type},
        )

    # =========================================================================
    # Context Manager for Timing
    # =========================================================================

    def time_analysis(
        self,
        asset_id: str,
        surface_type: SurfaceType,
        insulation_type: InsulationType,
    ) -> "_AnalysisTimer":
        """
        Context manager to time an analysis operation.

        Args:
            asset_id: Asset identifier
            surface_type: Type of insulated surface
            insulation_type: Type of insulation material

        Returns:
            Timer context manager
        """
        return _AnalysisTimer(self, asset_id, surface_type, insulation_type)

    # =========================================================================
    # Summary and Export Methods
    # =========================================================================

    def get_metrics_summary(
        self,
        time_window: Optional[timedelta] = None,
    ) -> MetricsSummary:
        """
        Get summary of metrics over a time window.

        Args:
            time_window: Time window for summary (default: all history)

        Returns:
            MetricsSummary with aggregated statistics
        """
        now = datetime.now(timezone.utc)
        start = now - time_window if time_window else datetime.min.replace(tzinfo=timezone.utc)

        with self._lock:
            analyses = [
                a for a in self._analysis_history
                if a.timestamp >= start
            ]

        if not analyses:
            return MetricsSummary(
                time_window_start=start,
                time_window_end=now,
            )

        successful = [a for a in analyses if a.success]
        failed = [a for a in analyses if not a.success]
        durations = [a.duration_seconds for a in analyses]
        condition_scores = [a.condition_score for a in successful if a.condition_score > 0]
        unique_assets = set(a.asset_id for a in analyses)

        total_hot_spots = sum(a.hot_spots_detected for a in successful)
        critical_hot_spots = sum(
            a.hot_spot_severities.get(HotSpotSeverity.CRITICAL, 0)
            for a in successful
        )

        return MetricsSummary(
            time_window_start=start,
            time_window_end=now,
            total_analyses=len(analyses),
            successful_analyses=len(successful),
            failed_analyses=len(failed),
            avg_duration_seconds=sum(durations) / len(durations) if durations else 0.0,
            max_duration_seconds=max(durations) if durations else 0.0,
            total_heat_loss_watts=sum(a.heat_loss_watts for a in successful),
            avg_condition_score=(
                sum(condition_scores) / len(condition_scores)
                if condition_scores else 0.0
            ),
            min_condition_score=min(condition_scores) if condition_scores else 1.0,
            total_hot_spots=total_hot_spots,
            critical_hot_spots=critical_hot_spots,
            total_repair_recommendations=sum(1 for a in successful if a.repair_recommended),
            total_projected_savings_usd=sum(a.projected_savings_usd for a in successful),
            assets_analyzed=len(unique_assets),
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary containing all metric values
        """
        with self._lock:
            return {
                "counters": {k: dict(v) for k, v in self._counters.items()},
                "gauges": {k: dict(v) for k, v in self._gauges.items()},
                "histograms": {
                    name: {
                        key: {
                            "buckets": [(b.le, b.count) for b in buckets],
                            "sum": self._histogram_sums[name][key],
                            "count": self._histogram_counts[name][key],
                        }
                        for key, buckets in values.items()
                    }
                    for name, values in self._histograms.items()
                },
            }

    def get_metrics_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        with self._lock:
            # Counters
            for name, values in self._counters.items():
                definition = METRICS_DEFINITIONS.get(name, {})
                help_text = definition.get("help", f"{name} counter")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for labels_key, value in values.items():
                    if labels_key:
                        lines.append(f"{name}{{{labels_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Gauges
            for name, values in self._gauges.items():
                definition = METRICS_DEFINITIONS.get(name, {})
                help_text = definition.get("help", f"{name} gauge")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                for labels_key, value in values.items():
                    if labels_key:
                        lines.append(f"{name}{{{labels_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Histograms
            for name, values in self._histograms.items():
                definition = METRICS_DEFINITIONS.get(name, {})
                help_text = definition.get("help", f"{name} histogram")
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} histogram")
                for labels_key, buckets in values.items():
                    for bucket in buckets:
                        le_label = f'le="{bucket.le}"'
                        if labels_key:
                            full_labels = f"{labels_key},{le_label}"
                        else:
                            full_labels = le_label
                        lines.append(f"{name}_bucket{{{full_labels}}} {bucket.count}")

                    # Sum and count
                    if labels_key:
                        lines.append(
                            f"{name}_sum{{{labels_key}}} "
                            f"{self._histogram_sums[name][labels_key]}"
                        )
                        lines.append(
                            f"{name}_count{{{labels_key}}} "
                            f"{self._histogram_counts[name][labels_key]}"
                        )
                    else:
                        lines.append(f"{name}_sum {self._histogram_sums[name][labels_key]}")
                        lines.append(f"{name}_count {self._histogram_counts[name][labels_key]}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._init_standard_metrics()
            self._analysis_history.clear()

        logger.info("Metrics collector reset")


# =============================================================================
# Timer Context Manager
# =============================================================================

class _AnalysisTimer:
    """Context manager for timing analysis operations."""

    def __init__(
        self,
        collector: InsulscanMetricsCollector,
        asset_id: str,
        surface_type: SurfaceType,
        insulation_type: InsulationType,
    ) -> None:
        """
        Initialize analysis timer.

        Args:
            collector: Parent metrics collector
            asset_id: Asset identifier
            surface_type: Type of insulated surface
            insulation_type: Type of insulation material
        """
        self.collector = collector
        self.asset_id = asset_id
        self.surface_type = surface_type
        self.insulation_type = insulation_type
        self.start_time: float = 0.0
        self.duration: float = 0.0
        self.success: bool = True

    def __enter__(self) -> "_AnalysisTimer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and record metrics."""
        self.duration = time.perf_counter() - self.start_time
        self.success = exc_type is None

        # Record basic analysis without full details
        labels = {
            "asset_id": self.asset_id,
            "surface_type": self.surface_type.value,
            "insulation_type": self.insulation_type.value,
        }

        self.collector.observe_histogram(
            "insulscan_analysis_duration_seconds",
            self.duration,
            labels=labels,
        )

        self.collector.inc_counter(
            "insulscan_analyses_total",
            labels={**labels, "status": "success" if self.success else "failure"},
        )

        if not self.success:
            self.collector.inc_counter(
                "insulscan_errors_total",
                labels={"asset_id": self.asset_id, "error_type": "analysis_exception"},
            )


# =============================================================================
# Global Instance
# =============================================================================

_metrics_collector: Optional[InsulscanMetricsCollector] = None


def get_metrics_collector() -> InsulscanMetricsCollector:
    """
    Get or create the global metrics collector.

    Returns:
        Global InsulscanMetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = InsulscanMetricsCollector()
    return _metrics_collector
