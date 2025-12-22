"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Metrics Collection

This module provides Prometheus-compatible metrics collection for steam system
optimization, including computation latency, optimization results, recommendation
outcomes, API metrics, and steam KPIs.

Key Metrics:
    - Computation latency (histogram)
    - Optimization frequency and results
    - Recommendation acceptance rate
    - Trap failure detection accuracy
    - Steam system KPIs (efficiency, recovery rate, etc.)

Example:
    >>> collector = MetricsCollector(namespace="unifiedsteam")
    >>> collector.record_computation_time("desuperheater_optimization", 150.5)
    >>> collector.record_steam_kpi("steam_efficiency_percent", 92.5)
    >>> prometheus_output = collector.to_prometheus()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import logging
import time
import hashlib

logger = logging.getLogger(__name__)


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
class SteamMetrics:
    """Steam system operating metrics."""
    system_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Pressure (bar)
    header_pressure_bar: float = 0.0
    process_pressure_bar: float = 0.0
    condensate_pressure_bar: float = 0.0

    # Temperature (Celsius)
    steam_temperature_c: float = 0.0
    condensate_temperature_c: float = 0.0
    feedwater_temperature_c: float = 0.0

    # Flow (kg/h)
    steam_flow_kg_h: float = 0.0
    condensate_flow_kg_h: float = 0.0
    makeup_water_kg_h: float = 0.0

    # Quality
    steam_quality_percent: float = 100.0
    superheat_degree_c: float = 0.0

    # Efficiency
    boiler_efficiency_percent: float = 0.0
    condensate_recovery_percent: float = 0.0
    overall_efficiency_percent: float = 0.0


@dataclass
class OptimizationMetrics:
    """Optimization operation metrics."""
    system_id: str
    optimization_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Timing
    computation_time_ms: float = 0.0
    setup_time_ms: float = 0.0
    solve_time_ms: float = 0.0

    # Results
    objective_value: float = 0.0
    improvement_percent: float = 0.0
    constraints_satisfied: bool = True
    feasible: bool = True

    # Savings
    energy_savings_kwh: float = 0.0
    cost_savings_usd: float = 0.0
    co2_reduction_kg: float = 0.0


@dataclass
class TrapMetrics:
    """Steam trap monitoring metrics."""
    system_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Trap counts
    total_traps: int = 0
    operational_traps: int = 0
    failed_traps: int = 0
    blow_through_traps: int = 0
    blocked_traps: int = 0
    unknown_status_traps: int = 0

    # Detection metrics
    traps_inspected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Loss metrics
    total_steam_loss_kg_h: float = 0.0
    estimated_annual_loss_usd: float = 0.0

    @property
    def detection_accuracy(self) -> float:
        """Calculate detection accuracy."""
        total = (
            self.true_positives + self.false_positives +
            self.true_negatives + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        if (self.true_positives + self.false_positives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity)."""
        if (self.true_positives + self.false_negatives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)


@dataclass
class DesuperheaterMetrics:
    """Desuperheater optimization metrics."""
    system_id: str
    desuperheater_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Operating conditions
    inlet_temperature_c: float = 0.0
    outlet_temperature_c: float = 0.0
    target_temperature_c: float = 0.0
    spray_water_flow_kg_h: float = 0.0

    # Performance
    temperature_deviation_c: float = 0.0
    control_error_percent: float = 0.0
    efficiency_percent: float = 0.0

    # Optimization
    optimized_setpoint_c: float = 0.0
    energy_savings_percent: float = 0.0


@dataclass
class CondensateMetrics:
    """Condensate recovery metrics."""
    system_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Recovery rates
    recovery_rate_percent: float = 0.0
    flash_steam_recovered_percent: float = 0.0

    # Flow
    condensate_collected_kg_h: float = 0.0
    condensate_returned_kg_h: float = 0.0
    flash_steam_kg_h: float = 0.0

    # Losses
    flash_loss_kg_h: float = 0.0
    leak_loss_kg_h: float = 0.0
    drain_loss_kg_h: float = 0.0

    # Energy
    energy_recovered_kw: float = 0.0
    energy_lost_kw: float = 0.0


@dataclass
class MetricsSummary:
    """Summary of metrics over a time window."""
    time_window_start: datetime
    time_window_end: datetime
    system_id: str

    # Computation metrics
    total_computations: int = 0
    avg_computation_time_ms: float = 0.0
    p50_computation_time_ms: float = 0.0
    p95_computation_time_ms: float = 0.0
    p99_computation_time_ms: float = 0.0

    # Optimization metrics
    optimizations_run: int = 0
    optimizations_successful: int = 0
    total_savings_usd: float = 0.0

    # Recommendation metrics
    recommendations_generated: int = 0
    recommendations_accepted: int = 0
    recommendations_rejected: int = 0
    acceptance_rate_percent: float = 0.0

    # API metrics
    api_requests_total: int = 0
    api_errors_total: int = 0
    avg_api_latency_ms: float = 0.0

    # Trap metrics
    trap_failures_detected: int = 0
    trap_detection_accuracy_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "time_window_start": self.time_window_start.isoformat(),
            "time_window_end": self.time_window_end.isoformat(),
            "system_id": self.system_id,
            "computation": {
                "total": self.total_computations,
                "avg_time_ms": self.avg_computation_time_ms,
                "p50_time_ms": self.p50_computation_time_ms,
                "p95_time_ms": self.p95_computation_time_ms,
                "p99_time_ms": self.p99_computation_time_ms,
            },
            "optimization": {
                "total": self.optimizations_run,
                "successful": self.optimizations_successful,
                "success_rate_percent": (
                    (self.optimizations_successful / self.optimizations_run * 100)
                    if self.optimizations_run > 0 else 0.0
                ),
                "total_savings_usd": self.total_savings_usd,
            },
            "recommendations": {
                "generated": self.recommendations_generated,
                "accepted": self.recommendations_accepted,
                "rejected": self.recommendations_rejected,
                "acceptance_rate_percent": self.acceptance_rate_percent,
            },
            "api": {
                "requests_total": self.api_requests_total,
                "errors_total": self.api_errors_total,
                "error_rate_percent": (
                    (self.api_errors_total / self.api_requests_total * 100)
                    if self.api_requests_total > 0 else 0.0
                ),
                "avg_latency_ms": self.avg_api_latency_ms,
            },
            "trap_detection": {
                "failures_detected": self.trap_failures_detected,
                "accuracy_percent": self.trap_detection_accuracy_percent,
            },
        }


class MetricsCollector:
    """
    Prometheus-compatible metrics collector for steam system optimization.

    This class provides comprehensive metrics collection including computation
    latency, optimization results, recommendation outcomes, API metrics, and
    steam-specific KPIs with Prometheus text format export.

    Attributes:
        namespace: Metric namespace prefix

    Example:
        >>> collector = MetricsCollector(namespace="unifiedsteam")
        >>> collector.record_computation_time("optimization", 150.5)
        >>> collector.record_optimization_result(result)
        >>> print(collector.to_prometheus())
    """

    # Default histogram buckets (milliseconds)
    DEFAULT_LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    def __init__(
        self,
        namespace: str = "unifiedsteam",
        system_id: str = "default",
    ) -> None:
        """
        Initialize MetricsCollector.

        Args:
            namespace: Metric namespace prefix
            system_id: System identifier for labeling
        """
        self.namespace = namespace
        self.system_id = system_id
        self._start_time = time.time()

        # Metric storage
        self._gauges: Dict[str, MetricValue] = {}
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

        # Time-series data for summaries
        self._computation_times: List[tuple] = []  # (timestamp, operation, duration)
        self._optimization_results: List[tuple] = []  # (timestamp, result_dict)
        self._recommendation_outcomes: List[tuple] = []  # (timestamp, rec_id, outcome)
        self._api_requests: List[tuple] = []  # (timestamp, endpoint, status, latency)
        self._steam_kpis: Dict[str, List[tuple]] = {}  # kpi_name -> [(timestamp, value)]

        # Trap detection tracking
        self._trap_detection_results: List[tuple] = []  # (timestamp, predicted, actual)

        # Data retention
        self._max_data_points = 10000
        self._max_data_age_hours = 24

        # Register standard metrics
        self._register_standard_metrics()

        logger.info(
            "MetricsCollector initialized: namespace=%s, system_id=%s",
            namespace,
            system_id,
        )

    def _register_standard_metrics(self) -> None:
        """Register standard application metrics."""
        self._set_gauge(
            "info",
            1.0,
            labels={"version": "1.0.0", "agent": "GL-003", "codename": "UNIFIEDSTEAM"},
            help_text="Application info",
        )

    def _set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        help_text: str = "",
    ) -> None:
        """Set gauge metric value."""
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

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels to string key."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def _cleanup_old_data(self) -> None:
        """Remove data older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_data_age_hours)

        self._computation_times = [
            x for x in self._computation_times if x[0] > cutoff
        ]
        self._optimization_results = [
            x for x in self._optimization_results if x[0] > cutoff
        ]
        self._recommendation_outcomes = [
            x for x in self._recommendation_outcomes if x[0] > cutoff
        ]
        self._api_requests = [
            x for x in self._api_requests if x[0] > cutoff
        ]
        self._trap_detection_results = [
            x for x in self._trap_detection_results if x[0] > cutoff
        ]

        for kpi_name in self._steam_kpis:
            self._steam_kpis[kpi_name] = [
                x for x in self._steam_kpis[kpi_name] if x[0] > cutoff
            ]

    def record_computation_time(
        self,
        operation: str,
        duration_ms: float,
    ) -> None:
        """
        Record computation time for an operation.

        Args:
            operation: Operation name (e.g., "desuperheater_optimization")
            duration_ms: Duration in milliseconds
        """
        now = datetime.now(timezone.utc)
        self._computation_times.append((now, operation, duration_ms))

        # Update histogram
        self._observe_histogram(
            "computation_duration_ms",
            duration_ms,
            labels={"operation": operation},
        )

        # Update counter
        self._increment_counter(
            "computations_total",
            labels={"operation": operation},
        )

        logger.debug(
            "Recorded computation: operation=%s, duration=%.2fms",
            operation,
            duration_ms,
        )

    def record_optimization_result(
        self,
        result: OptimizationMetrics,
    ) -> None:
        """
        Record optimization result metrics.

        Args:
            result: OptimizationMetrics instance
        """
        now = datetime.now(timezone.utc)
        result_dict = {
            "optimization_type": result.optimization_type,
            "computation_time_ms": result.computation_time_ms,
            "objective_value": result.objective_value,
            "improvement_percent": result.improvement_percent,
            "feasible": result.feasible,
            "energy_savings_kwh": result.energy_savings_kwh,
            "cost_savings_usd": result.cost_savings_usd,
            "co2_reduction_kg": result.co2_reduction_kg,
        }
        self._optimization_results.append((now, result_dict))

        labels = {"optimization_type": result.optimization_type}

        # Update gauges
        self._set_gauge(
            "optimization_objective_value",
            result.objective_value,
            labels=labels,
        )
        self._set_gauge(
            "optimization_improvement_percent",
            result.improvement_percent,
            labels=labels,
        )

        # Update counters
        self._increment_counter("optimizations_total", labels=labels)
        if result.feasible:
            self._increment_counter("optimizations_successful_total", labels=labels)
        else:
            self._increment_counter("optimizations_failed_total", labels=labels)

        # Update savings
        self._increment_counter(
            "optimization_energy_savings_kwh_total",
            result.energy_savings_kwh,
            labels=labels,
        )
        self._increment_counter(
            "optimization_cost_savings_usd_total",
            result.cost_savings_usd,
            labels=labels,
        )
        self._increment_counter(
            "optimization_co2_reduction_kg_total",
            result.co2_reduction_kg,
            labels=labels,
        )

        logger.debug(
            "Recorded optimization: type=%s, improvement=%.2f%%, savings=$%.2f",
            result.optimization_type,
            result.improvement_percent,
            result.cost_savings_usd,
        )

    def record_recommendation_outcome(
        self,
        recommendation_id: str,
        outcome: str,  # "accepted", "rejected", "pending", "expired"
    ) -> None:
        """
        Record recommendation outcome.

        Args:
            recommendation_id: Recommendation identifier
            outcome: Outcome status
        """
        now = datetime.now(timezone.utc)
        self._recommendation_outcomes.append((now, recommendation_id, outcome))

        labels = {"outcome": outcome}
        self._increment_counter("recommendations_total", labels=labels)

        logger.debug(
            "Recorded recommendation outcome: id=%s, outcome=%s",
            recommendation_id[:8],
            outcome,
        )

    def record_api_request(
        self,
        endpoint: str,
        status: int,
        latency_ms: float,
    ) -> None:
        """
        Record API request metrics.

        Args:
            endpoint: API endpoint path
            status: HTTP status code
            latency_ms: Request latency in milliseconds
        """
        now = datetime.now(timezone.utc)
        self._api_requests.append((now, endpoint, status, latency_ms))

        status_category = f"{status // 100}xx"
        labels = {"endpoint": endpoint, "status": status_category}

        # Update counters
        self._increment_counter("api_requests_total", labels=labels)
        if status >= 400:
            self._increment_counter(
                "api_errors_total",
                labels={"endpoint": endpoint, "status": str(status)},
            )

        # Update histogram
        self._observe_histogram(
            "api_request_duration_ms",
            latency_ms,
            labels={"endpoint": endpoint},
        )

    def record_steam_kpi(
        self,
        kpi_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a steam system KPI.

        Args:
            kpi_name: KPI name (e.g., "steam_efficiency_percent")
            value: KPI value
            labels: Optional additional labels
        """
        now = datetime.now(timezone.utc)

        if kpi_name not in self._steam_kpis:
            self._steam_kpis[kpi_name] = []
        self._steam_kpis[kpi_name].append((now, value))

        # Update gauge
        metric_labels = labels or {}
        self._set_gauge(kpi_name, value, labels=metric_labels)

        logger.debug("Recorded KPI: %s=%.4f", kpi_name, value)

    def record_steam_metrics(self, metrics: SteamMetrics) -> None:
        """Record all steam operating metrics."""
        labels = {"system_id": metrics.system_id}

        # Pressure
        self._set_gauge("header_pressure_bar", metrics.header_pressure_bar, labels)
        self._set_gauge("process_pressure_bar", metrics.process_pressure_bar, labels)

        # Temperature
        self._set_gauge("steam_temperature_c", metrics.steam_temperature_c, labels)
        self._set_gauge("condensate_temperature_c", metrics.condensate_temperature_c, labels)

        # Flow
        self._set_gauge("steam_flow_kg_h", metrics.steam_flow_kg_h, labels)
        self._set_gauge("condensate_flow_kg_h", metrics.condensate_flow_kg_h, labels)

        # Quality
        self._set_gauge("steam_quality_percent", metrics.steam_quality_percent, labels)
        self._set_gauge("superheat_degree_c", metrics.superheat_degree_c, labels)

        # Efficiency
        self._set_gauge("boiler_efficiency_percent", metrics.boiler_efficiency_percent, labels)
        self._set_gauge("condensate_recovery_percent", metrics.condensate_recovery_percent, labels)
        self._set_gauge("overall_efficiency_percent", metrics.overall_efficiency_percent, labels)

    def record_trap_metrics(self, metrics: TrapMetrics) -> None:
        """Record steam trap metrics."""
        labels = {"system_id": metrics.system_id}

        # Trap counts
        self._set_gauge("traps_total", float(metrics.total_traps), labels)
        self._set_gauge("traps_operational", float(metrics.operational_traps), labels)
        self._set_gauge("traps_failed", float(metrics.failed_traps), labels)
        self._set_gauge("traps_blow_through", float(metrics.blow_through_traps), labels)
        self._set_gauge("traps_blocked", float(metrics.blocked_traps), labels)

        # Detection metrics
        self._set_gauge("trap_detection_accuracy", metrics.detection_accuracy, labels)
        self._set_gauge("trap_detection_precision", metrics.precision, labels)
        self._set_gauge("trap_detection_recall", metrics.recall, labels)

        # Loss metrics
        self._set_gauge("steam_loss_kg_h", metrics.total_steam_loss_kg_h, labels)
        self._set_gauge("estimated_annual_loss_usd", metrics.estimated_annual_loss_usd, labels)

    def record_trap_detection_result(
        self,
        predicted_failed: bool,
        actual_failed: bool,
    ) -> None:
        """
        Record trap detection result for accuracy tracking.

        Args:
            predicted_failed: Model prediction
            actual_failed: Actual trap status
        """
        now = datetime.now(timezone.utc)
        self._trap_detection_results.append((now, predicted_failed, actual_failed))

        # Update counters
        if predicted_failed and actual_failed:
            self._increment_counter("trap_detection_true_positives_total")
        elif predicted_failed and not actual_failed:
            self._increment_counter("trap_detection_false_positives_total")
        elif not predicted_failed and actual_failed:
            self._increment_counter("trap_detection_false_negatives_total")
        else:
            self._increment_counter("trap_detection_true_negatives_total")

    def record_desuperheater_metrics(self, metrics: DesuperheaterMetrics) -> None:
        """Record desuperheater metrics."""
        labels = {
            "system_id": metrics.system_id,
            "desuperheater_id": metrics.desuperheater_id,
        }

        self._set_gauge("desuperheater_inlet_temp_c", metrics.inlet_temperature_c, labels)
        self._set_gauge("desuperheater_outlet_temp_c", metrics.outlet_temperature_c, labels)
        self._set_gauge("desuperheater_target_temp_c", metrics.target_temperature_c, labels)
        self._set_gauge("desuperheater_spray_flow_kg_h", metrics.spray_water_flow_kg_h, labels)
        self._set_gauge("desuperheater_temp_deviation_c", metrics.temperature_deviation_c, labels)
        self._set_gauge("desuperheater_efficiency_percent", metrics.efficiency_percent, labels)

    def record_condensate_metrics(self, metrics: CondensateMetrics) -> None:
        """Record condensate recovery metrics."""
        labels = {"system_id": metrics.system_id}

        self._set_gauge("condensate_recovery_rate_percent", metrics.recovery_rate_percent, labels)
        self._set_gauge("flash_steam_recovered_percent", metrics.flash_steam_recovered_percent, labels)
        self._set_gauge("condensate_collected_kg_h", metrics.condensate_collected_kg_h, labels)
        self._set_gauge("condensate_returned_kg_h", metrics.condensate_returned_kg_h, labels)
        self._set_gauge("flash_steam_kg_h", metrics.flash_steam_kg_h, labels)
        self._set_gauge("condensate_flash_loss_kg_h", metrics.flash_loss_kg_h, labels)
        self._set_gauge("condensate_energy_recovered_kw", metrics.energy_recovered_kw, labels)

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

        # Filter computation times
        recent_computations = [
            x for x in self._computation_times if x[0] > cutoff
        ]
        computation_durations = [x[2] for x in recent_computations]

        # Calculate percentiles
        p50, p95, p99 = 0.0, 0.0, 0.0
        if computation_durations:
            sorted_durations = sorted(computation_durations)
            n = len(sorted_durations)
            p50 = sorted_durations[int(n * 0.5)] if n > 0 else 0.0
            p95 = sorted_durations[int(n * 0.95)] if n > 0 else 0.0
            p99 = sorted_durations[int(n * 0.99)] if n > 0 else 0.0

        # Filter optimization results
        recent_optimizations = [
            x for x in self._optimization_results if x[0] > cutoff
        ]
        successful_optimizations = [
            x for x in recent_optimizations if x[1].get("feasible", False)
        ]
        total_savings = sum(
            x[1].get("cost_savings_usd", 0.0) for x in recent_optimizations
        )

        # Filter recommendations
        recent_recommendations = [
            x for x in self._recommendation_outcomes if x[0] > cutoff
        ]
        accepted = len([x for x in recent_recommendations if x[2] == "accepted"])
        rejected = len([x for x in recent_recommendations if x[2] == "rejected"])
        total_recs = accepted + rejected
        acceptance_rate = (accepted / total_recs * 100) if total_recs > 0 else 0.0

        # Filter API requests
        recent_api_requests = [
            x for x in self._api_requests if x[0] > cutoff
        ]
        api_errors = len([x for x in recent_api_requests if x[2] >= 400])
        api_latencies = [x[3] for x in recent_api_requests]
        avg_api_latency = (
            sum(api_latencies) / len(api_latencies)
            if api_latencies else 0.0
        )

        # Calculate trap detection accuracy
        recent_trap_detections = [
            x for x in self._trap_detection_results if x[0] > cutoff
        ]
        trap_accuracy = 0.0
        trap_failures = 0
        if recent_trap_detections:
            correct = len([
                x for x in recent_trap_detections if x[1] == x[2]
            ])
            trap_accuracy = correct / len(recent_trap_detections) * 100
            trap_failures = len([x for x in recent_trap_detections if x[2]])

        return MetricsSummary(
            time_window_start=cutoff,
            time_window_end=now,
            system_id=self.system_id,
            total_computations=len(recent_computations),
            avg_computation_time_ms=(
                sum(computation_durations) / len(computation_durations)
                if computation_durations else 0.0
            ),
            p50_computation_time_ms=p50,
            p95_computation_time_ms=p95,
            p99_computation_time_ms=p99,
            optimizations_run=len(recent_optimizations),
            optimizations_successful=len(successful_optimizations),
            total_savings_usd=total_savings,
            recommendations_generated=len(recent_recommendations),
            recommendations_accepted=accepted,
            recommendations_rejected=rejected,
            acceptance_rate_percent=acceptance_rate,
            api_requests_total=len(recent_api_requests),
            api_errors_total=api_errors,
            avg_api_latency_ms=avg_api_latency,
            trap_failures_detected=trap_failures,
            trap_detection_accuracy_percent=trap_accuracy,
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
            for bucket in self.DEFAULT_LATENCY_BUCKETS:
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
            lines.append(f"{name}_sum{labels_part} {sum(values):.2f}")
            lines.append(f"{name}_count{labels_part} {len(values)}")
            lines.append("")

        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
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
        self._gauges.clear()
        self._counters.clear()
        self._histograms.clear()
        self._computation_times.clear()
        self._optimization_results.clear()
        self._recommendation_outcomes.clear()
        self._api_requests.clear()
        self._steam_kpis.clear()
        self._trap_detection_results.clear()
        self._register_standard_metrics()
        logger.info("Metrics reset")
