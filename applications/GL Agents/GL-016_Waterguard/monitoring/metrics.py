"""
Prometheus Metrics for GL-016 Waterguard

This module provides Prometheus-style metrics collection for the Waterguard
boiler water chemistry optimization agent. Metrics cover chemistry,
compliance, control, savings, and system performance.

Metric Categories:
    - Chemistry metrics: conductivity, pH, silica, phosphate, DO, etc.
    - Compliance metrics: constraint distance, violation counts
    - Control metrics: blowdown rate, dosing rate, cycles of concentration
    - Savings metrics: water saved, energy saved, chemical saved
    - System metrics: recommendations generated, commands executed, latencies

Example:
    >>> metrics = WaterguardMetrics()
    >>> metrics.record_chemistry_reading("boiler-001", "conductivity", 1250.5, "uS/cm")
    >>> metrics.record_constraint_distance("boiler-001", "conductivity_max", 0.85)
    >>> metrics.record_water_savings("boiler-001", 150.5)
    >>> print(metrics.export())

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC TYPES
# =============================================================================

class Counter:
    """Prometheus-style counter metric (monotonically increasing)."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """
        Initialize counter.

        Args:
            name: Metric name
            description: Metric description/help text
            labels: Label names for this metric
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **label_values) -> None:
        """
        Increment counter.

        Args:
            amount: Amount to increment (must be positive)
            **label_values: Label values
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented")
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def get(self, **label_values) -> float:
        """Get counter value for given labels."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        for key, value in self._values.items():
            if self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Gauge:
    """Prometheus-style gauge metric (can increase or decrease)."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """
        Initialize gauge.

        Args:
            name: Metric name
            description: Metric description/help text
            labels: Label names for this metric
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}

    def set(self, value: float, **label_values) -> None:
        """Set gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = value

    def inc(self, amount: float = 1.0, **label_values) -> None:
        """Increment gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount: float = 1.0, **label_values) -> None:
        """Decrement gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = self._values.get(key, 0) - amount

    def get(self, **label_values) -> float:
        """Get gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        for key, value in self._values.items():
            if self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Histogram:
    """Prometheus-style histogram metric for distributions."""

    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize histogram.

        Args:
            name: Metric name
            description: Metric description/help text
            buckets: Histogram bucket boundaries
            labels: Label names for this metric
        """
        self.name = name
        self.description = description
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.labels = labels or []
        self._values: Dict[tuple, List[float]] = {}

    def observe(self, value: float, **label_values) -> None:
        """Record an observation."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]

        for key, observations in self._values.items():
            label_prefix = ""
            if self.labels:
                label_prefix = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                ) + ","

            # Calculate bucket counts
            for bucket in self.buckets:
                count = sum(1 for o in observations if o <= bucket)
                lines.append(
                    f'{self.name}_bucket{{{label_prefix}le="{bucket}"}} {count}'
                )
            lines.append(
                f'{self.name}_bucket{{{label_prefix}le="+Inf"}} {len(observations)}'
            )

            # Sum and count
            label_str = label_prefix[:-1] if label_prefix else ""
            lines.append(
                f"{self.name}_sum{{{label_str}}} {sum(observations)}"
            )
            lines.append(
                f"{self.name}_count{{{label_str}}} {len(observations)}"
            )

        return "\n".join(lines)


class Summary:
    """Prometheus-style summary metric for quantiles."""

    def __init__(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize summary.

        Args:
            name: Metric name
            description: Metric description/help text
            quantiles: Quantiles to calculate
            labels: Label names for this metric
        """
        self.name = name
        self.description = description
        self.quantiles = quantiles or [0.5, 0.9, 0.99]
        self.labels = labels or []
        self._values: Dict[tuple, List[float]] = {}

    def observe(self, value: float, **label_values) -> None:
        """Record an observation."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        if key not in self._values:
            self._values[key] = []
        self._values[key].append(value)

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} summary",
        ]

        for key, observations in self._values.items():
            if not observations:
                continue

            label_prefix = ""
            if self.labels:
                label_prefix = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                ) + ","

            sorted_obs = sorted(observations)
            n = len(sorted_obs)

            for quantile in self.quantiles:
                idx = int(quantile * n)
                idx = min(idx, n - 1)
                value = sorted_obs[idx]
                lines.append(
                    f'{self.name}{{quantile="{quantile}",{label_prefix[:-1]}}} {value}'
                )

            label_str = label_prefix[:-1] if label_prefix else ""
            lines.append(
                f"{self.name}_sum{{{label_str}}} {sum(observations)}"
            )
            lines.append(
                f"{self.name}_count{{{label_str}}} {len(observations)}"
            )

        return "\n".join(lines)


# =============================================================================
# WATERGUARD METRICS COLLECTOR
# =============================================================================

class WaterguardMetrics:
    """
    Prometheus metrics collector for Waterguard water chemistry agent.

    Collects and exports metrics for monitoring and observability of
    water chemistry operations, compliance, and savings.

    Attributes:
        prefix: Metric name prefix

    Example:
        >>> metrics = WaterguardMetrics()
        >>> metrics.record_chemistry_reading("boiler-001", "conductivity", 1250.5, "uS/cm")
        >>> metrics.record_constraint_distance("boiler-001", "conductivity_max", 0.85)
    """

    def __init__(self, prefix: str = "greenlang_waterguard"):
        """
        Initialize the metrics collector.

        Args:
            prefix: Metric name prefix (default: greenlang_waterguard)
        """
        self.prefix = prefix

        # Chemistry metrics (Gauges - current values)
        self.conductivity = Gauge(
            f"{prefix}_conductivity_us_cm",
            "Boiler water conductivity in uS/cm",
            labels=["asset_id"],
        )
        self.ph = Gauge(
            f"{prefix}_ph",
            "Boiler water pH",
            labels=["asset_id"],
        )
        self.silica = Gauge(
            f"{prefix}_silica_ppm",
            "Boiler water silica in ppm",
            labels=["asset_id"],
        )
        self.phosphate = Gauge(
            f"{prefix}_phosphate_ppm",
            "Boiler water phosphate in ppm",
            labels=["asset_id"],
        )
        self.dissolved_oxygen = Gauge(
            f"{prefix}_dissolved_oxygen_ppb",
            "Feedwater dissolved oxygen in ppb",
            labels=["asset_id"],
        )
        self.tds = Gauge(
            f"{prefix}_tds_ppm",
            "Total dissolved solids in ppm",
            labels=["asset_id"],
        )
        self.alkalinity = Gauge(
            f"{prefix}_alkalinity_ppm",
            "Boiler water alkalinity in ppm CaCO3",
            labels=["asset_id"],
        )
        self.hardness = Gauge(
            f"{prefix}_hardness_ppm",
            "Feedwater hardness in ppm CaCO3",
            labels=["asset_id"],
        )
        self.cycles_of_concentration = Gauge(
            f"{prefix}_cycles_of_concentration",
            "Current cycles of concentration",
            labels=["asset_id"],
        )

        # Compliance metrics
        self.constraint_distance = Gauge(
            f"{prefix}_constraint_distance",
            "Normalized distance to constraint (0-1, 1=at limit)",
            labels=["asset_id", "constraint"],
        )
        self.constraint_violations_total = Counter(
            f"{prefix}_constraint_violations_total",
            "Total constraint violations",
            labels=["asset_id", "constraint", "severity"],
        )
        self.compliance_score = Gauge(
            f"{prefix}_compliance_score_pct",
            "Overall compliance score percentage",
            labels=["asset_id"],
        )

        # Control metrics
        self.blowdown_rate = Gauge(
            f"{prefix}_blowdown_rate_pct",
            "Current blowdown rate percentage",
            labels=["asset_id"],
        )
        self.blowdown_valve_position = Gauge(
            f"{prefix}_blowdown_valve_position_pct",
            "Blowdown valve position percentage",
            labels=["asset_id"],
        )
        self.dosing_rate = Gauge(
            f"{prefix}_dosing_rate_ml_min",
            "Chemical dosing rate in mL/min",
            labels=["asset_id", "chemical"],
        )
        self.dosing_pump_running = Gauge(
            f"{prefix}_dosing_pump_running",
            "Dosing pump running status (1=on, 0=off)",
            labels=["asset_id", "pump_id"],
        )

        # Savings metrics (Counters - cumulative)
        self.water_saved_gallons = Counter(
            f"{prefix}_water_saved_gallons_total",
            "Total water saved in gallons",
            labels=["asset_id"],
        )
        self.energy_saved_mmbtu = Counter(
            f"{prefix}_energy_saved_mmbtu_total",
            "Total energy saved in MMBtu",
            labels=["asset_id"],
        )
        self.chemical_saved_usd = Counter(
            f"{prefix}_chemical_saved_usd_total",
            "Total chemical cost saved in USD",
            labels=["asset_id"],
        )
        self.water_saved_rate = Gauge(
            f"{prefix}_water_saved_rate_gph",
            "Current water savings rate in gallons per hour",
            labels=["asset_id"],
        )
        self.energy_saved_rate = Gauge(
            f"{prefix}_energy_saved_rate_mmbtu_hr",
            "Current energy savings rate in MMBtu/hr",
            labels=["asset_id"],
        )

        # System metrics
        self.recommendations_generated_total = Counter(
            f"{prefix}_recommendations_generated_total",
            "Total recommendations generated",
            labels=["asset_id", "type", "status"],
        )
        self.commands_executed_total = Counter(
            f"{prefix}_commands_executed_total",
            "Total commands executed",
            labels=["asset_id", "type", "status"],
        )
        self.calculations_total = Counter(
            f"{prefix}_calculations_total",
            "Total chemistry calculations performed",
            labels=["asset_id"],
        )

        # Latency histograms
        self.calculation_latency = Histogram(
            f"{prefix}_calculation_latency_seconds",
            "Chemistry calculation latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            labels=["asset_id"],
        )
        self.recommendation_latency = Histogram(
            f"{prefix}_recommendation_latency_seconds",
            "Recommendation generation latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            labels=["asset_id"],
        )
        self.command_latency = Histogram(
            f"{prefix}_command_latency_seconds",
            "Command execution latency in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            labels=["asset_id"],
        )

        # Analyzer status
        self.analyzer_status = Gauge(
            f"{prefix}_analyzer_status",
            "Analyzer status (1=online, 0=offline)",
            labels=["asset_id", "analyzer_id", "parameter"],
        )
        self.analyzer_last_reading = Gauge(
            f"{prefix}_analyzer_last_reading_timestamp",
            "Timestamp of last analyzer reading (unix epoch)",
            labels=["asset_id", "analyzer_id"],
        )

        logger.info(f"WaterguardMetrics initialized with prefix: {prefix}")

    def record_chemistry_reading(
        self,
        asset_id: str,
        parameter: str,
        value: float,
        unit: str,
    ) -> None:
        """
        Record a chemistry parameter reading.

        Args:
            asset_id: Asset identifier
            parameter: Parameter name (conductivity, ph, silica, etc.)
            value: Parameter value
            unit: Engineering unit
        """
        if parameter == "conductivity":
            self.conductivity.set(value, asset_id=asset_id)
        elif parameter == "ph":
            self.ph.set(value, asset_id=asset_id)
        elif parameter == "silica":
            self.silica.set(value, asset_id=asset_id)
        elif parameter == "phosphate":
            self.phosphate.set(value, asset_id=asset_id)
        elif parameter == "dissolved_oxygen":
            self.dissolved_oxygen.set(value, asset_id=asset_id)
        elif parameter == "tds":
            self.tds.set(value, asset_id=asset_id)
        elif parameter == "alkalinity":
            self.alkalinity.set(value, asset_id=asset_id)
        elif parameter == "hardness":
            self.hardness.set(value, asset_id=asset_id)
        elif parameter == "cycles":
            self.cycles_of_concentration.set(value, asset_id=asset_id)

    def record_constraint_distance(
        self,
        asset_id: str,
        constraint: str,
        distance: float,
    ) -> None:
        """
        Record distance to a constraint (0-1 normalized).

        Args:
            asset_id: Asset identifier
            constraint: Constraint name
            distance: Distance to constraint (0=far, 1=at limit)
        """
        self.constraint_distance.set(distance, asset_id=asset_id, constraint=constraint)

    def record_constraint_violation(
        self,
        asset_id: str,
        constraint: str,
        severity: str,
    ) -> None:
        """
        Record a constraint violation.

        Args:
            asset_id: Asset identifier
            constraint: Constraint that was violated
            severity: Violation severity (info, warning, critical)
        """
        self.constraint_violations_total.inc(
            asset_id=asset_id, constraint=constraint, severity=severity
        )

    def record_compliance_score(self, asset_id: str, score: float) -> None:
        """Record overall compliance score percentage."""
        self.compliance_score.set(score, asset_id=asset_id)

    def record_blowdown_rate(self, asset_id: str, rate: float) -> None:
        """Record current blowdown rate percentage."""
        self.blowdown_rate.set(rate, asset_id=asset_id)

    def record_dosing_rate(
        self,
        asset_id: str,
        chemical: str,
        rate: float,
    ) -> None:
        """
        Record chemical dosing rate.

        Args:
            asset_id: Asset identifier
            chemical: Chemical name (phosphate, oxygen_scavenger, amine)
            rate: Dosing rate in mL/min
        """
        self.dosing_rate.set(rate, asset_id=asset_id, chemical=chemical)

    def record_water_savings(self, asset_id: str, gallons: float) -> None:
        """Record water savings (incremental)."""
        self.water_saved_gallons.inc(gallons, asset_id=asset_id)

    def record_energy_savings(self, asset_id: str, mmbtu: float) -> None:
        """Record energy savings (incremental)."""
        self.energy_saved_mmbtu.inc(mmbtu, asset_id=asset_id)

    def record_chemical_savings(self, asset_id: str, usd: float) -> None:
        """Record chemical cost savings (incremental)."""
        self.chemical_saved_usd.inc(usd, asset_id=asset_id)

    def record_recommendation(
        self,
        asset_id: str,
        rec_type: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """
        Record a recommendation generation.

        Args:
            asset_id: Asset identifier
            rec_type: Recommendation type
            status: Status (generated, implemented, rejected)
            latency_seconds: Generation latency
        """
        self.recommendations_generated_total.inc(
            asset_id=asset_id, type=rec_type, status=status
        )
        self.recommendation_latency.observe(latency_seconds, asset_id=asset_id)

    def record_command(
        self,
        asset_id: str,
        cmd_type: str,
        status: str,
        latency_seconds: float,
    ) -> None:
        """
        Record a command execution.

        Args:
            asset_id: Asset identifier
            cmd_type: Command type
            status: Status (success, failed, timeout)
            latency_seconds: Execution latency
        """
        self.commands_executed_total.inc(
            asset_id=asset_id, type=cmd_type, status=status
        )
        self.command_latency.observe(latency_seconds, asset_id=asset_id)

    def record_calculation(self, asset_id: str, latency_seconds: float) -> None:
        """
        Record a chemistry calculation.

        Args:
            asset_id: Asset identifier
            latency_seconds: Calculation latency
        """
        self.calculations_total.inc(asset_id=asset_id)
        self.calculation_latency.observe(latency_seconds, asset_id=asset_id)

    def record_analyzer_status(
        self,
        asset_id: str,
        analyzer_id: str,
        parameter: str,
        online: bool,
    ) -> None:
        """
        Record analyzer status.

        Args:
            asset_id: Asset identifier
            analyzer_id: Analyzer identifier
            parameter: Parameter measured by analyzer
            online: Whether analyzer is online
        """
        self.analyzer_status.set(
            1 if online else 0,
            asset_id=asset_id,
            analyzer_id=analyzer_id,
            parameter=parameter,
        )
        if online:
            self.analyzer_last_reading.set(
                time.time(),
                asset_id=asset_id,
                analyzer_id=analyzer_id,
            )

    def export(self) -> str:
        """Export all metrics in Prometheus text format."""
        metrics = [
            # Chemistry
            self.conductivity,
            self.ph,
            self.silica,
            self.phosphate,
            self.dissolved_oxygen,
            self.tds,
            self.alkalinity,
            self.hardness,
            self.cycles_of_concentration,
            # Compliance
            self.constraint_distance,
            self.constraint_violations_total,
            self.compliance_score,
            # Control
            self.blowdown_rate,
            self.blowdown_valve_position,
            self.dosing_rate,
            self.dosing_pump_running,
            # Savings
            self.water_saved_gallons,
            self.energy_saved_mmbtu,
            self.chemical_saved_usd,
            self.water_saved_rate,
            self.energy_saved_rate,
            # System
            self.recommendations_generated_total,
            self.commands_executed_total,
            self.calculations_total,
            self.calculation_latency,
            self.recommendation_latency,
            self.command_latency,
            # Analyzers
            self.analyzer_status,
            self.analyzer_last_reading,
        ]

        return "\n\n".join(m.export() for m in metrics if m._values)


# =============================================================================
# METRICS HTTP HANDLER
# =============================================================================

class MetricsHTTPHandler:
    """
    HTTP handler for Prometheus metrics endpoint.

    In production, this would be integrated with an HTTP server like FastAPI.
    """

    def __init__(self, metrics: WaterguardMetrics, port: int = 9091):
        """
        Initialize the metrics HTTP handler.

        Args:
            metrics: WaterguardMetrics instance
            port: HTTP port for metrics endpoint
        """
        self.metrics = metrics
        self.port = port
        self._running = False

    async def handle_metrics(self) -> str:
        """Handle /metrics endpoint."""
        return self.metrics.export()

    async def start(self) -> None:
        """Start the metrics HTTP server."""
        self._running = True
        logger.info(f"Metrics endpoint available at http://localhost:{self.port}/metrics")

    async def stop(self) -> None:
        """Stop the metrics HTTP server."""
        self._running = False
        logger.info("Metrics endpoint stopped")


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """Context manager for timing operations and recording to metrics."""

    def __init__(
        self,
        metrics: WaterguardMetrics,
        metric_type: str,
        asset_id: str,
        **extra_labels,
    ):
        """
        Initialize the timer.

        Args:
            metrics: WaterguardMetrics instance
            metric_type: Type of metric (calculation, recommendation, command)
            asset_id: Asset identifier
            **extra_labels: Additional labels for the metric
        """
        self.metrics = metrics
        self.metric_type = metric_type
        self.asset_id = asset_id
        self.extra_labels = extra_labels
        self._start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record metric."""
        if self._start_time is not None:
            duration = time.perf_counter() - self._start_time

            if self.metric_type == "calculation":
                self.metrics.record_calculation(self.asset_id, duration)
            elif self.metric_type == "recommendation":
                self.metrics.record_recommendation(
                    self.asset_id,
                    self.extra_labels.get("type", "unknown"),
                    "success" if exc_type is None else "failed",
                    duration,
                )
            elif self.metric_type == "command":
                self.metrics.record_command(
                    self.asset_id,
                    self.extra_labels.get("type", "unknown"),
                    "success" if exc_type is None else "failed",
                    duration,
                )
