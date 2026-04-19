"""
GL-002 FLAMEGUARD - Metrics Collection

Prometheus-format metrics for boiler monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: str = "gauge"  # gauge, counter, histogram, summary


@dataclass
class BoilerMetrics:
    """Boiler operating metrics."""
    boiler_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Pressure
    drum_pressure_psig: float = 0.0
    fuel_pressure_psig: float = 0.0
    air_pressure_in_wc: float = 0.0

    # Level
    drum_level_inches: float = 0.0

    # Flow
    steam_flow_klb_hr: float = 0.0
    feedwater_flow_klb_hr: float = 0.0
    fuel_flow_scfh: float = 0.0
    blowdown_flow_klb_hr: float = 0.0

    # Temperature
    steam_temperature_f: float = 0.0
    feedwater_temperature_f: float = 0.0
    flue_gas_temperature_f: float = 0.0
    ambient_temperature_f: float = 70.0

    # Operating state
    load_percent: float = 0.0
    firing: bool = False
    state: str = "offline"


@dataclass
class CombustionMetrics:
    """Combustion analysis metrics."""
    boiler_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # O2 control
    o2_percent: float = 0.0
    o2_setpoint: float = 0.0
    o2_error: float = 0.0

    # CO monitoring
    co_ppm: float = 0.0
    co_limit_ppm: float = 400.0

    # Excess air
    excess_air_percent: float = 0.0
    stoichiometric_ratio: float = 0.0

    # Air-fuel ratio
    air_fuel_ratio: float = 0.0
    air_fuel_ratio_target: float = 0.0


@dataclass
class EfficiencyMetrics:
    """Efficiency calculation metrics."""
    boiler_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Efficiency
    gross_efficiency_percent: float = 0.0
    net_efficiency_percent: float = 0.0
    fuel_efficiency_percent: float = 0.0

    # Losses
    stack_loss_percent: float = 0.0
    radiation_loss_percent: float = 0.0
    blowdown_loss_percent: float = 0.0
    unaccounted_loss_percent: float = 0.0

    # Heat
    heat_input_mmbtu_hr: float = 0.0
    heat_output_mmbtu_hr: float = 0.0


@dataclass
class SafetyMetrics:
    """Safety system metrics."""
    boiler_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # BMS state
    bms_state: str = "offline"
    flame_proven: bool = False
    flame_signal_percent: float = 0.0

    # Interlocks
    interlocks_normal: int = 0
    interlocks_alarm: int = 0
    interlocks_trip: int = 0
    interlocks_bypassed: int = 0

    # Trip tracking
    total_trips: int = 0
    trips_last_24h: int = 0
    time_since_last_trip_hours: float = 0.0


class MetricsCollector:
    """
    Prometheus-compatible metrics collector.

    Features:
    - Gauge, counter, histogram, summary metrics
    - Label support
    - Metric aggregation
    - Prometheus text format export
    """

    def __init__(self, namespace: str = "flameguard") -> None:
        self.namespace = namespace
        self._metrics: Dict[str, MetricValue] = {}
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._start_time = time.time()

        # Register standard metrics
        self._register_standard_metrics()

        logger.info(f"MetricsCollector initialized: {namespace}")

    def _register_standard_metrics(self) -> None:
        """Register standard application metrics."""
        self.set_gauge("info", 1.0, {"version": "1.0.0", "agent": "GL-002"})

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge metric value."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._labels_to_key(labels)

        self._metrics[f"{full_name}{label_key}"] = MetricValue(
            name=full_name,
            value=value,
            labels=labels,
            metric_type="gauge",
        )

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment counter metric."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._labels_to_key(labels)
        key = f"{full_name}{label_key}"

        if key not in self._counters:
            self._counters[key] = 0.0
        self._counters[key] += value

        self._metrics[key] = MetricValue(
            name=full_name,
            value=self._counters[key],
            labels=labels,
            metric_type="counter",
        )

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        """Add observation to histogram."""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        label_key = self._labels_to_key(labels)
        key = f"{full_name}{label_key}"

        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

        # Keep only last 1000 observations
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels to string key."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def record_boiler_metrics(self, metrics: BoilerMetrics) -> None:
        """Record all boiler operating metrics."""
        labels = {"boiler_id": metrics.boiler_id}

        # Pressure
        self.set_gauge("drum_pressure_psig", metrics.drum_pressure_psig, labels)
        self.set_gauge("fuel_pressure_psig", metrics.fuel_pressure_psig, labels)
        self.set_gauge("air_pressure_in_wc", metrics.air_pressure_in_wc, labels)

        # Level
        self.set_gauge("drum_level_inches", metrics.drum_level_inches, labels)

        # Flow
        self.set_gauge("steam_flow_klb_hr", metrics.steam_flow_klb_hr, labels)
        self.set_gauge("feedwater_flow_klb_hr", metrics.feedwater_flow_klb_hr, labels)
        self.set_gauge("fuel_flow_scfh", metrics.fuel_flow_scfh, labels)

        # Temperature
        self.set_gauge("steam_temperature_f", metrics.steam_temperature_f, labels)
        self.set_gauge("feedwater_temperature_f", metrics.feedwater_temperature_f, labels)
        self.set_gauge("flue_gas_temperature_f", metrics.flue_gas_temperature_f, labels)

        # Operating state
        self.set_gauge("load_percent", metrics.load_percent, labels)
        self.set_gauge("firing", 1.0 if metrics.firing else 0.0, labels)

    def record_combustion_metrics(self, metrics: CombustionMetrics) -> None:
        """Record combustion analysis metrics."""
        labels = {"boiler_id": metrics.boiler_id}

        self.set_gauge("o2_percent", metrics.o2_percent, labels)
        self.set_gauge("o2_setpoint", metrics.o2_setpoint, labels)
        self.set_gauge("o2_error", metrics.o2_error, labels)
        self.set_gauge("co_ppm", metrics.co_ppm, labels)
        self.set_gauge("excess_air_percent", metrics.excess_air_percent, labels)
        self.set_gauge("air_fuel_ratio", metrics.air_fuel_ratio, labels)

    def record_efficiency_metrics(self, metrics: EfficiencyMetrics) -> None:
        """Record efficiency calculation metrics."""
        labels = {"boiler_id": metrics.boiler_id}

        self.set_gauge("gross_efficiency_percent", metrics.gross_efficiency_percent, labels)
        self.set_gauge("net_efficiency_percent", metrics.net_efficiency_percent, labels)
        self.set_gauge("stack_loss_percent", metrics.stack_loss_percent, labels)
        self.set_gauge("radiation_loss_percent", metrics.radiation_loss_percent, labels)
        self.set_gauge("heat_input_mmbtu_hr", metrics.heat_input_mmbtu_hr, labels)
        self.set_gauge("heat_output_mmbtu_hr", metrics.heat_output_mmbtu_hr, labels)

    def record_safety_metrics(self, metrics: SafetyMetrics) -> None:
        """Record safety system metrics."""
        labels = {"boiler_id": metrics.boiler_id}

        self.set_gauge("flame_proven", 1.0 if metrics.flame_proven else 0.0, labels)
        self.set_gauge("flame_signal_percent", metrics.flame_signal_percent, labels)
        self.set_gauge("interlocks_normal", float(metrics.interlocks_normal), labels)
        self.set_gauge("interlocks_alarm", float(metrics.interlocks_alarm), labels)
        self.set_gauge("interlocks_trip", float(metrics.interlocks_trip), labels)
        self.set_gauge("interlocks_bypassed", float(metrics.interlocks_bypassed), labels)
        self.set_gauge("total_trips", float(metrics.total_trips), labels)

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Add uptime
        uptime = time.time() - self._start_time
        lines.append(f"# TYPE {self.namespace}_uptime_seconds gauge")
        lines.append(f"{self.namespace}_uptime_seconds {uptime:.2f}")

        # Group by metric name
        metrics_by_name: Dict[str, List[MetricValue]] = {}
        for metric in self._metrics.values():
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric)

        # Output each metric
        for name, metrics in sorted(metrics_by_name.items()):
            metric_type = metrics[0].metric_type
            lines.append(f"# TYPE {name} {metric_type}")

            for m in metrics:
                label_str = ""
                if m.labels:
                    parts = [f'{k}="{v}"' for k, v in sorted(m.labels.items())]
                    label_str = "{" + ",".join(parts) + "}"
                lines.append(f"{m.name}{label_str} {m.value}")

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
            for key, m in self._metrics.items()
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._histograms.clear()
        self._register_standard_metrics()
