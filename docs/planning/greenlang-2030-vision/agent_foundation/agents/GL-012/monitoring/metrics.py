# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Steam Quality Controller Metrics
==================================================

Production-ready Prometheus metrics collection for the GL-012 STEAMQUAL
SteamQualityController agent. Provides comprehensive observability for
steam quality monitoring, control actions, and operational performance.

This module implements Prometheus-style metrics following GreenLang's
standardized monitoring patterns with agent-specific metrics for:
- Steam quality parameters (dryness, pressure, temperature)
- Control system performance (valves, desuperheaters)
- Quality calculations and violations
- Cache performance and error tracking

Metrics Categories:
1. Steam Quality Gauges (6 metrics)
2. Control System Metrics (4 metrics)
3. Calculation Counters/Histograms (4 metrics)
4. Violation and Alert Counters (3 metrics)
5. Cache and Error Metrics (2 metrics)

Total: 19 agent-specific metrics + standard baseline metrics

Usage:
    >>> from monitoring.metrics import MetricsCollector
    >>>
    >>> collector = MetricsCollector(
    ...     agent_id="GL-012",
    ...     agent_name="SteamQualityController"
    ... )
    >>>
    >>> # Record quality calculation
    >>> collector.record_quality_calculation(duration=0.045, result={"dryness": 0.98})
    >>>
    >>> # Record control action
    >>> collector.record_control_action(action_type="valve_adjustment", success=True)
    >>>
    >>> # Export Prometheus metrics
    >>> prometheus_text = collector.export_prometheus()

Author: GreenLang Team
License: Proprietary
"""

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


# Try to import Prometheus client
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        start_http_server,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Using stub implementations.")

    # Stub implementations for when Prometheus is not available
    class Counter:
        """Stub Counter for when Prometheus is unavailable."""
        def __init__(self, *args, **kwargs):
            self._value = 0
            self._labels = {}

        def inc(self, amount: float = 1) -> None:
            self._value += amount

        def labels(self, *args, **kwargs):
            return self

    class Gauge:
        """Stub Gauge for when Prometheus is unavailable."""
        def __init__(self, *args, **kwargs):
            self._value = 0

        def set(self, value: float) -> None:
            self._value = value

        def inc(self, amount: float = 1) -> None:
            self._value += amount

        def dec(self, amount: float = 1) -> None:
            self._value -= amount

        def labels(self, *args, **kwargs):
            return self

    class Histogram:
        """Stub Histogram for when Prometheus is unavailable."""
        def __init__(self, *args, **kwargs):
            self._observations = []

        def observe(self, value: float) -> None:
            self._observations.append(value)

        def labels(self, *args, **kwargs):
            return self

    class Summary:
        """Stub Summary for when Prometheus is unavailable."""
        def __init__(self, *args, **kwargs):
            self._observations = []

        def observe(self, value: float) -> None:
            self._observations.append(value)

        def labels(self, *args, **kwargs):
            return self

    class Info:
        """Stub Info for when Prometheus is unavailable."""
        def __init__(self, *args, **kwargs):
            self._info = {}

        def info(self, data: Dict[str, str]) -> None:
            self._info = data

    class CollectorRegistry:
        """Stub CollectorRegistry for when Prometheus is unavailable."""
        pass

    def generate_latest(registry=None) -> bytes:
        return b""

    CONTENT_TYPE_LATEST = "text/plain"

    def start_http_server(port: int, registry=None) -> None:
        pass


class MetricType(Enum):
    """Types of metrics supported by the collector."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Single metric value with metadata for internal tracking."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "metric_type": self.metric_type.value,
        }


@dataclass
class OperationalState:
    """Current operational state of the steam quality controller."""
    steam_quality_index: float = 0.0
    steam_dryness_fraction: float = 0.0
    steam_pressure_bar: float = 0.0
    steam_temperature_c: float = 0.0
    desuperheater_injection_rate_kg_hr: float = 0.0
    control_valve_position_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MetricsBuffer:
    """Thread-safe buffer for collecting metric history."""

    def __init__(self, max_size: int = 10000):
        """
        Initialize metrics buffer.

        Args:
            max_size: Maximum number of metrics to store
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, metric: MetricValue) -> None:
        """
        Add metric to buffer thread-safely.

        Args:
            metric: Metric value to add
        """
        with self.lock:
            self.buffer.append(metric)

    def get_recent(self, seconds: int = 60) -> List[MetricValue]:
        """
        Get metrics from last N seconds.

        Args:
            seconds: Number of seconds to look back

        Returns:
            List of recent metric values
        """
        cutoff = time.time() - seconds
        with self.lock:
            return [m for m in self.buffer if m.timestamp >= cutoff]

    def clear_old(self, max_age_seconds: int = 3600) -> int:
        """
        Clear metrics older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age of metrics to keep

        Returns:
            Number of metrics cleared
        """
        cutoff = time.time() - max_age_seconds
        cleared = 0
        with self.lock:
            while self.buffer and self.buffer[0].timestamp < cutoff:
                self.buffer.popleft()
                cleared += 1
        return cleared

    def __len__(self) -> int:
        """Return current buffer size."""
        with self.lock:
            return len(self.buffer)


class MetricsCollector:
    """
    Prometheus-style metrics collector for GL-012 STEAMQUAL agent.

    Provides comprehensive metrics collection for steam quality monitoring
    including real-time gauges, calculation performance histograms, and
    violation counters. Supports both Prometheus export and internal
    metrics buffer for analysis.

    Attributes:
        agent_id: Agent identifier (GL-012)
        agent_name: Human-readable agent name
        registry: Prometheus collector registry
        buffer: Internal metrics buffer for history

    Example:
        >>> collector = MetricsCollector("GL-012", "SteamQualityController")
        >>> collector.update_operational_state(OperationalState(
        ...     steam_dryness_fraction=0.98,
        ...     steam_pressure_bar=10.5
        ... ))
        >>> collector.record_quality_calculation(0.045, {"quality": 0.98})
        >>> snapshot = collector.get_metrics_snapshot()
    """

    # Default histogram buckets for calculation duration
    DURATION_BUCKETS = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    )

    def __init__(
        self,
        agent_id: str = "GL-012",
        agent_name: str = "SteamQualityController",
        registry: Optional[CollectorRegistry] = None,
        buffer_size: int = 10000,
        enable_prometheus: bool = True,
    ):
        """
        Initialize the metrics collector.

        Args:
            agent_id: Agent identifier (default: GL-012)
            agent_name: Human-readable agent name
            registry: Optional Prometheus registry (creates new if None)
            buffer_size: Size of internal metrics buffer
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # Create metric prefix from agent_id (GL-012 -> gl012)
        self.metric_prefix = agent_id.lower().replace("-", "")

        # Initialize registry
        if self.enable_prometheus:
            self.registry = registry or CollectorRegistry()
        else:
            self.registry = None

        # Internal metrics buffer
        self.buffer = MetricsBuffer(max_size=buffer_size)

        # Thread safety
        self.lock = threading.Lock()

        # Track start time for uptime calculations
        self.start_time = time.time()

        # Current operational state
        self._current_state = OperationalState()

        # Internal counters for non-Prometheus mode
        self._internal_counters: Dict[str, float] = defaultdict(float)
        self._internal_gauges: Dict[str, float] = {}

        # Initialize all metrics
        self._init_steam_quality_metrics()
        self._init_control_metrics()
        self._init_calculation_metrics()
        self._init_violation_metrics()
        self._init_cache_error_metrics()
        self._init_agent_info()

        logger.info(
            f"MetricsCollector initialized for {agent_id} ({agent_name}) "
            f"[Prometheus: {self.enable_prometheus}]"
        )

    # =========================================================================
    # STEAM QUALITY METRICS (Gauges)
    # =========================================================================

    def _init_steam_quality_metrics(self) -> None:
        """Initialize steam quality gauge metrics."""
        if not self.enable_prometheus:
            return

        # Metric 1: Steam Quality Index (composite quality score 0-100)
        self.steam_quality_index = Gauge(
            f"{self.metric_prefix}_steam_quality_index",
            "Steam quality index (0-100 scale)",
            registry=self.registry
        )

        # Metric 2: Steam Dryness Fraction (0.0-1.0)
        self.steam_dryness_fraction = Gauge(
            f"{self.metric_prefix}_steam_dryness_fraction",
            "Steam dryness fraction (0.0=saturated liquid, 1.0=saturated vapor)",
            registry=self.registry
        )

        # Metric 3: Steam Pressure (bar)
        self.steam_pressure_bar = Gauge(
            f"{self.metric_prefix}_steam_pressure_bar",
            "Steam pressure in bar",
            registry=self.registry
        )

        # Metric 4: Steam Temperature (Celsius)
        self.steam_temperature_c = Gauge(
            f"{self.metric_prefix}_steam_temperature_c",
            "Steam temperature in Celsius",
            registry=self.registry
        )

        # Metric 5: Desuperheater Injection Rate (kg/hr)
        self.desuperheater_injection_rate_kg_hr = Gauge(
            f"{self.metric_prefix}_desuperheater_injection_rate_kg_hr",
            "Desuperheater water injection rate in kg/hr",
            registry=self.registry
        )

        # Metric 6: Control Valve Position (percent)
        self.control_valve_position_percent = Gauge(
            f"{self.metric_prefix}_control_valve_position_percent",
            "Control valve position (0-100%)",
            registry=self.registry
        )

    # =========================================================================
    # CONTROL SYSTEM METRICS
    # =========================================================================

    def _init_control_metrics(self) -> None:
        """Initialize control system metrics."""
        if not self.enable_prometheus:
            return

        # Metric 7: Control Actions Total (counter by action_type)
        self.control_actions_total = Counter(
            f"{self.metric_prefix}_control_actions_total",
            "Total control actions executed",
            ["action_type", "success"],
            registry=self.registry
        )

        # Metric 8: Control Action Duration (histogram)
        self.control_action_duration_seconds = Histogram(
            f"{self.metric_prefix}_control_action_duration_seconds",
            "Control action execution duration in seconds",
            ["action_type"],
            buckets=self.DURATION_BUCKETS,
            registry=self.registry
        )

        # Metric 9: Active Control Operations
        self.active_control_operations = Gauge(
            f"{self.metric_prefix}_active_control_operations",
            "Number of control operations currently in progress",
            ["operation_type"],
            registry=self.registry
        )

        # Metric 10: Control Setpoint Deviation
        self.control_setpoint_deviation = Gauge(
            f"{self.metric_prefix}_control_setpoint_deviation",
            "Current deviation from control setpoint",
            ["parameter"],
            registry=self.registry
        )

    # =========================================================================
    # CALCULATION METRICS
    # =========================================================================

    def _init_calculation_metrics(self) -> None:
        """Initialize calculation performance metrics."""
        if not self.enable_prometheus:
            return

        # Metric 11: Quality Calculations Total (counter)
        self.quality_calculations_total = Counter(
            f"{self.metric_prefix}_quality_calculations_total",
            "Total steam quality calculations performed",
            ["result_status"],
            registry=self.registry
        )

        # Metric 12: Quality Calculation Duration (histogram)
        self.quality_calculation_duration_seconds = Histogram(
            f"{self.metric_prefix}_quality_calculation_duration_seconds",
            "Steam quality calculation duration in seconds",
            buckets=self.DURATION_BUCKETS,
            registry=self.registry
        )

        # Metric 13: Calculation Errors Total
        self.calculation_errors_total = Counter(
            f"{self.metric_prefix}_calculation_errors_total",
            "Total calculation errors",
            ["error_type"],
            registry=self.registry
        )

        # Metric 14: Calculation Queue Depth
        self.calculation_queue_depth = Gauge(
            f"{self.metric_prefix}_calculation_queue_depth",
            "Number of calculations waiting in queue",
            registry=self.registry
        )

    # =========================================================================
    # VIOLATION AND ALERT METRICS
    # =========================================================================

    def _init_violation_metrics(self) -> None:
        """Initialize violation and alert tracking metrics."""
        if not self.enable_prometheus:
            return

        # Metric 15: Quality Violations Total (counter by violation_type)
        self.quality_violations_total = Counter(
            f"{self.metric_prefix}_quality_violations_total",
            "Total steam quality violations detected",
            ["violation_type"],
            registry=self.registry
        )

        # Metric 16: Steam Quality Alerts (counter by severity)
        self.steam_quality_alerts = Counter(
            f"{self.metric_prefix}_steam_quality_alerts",
            "Total steam quality alerts raised",
            ["severity", "alert_type"],
            registry=self.registry
        )

        # Metric 17: Active Alerts Gauge
        self.active_alerts_gauge = Gauge(
            f"{self.metric_prefix}_active_alerts",
            "Number of currently active alerts",
            ["severity"],
            registry=self.registry
        )

    # =========================================================================
    # CACHE AND ERROR METRICS
    # =========================================================================

    def _init_cache_error_metrics(self) -> None:
        """Initialize cache performance and error metrics."""
        if not self.enable_prometheus:
            return

        # Metric 18: Cache Hit Ratio (gauge 0.0-1.0)
        self.cache_hit_ratio = Gauge(
            f"{self.metric_prefix}_cache_hit_ratio",
            "Cache hit ratio (0.0-1.0)",
            registry=self.registry
        )

        # Metric 19: Agent Errors Total (counter)
        self.agent_errors_total = Counter(
            f"{self.metric_prefix}_agent_errors_total",
            "Total agent errors encountered",
            ["error_category", "component"],
            registry=self.registry
        )

        # Additional cache metrics
        self.cache_hits_total = Counter(
            f"{self.metric_prefix}_cache_hits_total",
            "Total cache hits",
            registry=self.registry
        )

        self.cache_misses_total = Counter(
            f"{self.metric_prefix}_cache_misses_total",
            "Total cache misses",
            registry=self.registry
        )

    # =========================================================================
    # AGENT INFO
    # =========================================================================

    def _init_agent_info(self) -> None:
        """Initialize agent information metric."""
        if not self.enable_prometheus:
            return

        self.agent_info = Info(
            f"{self.metric_prefix}_agent_info",
            f"{self.agent_id} STEAMQUAL agent information",
            registry=self.registry
        )

        # Set initial info
        self.agent_info.info({
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "codename": "STEAMQUAL",
            "version": "1.0.0",
            "domain": "steam_quality_control",
        })

    # =========================================================================
    # RECORDING METHODS
    # =========================================================================

    def record_quality_calculation(
        self,
        duration: float,
        result: Dict[str, Any],
        success: bool = True
    ) -> None:
        """
        Record a steam quality calculation.

        Args:
            duration: Calculation duration in seconds
            result: Calculation result dictionary
            success: Whether calculation was successful
        """
        status = "success" if success else "error"

        if self.enable_prometheus:
            self.quality_calculations_total.labels(result_status=status).inc()
            self.quality_calculation_duration_seconds.observe(duration)
        else:
            self._internal_counters[f"quality_calculations_{status}"] += 1

        # Add to internal buffer
        self.buffer.add(MetricValue(
            name="quality_calculation",
            value=duration,
            labels={"status": status},
            metric_type=MetricType.HISTOGRAM
        ))

        logger.debug(
            f"Recorded quality calculation: duration={duration:.4f}s, "
            f"status={status}"
        )

    def record_control_action(
        self,
        action_type: str,
        success: bool,
        duration: Optional[float] = None
    ) -> None:
        """
        Record a control action execution.

        Args:
            action_type: Type of control action (e.g., valve_adjustment,
                         desuperheater_modulation)
            success: Whether action was successful
            duration: Optional action duration in seconds
        """
        success_str = "true" if success else "false"

        if self.enable_prometheus:
            self.control_actions_total.labels(
                action_type=action_type,
                success=success_str
            ).inc()

            if duration is not None:
                self.control_action_duration_seconds.labels(
                    action_type=action_type
                ).observe(duration)
        else:
            self._internal_counters[f"control_action_{action_type}_{success_str}"] += 1

        # Add to internal buffer
        self.buffer.add(MetricValue(
            name="control_action",
            value=1,
            labels={"action_type": action_type, "success": success_str},
            metric_type=MetricType.COUNTER
        ))

        logger.debug(
            f"Recorded control action: type={action_type}, success={success}"
        )

    def record_quality_violation(
        self,
        violation_type: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a steam quality violation.

        Args:
            violation_type: Type of violation (e.g., low_dryness,
                           pressure_deviation, temperature_deviation)
            severity: Violation severity (info, warning, critical)
            details: Optional additional details
        """
        if self.enable_prometheus:
            self.quality_violations_total.labels(violation_type=violation_type).inc()
            self.steam_quality_alerts.labels(
                severity=severity,
                alert_type=violation_type
            ).inc()
        else:
            self._internal_counters[f"violation_{violation_type}"] += 1

        # Add to internal buffer
        self.buffer.add(MetricValue(
            name="quality_violation",
            value=1,
            labels={"violation_type": violation_type, "severity": severity},
            metric_type=MetricType.COUNTER
        ))

        logger.warning(
            f"Recorded quality violation: type={violation_type}, "
            f"severity={severity}, details={details}"
        )

    def update_operational_state(self, state: OperationalState) -> None:
        """
        Update all operational state gauges.

        Args:
            state: Current operational state of the steam quality controller
        """
        with self.lock:
            self._current_state = state

        if self.enable_prometheus:
            self.steam_quality_index.set(state.steam_quality_index)
            self.steam_dryness_fraction.set(state.steam_dryness_fraction)
            self.steam_pressure_bar.set(state.steam_pressure_bar)
            self.steam_temperature_c.set(state.steam_temperature_c)
            self.desuperheater_injection_rate_kg_hr.set(
                state.desuperheater_injection_rate_kg_hr
            )
            self.control_valve_position_percent.set(
                state.control_valve_position_percent
            )
        else:
            self._internal_gauges["steam_quality_index"] = state.steam_quality_index
            self._internal_gauges["steam_dryness_fraction"] = state.steam_dryness_fraction
            self._internal_gauges["steam_pressure_bar"] = state.steam_pressure_bar
            self._internal_gauges["steam_temperature_c"] = state.steam_temperature_c
            self._internal_gauges["desuperheater_injection_rate_kg_hr"] = (
                state.desuperheater_injection_rate_kg_hr
            )
            self._internal_gauges["control_valve_position_percent"] = (
                state.control_valve_position_percent
            )

        # Add to buffer for history tracking
        for metric_name, value in state.to_dict().items():
            if metric_name != "timestamp" and isinstance(value, (int, float)):
                self.buffer.add(MetricValue(
                    name=metric_name,
                    value=value,
                    metric_type=MetricType.GAUGE
                ))

        logger.debug(f"Updated operational state: {state}")

    def update_cache_hit_ratio(self, hits: int, misses: int) -> None:
        """
        Update cache hit ratio based on hits and misses.

        Args:
            hits: Number of cache hits
            misses: Number of cache misses
        """
        total = hits + misses
        ratio = hits / total if total > 0 else 0.0

        if self.enable_prometheus:
            self.cache_hit_ratio.set(ratio)
            self.cache_hits_total.inc(hits)
            self.cache_misses_total.inc(misses)
        else:
            self._internal_gauges["cache_hit_ratio"] = ratio

        self.buffer.add(MetricValue(
            name="cache_hit_ratio",
            value=ratio,
            metric_type=MetricType.GAUGE
        ))

    def record_agent_error(
        self,
        error_category: str,
        component: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record an agent error.

        Args:
            error_category: Category of error (e.g., validation, calculation,
                           integration)
            component: Component where error occurred
            error_message: Optional error message for logging
        """
        if self.enable_prometheus:
            self.agent_errors_total.labels(
                error_category=error_category,
                component=component
            ).inc()
        else:
            self._internal_counters[f"error_{error_category}_{component}"] += 1

        self.buffer.add(MetricValue(
            name="agent_error",
            value=1,
            labels={"error_category": error_category, "component": component},
            metric_type=MetricType.COUNTER
        ))

        logger.error(
            f"Agent error recorded: category={error_category}, "
            f"component={component}, message={error_message}"
        )

    # =========================================================================
    # CONTEXT MANAGERS
    # =========================================================================

    @contextmanager
    def track_calculation(self, calculation_type: str = "quality"):
        """
        Context manager for tracking calculation duration.

        Args:
            calculation_type: Type of calculation being tracked

        Yields:
            None

        Example:
            >>> with collector.track_calculation("dryness"):
            ...     result = calculate_dryness()
        """
        start_time = time.perf_counter()
        success = True

        try:
            yield
        except Exception as e:
            success = False
            self.record_agent_error(
                error_category="calculation",
                component=calculation_type,
                error_message=str(e)
            )
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.record_quality_calculation(
                duration=duration,
                result={"calculation_type": calculation_type},
                success=success
            )

    @contextmanager
    def track_control_action(self, action_type: str):
        """
        Context manager for tracking control action execution.

        Args:
            action_type: Type of control action

        Yields:
            None

        Example:
            >>> with collector.track_control_action("valve_adjustment"):
            ...     adjust_valve(position=50)
        """
        start_time = time.perf_counter()
        success = True

        if self.enable_prometheus:
            self.active_control_operations.labels(operation_type=action_type).inc()

        try:
            yield
        except Exception as e:
            success = False
            self.record_agent_error(
                error_category="control",
                component=action_type,
                error_message=str(e)
            )
            raise
        finally:
            duration = time.perf_counter() - start_time

            if self.enable_prometheus:
                self.active_control_operations.labels(operation_type=action_type).dec()

            self.record_control_action(
                action_type=action_type,
                success=success,
                duration=duration
            )

    # =========================================================================
    # SNAPSHOT AND EXPORT METHODS
    # =========================================================================

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of all current metric values.

        Returns:
            Dictionary containing all current metric values and metadata
        """
        with self.lock:
            current_state = self._current_state.to_dict()

        snapshot = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "prometheus_enabled": self.enable_prometheus,
            "buffer_size": len(self.buffer),
            "operational_state": current_state,
        }

        # Add internal counters if not using Prometheus
        if not self.enable_prometheus:
            snapshot["counters"] = dict(self._internal_counters)
            snapshot["gauges"] = dict(self._internal_gauges)

        # Add recent metrics summary
        recent_metrics = self.buffer.get_recent(60)
        snapshot["metrics_last_minute"] = {
            "total_count": len(recent_metrics),
            "calculations": len([m for m in recent_metrics
                               if m.name == "quality_calculation"]),
            "control_actions": len([m for m in recent_metrics
                                   if m.name == "control_action"]),
            "violations": len([m for m in recent_metrics
                             if m.name == "quality_violation"]),
            "errors": len([m for m in recent_metrics
                          if m.name == "agent_error"]),
        }

        return snapshot

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus metrics text format string

        Raises:
            RuntimeError: If Prometheus is not available
        """
        if not self.enable_prometheus:
            logger.warning("Prometheus not available, returning empty metrics")
            return ""

        try:
            metrics_bytes = generate_latest(self.registry)
            return metrics_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            raise RuntimeError(f"Prometheus export failed: {e}") from e

    def start_prometheus_server(self, port: int = 9012) -> None:
        """
        Start Prometheus HTTP metrics server.

        Args:
            port: Port to serve metrics on (default: 9012 for GL-012)

        Raises:
            RuntimeError: If Prometheus is not available
        """
        if not self.enable_prometheus:
            raise RuntimeError("Prometheus is not available")

        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            raise

    def reset_counters(self) -> None:
        """
        Reset all internal counters (for testing purposes).

        Note: This does not reset Prometheus counters as they are
        monotonically increasing by design.
        """
        self._internal_counters.clear()
        self.buffer.clear_old(max_age_seconds=0)
        logger.info("Internal counters reset")

    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get metrics relevant for health checking.

        Returns:
            Dictionary of health-relevant metrics
        """
        recent = self.buffer.get_recent(60)
        error_count = len([m for m in recent if m.name == "agent_error"])
        violation_count = len([m for m in recent if m.name == "quality_violation"])

        with self.lock:
            dryness = self._current_state.steam_dryness_fraction
            quality_index = self._current_state.steam_quality_index

        return {
            "errors_last_minute": error_count,
            "violations_last_minute": violation_count,
            "current_dryness_fraction": dryness,
            "current_quality_index": quality_index,
            "buffer_utilization": len(self.buffer) / self.buffer.max_size,
            "uptime_seconds": time.time() - self.start_time,
            "prometheus_enabled": self.enable_prometheus,
        }


# Module-level convenience functions

_default_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> Optional[MetricsCollector]:
    """
    Get the default metrics collector instance.

    Returns:
        Default MetricsCollector or None if not initialized
    """
    return _default_collector


def init_metrics_collector(
    agent_id: str = "GL-012",
    agent_name: str = "SteamQualityController",
    **kwargs
) -> MetricsCollector:
    """
    Initialize and return the default metrics collector.

    Args:
        agent_id: Agent identifier
        agent_name: Human-readable agent name
        **kwargs: Additional arguments for MetricsCollector

    Returns:
        Initialized MetricsCollector instance
    """
    global _default_collector
    _default_collector = MetricsCollector(
        agent_id=agent_id,
        agent_name=agent_name,
        **kwargs
    )
    return _default_collector


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "MetricType",
    "MetricValue",
    "OperationalState",
    "MetricsBuffer",
    "MetricsCollector",
    "get_metrics_collector",
    "init_metrics_collector",
]
