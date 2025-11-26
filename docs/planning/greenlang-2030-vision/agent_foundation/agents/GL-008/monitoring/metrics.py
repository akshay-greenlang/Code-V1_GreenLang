"""
GL-008 SteamTrapInspector Metrics Collection

This module implements comprehensive Prometheus metrics for steam trap inspection,
failure detection, energy loss tracking, and fleet health monitoring.

The metrics follow GreenLang's observability standards with:
- 50+ metrics covering all inspection aspects
- Multi-dimensional labeling (facility, trap_type, severity, etc.)
- Performance tracking (P50, P95, P99 latencies)
- Business metrics (energy loss, cost savings, CO2 reduction)

Example:
    >>> from metrics import metrics_collector
    >>> metrics_collector.record_inspection(
    ...     trap_id="ST-001",
    ...     facility="Plant-A",
    ...     trap_type="inverted_bucket",
    ...     status="healthy",
    ...     duration_ms=2456.3
    ... )
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    Enum,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
import logging
from dataclasses import dataclass
from enum import Enum as PyEnum

logger = logging.getLogger(__name__)


class TrapStatus(PyEnum):
    """Steam trap health status classifications."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FailureMode(PyEnum):
    """Steam trap failure mode classifications."""
    BLOW_THROUGH = "blow_through"
    PLUGGED = "plugged"
    COLD = "cold"
    LEAKING = "leaking"
    CYCLING_SLOW = "cycling_slow"
    CYCLING_FAST = "cycling_fast"
    MECHANICAL_DAMAGE = "mechanical_damage"
    SCALE_BUILDUP = "scale_buildup"
    UNKNOWN = "unknown"


class InspectionMethod(PyEnum):
    """Inspection method types."""
    ACOUSTIC = "acoustic"
    THERMAL = "thermal"
    ULTRASONIC = "ultrasonic"
    VISUAL = "visual"
    COMBINED = "combined"


@dataclass
class InspectionMetrics:
    """Container for inspection event metrics."""
    trap_id: str
    facility: str
    trap_type: str
    status: TrapStatus
    duration_ms: float
    method: InspectionMethod
    confidence: float
    timestamp: datetime

    # Optional failure details
    failure_mode: Optional[FailureMode] = None
    severity: Optional[str] = None
    energy_loss_kw: Optional[float] = None
    steam_loss_kg_hr: Optional[float] = None
    cost_impact_usd_yr: Optional[float] = None


@dataclass
class FleetMetrics:
    """Container for fleet-wide metrics."""
    facility: str
    total_traps: int
    healthy_count: int
    degraded_count: int
    failed_count: int
    critical_count: int
    total_energy_loss_kw: float
    total_cost_impact_usd_yr: float
    avg_health_score: float


class MetricsCollector:
    """
    Comprehensive metrics collector for GL-008 SteamTrapInspector.

    Implements 50+ Prometheus metrics covering:
    - Inspection performance and throughput
    - Failure detection accuracy
    - Energy loss tracking
    - Fleet health monitoring
    - Alert generation
    - System performance

    All metrics are labeled for multi-dimensional analysis.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector with Prometheus registry.

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        logger.info("GL-008 MetricsCollector initialized with 50+ metrics")

    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""

        # ============================================================
        # INSPECTION METRICS
        # ============================================================

        # Counter: Total inspections performed
        self.inspections_total = Counter(
            'gl008_inspections_total',
            'Total steam trap inspections performed',
            ['facility', 'trap_type', 'method', 'status'],
            registry=self.registry
        )

        # Counter: Inspection failures (system errors)
        self.inspection_errors_total = Counter(
            'gl008_inspection_errors_total',
            'Total inspection system errors',
            ['facility', 'error_type', 'method'],
            registry=self.registry
        )

        # Counter: Inspections by shift/time
        self.inspections_by_shift_total = Counter(
            'gl008_inspections_by_shift_total',
            'Inspections performed by shift',
            ['facility', 'shift', 'day_of_week'],
            registry=self.registry
        )

        # Histogram: Inspection duration
        self.inspection_duration = Histogram(
            'gl008_inspection_duration_seconds',
            'Time to complete trap inspection',
            ['facility', 'trap_type', 'method'],
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0],
            registry=self.registry
        )

        # Histogram: Acoustic analysis time
        self.acoustic_analysis_time = Histogram(
            'gl008_acoustic_analysis_seconds',
            'Time for acoustic signature analysis',
            ['facility', 'trap_type'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
            registry=self.registry
        )

        # Histogram: Thermal analysis time
        self.thermal_analysis_time = Histogram(
            'gl008_thermal_analysis_seconds',
            'Time for thermal imaging analysis',
            ['facility', 'trap_type'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
            registry=self.registry
        )

        # Histogram: Ultrasonic analysis time
        self.ultrasonic_analysis_time = Histogram(
            'gl008_ultrasonic_analysis_seconds',
            'Time for ultrasonic data analysis',
            ['facility', 'trap_type'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
            registry=self.registry
        )

        # Gauge: Detection confidence score
        self.detection_confidence = Gauge(
            'gl008_detection_confidence',
            'Confidence score of failure detection',
            ['facility', 'trap_id', 'method'],
            registry=self.registry
        )

        # Histogram: End-to-end inspection latency
        self.inspection_latency = Histogram(
            'gl008_inspection_latency_seconds',
            'End-to-end inspection latency including processing',
            ['facility', 'method'],
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
            registry=self.registry
        )

        # ============================================================
        # FAILURE DETECTION METRICS
        # ============================================================

        # Counter: Total failures detected
        self.failures_detected_total = Counter(
            'gl008_failures_detected_total',
            'Total steam trap failures detected',
            ['facility', 'trap_type', 'failure_mode', 'severity'],
            registry=self.registry
        )

        # Counter: False positives
        self.false_positives_total = Counter(
            'gl008_false_positives_total',
            'False positive detections (confirmed by technician)',
            ['facility', 'trap_type', 'failure_mode'],
            registry=self.registry
        )

        # Counter: False negatives
        self.false_negatives_total = Counter(
            'gl008_false_negatives_total',
            'False negative detections (missed failures)',
            ['facility', 'trap_type', 'failure_mode'],
            registry=self.registry
        )

        # Counter: True positives
        self.true_positives_total = Counter(
            'gl008_true_positives_total',
            'True positive detections (confirmed failures)',
            ['facility', 'trap_type', 'failure_mode', 'severity'],
            registry=self.registry
        )

        # Gauge: Current detection accuracy rate
        self.detection_accuracy = Gauge(
            'gl008_detection_accuracy_rate',
            'Current failure detection accuracy rate',
            ['facility', 'trap_type', 'time_window'],
            registry=self.registry
        )

        # Gauge: Precision (PPV)
        self.detection_precision = Gauge(
            'gl008_detection_precision',
            'Precision (positive predictive value)',
            ['facility', 'trap_type'],
            registry=self.registry
        )

        # Gauge: Recall (sensitivity)
        self.detection_recall = Gauge(
            'gl008_detection_recall',
            'Recall (sensitivity) rate',
            ['facility', 'trap_type'],
            registry=self.registry
        )

        # Gauge: F1 score
        self.detection_f1_score = Gauge(
            'gl008_detection_f1_score',
            'F1 score for failure detection',
            ['facility', 'trap_type'],
            registry=self.registry
        )

        # ============================================================
        # TRAP HEALTH METRICS
        # ============================================================

        # Gauge: Individual trap health score (0-100)
        self.trap_health_score = Gauge(
            'gl008_trap_health_score',
            'Individual trap health score (0-100)',
            ['facility', 'trap_id', 'trap_type'],
            registry=self.registry
        )

        # Gauge: Trap status (0=healthy, 1=degraded, 2=failed, 3=critical)
        self.trap_status = Gauge(
            'gl008_trap_status',
            'Trap status code (0=healthy, 1=degraded, 2=failed, 3=critical)',
            ['facility', 'trap_id', 'trap_type'],
            registry=self.registry
        )

        # Gauge: Days since last inspection
        self.days_since_inspection = Gauge(
            'gl008_days_since_inspection',
            'Days since trap was last inspected',
            ['facility', 'trap_id', 'trap_type'],
            registry=self.registry
        )

        # Gauge: Trap operating hours
        self.trap_operating_hours = Gauge(
            'gl008_trap_operating_hours',
            'Total operating hours for trap',
            ['facility', 'trap_id', 'trap_type'],
            registry=self.registry
        )

        # Gauge: Cycle count (for thermodynamic traps)
        self.trap_cycle_count = Gauge(
            'gl008_trap_cycle_count',
            'Total cycle count for thermodynamic traps',
            ['facility', 'trap_id'],
            registry=self.registry
        )

        # ============================================================
        # ENERGY LOSS METRICS
        # ============================================================

        # Gauge: Current energy loss per trap (kW)
        self.energy_loss_kw = Gauge(
            'gl008_energy_loss_kw',
            'Current energy loss in kilowatts',
            ['facility', 'trap_id', 'trap_type', 'failure_mode'],
            registry=self.registry
        )

        # Gauge: Steam loss rate (kg/hr)
        self.steam_loss_kg_hr = Gauge(
            'gl008_steam_loss_kg_hr',
            'Steam loss rate in kg per hour',
            ['facility', 'trap_id', 'trap_type', 'failure_mode'],
            registry=self.registry
        )

        # Gauge: Total facility energy loss (kW)
        self.total_energy_loss_kw = Gauge(
            'gl008_total_energy_loss_kw',
            'Total facility energy loss in kilowatts',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Total facility steam loss (kg/hr)
        self.total_steam_loss_kg_hr = Gauge(
            'gl008_total_steam_loss_kg_hr',
            'Total facility steam loss in kg per hour',
            ['facility'],
            registry=self.registry
        )

        # Counter: Cumulative energy wasted (kWh)
        self.energy_wasted_kwh_total = Counter(
            'gl008_energy_wasted_kwh_total',
            'Cumulative energy wasted in kilowatt-hours',
            ['facility', 'failure_mode'],
            registry=self.registry
        )

        # Counter: Cumulative steam wasted (kg)
        self.steam_wasted_kg_total = Counter(
            'gl008_steam_wasted_kg_total',
            'Cumulative steam wasted in kilograms',
            ['facility', 'failure_mode'],
            registry=self.registry
        )

        # ============================================================
        # COST & SAVINGS METRICS
        # ============================================================

        # Gauge: Annual cost impact per trap (USD/year)
        self.cost_impact_usd_yr = Gauge(
            'gl008_cost_impact_usd_per_year',
            'Annual cost impact of trap failure in USD',
            ['facility', 'trap_id', 'failure_mode'],
            registry=self.registry
        )

        # Gauge: Total facility cost impact (USD/year)
        self.total_cost_impact_usd_yr = Gauge(
            'gl008_total_cost_impact_usd_per_year',
            'Total facility annual cost impact in USD',
            ['facility'],
            registry=self.registry
        )

        # Counter: Avoided costs from early detection (USD)
        self.avoided_costs_usd_total = Counter(
            'gl008_avoided_costs_usd_total',
            'Cumulative avoided costs from early detection',
            ['facility'],
            registry=self.registry
        )

        # Counter: Maintenance costs (USD)
        self.maintenance_costs_usd_total = Counter(
            'gl008_maintenance_costs_usd_total',
            'Cumulative maintenance costs',
            ['facility', 'repair_type'],
            registry=self.registry
        )

        # Gauge: ROI from inspection program
        self.inspection_roi = Gauge(
            'gl008_inspection_roi',
            'Return on investment from inspection program',
            ['facility', 'time_window'],
            registry=self.registry
        )

        # ============================================================
        # CO2 EMISSIONS METRICS
        # ============================================================

        # Gauge: CO2 emissions from energy loss (kg CO2/hour)
        self.co2_emissions_kg_hr = Gauge(
            'gl008_co2_emissions_kg_per_hour',
            'CO2 emissions from energy loss in kg per hour',
            ['facility', 'trap_id'],
            registry=self.registry
        )

        # Counter: Total CO2 emissions (kg CO2)
        self.co2_emissions_kg_total = Counter(
            'gl008_co2_emissions_kg_total',
            'Cumulative CO2 emissions from steam trap failures',
            ['facility'],
            registry=self.registry
        )

        # Counter: Avoided CO2 emissions (kg CO2)
        self.co2_avoided_kg_total = Counter(
            'gl008_co2_avoided_kg_total',
            'Cumulative avoided CO2 emissions from repairs',
            ['facility'],
            registry=self.registry
        )

        # ============================================================
        # FLEET HEALTH METRICS
        # ============================================================

        # Gauge: Total traps in fleet
        self.fleet_total_traps = Gauge(
            'gl008_fleet_total_traps',
            'Total number of traps in fleet',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Healthy traps count
        self.fleet_healthy_count = Gauge(
            'gl008_fleet_healthy_count',
            'Number of healthy traps',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Degraded traps count
        self.fleet_degraded_count = Gauge(
            'gl008_fleet_degraded_count',
            'Number of degraded traps',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Failed traps count
        self.fleet_failed_count = Gauge(
            'gl008_fleet_failed_count',
            'Number of failed traps',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Critical traps count
        self.fleet_critical_count = Gauge(
            'gl008_fleet_critical_count',
            'Number of critical traps requiring immediate attention',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Fleet health score (0-100)
        self.fleet_health_score = Gauge(
            'gl008_fleet_health_score',
            'Overall fleet health score (0-100)',
            ['facility'],
            registry=self.registry
        )

        # Gauge: Fleet coverage (% inspected)
        self.fleet_coverage_pct = Gauge(
            'gl008_fleet_coverage_percent',
            'Percentage of fleet inspected in period',
            ['facility', 'time_window'],
            registry=self.registry
        )

        # ============================================================
        # ALERT METRICS
        # ============================================================

        # Counter: Total alerts generated
        self.alerts_total = Counter(
            'gl008_alerts_total',
            'Total alerts generated',
            ['facility', 'severity', 'alert_type'],
            registry=self.registry
        )

        # Counter: Alerts by resolution
        self.alerts_resolved_total = Counter(
            'gl008_alerts_resolved_total',
            'Alerts resolved',
            ['facility', 'severity', 'alert_type', 'resolution'],
            registry=self.registry
        )

        # Histogram: Alert response time
        self.alert_response_time = Histogram(
            'gl008_alert_response_seconds',
            'Time from alert generation to technician response',
            ['facility', 'severity'],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800],
            registry=self.registry
        )

        # Histogram: Alert resolution time
        self.alert_resolution_time = Histogram(
            'gl008_alert_resolution_seconds',
            'Time from alert generation to resolution',
            ['facility', 'severity', 'alert_type'],
            buckets=[300, 1800, 3600, 7200, 14400, 28800, 86400, 172800],
            registry=self.registry
        )

        # Gauge: Active alerts
        self.active_alerts = Gauge(
            'gl008_active_alerts',
            'Currently active alerts',
            ['facility', 'severity', 'alert_type'],
            registry=self.registry
        )

        # ============================================================
        # SYSTEM PERFORMANCE METRICS
        # ============================================================

        # Counter: API requests
        self.api_requests_total = Counter(
            'gl008_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )

        # Histogram: API latency
        self.api_latency = Histogram(
            'gl008_api_latency_seconds',
            'API request latency',
            ['endpoint', 'method'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )

        # Gauge: System health status
        self.system_health = Gauge(
            'gl008_system_health',
            'System health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )

        # Gauge: Database connection pool
        self.db_connections = Gauge(
            'gl008_db_connections',
            'Active database connections',
            ['pool'],
            registry=self.registry
        )

        # Counter: Cache hits/misses
        self.cache_operations_total = Counter(
            'gl008_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )

    # ============================================================
    # RECORDING METHODS
    # ============================================================

    def record_inspection(
        self,
        trap_id: str,
        facility: str,
        trap_type: str,
        status: str,
        duration_ms: float,
        method: str = "combined",
        confidence: float = 0.0,
        acoustic_time_ms: Optional[float] = None,
        thermal_time_ms: Optional[float] = None,
        ultrasonic_time_ms: Optional[float] = None
    ):
        """
        Record a completed inspection with all metrics.

        Args:
            trap_id: Unique trap identifier
            facility: Facility name
            trap_type: Type of steam trap
            status: Inspection result status
            duration_ms: Total inspection duration in milliseconds
            method: Inspection method used
            confidence: Detection confidence (0.0-1.0)
            acoustic_time_ms: Acoustic analysis time
            thermal_time_ms: Thermal analysis time
            ultrasonic_time_ms: Ultrasonic analysis time
        """
        duration_s = duration_ms / 1000.0

        # Record inspection counter
        self.inspections_total.labels(
            facility=facility,
            trap_type=trap_type,
            method=method,
            status=status
        ).inc()

        # Record duration histogram
        self.inspection_duration.labels(
            facility=facility,
            trap_type=trap_type,
            method=method
        ).observe(duration_s)

        # Record latency
        self.inspection_latency.labels(
            facility=facility,
            method=method
        ).observe(duration_s)

        # Record confidence
        if confidence > 0:
            self.detection_confidence.labels(
                facility=facility,
                trap_id=trap_id,
                method=method
            ).set(confidence)

        # Record analysis times
        if acoustic_time_ms:
            self.acoustic_analysis_time.labels(
                facility=facility,
                trap_type=trap_type
            ).observe(acoustic_time_ms / 1000.0)

        if thermal_time_ms:
            self.thermal_analysis_time.labels(
                facility=facility,
                trap_type=trap_type
            ).observe(thermal_time_ms / 1000.0)

        if ultrasonic_time_ms:
            self.ultrasonic_analysis_time.labels(
                facility=facility,
                trap_type=trap_type
            ).observe(ultrasonic_time_ms / 1000.0)

        logger.debug(f"Recorded inspection for {trap_id}: {status} in {duration_ms}ms")

    def record_failure(
        self,
        trap_id: str,
        facility: str,
        trap_type: str,
        failure_mode: str,
        severity: str,
        energy_loss_kw: float,
        steam_loss_kg_hr: float,
        cost_impact_usd_yr: float,
        co2_kg_hr: Optional[float] = None
    ):
        """
        Record a detected failure with energy and cost impacts.

        Args:
            trap_id: Unique trap identifier
            facility: Facility name
            trap_type: Type of steam trap
            failure_mode: Type of failure detected
            severity: Failure severity level
            energy_loss_kw: Energy loss in kilowatts
            steam_loss_kg_hr: Steam loss in kg/hour
            cost_impact_usd_yr: Annual cost impact in USD
            co2_kg_hr: CO2 emissions in kg/hour
        """
        # Record failure detection
        self.failures_detected_total.labels(
            facility=facility,
            trap_type=trap_type,
            failure_mode=failure_mode,
            severity=severity
        ).inc()

        # Record energy loss
        self.energy_loss_kw.labels(
            facility=facility,
            trap_id=trap_id,
            trap_type=trap_type,
            failure_mode=failure_mode
        ).set(energy_loss_kw)

        # Record steam loss
        self.steam_loss_kg_hr.labels(
            facility=facility,
            trap_id=trap_id,
            trap_type=trap_type,
            failure_mode=failure_mode
        ).set(steam_loss_kg_hr)

        # Record cost impact
        self.cost_impact_usd_yr.labels(
            facility=facility,
            trap_id=trap_id,
            failure_mode=failure_mode
        ).set(cost_impact_usd_yr)

        # Record CO2 emissions
        if co2_kg_hr:
            self.co2_emissions_kg_hr.labels(
                facility=facility,
                trap_id=trap_id
            ).set(co2_kg_hr)

        logger.info(f"Recorded failure for {trap_id}: {failure_mode} ({severity}), "
                   f"{energy_loss_kw:.2f} kW loss, ${cost_impact_usd_yr:,.0f}/yr impact")

    def update_trap_status(
        self,
        trap_id: str,
        facility: str,
        trap_type: str,
        health_score: float,
        status: TrapStatus,
        days_since_inspection: int,
        operating_hours: Optional[float] = None
    ):
        """
        Update trap health and status metrics.

        Args:
            trap_id: Unique trap identifier
            facility: Facility name
            trap_type: Type of steam trap
            health_score: Health score (0-100)
            status: Current trap status
            days_since_inspection: Days since last inspection
            operating_hours: Total operating hours
        """
        # Update health score
        self.trap_health_score.labels(
            facility=facility,
            trap_id=trap_id,
            trap_type=trap_type
        ).set(health_score)

        # Update status (encode as numeric)
        status_map = {
            TrapStatus.HEALTHY: 0,
            TrapStatus.DEGRADED: 1,
            TrapStatus.FAILED: 2,
            TrapStatus.CRITICAL: 3,
            TrapStatus.UNKNOWN: 4
        }
        self.trap_status.labels(
            facility=facility,
            trap_id=trap_id,
            trap_type=trap_type
        ).set(status_map[status])

        # Update inspection timing
        self.days_since_inspection.labels(
            facility=facility,
            trap_id=trap_id,
            trap_type=trap_type
        ).set(days_since_inspection)

        # Update operating hours if provided
        if operating_hours is not None:
            self.trap_operating_hours.labels(
                facility=facility,
                trap_id=trap_id,
                trap_type=trap_type
            ).set(operating_hours)

    def update_fleet_metrics(self, fleet_data: FleetMetrics):
        """
        Update fleet-wide health metrics.

        Args:
            fleet_data: Fleet metrics data container
        """
        facility = fleet_data.facility

        # Update counts
        self.fleet_total_traps.labels(facility=facility).set(fleet_data.total_traps)
        self.fleet_healthy_count.labels(facility=facility).set(fleet_data.healthy_count)
        self.fleet_degraded_count.labels(facility=facility).set(fleet_data.degraded_count)
        self.fleet_failed_count.labels(facility=facility).set(fleet_data.failed_count)
        self.fleet_critical_count.labels(facility=facility).set(fleet_data.critical_count)

        # Update totals
        self.total_energy_loss_kw.labels(facility=facility).set(fleet_data.total_energy_loss_kw)
        self.total_cost_impact_usd_yr.labels(facility=facility).set(fleet_data.total_cost_impact_usd_yr)

        # Update fleet health score
        self.fleet_health_score.labels(facility=facility).set(fleet_data.avg_health_score)

        logger.info(f"Updated fleet metrics for {facility}: "
                   f"{fleet_data.failed_count}/{fleet_data.total_traps} failures, "
                   f"health score {fleet_data.avg_health_score:.1f}")

    def record_alert(
        self,
        facility: str,
        severity: str,
        alert_type: str,
        increment_active: bool = True
    ):
        """
        Record alert generation.

        Args:
            facility: Facility name
            severity: Alert severity
            alert_type: Type of alert
            increment_active: Whether to increment active alerts gauge
        """
        self.alerts_total.labels(
            facility=facility,
            severity=severity,
            alert_type=alert_type
        ).inc()

        if increment_active:
            self.active_alerts.labels(
                facility=facility,
                severity=severity,
                alert_type=alert_type
            ).inc()

    def resolve_alert(
        self,
        facility: str,
        severity: str,
        alert_type: str,
        resolution: str,
        response_time_s: float,
        resolution_time_s: float
    ):
        """
        Record alert resolution.

        Args:
            facility: Facility name
            severity: Alert severity
            alert_type: Type of alert
            resolution: Resolution type
            response_time_s: Time to response in seconds
            resolution_time_s: Time to resolution in seconds
        """
        self.alerts_resolved_total.labels(
            facility=facility,
            severity=severity,
            alert_type=alert_type,
            resolution=resolution
        ).inc()

        self.alert_response_time.labels(
            facility=facility,
            severity=severity
        ).observe(response_time_s)

        self.alert_resolution_time.labels(
            facility=facility,
            severity=severity,
            alert_type=alert_type
        ).observe(resolution_time_s)

        # Decrement active alerts
        self.active_alerts.labels(
            facility=facility,
            severity=severity,
            alert_type=alert_type
        ).dec()

    def export_metrics(self) -> bytes:
        """
        Export all metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics as bytes
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type for HTTP responses."""
        return CONTENT_TYPE_LATEST


# Global singleton instance
metrics_collector = MetricsCollector()


# Convenience functions for common operations
def record_inspection(*args, **kwargs):
    """Convenience wrapper for metrics_collector.record_inspection()."""
    return metrics_collector.record_inspection(*args, **kwargs)


def record_failure(*args, **kwargs):
    """Convenience wrapper for metrics_collector.record_failure()."""
    return metrics_collector.record_failure(*args, **kwargs)


def update_trap_status(*args, **kwargs):
    """Convenience wrapper for metrics_collector.update_trap_status()."""
    return metrics_collector.update_trap_status(*args, **kwargs)


def update_fleet_metrics(*args, **kwargs):
    """Convenience wrapper for metrics_collector.update_fleet_metrics()."""
    return metrics_collector.update_fleet_metrics(*args, **kwargs)
