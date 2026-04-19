"""
GL-012 STEAMQUAL SteamQualityController - Dashboard Data Provider

This module provides dashboard data aggregation for UI consumption, including
real-time steam quality dashboards, KPI dashboards, separator performance
dashboards, and quality trend analysis views.

Dashboard Types:
    - RealTimeQualityDashboard: Live steam quality status across all separators
    - SeparatorDashboard: Individual separator performance and quality metrics
    - QualityTrendDashboard: Historical quality trends and pattern analysis
    - AlertSummaryDashboard: Alert statistics and active alarm overview
    - KPIDashboard: Key performance indicators for steam quality

Example:
    >>> provider = SteamQualityDashboardProvider(metrics_collector, alert_manager)
    >>> real_time = await provider.get_real_time_dashboard()
    >>> separator = await provider.get_separator_dashboard("SEP-001")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class TimeSeriesData:
    """Time series data for charts."""
    name: str
    unit: str
    timestamps: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "unit": self.unit,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "values": self.values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "avg_value": self.avg_value,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
        }

    def compute_stats(self) -> None:
        """Compute min, max, avg statistics."""
        if self.values:
            self.min_value = min(self.values)
            self.max_value = max(self.values)
            self.avg_value = sum(self.values) / len(self.values)


@dataclass
class DashboardWidget:
    """Generic dashboard widget."""
    widget_id: str
    widget_type: str  # gauge, chart, table, alert, stat, heatmap
    title: str
    data: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # row, col, width, height
    refresh_interval_s: float = 10.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type,
            "title": self.title,
            "data": self.data,
            "position": self.position,
            "refresh_interval_s": self.refresh_interval_s,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SeparatorStatus:
    """Status summary for a single separator."""
    separator_id: str
    status: str = "operational"  # operational, degraded, critical, offline
    dryness_fraction: float = 1.0
    carryover_risk: float = 0.0
    efficiency_percent: float = 95.0
    level_percent: float = 50.0
    active_alerts: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "separator_id": self.separator_id,
            "status": self.status,
            "dryness_fraction": self.dryness_fraction,
            "carryover_risk": self.carryover_risk,
            "efficiency_percent": self.efficiency_percent,
            "level_percent": self.level_percent,
            "active_alerts": self.active_alerts,
            "last_update": self.last_update.isoformat(),
        }


# =============================================================================
# Dashboard Data Classes
# =============================================================================

@dataclass
class RealTimeQualityDashboard:
    """Real-time dashboard data for live steam quality monitoring."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Overall system status
    system_status: str = "operational"  # operational, degraded, critical, offline
    total_separators: int = 0
    operational_separators: int = 0
    degraded_separators: int = 0
    critical_separators: int = 0

    # Aggregate quality metrics
    avg_dryness_fraction: float = 1.0
    min_dryness_fraction: float = 1.0
    max_carryover_risk: float = 0.0
    avg_separator_efficiency: float = 95.0

    # Alert summary
    active_alerts_count: int = 0
    critical_alerts_count: int = 0
    warning_alerts_count: int = 0

    # Separator statuses
    separator_statuses: List[SeparatorStatus] = field(default_factory=list)

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system_status": self.system_status,
            "separators": {
                "total": self.total_separators,
                "operational": self.operational_separators,
                "degraded": self.degraded_separators,
                "critical": self.critical_separators,
            },
            "quality": {
                "avg_dryness_fraction": self.avg_dryness_fraction,
                "min_dryness_fraction": self.min_dryness_fraction,
                "max_carryover_risk": self.max_carryover_risk,
                "avg_separator_efficiency": self.avg_separator_efficiency,
            },
            "alerts": {
                "active": self.active_alerts_count,
                "critical": self.critical_alerts_count,
                "warning": self.warning_alerts_count,
            },
            "separator_statuses": [s.to_dict() for s in self.separator_statuses],
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class SeparatorDashboard:
    """Dashboard data for individual separator monitoring."""
    separator_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Current status
    status: str = "operational"
    is_online: bool = True

    # Quality metrics
    dryness_fraction: float = 1.0
    moisture_content_percent: float = 0.0
    carryover_risk: float = 0.0

    # Separator performance
    efficiency_percent: float = 95.0
    pressure_drop_bar: float = 0.0
    level_percent: float = 50.0
    drain_rate_kg_h: float = 0.0

    # Operating conditions
    inlet_pressure_bar: float = 0.0
    outlet_pressure_bar: float = 0.0
    inlet_temperature_c: float = 0.0
    steam_flow_kg_h: float = 0.0
    steam_velocity_m_s: float = 0.0

    # Risk indicators
    water_hammer_risk: float = 0.0
    flooding_risk: float = 0.0

    # Alert summary
    active_alerts: int = 0
    recent_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Historical data
    dryness_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(
            name="dryness_fraction",
            unit="fraction",
            threshold_low=0.95,
        )
    )
    carryover_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(
            name="carryover_risk",
            unit="score",
            threshold_high=0.3,
        )
    )
    efficiency_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(
            name="efficiency",
            unit="%",
            threshold_low=90.0,
        )
    )

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "separator_id": self.separator_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "is_online": self.is_online,
            "quality": {
                "dryness_fraction": self.dryness_fraction,
                "moisture_content_percent": self.moisture_content_percent,
                "carryover_risk": self.carryover_risk,
            },
            "performance": {
                "efficiency_percent": self.efficiency_percent,
                "pressure_drop_bar": self.pressure_drop_bar,
                "level_percent": self.level_percent,
                "drain_rate_kg_h": self.drain_rate_kg_h,
            },
            "operating_conditions": {
                "inlet_pressure_bar": self.inlet_pressure_bar,
                "outlet_pressure_bar": self.outlet_pressure_bar,
                "inlet_temperature_c": self.inlet_temperature_c,
                "steam_flow_kg_h": self.steam_flow_kg_h,
                "steam_velocity_m_s": self.steam_velocity_m_s,
            },
            "risks": {
                "water_hammer_risk": self.water_hammer_risk,
                "flooding_risk": self.flooding_risk,
            },
            "alerts": {
                "active_count": self.active_alerts,
                "recent": self.recent_alerts,
            },
            "history": {
                "dryness": self.dryness_history.to_dict(),
                "carryover": self.carryover_history.to_dict(),
                "efficiency": self.efficiency_history.to_dict(),
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class QualityTrendDashboard:
    """Dashboard data for quality trend analysis."""
    time_window_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=24)
    )
    time_window_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Trend summary
    dryness_trend: str = "stable"  # improving, stable, declining
    carryover_trend: str = "stable"
    efficiency_trend: str = "stable"

    # Statistics over window
    avg_dryness_fraction: float = 0.0
    min_dryness_fraction: float = 0.0
    max_dryness_fraction: float = 0.0
    dryness_std_dev: float = 0.0

    avg_carryover_risk: float = 0.0
    max_carryover_risk: float = 0.0
    carryover_events: int = 0

    avg_efficiency_percent: float = 0.0
    min_efficiency_percent: float = 0.0

    # Event counts
    low_dryness_events: int = 0
    high_moisture_events: int = 0
    separator_flooding_events: int = 0
    water_hammer_events: int = 0
    total_quality_events: int = 0

    # Time series data
    dryness_trend_data: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="dryness_trend", unit="fraction")
    )
    carryover_trend_data: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="carryover_trend", unit="score")
    )
    efficiency_trend_data: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="efficiency_trend", unit="%")
    )
    event_count_data: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="event_count", unit="count")
    )

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "time_window": {
                "start": self.time_window_start.isoformat(),
                "end": self.time_window_end.isoformat(),
            },
            "timestamp": self.timestamp.isoformat(),
            "trends": {
                "dryness": self.dryness_trend,
                "carryover": self.carryover_trend,
                "efficiency": self.efficiency_trend,
            },
            "dryness_stats": {
                "avg": self.avg_dryness_fraction,
                "min": self.min_dryness_fraction,
                "max": self.max_dryness_fraction,
                "std_dev": self.dryness_std_dev,
            },
            "carryover_stats": {
                "avg": self.avg_carryover_risk,
                "max": self.max_carryover_risk,
                "events": self.carryover_events,
            },
            "efficiency_stats": {
                "avg": self.avg_efficiency_percent,
                "min": self.min_efficiency_percent,
            },
            "events": {
                "total": self.total_quality_events,
                "low_dryness": self.low_dryness_events,
                "high_moisture": self.high_moisture_events,
                "separator_flooding": self.separator_flooding_events,
                "water_hammer": self.water_hammer_events,
            },
            "trend_data": {
                "dryness": self.dryness_trend_data.to_dict(),
                "carryover": self.carryover_trend_data.to_dict(),
                "efficiency": self.efficiency_trend_data.to_dict(),
                "events": self.event_count_data.to_dict(),
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class AlertSummaryDashboard:
    """Dashboard data for alert summary and statistics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Current alert counts
    total_active: int = 0
    critical_count: int = 0
    warning_count: int = 0
    advisory_count: int = 0
    info_count: int = 0

    # Alert counts by type
    low_dryness_active: int = 0
    high_moisture_active: int = 0
    carryover_risk_active: int = 0
    separator_flooding_active: int = 0
    water_hammer_active: int = 0
    data_quality_active: int = 0

    # Historical statistics (last 24 hours)
    alerts_created_24h: int = 0
    alerts_resolved_24h: int = 0
    alerts_acknowledged_24h: int = 0
    avg_resolution_time_minutes: float = 0.0

    # Active alerts detail
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Alert type distribution
    alert_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Severity distribution
    severity_distribution: Dict[str, int] = field(default_factory=dict)

    # Top alerting separators
    top_alerting_separators: List[Dict[str, Any]] = field(default_factory=list)

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "active": {
                "total": self.total_active,
                "critical": self.critical_count,
                "warning": self.warning_count,
                "advisory": self.advisory_count,
                "info": self.info_count,
            },
            "by_type": {
                "low_dryness": self.low_dryness_active,
                "high_moisture": self.high_moisture_active,
                "carryover_risk": self.carryover_risk_active,
                "separator_flooding": self.separator_flooding_active,
                "water_hammer": self.water_hammer_active,
                "data_quality": self.data_quality_active,
            },
            "statistics_24h": {
                "created": self.alerts_created_24h,
                "resolved": self.alerts_resolved_24h,
                "acknowledged": self.alerts_acknowledged_24h,
                "avg_resolution_time_minutes": self.avg_resolution_time_minutes,
            },
            "active_alerts": self.active_alerts,
            "alert_type_distribution": self.alert_type_distribution,
            "severity_distribution": self.severity_distribution,
            "top_alerting_separators": self.top_alerting_separators,
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class KPIDashboard:
    """Dashboard data for key performance indicators."""
    time_window_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=24)
    )
    time_window_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Quality KPIs
    avg_dryness_fraction: float = 0.0
    dryness_target: float = 0.97
    dryness_achievement_percent: float = 0.0

    carryover_events_count: int = 0
    carryover_target: int = 0
    carryover_achievement_percent: float = 100.0

    # Efficiency KPIs
    avg_separator_efficiency: float = 0.0
    efficiency_target: float = 95.0
    efficiency_achievement_percent: float = 0.0

    # Availability KPIs
    system_availability_percent: float = 0.0
    availability_target: float = 99.5
    availability_achievement_percent: float = 0.0

    # Response KPIs
    avg_alert_response_time_minutes: float = 0.0
    response_target_minutes: float = 15.0
    response_achievement_percent: float = 0.0

    # Calculation performance
    calculations_total: int = 0
    calculation_success_rate: float = 0.0
    avg_calculation_time_ms: float = 0.0

    # Time series data
    dryness_kpi_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="dryness_kpi", unit="fraction")
    )
    efficiency_kpi_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="efficiency_kpi", unit="%")
    )
    availability_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="availability", unit="%")
    )

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "time_window": {
                "start": self.time_window_start.isoformat(),
                "end": self.time_window_end.isoformat(),
            },
            "timestamp": self.timestamp.isoformat(),
            "quality_kpis": {
                "dryness": {
                    "value": self.avg_dryness_fraction,
                    "target": self.dryness_target,
                    "achievement_percent": self.dryness_achievement_percent,
                },
                "carryover_events": {
                    "count": self.carryover_events_count,
                    "target": self.carryover_target,
                    "achievement_percent": self.carryover_achievement_percent,
                },
            },
            "efficiency_kpis": {
                "separator_efficiency": {
                    "value": self.avg_separator_efficiency,
                    "target": self.efficiency_target,
                    "achievement_percent": self.efficiency_achievement_percent,
                },
            },
            "availability_kpis": {
                "system_availability": {
                    "value": self.system_availability_percent,
                    "target": self.availability_target,
                    "achievement_percent": self.availability_achievement_percent,
                },
            },
            "response_kpis": {
                "alert_response_time": {
                    "value_minutes": self.avg_alert_response_time_minutes,
                    "target_minutes": self.response_target_minutes,
                    "achievement_percent": self.response_achievement_percent,
                },
            },
            "performance": {
                "calculations_total": self.calculations_total,
                "success_rate": self.calculation_success_rate,
                "avg_time_ms": self.avg_calculation_time_ms,
            },
            "history": {
                "dryness": self.dryness_kpi_history.to_dict(),
                "efficiency": self.efficiency_kpi_history.to_dict(),
                "availability": self.availability_history.to_dict(),
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


# =============================================================================
# Dashboard Provider
# =============================================================================

class SteamQualityDashboardProvider:
    """
    Dashboard data provider for steam quality monitoring UI.

    This class aggregates data from metrics collectors, alert managers, and
    other sources to provide comprehensive dashboard views for the steam
    quality control UI.

    Attributes:
        metrics_collector: Optional SteamQualityMetricsCollector instance
        alert_manager: Optional SteamQualityAlertManager instance

    Example:
        >>> provider = SteamQualityDashboardProvider(metrics_collector, alert_manager)
        >>> real_time = await provider.get_real_time_dashboard()
        >>> separator = await provider.get_separator_dashboard("SEP-001")
    """

    def __init__(
        self,
        metrics_collector: Optional[Any] = None,
        alert_manager: Optional[Any] = None,
        data_store: Optional[Any] = None,
    ) -> None:
        """
        Initialize SteamQualityDashboardProvider.

        Args:
            metrics_collector: SteamQualityMetricsCollector instance
            alert_manager: SteamQualityAlertManager instance
            data_store: Optional data store for historical data
        """
        self._metrics = metrics_collector
        self._alerts = alert_manager
        self._data_store = data_store

        # Cache for dashboard data
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl_s = 5.0  # 5 second cache

        logger.info("SteamQualityDashboardProvider initialized")

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached dashboard data if not expired."""
        if cache_key not in self._cache:
            return None

        cached_time, data = self._cache[cache_key]
        age = (datetime.now(timezone.utc) - cached_time).total_seconds()

        if age > self._cache_ttl_s:
            del self._cache[cache_key]
            return None

        return data

    def _set_cached(self, cache_key: str, data: Any) -> None:
        """Cache dashboard data."""
        self._cache[cache_key] = (datetime.now(timezone.utc), data)

    def _extract_metric_value(
        self,
        metrics: Dict[str, Any],
        metric_name: str,
        default: float = 0.0,
    ) -> float:
        """Extract metric value from metrics dictionary."""
        for key, value in metrics.items():
            if metric_name in key:
                return float(value.get("value", default))
        return default

    # =========================================================================
    # Real-Time Dashboard
    # =========================================================================

    async def get_real_time_dashboard(self) -> RealTimeQualityDashboard:
        """
        Get real-time dashboard data for all separators.

        Returns:
            RealTimeQualityDashboard with current system status
        """
        cache_key = "real_time"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = RealTimeQualityDashboard()

        # Populate from metrics collector
        if self._metrics:
            metrics = self._metrics.get_metrics()
            summary = self._metrics.get_metrics_summary(timedelta(hours=1))

            dashboard.avg_dryness_fraction = summary.avg_dryness_fraction
            dashboard.min_dryness_fraction = summary.min_dryness_fraction
            dashboard.max_carryover_risk = summary.max_carryover_risk
            dashboard.avg_separator_efficiency = summary.avg_separator_efficiency

            # Build separator statuses from metrics
            separator_ids = set()
            for key in metrics:
                if "separator_id" in str(metrics[key].get("labels", {})):
                    sep_id = metrics[key]["labels"].get("separator_id")
                    if sep_id:
                        separator_ids.add(sep_id)

            for sep_id in separator_ids:
                status = SeparatorStatus(
                    separator_id=sep_id,
                    dryness_fraction=self._extract_metric_value(
                        metrics, f"dryness_fraction{{separator_id=\"{sep_id}\""
                    ),
                    carryover_risk=self._extract_metric_value(
                        metrics, f"carryover_risk{{separator_id=\"{sep_id}\""
                    ),
                    efficiency_percent=self._extract_metric_value(
                        metrics, f"separator_efficiency{{separator_id=\"{sep_id}\""
                    ),
                )

                # Determine status
                if status.dryness_fraction < 0.90 or status.carryover_risk > 0.7:
                    status.status = "critical"
                elif status.dryness_fraction < 0.95 or status.carryover_risk > 0.3:
                    status.status = "degraded"
                else:
                    status.status = "operational"

                dashboard.separator_statuses.append(status)

            dashboard.total_separators = len(dashboard.separator_statuses)
            dashboard.operational_separators = len([
                s for s in dashboard.separator_statuses if s.status == "operational"
            ])
            dashboard.degraded_separators = len([
                s for s in dashboard.separator_statuses if s.status == "degraded"
            ])
            dashboard.critical_separators = len([
                s for s in dashboard.separator_statuses if s.status == "critical"
            ])

        # Populate from alert manager
        if self._alerts:
            active_alerts = self._alerts.get_active_alerts()
            dashboard.active_alerts_count = len(active_alerts)
            dashboard.critical_alerts_count = len([
                a for a in active_alerts if a.severity.value >= 3
            ])
            dashboard.warning_alerts_count = len([
                a for a in active_alerts if a.severity.value == 2
            ])

        # Determine system status
        if dashboard.critical_separators > 0 or dashboard.critical_alerts_count > 0:
            dashboard.system_status = "critical"
        elif dashboard.degraded_separators > 0 or dashboard.warning_alerts_count > 0:
            dashboard.system_status = "degraded"
        else:
            dashboard.system_status = "operational"

        # Create widgets
        dashboard.widgets = self._create_real_time_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_real_time_widgets(
        self,
        dashboard: RealTimeQualityDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for real-time dashboard."""
        widgets = []

        # System status gauge
        widgets.append(DashboardWidget(
            widget_id="system_status",
            widget_type="gauge",
            title="System Status",
            data={
                "value": dashboard.system_status,
                "color": {
                    "operational": "green",
                    "degraded": "yellow",
                    "critical": "red",
                    "offline": "gray",
                }.get(dashboard.system_status, "gray"),
            },
            position={"row": 0, "col": 0, "width": 2, "height": 1},
        ))

        # Average dryness gauge
        widgets.append(DashboardWidget(
            widget_id="avg_dryness",
            widget_type="gauge",
            title="Avg Dryness Fraction",
            data={
                "value": dashboard.avg_dryness_fraction,
                "min": 0.85,
                "max": 1.0,
                "thresholds": {"green": 0.97, "yellow": 0.95, "red": 0.0},
            },
            position={"row": 0, "col": 2, "width": 2, "height": 1},
        ))

        # Max carryover risk gauge
        widgets.append(DashboardWidget(
            widget_id="max_carryover",
            widget_type="gauge",
            title="Max Carryover Risk",
            data={
                "value": dashboard.max_carryover_risk,
                "min": 0.0,
                "max": 1.0,
                "thresholds": {"green": 0.0, "yellow": 0.3, "red": 0.5},
            },
            position={"row": 0, "col": 4, "width": 2, "height": 1},
        ))

        # Active alerts stat
        widgets.append(DashboardWidget(
            widget_id="active_alerts",
            widget_type="stat",
            title="Active Alerts",
            data={
                "value": dashboard.active_alerts_count,
                "critical": dashboard.critical_alerts_count,
                "warning": dashboard.warning_alerts_count,
            },
            position={"row": 0, "col": 6, "width": 2, "height": 1},
        ))

        # Separator status summary
        widgets.append(DashboardWidget(
            widget_id="separator_summary",
            widget_type="chart",
            title="Separator Status",
            data={
                "type": "donut",
                "series": [
                    {"name": "Operational", "value": dashboard.operational_separators, "color": "green"},
                    {"name": "Degraded", "value": dashboard.degraded_separators, "color": "yellow"},
                    {"name": "Critical", "value": dashboard.critical_separators, "color": "red"},
                ],
            },
            position={"row": 1, "col": 0, "width": 3, "height": 2},
        ))

        # Separator status table
        widgets.append(DashboardWidget(
            widget_id="separator_table",
            widget_type="table",
            title="Separator Details",
            data={
                "columns": [
                    {"id": "separator_id", "label": "Separator"},
                    {"id": "status", "label": "Status"},
                    {"id": "dryness_fraction", "label": "Dryness"},
                    {"id": "carryover_risk", "label": "Carryover Risk"},
                    {"id": "efficiency_percent", "label": "Efficiency"},
                ],
                "rows": [s.to_dict() for s in dashboard.separator_statuses],
            },
            position={"row": 1, "col": 3, "width": 5, "height": 2},
        ))

        return widgets

    # =========================================================================
    # Separator Dashboard
    # =========================================================================

    async def get_separator_dashboard(
        self,
        separator_id: str,
        time_window: Optional[timedelta] = None,
    ) -> SeparatorDashboard:
        """
        Get dashboard data for a specific separator.

        Args:
            separator_id: Separator identifier
            time_window: Time window for historical data (default: 1 hour)

        Returns:
            SeparatorDashboard with separator-specific metrics
        """
        if time_window is None:
            time_window = timedelta(hours=1)

        cache_key = f"separator:{separator_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = SeparatorDashboard(separator_id=separator_id)

        # Populate from metrics collector
        if self._metrics:
            metrics = self._metrics.get_metrics()

            # Find metrics for this separator
            for key, value in metrics.items():
                labels = value.get("labels", {})
                if labels.get("separator_id") == separator_id:
                    metric_name = value.get("name", "")
                    metric_value = value.get("value", 0.0)

                    if "dryness_fraction" in metric_name:
                        dashboard.dryness_fraction = metric_value
                        dashboard.moisture_content_percent = (1.0 - metric_value) * 100
                    elif "carryover_risk" in metric_name:
                        dashboard.carryover_risk = metric_value
                    elif "separator_efficiency" in metric_name:
                        dashboard.efficiency_percent = metric_value
                    elif "separator_level" in metric_name:
                        dashboard.level_percent = metric_value
                    elif "drain_rate" in metric_name:
                        dashboard.drain_rate_kg_h = metric_value
                    elif "inlet_pressure" in metric_name:
                        dashboard.inlet_pressure_bar = metric_value
                    elif "outlet_pressure" in metric_name:
                        dashboard.outlet_pressure_bar = metric_value
                    elif "steam_flow" in metric_name:
                        dashboard.steam_flow_kg_h = metric_value
                    elif "water_hammer_risk" in metric_name:
                        dashboard.water_hammer_risk = metric_value
                    elif "flooding_risk" in metric_name:
                        dashboard.flooding_risk = metric_value

        # Determine status
        if dashboard.dryness_fraction < 0.90 or dashboard.carryover_risk > 0.7:
            dashboard.status = "critical"
        elif dashboard.dryness_fraction < 0.95 or dashboard.carryover_risk > 0.3:
            dashboard.status = "degraded"
        else:
            dashboard.status = "operational"

        # Populate from alert manager
        if self._alerts:
            from .alerting import AlertFilter
            filters = AlertFilter(separator_id=separator_id)
            active_alerts = self._alerts.get_active_alerts(filters)
            dashboard.active_alerts = len(active_alerts)
            dashboard.recent_alerts = [
                {
                    "alert_id": a.alert_id[:8],
                    "type": a.alert_type.value,
                    "severity": a.severity.name_display,
                    "message": a.message[:100],
                    "created_at": a.created_at.isoformat(),
                }
                for a in active_alerts[:5]
            ]

        # Create widgets
        dashboard.widgets = self._create_separator_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_separator_widgets(
        self,
        dashboard: SeparatorDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for separator dashboard."""
        widgets = []

        # Status indicator
        widgets.append(DashboardWidget(
            widget_id="status",
            widget_type="stat",
            title="Status",
            data={
                "value": dashboard.status.upper(),
                "color": {
                    "operational": "green",
                    "degraded": "yellow",
                    "critical": "red",
                }.get(dashboard.status, "gray"),
            },
            position={"row": 0, "col": 0, "width": 1, "height": 1},
        ))

        # Dryness gauge
        widgets.append(DashboardWidget(
            widget_id="dryness",
            widget_type="gauge",
            title="Dryness Fraction",
            data={
                "value": dashboard.dryness_fraction,
                "min": 0.85,
                "max": 1.0,
                "thresholds": {"green": 0.97, "yellow": 0.95, "red": 0.0},
            },
            position={"row": 0, "col": 1, "width": 2, "height": 1},
        ))

        # Carryover risk gauge
        widgets.append(DashboardWidget(
            widget_id="carryover",
            widget_type="gauge",
            title="Carryover Risk",
            data={
                "value": dashboard.carryover_risk,
                "min": 0.0,
                "max": 1.0,
                "thresholds": {"green": 0.0, "yellow": 0.3, "red": 0.5},
            },
            position={"row": 0, "col": 3, "width": 2, "height": 1},
        ))

        # Efficiency gauge
        widgets.append(DashboardWidget(
            widget_id="efficiency",
            widget_type="gauge",
            title="Efficiency",
            data={
                "value": dashboard.efficiency_percent,
                "min": 80,
                "max": 100,
                "unit": "%",
                "thresholds": {"green": 95, "yellow": 90, "red": 0},
            },
            position={"row": 0, "col": 5, "width": 2, "height": 1},
        ))

        # Level gauge
        widgets.append(DashboardWidget(
            widget_id="level",
            widget_type="gauge",
            title="Level",
            data={
                "value": dashboard.level_percent,
                "min": 0,
                "max": 100,
                "unit": "%",
                "thresholds": {"green": 0, "yellow": 70, "red": 85},
            },
            position={"row": 0, "col": 7, "width": 1, "height": 1},
        ))

        # Historical trend chart
        widgets.append(DashboardWidget(
            widget_id="dryness_trend",
            widget_type="chart",
            title="Dryness Trend",
            data={
                "type": "line",
                "series": dashboard.dryness_history.to_dict(),
            },
            position={"row": 1, "col": 0, "width": 4, "height": 2},
        ))

        # Carryover trend chart
        widgets.append(DashboardWidget(
            widget_id="carryover_trend",
            widget_type="chart",
            title="Carryover Risk Trend",
            data={
                "type": "line",
                "series": dashboard.carryover_history.to_dict(),
            },
            position={"row": 1, "col": 4, "width": 4, "height": 2},
        ))

        return widgets

    # =========================================================================
    # Quality Trend Dashboard
    # =========================================================================

    async def get_quality_trend_dashboard(
        self,
        time_window: Optional[timedelta] = None,
    ) -> QualityTrendDashboard:
        """
        Get quality trend dashboard data.

        Args:
            time_window: Time window for trend analysis (default: 24 hours)

        Returns:
            QualityTrendDashboard with trend analysis
        """
        if time_window is None:
            time_window = timedelta(hours=24)

        now = datetime.now(timezone.utc)
        start = now - time_window

        cache_key = f"trend:{time_window.total_seconds()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = QualityTrendDashboard(
            time_window_start=start,
            time_window_end=now,
        )

        # Populate from metrics collector
        if self._metrics:
            summary = self._metrics.get_metrics_summary(time_window)

            dashboard.avg_dryness_fraction = summary.avg_dryness_fraction
            dashboard.min_dryness_fraction = summary.min_dryness_fraction
            dashboard.max_dryness_fraction = summary.max_dryness_fraction

            dashboard.avg_carryover_risk = summary.avg_carryover_risk
            dashboard.max_carryover_risk = summary.max_carryover_risk
            dashboard.carryover_events = summary.carryover_events_count

            dashboard.avg_efficiency_percent = summary.avg_separator_efficiency
            dashboard.min_efficiency_percent = summary.min_separator_efficiency

            dashboard.low_dryness_events = summary.low_dryness_events
            dashboard.high_moisture_events = summary.high_moisture_events
            dashboard.separator_flooding_events = summary.separator_flooding_events
            dashboard.water_hammer_events = summary.water_hammer_risk_events
            dashboard.total_quality_events = summary.total_events

        # Create widgets
        dashboard.widgets = self._create_trend_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_trend_widgets(
        self,
        dashboard: QualityTrendDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for trend dashboard."""
        widgets = []

        # Trend indicators
        widgets.append(DashboardWidget(
            widget_id="dryness_trend_indicator",
            widget_type="stat",
            title="Dryness Trend",
            data={
                "trend": dashboard.dryness_trend,
                "value": dashboard.avg_dryness_fraction,
                "icon": {
                    "improving": "arrow_up",
                    "stable": "minus",
                    "declining": "arrow_down",
                }.get(dashboard.dryness_trend, "minus"),
            },
            position={"row": 0, "col": 0, "width": 2, "height": 1},
        ))

        widgets.append(DashboardWidget(
            widget_id="carryover_trend_indicator",
            widget_type="stat",
            title="Carryover Trend",
            data={
                "trend": dashboard.carryover_trend,
                "value": dashboard.avg_carryover_risk,
            },
            position={"row": 0, "col": 2, "width": 2, "height": 1},
        ))

        widgets.append(DashboardWidget(
            widget_id="efficiency_trend_indicator",
            widget_type="stat",
            title="Efficiency Trend",
            data={
                "trend": dashboard.efficiency_trend,
                "value": dashboard.avg_efficiency_percent,
            },
            position={"row": 0, "col": 4, "width": 2, "height": 1},
        ))

        # Event count summary
        widgets.append(DashboardWidget(
            widget_id="event_summary",
            widget_type="stat",
            title="Quality Events",
            data={
                "total": dashboard.total_quality_events,
                "by_type": {
                    "low_dryness": dashboard.low_dryness_events,
                    "high_moisture": dashboard.high_moisture_events,
                    "flooding": dashboard.separator_flooding_events,
                    "water_hammer": dashboard.water_hammer_events,
                },
            },
            position={"row": 0, "col": 6, "width": 2, "height": 1},
        ))

        # Trend chart
        widgets.append(DashboardWidget(
            widget_id="quality_trend_chart",
            widget_type="chart",
            title="Quality Metrics Trend",
            data={
                "type": "multi_line",
                "series": [
                    dashboard.dryness_trend_data.to_dict(),
                    dashboard.carryover_trend_data.to_dict(),
                ],
            },
            position={"row": 1, "col": 0, "width": 8, "height": 2},
        ))

        return widgets

    # =========================================================================
    # Alert Summary Dashboard
    # =========================================================================

    async def get_alert_summary_dashboard(self) -> AlertSummaryDashboard:
        """
        Get alert summary dashboard data.

        Returns:
            AlertSummaryDashboard with alert statistics
        """
        cache_key = "alert_summary"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = AlertSummaryDashboard()

        if self._alerts:
            active_alerts = self._alerts.get_active_alerts()
            stats = self._alerts.get_statistics()

            dashboard.total_active = len(active_alerts)
            dashboard.critical_count = len([a for a in active_alerts if a.severity.value == 3])
            dashboard.warning_count = len([a for a in active_alerts if a.severity.value == 2])
            dashboard.advisory_count = len([a for a in active_alerts if a.severity.value == 1])
            dashboard.info_count = len([a for a in active_alerts if a.severity.value == 0])

            # Count by type
            for a in active_alerts:
                type_name = a.alert_type.value
                if type_name == "low_dryness":
                    dashboard.low_dryness_active += 1
                elif type_name == "high_moisture":
                    dashboard.high_moisture_active += 1
                elif type_name == "carryover_risk":
                    dashboard.carryover_risk_active += 1
                elif type_name == "separator_flooding":
                    dashboard.separator_flooding_active += 1
                elif type_name == "water_hammer_risk":
                    dashboard.water_hammer_active += 1
                elif type_name == "data_quality_degraded":
                    dashboard.data_quality_active += 1

            dashboard.alerts_created_24h = stats.get("alerts_created", 0)
            dashboard.alerts_resolved_24h = stats.get("alerts_resolved", 0)
            dashboard.alerts_acknowledged_24h = stats.get("alerts_acknowledged", 0)

            # Active alerts detail
            dashboard.active_alerts = [
                a.to_dict() for a in active_alerts[:20]
            ]

        # Create widgets
        dashboard.widgets = self._create_alert_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_alert_widgets(
        self,
        dashboard: AlertSummaryDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for alert summary dashboard."""
        widgets = []

        # Alert count by severity
        widgets.append(DashboardWidget(
            widget_id="severity_breakdown",
            widget_type="chart",
            title="Alerts by Severity",
            data={
                "type": "bar",
                "series": [
                    {"name": "Critical", "value": dashboard.critical_count, "color": "red"},
                    {"name": "Warning", "value": dashboard.warning_count, "color": "orange"},
                    {"name": "Advisory", "value": dashboard.advisory_count, "color": "yellow"},
                    {"name": "Info", "value": dashboard.info_count, "color": "blue"},
                ],
            },
            position={"row": 0, "col": 0, "width": 4, "height": 2},
        ))

        # Alert count by type
        widgets.append(DashboardWidget(
            widget_id="type_breakdown",
            widget_type="chart",
            title="Alerts by Type",
            data={
                "type": "pie",
                "series": [
                    {"name": "Low Dryness", "value": dashboard.low_dryness_active},
                    {"name": "High Moisture", "value": dashboard.high_moisture_active},
                    {"name": "Carryover Risk", "value": dashboard.carryover_risk_active},
                    {"name": "Flooding", "value": dashboard.separator_flooding_active},
                    {"name": "Water Hammer", "value": dashboard.water_hammer_active},
                    {"name": "Data Quality", "value": dashboard.data_quality_active},
                ],
            },
            position={"row": 0, "col": 4, "width": 4, "height": 2},
        ))

        # Active alerts table
        widgets.append(DashboardWidget(
            widget_id="active_alerts_table",
            widget_type="table",
            title="Active Alerts",
            data={
                "columns": [
                    {"id": "severity_code", "label": "Severity"},
                    {"id": "alert_type", "label": "Type"},
                    {"id": "source", "label": "Source"},
                    {"id": "message", "label": "Message"},
                    {"id": "created_at", "label": "Time"},
                ],
                "rows": dashboard.active_alerts,
            },
            position={"row": 2, "col": 0, "width": 8, "height": 3},
        ))

        return widgets

    # =========================================================================
    # KPI Dashboard
    # =========================================================================

    async def get_kpi_dashboard(
        self,
        time_window: Optional[timedelta] = None,
    ) -> KPIDashboard:
        """
        Get KPI dashboard data.

        Args:
            time_window: Time window for KPI calculation (default: 24 hours)

        Returns:
            KPIDashboard with KPI metrics
        """
        if time_window is None:
            time_window = timedelta(hours=24)

        now = datetime.now(timezone.utc)
        start = now - time_window

        cache_key = f"kpi:{time_window.total_seconds()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = KPIDashboard(
            time_window_start=start,
            time_window_end=now,
        )

        # Populate from metrics collector
        if self._metrics:
            summary = self._metrics.get_metrics_summary(time_window)

            dashboard.avg_dryness_fraction = summary.avg_dryness_fraction
            dashboard.dryness_achievement_percent = min(
                100.0,
                (dashboard.avg_dryness_fraction / dashboard.dryness_target) * 100
            )

            dashboard.carryover_events_count = summary.carryover_events_count
            dashboard.carryover_achievement_percent = max(
                0.0,
                100.0 - (summary.carryover_events_count * 10)  # -10% per event
            )

            dashboard.avg_separator_efficiency = summary.avg_separator_efficiency
            dashboard.efficiency_achievement_percent = min(
                100.0,
                (summary.avg_separator_efficiency / dashboard.efficiency_target) * 100
            )

            dashboard.calculations_total = summary.total_calculations
            dashboard.calculation_success_rate = (
                (1.0 - summary.calculation_error_rate) * 100
            )
            dashboard.avg_calculation_time_ms = summary.avg_calculation_time_s * 1000

        # Create widgets
        dashboard.widgets = self._create_kpi_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_kpi_widgets(
        self,
        dashboard: KPIDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for KPI dashboard."""
        widgets = []

        # Dryness KPI
        widgets.append(DashboardWidget(
            widget_id="dryness_kpi",
            widget_type="gauge",
            title="Dryness Achievement",
            data={
                "value": dashboard.dryness_achievement_percent,
                "target": 100,
                "unit": "%",
                "actual_value": dashboard.avg_dryness_fraction,
                "target_value": dashboard.dryness_target,
            },
            position={"row": 0, "col": 0, "width": 2, "height": 1},
        ))

        # Efficiency KPI
        widgets.append(DashboardWidget(
            widget_id="efficiency_kpi",
            widget_type="gauge",
            title="Efficiency Achievement",
            data={
                "value": dashboard.efficiency_achievement_percent,
                "target": 100,
                "unit": "%",
                "actual_value": dashboard.avg_separator_efficiency,
                "target_value": dashboard.efficiency_target,
            },
            position={"row": 0, "col": 2, "width": 2, "height": 1},
        ))

        # Carryover events
        widgets.append(DashboardWidget(
            widget_id="carryover_kpi",
            widget_type="stat",
            title="Carryover Events",
            data={
                "value": dashboard.carryover_events_count,
                "target": dashboard.carryover_target,
                "achievement_percent": dashboard.carryover_achievement_percent,
            },
            position={"row": 0, "col": 4, "width": 2, "height": 1},
        ))

        # Calculation performance
        widgets.append(DashboardWidget(
            widget_id="calc_performance",
            widget_type="stat",
            title="Calculation Performance",
            data={
                "total": dashboard.calculations_total,
                "success_rate": dashboard.calculation_success_rate,
                "avg_time_ms": dashboard.avg_calculation_time_ms,
            },
            position={"row": 0, "col": 6, "width": 2, "height": 1},
        ))

        # KPI trend chart
        widgets.append(DashboardWidget(
            widget_id="kpi_trend",
            widget_type="chart",
            title="KPI Trends",
            data={
                "type": "multi_line",
                "series": [
                    dashboard.dryness_kpi_history.to_dict(),
                    dashboard.efficiency_kpi_history.to_dict(),
                ],
            },
            position={"row": 1, "col": 0, "width": 8, "height": 2},
        ))

        return widgets
