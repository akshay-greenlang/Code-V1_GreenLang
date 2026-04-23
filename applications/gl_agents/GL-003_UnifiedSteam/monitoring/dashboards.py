"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Dashboard Data Provider

This module provides dashboard data aggregation for UI consumption, including
real-time dashboards, KPI dashboards, optimization dashboards, trap health
dashboards, and climate impact dashboards.

Dashboard Types:
    - RealTimeDashboard: Live steam system status
    - KPIDashboard: Key performance indicators over time
    - OptimizationDashboard: Optimization results and recommendations
    - TrapHealthDashboard: Steam trap health and failure analysis
    - ClimateImpactDashboard: Environmental impact metrics

Example:
    >>> provider = DashboardProvider(metrics_collector)
    >>> real_time = await provider.get_real_time_dashboard("SITE-001")
    >>> kpi = await provider.get_kpi_dashboard(timedelta(hours=24))
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


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
    widget_type: str  # gauge, chart, table, alert, stat
    title: str
    data: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)  # row, col, width, height
    refresh_interval_s: float = 30.0
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
class RealTimeDashboard:
    """Real-time dashboard data for live monitoring."""
    site_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # System status
    system_status: str = "operational"  # operational, degraded, critical, offline
    active_alerts_count: int = 0
    critical_alerts_count: int = 0

    # Steam metrics
    steam_flow_kg_h: float = 0.0
    steam_pressure_bar: float = 0.0
    steam_temperature_c: float = 0.0
    steam_quality_percent: float = 100.0

    # Efficiency metrics
    boiler_efficiency_percent: float = 0.0
    condensate_recovery_percent: float = 0.0
    overall_efficiency_percent: float = 0.0

    # Trap status summary
    total_traps: int = 0
    healthy_traps: int = 0
    failed_traps: int = 0
    estimated_loss_kg_h: float = 0.0

    # Desuperheater status
    desuperheater_active: bool = False
    desuperheater_outlet_temp_c: float = 0.0
    desuperheater_target_temp_c: float = 0.0

    # Recent activity
    recent_optimizations: int = 0
    pending_recommendations: int = 0

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "site_id": self.site_id,
            "timestamp": self.timestamp.isoformat(),
            "system_status": self.system_status,
            "alerts": {
                "active": self.active_alerts_count,
                "critical": self.critical_alerts_count,
            },
            "steam": {
                "flow_kg_h": self.steam_flow_kg_h,
                "pressure_bar": self.steam_pressure_bar,
                "temperature_c": self.steam_temperature_c,
                "quality_percent": self.steam_quality_percent,
            },
            "efficiency": {
                "boiler_percent": self.boiler_efficiency_percent,
                "condensate_recovery_percent": self.condensate_recovery_percent,
                "overall_percent": self.overall_efficiency_percent,
            },
            "traps": {
                "total": self.total_traps,
                "healthy": self.healthy_traps,
                "failed": self.failed_traps,
                "estimated_loss_kg_h": self.estimated_loss_kg_h,
            },
            "desuperheater": {
                "active": self.desuperheater_active,
                "outlet_temp_c": self.desuperheater_outlet_temp_c,
                "target_temp_c": self.desuperheater_target_temp_c,
            },
            "activity": {
                "recent_optimizations": self.recent_optimizations,
                "pending_recommendations": self.pending_recommendations,
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class KPIDashboard:
    """KPI dashboard data for performance monitoring."""
    time_window_start: datetime
    time_window_end: datetime
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Efficiency KPIs
    avg_boiler_efficiency_percent: float = 0.0
    avg_condensate_recovery_percent: float = 0.0
    avg_overall_efficiency_percent: float = 0.0
    efficiency_trend: str = "stable"  # improving, stable, declining

    # Energy KPIs
    total_steam_produced_kg: float = 0.0
    total_energy_consumed_kwh: float = 0.0
    energy_intensity_kwh_per_kg: float = 0.0

    # Cost KPIs
    total_operating_cost_usd: float = 0.0
    cost_per_kg_steam_usd: float = 0.0
    savings_realized_usd: float = 0.0

    # Reliability KPIs
    system_availability_percent: float = 0.0
    trap_failure_rate_percent: float = 0.0
    unplanned_downtime_hours: float = 0.0

    # Time series data
    efficiency_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="efficiency", unit="%")
    )
    energy_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="energy_consumption", unit="kWh")
    )
    cost_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="operating_cost", unit="USD")
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
            "efficiency": {
                "avg_boiler_percent": self.avg_boiler_efficiency_percent,
                "avg_condensate_recovery_percent": self.avg_condensate_recovery_percent,
                "avg_overall_percent": self.avg_overall_efficiency_percent,
                "trend": self.efficiency_trend,
                "history": self.efficiency_history.to_dict(),
            },
            "energy": {
                "total_steam_produced_kg": self.total_steam_produced_kg,
                "total_energy_consumed_kwh": self.total_energy_consumed_kwh,
                "intensity_kwh_per_kg": self.energy_intensity_kwh_per_kg,
                "history": self.energy_history.to_dict(),
            },
            "cost": {
                "total_operating_usd": self.total_operating_cost_usd,
                "per_kg_steam_usd": self.cost_per_kg_steam_usd,
                "savings_realized_usd": self.savings_realized_usd,
                "history": self.cost_history.to_dict(),
            },
            "reliability": {
                "availability_percent": self.system_availability_percent,
                "trap_failure_rate_percent": self.trap_failure_rate_percent,
                "unplanned_downtime_hours": self.unplanned_downtime_hours,
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class OptimizationDashboard:
    """Optimization dashboard data."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optimization summary
    optimizations_today: int = 0
    optimizations_successful: int = 0
    success_rate_percent: float = 0.0

    # Savings summary
    energy_savings_kwh_today: float = 0.0
    cost_savings_usd_today: float = 0.0
    co2_reduction_kg_today: float = 0.0
    projected_annual_savings_usd: float = 0.0

    # Active optimizations
    active_desuperheater_optimizations: int = 0
    active_condensate_optimizations: int = 0
    active_trap_optimizations: int = 0

    # Recommendations
    pending_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_acceptance_rate_percent: float = 0.0

    # Performance
    avg_optimization_time_ms: float = 0.0
    p95_optimization_time_ms: float = 0.0

    # History
    savings_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="daily_savings", unit="USD")
    )
    optimization_count_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="optimization_count", unit="count")
    )

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "optimizations_today": self.optimizations_today,
                "optimizations_successful": self.optimizations_successful,
                "success_rate_percent": self.success_rate_percent,
            },
            "savings": {
                "energy_kwh_today": self.energy_savings_kwh_today,
                "cost_usd_today": self.cost_savings_usd_today,
                "co2_kg_today": self.co2_reduction_kg_today,
                "projected_annual_usd": self.projected_annual_savings_usd,
                "history": self.savings_history.to_dict(),
            },
            "active_optimizations": {
                "desuperheater": self.active_desuperheater_optimizations,
                "condensate": self.active_condensate_optimizations,
                "trap": self.active_trap_optimizations,
            },
            "recommendations": {
                "pending": self.pending_recommendations,
                "acceptance_rate_percent": self.recommendation_acceptance_rate_percent,
            },
            "performance": {
                "avg_time_ms": self.avg_optimization_time_ms,
                "p95_time_ms": self.p95_optimization_time_ms,
                "history": self.optimization_count_history.to_dict(),
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class TrapHealthDashboard:
    """Steam trap health dashboard data."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Overall status
    total_traps: int = 0
    operational_traps: int = 0
    failed_traps: int = 0
    suspect_traps: int = 0
    unknown_status_traps: int = 0

    # Failure breakdown
    blow_through_count: int = 0
    blocked_count: int = 0
    leaking_count: int = 0

    # Detection metrics
    detection_accuracy_percent: float = 0.0
    detection_precision_percent: float = 0.0
    detection_recall_percent: float = 0.0

    # Loss metrics
    total_steam_loss_kg_h: float = 0.0
    total_energy_loss_kw: float = 0.0
    estimated_annual_loss_usd: float = 0.0

    # Trap details (top failures)
    top_failing_traps: List[Dict[str, Any]] = field(default_factory=list)

    # Historical data
    failure_rate_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="failure_rate", unit="%")
    )
    loss_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="steam_loss", unit="kg/h")
    )

    # Maintenance queue
    maintenance_queue: List[Dict[str, Any]] = field(default_factory=list)
    maintenance_queue_count: int = 0

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": {
                "total": self.total_traps,
                "operational": self.operational_traps,
                "failed": self.failed_traps,
                "suspect": self.suspect_traps,
                "unknown": self.unknown_status_traps,
            },
            "failure_breakdown": {
                "blow_through": self.blow_through_count,
                "blocked": self.blocked_count,
                "leaking": self.leaking_count,
            },
            "detection_metrics": {
                "accuracy_percent": self.detection_accuracy_percent,
                "precision_percent": self.detection_precision_percent,
                "recall_percent": self.detection_recall_percent,
            },
            "losses": {
                "steam_loss_kg_h": self.total_steam_loss_kg_h,
                "energy_loss_kw": self.total_energy_loss_kw,
                "estimated_annual_loss_usd": self.estimated_annual_loss_usd,
            },
            "top_failing_traps": self.top_failing_traps,
            "history": {
                "failure_rate": self.failure_rate_history.to_dict(),
                "steam_loss": self.loss_history.to_dict(),
            },
            "maintenance": {
                "queue": self.maintenance_queue,
                "queue_count": self.maintenance_queue_count,
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class ClimateImpactDashboard:
    """Climate and environmental impact dashboard data."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reporting_period_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30)
    )
    reporting_period_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # CO2 emissions
    total_co2_emissions_kg: float = 0.0
    co2_emissions_avoided_kg: float = 0.0
    co2_intensity_kg_per_ton_steam: float = 0.0
    co2_reduction_percent: float = 0.0

    # Energy metrics
    total_energy_consumed_kwh: float = 0.0
    energy_saved_kwh: float = 0.0
    renewable_energy_percent: float = 0.0

    # Water metrics
    water_consumed_m3: float = 0.0
    water_recycled_m3: float = 0.0
    water_recycling_rate_percent: float = 0.0

    # Fuel metrics
    fuel_consumed_units: float = 0.0  # natural gas m3 or equivalent
    fuel_type: str = "natural_gas"
    fuel_saved_units: float = 0.0

    # Compliance
    ghg_protocol_scope1_kg: float = 0.0
    ghg_protocol_scope2_kg: float = 0.0
    iso50001_compliant: bool = False

    # Time series
    co2_emissions_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="co2_emissions", unit="kg")
    )
    energy_consumption_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="energy_consumption", unit="kWh")
    )
    co2_intensity_history: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="co2_intensity", unit="kg/ton")
    )

    # Targets and progress
    annual_co2_target_kg: float = 0.0
    annual_co2_progress_percent: float = 0.0
    annual_energy_target_kwh: float = 0.0
    annual_energy_progress_percent: float = 0.0

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reporting_period": {
                "start": self.reporting_period_start.isoformat(),
                "end": self.reporting_period_end.isoformat(),
            },
            "co2": {
                "total_emissions_kg": self.total_co2_emissions_kg,
                "emissions_avoided_kg": self.co2_emissions_avoided_kg,
                "intensity_kg_per_ton_steam": self.co2_intensity_kg_per_ton_steam,
                "reduction_percent": self.co2_reduction_percent,
                "history": self.co2_emissions_history.to_dict(),
            },
            "energy": {
                "total_consumed_kwh": self.total_energy_consumed_kwh,
                "saved_kwh": self.energy_saved_kwh,
                "renewable_percent": self.renewable_energy_percent,
                "history": self.energy_consumption_history.to_dict(),
            },
            "water": {
                "consumed_m3": self.water_consumed_m3,
                "recycled_m3": self.water_recycled_m3,
                "recycling_rate_percent": self.water_recycling_rate_percent,
            },
            "fuel": {
                "consumed_units": self.fuel_consumed_units,
                "type": self.fuel_type,
                "saved_units": self.fuel_saved_units,
            },
            "compliance": {
                "ghg_scope1_kg": self.ghg_protocol_scope1_kg,
                "ghg_scope2_kg": self.ghg_protocol_scope2_kg,
                "iso50001_compliant": self.iso50001_compliant,
            },
            "targets": {
                "annual_co2_target_kg": self.annual_co2_target_kg,
                "annual_co2_progress_percent": self.annual_co2_progress_percent,
                "annual_energy_target_kwh": self.annual_energy_target_kwh,
                "annual_energy_progress_percent": self.annual_energy_progress_percent,
            },
            "co2_intensity_history": self.co2_intensity_history.to_dict(),
            "widgets": [w.to_dict() for w in self.widgets],
        }


class DashboardProvider:
    """
    Dashboard data provider for UI consumption.

    This class aggregates data from metrics collectors, alert managers, and
    other sources to provide comprehensive dashboard views for the steam
    system optimization UI.

    Attributes:
        metrics_collector: Optional MetricsCollector instance
        alert_manager: Optional AlertManager instance

    Example:
        >>> provider = DashboardProvider(metrics_collector, alert_manager)
        >>> real_time = await provider.get_real_time_dashboard("SITE-001")
        >>> kpi = await provider.get_kpi_dashboard(timedelta(hours=24))
    """

    def __init__(
        self,
        metrics_collector: Optional[Any] = None,
        alert_manager: Optional[Any] = None,
        data_store: Optional[Any] = None,
    ) -> None:
        """
        Initialize DashboardProvider.

        Args:
            metrics_collector: MetricsCollector instance for metric data
            alert_manager: AlertManager instance for alert data
            data_store: Optional data store for historical data
        """
        self._metrics = metrics_collector
        self._alerts = alert_manager
        self._data_store = data_store

        # Cache for dashboard data
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl_s = 10.0  # 10 second cache

        logger.info("DashboardProvider initialized")

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

    async def get_real_time_dashboard(
        self,
        site_id: str,
    ) -> RealTimeDashboard:
        """
        Get real-time dashboard data for a site.

        Args:
            site_id: Site identifier

        Returns:
            RealTimeDashboard with current system status
        """
        cache_key = f"real_time:{site_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = RealTimeDashboard(site_id=site_id)

        # Populate from metrics collector
        if self._metrics:
            metrics = self._metrics.get_metrics()

            # Extract steam metrics
            dashboard.steam_flow_kg_h = self._extract_metric_value(
                metrics, "steam_flow_kg_h"
            )
            dashboard.steam_pressure_bar = self._extract_metric_value(
                metrics, "header_pressure_bar"
            )
            dashboard.steam_temperature_c = self._extract_metric_value(
                metrics, "steam_temperature_c"
            )
            dashboard.steam_quality_percent = self._extract_metric_value(
                metrics, "steam_quality_percent", default=100.0
            )

            # Extract efficiency metrics
            dashboard.boiler_efficiency_percent = self._extract_metric_value(
                metrics, "boiler_efficiency_percent"
            )
            dashboard.condensate_recovery_percent = self._extract_metric_value(
                metrics, "condensate_recovery_percent"
            )
            dashboard.overall_efficiency_percent = self._extract_metric_value(
                metrics, "overall_efficiency_percent"
            )

            # Extract trap metrics
            dashboard.total_traps = int(self._extract_metric_value(
                metrics, "traps_total"
            ))
            dashboard.healthy_traps = int(self._extract_metric_value(
                metrics, "traps_operational"
            ))
            dashboard.failed_traps = int(self._extract_metric_value(
                metrics, "traps_failed"
            ))
            dashboard.estimated_loss_kg_h = self._extract_metric_value(
                metrics, "steam_loss_kg_h"
            )

        # Populate from alert manager
        if self._alerts:
            active_alerts = self._alerts.get_active_alerts()
            dashboard.active_alerts_count = len(active_alerts)
            dashboard.critical_alerts_count = len([
                a for a in active_alerts
                if a.severity.value in ["critical", "emergency"]
            ])

        # Determine system status
        if dashboard.critical_alerts_count > 0:
            dashboard.system_status = "critical"
        elif dashboard.active_alerts_count > 3:
            dashboard.system_status = "degraded"
        else:
            dashboard.system_status = "operational"

        # Create widgets
        dashboard.widgets = self._create_real_time_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

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

    def _create_real_time_widgets(
        self,
        dashboard: RealTimeDashboard,
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

        # Steam flow gauge
        widgets.append(DashboardWidget(
            widget_id="steam_flow",
            widget_type="gauge",
            title="Steam Flow",
            data={
                "value": dashboard.steam_flow_kg_h,
                "unit": "kg/h",
                "min": 0,
                "max": 10000,
            },
            position={"row": 0, "col": 2, "width": 2, "height": 1},
        ))

        # Efficiency stat
        widgets.append(DashboardWidget(
            widget_id="efficiency",
            widget_type="stat",
            title="Overall Efficiency",
            data={
                "value": dashboard.overall_efficiency_percent,
                "unit": "%",
                "trend": "stable",
            },
            position={"row": 0, "col": 4, "width": 2, "height": 1},
        ))

        # Trap health
        widgets.append(DashboardWidget(
            widget_id="trap_health",
            widget_type="stat",
            title="Trap Health",
            data={
                "healthy": dashboard.healthy_traps,
                "failed": dashboard.failed_traps,
                "total": dashboard.total_traps,
                "health_percent": (
                    dashboard.healthy_traps / dashboard.total_traps * 100
                    if dashboard.total_traps > 0 else 100.0
                ),
            },
            position={"row": 1, "col": 0, "width": 2, "height": 1},
        ))

        # Active alerts
        widgets.append(DashboardWidget(
            widget_id="alerts",
            widget_type="alert",
            title="Active Alerts",
            data={
                "total": dashboard.active_alerts_count,
                "critical": dashboard.critical_alerts_count,
            },
            position={"row": 1, "col": 2, "width": 2, "height": 1},
        ))

        return widgets

    async def get_kpi_dashboard(
        self,
        time_window: Optional[timedelta] = None,
    ) -> KPIDashboard:
        """
        Get KPI dashboard data.

        Args:
            time_window: Time window for KPIs (default: 24 hours)

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

            dashboard.avg_optimization_time_ms = summary.avg_computation_time_ms

            # Calculate efficiency trend (placeholder)
            dashboard.efficiency_trend = "stable"

        # Create widgets
        dashboard.widgets = self._create_kpi_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_kpi_widgets(self, dashboard: KPIDashboard) -> List[DashboardWidget]:
        """Create widgets for KPI dashboard."""
        widgets = []

        # Efficiency trend chart
        widgets.append(DashboardWidget(
            widget_id="efficiency_trend",
            widget_type="chart",
            title="Efficiency Trend",
            data={
                "type": "line",
                "series": dashboard.efficiency_history.to_dict(),
            },
            position={"row": 0, "col": 0, "width": 6, "height": 2},
        ))

        # Reliability stats
        widgets.append(DashboardWidget(
            widget_id="reliability",
            widget_type="stat",
            title="System Availability",
            data={
                "value": dashboard.system_availability_percent,
                "unit": "%",
                "target": 99.5,
            },
            position={"row": 0, "col": 6, "width": 2, "height": 1},
        ))

        # Cost savings
        widgets.append(DashboardWidget(
            widget_id="cost_savings",
            widget_type="stat",
            title="Savings Realized",
            data={
                "value": dashboard.savings_realized_usd,
                "unit": "USD",
                "period": "today",
            },
            position={"row": 1, "col": 6, "width": 2, "height": 1},
        ))

        return widgets

    async def get_optimization_dashboard(self) -> OptimizationDashboard:
        """
        Get optimization dashboard data.

        Returns:
            OptimizationDashboard with optimization metrics
        """
        cache_key = "optimization"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = OptimizationDashboard()

        # Populate from metrics collector
        if self._metrics:
            summary = self._metrics.get_metrics_summary(timedelta(hours=24))

            dashboard.optimizations_today = summary.optimizations_run
            dashboard.optimizations_successful = summary.optimizations_successful
            dashboard.success_rate_percent = (
                summary.optimizations_successful / summary.optimizations_run * 100
                if summary.optimizations_run > 0 else 0.0
            )

            dashboard.cost_savings_usd_today = summary.total_savings_usd
            dashboard.recommendation_acceptance_rate_percent = summary.acceptance_rate_percent
            dashboard.avg_optimization_time_ms = summary.avg_computation_time_ms
            dashboard.p95_optimization_time_ms = summary.p95_computation_time_ms

            # Project annual savings
            daily_savings = summary.total_savings_usd
            dashboard.projected_annual_savings_usd = daily_savings * 365

        # Create widgets
        dashboard.widgets = self._create_optimization_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_optimization_widgets(
        self,
        dashboard: OptimizationDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for optimization dashboard."""
        widgets = []

        # Success rate gauge
        widgets.append(DashboardWidget(
            widget_id="success_rate",
            widget_type="gauge",
            title="Optimization Success Rate",
            data={
                "value": dashboard.success_rate_percent,
                "unit": "%",
                "thresholds": {"green": 95, "yellow": 80, "red": 0},
            },
            position={"row": 0, "col": 0, "width": 2, "height": 1},
        ))

        # Daily savings stat
        widgets.append(DashboardWidget(
            widget_id="daily_savings",
            widget_type="stat",
            title="Daily Savings",
            data={
                "value": dashboard.cost_savings_usd_today,
                "unit": "USD",
                "trend": "up",
            },
            position={"row": 0, "col": 2, "width": 2, "height": 1},
        ))

        # Pending recommendations table
        widgets.append(DashboardWidget(
            widget_id="pending_recommendations",
            widget_type="table",
            title="Pending Recommendations",
            data={
                "rows": dashboard.pending_recommendations[:5],
                "columns": ["type", "savings_usd", "priority", "created_at"],
            },
            position={"row": 1, "col": 0, "width": 4, "height": 2},
        ))

        return widgets

    async def get_trap_health_dashboard(self) -> TrapHealthDashboard:
        """
        Get trap health dashboard data.

        Returns:
            TrapHealthDashboard with trap health metrics
        """
        cache_key = "trap_health"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = TrapHealthDashboard()

        # Populate from metrics collector
        if self._metrics:
            metrics = self._metrics.get_metrics()

            dashboard.total_traps = int(self._extract_metric_value(
                metrics, "traps_total"
            ))
            dashboard.operational_traps = int(self._extract_metric_value(
                metrics, "traps_operational"
            ))
            dashboard.failed_traps = int(self._extract_metric_value(
                metrics, "traps_failed"
            ))
            dashboard.blow_through_count = int(self._extract_metric_value(
                metrics, "traps_blow_through"
            ))
            dashboard.blocked_count = int(self._extract_metric_value(
                metrics, "traps_blocked"
            ))

            dashboard.detection_accuracy_percent = self._extract_metric_value(
                metrics, "trap_detection_accuracy"
            ) * 100
            dashboard.detection_precision_percent = self._extract_metric_value(
                metrics, "trap_detection_precision"
            ) * 100
            dashboard.detection_recall_percent = self._extract_metric_value(
                metrics, "trap_detection_recall"
            ) * 100

            dashboard.total_steam_loss_kg_h = self._extract_metric_value(
                metrics, "steam_loss_kg_h"
            )

        # Calculate annual loss (placeholder)
        # Assuming $10/ton steam, 8760 hours/year
        dashboard.estimated_annual_loss_usd = (
            dashboard.total_steam_loss_kg_h * 8760 / 1000 * 10
        )

        # Create widgets
        dashboard.widgets = self._create_trap_health_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_trap_health_widgets(
        self,
        dashboard: TrapHealthDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for trap health dashboard."""
        widgets = []

        # Trap status pie chart
        widgets.append(DashboardWidget(
            widget_id="trap_status",
            widget_type="chart",
            title="Trap Status Distribution",
            data={
                "type": "pie",
                "series": [
                    {"name": "Operational", "value": dashboard.operational_traps},
                    {"name": "Failed", "value": dashboard.failed_traps},
                    {"name": "Suspect", "value": dashboard.suspect_traps},
                    {"name": "Unknown", "value": dashboard.unknown_status_traps},
                ],
            },
            position={"row": 0, "col": 0, "width": 3, "height": 2},
        ))

        # Detection accuracy gauge
        widgets.append(DashboardWidget(
            widget_id="detection_accuracy",
            widget_type="gauge",
            title="Detection Accuracy",
            data={
                "value": dashboard.detection_accuracy_percent,
                "unit": "%",
                "target": 95,
            },
            position={"row": 0, "col": 3, "width": 2, "height": 1},
        ))

        # Steam loss stat
        widgets.append(DashboardWidget(
            widget_id="steam_loss",
            widget_type="stat",
            title="Steam Loss",
            data={
                "value": dashboard.total_steam_loss_kg_h,
                "unit": "kg/h",
                "annual_cost_usd": dashboard.estimated_annual_loss_usd,
            },
            position={"row": 0, "col": 5, "width": 2, "height": 1},
        ))

        # Failure breakdown
        widgets.append(DashboardWidget(
            widget_id="failure_breakdown",
            widget_type="chart",
            title="Failure Types",
            data={
                "type": "bar",
                "series": [
                    {"name": "Blow-through", "value": dashboard.blow_through_count},
                    {"name": "Blocked", "value": dashboard.blocked_count},
                    {"name": "Leaking", "value": dashboard.leaking_count},
                ],
            },
            position={"row": 1, "col": 3, "width": 4, "height": 1},
        ))

        return widgets

    async def get_climate_impact_dashboard(self) -> ClimateImpactDashboard:
        """
        Get climate impact dashboard data.

        Returns:
            ClimateImpactDashboard with environmental metrics
        """
        cache_key = "climate_impact"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = ClimateImpactDashboard()

        # Populate from metrics collector
        if self._metrics:
            metrics = self._metrics.get_metrics()
            summary = self._metrics.get_metrics_summary(timedelta(days=30))

            # Calculate CO2 from energy savings
            # Assuming 0.5 kg CO2 per kWh (natural gas)
            energy_savings = summary.total_savings_usd / 0.10  # Rough conversion
            dashboard.co2_emissions_avoided_kg = energy_savings * 0.5

        # Create widgets
        dashboard.widgets = self._create_climate_impact_widgets(dashboard)

        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_climate_impact_widgets(
        self,
        dashboard: ClimateImpactDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for climate impact dashboard."""
        widgets = []

        # CO2 reduction stat
        widgets.append(DashboardWidget(
            widget_id="co2_reduction",
            widget_type="stat",
            title="CO2 Avoided",
            data={
                "value": dashboard.co2_emissions_avoided_kg,
                "unit": "kg",
                "period": "30 days",
            },
            position={"row": 0, "col": 0, "width": 2, "height": 1},
        ))

        # Energy savings stat
        widgets.append(DashboardWidget(
            widget_id="energy_savings",
            widget_type="stat",
            title="Energy Saved",
            data={
                "value": dashboard.energy_saved_kwh,
                "unit": "kWh",
                "period": "30 days",
            },
            position={"row": 0, "col": 2, "width": 2, "height": 1},
        ))

        # CO2 emissions trend
        widgets.append(DashboardWidget(
            widget_id="co2_trend",
            widget_type="chart",
            title="CO2 Emissions Trend",
            data={
                "type": "area",
                "series": dashboard.co2_emissions_history.to_dict(),
            },
            position={"row": 1, "col": 0, "width": 6, "height": 2},
        ))

        # Target progress
        widgets.append(DashboardWidget(
            widget_id="target_progress",
            widget_type="gauge",
            title="Annual Target Progress",
            data={
                "value": dashboard.annual_co2_progress_percent,
                "unit": "%",
                "target": 100,
            },
            position={"row": 0, "col": 4, "width": 2, "height": 1},
        ))

        # Compliance status
        widgets.append(DashboardWidget(
            widget_id="compliance",
            widget_type="stat",
            title="ISO 50001 Compliance",
            data={
                "value": "Compliant" if dashboard.iso50001_compliant else "Non-compliant",
                "color": "green" if dashboard.iso50001_compliant else "red",
            },
            position={"row": 0, "col": 6, "width": 2, "height": 1},
        ))

        return widgets
