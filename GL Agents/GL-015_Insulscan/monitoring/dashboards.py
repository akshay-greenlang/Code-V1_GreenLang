"""
GL-015 INSULSCAN - Grafana Dashboard Definitions

This module provides Grafana dashboard definitions and data providers
for insulation scanning and thermal assessment visualization, including
heat loss overviews, asset condition heatmaps, trend analysis, and
ROI tracking dashboards.

Dashboard Types:
    - HeatLossOverviewDashboard: System-wide heat loss visualization
    - AssetConditionHeatmap: Visual heatmap of asset conditions
    - TrendAnalysisDashboard: Historical trend analysis
    - ROITrackingDashboard: Return on investment tracking
    - AlertOverviewDashboard: Alert status and statistics

Output Formats:
    - Grafana JSON dashboard model
    - Dashboard widget data for custom UIs
    - Time-series data for charts

Example:
    >>> provider = InsulscanDashboardProvider(metrics_collector, alert_manager)
    >>> heat_loss_dashboard = await provider.get_heat_loss_overview()
    >>> grafana_json = provider.export_grafana_dashboard("heat_loss_overview")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
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
    """Generic dashboard widget definition."""
    widget_id: str
    widget_type: str  # gauge, chart, table, heatmap, stat, alert
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
class AssetStatus:
    """Status summary for a single asset."""
    asset_id: str
    surface_type: str = "unknown"
    insulation_type: str = "unknown"
    status: str = "healthy"  # healthy, degraded, critical, offline
    heat_loss_watts: float = 0.0
    condition_score: float = 1.0
    hot_spots_count: int = 0
    projected_savings_usd: float = 0.0
    active_alerts: int = 0
    last_scan: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "asset_id": self.asset_id,
            "surface_type": self.surface_type,
            "insulation_type": self.insulation_type,
            "status": self.status,
            "heat_loss_watts": self.heat_loss_watts,
            "condition_score": self.condition_score,
            "hot_spots_count": self.hot_spots_count,
            "projected_savings_usd": self.projected_savings_usd,
            "active_alerts": self.active_alerts,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
        }


# =============================================================================
# Dashboard Data Classes
# =============================================================================

@dataclass
class HeatLossOverviewDashboard:
    """Dashboard data for system-wide heat loss overview."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # System summary
    total_assets: int = 0
    assets_scanned_today: int = 0
    total_heat_loss_watts: float = 0.0
    total_heat_loss_kw: float = 0.0
    avg_condition_score: float = 1.0

    # Heat loss breakdown by surface type
    heat_loss_by_surface: Dict[str, float] = field(default_factory=dict)

    # Heat loss breakdown by insulation type
    heat_loss_by_insulation: Dict[str, float] = field(default_factory=dict)

    # Top heat loss assets
    top_heat_loss_assets: List[AssetStatus] = field(default_factory=list)

    # Alert summary
    active_alerts: int = 0
    critical_alerts: int = 0

    # Savings potential
    total_projected_savings_usd: float = 0.0
    repairs_recommended: int = 0

    # Time series
    heat_loss_trend: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="heat_loss", unit="kW")
    )

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_assets": self.total_assets,
                "assets_scanned_today": self.assets_scanned_today,
                "total_heat_loss_watts": self.total_heat_loss_watts,
                "total_heat_loss_kw": self.total_heat_loss_kw,
                "avg_condition_score": self.avg_condition_score,
            },
            "breakdown": {
                "by_surface_type": self.heat_loss_by_surface,
                "by_insulation_type": self.heat_loss_by_insulation,
            },
            "top_heat_loss_assets": [a.to_dict() for a in self.top_heat_loss_assets],
            "alerts": {
                "active": self.active_alerts,
                "critical": self.critical_alerts,
            },
            "savings": {
                "total_projected_usd": self.total_projected_savings_usd,
                "repairs_recommended": self.repairs_recommended,
            },
            "trend": self.heat_loss_trend.to_dict(),
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class AssetConditionHeatmapDashboard:
    """Dashboard data for asset condition heatmap visualization."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Grid dimensions
    grid_rows: int = 0
    grid_cols: int = 0

    # Asset statuses
    asset_statuses: List[AssetStatus] = field(default_factory=list)

    # Heatmap data: list of [row, col, value] for condition scores
    heatmap_data: List[List[float]] = field(default_factory=list)

    # Labels for axes
    row_labels: List[str] = field(default_factory=list)
    col_labels: List[str] = field(default_factory=list)

    # Statistics
    healthy_count: int = 0
    degraded_count: int = 0
    critical_count: int = 0
    offline_count: int = 0

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "grid": {
                "rows": self.grid_rows,
                "cols": self.grid_cols,
                "row_labels": self.row_labels,
                "col_labels": self.col_labels,
            },
            "heatmap_data": self.heatmap_data,
            "asset_statuses": [a.to_dict() for a in self.asset_statuses],
            "statistics": {
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "critical": self.critical_count,
                "offline": self.offline_count,
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class TrendAnalysisDashboard:
    """Dashboard data for historical trend analysis."""
    time_window_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=7)
    )
    time_window_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Trend indicators
    heat_loss_trend: str = "stable"  # improving, stable, declining
    condition_trend: str = "stable"
    hot_spots_trend: str = "stable"

    # Statistics over window
    avg_heat_loss_watts: float = 0.0
    max_heat_loss_watts: float = 0.0
    avg_condition_score: float = 0.0
    min_condition_score: float = 1.0
    total_hot_spots_detected: int = 0
    total_repairs_recommended: int = 0

    # Time series data
    heat_loss_series: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="heat_loss_trend", unit="kW")
    )
    condition_series: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="condition_trend", unit="score")
    )
    hot_spots_series: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="hot_spots_trend", unit="count")
    )
    analyses_series: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="analyses_trend", unit="count")
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
                "heat_loss": self.heat_loss_trend,
                "condition": self.condition_trend,
                "hot_spots": self.hot_spots_trend,
            },
            "statistics": {
                "avg_heat_loss_watts": self.avg_heat_loss_watts,
                "max_heat_loss_watts": self.max_heat_loss_watts,
                "avg_condition_score": self.avg_condition_score,
                "min_condition_score": self.min_condition_score,
                "total_hot_spots": self.total_hot_spots_detected,
                "total_repairs": self.total_repairs_recommended,
            },
            "series": {
                "heat_loss": self.heat_loss_series.to_dict(),
                "condition": self.condition_series.to_dict(),
                "hot_spots": self.hot_spots_series.to_dict(),
                "analyses": self.analyses_series.to_dict(),
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class ROITrackingDashboard:
    """Dashboard data for return on investment tracking."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_period: str = "monthly"  # daily, weekly, monthly, yearly

    # Current period values
    period_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=30)
    )
    period_end: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Heat loss costs
    estimated_heat_loss_cost_usd: float = 0.0
    heat_loss_reduction_usd: float = 0.0

    # Repair economics
    repairs_completed: int = 0
    repair_cost_total_usd: float = 0.0
    projected_annual_savings_usd: float = 0.0
    actual_savings_realized_usd: float = 0.0

    # ROI metrics
    simple_payback_months: float = 0.0
    roi_percent: float = 0.0
    net_present_value_usd: float = 0.0

    # Comparison to baseline
    baseline_heat_loss_kw: float = 0.0
    current_heat_loss_kw: float = 0.0
    heat_loss_reduction_percent: float = 0.0

    # Time series
    savings_trend: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="savings_trend", unit="USD")
    )
    roi_trend: TimeSeriesData = field(
        default_factory=lambda: TimeSeriesData(name="roi_trend", unit="%")
    )

    # Widgets
    widgets: List[DashboardWidget] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "time_period": self.time_period,
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "heat_loss_costs": {
                "estimated_cost_usd": self.estimated_heat_loss_cost_usd,
                "reduction_usd": self.heat_loss_reduction_usd,
            },
            "repair_economics": {
                "repairs_completed": self.repairs_completed,
                "total_cost_usd": self.repair_cost_total_usd,
                "projected_savings_usd": self.projected_annual_savings_usd,
                "actual_savings_usd": self.actual_savings_realized_usd,
            },
            "roi_metrics": {
                "simple_payback_months": self.simple_payback_months,
                "roi_percent": self.roi_percent,
                "npv_usd": self.net_present_value_usd,
            },
            "baseline_comparison": {
                "baseline_heat_loss_kw": self.baseline_heat_loss_kw,
                "current_heat_loss_kw": self.current_heat_loss_kw,
                "reduction_percent": self.heat_loss_reduction_percent,
            },
            "trends": {
                "savings": self.savings_trend.to_dict(),
                "roi": self.roi_trend.to_dict(),
            },
            "widgets": [w.to_dict() for w in self.widgets],
        }


@dataclass
class AlertOverviewDashboard:
    """Dashboard data for alert overview and statistics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Current alert counts
    total_active: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Alert counts by type
    hot_spot_alerts: int = 0
    degradation_alerts: int = 0
    heat_loss_alerts: int = 0
    system_alerts: int = 0

    # Historical statistics (last 24 hours)
    alerts_created_24h: int = 0
    alerts_resolved_24h: int = 0
    alerts_acknowledged_24h: int = 0
    avg_resolution_time_minutes: float = 0.0

    # Top alerting assets
    top_alerting_assets: List[Dict[str, Any]] = field(default_factory=list)

    # Active alerts detail
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)

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
                "info": self.info_count,
            },
            "by_type": {
                "hot_spot": self.hot_spot_alerts,
                "degradation": self.degradation_alerts,
                "heat_loss": self.heat_loss_alerts,
                "system": self.system_alerts,
            },
            "statistics_24h": {
                "created": self.alerts_created_24h,
                "resolved": self.alerts_resolved_24h,
                "acknowledged": self.alerts_acknowledged_24h,
                "avg_resolution_minutes": self.avg_resolution_time_minutes,
            },
            "top_alerting_assets": self.top_alerting_assets,
            "active_alerts": self.active_alerts,
            "widgets": [w.to_dict() for w in self.widgets],
        }


# =============================================================================
# Grafana Dashboard Templates
# =============================================================================

class GrafanaDashboardBuilder:
    """Builder for Grafana dashboard JSON models."""

    AGENT_ID = "GL-015"
    AGENT_NAME = "INSULSCAN"

    def __init__(
        self,
        datasource: str = "prometheus",
        refresh_interval: str = "10s",
    ) -> None:
        """
        Initialize Grafana dashboard builder.

        Args:
            datasource: Prometheus datasource name
            refresh_interval: Dashboard refresh interval
        """
        self.datasource = datasource
        self.refresh_interval = refresh_interval

    def build_heat_loss_overview(self) -> Dict[str, Any]:
        """Build Grafana dashboard for heat loss overview."""
        return {
            "title": "INSULSCAN - Heat Loss Overview",
            "uid": "insulscan-heat-loss",
            "tags": ["insulscan", "heat-loss", "thermal"],
            "timezone": "browser",
            "refresh": self.refresh_interval,
            "schemaVersion": 38,
            "panels": [
                # Total heat loss stat
                {
                    "id": 1,
                    "type": "stat",
                    "title": "Total Heat Loss",
                    "gridPos": {"x": 0, "y": 0, "w": 4, "h": 4},
                    "targets": [
                        {
                            "expr": "sum(insulscan_heat_loss_watts)",
                            "legendFormat": "Total",
                        }
                    ],
                    "options": {
                        "reduceOptions": {"calcs": ["lastNotNull"]},
                        "colorMode": "value",
                    },
                    "fieldConfig": {
                        "defaults": {
                            "unit": "watt",
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 10000},
                                    {"color": "red", "value": 50000},
                                ],
                            },
                        },
                    },
                },
                # Average condition score
                {
                    "id": 2,
                    "type": "gauge",
                    "title": "Avg Condition Score",
                    "gridPos": {"x": 4, "y": 0, "w": 4, "h": 4},
                    "targets": [
                        {
                            "expr": "avg(insulscan_condition_score)",
                            "legendFormat": "Avg Score",
                        }
                    ],
                    "options": {
                        "minVizWidth": 75,
                        "minVizHeight": 75,
                        "showThresholdLabels": False,
                    },
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "yellow", "value": 0.5},
                                    {"color": "green", "value": 0.8},
                                ],
                            },
                        },
                    },
                },
                # Active alerts
                {
                    "id": 3,
                    "type": "stat",
                    "title": "Active Alerts",
                    "gridPos": {"x": 8, "y": 0, "w": 4, "h": 4},
                    "targets": [
                        {
                            "expr": "count(ALERTS{alertname=~\"insulscan_.*\", alertstate=\"firing\"})",
                            "legendFormat": "Active",
                        }
                    ],
                    "options": {
                        "colorMode": "value",
                    },
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 1},
                                    {"color": "red", "value": 5},
                                ],
                            },
                        },
                    },
                },
                # Analyses performed
                {
                    "id": 4,
                    "type": "stat",
                    "title": "Analyses Today",
                    "gridPos": {"x": 12, "y": 0, "w": 4, "h": 4},
                    "targets": [
                        {
                            "expr": "increase(insulscan_analyses_total[24h])",
                            "legendFormat": "Analyses",
                        }
                    ],
                },
                # Heat loss time series
                {
                    "id": 5,
                    "type": "timeseries",
                    "title": "Heat Loss Trend",
                    "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(insulscan_heat_loss_watts) by (surface_type)",
                            "legendFormat": "{{surface_type}}",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "watt",
                            "custom": {
                                "drawStyle": "line",
                                "lineInterpolation": "smooth",
                                "fillOpacity": 10,
                            },
                        },
                    },
                },
                # Heat loss by surface type pie
                {
                    "id": 6,
                    "type": "piechart",
                    "title": "Heat Loss by Surface Type",
                    "gridPos": {"x": 12, "y": 4, "w": 6, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(insulscan_heat_loss_watts) by (surface_type)",
                            "legendFormat": "{{surface_type}}",
                        }
                    ],
                },
                # Hot spots detected
                {
                    "id": 7,
                    "type": "bargauge",
                    "title": "Hot Spots by Severity",
                    "gridPos": {"x": 18, "y": 4, "w": 6, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(insulscan_hot_spots_detected) by (severity)",
                            "legendFormat": "{{severity}}",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 5},
                                    {"color": "red", "value": 10},
                                ],
                            },
                        },
                    },
                },
                # Top heat loss assets table
                {
                    "id": 8,
                    "type": "table",
                    "title": "Top Heat Loss Assets",
                    "gridPos": {"x": 0, "y": 12, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "topk(10, insulscan_heat_loss_watts)",
                            "format": "table",
                            "instant": True,
                        }
                    ],
                    "transformations": [
                        {
                            "id": "organize",
                            "options": {
                                "excludeByName": {"Time": True, "__name__": True},
                                "indexByName": {},
                                "renameByName": {
                                    "asset_id": "Asset",
                                    "surface_type": "Surface",
                                    "Value": "Heat Loss (W)",
                                },
                            },
                        }
                    ],
                },
                # Condition score distribution
                {
                    "id": 9,
                    "type": "histogram",
                    "title": "Condition Score Distribution",
                    "gridPos": {"x": 12, "y": 12, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "insulscan_condition_score",
                            "legendFormat": "{{asset_id}}",
                        }
                    ],
                    "options": {
                        "bucketSize": 0.1,
                    },
                },
            ],
            "templating": {
                "list": [
                    {
                        "name": "surface_type",
                        "type": "query",
                        "datasource": self.datasource,
                        "query": 'label_values(insulscan_heat_loss_watts, surface_type)',
                        "includeAll": True,
                        "multi": True,
                    },
                    {
                        "name": "asset_id",
                        "type": "query",
                        "datasource": self.datasource,
                        "query": 'label_values(insulscan_heat_loss_watts, asset_id)',
                        "includeAll": True,
                        "multi": True,
                    },
                ],
            },
        }

    def build_asset_condition_heatmap(self) -> Dict[str, Any]:
        """Build Grafana dashboard for asset condition heatmap."""
        return {
            "title": "INSULSCAN - Asset Condition Heatmap",
            "uid": "insulscan-condition-heatmap",
            "tags": ["insulscan", "condition", "heatmap"],
            "timezone": "browser",
            "refresh": self.refresh_interval,
            "schemaVersion": 38,
            "panels": [
                # Condition heatmap
                {
                    "id": 1,
                    "type": "heatmap",
                    "title": "Asset Condition Heatmap",
                    "gridPos": {"x": 0, "y": 0, "w": 24, "h": 12},
                    "targets": [
                        {
                            "expr": "insulscan_condition_score",
                            "legendFormat": "{{asset_id}}",
                        }
                    ],
                    "options": {
                        "cellGap": 1,
                        "color": {
                            "mode": "scheme",
                            "scheme": "RdYlGn",
                            "steps": 128,
                        },
                        "yAxis": {
                            "axisPlacement": "left",
                        },
                    },
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                        },
                    },
                },
                # Status summary
                {
                    "id": 2,
                    "type": "piechart",
                    "title": "Asset Status Distribution",
                    "gridPos": {"x": 0, "y": 12, "w": 8, "h": 8},
                    "targets": [
                        {
                            "expr": "count(insulscan_condition_score >= 0.8)",
                            "legendFormat": "Healthy",
                        },
                        {
                            "expr": "count(insulscan_condition_score >= 0.5 and insulscan_condition_score < 0.8)",
                            "legendFormat": "Degraded",
                        },
                        {
                            "expr": "count(insulscan_condition_score < 0.5)",
                            "legendFormat": "Critical",
                        },
                    ],
                    "options": {
                        "pieType": "pie",
                        "displayLabels": ["value", "percent"],
                    },
                },
                # Critical assets list
                {
                    "id": 3,
                    "type": "table",
                    "title": "Critical Assets (Score < 0.5)",
                    "gridPos": {"x": 8, "y": 12, "w": 16, "h": 8},
                    "targets": [
                        {
                            "expr": "insulscan_condition_score < 0.5",
                            "format": "table",
                            "instant": True,
                        }
                    ],
                },
            ],
        }

    def build_trend_analysis(self) -> Dict[str, Any]:
        """Build Grafana dashboard for trend analysis."""
        return {
            "title": "INSULSCAN - Trend Analysis",
            "uid": "insulscan-trends",
            "tags": ["insulscan", "trends", "analysis"],
            "timezone": "browser",
            "refresh": self.refresh_interval,
            "schemaVersion": 38,
            "panels": [
                # Heat loss trend
                {
                    "id": 1,
                    "type": "timeseries",
                    "title": "Heat Loss Trend (7 Days)",
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(insulscan_heat_loss_watts)",
                            "legendFormat": "Total Heat Loss",
                        },
                        {
                            "expr": "avg(insulscan_heat_loss_watts)",
                            "legendFormat": "Avg per Asset",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "watt"},
                    },
                },
                # Condition trend
                {
                    "id": 2,
                    "type": "timeseries",
                    "title": "Condition Score Trend (7 Days)",
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "avg(insulscan_condition_score)",
                            "legendFormat": "Avg Condition",
                        },
                        {
                            "expr": "min(insulscan_condition_score)",
                            "legendFormat": "Min Condition",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {"min": 0, "max": 1},
                    },
                },
                # Hot spots trend
                {
                    "id": 3,
                    "type": "timeseries",
                    "title": "Hot Spots Detected (7 Days)",
                    "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "increase(insulscan_hot_spots_detected[1d])",
                            "legendFormat": "{{severity}}",
                        }
                    ],
                },
                # Analyses performed trend
                {
                    "id": 4,
                    "type": "timeseries",
                    "title": "Analyses Performed (7 Days)",
                    "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "increase(insulscan_analyses_total[1d])",
                            "legendFormat": "{{status}}",
                        }
                    ],
                },
                # Analysis latency
                {
                    "id": 5,
                    "type": "timeseries",
                    "title": "Analysis Duration (P95)",
                    "gridPos": {"x": 0, "y": 16, "w": 24, "h": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(insulscan_analysis_duration_seconds_bucket[5m]))",
                            "legendFormat": "P95 Duration",
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(insulscan_analysis_duration_seconds_bucket[5m]))",
                            "legendFormat": "P50 Duration",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "s"},
                    },
                },
            ],
        }

    def build_roi_tracking(self) -> Dict[str, Any]:
        """Build Grafana dashboard for ROI tracking."""
        return {
            "title": "INSULSCAN - ROI Tracking",
            "uid": "insulscan-roi",
            "tags": ["insulscan", "roi", "savings"],
            "timezone": "browser",
            "refresh": "1m",
            "schemaVersion": 38,
            "panels": [
                # Total projected savings
                {
                    "id": 1,
                    "type": "stat",
                    "title": "Total Projected Savings",
                    "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "sum(insulscan_energy_savings_usd)",
                            "legendFormat": "Projected",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "currencyUSD",
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                ],
                            },
                        },
                    },
                },
                # Repairs recommended
                {
                    "id": 2,
                    "type": "stat",
                    "title": "Repairs Recommended",
                    "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "sum(insulscan_repair_recommendations_total)",
                            "legendFormat": "Total",
                        }
                    ],
                },
                # Critical repairs
                {
                    "id": 3,
                    "type": "stat",
                    "title": "Critical Priority Repairs",
                    "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": 'sum(insulscan_repair_recommendations_total{priority="critical"})',
                            "legendFormat": "Critical",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 1},
                                ],
                            },
                        },
                    },
                },
                # Heat loss reduction
                {
                    "id": 4,
                    "type": "gauge",
                    "title": "Heat Loss Reduction vs Baseline",
                    "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "100 * (1 - sum(insulscan_heat_loss_watts) / 100000)",
                            "legendFormat": "Reduction %",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                        },
                    },
                },
                # Savings trend
                {
                    "id": 5,
                    "type": "timeseries",
                    "title": "Projected Savings Trend",
                    "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(insulscan_energy_savings_usd)",
                            "legendFormat": "Total Savings",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "currencyUSD"},
                    },
                },
                # Repairs by priority
                {
                    "id": 6,
                    "type": "piechart",
                    "title": "Repairs by Priority",
                    "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "sum(insulscan_repair_recommendations_total) by (priority)",
                            "legendFormat": "{{priority}}",
                        }
                    ],
                },
                # Top savings opportunities
                {
                    "id": 7,
                    "type": "table",
                    "title": "Top Savings Opportunities",
                    "gridPos": {"x": 0, "y": 12, "w": 24, "h": 8},
                    "targets": [
                        {
                            "expr": "topk(10, insulscan_energy_savings_usd)",
                            "format": "table",
                            "instant": True,
                        }
                    ],
                    "transformations": [
                        {
                            "id": "organize",
                            "options": {
                                "renameByName": {
                                    "asset_id": "Asset",
                                    "surface_type": "Type",
                                    "Value": "Annual Savings (USD)",
                                },
                            },
                        }
                    ],
                },
            ],
        }


# =============================================================================
# Dashboard Provider
# =============================================================================

class InsulscanDashboardProvider:
    """
    Dashboard data provider for INSULSCAN monitoring UI.

    This class aggregates data from metrics collectors, alert managers,
    and other sources to provide comprehensive dashboard views.

    Example:
        >>> provider = InsulscanDashboardProvider(metrics_collector, alert_manager)
        >>> overview = await provider.get_heat_loss_overview()
        >>> grafana_json = provider.export_grafana_dashboard("heat_loss_overview")
    """

    def __init__(
        self,
        metrics_collector: Optional[Any] = None,
        alert_manager: Optional[Any] = None,
        data_store: Optional[Any] = None,
    ) -> None:
        """
        Initialize InsulscanDashboardProvider.

        Args:
            metrics_collector: InsulscanMetricsCollector instance
            alert_manager: InsulscanAlertManager instance
            data_store: Optional data store for historical data
        """
        self._metrics = metrics_collector
        self._alerts = alert_manager
        self._data_store = data_store
        self._grafana_builder = GrafanaDashboardBuilder()

        # Cache for dashboard data
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl_s = 5.0

        logger.info("InsulscanDashboardProvider initialized")

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

    # =========================================================================
    # Dashboard Data Methods
    # =========================================================================

    async def get_heat_loss_overview(self) -> HeatLossOverviewDashboard:
        """
        Get heat loss overview dashboard data.

        Returns:
            HeatLossOverviewDashboard with current heat loss data
        """
        cache_key = "heat_loss_overview"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = HeatLossOverviewDashboard()

        if self._metrics:
            summary = self._metrics.get_metrics_summary(timedelta(hours=24))
            dashboard.total_heat_loss_watts = summary.total_heat_loss_watts
            dashboard.total_heat_loss_kw = summary.total_heat_loss_watts / 1000
            dashboard.avg_condition_score = summary.avg_condition_score
            dashboard.assets_scanned_today = summary.assets_analyzed
            dashboard.total_projected_savings_usd = summary.total_projected_savings_usd
            dashboard.repairs_recommended = summary.total_repair_recommendations

        if self._alerts:
            active = self._alerts.get_active_alerts()
            dashboard.active_alerts = len(active)
            dashboard.critical_alerts = len([
                a for a in active if a.severity.value >= 2
            ])

        dashboard.widgets = self._create_heat_loss_widgets(dashboard)
        self._set_cached(cache_key, dashboard)
        return dashboard

    def _create_heat_loss_widgets(
        self,
        dashboard: HeatLossOverviewDashboard,
    ) -> List[DashboardWidget]:
        """Create widgets for heat loss overview dashboard."""
        widgets = []

        widgets.append(DashboardWidget(
            widget_id="total_heat_loss",
            widget_type="stat",
            title="Total Heat Loss",
            data={
                "value": dashboard.total_heat_loss_kw,
                "unit": "kW",
                "thresholds": {"green": 0, "yellow": 50, "red": 100},
            },
            position={"row": 0, "col": 0, "width": 3, "height": 1},
        ))

        widgets.append(DashboardWidget(
            widget_id="avg_condition",
            widget_type="gauge",
            title="Avg Condition Score",
            data={
                "value": dashboard.avg_condition_score,
                "min": 0,
                "max": 1,
                "thresholds": {"red": 0, "yellow": 0.5, "green": 0.8},
            },
            position={"row": 0, "col": 3, "width": 3, "height": 1},
        ))

        widgets.append(DashboardWidget(
            widget_id="projected_savings",
            widget_type="stat",
            title="Projected Savings",
            data={
                "value": dashboard.total_projected_savings_usd,
                "unit": "USD",
            },
            position={"row": 0, "col": 6, "width": 3, "height": 1},
        ))

        widgets.append(DashboardWidget(
            widget_id="active_alerts",
            widget_type="stat",
            title="Active Alerts",
            data={
                "value": dashboard.active_alerts,
                "critical": dashboard.critical_alerts,
            },
            position={"row": 0, "col": 9, "width": 3, "height": 1},
        ))

        return widgets

    async def get_asset_condition_heatmap(self) -> AssetConditionHeatmapDashboard:
        """Get asset condition heatmap dashboard data."""
        cache_key = "condition_heatmap"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = AssetConditionHeatmapDashboard()
        # In production, this would populate from actual data
        self._set_cached(cache_key, dashboard)
        return dashboard

    async def get_trend_analysis(
        self,
        time_window: Optional[timedelta] = None,
    ) -> TrendAnalysisDashboard:
        """Get trend analysis dashboard data."""
        if time_window is None:
            time_window = timedelta(days=7)

        cache_key = f"trends:{time_window.total_seconds()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        now = datetime.now(timezone.utc)
        dashboard = TrendAnalysisDashboard(
            time_window_start=now - time_window,
            time_window_end=now,
        )

        if self._metrics:
            summary = self._metrics.get_metrics_summary(time_window)
            dashboard.avg_heat_loss_watts = summary.total_heat_loss_watts / max(1, summary.assets_analyzed)
            dashboard.max_heat_loss_watts = summary.total_heat_loss_watts  # Simplified
            dashboard.avg_condition_score = summary.avg_condition_score
            dashboard.min_condition_score = summary.min_condition_score
            dashboard.total_hot_spots_detected = summary.total_hot_spots
            dashboard.total_repairs_recommended = summary.total_repair_recommendations

        self._set_cached(cache_key, dashboard)
        return dashboard

    async def get_roi_tracking(self) -> ROITrackingDashboard:
        """Get ROI tracking dashboard data."""
        cache_key = "roi_tracking"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = ROITrackingDashboard()

        if self._metrics:
            summary = self._metrics.get_metrics_summary(timedelta(days=30))
            dashboard.projected_annual_savings_usd = summary.total_projected_savings_usd
            dashboard.repairs_completed = summary.total_repair_recommendations

        self._set_cached(cache_key, dashboard)
        return dashboard

    async def get_alert_overview(self) -> AlertOverviewDashboard:
        """Get alert overview dashboard data."""
        cache_key = "alert_overview"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        dashboard = AlertOverviewDashboard()

        if self._alerts:
            active = self._alerts.get_active_alerts()
            stats = self._alerts.get_statistics()

            dashboard.total_active = len(active)
            dashboard.critical_count = len([a for a in active if a.severity.value == 2])
            dashboard.warning_count = len([a for a in active if a.severity.value == 1])
            dashboard.info_count = len([a for a in active if a.severity.value == 0])

            dashboard.alerts_created_24h = stats.get("alerts_created", 0)
            dashboard.alerts_resolved_24h = stats.get("alerts_resolved", 0)
            dashboard.alerts_acknowledged_24h = stats.get("alerts_acknowledged", 0)

            dashboard.active_alerts = [a.to_dict() for a in active[:20]]

        self._set_cached(cache_key, dashboard)
        return dashboard

    # =========================================================================
    # Grafana Export Methods
    # =========================================================================

    def export_grafana_dashboard(self, dashboard_type: str) -> Dict[str, Any]:
        """
        Export a Grafana dashboard JSON.

        Args:
            dashboard_type: Type of dashboard to export

        Returns:
            Grafana dashboard JSON model
        """
        dashboard_builders = {
            "heat_loss_overview": self._grafana_builder.build_heat_loss_overview,
            "asset_condition_heatmap": self._grafana_builder.build_asset_condition_heatmap,
            "trend_analysis": self._grafana_builder.build_trend_analysis,
            "roi_tracking": self._grafana_builder.build_roi_tracking,
        }

        builder = dashboard_builders.get(dashboard_type)
        if builder:
            return builder()
        else:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")

    def export_all_grafana_dashboards(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all Grafana dashboards.

        Returns:
            Dictionary of dashboard type to Grafana JSON model
        """
        return {
            "heat_loss_overview": self._grafana_builder.build_heat_loss_overview(),
            "asset_condition_heatmap": self._grafana_builder.build_asset_condition_heatmap(),
            "trend_analysis": self._grafana_builder.build_trend_analysis(),
            "roi_tracking": self._grafana_builder.build_roi_tracking(),
        }


# =============================================================================
# Global Instance
# =============================================================================

_dashboard_provider: Optional[InsulscanDashboardProvider] = None


def get_dashboard_provider() -> InsulscanDashboardProvider:
    """
    Get or create the global dashboard provider.

    Returns:
        Global InsulscanDashboardProvider instance
    """
    global _dashboard_provider
    if _dashboard_provider is None:
        _dashboard_provider = InsulscanDashboardProvider()
    return _dashboard_provider
