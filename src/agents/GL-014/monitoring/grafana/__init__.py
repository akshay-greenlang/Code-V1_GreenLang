# -*- coding: utf-8 -*-
"""
Grafana Dashboard Definitions for GL-014 EXCHANGER-PRO.

This module provides JSON dashboard configurations for Grafana:
- Heat Exchanger Overview - Fleet status summary
- Fouling Monitoring - Fouling trends and predictions
- Performance Trends - Thermal performance over time
- Cleaning Schedule Status - Maintenance planning
- Economic Impact - Cost analysis and savings
- Fleet Comparison - Cross-exchanger benchmarking

Example:
    >>> from monitoring.grafana import export_dashboard_json, get_all_dashboards
    >>> dashboards = get_all_dashboards()
    >>> for name, dashboard in dashboards.items():
    ...     export_dashboard_json(dashboard, f"{name}.json")

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PANEL TYPE ENUMERATION
# =============================================================================

class PanelType(Enum):
    """Grafana panel types."""
    STAT = "stat"
    GAUGE = "gauge"
    GRAPH = "graph"
    TIME_SERIES = "timeseries"
    TABLE = "table"
    HEATMAP = "heatmap"
    BAR_GAUGE = "bargauge"
    PIE_CHART = "piechart"
    TEXT = "text"
    ALERT_LIST = "alertlist"
    LOGS = "logs"


class DataSourceType(Enum):
    """Grafana data source types."""
    PROMETHEUS = "prometheus"
    POSTGRESQL = "postgres"
    INFLUXDB = "influxdb"
    ELASTICSEARCH = "elasticsearch"


# =============================================================================
# DASHBOARD BUILDER CLASS
# =============================================================================

@dataclass
class GrafanaPanel:
    """
    Configuration for a Grafana panel.

    Attributes:
        id: Unique panel ID
        title: Panel title
        panel_type: Type of panel
        grid_pos: Grid position (x, y, w, h)
        targets: List of query targets
        options: Panel-specific options
    """
    id: int
    title: str
    panel_type: PanelType
    grid_pos: Dict[str, int]
    targets: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Grafana JSON format."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.panel_type.value,
            "gridPos": self.grid_pos,
            "targets": self.targets,
            "options": self.options,
            "fieldConfig": self.field_config,
            "description": self.description,
        }


class GrafanaDashboardBuilder:
    """
    Builder class for creating Grafana dashboards.

    Provides a fluent interface for constructing dashboard JSON.

    Example:
        >>> builder = GrafanaDashboardBuilder("Heat Exchanger Overview")
        >>> builder.add_row("Summary")
        >>> builder.add_stat_panel("Total Exchangers", "gl014_exchangers_monitored")
        >>> dashboard = builder.build()
    """

    def __init__(
        self,
        title: str,
        uid: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        refresh: str = "30s"
    ):
        """
        Initialize the dashboard builder.

        Args:
            title: Dashboard title
            uid: Unique dashboard ID
            description: Dashboard description
            tags: Dashboard tags
            refresh: Auto-refresh interval
        """
        self.title = title
        self.uid = uid or title.lower().replace(" ", "-")
        self.description = description
        self.tags = tags or ["gl-014", "heat-exchangers"]
        self.refresh = refresh
        self.panels: List[GrafanaPanel] = []
        self._panel_id = 1
        self._current_y = 0
        self._datasource = "Prometheus"

    def set_datasource(self, datasource: str) -> "GrafanaDashboardBuilder":
        """Set the default data source."""
        self._datasource = datasource
        return self

    def add_row(
        self,
        title: str,
        collapsed: bool = False
    ) -> "GrafanaDashboardBuilder":
        """
        Add a row to the dashboard.

        Args:
            title: Row title
            collapsed: Whether row is collapsed

        Returns:
            Self for method chaining
        """
        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.TEXT,
            grid_pos={"x": 0, "y": self._current_y, "w": 24, "h": 1},
            options={"mode": "html", "content": f"<h3>{title}</h3>"},
        )
        panel.to_dict()["type"] = "row"
        panel.to_dict()["collapsed"] = collapsed

        self.panels.append(panel)
        self._panel_id += 1
        self._current_y += 1

        return self

    def add_stat_panel(
        self,
        title: str,
        query: str,
        unit: str = "",
        thresholds: Optional[List[Dict]] = None,
        grid_pos: Optional[Dict[str, int]] = None
    ) -> "GrafanaDashboardBuilder":
        """
        Add a stat panel.

        Args:
            title: Panel title
            query: Prometheus query
            unit: Value unit
            thresholds: Color thresholds
            grid_pos: Custom grid position

        Returns:
            Self for method chaining
        """
        if grid_pos is None:
            grid_pos = {"x": 0, "y": self._current_y, "w": 4, "h": 4}

        thresholds = thresholds or [
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ]

        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.STAT,
            grid_pos=grid_pos,
            targets=[{
                "expr": query,
                "refId": "A",
                "datasource": {"type": "prometheus", "uid": self._datasource},
            }],
            options={
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
            },
            field_config={
                "defaults": {
                    "unit": unit,
                    "thresholds": {"mode": "absolute", "steps": thresholds},
                },
                "overrides": [],
            },
        )

        self.panels.append(panel)
        self._panel_id += 1

        return self

    def add_gauge_panel(
        self,
        title: str,
        query: str,
        min_val: float = 0,
        max_val: float = 100,
        unit: str = "percent",
        thresholds: Optional[List[Dict]] = None,
        grid_pos: Optional[Dict[str, int]] = None
    ) -> "GrafanaDashboardBuilder":
        """
        Add a gauge panel.

        Args:
            title: Panel title
            query: Prometheus query
            min_val: Minimum value
            max_val: Maximum value
            unit: Value unit
            thresholds: Color thresholds
            grid_pos: Custom grid position

        Returns:
            Self for method chaining
        """
        if grid_pos is None:
            grid_pos = {"x": 0, "y": self._current_y, "w": 6, "h": 6}

        thresholds = thresholds or [
            {"color": "red", "value": None},
            {"color": "yellow", "value": 50},
            {"color": "green", "value": 80},
        ]

        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.GAUGE,
            grid_pos=grid_pos,
            targets=[{
                "expr": query,
                "refId": "A",
                "datasource": {"type": "prometheus", "uid": self._datasource},
            }],
            options={
                "reduceOptions": {"calcs": ["lastNotNull"]},
                "orientation": "auto",
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
            },
            field_config={
                "defaults": {
                    "unit": unit,
                    "min": min_val,
                    "max": max_val,
                    "thresholds": {"mode": "absolute", "steps": thresholds},
                },
                "overrides": [],
            },
        )

        self.panels.append(panel)
        self._panel_id += 1

        return self

    def add_time_series_panel(
        self,
        title: str,
        queries: List[Dict[str, str]],
        unit: str = "",
        legend_mode: str = "list",
        grid_pos: Optional[Dict[str, int]] = None
    ) -> "GrafanaDashboardBuilder":
        """
        Add a time series panel.

        Args:
            title: Panel title
            queries: List of {expr, legendFormat} dictionaries
            unit: Value unit
            legend_mode: Legend display mode
            grid_pos: Custom grid position

        Returns:
            Self for method chaining
        """
        if grid_pos is None:
            grid_pos = {"x": 0, "y": self._current_y, "w": 12, "h": 8}

        targets = []
        for i, q in enumerate(queries):
            targets.append({
                "expr": q.get("expr", ""),
                "legendFormat": q.get("legendFormat", ""),
                "refId": chr(65 + i),  # A, B, C, ...
                "datasource": {"type": "prometheus", "uid": self._datasource},
            })

        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.TIME_SERIES,
            grid_pos=grid_pos,
            targets=targets,
            options={
                "legend": {"displayMode": legend_mode, "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
            field_config={
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "showPoints": "never",
                    },
                },
                "overrides": [],
            },
        )

        self.panels.append(panel)
        self._panel_id += 1

        return self

    def add_table_panel(
        self,
        title: str,
        query: str,
        columns: Optional[List[Dict[str, str]]] = None,
        grid_pos: Optional[Dict[str, int]] = None
    ) -> "GrafanaDashboardBuilder":
        """
        Add a table panel.

        Args:
            title: Panel title
            query: Prometheus query
            columns: Column configurations
            grid_pos: Custom grid position

        Returns:
            Self for method chaining
        """
        if grid_pos is None:
            grid_pos = {"x": 0, "y": self._current_y, "w": 24, "h": 8}

        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.TABLE,
            grid_pos=grid_pos,
            targets=[{
                "expr": query,
                "format": "table",
                "instant": True,
                "refId": "A",
                "datasource": {"type": "prometheus", "uid": self._datasource},
            }],
            options={
                "showHeader": True,
                "sortBy": [{"desc": True, "displayName": "Value"}],
            },
            field_config={
                "defaults": {},
                "overrides": [],
            },
        )

        self.panels.append(panel)
        self._panel_id += 1

        return self

    def add_alert_list_panel(
        self,
        title: str,
        grid_pos: Optional[Dict[str, int]] = None
    ) -> "GrafanaDashboardBuilder":
        """
        Add an alert list panel.

        Args:
            title: Panel title
            grid_pos: Custom grid position

        Returns:
            Self for method chaining
        """
        if grid_pos is None:
            grid_pos = {"x": 0, "y": self._current_y, "w": 12, "h": 8}

        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.ALERT_LIST,
            grid_pos=grid_pos,
            targets=[],
            options={
                "showOptions": "current",
                "stateFilter": {"ok": True, "alerting": True, "pending": True},
                "sortOrder": 1,
                "dashboardAlerts": True,
                "alertName": "",
                "dashboardTitle": "",
                "maxItems": 10,
                "tags": self.tags,
            },
        )

        self.panels.append(panel)
        self._panel_id += 1

        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the final dashboard JSON.

        Returns:
            Complete dashboard JSON structure
        """
        return {
            "uid": self.uid,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "timezone": "browser",
            "refresh": self.refresh,
            "schemaVersion": 38,
            "version": 1,
            "editable": True,
            "graphTooltip": 1,
            "panels": [p.to_dict() for p in self.panels],
            "templating": {
                "list": [
                    {
                        "name": "exchanger_id",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self._datasource},
                        "query": 'label_values(gl014_health_index, exchanger_id)',
                        "refresh": 2,
                        "multi": True,
                        "includeAll": True,
                    },
                    {
                        "name": "location",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self._datasource},
                        "query": 'label_values(gl014_health_index, location)',
                        "refresh": 2,
                        "multi": True,
                        "includeAll": True,
                    },
                ]
            },
            "time": {"from": "now-6h", "to": "now"},
            "annotations": {"list": []},
            "links": [],
        }


# =============================================================================
# PRE-BUILT DASHBOARD DEFINITIONS
# =============================================================================

def _build_heat_exchanger_overview_dashboard() -> Dict[str, Any]:
    """Build Heat Exchanger Overview dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-014 Heat Exchanger Overview",
        uid="gl014-overview",
        description="Overview of all monitored heat exchangers",
        tags=["gl-014", "heat-exchangers", "overview"],
    )

    # Summary Row
    builder.add_stat_panel(
        "Total Exchangers",
        'sum(gl014_exchangers_monitored)',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Active Alerts",
        'count(ALERTS{alertstate="firing", job=~"gl-014.*"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Avg Health Index",
        'avg(gl014_health_index)',
        unit="percent",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "green", "value": 85},
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Cleanings Pending",
        'sum(gl014_cleaning_schedules_generated_total{urgency="urgent"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 3},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Est. Monthly Savings",
        'sum(gl014_estimated_savings_usd{savings_type="monthly"})',
        unit="currencyUSD",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Data Points/min",
        'sum(rate(gl014_data_points_processed_total[5m])) * 60',
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Health Index Distribution
    builder.add_time_series_panel(
        "Health Index Trend",
        [
            {"expr": 'avg(gl014_health_index{exchanger_id=~"$exchanger_id"})', "legendFormat": "Average"},
            {"expr": 'min(gl014_health_index{exchanger_id=~"$exchanger_id"})', "legendFormat": "Minimum"},
            {"expr": 'max(gl014_health_index{exchanger_id=~"$exchanger_id"})', "legendFormat": "Maximum"},
        ],
        unit="percent",
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Alert List
    builder.add_alert_list_panel(
        "Active Alerts",
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Exchanger Status Table
    builder.add_table_panel(
        "Exchanger Status",
        'gl014_health_index{exchanger_id=~"$exchanger_id"}',
        grid_pos={"x": 0, "y": 12, "w": 24, "h": 8},
    )

    return builder.build()


def _build_fouling_monitoring_dashboard() -> Dict[str, Any]:
    """Build Fouling Monitoring dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-014 Fouling Monitoring",
        uid="gl014-fouling",
        description="Detailed fouling trends and predictions",
        tags=["gl-014", "heat-exchangers", "fouling"],
    )

    # Current Fouling Status
    builder.add_gauge_panel(
        "Shell Side Fouling",
        'gl014_fouling_resistance_m2kw{exchanger_id=~"$exchanger_id", side="shell"}',
        min_val=0,
        max_val=0.002,
        unit="m2-K/W",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.0005},
            {"color": "red", "value": 0.001},
        ],
        grid_pos={"x": 0, "y": 0, "w": 6, "h": 6},
    )

    builder.add_gauge_panel(
        "Tube Side Fouling",
        'gl014_fouling_resistance_m2kw{exchanger_id=~"$exchanger_id", side="tube"}',
        min_val=0,
        max_val=0.002,
        unit="m2-K/W",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.0005},
            {"color": "red", "value": 0.001},
        ],
        grid_pos={"x": 6, "y": 0, "w": 6, "h": 6},
    )

    builder.add_stat_panel(
        "Days to Cleaning",
        'gl014_days_to_cleaning{exchanger_id=~"$exchanger_id"}',
        unit="d",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 14},
            {"color": "green", "value": 30},
        ],
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 6},
    )

    builder.add_stat_panel(
        "Fouling Rate",
        'gl014_fouling_rate_per_day{exchanger_id=~"$exchanger_id"}',
        unit="m2-K/W/d",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 6},
    )

    builder.add_stat_panel(
        "Fouling Alerts",
        'sum(gl014_fouling_alerts_total{exchanger_id=~"$exchanger_id"})',
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 6},
    )

    # Fouling Trend
    builder.add_time_series_panel(
        "Fouling Resistance Trend",
        [
            {"expr": 'gl014_fouling_resistance_m2kw{exchanger_id=~"$exchanger_id", side="shell"}', "legendFormat": "{{exchanger_id}} Shell"},
            {"expr": 'gl014_fouling_resistance_m2kw{exchanger_id=~"$exchanger_id", side="tube"}', "legendFormat": "{{exchanger_id}} Tube"},
        ],
        unit="m2-K/W",
        grid_pos={"x": 0, "y": 6, "w": 24, "h": 8},
    )

    # Fouling by Location
    builder.add_time_series_panel(
        "Fouling by Location",
        [
            {"expr": 'avg(gl014_fouling_resistance_m2kw{location=~"$location"}) by (location)', "legendFormat": "{{location}}"},
        ],
        unit="m2-K/W",
        grid_pos={"x": 0, "y": 14, "w": 12, "h": 8},
    )

    # Fouling Distribution
    builder.add_table_panel(
        "Fouling Status by Exchanger",
        'topk(20, gl014_fouling_resistance_m2kw{exchanger_id=~"$exchanger_id"})',
        grid_pos={"x": 12, "y": 14, "w": 12, "h": 8},
    )

    return builder.build()


def _build_performance_trends_dashboard() -> Dict[str, Any]:
    """Build Performance Trends dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-014 Performance Trends",
        uid="gl014-performance",
        description="Thermal performance metrics over time",
        tags=["gl-014", "heat-exchangers", "performance"],
    )

    # Summary Stats
    builder.add_gauge_panel(
        "Thermal Efficiency",
        'gl014_thermal_efficiency{exchanger_id=~"$exchanger_id"}',
        min_val=0,
        max_val=100,
        unit="percent",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "green", "value": 85},
        ],
        grid_pos={"x": 0, "y": 0, "w": 6, "h": 6},
    )

    builder.add_gauge_panel(
        "U-Value Ratio",
        'gl014_u_value_ratio{exchanger_id=~"$exchanger_id"}',
        min_val=0,
        max_val=1,
        unit="percentunit",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 0.7},
            {"color": "green", "value": 0.85},
        ],
        grid_pos={"x": 6, "y": 0, "w": 6, "h": 6},
    )

    builder.add_stat_panel(
        "Heat Duty (kW)",
        'gl014_heat_duty_kw{exchanger_id=~"$exchanger_id"}',
        unit="kW",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 6},
    )

    builder.add_stat_panel(
        "LMTD (K)",
        'gl014_lmtd_k{exchanger_id=~"$exchanger_id"}',
        unit="K",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 6},
    )

    builder.add_stat_panel(
        "Approach Temp",
        'gl014_approach_temp_k{exchanger_id=~"$exchanger_id"}',
        unit="K",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 6},
    )

    # Performance Trends
    builder.add_time_series_panel(
        "Thermal Efficiency Trend",
        [
            {"expr": 'gl014_thermal_efficiency{exchanger_id=~"$exchanger_id"}', "legendFormat": "{{exchanger_id}}"},
        ],
        unit="percent",
        grid_pos={"x": 0, "y": 6, "w": 12, "h": 8},
    )

    builder.add_time_series_panel(
        "U-Value Trend",
        [
            {"expr": 'gl014_u_value_wm2k{exchanger_id=~"$exchanger_id"}', "legendFormat": "{{exchanger_id}}"},
        ],
        unit="W/m2-K",
        grid_pos={"x": 12, "y": 6, "w": 12, "h": 8},
    )

    # Pressure Drop
    builder.add_time_series_panel(
        "Pressure Drop Ratio",
        [
            {"expr": 'gl014_pressure_drop_ratio{exchanger_id=~"$exchanger_id", side="shell"}', "legendFormat": "{{exchanger_id}} Shell"},
            {"expr": 'gl014_pressure_drop_ratio{exchanger_id=~"$exchanger_id", side="tube"}', "legendFormat": "{{exchanger_id}} Tube"},
        ],
        unit="percentunit",
        grid_pos={"x": 0, "y": 14, "w": 24, "h": 8},
    )

    return builder.build()


def _build_cleaning_schedule_dashboard() -> Dict[str, Any]:
    """Build Cleaning Schedule Status dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-014 Cleaning Schedule",
        uid="gl014-cleaning",
        description="Cleaning schedule and maintenance planning",
        tags=["gl-014", "heat-exchangers", "maintenance", "cleaning"],
    )

    # Summary Stats
    builder.add_stat_panel(
        "Cleanings This Month",
        'sum(increase(gl014_cleaning_schedules_generated_total[30d]))',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Urgent Cleanings",
        'sum(gl014_cleaning_schedules_generated_total{urgency="urgent"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 2},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Critical Cleanings",
        'sum(gl014_cleaning_schedules_generated_total{urgency="critical"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 3},
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Avg Days to Cleaning",
        'avg(gl014_days_to_cleaning{exchanger_id=~"$exchanger_id"})',
        unit="d",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Est. Cleaning Cost",
        'sum(gl014_estimated_cleaning_cost_usd{exchanger_id=~"$exchanger_id"})',
        unit="currencyUSD",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Est. ROI",
        'avg(gl014_cleaning_roi_percent{exchanger_id=~"$exchanger_id"})',
        unit="percent",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Cleaning Timeline
    builder.add_time_series_panel(
        "Cleaning Schedules by Urgency",
        [
            {"expr": 'sum(gl014_cleaning_schedules_generated_total{urgency="routine"}) by (urgency)', "legendFormat": "Routine"},
            {"expr": 'sum(gl014_cleaning_schedules_generated_total{urgency="planned"}) by (urgency)', "legendFormat": "Planned"},
            {"expr": 'sum(gl014_cleaning_schedules_generated_total{urgency="urgent"}) by (urgency)', "legendFormat": "Urgent"},
            {"expr": 'sum(gl014_cleaning_schedules_generated_total{urgency="critical"}) by (urgency)', "legendFormat": "Critical"},
        ],
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Cleaning by Method
    builder.add_time_series_panel(
        "Cleaning by Method",
        [
            {"expr": 'sum(gl014_cleaning_schedules_generated_total) by (cleaning_method)', "legendFormat": "{{cleaning_method}}"},
        ],
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Upcoming Cleanings Table
    builder.add_table_panel(
        "Upcoming Cleanings",
        'gl014_days_to_cleaning{exchanger_id=~"$exchanger_id"}',
        grid_pos={"x": 0, "y": 12, "w": 24, "h": 8},
    )

    return builder.build()


def _build_economic_impact_dashboard() -> Dict[str, Any]:
    """Build Economic Impact dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-014 Economic Impact",
        uid="gl014-economic",
        description="Cost analysis and potential savings",
        tags=["gl-014", "heat-exchangers", "economics"],
    )

    # Summary Stats
    builder.add_stat_panel(
        "Annual Energy Loss",
        'sum(gl014_annual_energy_loss_mwh{exchanger_id=~"$exchanger_id"})',
        unit="MWh",
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Annual Cost Impact",
        'sum(gl014_annual_cost_loss_usd{exchanger_id=~"$exchanger_id"})',
        unit="currencyUSD",
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Potential Savings",
        'sum(gl014_potential_savings_usd{exchanger_id=~"$exchanger_id"})',
        unit="currencyUSD",
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Avg Payback (days)",
        'avg(gl014_payback_days{exchanger_id=~"$exchanger_id"})',
        unit="d",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Cleaning ROI",
        'avg(gl014_cleaning_roi_percent{exchanger_id=~"$exchanger_id"})',
        unit="percent",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Total Cleaning Cost",
        'sum(gl014_estimated_cleaning_cost_usd{exchanger_id=~"$exchanger_id"})',
        unit="currencyUSD",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Cost Trends
    builder.add_time_series_panel(
        "Energy Cost Trend",
        [
            {"expr": 'sum(gl014_energy_cost_loss_usd{exchanger_id=~"$exchanger_id"}) by (exchanger_id)', "legendFormat": "{{exchanger_id}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Savings Opportunity
    builder.add_time_series_panel(
        "Potential Savings Trend",
        [
            {"expr": 'sum(gl014_potential_savings_usd{exchanger_id=~"$exchanger_id"}) by (savings_type)', "legendFormat": "{{savings_type}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Economic Impact Table
    builder.add_table_panel(
        "Economic Impact by Exchanger",
        'gl014_annual_cost_loss_usd{exchanger_id=~"$exchanger_id"}',
        grid_pos={"x": 0, "y": 12, "w": 24, "h": 8},
    )

    return builder.build()


def _build_fleet_comparison_dashboard() -> Dict[str, Any]:
    """Build Fleet Comparison dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-014 Fleet Comparison",
        uid="gl014-fleet",
        description="Cross-exchanger benchmarking and comparison",
        tags=["gl-014", "heat-exchangers", "fleet", "benchmarking"],
    )

    # Fleet Statistics
    builder.add_stat_panel(
        "Fleet Size",
        'count(gl014_health_index)',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Healthy (>85%)",
        'count(gl014_health_index > 85)',
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 5},
            {"color": "green", "value": 10},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Degraded (70-85%)",
        'count(gl014_health_index >= 70 and gl014_health_index <= 85)',
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Poor (<70%)",
        'count(gl014_health_index < 70)',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 2},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Best Performer",
        'max(gl014_health_index)',
        unit="percent",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Worst Performer",
        'min(gl014_health_index)',
        unit="percent",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "green", "value": 85},
        ],
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Health Index Comparison
    builder.add_time_series_panel(
        "Health Index by Exchanger",
        [
            {"expr": 'gl014_health_index{exchanger_id=~"$exchanger_id"}', "legendFormat": "{{exchanger_id}}"},
        ],
        unit="percent",
        grid_pos={"x": 0, "y": 4, "w": 24, "h": 8},
    )

    # Fouling Comparison
    builder.add_time_series_panel(
        "Fouling Resistance Comparison",
        [
            {"expr": 'gl014_fouling_resistance_m2kw{exchanger_id=~"$exchanger_id"}', "legendFormat": "{{exchanger_id}} {{side}}"},
        ],
        unit="m2-K/W",
        grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
    )

    # Efficiency Comparison
    builder.add_time_series_panel(
        "Thermal Efficiency Comparison",
        [
            {"expr": 'gl014_thermal_efficiency{exchanger_id=~"$exchanger_id"}', "legendFormat": "{{exchanger_id}}"},
        ],
        unit="percent",
        grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
    )

    # Fleet Ranking Table
    builder.add_table_panel(
        "Fleet Performance Ranking",
        'topk(50, gl014_health_index)',
        grid_pos={"x": 0, "y": 20, "w": 24, "h": 8},
    )

    return builder.build()


# =============================================================================
# PRE-BUILT DASHBOARD INSTANCES
# =============================================================================

HEAT_EXCHANGER_OVERVIEW_DASHBOARD = _build_heat_exchanger_overview_dashboard()
FOULING_MONITORING_DASHBOARD = _build_fouling_monitoring_dashboard()
PERFORMANCE_TRENDS_DASHBOARD = _build_performance_trends_dashboard()
CLEANING_SCHEDULE_DASHBOARD = _build_cleaning_schedule_dashboard()
ECONOMIC_IMPACT_DASHBOARD = _build_economic_impact_dashboard()
FLEET_COMPARISON_DASHBOARD = _build_fleet_comparison_dashboard()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_dashboards() -> Dict[str, Dict[str, Any]]:
    """
    Get all pre-built dashboards.

    Returns:
        Dictionary mapping dashboard names to JSON configurations
    """
    return {
        "overview": HEAT_EXCHANGER_OVERVIEW_DASHBOARD,
        "fouling": FOULING_MONITORING_DASHBOARD,
        "performance": PERFORMANCE_TRENDS_DASHBOARD,
        "cleaning": CLEANING_SCHEDULE_DASHBOARD,
        "economic": ECONOMIC_IMPACT_DASHBOARD,
        "fleet": FLEET_COMPARISON_DASHBOARD,
    }


def export_dashboard_json(
    dashboard: Dict[str, Any],
    filepath: str,
    indent: int = 2
) -> None:
    """
    Export a dashboard to JSON file.

    Args:
        dashboard: Dashboard configuration dictionary
        filepath: Output file path
        indent: JSON indentation level
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dashboard, f, indent=indent)

    logger.info(f"Dashboard exported to: {filepath}")


def import_dashboard_json(filepath: str) -> Dict[str, Any]:
    """
    Import a dashboard from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Dashboard configuration dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


__all__ = [
    # Dashboard Builder
    "GrafanaDashboardBuilder",
    "GrafanaPanel",

    # Panel Types
    "PanelType",
    "DataSourceType",

    # Pre-built Dashboards
    "HEAT_EXCHANGER_OVERVIEW_DASHBOARD",
    "FOULING_MONITORING_DASHBOARD",
    "PERFORMANCE_TRENDS_DASHBOARD",
    "CLEANING_SCHEDULE_DASHBOARD",
    "ECONOMIC_IMPACT_DASHBOARD",
    "FLEET_COMPARISON_DASHBOARD",

    # Utility Functions
    "get_all_dashboards",
    "export_dashboard_json",
    "import_dashboard_json",
]
