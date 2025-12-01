# -*- coding: utf-8 -*-
"""
Grafana Dashboard Definitions for GL-015 INSULSCAN.

This module provides JSON dashboard configurations for Grafana:
- Insulation Health Overview - Fleet status summary
- Heat Loss Monitoring - Energy loss trends and analysis
- Thermal Imaging Results - Inspection findings and hotspots
- Repair Tracking - Maintenance prioritization and scheduling
- Energy Impact - Cost analysis and carbon emissions
- Facility Comparison - Cross-facility benchmarking

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
        >>> builder = GrafanaDashboardBuilder("Insulation Health Overview")
        >>> builder.add_row("Summary")
        >>> builder.add_stat_panel("Total Facilities", "gl015_facilities_monitored")
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
        self.uid = uid or title.lower().replace(" ", "-").replace("gl-015-", "gl015-")
        self.description = description
        self.tags = tags or ["gl-015", "insulation", "thermal-imaging"]
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

    def add_heatmap_panel(
        self,
        title: str,
        query: str,
        grid_pos: Optional[Dict[str, int]] = None
    ) -> "GrafanaDashboardBuilder":
        """
        Add a heatmap panel (useful for thermal imaging data).

        Args:
            title: Panel title
            query: Prometheus query
            grid_pos: Custom grid position

        Returns:
            Self for method chaining
        """
        if grid_pos is None:
            grid_pos = {"x": 0, "y": self._current_y, "w": 12, "h": 8}

        panel = GrafanaPanel(
            id=self._panel_id,
            title=title,
            panel_type=PanelType.HEATMAP,
            grid_pos=grid_pos,
            targets=[{
                "expr": query,
                "refId": "A",
                "datasource": {"type": "prometheus", "uid": self._datasource},
            }],
            options={
                "color": {
                    "mode": "scheme",
                    "scheme": "Turbo",
                },
                "yAxis": {"unit": "celsius"},
            },
            field_config={
                "defaults": {},
                "overrides": [],
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
                        "name": "facility_id",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self._datasource},
                        "query": 'label_values(gl015_total_heat_loss_watts, facility_id)',
                        "refresh": 2,
                        "multi": True,
                        "includeAll": True,
                    },
                    {
                        "name": "zone",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self._datasource},
                        "query": 'label_values(gl015_total_heat_loss_watts, zone)',
                        "refresh": 2,
                        "multi": True,
                        "includeAll": True,
                    },
                    {
                        "name": "equipment_type",
                        "type": "query",
                        "datasource": {"type": "prometheus", "uid": self._datasource},
                        "query": 'label_values(gl015_equipment_by_condition, equipment_type)',
                        "refresh": 2,
                        "multi": True,
                        "includeAll": True,
                    },
                ]
            },
            "time": {"from": "now-24h", "to": "now"},
            "annotations": {"list": []},
            "links": [],
        }


# =============================================================================
# PRE-BUILT DASHBOARD DEFINITIONS
# =============================================================================

def _build_insulation_health_overview_dashboard() -> Dict[str, Any]:
    """Build Insulation Health Overview dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-015 Insulation Health Overview",
        uid="gl015-overview",
        description="Overview of insulation health across all monitored facilities",
        tags=["gl-015", "insulation", "overview"],
    )

    # Summary Row - Key Metrics
    builder.add_stat_panel(
        "Total Facilities",
        'count(count by (facility_id)(gl015_total_heat_loss_watts))',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Active Alerts",
        'count(ALERTS{alertstate="firing", job=~"gl-015.*"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Avg Insulation Efficiency",
        'avg(gl015_insulation_efficiency_percent{facility_id=~"$facility_id"})',
        unit="percent",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "green", "value": 85},
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Critical Equipment",
        'sum(gl015_equipment_by_condition{condition_severity="critical"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 2},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Monthly Energy Cost",
        'sum(gl015_energy_cost_dollars{period="monthly", facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Images Processed Today",
        'sum(increase(gl015_thermal_images_processed_total[24h]))',
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Efficiency Trend
    builder.add_time_series_panel(
        "Insulation Efficiency Trend",
        [
            {"expr": 'avg(gl015_insulation_efficiency_percent{facility_id=~"$facility_id"})', "legendFormat": "Average"},
            {"expr": 'min(gl015_insulation_efficiency_percent{facility_id=~"$facility_id"})', "legendFormat": "Minimum"},
            {"expr": 'max(gl015_insulation_efficiency_percent{facility_id=~"$facility_id"})', "legendFormat": "Maximum"},
        ],
        unit="percent",
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Alert List
    builder.add_alert_list_panel(
        "Active Alerts",
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Equipment by Condition
    builder.add_time_series_panel(
        "Equipment by Condition Severity",
        [
            {"expr": 'sum(gl015_equipment_by_condition{condition_severity="good"}) by (condition_severity)', "legendFormat": "Good"},
            {"expr": 'sum(gl015_equipment_by_condition{condition_severity="fair"}) by (condition_severity)', "legendFormat": "Fair"},
            {"expr": 'sum(gl015_equipment_by_condition{condition_severity="poor"}) by (condition_severity)', "legendFormat": "Poor"},
            {"expr": 'sum(gl015_equipment_by_condition{condition_severity="critical"}) by (condition_severity)', "legendFormat": "Critical"},
        ],
        grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
    )

    # Inspection Activity
    builder.add_time_series_panel(
        "Daily Inspection Activity",
        [
            {"expr": 'sum(increase(gl015_inspections_completed_total[1d]))', "legendFormat": "Inspections"},
            {"expr": 'sum(increase(gl015_thermal_images_processed_total[1d]))', "legendFormat": "Images Processed"},
        ],
        grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
    )

    # Facility Status Table
    builder.add_table_panel(
        "Facility Status Summary",
        'avg(gl015_insulation_efficiency_percent) by (facility_id)',
        grid_pos={"x": 0, "y": 20, "w": 24, "h": 8},
    )

    return builder.build()


def _build_heat_loss_monitoring_dashboard() -> Dict[str, Any]:
    """Build Heat Loss Monitoring dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-015 Heat Loss Monitoring",
        uid="gl015-heat-loss",
        description="Detailed heat loss analysis and trends",
        tags=["gl-015", "insulation", "heat-loss", "energy"],
    )

    # Heat Loss Summary
    builder.add_stat_panel(
        "Total Heat Loss (kW)",
        'sum(gl015_total_heat_loss_watts{facility_id=~"$facility_id"}) / 1000',
        unit="kW",
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Avg Heat Loss/m",
        'avg(gl015_heat_loss_per_meter_wm{facility_id=~"$facility_id"})',
        unit="W/m",
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 50},
            {"color": "red", "value": 100},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Daily Energy Cost",
        'sum(gl015_energy_cost_dollars{period="daily", facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Monthly Energy Cost",
        'sum(gl015_energy_cost_dollars{period="monthly", facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Annual Energy Cost",
        'sum(gl015_energy_cost_dollars{period="annual", facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "CO2 Emissions (kg)",
        'sum(gl015_carbon_emissions_kg{facility_id=~"$facility_id"})',
        unit="kg",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Heat Loss by Zone
    builder.add_time_series_panel(
        "Heat Loss by Zone",
        [
            {"expr": 'sum(gl015_total_heat_loss_watts{facility_id=~"$facility_id"}) by (zone)', "legendFormat": "{{zone}}"},
        ],
        unit="W",
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Heat Loss by Equipment Type
    builder.add_time_series_panel(
        "Heat Loss by Equipment Type",
        [
            {"expr": 'sum(gl015_total_heat_loss_watts{facility_id=~"$facility_id"}) by (equipment_type)', "legendFormat": "{{equipment_type}}"},
        ],
        unit="W",
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Heat Loss per Meter Trend
    builder.add_time_series_panel(
        "Heat Loss per Meter Trend",
        [
            {"expr": 'gl015_heat_loss_per_meter_wm{facility_id=~"$facility_id"}', "legendFormat": "{{insulation_type}} - {{condition}}"},
        ],
        unit="W/m",
        grid_pos={"x": 0, "y": 12, "w": 24, "h": 8},
    )

    # Ambient Temperature vs Heat Loss
    builder.add_time_series_panel(
        "Ambient Temperature Impact",
        [
            {"expr": 'avg(gl015_ambient_temperature_celsius{facility_id=~"$facility_id"})', "legendFormat": "Ambient Temp (C)"},
            {"expr": 'sum(gl015_total_heat_loss_watts{facility_id=~"$facility_id"}) / 10000', "legendFormat": "Heat Loss (x10kW)"},
        ],
        grid_pos={"x": 0, "y": 20, "w": 12, "h": 8},
    )

    # Energy Cost Trend
    builder.add_time_series_panel(
        "Energy Cost Trend",
        [
            {"expr": 'sum(gl015_energy_cost_dollars{period="daily", facility_id=~"$facility_id"}) by (facility_id)', "legendFormat": "{{facility_id}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 12, "y": 20, "w": 12, "h": 8},
    )

    return builder.build()


def _build_thermal_imaging_results_dashboard() -> Dict[str, Any]:
    """Build Thermal Imaging Results dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-015 Thermal Imaging Results",
        uid="gl015-thermal",
        description="Thermal imaging inspection findings and anomaly detection",
        tags=["gl-015", "thermal-imaging", "inspection", "hotspots"],
    )

    # Inspection Summary
    builder.add_stat_panel(
        "Total Inspections",
        'sum(gl015_inspections_completed_total{facility_id=~"$facility_id"})',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Images Processed",
        'sum(gl015_thermal_images_processed_total{facility_id=~"$facility_id"})',
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Hotspots Detected",
        'sum(gl015_hotspots_detected_total{facility_id=~"$facility_id"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 10},
            {"color": "red", "value": 25},
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Critical Hotspots",
        'sum(gl015_hotspots_detected_total{facility_id=~"$facility_id", severity="critical"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 3},
        ],
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Anomalies Classified",
        'sum(gl015_anomalies_classified_total{facility_id=~"$facility_id"})',
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Inspection Coverage",
        'avg(gl015_inspection_coverage_percent{facility_id=~"$facility_id"})',
        unit="percent",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Hotspots by Severity
    builder.add_time_series_panel(
        "Hotspots by Severity",
        [
            {"expr": 'sum(gl015_hotspots_detected_total{facility_id=~"$facility_id", severity="low"})', "legendFormat": "Low"},
            {"expr": 'sum(gl015_hotspots_detected_total{facility_id=~"$facility_id", severity="medium"})', "legendFormat": "Medium"},
            {"expr": 'sum(gl015_hotspots_detected_total{facility_id=~"$facility_id", severity="high"})', "legendFormat": "High"},
            {"expr": 'sum(gl015_hotspots_detected_total{facility_id=~"$facility_id", severity="critical"})', "legendFormat": "Critical"},
        ],
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Anomaly Types Distribution
    builder.add_time_series_panel(
        "Anomaly Types Distribution",
        [
            {"expr": 'sum(gl015_anomalies_classified_total{facility_id=~"$facility_id"}) by (anomaly_type)', "legendFormat": "{{anomaly_type}}"},
        ],
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Surface Temperature Trend
    builder.add_time_series_panel(
        "Surface Temperature Readings",
        [
            {"expr": 'avg(gl015_surface_temperature_distribution_celsius_sum{facility_id=~"$facility_id"} / gl015_surface_temperature_distribution_celsius_count)', "legendFormat": "Avg Surface Temp"},
        ],
        unit="celsius",
        grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
    )

    # Image Processing Performance
    builder.add_time_series_panel(
        "Image Processing Duration (P95)",
        [
            {"expr": 'histogram_quantile(0.95, rate(gl015_image_processing_duration_seconds_bucket[5m]))', "legendFormat": "P95 Duration"},
        ],
        unit="s",
        grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
    )

    # Camera Status
    builder.add_table_panel(
        "Camera Connection Status",
        'gl015_camera_connection_status',
        grid_pos={"x": 0, "y": 20, "w": 12, "h": 6},
    )

    # Recent Inspections
    builder.add_table_panel(
        "Recent Inspection Summary",
        'sum(gl015_inspections_completed_total{facility_id=~"$facility_id"}) by (facility_id, inspection_type)',
        grid_pos={"x": 12, "y": 20, "w": 12, "h": 6},
    )

    return builder.build()


def _build_repair_tracking_dashboard() -> Dict[str, Any]:
    """Build Repair Tracking dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-015 Repair Tracking",
        uid="gl015-repairs",
        description="Maintenance prioritization and repair scheduling",
        tags=["gl-015", "insulation", "maintenance", "repairs"],
    )

    # Repair Summary
    builder.add_stat_panel(
        "Total Repairs Needed",
        'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id"})',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Emergency Repairs",
        'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id", urgency="emergency"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 3},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Urgent Repairs",
        'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id", urgency="urgent"})',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 3},
            {"color": "red", "value": 5},
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Est. Total Repair Cost",
        'sum(gl015_repair_cost_estimate_dollars{facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "CMMS Work Orders",
        'sum(gl015_cmms_work_orders_created_total{facility_id=~"$facility_id"})',
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Avg Time to Failure",
        'avg(gl015_time_to_failure_days{facility_id=~"$facility_id"})',
        unit="d",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 30},
            {"color": "green", "value": 90},
        ],
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Repairs by Urgency
    builder.add_time_series_panel(
        "Repairs by Urgency",
        [
            {"expr": 'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id", urgency="routine"})', "legendFormat": "Routine"},
            {"expr": 'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id", urgency="planned"})', "legendFormat": "Planned"},
            {"expr": 'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id", urgency="urgent"})', "legendFormat": "Urgent"},
            {"expr": 'sum(gl015_repairs_prioritized_total{facility_id=~"$facility_id", urgency="emergency"})', "legendFormat": "Emergency"},
        ],
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Repair Cost by Type
    builder.add_time_series_panel(
        "Repair Cost by Type",
        [
            {"expr": 'sum(gl015_repair_cost_estimate_dollars{facility_id=~"$facility_id"}) by (repair_type)', "legendFormat": "{{repair_type}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Time to Failure Distribution
    builder.add_time_series_panel(
        "Time to Failure by Equipment",
        [
            {"expr": 'gl015_time_to_failure_days{facility_id=~"$facility_id"}', "legendFormat": "{{equipment_id}}"},
        ],
        unit="d",
        grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
    )

    # Degradation Rate Trend
    builder.add_time_series_panel(
        "Degradation Rate Trend",
        [
            {"expr": 'avg(gl015_average_degradation_rate_percent_per_year{facility_id=~"$facility_id"}) by (insulation_type)', "legendFormat": "{{insulation_type}}"},
        ],
        unit="percent",
        grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
    )

    # Equipment Needing Repair Table
    builder.add_table_panel(
        "Equipment Requiring Repair",
        'topk(20, gl015_repairs_prioritized_total{facility_id=~"$facility_id"})',
        grid_pos={"x": 0, "y": 20, "w": 24, "h": 8},
    )

    return builder.build()


def _build_energy_impact_dashboard() -> Dict[str, Any]:
    """Build Energy Impact dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-015 Energy Impact",
        uid="gl015-energy",
        description="Energy cost analysis and carbon emissions tracking",
        tags=["gl-015", "insulation", "energy", "carbon", "sustainability"],
    )

    # Energy Summary
    builder.add_stat_panel(
        "Annual Energy Loss (MWh)",
        'sum(gl015_total_heat_loss_watts{facility_id=~"$facility_id"}) * 8760 / 1000000',
        unit="MWh",
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Annual Energy Cost",
        'sum(gl015_energy_cost_dollars{period="annual", facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Monthly Energy Cost",
        'sum(gl015_energy_cost_dollars{period="monthly", facility_id=~"$facility_id"})',
        unit="currencyUSD",
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Annual CO2 Emissions",
        'sum(gl015_carbon_emissions_kg{facility_id=~"$facility_id"}) * 12',
        unit="kg",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Scope 1 Emissions",
        'sum(gl015_carbon_emissions_kg{facility_id=~"$facility_id", scope="scope1"})',
        unit="kg",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Potential Savings",
        'sum(gl015_repair_cost_estimate_dollars{facility_id=~"$facility_id", urgency=~"urgent|emergency"})',
        unit="currencyUSD",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Energy Cost Trend
    builder.add_time_series_panel(
        "Energy Cost Trend by Facility",
        [
            {"expr": 'sum(gl015_energy_cost_dollars{period="daily", facility_id=~"$facility_id"}) by (facility_id)', "legendFormat": "{{facility_id}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
    )

    # Carbon Emissions Trend
    builder.add_time_series_panel(
        "Carbon Emissions Trend",
        [
            {"expr": 'sum(gl015_carbon_emissions_kg{facility_id=~"$facility_id"}) by (fuel_type)', "legendFormat": "{{fuel_type}}"},
        ],
        unit="kg",
        grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
    )

    # Energy Cost by Tariff Type
    builder.add_time_series_panel(
        "Energy Cost by Tariff Type",
        [
            {"expr": 'sum(gl015_energy_cost_dollars{facility_id=~"$facility_id"}) by (tariff_type)', "legendFormat": "{{tariff_type}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
    )

    # Emissions by Scope
    builder.add_time_series_panel(
        "Emissions by Scope",
        [
            {"expr": 'sum(gl015_carbon_emissions_kg{facility_id=~"$facility_id", scope="scope1"})', "legendFormat": "Scope 1 (Direct)"},
            {"expr": 'sum(gl015_carbon_emissions_kg{facility_id=~"$facility_id", scope="scope2"})', "legendFormat": "Scope 2 (Indirect)"},
        ],
        unit="kg",
        grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
    )

    # Energy Impact by Zone Table
    builder.add_table_panel(
        "Energy Impact by Zone",
        'sum(gl015_energy_cost_dollars{period="monthly", facility_id=~"$facility_id"}) by (facility_id, zone)',
        grid_pos={"x": 0, "y": 20, "w": 24, "h": 8},
    )

    return builder.build()


def _build_facility_comparison_dashboard() -> Dict[str, Any]:
    """Build Facility Comparison dashboard."""
    builder = GrafanaDashboardBuilder(
        title="GL-015 Facility Comparison",
        uid="gl015-comparison",
        description="Cross-facility benchmarking and performance comparison",
        tags=["gl-015", "insulation", "comparison", "benchmarking"],
    )

    # Fleet Statistics
    builder.add_stat_panel(
        "Total Facilities",
        'count(count by (facility_id)(gl015_total_heat_loss_watts))',
        grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Best Efficiency",
        'max(gl015_insulation_efficiency_percent)',
        unit="percent",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 80},
            {"color": "green", "value": 90},
        ],
        grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Worst Efficiency",
        'min(gl015_insulation_efficiency_percent)',
        unit="percent",
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "green", "value": 85},
        ],
        grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Fleet Avg Efficiency",
        'avg(gl015_insulation_efficiency_percent)',
        unit="percent",
        grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Total Heat Loss (MW)",
        'sum(gl015_total_heat_loss_watts) / 1000000',
        unit="MW",
        grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
    )

    builder.add_stat_panel(
        "Total Energy Cost",
        'sum(gl015_energy_cost_dollars{period="monthly"})',
        unit="currencyUSD",
        grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
    )

    # Efficiency Comparison by Facility
    builder.add_time_series_panel(
        "Insulation Efficiency by Facility",
        [
            {"expr": 'avg(gl015_insulation_efficiency_percent) by (facility_id)', "legendFormat": "{{facility_id}}"},
        ],
        unit="percent",
        grid_pos={"x": 0, "y": 4, "w": 24, "h": 8},
    )

    # Heat Loss Comparison
    builder.add_time_series_panel(
        "Heat Loss by Facility",
        [
            {"expr": 'sum(gl015_total_heat_loss_watts) by (facility_id) / 1000', "legendFormat": "{{facility_id}}"},
        ],
        unit="kW",
        grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
    )

    # Energy Cost Comparison
    builder.add_time_series_panel(
        "Monthly Energy Cost by Facility",
        [
            {"expr": 'sum(gl015_energy_cost_dollars{period="monthly"}) by (facility_id)', "legendFormat": "{{facility_id}}"},
        ],
        unit="currencyUSD",
        grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
    )

    # Critical Equipment Comparison
    builder.add_time_series_panel(
        "Critical Equipment by Facility",
        [
            {"expr": 'sum(gl015_equipment_by_condition{condition_severity="critical"}) by (facility_id)', "legendFormat": "{{facility_id}}"},
        ],
        grid_pos={"x": 0, "y": 20, "w": 12, "h": 8},
    )

    # Inspection Coverage Comparison
    builder.add_time_series_panel(
        "Inspection Coverage by Facility",
        [
            {"expr": 'avg(gl015_inspection_coverage_percent) by (facility_id)', "legendFormat": "{{facility_id}}"},
        ],
        unit="percent",
        grid_pos={"x": 12, "y": 20, "w": 12, "h": 8},
    )

    # Facility Ranking Table
    builder.add_table_panel(
        "Facility Performance Ranking",
        'topk(50, avg(gl015_insulation_efficiency_percent) by (facility_id))',
        grid_pos={"x": 0, "y": 28, "w": 24, "h": 8},
    )

    return builder.build()


# =============================================================================
# PRE-BUILT DASHBOARD INSTANCES
# =============================================================================

INSULATION_HEALTH_OVERVIEW_DASHBOARD = _build_insulation_health_overview_dashboard()
HEAT_LOSS_MONITORING_DASHBOARD = _build_heat_loss_monitoring_dashboard()
THERMAL_IMAGING_RESULTS_DASHBOARD = _build_thermal_imaging_results_dashboard()
REPAIR_TRACKING_DASHBOARD = _build_repair_tracking_dashboard()
ENERGY_IMPACT_DASHBOARD = _build_energy_impact_dashboard()
FACILITY_COMPARISON_DASHBOARD = _build_facility_comparison_dashboard()


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
        "overview": INSULATION_HEALTH_OVERVIEW_DASHBOARD,
        "heat_loss": HEAT_LOSS_MONITORING_DASHBOARD,
        "thermal_imaging": THERMAL_IMAGING_RESULTS_DASHBOARD,
        "repairs": REPAIR_TRACKING_DASHBOARD,
        "energy": ENERGY_IMPACT_DASHBOARD,
        "comparison": FACILITY_COMPARISON_DASHBOARD,
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
    "INSULATION_HEALTH_OVERVIEW_DASHBOARD",
    "HEAT_LOSS_MONITORING_DASHBOARD",
    "THERMAL_IMAGING_RESULTS_DASHBOARD",
    "REPAIR_TRACKING_DASHBOARD",
    "ENERGY_IMPACT_DASHBOARD",
    "FACILITY_COMPARISON_DASHBOARD",

    # Utility Functions
    "get_all_dashboards",
    "export_dashboard_json",
    "import_dashboard_json",
]
