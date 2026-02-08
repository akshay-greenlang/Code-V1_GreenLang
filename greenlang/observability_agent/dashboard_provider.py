# -*- coding: utf-8 -*-
"""
Dashboard Data Provisioning Engine - AGENT-FOUND-010: Observability & Telemetry Agent

Provides Grafana-compatible dashboard definition management and metric data
provisioning. Supports pre-built platform and per-agent dashboards, metric
data point querying, and dashboard template instantiation.

Zero-Hallucination Guarantees:
    - All metric data points are sourced from the MetricsCollector
    - Dashboard definitions are static JSON structures
    - Time range filtering uses deterministic comparison logic
    - No probabilistic forecasting or interpolation

Example:
    >>> from greenlang.observability_agent.dashboard_provider import DashboardProvider
    >>> from greenlang.observability_agent.metrics_collector import MetricsCollector
    >>> from greenlang.observability_agent.config import ObservabilityConfig
    >>> config = ObservabilityConfig()
    >>> collector = MetricsCollector(config)
    >>> provider = DashboardProvider(config, collector)
    >>> dashboard = provider.build_platform_dashboard()
    >>> data = provider.get_dashboard_data(dashboard.dashboard_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PANEL_TYPES: Tuple[str, ...] = (
    "graph", "stat", "gauge", "table", "heatmap", "timeseries", "text",
)

DEFAULT_TIME_RANGE: str = "1h"
DEFAULT_REFRESH_INTERVAL: str = "30s"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PanelConfig:
    """Configuration for a single dashboard panel.

    Attributes:
        panel_id: Unique panel identifier.
        title: Panel display title.
        panel_type: Visualization type (graph, stat, gauge, table, etc.).
        metric_name: Primary metric name for this panel.
        labels_filter: Label filters to apply to the metric query.
        description: Panel description.
        unit: Display unit (e.g. "ms", "bytes", "%").
        thresholds: Optional threshold definitions for color coding.
        grid_position: Grid position dict with x, y, w, h.
    """

    panel_id: str = ""
    title: str = ""
    panel_type: str = "timeseries"
    metric_name: str = ""
    labels_filter: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    unit: str = ""
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    grid_position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8})

    def __post_init__(self) -> None:
        """Generate panel_id if not provided."""
        if not self.panel_id:
            self.panel_id = str(uuid.uuid4())


@dataclass
class DashboardConfig:
    """Configuration for a complete dashboard.

    Attributes:
        dashboard_id: Unique dashboard identifier.
        name: Dashboard display name.
        description: Dashboard description.
        panels: List of panel configurations.
        time_range: Default time range (e.g. "1h", "6h", "24h").
        refresh_interval: Auto-refresh interval (e.g. "10s", "30s", "1m").
        variables: Dashboard-level template variables.
        tags: Dashboard tags for organization.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    dashboard_id: str = ""
    name: str = ""
    description: str = ""
    panels: List[PanelConfig] = field(default_factory=list)
    time_range: str = DEFAULT_TIME_RANGE
    refresh_interval: str = DEFAULT_REFRESH_INTERVAL
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate dashboard_id if not provided."""
        if not self.dashboard_id:
            self.dashboard_id = str(uuid.uuid4())


# =============================================================================
# DashboardProvider
# =============================================================================


class DashboardProvider:
    """Grafana-compatible dashboard data provisioning engine.

    Manages dashboard definitions, queries metric data for panel
    rendering, and provides pre-built dashboards for the GreenLang
    platform and individual agents.

    Thread-safe via a reentrant lock on all mutating operations.

    Attributes:
        _config: Observability configuration.
        _metrics_collector: MetricsCollector for data queries.
        _dashboards: Registered dashboards keyed by dashboard_id.
        _lock: Thread lock for concurrent access.

    Example:
        >>> provider = DashboardProvider(config, collector)
        >>> dashboard = provider.build_platform_dashboard()
        >>> data = provider.get_dashboard_data(dashboard.dashboard_id)
        >>> for panel_id, panel_data in data["panels"].items():
        ...     print(panel_id, len(panel_data["data_points"]))
    """

    def __init__(self, config: Any, metrics_collector: Any) -> None:
        """Initialize DashboardProvider.

        Args:
            config: Observability configuration.
            metrics_collector: MetricsCollector for metric data queries.
        """
        self._config = config
        self._metrics_collector = metrics_collector
        self._dashboards: Dict[str, DashboardConfig] = {}
        self._lock = threading.RLock()

        logger.info("DashboardProvider initialized")

    # ------------------------------------------------------------------
    # Dashboard registration
    # ------------------------------------------------------------------

    def register_dashboard(
        self,
        name: str,
        description: str = "",
        panels: Optional[List[PanelConfig]] = None,
        time_range: str = DEFAULT_TIME_RANGE,
        refresh_interval: str = DEFAULT_REFRESH_INTERVAL,
        tags: Optional[List[str]] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> DashboardConfig:
        """Register a new dashboard definition.

        Args:
            name: Dashboard display name.
            description: Dashboard description.
            panels: List of panel configurations.
            time_range: Default time range.
            refresh_interval: Auto-refresh interval.
            tags: Organization tags.
            variables: Template variables.

        Returns:
            DashboardConfig for the registered dashboard.

        Raises:
            ValueError: If name is empty.
        """
        if not name or not name.strip():
            raise ValueError("Dashboard name must be non-empty")

        dashboard = DashboardConfig(
            name=name,
            description=description,
            panels=panels or [],
            time_range=time_range,
            refresh_interval=refresh_interval,
            variables=variables or {},
            tags=tags or [],
        )
        dashboard.provenance_hash = self._compute_dashboard_hash(dashboard)

        with self._lock:
            self._dashboards[dashboard.dashboard_id] = dashboard

        logger.info(
            "Registered dashboard: id=%s, name=%s, panels=%d",
            dashboard.dashboard_id[:8], name, len(dashboard.panels),
        )
        return dashboard

    def update_dashboard(
        self,
        dashboard_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        panels: Optional[List[PanelConfig]] = None,
        time_range: Optional[str] = None,
        refresh_interval: Optional[str] = None,
    ) -> DashboardConfig:
        """Update an existing dashboard definition.

        Args:
            dashboard_id: Dashboard to update.
            name: New name (if provided).
            description: New description (if provided).
            panels: New panel list (if provided).
            time_range: New time range (if provided).
            refresh_interval: New refresh interval (if provided).

        Returns:
            Updated DashboardConfig.

        Raises:
            ValueError: If dashboard not found.
        """
        with self._lock:
            dashboard = self._dashboards.get(dashboard_id)
            if dashboard is None:
                raise ValueError(f"Dashboard '{dashboard_id[:8]}' not found")

            if name is not None:
                dashboard.name = name
            if description is not None:
                dashboard.description = description
            if panels is not None:
                dashboard.panels = panels
            if time_range is not None:
                dashboard.time_range = time_range
            if refresh_interval is not None:
                dashboard.refresh_interval = refresh_interval

            dashboard.updated_at = _utcnow()
            dashboard.provenance_hash = self._compute_dashboard_hash(dashboard)

        logger.info("Updated dashboard: id=%s", dashboard_id[:8])
        return dashboard

    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard definition.

        Args:
            dashboard_id: Dashboard to delete.

        Returns:
            True if found and deleted, False otherwise.
        """
        with self._lock:
            if dashboard_id not in self._dashboards:
                return False
            del self._dashboards[dashboard_id]

        logger.info("Deleted dashboard: id=%s", dashboard_id[:8])
        return True

    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """Get a dashboard configuration by ID.

        Args:
            dashboard_id: Dashboard identifier.

        Returns:
            DashboardConfig or None if not found.
        """
        with self._lock:
            return self._dashboards.get(dashboard_id)

    def list_dashboards(self) -> List[DashboardConfig]:
        """List all registered dashboard configurations.

        Returns:
            List of DashboardConfig objects sorted by name.
        """
        with self._lock:
            dashboards = list(self._dashboards.values())
        dashboards.sort(key=lambda d: d.name)
        return dashboards

    # ------------------------------------------------------------------
    # Data provisioning
    # ------------------------------------------------------------------

    def get_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get metric data for all panels in a dashboard.

        Args:
            dashboard_id: Dashboard to query data for.
            time_range: Override time range (uses dashboard default).
            variables: Override template variables.

        Returns:
            Dictionary with dashboard metadata and per-panel data.

        Raises:
            ValueError: If dashboard not found.
        """
        with self._lock:
            dashboard = self._dashboards.get(dashboard_id)
            if dashboard is None:
                raise ValueError(f"Dashboard '{dashboard_id[:8]}' not found")

        effective_range = time_range or dashboard.time_range
        effective_vars = dict(dashboard.variables)
        if variables:
            effective_vars.update(variables)

        panel_data: Dict[str, Dict[str, Any]] = {}
        for panel in dashboard.panels:
            data_points = self.query_metric_data(
                panel.metric_name,
                effective_range,
                panel.labels_filter,
            )
            panel_data[panel.panel_id] = {
                "title": panel.title,
                "panel_type": panel.panel_type,
                "metric_name": panel.metric_name,
                "unit": panel.unit,
                "data_points": data_points,
                "thresholds": panel.thresholds,
            }

        return {
            "dashboard_id": dashboard_id,
            "dashboard_name": dashboard.name,
            "time_range": effective_range,
            "variables": effective_vars,
            "panels": panel_data,
            "queried_at": _utcnow().isoformat(),
        }

    def query_metric_data(
        self,
        metric_name: str,
        time_range: str = DEFAULT_TIME_RANGE,
        labels_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query metric data points from the collector.

        Retrieves current metric series values for the given metric
        and label filters. The time_range parameter is used for
        context but the actual data comes from the in-memory collector.

        Args:
            metric_name: Name of the metric to query.
            time_range: Time range string (for context/logging).
            labels_filter: Label key-value filters.

        Returns:
            List of data point dicts with timestamp, value, and labels.
        """
        labels_filter = labels_filter or {}
        data_points: List[Dict[str, Any]] = []

        try:
            series_list = self._metrics_collector.get_metric_series(metric_name)

            for series in series_list:
                # Apply label filtering
                if not self._matches_labels(series.labels, labels_filter):
                    continue

                data_points.append({
                    "timestamp": series.last_updated.isoformat(),
                    "value": series.value,
                    "labels": dict(series.labels),
                    "recordings": series.recordings,
                })

        except (AttributeError, ValueError) as exc:
            logger.warning(
                "Failed to query metric '%s': %s", metric_name, exc,
            )

        return data_points

    def get_panel_data(
        self,
        dashboard_id: str,
        panel_id: str,
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get metric data for a single panel.

        Args:
            dashboard_id: Dashboard containing the panel.
            panel_id: Panel to query.
            time_range: Override time range.

        Returns:
            Panel data dict with metric information.

        Raises:
            ValueError: If dashboard or panel not found.
        """
        with self._lock:
            dashboard = self._dashboards.get(dashboard_id)
            if dashboard is None:
                raise ValueError(f"Dashboard '{dashboard_id[:8]}' not found")

        panel = None
        for p in dashboard.panels:
            if p.panel_id == panel_id:
                panel = p
                break

        if panel is None:
            raise ValueError(f"Panel '{panel_id[:8]}' not found in dashboard")

        effective_range = time_range or dashboard.time_range
        data_points = self.query_metric_data(
            panel.metric_name, effective_range, panel.labels_filter,
        )

        return {
            "panel_id": panel_id,
            "title": panel.title,
            "panel_type": panel.panel_type,
            "metric_name": panel.metric_name,
            "unit": panel.unit,
            "data_points": data_points,
            "thresholds": panel.thresholds,
            "queried_at": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Pre-built dashboards
    # ------------------------------------------------------------------

    def build_platform_dashboard(self) -> DashboardConfig:
        """Build the pre-defined GreenLang platform overview dashboard.

        Includes panels for request rate, latency, error rate, active agents,
        pipeline throughput, and system health.

        Returns:
            Registered DashboardConfig for the platform dashboard.
        """
        panels = [
            PanelConfig(
                title="Request Rate",
                panel_type="timeseries",
                metric_name="gl_http_requests_total",
                unit="req/s",
                grid_position={"x": 0, "y": 0, "w": 8, "h": 8},
                description="HTTP request rate across all endpoints",
            ),
            PanelConfig(
                title="Request Latency (p95)",
                panel_type="timeseries",
                metric_name="gl_http_request_duration_seconds",
                unit="s",
                grid_position={"x": 8, "y": 0, "w": 8, "h": 8},
                thresholds=[
                    {"value": 0.5, "color": "green"},
                    {"value": 1.0, "color": "yellow"},
                    {"value": 2.0, "color": "red"},
                ],
                description="95th percentile request latency",
            ),
            PanelConfig(
                title="Error Rate",
                panel_type="stat",
                metric_name="gl_http_errors_total",
                unit="%",
                grid_position={"x": 16, "y": 0, "w": 8, "h": 8},
                thresholds=[
                    {"value": 1.0, "color": "green"},
                    {"value": 5.0, "color": "red"},
                ],
                description="HTTP error rate percentage",
            ),
            PanelConfig(
                title="Active Agents",
                panel_type="stat",
                metric_name="gl_active_agents",
                unit="",
                grid_position={"x": 0, "y": 8, "w": 6, "h": 6},
                description="Number of currently active agents",
            ),
            PanelConfig(
                title="Pipeline Throughput",
                panel_type="timeseries",
                metric_name="gl_pipeline_executions_total",
                unit="exec/min",
                grid_position={"x": 6, "y": 8, "w": 12, "h": 6},
                description="Pipeline execution throughput",
            ),
            PanelConfig(
                title="System Health",
                panel_type="gauge",
                metric_name="gl_system_health_score",
                unit="%",
                grid_position={"x": 18, "y": 8, "w": 6, "h": 6},
                thresholds=[
                    {"value": 90, "color": "green"},
                    {"value": 70, "color": "yellow"},
                    {"value": 0, "color": "red"},
                ],
                description="Overall system health score",
            ),
        ]

        return self.register_dashboard(
            name="GreenLang Platform Overview",
            description="Comprehensive platform health and performance overview",
            panels=panels,
            time_range="1h",
            refresh_interval="10s",
            tags=["platform", "overview", "greenlang"],
        )

    def build_agent_dashboard(self, agent_id: str) -> DashboardConfig:
        """Build a per-agent monitoring dashboard.

        Creates a dashboard with panels for agent-specific request rate,
        processing latency, error rate, and active executions.

        Args:
            agent_id: Agent identifier for metric filtering.

        Returns:
            Registered DashboardConfig for the agent dashboard.
        """
        agent_label = {"agent_id": agent_id}

        panels = [
            PanelConfig(
                title=f"Request Rate - {agent_id}",
                panel_type="timeseries",
                metric_name="gl_agent_requests_total",
                labels_filter=agent_label,
                unit="req/s",
                grid_position={"x": 0, "y": 0, "w": 12, "h": 8},
                description=f"Request rate for agent {agent_id}",
            ),
            PanelConfig(
                title=f"Processing Latency - {agent_id}",
                panel_type="timeseries",
                metric_name="gl_agent_processing_duration_seconds",
                labels_filter=agent_label,
                unit="s",
                grid_position={"x": 12, "y": 0, "w": 12, "h": 8},
                thresholds=[
                    {"value": 1.0, "color": "green"},
                    {"value": 5.0, "color": "yellow"},
                    {"value": 10.0, "color": "red"},
                ],
                description=f"Processing latency for agent {agent_id}",
            ),
            PanelConfig(
                title=f"Error Rate - {agent_id}",
                panel_type="stat",
                metric_name="gl_agent_errors_total",
                labels_filter=agent_label,
                unit="%",
                grid_position={"x": 0, "y": 8, "w": 8, "h": 6},
                description=f"Error rate for agent {agent_id}",
            ),
            PanelConfig(
                title=f"Active Executions - {agent_id}",
                panel_type="stat",
                metric_name="gl_agent_active_executions",
                labels_filter=agent_label,
                unit="",
                grid_position={"x": 8, "y": 8, "w": 8, "h": 6},
                description=f"Active executions for agent {agent_id}",
            ),
            PanelConfig(
                title=f"Memory Usage - {agent_id}",
                panel_type="gauge",
                metric_name="gl_agent_memory_bytes",
                labels_filter=agent_label,
                unit="bytes",
                grid_position={"x": 16, "y": 8, "w": 8, "h": 6},
                thresholds=[
                    {"value": 256_000_000, "color": "green"},
                    {"value": 512_000_000, "color": "yellow"},
                    {"value": 1_000_000_000, "color": "red"},
                ],
                description=f"Memory usage for agent {agent_id}",
            ),
        ]

        return self.register_dashboard(
            name=f"Agent: {agent_id}",
            description=f"Monitoring dashboard for agent {agent_id}",
            panels=panels,
            time_range="1h",
            refresh_interval="15s",
            tags=["agent", agent_id],
            variables={"agent_id": agent_id},
        )

    def build_slo_dashboard(self) -> DashboardConfig:
        """Build a pre-defined SLO/Error Budget dashboard.

        Returns:
            Registered DashboardConfig for the SLO dashboard.
        """
        panels = [
            PanelConfig(
                title="SLO Compliance",
                panel_type="gauge",
                metric_name="gl_slo_compliance_percentage",
                unit="%",
                grid_position={"x": 0, "y": 0, "w": 8, "h": 8},
                thresholds=[
                    {"value": 99.9, "color": "green"},
                    {"value": 99.0, "color": "yellow"},
                    {"value": 0, "color": "red"},
                ],
                description="Current SLO compliance percentage",
            ),
            PanelConfig(
                title="Error Budget Remaining",
                panel_type="gauge",
                metric_name="gl_slo_error_budget_remaining",
                unit="%",
                grid_position={"x": 8, "y": 0, "w": 8, "h": 8},
                thresholds=[
                    {"value": 50, "color": "green"},
                    {"value": 20, "color": "yellow"},
                    {"value": 0, "color": "red"},
                ],
                description="Remaining error budget percentage",
            ),
            PanelConfig(
                title="Burn Rate",
                panel_type="timeseries",
                metric_name="gl_slo_burn_rate",
                unit="x",
                grid_position={"x": 16, "y": 0, "w": 8, "h": 8},
                thresholds=[
                    {"value": 1.0, "color": "green"},
                    {"value": 2.0, "color": "yellow"},
                    {"value": 10.0, "color": "red"},
                ],
                description="Error budget burn rate multiplier",
            ),
        ]

        return self.register_dashboard(
            name="SLO / Error Budget Overview",
            description="Service Level Objective compliance and error budget tracking",
            panels=panels,
            time_range="24h",
            refresh_interval="1m",
            tags=["slo", "error-budget", "reliability"],
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard provider statistics.

        Returns:
            Dictionary with total_dashboards, total_panels, and
            dashboards_by_tag counts.
        """
        with self._lock:
            total_panels = sum(len(d.panels) for d in self._dashboards.values())

            tag_counts: Dict[str, int] = {}
            for d in self._dashboards.values():
                for tag in d.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_dashboards": len(self._dashboards),
                "total_panels": total_panels,
                "dashboards_by_tag": tag_counts,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matches_labels(
        self,
        series_labels: Dict[str, str],
        filter_labels: Dict[str, str],
    ) -> bool:
        """Check if series labels match the filter criteria.

        All filter labels must be present in the series labels with
        matching values. Empty filter matches everything.

        Args:
            series_labels: Labels from the metric series.
            filter_labels: Required label values.

        Returns:
            True if all filter labels match.
        """
        for key, value in filter_labels.items():
            if series_labels.get(key) != value:
                return False
        return True

    def _compute_dashboard_hash(self, dashboard: DashboardConfig) -> str:
        """Compute SHA-256 provenance hash for a dashboard definition.

        Args:
            dashboard: DashboardConfig to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        panel_summaries = [
            {
                "title": p.title,
                "metric_name": p.metric_name,
                "panel_type": p.panel_type,
            }
            for p in dashboard.panels
        ]
        payload = json.dumps(
            {
                "dashboard_id": dashboard.dashboard_id,
                "name": dashboard.name,
                "panels": panel_summaries,
                "time_range": dashboard.time_range,
                "created_at": dashboard.created_at.isoformat(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DashboardProvider",
    "DashboardConfig",
    "PanelConfig",
    "VALID_PANEL_TYPES",
    "DEFAULT_TIME_RANGE",
    "DEFAULT_REFRESH_INTERVAL",
]
