# -*- coding: utf-8 -*-
"""
Grafana SDK Pydantic v2 Models
=================================

Data models for all Grafana API objects used by the GreenLang Grafana SDK.
All models use Pydantic v2 with strict validation, sensible defaults, and
JSON-serializable output for direct Grafana API consumption.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GridPos(BaseModel):
    """Panel position within the Grafana dashboard grid (24-column layout)."""

    h: int = Field(default=8, ge=1, le=40, description="Panel height in grid units")
    w: int = Field(default=12, ge=1, le=24, description="Panel width in grid columns")
    x: int = Field(default=0, ge=0, le=23, description="Horizontal position (0-23)")
    y: int = Field(default=0, ge=0, description="Vertical position (row offset)")


class Target(BaseModel):
    """Data source query target for a panel.

    Supports Prometheus (PromQL), Loki (LogQL), and generic data source queries
    with legend formatting and instant/range query modes.
    """

    datasource: Optional[dict[str, str]] = Field(
        default=None,
        description="Datasource reference, e.g. {'type': 'prometheus', 'uid': 'thanos'}",
    )
    expr: str = Field(default="", description="PromQL or LogQL expression")
    legendFormat: str = Field(
        default="__auto",
        description="Legend template, e.g. '{{instance}}' or '__auto'",
    )
    refId: str = Field(default="A", description="Query reference ID (A, B, C, ...)")
    instant: bool = Field(default=False, description="True for instant query, False for range")
    interval: str = Field(default="", description="Minimum step interval, e.g. '15s'")
    editorMode: str = Field(default="code", description="Query editor mode: 'code' or 'builder'")
    hide: bool = Field(default=False, description="Hide this query from the panel")
    format: str = Field(default="time_series", description="Result format: time_series, table, heatmap")

    @field_validator("refId")
    @classmethod
    def validate_ref_id(cls, v: str) -> str:
        """Ensure refId is a non-empty uppercase letter sequence."""
        if not v or not v[0].isupper():
            raise ValueError("refId must start with an uppercase letter (A-Z)")
        return v


class Variable(BaseModel):
    """Dashboard template variable.

    Variables allow dynamic filtering of dashboard panels via drop-down selectors.
    Supports query, custom, datasource, interval, constant, and textbox types.
    """

    name: str = Field(..., min_length=1, description="Variable name used in queries as $name")
    label: str = Field(default="", description="Human-readable label displayed in the UI")
    type: str = Field(
        default="query",
        description="Variable type: query, custom, datasource, interval, constant, textbox",
    )
    query: str = Field(default="", description="Variable query expression or static values")
    datasource: Optional[dict[str, str]] = Field(
        default=None, description="Datasource for query variables"
    )
    refresh: int = Field(
        default=2,
        ge=0,
        le=2,
        description="Refresh: 0=never, 1=on dashboard load, 2=on time range change",
    )
    regex: str = Field(default="", description="Regex to filter/transform values")
    sort: int = Field(
        default=1,
        ge=0,
        le=8,
        description="Sort order: 0=disabled, 1=alpha-asc, 2=alpha-desc, 3=num-asc, etc.",
    )
    multi: bool = Field(default=False, description="Allow multi-value selection")
    includeAll: bool = Field(default=False, description="Add 'All' option")
    allValue: str = Field(default="", description="Custom value for 'All' option, e.g. '.*'")
    current: dict[str, Any] = Field(default_factory=dict, description="Currently selected value")
    hide: int = Field(default=0, ge=0, le=2, description="0=visible, 1=label only, 2=hidden")
    options: list[dict[str, Any]] = Field(
        default_factory=list, description="Selectable options list"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate variable type against allowed Grafana types."""
        allowed = {"query", "custom", "datasource", "interval", "constant", "textbox", "adhoc"}
        if v not in allowed:
            raise ValueError(f"Variable type must be one of {allowed}, got '{v}'")
        return v


class Annotation(BaseModel):
    """Dashboard annotation query.

    Annotations overlay time-based events on panels to mark deployments,
    incidents, or other notable events.
    """

    name: str = Field(default="Annotations & Alerts", description="Annotation layer name")
    datasource: Optional[dict[str, str]] = Field(
        default=None, description="Datasource for annotation queries"
    )
    enable: bool = Field(default=True, description="Whether this annotation layer is active")
    hide: bool = Field(default=False, description="Whether to hide the annotation layer")
    iconColor: str = Field(default="rgba(0, 211, 255, 1)", description="Annotation marker colour")
    type: str = Field(default="dashboard", description="Type: 'dashboard' or 'tags'")
    builtIn: int = Field(default=1, ge=0, le=1, description="1 for built-in annotations, 0 otherwise")
    expr: str = Field(default="", description="PromQL/LogQL expression for query annotations")
    step: str = Field(default="", description="Step interval for annotation queries")
    tags: list[str] = Field(default_factory=list, description="Tags to filter annotations")


class Panel(BaseModel):
    """Grafana dashboard panel.

    Represents a single visualization within a dashboard including its query
    targets, display options, field configuration, and grid position.
    """

    id: int = Field(default=0, ge=0, description="Unique panel ID within the dashboard")
    title: str = Field(default="", description="Panel title displayed at the top")
    type: str = Field(default="timeseries", description="Panel type: timeseries, stat, gauge, etc.")
    description: str = Field(default="", description="Panel description / tooltip text")
    datasource: Optional[dict[str, str]] = Field(
        default=None, description="Default datasource for all targets"
    )
    targets: list[Target] = Field(default_factory=list, description="Query targets for the panel")
    gridPos: GridPos = Field(default_factory=GridPos, description="Panel grid position and size")
    fieldConfig: dict[str, Any] = Field(
        default_factory=lambda: {"defaults": {}, "overrides": []},
        description="Field configuration: units, thresholds, colour, overrides",
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="Panel-specific display options"
    )
    transformations: list[dict[str, Any]] = Field(
        default_factory=list, description="Data transformations applied before display"
    )
    transparent: bool = Field(default=False, description="Transparent panel background")
    repeat: Optional[str] = Field(default=None, description="Variable name to repeat panel by")
    repeatDirection: str = Field(default="h", description="Repeat direction: 'h' (horizontal) or 'v'")
    maxPerRow: int = Field(default=4, ge=1, le=24, description="Max panels per row when repeating")
    links: list[dict[str, Any]] = Field(default_factory=list, description="Panel-level data links")
    collapsed: Optional[bool] = Field(
        default=None, description="For row panels only: whether the row is collapsed"
    )
    panels: Optional[list[Panel]] = Field(
        default=None, description="For row panels: nested panels when collapsed"
    )

    @field_validator("type")
    @classmethod
    def validate_panel_type(cls, v: str) -> str:
        """Validate panel type against known Grafana panel types."""
        known_types = {
            "timeseries", "stat", "gauge", "table", "barchart", "piechart",
            "heatmap", "logs", "text", "bargauge", "histogram", "news",
            "dashlist", "row", "canvas", "nodeGraph", "traces", "flamegraph",
            "geomap", "candlestick", "trend", "xychart", "datagrid",
            "state-timeline", "status-history",
        }
        if v not in known_types:
            raise ValueError(f"Unknown panel type '{v}'. Known types: {sorted(known_types)}")
        return v

    @field_validator("repeatDirection")
    @classmethod
    def validate_repeat_direction(cls, v: str) -> str:
        """Validate repeat direction."""
        if v not in ("h", "v"):
            raise ValueError("repeatDirection must be 'h' (horizontal) or 'v' (vertical)")
        return v


class Dashboard(BaseModel):
    """Complete Grafana dashboard model.

    Represents a full dashboard JSON structure compatible with the Grafana
    Dashboard API (POST /api/dashboards/db).
    """

    id: Optional[int] = Field(default=None, description="Dashboard numeric ID (null for new)")
    uid: str = Field(default="", max_length=40, description="Unique string identifier")
    title: str = Field(..., min_length=1, max_length=255, description="Dashboard title")
    description: str = Field(default="", description="Dashboard description")
    tags: list[str] = Field(default_factory=list, description="Tags for search and filtering")
    timezone: str = Field(default="browser", description="Timezone: 'browser', 'utc', or IANA tz")
    editable: bool = Field(default=True, description="Whether dashboard is editable by users")
    fiscalYearStartMonth: int = Field(
        default=0, ge=0, le=11, description="Fiscal year start month (0=Jan)"
    )
    graphTooltip: int = Field(
        default=1, ge=0, le=2, description="0=no sharing, 1=shared crosshair, 2=shared tooltip"
    )
    liveNow: bool = Field(default=False, description="Whether dashboard auto-refreshes to 'now'")
    panels: list[Panel] = Field(default_factory=list, description="Panels in the dashboard")
    refresh: str = Field(default="30s", description="Auto-refresh interval, e.g. '30s', '1m', '5m'")
    schemaVersion: int = Field(default=39, ge=1, description="Dashboard schema version")
    style: str = Field(default="dark", description="Dashboard style: 'dark' or 'light'")
    templating: dict[str, list[Variable]] = Field(
        default_factory=lambda: {"list": []},
        description="Template variable definitions",
    )
    annotations: dict[str, list[Annotation]] = Field(
        default_factory=lambda: {"list": []},
        description="Annotation query definitions",
    )
    time: dict[str, str] = Field(
        default_factory=lambda: {"from": "now-6h", "to": "now"},
        description="Default time range",
    )
    timepicker: dict[str, Any] = Field(
        default_factory=dict, description="Time picker configuration"
    )
    version: int = Field(default=0, ge=0, description="Dashboard version for concurrency control")
    links: list[dict[str, Any]] = Field(
        default_factory=list, description="Dashboard-level links"
    )

    @model_validator(mode="after")
    def assign_panel_ids(self) -> Dashboard:
        """Ensure all panels have unique, non-zero IDs."""
        existing_ids = {p.id for p in self.panels if p.id > 0}
        next_id = max(existing_ids, default=0) + 1
        for panel in self.panels:
            if panel.id == 0:
                while next_id in existing_ids:
                    next_id += 1
                panel.id = next_id
                existing_ids.add(next_id)
                next_id += 1
        return self


class DataSource(BaseModel):
    """Grafana data source configuration.

    Represents a data source as returned by the Grafana Data Source API
    (GET /api/datasources) and used for provisioning.
    """

    id: Optional[int] = Field(default=None, description="Data source numeric ID")
    uid: str = Field(default="", description="Unique string identifier")
    orgId: int = Field(default=1, ge=1, description="Organization ID")
    name: str = Field(..., min_length=1, description="Data source display name")
    type: str = Field(..., description="Plugin type: prometheus, loki, jaeger, postgres, etc.")
    access: str = Field(default="proxy", description="Access mode: 'proxy' or 'direct'")
    url: str = Field(default="", description="Data source URL")
    isDefault: bool = Field(default=False, description="Whether this is the default datasource")
    editable: bool = Field(default=False, description="Whether users can edit via UI")
    basicAuth: bool = Field(default=False, description="Enable basic authentication")
    basicAuthUser: str = Field(default="", description="Basic auth username")
    withCredentials: bool = Field(default=False, description="Send credentials with requests")
    jsonData: dict[str, Any] = Field(
        default_factory=dict, description="Plugin-specific configuration"
    )
    secureJsonData: dict[str, Any] = Field(
        default_factory=dict, description="Encrypted configuration (passwords, tokens)"
    )
    version: int = Field(default=1, ge=1, description="Data source version")
    readOnly: bool = Field(default=False, description="Read-only mode")

    @field_validator("access")
    @classmethod
    def validate_access(cls, v: str) -> str:
        """Validate access mode."""
        if v not in ("proxy", "direct"):
            raise ValueError("access must be 'proxy' or 'direct'")
        return v


class Folder(BaseModel):
    """Grafana folder.

    Folders organise dashboards into a hierarchy for access control and navigation.
    """

    id: Optional[int] = Field(default=None, description="Folder numeric ID")
    uid: str = Field(default="", max_length=40, description="Unique string identifier")
    title: str = Field(..., min_length=1, max_length=255, description="Folder display name")
    url: str = Field(default="", description="Folder URL path")
    hasAcl: bool = Field(default=False, description="Whether custom permissions are set")
    canSave: bool = Field(default=True, description="Whether the current user can save to folder")
    canEdit: bool = Field(default=True, description="Whether the current user can edit the folder")
    canAdmin: bool = Field(default=False, description="Whether the current user can admin the folder")
    canDelete: bool = Field(default=False, description="Whether the current user can delete")
    createdBy: str = Field(default="", description="Creator username")
    updatedBy: str = Field(default="", description="Last updater username")
    version: int = Field(default=0, ge=0, description="Folder version for concurrency")
    parentUid: str = Field(default="", description="Parent folder UID for nested folders")


class FolderPermission(BaseModel):
    """Folder permission assignment.

    Grants a role, team, or user a specific permission level on a folder.
    """

    role: str = Field(default="", description="Grafana org role: Viewer, Editor, Admin")
    teamId: int = Field(default=0, ge=0, description="Team ID (0 if role-based)")
    userId: int = Field(default=0, ge=0, description="User ID (0 if role or team-based)")
    permission: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Permission level: 1=View, 2=Edit, 4=Admin",
    )

    @field_validator("permission")
    @classmethod
    def validate_permission(cls, v: int) -> int:
        """Validate permission level against Grafana API values."""
        if v not in (1, 2, 4):
            raise ValueError("permission must be 1 (View), 2 (Edit), or 4 (Admin)")
        return v


class AlertRule(BaseModel):
    """Grafana Unified Alerting rule.

    Represents an alert rule within the Grafana Unified Alerting system
    (POST /api/v1/provisioning/alert-rules).
    """

    uid: str = Field(default="", description="Alert rule UID")
    orgID: int = Field(default=1, ge=1, description="Organization ID")
    folderUID: str = Field(default="", description="Folder UID containing this rule")
    ruleGroup: str = Field(default="", description="Alert rule group name")
    title: str = Field(..., min_length=1, description="Alert rule title")
    condition: str = Field(default="C", description="Condition refId that triggers the alert")
    data: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alert query and condition definitions (GrafanaAlertQuery[])",
    )
    noDataState: str = Field(
        default="NoData",
        description="State when no data: NoData, Alerting, OK",
    )
    execErrState: str = Field(
        default="Error",
        description="State on execution error: Error, Alerting, OK",
    )
    for_: str = Field(
        default="5m",
        alias="for",
        description="Duration condition must be true before firing, e.g. '5m'",
    )
    labels: dict[str, str] = Field(
        default_factory=dict, description="Labels attached to the alert, e.g. severity"
    )
    annotations: dict[str, str] = Field(
        default_factory=dict, description="Annotations: summary, description, runbook_url"
    )
    isPaused: bool = Field(default=False, description="Whether the alert rule is paused")

    model_config = {"populate_by_name": True}

    @field_validator("noDataState")
    @classmethod
    def validate_no_data_state(cls, v: str) -> str:
        """Validate noDataState against allowed values."""
        allowed = {"NoData", "Alerting", "OK"}
        if v not in allowed:
            raise ValueError(f"noDataState must be one of {allowed}")
        return v

    @field_validator("execErrState")
    @classmethod
    def validate_exec_err_state(cls, v: str) -> str:
        """Validate execErrState against allowed values."""
        allowed = {"Error", "Alerting", "OK"}
        if v not in allowed:
            raise ValueError(f"execErrState must be one of {allowed}")
        return v


class DashboardSearchResult(BaseModel):
    """Dashboard search result from Grafana Search API.

    Returned by GET /api/search when querying for dashboards and folders.
    """

    id: int = Field(default=0, description="Result numeric ID")
    uid: str = Field(default="", description="Unique string identifier")
    title: str = Field(default="", description="Dashboard or folder title")
    uri: str = Field(default="", description="URI path, e.g. 'db/my-dashboard'")
    url: str = Field(default="", description="Full URL path")
    slug: str = Field(default="", description="URL-safe slug")
    type: str = Field(default="dash-db", description="Result type: dash-db, dash-folder")
    tags: list[str] = Field(default_factory=list, description="Dashboard tags")
    isStarred: bool = Field(default=False, description="Whether user starred this item")
    folderId: int = Field(default=0, ge=0, description="Parent folder ID")
    folderUid: str = Field(default="", description="Parent folder UID")
    folderTitle: str = Field(default="", description="Parent folder title")
    folderUrl: str = Field(default="", description="Parent folder URL")
    sortMeta: int = Field(default=0, description="Sort metadata value")


class HealthStatus(BaseModel):
    """Grafana health check response.

    Returned by GET /api/health to verify Grafana server status.
    """

    commit: str = Field(default="", description="Git commit hash of the Grafana build")
    database: str = Field(default="ok", description="Database connection status")
    version: str = Field(default="", description="Grafana version string")


class ContactPoint(BaseModel):
    """Grafana alerting contact point.

    Defines a notification receiver for alerts (Slack, PagerDuty, email, etc.).
    Used with POST /api/v1/provisioning/contact-points.
    """

    uid: str = Field(default="", description="Contact point UID")
    name: str = Field(..., min_length=1, description="Contact point name")
    type: str = Field(
        ...,
        description="Receiver type: slack, pagerduty, email, webhook, teams, opsgenie, etc.",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific settings (URL, integration key, addresses, etc.)",
    )
    disableResolveMessage: bool = Field(
        default=False, description="Disable automatic resolve notifications"
    )
    provenance: str = Field(default="", description="Provenance: 'api', 'file', or ''")

    @field_validator("type")
    @classmethod
    def validate_contact_type(cls, v: str) -> str:
        """Validate contact point type against known receiver types."""
        known = {
            "slack", "pagerduty", "email", "webhook", "teams",
            "opsgenie", "victorops", "pushover", "telegram",
            "discord", "googlechat", "sensugo", "threema",
            "kafka", "line", "dingding",
        }
        if v not in known:
            raise ValueError(f"Unknown contact point type '{v}'. Known types: {sorted(known)}")
        return v


class NotificationPolicy(BaseModel):
    """Grafana notification policy tree node.

    Defines routing rules for alerts to contact points, including grouping,
    timing, and nested child routes.
    """

    receiver: str = Field(default="", description="Contact point name to send to")
    group_by: list[str] = Field(
        default_factory=lambda: ["grafana_folder", "alertname"],
        description="Labels to group alerts by",
    )
    group_wait: str = Field(default="30s", description="Wait before sending first notification")
    group_interval: str = Field(default="5m", description="Interval between group notifications")
    repeat_interval: str = Field(default="4h", description="Repeat interval for firing alerts")
    object_matchers: list[list[str]] = Field(
        default_factory=list,
        description="Label matchers: [['label', 'op', 'value'], ...] where op is =, !=, =~, !~",
    )
    mute_time_intervals: list[str] = Field(
        default_factory=list, description="Mute timing names to apply"
    )
    routes: list[NotificationPolicy] = Field(
        default_factory=list, description="Child notification routes"
    )
    continue_: bool = Field(
        default=False,
        alias="continue",
        description="Continue matching after this route matches",
    )
    provenance: str = Field(default="", description="Provenance: 'api', 'file', or ''")

    model_config = {"populate_by_name": True}


# Enable forward references for self-referencing models.
Panel.model_rebuild()
NotificationPolicy.model_rebuild()
