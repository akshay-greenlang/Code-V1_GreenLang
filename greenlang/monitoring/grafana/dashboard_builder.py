# -*- coding: utf-8 -*-
"""
Grafana Dashboard Builder
==========================

Fluent builder for constructing Grafana dashboards programmatically.
Handles auto-layout on the 24-column grid, auto-incrementing panel IDs,
and serialisation to Grafana-compatible JSON.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.monitoring.grafana.models import (
    Annotation,
    Dashboard,
    GridPos,
    Panel,
    Variable,
)

logger = logging.getLogger(__name__)

# Grafana uses a 24-column grid
_GRID_COLUMNS = 24


class DashboardBuilder:
    """Fluent builder for Grafana dashboards.

    Provides a chainable API for constructing dashboards with automatic
    panel ID assignment and grid layout management.

    Usage::

        dashboard = (
            DashboardBuilder()
            .with_title("Platform Overview")
            .with_uid("gl-platform-overview")
            .with_tags(["greenlang", "overview"])
            .with_time_range("now-6h", "now")
            .with_refresh("30s")
            .add_variable(Variable(name="namespace", query="label_values(namespace)"))
            .add_row("Request Metrics")
            .add_panel(some_panel)
            .build()
        )
    """

    def __init__(self) -> None:
        self._title: str = "New Dashboard"
        self._uid: str = ""
        self._description: str = ""
        self._tags: list[str] = []
        self._time_from: str = "now-6h"
        self._time_to: str = "now"
        self._refresh: str = "30s"
        self._timezone: str = "browser"
        self._editable: bool = True
        self._schema_version: int = 39
        self._graph_tooltip: int = 1
        self._style: str = "dark"
        self._version: int = 0
        self._links: list[dict[str, Any]] = []
        self._variables: list[Variable] = []
        self._annotations: list[Annotation] = []
        self._panels: list[Panel] = []
        self._next_panel_id: int = 1
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._current_row_height: int = 0

    # -- Title / metadata ----------------------------------------------------

    def with_title(self, title: str) -> DashboardBuilder:
        """Set the dashboard title.

        Args:
            title: Dashboard title (1-255 characters).

        Returns:
            Self for chaining.
        """
        self._title = title
        return self

    def with_uid(self, uid: str) -> DashboardBuilder:
        """Set the dashboard UID.

        Args:
            uid: Unique identifier (max 40 characters).

        Returns:
            Self for chaining.
        """
        self._uid = uid
        return self

    def with_description(self, description: str) -> DashboardBuilder:
        """Set the dashboard description.

        Args:
            description: Dashboard description text.

        Returns:
            Self for chaining.
        """
        self._description = description
        return self

    def with_tags(self, tags: list[str]) -> DashboardBuilder:
        """Set dashboard tags for search and filtering.

        Args:
            tags: List of tag strings.

        Returns:
            Self for chaining.
        """
        self._tags = tags
        return self

    # -- Time / display ------------------------------------------------------

    def with_time_range(self, from_: str, to: str) -> DashboardBuilder:
        """Set the default time range.

        Args:
            from_: Start time expression (e.g. 'now-6h', 'now-1d').
            to: End time expression (e.g. 'now').

        Returns:
            Self for chaining.
        """
        self._time_from = from_
        self._time_to = to
        return self

    def with_refresh(self, interval: str) -> DashboardBuilder:
        """Set the auto-refresh interval.

        Args:
            interval: Refresh interval (e.g. '10s', '30s', '1m', '5m').

        Returns:
            Self for chaining.
        """
        self._refresh = interval
        return self

    def with_timezone(self, timezone: str) -> DashboardBuilder:
        """Set the dashboard timezone.

        Args:
            timezone: Timezone string ('browser', 'utc', or IANA timezone).

        Returns:
            Self for chaining.
        """
        self._timezone = timezone
        return self

    def with_editable(self, editable: bool) -> DashboardBuilder:
        """Set whether the dashboard is editable.

        Args:
            editable: True to allow editing, False for read-only.

        Returns:
            Self for chaining.
        """
        self._editable = editable
        return self

    def with_schema_version(self, version: int) -> DashboardBuilder:
        """Set the dashboard schema version.

        Args:
            version: Grafana schema version (default 39).

        Returns:
            Self for chaining.
        """
        self._schema_version = version
        return self

    def with_graph_tooltip(self, mode: int) -> DashboardBuilder:
        """Set the graph tooltip sharing mode.

        Args:
            mode: 0 = no sharing, 1 = shared crosshair, 2 = shared tooltip.

        Returns:
            Self for chaining.
        """
        self._graph_tooltip = mode
        return self

    def with_version(self, version: int) -> DashboardBuilder:
        """Set the dashboard version for concurrency control.

        Args:
            version: Dashboard version number.

        Returns:
            Self for chaining.
        """
        self._version = version
        return self

    # -- Links ---------------------------------------------------------------

    def add_link(
        self,
        title: str,
        url: str = "",
        link_type: str = "link",
        tags: Optional[list[str]] = None,
        target_blank: bool = True,
        icon: str = "external link",
    ) -> DashboardBuilder:
        """Add a dashboard link.

        Args:
            title: Link display text.
            url: URL target (for 'link' type).
            link_type: 'link' for URL or 'dashboards' for tag-based links.
            tags: Tags filter (for 'dashboards' type).
            target_blank: Open in new tab.
            icon: Icon name.

        Returns:
            Self for chaining.
        """
        link: dict[str, Any] = {
            "title": title,
            "type": link_type,
            "targetBlank": target_blank,
            "icon": icon,
        }
        if url:
            link["url"] = url
        if tags:
            link["tags"] = tags
        self._links.append(link)
        return self

    # -- Variables and annotations -------------------------------------------

    def add_variable(self, variable: Variable) -> DashboardBuilder:
        """Add a template variable.

        Args:
            variable: Variable model to add.

        Returns:
            Self for chaining.
        """
        self._variables.append(variable)
        return self

    def add_annotation(self, annotation: Annotation) -> DashboardBuilder:
        """Add an annotation query layer.

        Args:
            annotation: Annotation model to add.

        Returns:
            Self for chaining.
        """
        self._annotations.append(annotation)
        return self

    # -- Panels and rows -----------------------------------------------------

    def add_row(self, title: str, collapsed: bool = False) -> DashboardBuilder:
        """Add a row separator panel.

        Rows act as visual and logical separators. When collapsed, nested
        panels are hidden until the row is expanded.

        Args:
            title: Row title text.
            collapsed: Whether the row starts collapsed.

        Returns:
            Self for chaining.
        """
        # Move cursor to start of next row
        if self._cursor_x > 0:
            self._cursor_y += self._current_row_height
            self._cursor_x = 0
            self._current_row_height = 0

        row_panel = Panel(
            id=self._next_panel_id,
            title=title,
            type="row",
            gridPos=GridPos(h=1, w=_GRID_COLUMNS, x=0, y=self._cursor_y),
            collapsed=collapsed,
        )
        self._next_panel_id += 1
        self._panels.append(row_panel)

        # Advance cursor past the row
        self._cursor_y += 1
        self._current_row_height = 0
        return self

    def add_panel(self, panel: Panel) -> DashboardBuilder:
        """Add a panel with automatic grid positioning.

        The builder maintains a cursor that flows left-to-right, top-to-bottom
        within the 24-column grid. Panels that do not fit on the current row
        wrap to the next row automatically.

        Args:
            panel: Panel model to add. Its gridPos will be updated with
                   auto-calculated x/y values. The panel ID will be assigned
                   automatically.

        Returns:
            Self for chaining.
        """
        w = panel.gridPos.w
        h = panel.gridPos.h

        # Wrap to next row if panel does not fit
        if self._cursor_x + w > _GRID_COLUMNS:
            self._cursor_y += self._current_row_height
            self._cursor_x = 0
            self._current_row_height = 0

        # Assign position and ID
        panel.gridPos = GridPos(h=h, w=w, x=self._cursor_x, y=self._cursor_y)
        panel.id = self._next_panel_id
        self._next_panel_id += 1

        self._panels.append(panel)

        # Advance cursor
        self._cursor_x += w
        self._current_row_height = max(self._current_row_height, h)

        return self

    def add_panel_full_width(self, panel: Panel) -> DashboardBuilder:
        """Add a panel spanning the full 24-column width.

        Forces a new row if the cursor is not at position 0, then places
        the panel at full width.

        Args:
            panel: Panel model (width will be set to 24).

        Returns:
            Self for chaining.
        """
        panel.gridPos = GridPos(h=panel.gridPos.h, w=_GRID_COLUMNS, x=0, y=0)
        return self.add_panel(panel)

    # -- Build ---------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build the dashboard as a Grafana-compatible JSON dict.

        Creates a validated Dashboard model and serialises it to a dict
        suitable for the Grafana Dashboard API (POST /api/dashboards/db).

        Returns:
            Dashboard JSON dict ready for API submission.
        """
        dashboard = Dashboard(
            uid=self._uid,
            title=self._title,
            description=self._description,
            tags=self._tags,
            timezone=self._timezone,
            editable=self._editable,
            graphTooltip=self._graph_tooltip,
            schemaVersion=self._schema_version,
            style=self._style,
            version=self._version,
            panels=self._panels,
            refresh=self._refresh,
            time={"from": self._time_from, "to": self._time_to},
            templating={"list": self._variables},
            annotations={"list": self._annotations},
            links=self._links,
        )

        result = dashboard.model_dump(exclude_none=True)
        logger.info(
            "Dashboard built: title=%s uid=%s panels=%d variables=%d",
            self._title,
            self._uid,
            len(self._panels),
            len(self._variables),
        )
        return result

    def build_model(self) -> Dashboard:
        """Build the dashboard as a validated Dashboard model.

        Returns:
            Dashboard Pydantic model instance.
        """
        return Dashboard(
            uid=self._uid,
            title=self._title,
            description=self._description,
            tags=self._tags,
            timezone=self._timezone,
            editable=self._editable,
            graphTooltip=self._graph_tooltip,
            schemaVersion=self._schema_version,
            style=self._style,
            version=self._version,
            panels=self._panels,
            refresh=self._refresh,
            time={"from": self._time_from, "to": self._time_to},
            templating={"list": self._variables},
            annotations={"list": self._annotations},
            links=self._links,
        )
