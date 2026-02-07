# -*- coding: utf-8 -*-
"""
Grafana Panel Builder
======================

Fluent builder for constructing Grafana panels with factory methods for
common panel types and chainable configuration.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from greenlang.monitoring.grafana.models import GridPos, Panel, Target

logger = logging.getLogger(__name__)


class PanelBuilder:
    """Fluent builder for Grafana dashboard panels.

    Provides factory class methods for common panel types and chainable
    configuration methods for targets, thresholds, field config, and display.

    Usage::

        panel = (
            PanelBuilder.timeseries("Request Rate")
            .with_target("thanos", 'sum(rate(http_requests_total[5m]))')
            .with_unit("reqps")
            .with_grid_pos(w=12, h=8)
            .with_legend(display_mode="table", placement="bottom")
            .build()
        )
    """

    def __init__(self, title: str, panel_type: str) -> None:
        self._title = title
        self._type = panel_type
        self._description: str = ""
        self._datasource: Optional[dict[str, str]] = None
        self._targets: list[Target] = []
        self._grid_pos = GridPos()
        self._field_config: dict[str, Any] = {"defaults": {}, "overrides": []}
        self._options: dict[str, Any] = {}
        self._transformations: list[dict[str, Any]] = []
        self._transparent: bool = False
        self._repeat: Optional[str] = None
        self._repeat_direction: str = "h"
        self._max_per_row: int = 4
        self._links: list[dict[str, Any]] = []
        self._next_ref_id_ord: int = ord("A")

    # -- Factory methods for common panel types ------------------------------

    @classmethod
    def stat(cls, title: str) -> PanelBuilder:
        """Create a Stat panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for stat type.
        """
        builder = cls(title, "stat")
        builder._options = {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"], "fields": ""},
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto",
            "textMode": "auto",
        }
        builder._grid_pos = GridPos(w=6, h=4)
        return builder

    @classmethod
    def gauge(cls, title: str) -> PanelBuilder:
        """Create a Gauge panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for gauge type.
        """
        builder = cls(title, "gauge")
        builder._options = {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"], "fields": ""},
            "showThresholdLabels": False,
            "showThresholdMarkers": True,
        }
        builder._grid_pos = GridPos(w=6, h=6)
        return builder

    @classmethod
    def timeseries(cls, title: str) -> PanelBuilder:
        """Create a Time Series panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for timeseries type.
        """
        builder = cls(title, "timeseries")
        builder._options = {
            "tooltip": {"mode": "single", "sort": "none"},
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        }
        builder._field_config = {
            "defaults": {
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "linear",
                    "barAlignment": 0,
                    "lineWidth": 1,
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "spanNulls": False,
                    "showPoints": "auto",
                    "pointSize": 5,
                    "stacking": {"mode": "none", "group": "A"},
                    "axisPlacement": "auto",
                    "scaleDistribution": {"type": "linear"},
                },
            },
            "overrides": [],
        }
        builder._grid_pos = GridPos(w=12, h=8)
        return builder

    @classmethod
    def table(cls, title: str) -> PanelBuilder:
        """Create a Table panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for table type.
        """
        builder = cls(title, "table")
        builder._options = {
            "showHeader": True,
            "sortBy": [],
            "cellHeight": "sm",
            "footer": {"show": False, "reducer": ["sum"], "countRows": False},
        }
        builder._grid_pos = GridPos(w=24, h=8)
        return builder

    @classmethod
    def barchart(cls, title: str) -> PanelBuilder:
        """Create a Bar Chart panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for barchart type.
        """
        builder = cls(title, "barchart")
        builder._options = {
            "orientation": "auto",
            "xTickLabelRotation": 0,
            "xTickLabelSpacing": 0,
            "showValue": "auto",
            "stacking": "none",
            "groupWidth": 0.7,
            "barWidth": 0.97,
            "barRadius": 0,
            "legend": {"displayMode": "list", "placement": "bottom"},
            "tooltip": {"mode": "single", "sort": "none"},
        }
        builder._grid_pos = GridPos(w=12, h=8)
        return builder

    @classmethod
    def piechart(cls, title: str) -> PanelBuilder:
        """Create a Pie Chart panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for piechart type.
        """
        builder = cls(title, "piechart")
        builder._options = {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"], "fields": ""},
            "pieType": "pie",
            "tooltip": {"mode": "single", "sort": "none"},
            "legend": {"displayMode": "list", "placement": "right", "showLegend": True},
        }
        builder._grid_pos = GridPos(w=8, h=8)
        return builder

    @classmethod
    def heatmap(cls, title: str) -> PanelBuilder:
        """Create a Heatmap panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for heatmap type.
        """
        builder = cls(title, "heatmap")
        builder._options = {
            "calculate": True,
            "cellGap": 1,
            "color": {"mode": "scheme", "scheme": "Oranges", "steps": 64},
            "yAxis": {"axisPlacement": "left"},
            "tooltip": {"show": True, "yHistogram": False},
            "legend": {"show": True},
        }
        builder._grid_pos = GridPos(w=12, h=8)
        return builder

    @classmethod
    def logs(cls, title: str) -> PanelBuilder:
        """Create a Logs panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for logs type.
        """
        builder = cls(title, "logs")
        builder._options = {
            "showTime": True,
            "showLabels": False,
            "showCommonLabels": False,
            "wrapLogMessage": True,
            "prettifyLogMessage": False,
            "enableLogDetails": True,
            "dedupStrategy": "none",
            "sortOrder": "Descending",
        }
        builder._datasource = {"type": "loki", "uid": "loki"}
        builder._grid_pos = GridPos(w=24, h=10)
        return builder

    @classmethod
    def text(cls, title: str, content: str = "", mode: str = "markdown") -> PanelBuilder:
        """Create a Text panel builder.

        Args:
            title: Panel title.
            content: Markdown or HTML content.
            mode: Content mode ('markdown' or 'html').

        Returns:
            PanelBuilder configured for text type.
        """
        builder = cls(title, "text")
        builder._options = {"mode": mode, "content": content}
        builder._grid_pos = GridPos(w=24, h=4)
        return builder

    @classmethod
    def bargauge(cls, title: str) -> PanelBuilder:
        """Create a Bar Gauge panel builder.

        Args:
            title: Panel title.

        Returns:
            PanelBuilder configured for bargauge type.
        """
        builder = cls(title, "bargauge")
        builder._options = {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"], "fields": ""},
            "orientation": "horizontal",
            "displayMode": "gradient",
            "showUnfilled": True,
            "minVizWidth": 8,
            "minVizHeight": 16,
        }
        builder._grid_pos = GridPos(w=8, h=6)
        return builder

    # -- Target / query methods ----------------------------------------------

    def with_target(
        self,
        datasource_uid: str,
        expr: str,
        legend: str = "__auto",
        instant: bool = False,
        datasource_type: str = "prometheus",
        interval: str = "",
        format_: str = "time_series",
    ) -> PanelBuilder:
        """Add a Prometheus/Thanos query target.

        Args:
            datasource_uid: Data source UID (e.g. 'thanos', 'prometheus').
            expr: PromQL expression.
            legend: Legend format template.
            instant: True for instant query.
            datasource_type: Data source type.
            interval: Minimum step interval.
            format_: Result format.

        Returns:
            Self for chaining.
        """
        ref_id = chr(self._next_ref_id_ord)
        self._next_ref_id_ord += 1

        target = Target(
            datasource={"type": datasource_type, "uid": datasource_uid},
            expr=expr,
            legendFormat=legend,
            refId=ref_id,
            instant=instant,
            interval=interval,
            format=format_,
        )
        self._targets.append(target)

        # Set panel datasource to match first target
        if self._datasource is None:
            self._datasource = {"type": datasource_type, "uid": datasource_uid}

        return self

    def with_loki_target(
        self,
        expr: str,
        legend: str = "",
        datasource_uid: str = "loki",
    ) -> PanelBuilder:
        """Add a Loki LogQL query target.

        Args:
            expr: LogQL expression.
            legend: Legend format template.
            datasource_uid: Loki data source UID.

        Returns:
            Self for chaining.
        """
        return self.with_target(
            datasource_uid=datasource_uid,
            expr=expr,
            legend=legend,
            datasource_type="loki",
        )

    # -- Grid position -------------------------------------------------------

    def with_grid_pos(
        self,
        w: int = 12,
        h: int = 8,
        x: int = 0,
        y: int = 0,
    ) -> PanelBuilder:
        """Set the panel grid position and size.

        Args:
            w: Width in grid columns (1-24).
            h: Height in grid units (1-40).
            x: Horizontal position (0-23).
            y: Vertical position.

        Returns:
            Self for chaining.
        """
        self._grid_pos = GridPos(w=w, h=h, x=x, y=y)
        return self

    # -- Display options -----------------------------------------------------

    def with_description(self, description: str) -> PanelBuilder:
        """Set the panel description / tooltip.

        Args:
            description: Description text.

        Returns:
            Self for chaining.
        """
        self._description = description
        return self

    def with_transparent(self, transparent: bool = True) -> PanelBuilder:
        """Set the panel background transparency.

        Args:
            transparent: True for transparent background.

        Returns:
            Self for chaining.
        """
        self._transparent = transparent
        return self

    def with_repeat(
        self,
        variable: str,
        direction: str = "h",
        max_per_row: int = 4,
    ) -> PanelBuilder:
        """Configure panel repetition by variable.

        Args:
            variable: Template variable name to repeat by.
            direction: 'h' for horizontal or 'v' for vertical.
            max_per_row: Maximum panels per row.

        Returns:
            Self for chaining.
        """
        self._repeat = variable
        self._repeat_direction = direction
        self._max_per_row = max_per_row
        return self

    # -- Field configuration -------------------------------------------------

    def with_unit(self, unit: str) -> PanelBuilder:
        """Set the display unit for values.

        Args:
            unit: Grafana unit string (e.g. 'reqps', 'bytes', 'percent',
                  'short', 'dtdurations', 's', 'ms').

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["unit"] = unit
        return self

    def with_decimals(self, decimals: int) -> PanelBuilder:
        """Set the number of decimal places.

        Args:
            decimals: Number of decimal places (0-10).

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["decimals"] = decimals
        return self

    def with_min(self, value: float) -> PanelBuilder:
        """Set the minimum display value.

        Args:
            value: Minimum value.

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["min"] = value
        return self

    def with_max(self, value: float) -> PanelBuilder:
        """Set the maximum display value.

        Args:
            value: Maximum value.

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["max"] = value
        return self

    def with_threshold(
        self,
        value: float,
        color: str = "red",
    ) -> PanelBuilder:
        """Add a threshold step.

        Thresholds define colour boundaries for stat, gauge, and table panels.

        Args:
            value: Threshold value.
            color: Colour name or hex code.

        Returns:
            Self for chaining.
        """
        defaults = self._field_config["defaults"]
        if "thresholds" not in defaults:
            defaults["thresholds"] = {
                "mode": "absolute",
                "steps": [{"color": "green", "value": None}],
            }
        defaults["thresholds"]["steps"].append({"color": color, "value": value})
        return self

    def with_color_mode(self, mode: str) -> PanelBuilder:
        """Set the colour mode.

        Args:
            mode: Colour mode ('fixed', 'thresholds', 'palette-classic',
                  'continuous-GrYlRd', 'continuous-BlYlRd').

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["color"] = {"mode": mode}
        return self

    def with_color_scheme(self, scheme: str) -> PanelBuilder:
        """Set the colour scheme for the panel.

        Args:
            scheme: Scheme name (e.g. 'palette-classic', 'Greens', 'Blues').

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["color"] = {"mode": "palette-classic-by-name"}
        if "color" in self._field_config["defaults"]:
            self._field_config["defaults"]["color"]["fixedColor"] = scheme
        return self

    def with_no_value(self, text: str) -> PanelBuilder:
        """Set the text displayed when there is no data.

        Args:
            text: No-data display text (e.g. 'N/A', '0', '-').

        Returns:
            Self for chaining.
        """
        self._field_config["defaults"]["noValue"] = text
        return self

    def with_override(
        self,
        matcher: dict[str, Any],
        properties: list[dict[str, Any]],
    ) -> PanelBuilder:
        """Add a field override.

        Args:
            matcher: Override matcher (e.g. {'id': 'byName', 'options': 'errors'}).
            properties: List of property overrides
                (e.g. [{'id': 'color', 'value': {'mode': 'fixed', 'fixedColor': 'red'}}]).

        Returns:
            Self for chaining.
        """
        self._field_config["overrides"].append({
            "matcher": matcher,
            "properties": properties,
        })
        return self

    def with_mapping(
        self,
        value: Any,
        display_text: str,
        color: str = "",
    ) -> PanelBuilder:
        """Add a value mapping.

        Args:
            value: Value to map.
            display_text: Display text for the mapped value.
            color: Optional colour for the mapping.

        Returns:
            Self for chaining.
        """
        defaults = self._field_config["defaults"]
        if "mappings" not in defaults:
            defaults["mappings"] = []

        mapping: dict[str, Any] = {
            "type": "value",
            "options": {
                str(value): {"text": display_text, "index": len(defaults["mappings"])},
            },
        }
        if color:
            mapping["options"][str(value)]["color"] = color
        defaults["mappings"].append(mapping)
        return self

    # -- Panel options -------------------------------------------------------

    def with_legend(
        self,
        display_mode: str = "list",
        placement: str = "bottom",
        show: bool = True,
    ) -> PanelBuilder:
        """Configure the legend display.

        Args:
            display_mode: 'list', 'table', or 'hidden'.
            placement: 'bottom' or 'right'.
            show: Whether to show the legend.

        Returns:
            Self for chaining.
        """
        self._options["legend"] = {
            "displayMode": display_mode,
            "placement": placement,
            "showLegend": show,
        }
        return self

    def with_tooltip(self, mode: str = "single", sort: str = "none") -> PanelBuilder:
        """Configure the tooltip behaviour.

        Args:
            mode: 'single', 'multi', or 'none'.
            sort: 'none', 'asc', or 'desc'.

        Returns:
            Self for chaining.
        """
        self._options["tooltip"] = {"mode": mode, "sort": sort}
        return self

    def with_axis(
        self,
        label: str = "",
        placement: str = "auto",
        soft_min: Optional[float] = None,
        soft_max: Optional[float] = None,
    ) -> PanelBuilder:
        """Configure the Y-axis.

        Args:
            label: Axis label text.
            placement: 'auto', 'left', 'right', or 'hidden'.
            soft_min: Soft minimum value.
            soft_max: Soft maximum value.

        Returns:
            Self for chaining.
        """
        custom = self._field_config["defaults"].setdefault("custom", {})
        custom["axisPlacement"] = placement
        if label:
            custom["axisLabel"] = label
        if soft_min is not None:
            custom["axisSoftMin"] = soft_min
        if soft_max is not None:
            custom["axisSoftMax"] = soft_max
        return self

    # -- Links and transformations -------------------------------------------

    def add_link(
        self,
        title: str,
        url: str,
        target_blank: bool = True,
    ) -> PanelBuilder:
        """Add a data link to the panel.

        Args:
            title: Link display text.
            url: Target URL (supports Grafana variables).
            target_blank: Open in new tab.

        Returns:
            Self for chaining.
        """
        self._links.append({
            "title": title,
            "url": url,
            "targetBlank": target_blank,
        })
        return self

    def add_transformation(
        self,
        transform_id: str,
        options: Optional[dict[str, Any]] = None,
    ) -> PanelBuilder:
        """Add a data transformation.

        Args:
            transform_id: Transformation ID (e.g. 'organize', 'reduce',
                          'merge', 'filterByValue', 'groupBy').
            options: Transformation options.

        Returns:
            Self for chaining.
        """
        self._transformations.append({
            "id": transform_id,
            "options": options or {},
        })
        return self

    # -- Build ---------------------------------------------------------------

    def build(self) -> Panel:
        """Build the panel as a validated Panel model.

        Returns:
            Panel Pydantic model instance.
        """
        panel = Panel(
            title=self._title,
            type=self._type,
            description=self._description,
            datasource=self._datasource,
            targets=self._targets,
            gridPos=self._grid_pos,
            fieldConfig=self._field_config,
            options=self._options,
            transformations=self._transformations,
            transparent=self._transparent,
            repeat=self._repeat,
            repeatDirection=self._repeat_direction,
            maxPerRow=self._max_per_row,
            links=self._links,
        )
        logger.debug(
            "Panel built: title=%s type=%s targets=%d",
            self._title,
            self._type,
            len(self._targets),
        )
        return panel

    def build_dict(self) -> dict[str, Any]:
        """Build the panel as a JSON-serializable dict.

        Returns:
            Panel dict ready for dashboard embedding.
        """
        return self.build().model_dump(exclude_none=True)
