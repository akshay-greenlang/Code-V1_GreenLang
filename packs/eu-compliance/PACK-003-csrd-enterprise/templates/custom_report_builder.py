"""
CustomReportBuilderTemplate - User-defined report composition for CSRD Enterprise Pack.

This module implements a flexible report builder with a widget library (30+
widget types), layout management, validation, and multi-format rendering.
Users can compose custom reports from available widgets and save/load layouts.

Example:
    >>> template = CustomReportBuilderTemplate()
    >>> layout = ReportLayout(name="My Report", widgets=[...])
    >>> html = template.render_html({"layout": layout.dict(), "data": {...}})
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Widget and Layout Models (Pydantic-compatible plain dicts)
# ------------------------------------------------------------------

class WidgetType(str, Enum):
    """Supported widget types for custom reports."""
    KPI_CARD = "kpi_card"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    TABLE = "table"
    TEXT_BLOCK = "text_block"
    IMAGE = "image"
    SEPARATOR = "separator"
    MAP = "map"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    WATERFALL = "waterfall"
    SPARKLINE = "sparkline"
    STACKED_BAR = "stacked_bar"
    AREA_CHART = "area_chart"
    SCATTER_PLOT = "scatter_plot"
    DONUT_CHART = "donut_chart"
    RADAR_CHART = "radar_chart"
    BUBBLE_CHART = "bubble_chart"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    PROGRESS_BAR = "progress_bar"
    METRIC_COMPARISON = "metric_comparison"
    TIMELINE = "timeline"
    CALLOUT_BOX = "callout_box"
    ICON_STAT = "icon_stat"
    RANKING_LIST = "ranking_list"
    COMPARISON_TABLE = "comparison_table"
    MARKDOWN_BLOCK = "markdown_block"
    HEADER_BLOCK = "header_block"


WIDGET_CATALOG: List[Dict[str, Any]] = [
    {"type": "kpi_card", "label": "KPI Card", "description": "Single metric display with trend indicator", "min_width": 1, "min_height": 1},
    {"type": "bar_chart", "label": "Bar Chart", "description": "Vertical or horizontal bar chart", "min_width": 2, "min_height": 2},
    {"type": "line_chart", "label": "Line Chart", "description": "Time series line chart with multiple series", "min_width": 2, "min_height": 2},
    {"type": "pie_chart", "label": "Pie Chart", "description": "Proportional pie/ring chart", "min_width": 2, "min_height": 2},
    {"type": "table", "label": "Data Table", "description": "Sortable data table with pagination", "min_width": 2, "min_height": 2},
    {"type": "text_block", "label": "Text Block", "description": "Rich text content block", "min_width": 1, "min_height": 1},
    {"type": "image", "label": "Image", "description": "Embedded image or logo", "min_width": 1, "min_height": 1},
    {"type": "separator", "label": "Separator", "description": "Horizontal line separator", "min_width": 4, "min_height": 1},
    {"type": "map", "label": "Map", "description": "Geographic map with data points", "min_width": 2, "min_height": 2},
    {"type": "gauge", "label": "Gauge", "description": "Circular gauge/meter display", "min_width": 1, "min_height": 1},
    {"type": "heatmap", "label": "Heatmap", "description": "Color-coded matrix heatmap", "min_width": 2, "min_height": 2},
    {"type": "treemap", "label": "Treemap", "description": "Hierarchical area-proportional visualization", "min_width": 2, "min_height": 2},
    {"type": "sankey", "label": "Sankey Diagram", "description": "Flow diagram showing transfers", "min_width": 3, "min_height": 2},
    {"type": "waterfall", "label": "Waterfall Chart", "description": "Cumulative effect waterfall", "min_width": 2, "min_height": 2},
    {"type": "sparkline", "label": "Sparkline", "description": "Inline mini trend chart", "min_width": 1, "min_height": 1},
    {"type": "stacked_bar", "label": "Stacked Bar", "description": "Stacked bar chart for composition", "min_width": 2, "min_height": 2},
    {"type": "area_chart", "label": "Area Chart", "description": "Filled area time series", "min_width": 2, "min_height": 2},
    {"type": "scatter_plot", "label": "Scatter Plot", "description": "X-Y scatter plot with tooltips", "min_width": 2, "min_height": 2},
    {"type": "donut_chart", "label": "Donut Chart", "description": "Ring chart with center metric", "min_width": 2, "min_height": 2},
    {"type": "radar_chart", "label": "Radar Chart", "description": "Multi-axis radar/spider chart", "min_width": 2, "min_height": 2},
    {"type": "bubble_chart", "label": "Bubble Chart", "description": "Scatter with size dimension", "min_width": 2, "min_height": 2},
    {"type": "histogram", "label": "Histogram", "description": "Frequency distribution chart", "min_width": 2, "min_height": 2},
    {"type": "box_plot", "label": "Box Plot", "description": "Statistical distribution box plot", "min_width": 2, "min_height": 2},
    {"type": "progress_bar", "label": "Progress Bar", "description": "Horizontal progress indicator", "min_width": 2, "min_height": 1},
    {"type": "metric_comparison", "label": "Metric Comparison", "description": "Side-by-side metric comparison", "min_width": 2, "min_height": 1},
    {"type": "timeline", "label": "Timeline", "description": "Chronological event timeline", "min_width": 3, "min_height": 2},
    {"type": "callout_box", "label": "Callout Box", "description": "Highlighted callout/alert box", "min_width": 2, "min_height": 1},
    {"type": "icon_stat", "label": "Icon Stat", "description": "Icon with stat value and label", "min_width": 1, "min_height": 1},
    {"type": "ranking_list", "label": "Ranking List", "description": "Ordered ranking with bars", "min_width": 2, "min_height": 2},
    {"type": "comparison_table", "label": "Comparison Table", "description": "Side-by-side comparison grid", "min_width": 3, "min_height": 2},
    {"type": "markdown_block", "label": "Markdown Block", "description": "Rendered markdown content", "min_width": 2, "min_height": 1},
    {"type": "header_block", "label": "Header Block", "description": "Section header with subtitle", "min_width": 4, "min_height": 1},
]


def _make_widget(
    widget_type: str,
    title: str,
    data_source: str = "",
    config: Optional[Dict[str, Any]] = None,
    row: int = 0,
    col: int = 0,
    width: int = 2,
    height: int = 2,
) -> Dict[str, Any]:
    """Create a widget dict.

    Args:
        widget_type: One of WidgetType values.
        title: Widget display title.
        data_source: Data source identifier.
        config: Widget-specific configuration.
        row: Grid row position (0-based).
        col: Grid column position (0-based).
        width: Widget width in grid units.
        height: Widget height in grid units.

    Returns:
        Widget dict.
    """
    return {
        "widget_id": str(uuid.uuid4())[:8],
        "widget_type": widget_type,
        "title": title,
        "data_source": data_source,
        "config": config or {},
        "position": {"row": row, "col": col, "width": width, "height": height},
    }


def _make_layout(
    name: str,
    description: str = "",
    widgets: Optional[List[Dict[str, Any]]] = None,
    page_size: str = "A4",
    orientation: str = "portrait",
    header: Optional[Dict[str, Any]] = None,
    footer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a report layout dict.

    Args:
        name: Layout display name.
        description: Layout description.
        widgets: List of widget dicts.
        page_size: Page size (A4, Letter, etc).
        orientation: portrait or landscape.
        header: Header configuration.
        footer: Footer configuration.

    Returns:
        Layout dict.
    """
    return {
        "layout_id": str(uuid.uuid4())[:12],
        "name": name,
        "description": description,
        "widgets": widgets or [],
        "page_size": page_size,
        "orientation": orientation,
        "header": header or {"title": name, "show_logo": True},
        "footer": footer or {"text": "Generated by GreenLang", "show_page_numbers": True},
    }


class CustomReportBuilderTemplate:
    """
    User-defined custom report builder template.

    Provides a widget library with 30+ widget types, layout management,
    validation, and multi-format rendering. Users can compose, save,
    and load custom report layouts.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
        layouts: In-memory layout storage.
    """

    GRID_COLUMNS = 4

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CustomReportBuilderTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.layouts: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Layout management methods
    # ------------------------------------------------------------------

    def add_widget(
        self, layout: Dict[str, Any], widget: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a widget to a layout.

        Args:
            layout: Layout dict.
            widget: Widget dict to add.

        Returns:
            Updated layout dict with widget added.
        """
        updated = dict(layout)
        updated["widgets"] = list(layout.get("widgets", []))
        updated["widgets"].append(widget)
        logger.info(
            "Added widget '%s' (%s) to layout '%s'",
            widget.get("title", ""),
            widget.get("widget_type", ""),
            layout.get("name", ""),
        )
        return updated

    def remove_widget(
        self, layout: Dict[str, Any], widget_id: str
    ) -> Dict[str, Any]:
        """Remove a widget from a layout by widget_id.

        Args:
            layout: Layout dict.
            widget_id: ID of widget to remove.

        Returns:
            Updated layout dict with widget removed.
        """
        updated = dict(layout)
        updated["widgets"] = [
            w for w in layout.get("widgets", [])
            if w.get("widget_id") != widget_id
        ]
        logger.info(
            "Removed widget '%s' from layout '%s'",
            widget_id, layout.get("name", ""),
        )
        return updated

    def save_layout(self, layout: Dict[str, Any]) -> str:
        """Save a layout as a reusable template (in-memory).

        Args:
            layout: Layout dict to save.

        Returns:
            Layout ID string.
        """
        layout_id = layout.get("layout_id", str(uuid.uuid4())[:12])
        self.layouts[layout_id] = dict(layout)
        logger.info("Saved layout '%s' with ID '%s'", layout.get("name", ""), layout_id)
        return layout_id

    def load_layout(self, layout_id: str) -> Dict[str, Any]:
        """Load a saved layout by ID.

        Args:
            layout_id: Layout identifier.

        Returns:
            Layout dict.

        Raises:
            KeyError: If layout_id not found.
        """
        if layout_id not in self.layouts:
            raise KeyError(f"Layout '{layout_id}' not found")
        return dict(self.layouts[layout_id])

    def list_widgets(self) -> List[Dict[str, Any]]:
        """List all available widget types with descriptions.

        Returns:
            List of widget type descriptors.
        """
        return list(WIDGET_CATALOG)

    def validate_layout(self, layout: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate a layout for issues.

        Checks for overlapping widgets, missing data sources, invalid
        widget types, and out-of-bounds positions.

        Args:
            layout: Layout dict to validate.

        Returns:
            List of validation issue dicts with 'severity' and 'message'.
        """
        issues: List[Dict[str, str]] = []
        widgets: List[Dict[str, Any]] = layout.get("widgets", [])
        valid_types = {w["type"] for w in WIDGET_CATALOG}

        for widget in widgets:
            wtype = widget.get("widget_type", "")
            wid = widget.get("widget_id", "unknown")
            title = widget.get("title", "Untitled")

            if wtype not in valid_types:
                issues.append({
                    "severity": "error",
                    "message": f"Widget '{title}' ({wid}) has invalid type '{wtype}'",
                })

            pos = widget.get("position", {})
            col = pos.get("col", 0)
            width = pos.get("width", 1)
            if col + width > self.GRID_COLUMNS:
                issues.append({
                    "severity": "warning",
                    "message": (
                        f"Widget '{title}' ({wid}) exceeds grid boundary: "
                        f"col={col}, width={width}, max_cols={self.GRID_COLUMNS}"
                    ),
                })

            if not widget.get("data_source") and wtype not in (
                "text_block", "separator", "image", "markdown_block",
                "header_block", "callout_box",
            ):
                issues.append({
                    "severity": "warning",
                    "message": f"Widget '{title}' ({wid}) has no data_source configured",
                })

        overlaps = self._check_overlaps(widgets)
        for overlap in overlaps:
            issues.append({
                "severity": "error",
                "message": (
                    f"Widgets '{overlap[0]}' and '{overlap[1]}' overlap "
                    f"at grid position"
                ),
            })

        return issues

    def render_layout(
        self, layout: Dict[str, Any], data: Dict[str, Any]
    ) -> str:
        """Render a complete report from a layout definition.

        This is a convenience method that calls render_html internally.

        Args:
            layout: Layout dict with widgets.
            data: Data dict for populating widgets.

        Returns:
            Rendered HTML string.
        """
        combined = {"layout": layout, "data": data}
        return self.render_html(combined)

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render custom report as Markdown.

        Args:
            data: Dict containing 'layout' and 'data' keys.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        layout = data.get("layout", {})
        report_data = data.get("data", {})
        sections: List[str] = []

        sections.append(self._render_md_header(layout))

        widgets = layout.get("widgets", [])
        sorted_widgets = sorted(
            widgets,
            key=lambda w: (
                w.get("position", {}).get("row", 0),
                w.get("position", {}).get("col", 0),
            ),
        )
        for widget in sorted_widgets:
            sections.append(self._render_md_widget(widget, report_data))

        sections.append(self._render_md_footer(layout))

        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render custom report as self-contained HTML.

        Args:
            data: Dict containing 'layout' and 'data' keys.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        layout = data.get("layout", {})
        report_data = data.get("data", {})
        css = self._build_css(layout)
        body_parts: List[str] = []

        body_parts.append(self._render_html_report_header(layout))

        widgets = layout.get("widgets", [])
        sorted_widgets = sorted(
            widgets,
            key=lambda w: (
                w.get("position", {}).get("row", 0),
                w.get("position", {}).get("col", 0),
            ),
        )

        body_parts.append("<div class=\"widget-grid\">")
        for widget in sorted_widgets:
            body_parts.append(self._render_html_widget(widget, report_data))
        body_parts.append("</div>")

        body_parts.append(self._render_html_report_footer(layout))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>{self._escape_html(layout.get('name', 'Custom Report'))}</title>\n"
            f"<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"report-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render custom report as structured JSON.

        Args:
            data: Dict containing 'layout' and 'data' keys.

        Returns:
            Structured dict with layout, widgets, and rendered data.
        """
        self.generated_at = datetime.utcnow()
        layout = data.get("layout", {})
        report_data = data.get("data", {})

        rendered_widgets = []
        for widget in layout.get("widgets", []):
            rendered_widgets.append(
                self._build_json_widget(widget, report_data)
            )

        result: Dict[str, Any] = {
            "template": "custom_report_builder",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "layout": {
                "layout_id": layout.get("layout_id", ""),
                "name": layout.get("name", "Custom Report"),
                "description": layout.get("description", ""),
                "page_size": layout.get("page_size", "A4"),
                "orientation": layout.get("orientation", "portrait"),
                "widget_count": len(rendered_widgets),
            },
            "widgets": rendered_widgets,
            "validation_issues": self.validate_layout(layout),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown widget renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, layout: Dict[str, Any]) -> str:
        """Render Markdown report header."""
        name = layout.get("name", "Custom Report")
        description = layout.get("description", "")
        ts = self._format_date(self.generated_at)
        header = layout.get("header", {})
        title = header.get("title", name)

        lines = [f"# {title}"]
        if description:
            lines.append(f"\n_{description}_")
        lines.append(f"\n**Generated:** {ts}")
        lines.append("\n---")
        return "\n".join(lines)

    def _render_md_widget(
        self, widget: Dict[str, Any], data: Dict[str, Any]
    ) -> str:
        """Render a single widget as Markdown.

        Args:
            widget: Widget dict.
            data: Report data dict.

        Returns:
            Markdown string for the widget.
        """
        wtype = widget.get("widget_type", "text_block")
        title = widget.get("title", "Widget")
        ds = widget.get("data_source", "")
        widget_data = data.get(ds, {}) if ds else {}
        config = widget.get("config", {})

        if wtype == "kpi_card":
            return self._render_md_kpi_widget(title, widget_data, config)
        elif wtype == "table":
            return self._render_md_table_widget(title, widget_data, config)
        elif wtype in ("bar_chart", "line_chart", "pie_chart", "area_chart",
                        "stacked_bar", "donut_chart", "scatter_plot",
                        "radar_chart", "bubble_chart", "histogram",
                        "box_plot", "treemap", "sankey", "waterfall",
                        "heatmap", "map"):
            return self._render_md_chart_widget(title, wtype, widget_data)
        elif wtype == "text_block":
            return self._render_md_text_widget(title, widget_data, config)
        elif wtype == "separator":
            return "---"
        elif wtype == "gauge":
            return self._render_md_gauge_widget(title, widget_data)
        elif wtype == "sparkline":
            return self._render_md_sparkline_widget(title, widget_data)
        elif wtype == "progress_bar":
            return self._render_md_progress_widget(title, widget_data)
        elif wtype == "metric_comparison":
            return self._render_md_comparison_widget(title, widget_data)
        elif wtype == "timeline":
            return self._render_md_timeline_widget(title, widget_data)
        elif wtype == "callout_box":
            return self._render_md_callout_widget(title, widget_data, config)
        elif wtype == "ranking_list":
            return self._render_md_ranking_widget(title, widget_data)
        elif wtype == "header_block":
            return self._render_md_header_block(title, config)
        elif wtype == "markdown_block":
            return widget_data.get("content", config.get("content", ""))
        elif wtype == "image":
            url = widget_data.get("url", config.get("url", ""))
            alt = widget_data.get("alt", title)
            return f"![{alt}]({url})" if url else ""
        elif wtype == "icon_stat":
            return self._render_md_kpi_widget(title, widget_data, config)
        elif wtype == "comparison_table":
            return self._render_md_table_widget(title, widget_data, config)
        else:
            return f"### {title}\n\n_[{wtype} widget]_"

    def _render_md_kpi_widget(
        self, title: str, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render KPI card widget in Markdown."""
        value = data.get("value", config.get("value", 0))
        unit = data.get("unit", config.get("unit", ""))
        trend = data.get("trend", "")
        return (
            f"### {title}\n\n"
            f"**{self._format_number(value)}**"
            f"{' ' + unit if unit else ''}"
            f"{' (' + trend + ')' if trend else ''}"
        )

    def _render_md_table_widget(
        self, title: str, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render table widget in Markdown."""
        headers: List[str] = data.get("headers", config.get("headers", []))
        rows: List[List[Any]] = data.get("rows", config.get("rows", []))
        if not headers:
            return f"### {title}\n\n_No table data._"

        lines = [f"### {title}", ""]
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _render_md_chart_widget(
        title: str, chart_type: str, data: Dict[str, Any]
    ) -> str:
        """Render chart placeholder in Markdown."""
        label = chart_type.replace("_", " ").title()
        return f"### {title}\n\n_[{label} Chart]_"

    @staticmethod
    def _render_md_text_widget(
        title: str, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render text block widget in Markdown."""
        content = data.get("content", config.get("content", ""))
        if title:
            return f"### {title}\n\n{content}"
        return content

    def _render_md_gauge_widget(self, title: str, data: Dict[str, Any]) -> str:
        """Render gauge widget in Markdown."""
        value = data.get("value", 0)
        max_val = data.get("max", 100)
        pct = (value / max_val * 100) if max_val else 0
        bar = self._text_bar(pct)
        return f"### {title}\n\n{self._format_number(value, 1)}/{max_val} {bar}"

    def _render_md_sparkline_widget(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render sparkline widget in Markdown."""
        values = data.get("values", [])
        if not values:
            return f"### {title}\n\n_No data_"
        current = values[-1]
        trend = "Up" if len(values) > 1 and values[-1] > values[0] else "Down"
        return f"### {title}\n\nCurrent: {self._format_number(current)} ({trend})"

    def _render_md_progress_widget(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render progress bar widget in Markdown."""
        value = data.get("value", 0)
        target = data.get("target", 100)
        pct = (value / target * 100) if target else 0
        bar = self._text_bar(pct)
        return f"### {title}\n\n{self._format_percentage(pct)} {bar}"

    def _render_md_comparison_widget(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render metric comparison widget in Markdown."""
        items: List[Dict[str, Any]] = data.get("items", [])
        if not items:
            return f"### {title}\n\n_No comparison data._"

        lines = [f"### {title}", ""]
        for item in items:
            label = item.get("label", "-")
            value = self._format_number(item.get("value", 0))
            lines.append(f"- **{label}:** {value}")
        return "\n".join(lines)

    @staticmethod
    def _render_md_timeline_widget(
        title: str, data: Dict[str, Any]
    ) -> str:
        """Render timeline widget in Markdown."""
        events: List[Dict[str, Any]] = data.get("events", [])
        if not events:
            return f"### {title}\n\n_No timeline data._"

        lines = [f"### {title}", ""]
        for evt in events:
            date = evt.get("date", "-")
            label = evt.get("label", "-")
            desc = evt.get("description", "")
            lines.append(f"- **{date}** - {label}{': ' + desc if desc else ''}")
        return "\n".join(lines)

    @staticmethod
    def _render_md_callout_widget(
        title: str, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render callout box widget in Markdown."""
        content = data.get("content", config.get("content", ""))
        style = data.get("style", config.get("style", "info"))
        return f"> **{style.upper()}: {title}**\n> {content}"

    def _render_md_ranking_widget(
        self, title: str, data: Dict[str, Any]
    ) -> str:
        """Render ranking list widget in Markdown."""
        items: List[Dict[str, Any]] = data.get("items", [])
        if not items:
            return f"### {title}\n\n_No ranking data._"

        lines = [f"### {title}", ""]
        for idx, item in enumerate(items, 1):
            label = item.get("label", "-")
            value = self._format_number(item.get("value", 0))
            lines.append(f"{idx}. **{label}** - {value}")
        return "\n".join(lines)

    @staticmethod
    def _render_md_header_block(title: str, config: Dict[str, Any]) -> str:
        """Render header block widget in Markdown."""
        subtitle = config.get("subtitle", "")
        level = config.get("level", 2)
        header = "#" * level + " " + title
        if subtitle:
            return f"{header}\n\n_{subtitle}_"
        return header

    def _render_md_footer(self, layout: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        footer = layout.get("footer", {})
        text = footer.get("text", "Generated by GreenLang")
        ts = self._format_date(self.generated_at)
        return f"---\n_{text} | {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML widget renderers
    # ------------------------------------------------------------------

    def _build_css(self, layout: Dict[str, Any]) -> str:
        """Build CSS for custom report."""
        return """
:root {
    --primary: #1a56db; --primary-light: #e1effe; --success: #057a55;
    --warning: #e3a008; --danger: #e02424; --bg: #f9fafb;
    --card-bg: #fff; --text: #1f2937; --text-muted: #6b7280;
    --border: #e5e7eb;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.report-container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.report-header { background: var(--primary); color: #fff; padding: 24px 32px;
    border-radius: 12px; margin-bottom: 24px; }
.report-header h1 { font-size: 24px; }
.report-header .desc { opacity: 0.85; margin-top: 4px; font-size: 13px; }
.widget-grid { display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin-bottom: 24px; }
.widget { background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.widget-title { font-size: 13px; font-weight: 600; color: var(--primary);
    margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.widget-kpi .kpi-val { font-size: 28px; font-weight: 700; color: var(--primary); }
.widget-kpi .kpi-unit { font-size: 12px; color: var(--text-muted); }
.widget-kpi .kpi-trend { font-size: 12px; margin-top: 4px; }
.widget-table table { width: 100%; border-collapse: collapse; }
.widget-table th { background: var(--primary-light); color: var(--primary);
    padding: 6px 10px; font-size: 11px; text-align: left; }
.widget-table td { padding: 6px 10px; border-bottom: 1px solid var(--border);
    font-size: 12px; }
.widget-chart { background: #f3f4f6; border: 2px dashed #d1d5db;
    border-radius: 6px; padding: 20px; text-align: center;
    color: var(--text-muted); font-size: 13px; min-height: 100px; }
.widget-text { font-size: 13px; line-height: 1.6; color: #374151; }
.widget-separator { border-top: 2px solid var(--border); margin: 8px 0; }
.widget-gauge .gauge-circle { width: 60px; height: 60px; border-radius: 50%;
    border: 5px solid var(--border); display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 16px; margin: 0 auto; }
.widget-gauge .gauge-circle.good { border-color: var(--success); color: var(--success); }
.widget-gauge .gauge-circle.fair { border-color: var(--warning); color: var(--warning); }
.widget-gauge .gauge-circle.poor { border-color: var(--danger); color: var(--danger); }
.widget-progress { margin-top: 8px; }
.widget-progress .bar-bg { height: 16px; background: #e5e7eb; border-radius: 8px;
    overflow: hidden; }
.widget-progress .bar-fill { height: 100%; background: var(--primary);
    border-radius: 8px; }
.widget-progress .bar-label { font-size: 12px; text-align: center; margin-top: 4px;
    color: var(--text-muted); }
.widget-callout { border-left: 4px solid var(--primary); padding: 12px;
    background: var(--primary-light); border-radius: 0 6px 6px 0; font-size: 13px; }
.widget-callout.warning { border-color: var(--warning); background: #fef9c3; }
.widget-callout.danger { border-color: var(--danger); background: #fde8e8; }
.widget-callout.success { border-color: var(--success); background: #d1fae5; }
.widget-ranking { list-style: none; }
.widget-ranking li { display: flex; align-items: center; padding: 4px 0;
    font-size: 12px; }
.widget-ranking .rank { width: 24px; font-weight: 700; color: var(--primary); }
.widget-ranking .rank-bar { height: 10px; background: var(--primary);
    border-radius: 5px; margin: 0 8px; }
.widget-ranking .rank-value { color: var(--text-muted); font-size: 11px; }
.widget-timeline { border-left: 2px solid var(--primary); padding-left: 16px; }
.widget-timeline .tl-event { margin-bottom: 12px; }
.widget-timeline .tl-date { font-size: 11px; font-weight: 600; color: var(--primary); }
.widget-timeline .tl-label { font-size: 13px; }
.widget-sparkline { display: flex; align-items: flex-end; gap: 1px; height: 24px; }
.widget-sparkline .spark-bar { width: 3px; background: var(--primary);
    border-radius: 1px; min-height: 2px; }
.widget-header-block { padding: 8px 0; }
.widget-header-block h2 { font-size: 18px; color: var(--primary); }
.widget-header-block .subtitle { font-size: 12px; color: var(--text-muted); }
.report-footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_report_header(self, layout: Dict[str, Any]) -> str:
        """Render HTML report header."""
        header = layout.get("header", {})
        name = self._escape_html(header.get("title", layout.get("name", "Custom Report")))
        desc = layout.get("description", "")
        ts = self._format_date(self.generated_at)
        desc_html = f"<div class=\"desc\">{self._escape_html(desc)}</div>" if desc else ""
        return (
            f"<div class=\"report-header\">\n"
            f"  <h1>{name}</h1>\n"
            f"  {desc_html}\n"
            f"  <div class=\"desc\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_widget(
        self, widget: Dict[str, Any], data: Dict[str, Any]
    ) -> str:
        """Render a single widget as HTML.

        Args:
            widget: Widget dict.
            data: Report data dict.

        Returns:
            HTML string for the widget.
        """
        wtype = widget.get("widget_type", "text_block")
        title = self._escape_html(widget.get("title", "Widget"))
        ds = widget.get("data_source", "")
        widget_data = data.get(ds, {}) if ds else {}
        config = widget.get("config", {})
        pos = widget.get("position", {})
        col_span = pos.get("width", 1)
        row_span = pos.get("height", 1)

        style = (
            f"grid-column: span {min(col_span, self.GRID_COLUMNS)}; "
            f"grid-row: span {row_span};"
        )

        content = self._render_widget_content_html(
            wtype, title, widget_data, config
        )

        title_html = (
            f"<div class=\"widget-title\">{title}</div>"
            if wtype not in ("separator", "header_block")
            else ""
        )

        return (
            f"<div class=\"widget\" style=\"{style}\">\n"
            f"  {title_html}\n"
            f"  {content}\n"
            f"</div>"
        )

    def _render_widget_content_html(
        self,
        wtype: str,
        title: str,
        data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> str:
        """Render the inner content of a widget based on type.

        Args:
            wtype: Widget type string.
            title: Widget title.
            data: Widget-specific data.
            config: Widget configuration.

        Returns:
            HTML content string.
        """
        if wtype == "kpi_card" or wtype == "icon_stat":
            return self._html_kpi_content(data, config)
        elif wtype == "table" or wtype == "comparison_table":
            return self._html_table_content(data, config)
        elif wtype in ("bar_chart", "line_chart", "pie_chart", "area_chart",
                        "stacked_bar", "donut_chart", "scatter_plot",
                        "radar_chart", "bubble_chart", "histogram",
                        "box_plot", "treemap", "sankey", "waterfall",
                        "heatmap", "map"):
            label = wtype.replace("_", " ").title()
            return f"<div class=\"widget-chart\">[{label}]</div>"
        elif wtype == "text_block" or wtype == "markdown_block":
            content = data.get("content", config.get("content", ""))
            return f"<div class=\"widget-text\">{self._escape_html(content)}</div>"
        elif wtype == "separator":
            return "<div class=\"widget-separator\"></div>"
        elif wtype == "image":
            url = data.get("url", config.get("url", ""))
            alt = self._escape_html(data.get("alt", title))
            if url:
                return f"<img src=\"{self._escape_html(url)}\" alt=\"{alt}\" style=\"max-width:100%\">"
            return "<div class=\"widget-chart\">[Image Placeholder]</div>"
        elif wtype == "gauge":
            return self._html_gauge_content(data)
        elif wtype == "sparkline":
            return self._html_sparkline_content(data)
        elif wtype == "progress_bar":
            return self._html_progress_content(data)
        elif wtype == "metric_comparison":
            return self._html_comparison_content(data)
        elif wtype == "timeline":
            return self._html_timeline_content(data)
        elif wtype == "callout_box":
            return self._html_callout_content(data, config)
        elif wtype == "ranking_list":
            return self._html_ranking_content(data)
        elif wtype == "header_block":
            return self._html_header_block_content(title, config)
        else:
            return f"<div class=\"widget-chart\">[{wtype}]</div>"

    def _html_kpi_content(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render KPI card HTML content."""
        value = data.get("value", config.get("value", 0))
        unit = data.get("unit", config.get("unit", ""))
        trend = data.get("trend", "")
        return (
            f"<div class=\"widget-kpi\">\n"
            f"  <div class=\"kpi-val\">{self._format_number(value)}</div>\n"
            f"  <div class=\"kpi-unit\">{unit}</div>\n"
            f"  <div class=\"kpi-trend\">{self._escape_html(trend)}</div>\n"
            f"</div>"
        )

    def _html_table_content(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render table HTML content."""
        headers: List[str] = data.get("headers", config.get("headers", []))
        rows: List[List[Any]] = data.get("rows", config.get("rows", []))
        if not headers:
            return "<p>No table data.</p>"

        h_cells = "".join(f"<th>{self._escape_html(str(h))}</th>" for h in headers)
        body = ""
        for row in rows:
            cells = "".join(f"<td>{self._escape_html(str(c))}</td>" for c in row)
            body += f"<tr>{cells}</tr>\n"

        return (
            f"<div class=\"widget-table\">\n"
            f"<table><thead><tr>{h_cells}</tr></thead>\n"
            f"<tbody>{body}</tbody></table>\n"
            f"</div>"
        )

    def _html_gauge_content(self, data: Dict[str, Any]) -> str:
        """Render gauge HTML content."""
        value = data.get("value", 0)
        max_val = data.get("max", 100)
        pct = (value / max_val * 100) if max_val else 0
        cls = "good" if pct >= 70 else "fair" if pct >= 40 else "poor"
        return (
            f"<div class=\"widget-gauge\">\n"
            f"  <div class=\"gauge-circle {cls}\">{self._format_number(value, 0)}</div>\n"
            f"</div>"
        )

    def _html_sparkline_content(self, data: Dict[str, Any]) -> str:
        """Render sparkline HTML content."""
        values = data.get("values", [])
        if not values:
            return "<p>No data</p>"

        max_val = max(values) if values else 1
        bars = ""
        for v in values[-30:]:
            h = (v / max_val * 24) if max_val else 2
            bars += f"<div class=\"spark-bar\" style=\"height:{max(2, h):.0f}px\"></div>"

        return f"<div class=\"widget-sparkline\">{bars}</div>"

    def _html_progress_content(self, data: Dict[str, Any]) -> str:
        """Render progress bar HTML content."""
        value = data.get("value", 0)
        target = data.get("target", 100)
        pct = (value / target * 100) if target else 0
        return (
            f"<div class=\"widget-progress\">\n"
            f"  <div class=\"bar-bg\">"
            f"<div class=\"bar-fill\" style=\"width:{min(pct, 100):.0f}%\"></div></div>\n"
            f"  <div class=\"bar-label\">{self._format_percentage(pct)}</div>\n"
            f"</div>"
        )

    def _html_comparison_content(self, data: Dict[str, Any]) -> str:
        """Render metric comparison HTML content."""
        items: List[Dict[str, Any]] = data.get("items", [])
        if not items:
            return "<p>No data</p>"

        rows = ""
        for item in items:
            label = self._escape_html(item.get("label", "-"))
            value = self._format_number(item.get("value", 0))
            rows += f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>\n"

        return f"<table><tbody>{rows}</tbody></table>"

    def _html_timeline_content(self, data: Dict[str, Any]) -> str:
        """Render timeline HTML content."""
        events: List[Dict[str, Any]] = data.get("events", [])
        if not events:
            return "<p>No timeline data.</p>"

        items = ""
        for evt in events:
            date = evt.get("date", "-")
            label = self._escape_html(evt.get("label", "-"))
            items += (
                f"<div class=\"tl-event\">\n"
                f"  <div class=\"tl-date\">{date}</div>\n"
                f"  <div class=\"tl-label\">{label}</div>\n"
                f"</div>\n"
            )
        return f"<div class=\"widget-timeline\">{items}</div>"

    def _html_callout_content(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Render callout box HTML content."""
        content = data.get("content", config.get("content", ""))
        style = data.get("style", config.get("style", "info"))
        return (
            f"<div class=\"widget-callout {style}\">"
            f"{self._escape_html(content)}</div>"
        )

    def _html_ranking_content(self, data: Dict[str, Any]) -> str:
        """Render ranking list HTML content."""
        items: List[Dict[str, Any]] = data.get("items", [])
        if not items:
            return "<p>No ranking data.</p>"

        max_val = max((i.get("value", 0) for i in items), default=1)
        lis = ""
        for idx, item in enumerate(items, 1):
            label = self._escape_html(item.get("label", "-"))
            value = item.get("value", 0)
            bar_width = (value / max_val * 120) if max_val else 0
            lis += (
                f"<li><span class=\"rank\">{idx}</span>"
                f"<span>{label}</span>"
                f"<span class=\"rank-bar\" style=\"width:{bar_width:.0f}px\"></span>"
                f"<span class=\"rank-value\">{self._format_number(value)}</span>"
                f"</li>\n"
            )

        return f"<ol class=\"widget-ranking\">{lis}</ol>"

    @staticmethod
    def _html_header_block_content(title: str, config: Dict[str, Any]) -> str:
        """Render header block HTML content."""
        subtitle = config.get("subtitle", "")
        sub_html = f"<div class=\"subtitle\">{subtitle}</div>" if subtitle else ""
        return (
            f"<div class=\"widget-header-block\">\n"
            f"  <h2>{title}</h2>\n"
            f"  {sub_html}\n"
            f"</div>"
        )

    def _render_html_report_footer(self, layout: Dict[str, Any]) -> str:
        """Render HTML report footer."""
        footer = layout.get("footer", {})
        text = self._escape_html(footer.get("text", "Generated by GreenLang"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"report-footer\">"
            f"{text} | {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _build_json_widget(
        self, widget: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build JSON representation of a widget with resolved data.

        Args:
            widget: Widget dict.
            data: Report data dict.

        Returns:
            Widget dict with resolved data.
        """
        ds = widget.get("data_source", "")
        widget_data = data.get(ds, {}) if ds else {}
        return {
            "widget_id": widget.get("widget_id", ""),
            "widget_type": widget.get("widget_type", ""),
            "title": widget.get("title", ""),
            "data_source": ds,
            "position": widget.get("position", {}),
            "config": widget.get("config", {}),
            "resolved_data": widget_data,
        }

    # ------------------------------------------------------------------
    # Overlap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_overlaps(
        widgets: List[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """Check for overlapping widgets in the grid.

        Args:
            widgets: List of widget dicts.

        Returns:
            List of (widget_id_a, widget_id_b) overlap pairs.
        """
        overlaps: List[Tuple[str, str]] = []
        occupied: Dict[Tuple[int, int], str] = {}

        for widget in widgets:
            wid = widget.get("widget_id", "unknown")
            pos = widget.get("position", {})
            row = pos.get("row", 0)
            col = pos.get("col", 0)
            width = pos.get("width", 1)
            height = pos.get("height", 1)

            for r in range(row, row + height):
                for c in range(col, col + width):
                    cell = (r, c)
                    if cell in occupied:
                        existing = occupied[cell]
                        pair = (existing, wid)
                        if pair not in overlaps and (wid, existing) not in overlaps:
                            overlaps.append(pair)
                    else:
                        occupied[cell] = wid

        return overlaps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format numeric value with thousands separator."""
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage."""
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string."""
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @staticmethod
    def _text_bar(value: float, max_val: float = 100.0, width: int = 15) -> str:
        """Create a text-based bar for Markdown tables."""
        filled = int((value / max_val) * width) if max_val else 0
        return "|" + "=" * filled + " " * (width - filled) + "|"
