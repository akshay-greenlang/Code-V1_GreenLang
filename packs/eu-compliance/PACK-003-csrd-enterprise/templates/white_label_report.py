"""
WhiteLabelReportTemplate - Branded report generator for CSRD Enterprise Pack.

This module implements a white-label report template with dynamic branding,
custom CSS injection, cover page, table of contents, and branded sections.
Tenants can apply their own logos, colors, fonts, and optional co-branding.

Example:
    >>> template = WhiteLabelReportTemplate()
    >>> brand = {"name": "Acme Corp", "primary_color": "#003366"}
    >>> html = template.render_html(data, brand)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class WhiteLabelReportTemplate:
    """
    White-label branded report template.

    Generates fully branded CSRD reports with dynamic headers, footers,
    cover pages, table of contents, and CSS variable injection.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    DEFAULT_BRAND: Dict[str, Any] = {
        "name": "Organization",
        "logo_url": "",
        "primary_color": "#1a56db",
        "secondary_color": "#057a55",
        "accent_color": "#e3a008",
        "text_color": "#1f2937",
        "background_color": "#ffffff",
        "font_family": "'Segoe UI', Roboto, sans-serif",
        "show_powered_by": True,
        "powered_by_text": "Powered by GreenLang",
        "custom_css": "",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize WhiteLabelReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(
        self, data: Dict[str, Any], brand: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render branded report as Markdown.

        Args:
            data: Report data with sections, charts, and metadata.
            brand: Brand configuration dict.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        b = self._merge_brand(brand)
        sections: List[str] = []

        sections.append(self._render_md_cover(data, b))
        sections.append(self._render_md_toc(data, b))

        for section in data.get("sections", []):
            sections.append(self._render_md_section(section, b))

        sections.append(self._render_md_footer(data, b))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(
        self, data: Dict[str, Any], brand: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render branded report as self-contained HTML.

        Args:
            data: Report data dict.
            brand: Brand configuration dict.

        Returns:
            Complete HTML string with inline branded styles.
        """
        self.generated_at = datetime.utcnow()
        b = self._merge_brand(brand)
        css = self._build_branded_css(b)
        body_parts: List[str] = []

        body_parts.append(self._render_html_cover(data, b))
        body_parts.append(self._render_html_toc(data, b))

        for section in data.get("sections", []):
            body_parts.append(self._render_html_section(section, b))

        body_parts.append(self._render_html_footer(data, b))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        html = (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>{self._escape_html(b['name'])} - CSRD Report</title>\n"
            f"<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"report-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )
        return html

    def render_json(
        self, data: Dict[str, Any], brand: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render report as structured JSON dict.

        Args:
            data: Report data dict.
            brand: Brand configuration dict.

        Returns:
            Structured dict with all report sections and branding.
        """
        self.generated_at = datetime.utcnow()
        b = self._merge_brand(brand)

        result: Dict[str, Any] = {
            "template": "white_label_report",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "brand": {
                "name": b["name"],
                "primary_color": b["primary_color"],
                "secondary_color": b["secondary_color"],
                "font_family": b["font_family"],
                "show_powered_by": b["show_powered_by"],
            },
            "cover": self._build_json_cover(data, b),
            "table_of_contents": self._build_json_toc(data),
            "sections": [
                self._build_json_section(s) for s in data.get("sections", [])
            ],
            "metadata": data.get("metadata", {}),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Brand helpers
    # ------------------------------------------------------------------

    def _merge_brand(self, brand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user brand settings with defaults.

        Args:
            brand: User-provided brand overrides.

        Returns:
            Complete brand configuration dict.
        """
        merged = dict(self.DEFAULT_BRAND)
        if brand:
            merged.update(brand)
        return merged

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_cover(self, data: Dict[str, Any], brand: Dict[str, Any]) -> str:
        """Render Markdown cover page."""
        company = brand.get("name", "Organization")
        year = data.get("reporting_year", datetime.utcnow().year)
        assurance = data.get("assurance_level", "Limited")
        subtitle = data.get("report_subtitle", "CSRD Sustainability Report")
        lines = [
            f"# {company}",
            f"## {subtitle}",
            "",
            f"**Reporting Year:** {year}",
            f"**Assurance Level:** {assurance}",
            f"**Generated:** {self._format_date(self.generated_at)}",
            "",
            "---",
        ]
        return "\n".join(lines)

    def _render_md_toc(self, data: Dict[str, Any], brand: Dict[str, Any]) -> str:
        """Render Markdown table of contents."""
        sections: List[Dict[str, Any]] = data.get("sections", [])
        if not sections:
            return ""

        lines = ["## Table of Contents", ""]
        for idx, section in enumerate(sections, 1):
            title = section.get("title", f"Section {idx}")
            anchor = title.lower().replace(" ", "-").replace("/", "")
            indent = "  " * (section.get("level", 1) - 1)
            lines.append(f"{indent}{idx}. [{title}](#{anchor})")

        return "\n".join(lines)

    def _render_md_section(
        self, section: Dict[str, Any], brand: Dict[str, Any]
    ) -> str:
        """Render a single Markdown section.

        Args:
            section: Section data with title, content, tables, charts.
            brand: Brand configuration.

        Returns:
            Markdown string for the section.
        """
        level = section.get("level", 2)
        title = section.get("title", "Section")
        header = "#" * level + " " + title

        parts = [header]

        if section.get("description"):
            parts.append(section["description"])

        if section.get("key_metrics"):
            parts.append(self._render_md_metrics_table(section["key_metrics"]))

        if section.get("tables"):
            for table_data in section["tables"]:
                parts.append(self._render_md_data_table(table_data))

        if section.get("charts"):
            for chart in section["charts"]:
                parts.append(self._render_md_chart_placeholder(chart))

        if section.get("narrative"):
            parts.append(section["narrative"])

        return "\n\n".join(parts)

    def _render_md_metrics_table(self, metrics: List[Dict[str, Any]]) -> str:
        """Render key metrics as a Markdown table."""
        lines = [
            "| Metric | Value | Unit | Trend |",
            "|--------|-------|------|-------|",
        ]
        for m in metrics:
            name = m.get("name", "-")
            value = self._format_number(m.get("value", 0))
            unit = m.get("unit", "")
            trend = m.get("trend", "-")
            lines.append(f"| {name} | {value} | {unit} | {trend} |")
        return "\n".join(lines)

    def _render_md_data_table(self, table_data: Dict[str, Any]) -> str:
        """Render a generic data table in Markdown."""
        title = table_data.get("title", "")
        headers: List[str] = table_data.get("headers", [])
        rows: List[List[str]] = table_data.get("rows", [])

        if not headers:
            return ""

        parts = []
        if title:
            parts.append(f"**{title}**")

        parts.append("| " + " | ".join(headers) + " |")
        parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            parts.append("| " + " | ".join(str(c) for c in row) + " |")

        return "\n".join(parts)

    @staticmethod
    def _render_md_chart_placeholder(chart: Dict[str, Any]) -> str:
        """Render a chart placeholder in Markdown."""
        chart_type = chart.get("type", "chart")
        title = chart.get("title", "Chart")
        return f"_[{chart_type}: {title}]_"

    def _render_md_footer(self, data: Dict[str, Any], brand: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        powered = ""
        if brand.get("show_powered_by", True):
            powered = f" | {brand.get('powered_by_text', 'Powered by GreenLang')}"
        return f"---\n_{brand['name']} - Generated {ts}{powered}_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_branded_css(self, brand: Dict[str, Any]) -> str:
        """Build CSS with brand variables injected.

        Args:
            brand: Brand configuration with colors and fonts.

        Returns:
            CSS string with brand variables.
        """
        custom_css = brand.get("custom_css", "")
        return f"""
:root {{
    --brand-primary: {brand['primary_color']};
    --brand-secondary: {brand['secondary_color']};
    --brand-accent: {brand['accent_color']};
    --brand-text: {brand['text_color']};
    --brand-bg: {brand['background_color']};
    --brand-font: {brand['font_family']};
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: var(--brand-font); background: #f9fafb;
    color: var(--brand-text); line-height: 1.6; }}
.report-container {{ max-width: 1100px; margin: 0 auto;
    background: var(--brand-bg); box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
.cover-page {{ background: var(--brand-primary); color: #fff; padding: 80px 60px;
    text-align: center; min-height: 400px; display: flex; flex-direction: column;
    justify-content: center; align-items: center; }}
.cover-page .logo {{ max-height: 80px; margin-bottom: 24px; }}
.cover-page h1 {{ font-size: 36px; font-weight: 700; margin-bottom: 8px; }}
.cover-page .subtitle {{ font-size: 20px; opacity: 0.9; }}
.cover-page .meta {{ margin-top: 32px; font-size: 14px; opacity: 0.8; }}
.cover-page .meta span {{ display: inline-block; margin: 0 16px; }}
.toc {{ padding: 40px 60px; border-bottom: 2px solid var(--brand-primary); }}
.toc h2 {{ color: var(--brand-primary); font-size: 24px; margin-bottom: 16px; }}
.toc-list {{ list-style: none; counter-reset: toc-counter; }}
.toc-list li {{ padding: 6px 0; border-bottom: 1px dotted #d1d5db; }}
.toc-list li::before {{ counter-increment: toc-counter;
    content: counter(toc-counter) "."; font-weight: 600;
    color: var(--brand-primary); margin-right: 8px; }}
.toc-list li a {{ color: var(--brand-text); text-decoration: none; }}
.toc-list .toc-page {{ float: right; color: #6b7280; }}
.report-section {{ padding: 40px 60px; border-bottom: 1px solid #e5e7eb; }}
.section-header {{ color: var(--brand-primary); font-size: 22px; font-weight: 600;
    padding-bottom: 8px; border-bottom: 3px solid var(--brand-primary);
    margin-bottom: 16px; }}
.section-description {{ margin-bottom: 16px; color: #4b5563; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px; margin-bottom: 20px; }}
.metric-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px;
    text-align: center; border-top: 3px solid var(--brand-primary); }}
.metric-card .metric-value {{ font-size: 28px; font-weight: 700;
    color: var(--brand-primary); }}
.metric-card .metric-label {{ font-size: 12px; color: #6b7280; margin-top: 2px; }}
.metric-card .metric-trend {{ font-size: 11px; margin-top: 4px; }}
.metric-card .metric-trend.up {{ color: #e02424; }}
.metric-card .metric-trend.down {{ color: #057a55; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
th {{ background: var(--brand-primary); color: #fff; padding: 10px 14px;
    text-align: left; font-size: 13px; font-weight: 600; }}
td {{ padding: 10px 14px; border-bottom: 1px solid #e5e7eb; font-size: 13px; }}
tr:nth-child(even) {{ background: #f9fafb; }}
.chart-placeholder {{ background: #f3f4f6; border: 2px dashed #d1d5db;
    border-radius: 8px; padding: 40px; text-align: center; color: #6b7280;
    margin-bottom: 16px; }}
.narrative {{ line-height: 1.8; color: #374151; margin-bottom: 16px; }}
.report-footer {{ padding: 24px 60px; background: #f9fafb;
    border-top: 2px solid var(--brand-primary); display: flex;
    justify-content: space-between; align-items: center; font-size: 12px;
    color: #6b7280; }}
.report-footer .powered-by {{ opacity: 0.7; }}
.branded-header {{ background: var(--brand-primary); color: #fff; padding: 12px 60px;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; }}
.branded-header .company-name {{ font-weight: 600; }}
{custom_css}
"""

    def _render_html_cover(self, data: Dict[str, Any], brand: Dict[str, Any]) -> str:
        """Render HTML cover page with branding."""
        company = self._escape_html(brand.get("name", "Organization"))
        logo_url = brand.get("logo_url", "")
        year = data.get("reporting_year", datetime.utcnow().year)
        assurance = data.get("assurance_level", "Limited")
        subtitle = self._escape_html(
            data.get("report_subtitle", "CSRD Sustainability Report")
        )
        ts = self._format_date(self.generated_at)

        logo_html = ""
        if logo_url:
            logo_html = (
                f"<img class=\"logo\" src=\"{self._escape_html(logo_url)}\" "
                f"alt=\"{company} Logo\">"
            )

        return (
            f"<div class=\"cover-page\">\n"
            f"  {logo_html}\n"
            f"  <h1>{company}</h1>\n"
            f"  <div class=\"subtitle\">{subtitle}</div>\n"
            f"  <div class=\"meta\">\n"
            f"    <span>Reporting Year: {year}</span>\n"
            f"    <span>Assurance Level: {assurance}</span>\n"
            f"    <span>Generated: {ts}</span>\n"
            f"  </div>\n"
            f"</div>"
        )

    def _render_html_toc(self, data: Dict[str, Any], brand: Dict[str, Any]) -> str:
        """Render HTML table of contents."""
        sections: List[Dict[str, Any]] = data.get("sections", [])
        if not sections:
            return ""

        items = ""
        for idx, section in enumerate(sections, 1):
            title = self._escape_html(section.get("title", f"Section {idx}"))
            anchor = title.lower().replace(" ", "-").replace("/", "")
            page = section.get("page_number", idx)
            indent = (section.get("level", 1) - 1) * 20
            items += (
                f"<li style=\"padding-left:{indent}px\">"
                f"<a href=\"#{anchor}\">{title}</a>"
                f"<span class=\"toc-page\">{page}</span></li>\n"
            )

        return (
            "<div class=\"toc\">\n"
            "  <h2>Table of Contents</h2>\n"
            f"  <ol class=\"toc-list\">{items}</ol>\n"
            "</div>"
        )

    def _render_html_section(
        self, section: Dict[str, Any], brand: Dict[str, Any]
    ) -> str:
        """Render a single HTML report section."""
        title = self._escape_html(section.get("title", "Section"))
        anchor = title.lower().replace(" ", "-").replace("/", "")
        parts = [
            f"<div class=\"report-section\" id=\"{anchor}\">\n"
            f"  <h2 class=\"section-header\">{title}</h2>"
        ]

        if section.get("description"):
            desc = self._escape_html(section["description"])
            parts.append(f"  <p class=\"section-description\">{desc}</p>")

        if section.get("key_metrics"):
            parts.append(self._render_html_metrics(section["key_metrics"]))

        if section.get("tables"):
            for table_data in section["tables"]:
                parts.append(self._render_html_data_table(table_data))

        if section.get("charts"):
            for chart in section["charts"]:
                parts.append(self._render_html_chart_placeholder(chart, brand))

        if section.get("narrative"):
            narrative = self._escape_html(section["narrative"])
            parts.append(f"  <div class=\"narrative\">{narrative}</div>")

        parts.append("</div>")
        return "\n".join(parts)

    def _render_html_metrics(self, metrics: List[Dict[str, Any]]) -> str:
        """Render HTML metric cards grid."""
        cards = ""
        for m in metrics:
            name = self._escape_html(m.get("name", "-"))
            value = self._format_number(m.get("value", 0))
            unit = m.get("unit", "")
            trend = m.get("trend_pct")
            trend_html = ""
            if trend is not None:
                cls = "down" if trend < 0 else "up"
                arrow = "v" if trend < 0 else "^"
                trend_html = (
                    f"<div class=\"metric-trend {cls}\">"
                    f"{arrow} {self._format_percentage(abs(trend))}</div>"
                )
            cards += (
                f"<div class=\"metric-card\">\n"
                f"  <div class=\"metric-value\">{value}</div>\n"
                f"  <div class=\"metric-label\">{name}"
                f"{' (' + unit + ')' if unit else ''}</div>\n"
                f"  {trend_html}\n"
                f"</div>\n"
            )
        return f"<div class=\"metrics-grid\">{cards}</div>"

    def _render_html_data_table(self, table_data: Dict[str, Any]) -> str:
        """Render an HTML data table."""
        title = table_data.get("title", "")
        headers: List[str] = table_data.get("headers", [])
        rows: List[List[Any]] = table_data.get("rows", [])

        if not headers:
            return ""

        parts = []
        if title:
            parts.append(f"<h3>{self._escape_html(title)}</h3>")

        header_cells = "".join(
            f"<th>{self._escape_html(h)}</th>" for h in headers
        )
        body_rows = ""
        for row in rows:
            cells = "".join(
                f"<td>{self._escape_html(str(c))}</td>" for c in row
            )
            body_rows += f"<tr>{cells}</tr>\n"

        parts.append(
            f"<table><thead><tr>{header_cells}</tr></thead>\n"
            f"<tbody>{body_rows}</tbody></table>"
        )
        return "\n".join(parts)

    def _render_html_chart_placeholder(
        self, chart: Dict[str, Any], brand: Dict[str, Any]
    ) -> str:
        """Render an HTML chart placeholder with brand color note."""
        chart_type = chart.get("type", "chart")
        title = self._escape_html(chart.get("title", "Chart"))
        primary = brand.get("primary_color", "#1a56db")
        return (
            f"<div class=\"chart-placeholder\">\n"
            f"  <strong>[{chart_type}]</strong> {title}<br>\n"
            f"  <small>Brand color: {primary}</small>\n"
            f"</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any], brand: Dict[str, Any]) -> str:
        """Render HTML footer with branding."""
        company = self._escape_html(brand.get("name", "Organization"))
        ts = self._format_date(self.generated_at)
        powered = ""
        if brand.get("show_powered_by", True):
            text = self._escape_html(
                brand.get("powered_by_text", "Powered by GreenLang")
            )
            powered = f"<span class=\"powered-by\">{text}</span>"

        return (
            f"<div class=\"report-footer\">\n"
            f"  <span>{company} | Generated {ts}</span>\n"
            f"  {powered}\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _build_json_cover(
        self, data: Dict[str, Any], brand: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build JSON cover page data."""
        return {
            "company_name": brand.get("name", "Organization"),
            "logo_url": brand.get("logo_url", ""),
            "reporting_year": data.get("reporting_year", datetime.utcnow().year),
            "assurance_level": data.get("assurance_level", "Limited"),
            "subtitle": data.get("report_subtitle", "CSRD Sustainability Report"),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    def _build_json_toc(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON table of contents."""
        return [
            {
                "title": s.get("title", f"Section {i}"),
                "level": s.get("level", 1),
                "page_number": s.get("page_number", i),
            }
            for i, s in enumerate(data.get("sections", []), 1)
        ]

    @staticmethod
    def _build_json_section(section: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON representation of a single section."""
        return {
            "title": section.get("title", "Section"),
            "level": section.get("level", 2),
            "description": section.get("description", ""),
            "key_metrics": section.get("key_metrics", []),
            "tables": section.get("tables", []),
            "charts": section.get("charts", []),
            "narrative": section.get("narrative", ""),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash.

        Args:
            content: Content to hash.

        Returns:
            Hexadecimal SHA-256 hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format a numeric value with thousands separator.

        Args:
            value: Numeric value.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage.

        Args:
            value: Numeric value.

        Returns:
            Percentage string.
        """
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string.

        Args:
            dt: Datetime object.

        Returns:
            Formatted date string.
        """
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Raw text.

        Returns:
            HTML-safe string.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
