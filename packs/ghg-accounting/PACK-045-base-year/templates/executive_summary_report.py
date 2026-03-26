# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReport - 2-Page Executive Summary for PACK-045.

Generates a concise executive summary covering base year status overview,
key emission metrics, recent changes and adjustments, target progress
highlights, and strategic recommendations.

Sections:
    1. Base Year Status Overview
    2. Key Emission Metrics
    3. Recent Changes & Adjustments
    4. Target Progress Highlights
    5. Strategic Recommendations

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


def _traffic_light(status: str) -> str:
    """Return traffic light label for status."""
    mapping = {"green": "GREEN", "amber": "AMBER", "red": "RED"}
    return mapping.get(status.lower(), status.upper())


def _traffic_css(status: str) -> str:
    """Return CSS class for traffic light status."""
    mapping = {"green": "tl-green", "amber": "tl-amber", "red": "tl-red"}
    return mapping.get(status.lower(), "tl-amber")


class ExecutiveSummaryReport:
    """
    Executive summary report template.

    Renders a concise 2-page executive overview of base year management
    status, key metrics, recent changes, target progress, and strategic
    recommendations. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ExecutiveSummaryReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveSummaryReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive summary as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_status_overview(data),
            self._md_key_metrics(data),
            self._md_recent_changes(data),
            self._md_target_highlights(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_status_overview(data),
            self._html_key_metrics(data),
            self._html_recent_changes(data),
            self._html_target_highlights(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "executive_summary_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "overall_status": self._get_val(data, "overall_status", "amber"),
            "key_metrics": data.get("key_metrics", {}),
            "recent_changes": data.get("recent_changes", []),
            "target_highlights": data.get("target_highlights", []),
            "recommendations": data.get("recommendations", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        status = _traffic_light(self._get_val(data, "overall_status", "amber"))
        return (
            f"# Executive Summary: Base Year Management - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Overall Status:** {status} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_status_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown base year status overview."""
        overview = data.get("status_overview", {})
        if not overview:
            return "## 1. Base Year Status\n\nNo status data available."
        lines = ["## 1. Base Year Status Overview", ""]
        areas = overview.get("areas", [])
        if areas:
            lines.extend([
                "| Area | Status | Last Updated | Notes |",
                "|------|--------|-------------|-------|",
            ])
            for a in areas:
                name = a.get("area", "")
                status = _traffic_light(a.get("status", "amber"))
                updated = a.get("last_updated", "-")
                notes = a.get("notes", "-")
                lines.append(f"| {name} | **{status}** | {updated} | {notes} |")
        return "\n".join(lines)

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown key emission metrics."""
        metrics = data.get("key_metrics", {})
        if not metrics:
            return "## 2. Key Metrics\n\nNo metrics available."
        base_total = metrics.get("base_year_total_tco2e", 0)
        current_total = metrics.get("current_total_tco2e", 0)
        adjustments = metrics.get("total_adjustments", 0)
        recalculations = metrics.get("recalculations_ytd", 0)
        data_quality = metrics.get("data_quality_score", 0)
        lines = [
            "## 2. Key Emission Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Base Year Total Emissions | {base_total:,.1f} tCO2e |",
            f"| Current Period Total | {current_total:,.1f} tCO2e |",
            f"| Adjustments Applied | {adjustments} |",
            f"| Recalculations YTD | {recalculations} |",
            f"| Data Quality Score | {data_quality:.0f}/100 |",
        ]
        return "\n".join(lines)

    def _md_recent_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown recent changes."""
        changes = data.get("recent_changes", [])
        if not changes:
            return "## 3. Recent Changes\n\nNo recent changes."
        lines = [
            "## 3. Recent Changes & Adjustments",
            "",
            "| Date | Type | Description | Impact (tCO2e) | Status |",
            "|------|------|------------|---------------|--------|",
        ]
        for c in changes:
            date = c.get("date", "")
            ctype = c.get("change_type", "")
            desc = c.get("description", "")
            impact = c.get("impact_tco2e", 0)
            status = c.get("status", "pending").upper()
            lines.append(f"| {date} | {ctype} | {desc} | {impact:+,.1f} | **{status}** |")
        return "\n".join(lines)

    def _md_target_highlights(self, data: Dict[str, Any]) -> str:
        """Render Markdown target progress highlights."""
        highlights = data.get("target_highlights", [])
        if not highlights:
            return ""
        lines = [
            "## 4. Target Progress Highlights",
            "",
            "| Target | Scope | Progress | On Track |",
            "|--------|-------|---------|----------|",
        ]
        for h in highlights:
            name = h.get("target_name", "")
            scope = h.get("scope", "")
            progress = f"{h.get('progress_pct', 0):.0f}%"
            on_track = "Yes" if h.get("on_track") else "No"
            lines.append(f"| {name} | {scope} | {progress} | **{on_track}** |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown strategic recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 5. Strategic Recommendations", ""]
        for r in recs:
            priority = r.get("priority", "medium").upper()
            action = r.get("action", "")
            owner = r.get("owner", "")
            timeline = r.get("timeline", "")
            lines.append(f"- **[{priority}]** {action}")
            if owner or timeline:
                lines.append(f"  - Owner: {owner} | Timeline: {timeline}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Executive Summary - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".tl-green{color:#2a9d8f;font-weight:700;}\n"
            ".tl-amber{color:#e9c46a;font-weight:700;}\n"
            ".tl-red{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".metric-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:160px;}\n"
            ".metric-value{font-size:1.4rem;font-weight:700;color:#1b263b;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        status = self._get_val(data, "overall_status", "amber")
        css = _traffic_css(status)
        label = _traffic_light(status)
        return (
            '<div class="section">\n'
            f"<h1>Executive Summary &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f'<strong>Status:</strong> <span class="{css}">{label}</span></p>\n<hr>\n</div>'
        )

    def _html_status_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML status overview table."""
        areas = data.get("status_overview", {}).get("areas", [])
        if not areas:
            return ""
        rows = ""
        for a in areas:
            name = a.get("area", "")
            status = a.get("status", "amber")
            css = _traffic_css(status)
            label = _traffic_light(status)
            notes = a.get("notes", "-")
            rows += f'<tr><td>{name}</td><td class="{css}">{label}</td><td>{notes}</td></tr>\n'
        return (
            '<div class="section">\n<h2>1. Status Overview</h2>\n'
            "<table><thead><tr><th>Area</th><th>Status</th>"
            "<th>Notes</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML key metrics as cards."""
        metrics = data.get("key_metrics", {})
        if not metrics:
            return ""
        base_total = metrics.get("base_year_total_tco2e", 0)
        current_total = metrics.get("current_total_tco2e", 0)
        dq = metrics.get("data_quality_score", 0)
        adjustments = metrics.get("total_adjustments", 0)
        return (
            '<div class="section">\n<h2>2. Key Metrics</h2>\n<div>\n'
            f'<div class="metric-card"><div class="metric-value">{base_total:,.0f}</div>'
            f'<div class="metric-label">Base Year tCO2e</div></div>\n'
            f'<div class="metric-card"><div class="metric-value">{current_total:,.0f}</div>'
            f'<div class="metric-label">Current tCO2e</div></div>\n'
            f'<div class="metric-card"><div class="metric-value">{dq:.0f}/100</div>'
            f'<div class="metric-label">Data Quality</div></div>\n'
            f'<div class="metric-card"><div class="metric-value">{adjustments}</div>'
            f'<div class="metric-label">Adjustments</div></div>\n'
            "</div>\n</div>"
        )

    def _html_recent_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML recent changes table."""
        changes = data.get("recent_changes", [])
        if not changes:
            return ""
        rows = ""
        for c in changes:
            date = c.get("date", "")
            ctype = c.get("change_type", "")
            desc = c.get("description", "")
            impact = c.get("impact_tco2e", 0)
            rows += f"<tr><td>{date}</td><td>{ctype}</td><td>{desc}</td><td>{impact:+,.1f}</td></tr>\n"
        return (
            '<div class="section">\n<h2>3. Recent Changes</h2>\n'
            "<table><thead><tr><th>Date</th><th>Type</th><th>Description</th>"
            "<th>Impact tCO2e</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_target_highlights(self, data: Dict[str, Any]) -> str:
        """Render HTML target progress highlights."""
        highlights = data.get("target_highlights", [])
        if not highlights:
            return ""
        rows = ""
        for h in highlights:
            name = h.get("target_name", "")
            progress = h.get("progress_pct", 0)
            on_track = h.get("on_track", False)
            css = "tl-green" if on_track else "tl-red"
            label = "Yes" if on_track else "No"
            rows += (
                f'<tr><td>{name}</td><td>{progress:.0f}%</td>'
                f'<td class="{css}"><strong>{label}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>4. Target Progress</h2>\n'
            "<table><thead><tr><th>Target</th><th>Progress</th>"
            "<th>On Track</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML strategic recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        items = ""
        for r in recs:
            priority = r.get("priority", "medium").upper()
            action = r.get("action", "")
            items += f"<li><strong>[{priority}]</strong> {action}</li>\n"
        return f'<div class="section">\n<h2>5. Recommendations</h2>\n<ul>{items}</ul>\n</div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
