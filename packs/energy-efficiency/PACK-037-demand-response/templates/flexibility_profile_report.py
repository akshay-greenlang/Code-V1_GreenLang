# -*- coding: utf-8 -*-
"""
FlexibilityProfileReportTemplate - Load flexibility assessment for PACK-037.

Generates comprehensive load flexibility assessment reports showing all
facility loads categorized by curtailability (fully curtailable, partially
curtailable, non-curtailable), total curtailment capacity by notification
time and duration, and load-specific operational constraints.

Sections:
    1. Flexibility Summary
    2. Load Inventory
    3. Curtailability Classification
    4. Notification Time Analysis
    5. Duration Capability Matrix
    6. Operational Constraints
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 2222 (DER aggregation)
    - EU Electricity Directive 2019/944 Art. 17 (demand response)
    - ISO/RTO baseline methodologies

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class FlexibilityProfileReportTemplate:
    """
    Load flexibility assessment report template.

    Renders facility load flexibility profiles showing curtailment capacity
    by notification time, duration limits, and operational constraints
    across markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FlexibilityProfileReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render flexibility profile report as Markdown.

        Args:
            data: Flexibility assessment engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_flexibility_summary(data),
            self._md_load_inventory(data),
            self._md_curtailability_classification(data),
            self._md_notification_time_analysis(data),
            self._md_duration_capability(data),
            self._md_operational_constraints(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render flexibility profile report as self-contained HTML.

        Args:
            data: Flexibility assessment engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_flexibility_summary(data),
            self._html_load_inventory(data),
            self._html_curtailability_classification(data),
            self._html_notification_time_analysis(data),
            self._html_duration_capability(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Flexibility Profile Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render flexibility profile report as structured JSON.

        Args:
            data: Flexibility assessment engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "flexibility_profile_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "flexibility_summary": self._json_flexibility_summary(data),
            "load_inventory": data.get("load_inventory", []),
            "curtailability_classification": self._json_curtailability(data),
            "notification_time_analysis": data.get("notification_time_analysis", []),
            "duration_capability": data.get("duration_capability", []),
            "operational_constraints": data.get("operational_constraints", []),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with facility metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Load Flexibility Profile Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Assessment Date:** {data.get('assessment_date', '')}  \n"
            f"**Peak Demand:** {self._format_power(data.get('peak_demand_kw', 0))}  \n"
            f"**Utility:** {data.get('utility', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 FlexibilityProfileReportTemplate v37.0.0\n\n---"
        )

    def _md_flexibility_summary(self, data: Dict[str, Any]) -> str:
        """Render flexibility summary section."""
        summary = data.get("flexibility_summary", {})
        total_kw = summary.get("total_curtailable_kw", 0)
        peak_kw = data.get("peak_demand_kw", 0)
        pct = self._pct(total_kw, peak_kw)
        return (
            "## 1. Flexibility Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Loads Assessed | {summary.get('total_loads', 0)} |\n"
            f"| Fully Curtailable Loads | {summary.get('fully_curtailable_count', 0)} |\n"
            f"| Partially Curtailable Loads | {summary.get('partially_curtailable_count', 0)} |\n"
            f"| Non-Curtailable Loads | {summary.get('non_curtailable_count', 0)} |\n"
            f"| Total Curtailable Capacity | {self._format_power(total_kw)} ({pct}) |\n"
            f"| Min Notification Time | {summary.get('min_notification_minutes', 0)} min |\n"
            f"| Max Sustained Duration | {summary.get('max_duration_hours', 0)} hrs |\n"
            f"| Flexibility Score | {self._fmt(summary.get('flexibility_score', 0), 1)}/100 |"
        )

    def _md_load_inventory(self, data: Dict[str, Any]) -> str:
        """Render load inventory table."""
        loads = data.get("load_inventory", [])
        if not loads:
            return "## 2. Load Inventory\n\n_No load data available._"
        lines = [
            "## 2. Load Inventory\n",
            "| # | Load Name | Category | Rated kW | Curtailable kW | Class |",
            "|---|-----------|----------|----------|---------------|-------|",
        ]
        for i, load in enumerate(loads, 1):
            lines.append(
                f"| {i} | {load.get('name', '-')} "
                f"| {load.get('category', '-')} "
                f"| {self._fmt(load.get('rated_kw', 0), 1)} "
                f"| {self._fmt(load.get('curtailable_kw', 0), 1)} "
                f"| {load.get('curtailability_class', '-')} |"
            )
        total_rated = sum(ld.get("rated_kw", 0) for ld in loads)
        total_curtailable = sum(ld.get("curtailable_kw", 0) for ld in loads)
        lines.append(
            f"| | **TOTAL** | | **{self._fmt(total_rated, 1)}** "
            f"| **{self._fmt(total_curtailable, 1)}** | |"
        )
        return "\n".join(lines)

    def _md_curtailability_classification(self, data: Dict[str, Any]) -> str:
        """Render curtailability classification breakdown."""
        classes = data.get("curtailability_classification", [])
        if not classes:
            classes = self._compute_curtailability(data)
        if not classes:
            return "## 3. Curtailability Classification\n\n_No classification data._"
        lines = [
            "## 3. Curtailability Classification\n",
            "| Classification | Load Count | Total kW | Share (%) |",
            "|---------------|-----------|----------|-----------|",
        ]
        for cls in classes:
            lines.append(
                f"| {cls.get('classification', '-')} "
                f"| {cls.get('count', 0)} "
                f"| {self._fmt(cls.get('total_kw', 0), 1)} "
                f"| {self._fmt(cls.get('share_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_notification_time_analysis(self, data: Dict[str, Any]) -> str:
        """Render notification time analysis section."""
        analysis = data.get("notification_time_analysis", [])
        if not analysis:
            return "## 4. Notification Time Analysis\n\n_No notification data._"
        lines = [
            "## 4. Notification Time Analysis\n",
            "| Notification Window | Available kW | Load Count | Cumulative kW |",
            "|--------------------|-----------:|----------:|--------------:|",
        ]
        for item in analysis:
            lines.append(
                f"| {item.get('window', '-')} "
                f"| {self._fmt(item.get('available_kw', 0), 1)} "
                f"| {item.get('load_count', 0)} "
                f"| {self._fmt(item.get('cumulative_kw', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_duration_capability(self, data: Dict[str, Any]) -> str:
        """Render duration capability matrix section."""
        matrix = data.get("duration_capability", [])
        if not matrix:
            return "## 5. Duration Capability Matrix\n\n_No duration data._"
        lines = [
            "## 5. Duration Capability Matrix\n",
            "| Duration | Available kW | % of Peak | Key Loads |",
            "|----------|-----------:|----------:|-----------|",
        ]
        for row in matrix:
            key_loads = ", ".join(row.get("key_loads", ["-"])[:3])
            lines.append(
                f"| {row.get('duration', '-')} "
                f"| {self._fmt(row.get('available_kw', 0), 1)} "
                f"| {self._fmt(row.get('pct_of_peak', 0))}% "
                f"| {key_loads} |"
            )
        return "\n".join(lines)

    def _md_operational_constraints(self, data: Dict[str, Any]) -> str:
        """Render operational constraints section."""
        constraints = data.get("operational_constraints", [])
        if not constraints:
            return "## 6. Operational Constraints\n\n_No constraints identified._"
        lines = ["## 6. Operational Constraints\n"]
        for c in constraints:
            lines.append(
                f"- **{c.get('load_name', '-')}**: {c.get('constraint', '-')} "
                f"(Impact: {c.get('impact', '-')}, "
                f"Workaround: {c.get('workaround', 'None')})"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Enroll fully curtailable loads in day-ahead DR programs",
                "Install automated load control for top 5 curtailable loads",
                "Evaluate thermal storage to extend curtailment duration",
                "Review non-curtailable loads for partial flexibility potential",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Load Flexibility Profile Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Peak Demand: {self._format_power(data.get("peak_demand_kw", 0))} | '
            f'Utility: {data.get("utility", "-")}</p>'
        )

    def _html_flexibility_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML flexibility summary cards."""
        s = data.get("flexibility_summary", {})
        return (
            '<h2>Flexibility Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Loads</span>'
            f'<span class="value">{s.get("total_loads", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Curtailable kW</span>'
            f'<span class="value">{self._fmt(s.get("total_curtailable_kw", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Min Notification</span>'
            f'<span class="value">{s.get("min_notification_minutes", 0)} min</span></div>\n'
            f'  <div class="card"><span class="label">Max Duration</span>'
            f'<span class="value">{s.get("max_duration_hours", 0)} hrs</span></div>\n'
            f'  <div class="card"><span class="label">Flexibility Score</span>'
            f'<span class="value">{self._fmt(s.get("flexibility_score", 0), 1)}/100</span></div>\n'
            '</div>'
        )

    def _html_load_inventory(self, data: Dict[str, Any]) -> str:
        """Render HTML load inventory table."""
        loads = data.get("load_inventory", [])
        rows = ""
        for ld in loads:
            cls_name = ld.get("curtailability_class", "").lower().replace(" ", "-")
            rows += (
                f'<tr><td>{ld.get("name", "-")}</td>'
                f'<td>{ld.get("category", "-")}</td>'
                f'<td>{self._fmt(ld.get("rated_kw", 0), 1)}</td>'
                f'<td>{self._fmt(ld.get("curtailable_kw", 0), 1)}</td>'
                f'<td><span class="class-{cls_name}">'
                f'{ld.get("curtailability_class", "-")}</span></td></tr>\n'
            )
        return (
            '<h2>Load Inventory</h2>\n'
            '<table>\n<tr><th>Load</th><th>Category</th>'
            f'<th>Rated kW</th><th>Curtailable kW</th><th>Class</th></tr>\n{rows}</table>'
        )

    def _html_curtailability_classification(self, data: Dict[str, Any]) -> str:
        """Render HTML curtailability classification."""
        classes = data.get("curtailability_classification", [])
        if not classes:
            classes = self._compute_curtailability(data)
        rows = ""
        for cls in classes:
            rows += (
                f'<tr><td>{cls.get("classification", "-")}</td>'
                f'<td>{cls.get("count", 0)}</td>'
                f'<td>{self._fmt(cls.get("total_kw", 0), 1)}</td>'
                f'<td>{self._fmt(cls.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Curtailability Classification</h2>\n'
            '<table>\n<tr><th>Classification</th><th>Count</th>'
            f'<th>Total kW</th><th>Share</th></tr>\n{rows}</table>'
        )

    def _html_notification_time_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML notification time analysis."""
        analysis = data.get("notification_time_analysis", [])
        rows = ""
        for item in analysis:
            rows += (
                f'<tr><td>{item.get("window", "-")}</td>'
                f'<td>{self._fmt(item.get("available_kw", 0), 1)}</td>'
                f'<td>{item.get("load_count", 0)}</td>'
                f'<td>{self._fmt(item.get("cumulative_kw", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Notification Time Analysis</h2>\n'
            '<table>\n<tr><th>Window</th><th>Available kW</th>'
            f'<th>Loads</th><th>Cumulative kW</th></tr>\n{rows}</table>'
        )

    def _html_duration_capability(self, data: Dict[str, Any]) -> str:
        """Render HTML duration capability matrix."""
        matrix = data.get("duration_capability", [])
        rows = ""
        for row in matrix:
            rows += (
                f'<tr><td>{row.get("duration", "-")}</td>'
                f'<td>{self._fmt(row.get("available_kw", 0), 1)}</td>'
                f'<td>{self._fmt(row.get("pct_of_peak", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Duration Capability</h2>\n'
            '<table>\n<tr><th>Duration</th><th>Available kW</th>'
            f'<th>% of Peak</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Enroll fully curtailable loads in day-ahead DR programs",
            "Install automated load control for top curtailable loads",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_flexibility_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON flexibility summary."""
        s = data.get("flexibility_summary", {})
        return {
            "total_loads": s.get("total_loads", 0),
            "fully_curtailable_count": s.get("fully_curtailable_count", 0),
            "partially_curtailable_count": s.get("partially_curtailable_count", 0),
            "non_curtailable_count": s.get("non_curtailable_count", 0),
            "total_curtailable_kw": s.get("total_curtailable_kw", 0),
            "min_notification_minutes": s.get("min_notification_minutes", 0),
            "max_duration_hours": s.get("max_duration_hours", 0),
            "flexibility_score": s.get("flexibility_score", 0),
        }

    def _json_curtailability(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON curtailability classification."""
        classes = data.get("curtailability_classification", [])
        if not classes:
            classes = self._compute_curtailability(data)
        return classes

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        loads = data.get("load_inventory", [])
        notification = data.get("notification_time_analysis", [])
        duration = data.get("duration_capability", [])
        classes = data.get("curtailability_classification", [])
        if not classes:
            classes = self._compute_curtailability(data)
        return {
            "curtailability_pie": {
                "type": "pie",
                "labels": [c.get("classification", "") for c in classes],
                "values": [c.get("total_kw", 0) for c in classes],
            },
            "notification_step": {
                "type": "bar",
                "labels": [n.get("window", "") for n in notification],
                "values": [n.get("cumulative_kw", 0) for n in notification],
            },
            "duration_curve": {
                "type": "line",
                "labels": [d.get("duration", "") for d in duration],
                "values": [d.get("available_kw", 0) for d in duration],
            },
            "load_waterfall": {
                "type": "waterfall",
                "items": [
                    {"label": ld.get("name", ""), "value": ld.get("curtailable_kw", 0)}
                    for ld in sorted(loads, key=lambda x: x.get("curtailable_kw", 0), reverse=True)[:10]
                ],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_curtailability(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compute curtailability classification from load inventory."""
        loads = data.get("load_inventory", [])
        if not loads:
            return []
        buckets: Dict[str, Dict[str, Any]] = {}
        total_kw = sum(ld.get("curtailable_kw", 0) for ld in loads)
        for ld in loads:
            cls = ld.get("curtailability_class", "Unknown")
            if cls not in buckets:
                buckets[cls] = {"classification": cls, "count": 0, "total_kw": 0}
            buckets[cls]["count"] += 1
            buckets[cls]["total_kw"] += ld.get("curtailable_kw", 0)
        for bucket in buckets.values():
            if total_kw > 0:
                bucket["share_pct"] = round((bucket["total_kw"] / total_kw) * 100, 1)
            else:
                bucket["share_pct"] = 0.0
        return sorted(buckets.values(), key=lambda x: x["total_kw"], reverse=True)

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".class-fully-curtailable{color:#198754;font-weight:700;}"
            ".class-partially-curtailable{color:#fd7e14;font-weight:600;}"
            ".class-non-curtailable{color:#dc3545;font-weight:500;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string (e.g., 'EUR 1,234.00').
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string (e.g., '1,234.0 kW').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 MWh').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
        return str(val)

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
