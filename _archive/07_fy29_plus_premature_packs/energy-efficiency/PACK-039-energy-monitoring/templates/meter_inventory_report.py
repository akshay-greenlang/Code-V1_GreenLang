# -*- coding: utf-8 -*-
"""
MeterInventoryReportTemplate - Meter registry for PACK-039.

Generates comprehensive meter inventory reports showing meter registry
with hierarchical topology, calibration status tracking, communication
protocol summary, coverage analysis by zone/system, and data quality
metrics per meter point.

Sections:
    1. Inventory Overview
    2. Meter Hierarchy
    3. Calibration Status
    4. Protocol Summary
    5. Coverage Analysis
    6. Data Quality by Meter
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy management systems - Metering plan)
    - IEC 62053 (Electricity metering equipment)
    - ASHRAE Guideline 14 (Measurement of energy, demand, and water savings)

Author: GreenLang Team
Version: 39.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class MeterInventoryReportTemplate:
    """
    Meter inventory report template.

    Renders meter registry reports showing hierarchical meter topology,
    calibration status, communication protocol summary, zone/system
    coverage analysis, and per-meter data quality metrics across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MeterInventoryReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render meter inventory report as Markdown.

        Args:
            data: Meter inventory engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_inventory_overview(data),
            self._md_meter_hierarchy(data),
            self._md_calibration_status(data),
            self._md_protocol_summary(data),
            self._md_coverage_analysis(data),
            self._md_data_quality(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render meter inventory report as self-contained HTML.

        Args:
            data: Meter inventory engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_inventory_overview(data),
            self._html_meter_hierarchy(data),
            self._html_calibration_status(data),
            self._html_protocol_summary(data),
            self._html_coverage_analysis(data),
            self._html_data_quality(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Meter Inventory Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render meter inventory report as structured JSON.

        Args:
            data: Meter inventory engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "meter_inventory_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "inventory_overview": self._json_inventory_overview(data),
            "meter_hierarchy": data.get("meter_hierarchy", []),
            "calibration_status": self._json_calibration_status(data),
            "protocol_summary": data.get("protocol_summary", []),
            "coverage_analysis": data.get("coverage_analysis", []),
            "data_quality": data.get("data_quality", []),
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
            f"# Meter Inventory Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Date:** {data.get('reporting_date', '')}  \n"
            f"**Total Meters:** {data.get('total_meters', 0)}  \n"
            f"**Active Meters:** {data.get('active_meters', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 MeterInventoryReportTemplate v39.0.0\n\n---"
        )

    def _md_inventory_overview(self, data: Dict[str, Any]) -> str:
        """Render inventory overview section."""
        overview = data.get("inventory_overview", {})
        total = data.get("total_meters", 0)
        active = data.get("active_meters", 0)
        utilization = self._pct(active, total) if total > 0 else "0.0%"
        return (
            "## 1. Inventory Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Meters | {self._fmt(total, 0)} |\n"
            f"| Active Meters | {self._fmt(active, 0)} |\n"
            f"| Inactive/Decommissioned | {self._fmt(overview.get('inactive_meters', 0), 0)} |\n"
            f"| Utilization Rate | {utilization} |\n"
            f"| Revenue-Grade | {self._fmt(overview.get('revenue_grade', 0), 0)} |\n"
            f"| Sub-Meters | {self._fmt(overview.get('sub_meters', 0), 0)} |\n"
            f"| Virtual Meters | {self._fmt(overview.get('virtual_meters', 0), 0)} |\n"
            f"| Average Age (years) | {self._fmt(overview.get('average_age_years', 0), 1)} |"
        )

    def _md_meter_hierarchy(self, data: Dict[str, Any]) -> str:
        """Render meter hierarchy section."""
        hierarchy = data.get("meter_hierarchy", [])
        if not hierarchy:
            return "## 2. Meter Hierarchy\n\n_No meter hierarchy data available._"
        lines = [
            "## 2. Meter Hierarchy\n",
            "| Meter ID | Name | Type | Parent | Level | System | Status |",
            "|----------|------|------|--------|------:|--------|--------|",
        ]
        for meter in hierarchy:
            lines.append(
                f"| {meter.get('meter_id', '-')} "
                f"| {meter.get('name', '-')} "
                f"| {meter.get('type', '-')} "
                f"| {meter.get('parent_id', '-')} "
                f"| {meter.get('level', 0)} "
                f"| {meter.get('system', '-')} "
                f"| {meter.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_calibration_status(self, data: Dict[str, Any]) -> str:
        """Render calibration status section."""
        calibrations = data.get("calibration_status", [])
        if not calibrations:
            return "## 3. Calibration Status\n\n_No calibration data available._"
        lines = [
            "## 3. Calibration Status\n",
            "| Meter ID | Last Calibration | Next Due | Accuracy Class | Status | Drift (%) |",
            "|----------|-----------------|----------|---------------|--------|----------:|",
        ]
        for cal in calibrations:
            lines.append(
                f"| {cal.get('meter_id', '-')} "
                f"| {cal.get('last_calibration', '-')} "
                f"| {cal.get('next_due', '-')} "
                f"| {cal.get('accuracy_class', '-')} "
                f"| {cal.get('status', '-')} "
                f"| {self._fmt(cal.get('drift_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_protocol_summary(self, data: Dict[str, Any]) -> str:
        """Render communication protocol summary section."""
        protocols = data.get("protocol_summary", [])
        if not protocols:
            return "## 4. Protocol Summary\n\n_No protocol data available._"
        lines = [
            "## 4. Protocol Summary\n",
            "| Protocol | Meter Count | % of Total | Avg Polling Interval | Success Rate |",
            "|----------|----------:|----------:|--------------------:|------------:|",
        ]
        total_meters = data.get("total_meters", 1)
        for proto in protocols:
            count = proto.get("meter_count", 0)
            pct = self._pct(count, total_meters)
            lines.append(
                f"| {proto.get('protocol', '-')} "
                f"| {self._fmt(count, 0)} "
                f"| {pct} "
                f"| {proto.get('avg_polling_interval', '-')} "
                f"| {self._fmt(proto.get('success_rate', 0))}% |"
            )
        return "\n".join(lines)

    def _md_coverage_analysis(self, data: Dict[str, Any]) -> str:
        """Render coverage analysis section."""
        coverage = data.get("coverage_analysis", [])
        if not coverage:
            return "## 5. Coverage Analysis\n\n_No coverage data available._"
        lines = [
            "## 5. Coverage Analysis\n",
            "| Zone / System | Total Load (kW) | Metered Load (kW) | Coverage (%) | Gap (kW) | Priority |",
            "|--------------|---------------:|------------------:|------------:|---------:|----------|",
        ]
        for zone in coverage:
            total_load = zone.get("total_load_kw", 0)
            metered_load = zone.get("metered_load_kw", 0)
            gap = total_load - metered_load
            cov_pct = self._pct(metered_load, total_load) if total_load > 0 else "0.0%"
            lines.append(
                f"| {zone.get('zone', '-')} "
                f"| {self._fmt(total_load, 1)} "
                f"| {self._fmt(metered_load, 1)} "
                f"| {cov_pct} "
                f"| {self._fmt(gap, 1)} "
                f"| {zone.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render data quality by meter section."""
        quality = data.get("data_quality", [])
        if not quality:
            return "## 6. Data Quality by Meter\n\n_No data quality metrics available._"
        lines = [
            "## 6. Data Quality by Meter\n",
            "| Meter ID | Completeness (%) | Timeliness (%) | Accuracy (%) | Overall Score | Issues |",
            "|----------|----------------:|---------------:|------------:|-------------:|-------:|",
        ]
        for dq in quality:
            lines.append(
                f"| {dq.get('meter_id', '-')} "
                f"| {self._fmt(dq.get('completeness_pct', 0))}% "
                f"| {self._fmt(dq.get('timeliness_pct', 0))}% "
                f"| {self._fmt(dq.get('accuracy_pct', 0))}% "
                f"| {self._fmt(dq.get('overall_score', 0))}% "
                f"| {dq.get('issue_count', 0)} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Schedule overdue calibrations for meters exceeding tolerance drift",
                "Install sub-meters in unmetered zones to close coverage gaps",
                "Upgrade legacy protocols to Modbus TCP or BACnet/IP for reliability",
                "Replace end-of-life meters approaching maximum service age",
            ]
        lines = ["## 7. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-039 Energy Monitoring Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Meter Inventory Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Total Meters: {data.get("total_meters", 0)} | '
            f'Active: {data.get("active_meters", 0)}</p>'
        )

    def _html_inventory_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML inventory overview cards."""
        o = data.get("inventory_overview", {})
        total = data.get("total_meters", 0)
        active = data.get("active_meters", 0)
        util = self._pct(active, total) if total > 0 else "0.0%"
        return (
            '<h2>Inventory Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Meters</span>'
            f'<span class="value">{self._fmt(total, 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Active</span>'
            f'<span class="value">{self._fmt(active, 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Utilization</span>'
            f'<span class="value">{util}</span></div>\n'
            f'  <div class="card"><span class="label">Revenue-Grade</span>'
            f'<span class="value">{self._fmt(o.get("revenue_grade", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Avg Age (yrs)</span>'
            f'<span class="value">{self._fmt(o.get("average_age_years", 0), 1)}</span></div>\n'
            '</div>'
        )

    def _html_meter_hierarchy(self, data: Dict[str, Any]) -> str:
        """Render HTML meter hierarchy table."""
        hierarchy = data.get("meter_hierarchy", [])
        rows = ""
        for meter in hierarchy:
            rows += (
                f'<tr><td>{meter.get("meter_id", "-")}</td>'
                f'<td>{meter.get("name", "-")}</td>'
                f'<td>{meter.get("type", "-")}</td>'
                f'<td>{meter.get("parent_id", "-")}</td>'
                f'<td>{meter.get("level", 0)}</td>'
                f'<td>{meter.get("system", "-")}</td>'
                f'<td>{meter.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Meter Hierarchy</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Name</th><th>Type</th>'
            f'<th>Parent</th><th>Level</th><th>System</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_calibration_status(self, data: Dict[str, Any]) -> str:
        """Render HTML calibration status table."""
        calibrations = data.get("calibration_status", [])
        rows = ""
        for cal in calibrations:
            status = cal.get("status", "").lower()
            cls = "severity-high" if status == "overdue" else ""
            rows += (
                f'<tr><td>{cal.get("meter_id", "-")}</td>'
                f'<td>{cal.get("last_calibration", "-")}</td>'
                f'<td>{cal.get("next_due", "-")}</td>'
                f'<td>{cal.get("accuracy_class", "-")}</td>'
                f'<td class="{cls}">{cal.get("status", "-")}</td>'
                f'<td>{self._fmt(cal.get("drift_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Calibration Status</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Last Calibration</th><th>Next Due</th>'
            f'<th>Accuracy Class</th><th>Status</th><th>Drift (%)</th></tr>\n{rows}</table>'
        )

    def _html_protocol_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML protocol summary table."""
        protocols = data.get("protocol_summary", [])
        total_meters = data.get("total_meters", 1)
        rows = ""
        for proto in protocols:
            count = proto.get("meter_count", 0)
            rows += (
                f'<tr><td>{proto.get("protocol", "-")}</td>'
                f'<td>{self._fmt(count, 0)}</td>'
                f'<td>{self._pct(count, total_meters)}</td>'
                f'<td>{proto.get("avg_polling_interval", "-")}</td>'
                f'<td>{self._fmt(proto.get("success_rate", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Protocol Summary</h2>\n'
            '<table>\n<tr><th>Protocol</th><th>Meter Count</th><th>% of Total</th>'
            f'<th>Polling Interval</th><th>Success Rate</th></tr>\n{rows}</table>'
        )

    def _html_coverage_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML coverage analysis table."""
        coverage = data.get("coverage_analysis", [])
        rows = ""
        for zone in coverage:
            total_load = zone.get("total_load_kw", 0)
            metered_load = zone.get("metered_load_kw", 0)
            gap = total_load - metered_load
            cov_pct = self._pct(metered_load, total_load) if total_load > 0 else "0.0%"
            rows += (
                f'<tr><td>{zone.get("zone", "-")}</td>'
                f'<td>{self._fmt(total_load, 1)}</td>'
                f'<td>{self._fmt(metered_load, 1)}</td>'
                f'<td>{cov_pct}</td>'
                f'<td>{self._fmt(gap, 1)}</td>'
                f'<td>{zone.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Coverage Analysis</h2>\n'
            '<table>\n<tr><th>Zone / System</th><th>Total Load (kW)</th>'
            '<th>Metered Load (kW)</th><th>Coverage (%)</th>'
            f'<th>Gap (kW)</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality table."""
        quality = data.get("data_quality", [])
        rows = ""
        for dq in quality:
            score = dq.get("overall_score", 0)
            cls = "severity-high" if score < 80 else ("severity-medium" if score < 95 else "")
            rows += (
                f'<tr><td>{dq.get("meter_id", "-")}</td>'
                f'<td>{self._fmt(dq.get("completeness_pct", 0))}%</td>'
                f'<td>{self._fmt(dq.get("timeliness_pct", 0))}%</td>'
                f'<td>{self._fmt(dq.get("accuracy_pct", 0))}%</td>'
                f'<td class="{cls}">{self._fmt(score)}%</td>'
                f'<td>{dq.get("issue_count", 0)}</td></tr>\n'
            )
        return (
            '<h2>Data Quality by Meter</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Completeness</th><th>Timeliness</th>'
            f'<th>Accuracy</th><th>Overall Score</th><th>Issues</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Schedule overdue calibrations for meters exceeding tolerance drift",
            "Install sub-meters in unmetered zones to close coverage gaps",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_inventory_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON inventory overview."""
        o = data.get("inventory_overview", {})
        return {
            "total_meters": data.get("total_meters", 0),
            "active_meters": data.get("active_meters", 0),
            "inactive_meters": o.get("inactive_meters", 0),
            "revenue_grade": o.get("revenue_grade", 0),
            "sub_meters": o.get("sub_meters", 0),
            "virtual_meters": o.get("virtual_meters", 0),
            "average_age_years": o.get("average_age_years", 0),
        }

    def _json_calibration_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON calibration status summary."""
        cals = data.get("calibration_status", [])
        overdue = sum(1 for c in cals if c.get("status", "").lower() == "overdue")
        return {
            "total_calibrated": len(cals),
            "overdue_count": overdue,
            "calibrations": cals,
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        protocols = data.get("protocol_summary", [])
        coverage = data.get("coverage_analysis", [])
        quality = data.get("data_quality", [])
        cals = data.get("calibration_status", [])
        return {
            "protocol_distribution": {
                "type": "pie",
                "labels": [p.get("protocol", "") for p in protocols],
                "values": [p.get("meter_count", 0) for p in protocols],
            },
            "coverage_gap": {
                "type": "stacked_bar",
                "labels": [z.get("zone", "") for z in coverage],
                "series": {
                    "metered": [z.get("metered_load_kw", 0) for z in coverage],
                    "unmetered": [
                        z.get("total_load_kw", 0) - z.get("metered_load_kw", 0)
                        for z in coverage
                    ],
                },
            },
            "data_quality_scores": {
                "type": "bar",
                "labels": [d.get("meter_id", "") for d in quality],
                "values": [d.get("overall_score", 0) for d in quality],
            },
            "calibration_status": {
                "type": "pie",
                "labels": list(set(c.get("status", "") for c in cals)),
                "values": [
                    sum(1 for c in cals if c.get("status", "") == s)
                    for s in set(c.get("status", "") for c in cals)
                ],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
            ".severity-high{color:#dc3545;font-weight:700;}"
            ".severity-medium{color:#fd7e14;font-weight:600;}"
            ".severity-low{color:#198754;font-weight:500;}"
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
