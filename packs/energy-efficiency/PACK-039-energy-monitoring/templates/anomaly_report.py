# -*- coding: utf-8 -*-
"""
AnomalyReportTemplate - Detected anomalies report for PACK-039.

Generates comprehensive anomaly detection reports showing detected
energy anomalies with severity classification, root cause analysis,
estimated waste quantification, investigation status tracking, and
resolution action management.

Sections:
    1. Anomaly Overview
    2. Anomaly Detail Table
    3. Severity Distribution
    4. Root Cause Analysis
    5. Estimated Waste
    6. Investigation Status
    7. Resolution Actions
    8. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISO 50001:2018 (Energy management - Significant energy uses)
    - ASHRAE Guideline 14 (Measurement of energy and demand savings)
    - EN 15232 (Building automation impact on energy performance)

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


class AnomalyReportTemplate:
    """
    Anomaly detection report template.

    Renders anomaly detection reports showing detected energy anomalies
    with severity classification, root cause analysis, estimated waste
    quantification, investigation status, and resolution actions across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AnomalyReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render anomaly report as Markdown.

        Args:
            data: Anomaly detection engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_anomaly_overview(data),
            self._md_anomaly_detail(data),
            self._md_severity_distribution(data),
            self._md_root_cause_analysis(data),
            self._md_estimated_waste(data),
            self._md_investigation_status(data),
            self._md_resolution_actions(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render anomaly report as self-contained HTML.

        Args:
            data: Anomaly detection engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_anomaly_overview(data),
            self._html_anomaly_detail(data),
            self._html_severity_distribution(data),
            self._html_root_cause_analysis(data),
            self._html_estimated_waste(data),
            self._html_investigation_status(data),
            self._html_resolution_actions(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Anomaly Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render anomaly report as structured JSON.

        Args:
            data: Anomaly detection engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "anomaly_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "anomaly_overview": self._json_anomaly_overview(data),
            "anomaly_detail": data.get("anomaly_detail", []),
            "severity_distribution": data.get("severity_distribution", {}),
            "root_cause_analysis": data.get("root_cause_analysis", []),
            "estimated_waste": self._json_estimated_waste(data),
            "investigation_status": data.get("investigation_status", []),
            "resolution_actions": data.get("resolution_actions", []),
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
            f"# Energy Anomaly Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '')}  \n"
            f"**Total Anomalies:** {data.get('total_anomalies', 0)}  \n"
            f"**Critical Anomalies:** {data.get('critical_anomalies', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 AnomalyReportTemplate v39.0.0\n\n---"
        )

    def _md_anomaly_overview(self, data: Dict[str, Any]) -> str:
        """Render anomaly overview section."""
        overview = data.get("anomaly_overview", {})
        return (
            "## 1. Anomaly Overview\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Anomalies Detected | {self._fmt(data.get('total_anomalies', 0), 0)} |\n"
            f"| Critical | {self._fmt(overview.get('critical', 0), 0)} |\n"
            f"| High | {self._fmt(overview.get('high', 0), 0)} |\n"
            f"| Medium | {self._fmt(overview.get('medium', 0), 0)} |\n"
            f"| Low | {self._fmt(overview.get('low', 0), 0)} |\n"
            f"| Estimated Total Waste | {self._format_energy(overview.get('total_waste_mwh', 0))} |\n"
            f"| Estimated Waste Cost | {self._format_currency(overview.get('total_waste_cost', 0))} |\n"
            f"| Open Investigations | {self._fmt(overview.get('open_investigations', 0), 0)} |"
        )

    def _md_anomaly_detail(self, data: Dict[str, Any]) -> str:
        """Render anomaly detail table section."""
        anomalies = data.get("anomaly_detail", [])
        if not anomalies:
            return "## 2. Anomaly Detail\n\n_No anomalies detected in the analysis period._"
        lines = [
            "## 2. Anomaly Detail\n",
            "| # | Timestamp | System | Type | Severity | Observed | Expected | Deviation |",
            "|---|-----------|--------|------|----------|--------:|---------:|----------:|",
        ]
        for i, a in enumerate(anomalies, 1):
            lines.append(
                f"| {i} | {a.get('timestamp', '-')} "
                f"| {a.get('system', '-')} "
                f"| {a.get('type', '-')} "
                f"| {a.get('severity', '-')} "
                f"| {self._fmt(a.get('observed_kwh', 0), 1)} kWh "
                f"| {self._fmt(a.get('expected_kwh', 0), 1)} kWh "
                f"| {self._fmt(a.get('deviation_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_severity_distribution(self, data: Dict[str, Any]) -> str:
        """Render severity distribution section."""
        dist = data.get("severity_distribution", {})
        if not dist:
            return "## 3. Severity Distribution\n\n_No severity data available._"
        total = data.get("total_anomalies", 1)
        lines = [
            "## 3. Severity Distribution\n",
            "| Severity | Count | % of Total | Avg Deviation (%) |",
            "|----------|------:|----------:|------------------:|",
        ]
        for sev in ["critical", "high", "medium", "low"]:
            info = dist.get(sev, {})
            count = info.get("count", 0) if isinstance(info, dict) else info
            avg_dev = info.get("avg_deviation_pct", 0) if isinstance(info, dict) else 0
            lines.append(
                f"| {sev.capitalize()} "
                f"| {self._fmt(count, 0)} "
                f"| {self._pct(count, total)} "
                f"| {self._fmt(avg_dev)}% |"
            )
        return "\n".join(lines)

    def _md_root_cause_analysis(self, data: Dict[str, Any]) -> str:
        """Render root cause analysis section."""
        causes = data.get("root_cause_analysis", [])
        if not causes:
            return "## 4. Root Cause Analysis\n\n_No root cause data available._"
        lines = [
            "## 4. Root Cause Analysis\n",
            "| Root Cause | Occurrences | % of Anomalies | Avg Waste (MWh) | Confidence |",
            "|------------|----------:|---------------:|--------------:|----------:|",
        ]
        total = data.get("total_anomalies", 1)
        for rc in causes:
            count = rc.get("occurrences", 0)
            lines.append(
                f"| {rc.get('cause', '-')} "
                f"| {self._fmt(count, 0)} "
                f"| {self._pct(count, total)} "
                f"| {self._fmt(rc.get('avg_waste_mwh', 0), 2)} "
                f"| {self._fmt(rc.get('confidence', 0))}% |"
            )
        return "\n".join(lines)

    def _md_estimated_waste(self, data: Dict[str, Any]) -> str:
        """Render estimated waste section."""
        waste = data.get("estimated_waste", {})
        if not waste:
            return "## 5. Estimated Waste\n\n_No waste estimation data available._"
        by_system = waste.get("by_system", [])
        lines = [
            "## 5. Estimated Waste\n",
            "| System | Waste (MWh) | Waste Cost | Annualized (MWh) | Priority |",
            "|--------|----------:|-----------:|-----------------:|----------|",
        ]
        for s in by_system:
            lines.append(
                f"| {s.get('system', '-')} "
                f"| {self._fmt(s.get('waste_mwh', 0), 2)} "
                f"| {self._format_currency(s.get('waste_cost', 0))} "
                f"| {self._fmt(s.get('annualized_mwh', 0), 1)} "
                f"| {s.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_investigation_status(self, data: Dict[str, Any]) -> str:
        """Render investigation status section."""
        investigations = data.get("investigation_status", [])
        if not investigations:
            return "## 6. Investigation Status\n\n_No open investigations._"
        lines = [
            "## 6. Investigation Status\n",
            "| Anomaly ID | Assignee | Status | Days Open | Findings |",
            "|------------|----------|--------|----------:|----------|",
        ]
        for inv in investigations:
            lines.append(
                f"| {inv.get('anomaly_id', '-')} "
                f"| {inv.get('assignee', '-')} "
                f"| {inv.get('status', '-')} "
                f"| {inv.get('days_open', 0)} "
                f"| {inv.get('findings', '-')} |"
            )
        return "\n".join(lines)

    def _md_resolution_actions(self, data: Dict[str, Any]) -> str:
        """Render resolution actions section."""
        actions = data.get("resolution_actions", [])
        if not actions:
            return "## 7. Resolution Actions\n\n_No resolution actions recorded._"
        lines = [
            "## 7. Resolution Actions\n",
            "| Anomaly ID | Action | Status | Due Date | Savings (MWh/yr) |",
            "|------------|--------|--------|----------|----------------:|",
        ]
        for a in actions:
            lines.append(
                f"| {a.get('anomaly_id', '-')} "
                f"| {a.get('action', '-')} "
                f"| {a.get('status', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {self._fmt(a.get('savings_mwh_yr', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Prioritize investigation of critical anomalies with highest waste impact",
                "Implement automated fault detection rules for recurring anomaly patterns",
                "Review operational schedules for systems with after-hours anomalies",
                "Establish anomaly response SLAs based on severity classification",
            ]
        lines = ["## 8. Recommendations\n"]
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
            f'<h1>Energy Anomaly Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Total Anomalies: {data.get("total_anomalies", 0)} | '
            f'Period: {data.get("analysis_period", "-")}</p>'
        )

    def _html_anomaly_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML anomaly overview cards."""
        o = data.get("anomaly_overview", {})
        return (
            '<h2>Anomaly Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Anomalies</span>'
            f'<span class="value">{data.get("total_anomalies", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Critical</span>'
            f'<span class="value severity-high">{o.get("critical", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Estimated Waste</span>'
            f'<span class="value">{self._fmt(o.get("total_waste_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Waste Cost</span>'
            f'<span class="value">{self._format_currency(o.get("total_waste_cost", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Open Investigations</span>'
            f'<span class="value">{o.get("open_investigations", 0)}</span></div>\n'
            '</div>'
        )

    def _html_anomaly_detail(self, data: Dict[str, Any]) -> str:
        """Render HTML anomaly detail table."""
        anomalies = data.get("anomaly_detail", [])
        rows = ""
        for i, a in enumerate(anomalies, 1):
            sev = a.get("severity", "low").lower()
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{a.get("timestamp", "-")}</td>'
                f'<td>{a.get("system", "-")}</td>'
                f'<td>{a.get("type", "-")}</td>'
                f'<td class="severity-{sev}">{a.get("severity", "-")}</td>'
                f'<td>{self._fmt(a.get("observed_kwh", 0), 1)}</td>'
                f'<td>{self._fmt(a.get("expected_kwh", 0), 1)}</td>'
                f'<td>{self._fmt(a.get("deviation_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Anomaly Detail</h2>\n'
            '<table>\n<tr><th>#</th><th>Timestamp</th><th>System</th><th>Type</th>'
            '<th>Severity</th><th>Observed (kWh)</th><th>Expected (kWh)</th>'
            f'<th>Deviation</th></tr>\n{rows}</table>'
        )

    def _html_severity_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML severity distribution table."""
        dist = data.get("severity_distribution", {})
        total = data.get("total_anomalies", 1)
        rows = ""
        for sev in ["critical", "high", "medium", "low"]:
            info = dist.get(sev, {})
            count = info.get("count", 0) if isinstance(info, dict) else info
            rows += (
                f'<tr><td class="severity-{sev}">{sev.capitalize()}</td>'
                f'<td>{self._fmt(count, 0)}</td>'
                f'<td>{self._pct(count, total)}</td></tr>\n'
            )
        return (
            '<h2>Severity Distribution</h2>\n'
            '<table>\n<tr><th>Severity</th><th>Count</th>'
            f'<th>% of Total</th></tr>\n{rows}</table>'
        )

    def _html_root_cause_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML root cause analysis table."""
        causes = data.get("root_cause_analysis", [])
        rows = ""
        for rc in causes:
            rows += (
                f'<tr><td>{rc.get("cause", "-")}</td>'
                f'<td>{self._fmt(rc.get("occurrences", 0), 0)}</td>'
                f'<td>{self._fmt(rc.get("avg_waste_mwh", 0), 2)}</td>'
                f'<td>{self._fmt(rc.get("confidence", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Root Cause Analysis</h2>\n'
            '<table>\n<tr><th>Root Cause</th><th>Occurrences</th>'
            f'<th>Avg Waste (MWh)</th><th>Confidence</th></tr>\n{rows}</table>'
        )

    def _html_estimated_waste(self, data: Dict[str, Any]) -> str:
        """Render HTML estimated waste table."""
        waste = data.get("estimated_waste", {})
        by_system = waste.get("by_system", [])
        rows = ""
        for s in by_system:
            rows += (
                f'<tr><td>{s.get("system", "-")}</td>'
                f'<td>{self._fmt(s.get("waste_mwh", 0), 2)}</td>'
                f'<td>{self._format_currency(s.get("waste_cost", 0))}</td>'
                f'<td>{self._fmt(s.get("annualized_mwh", 0), 1)}</td>'
                f'<td>{s.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Estimated Waste</h2>\n'
            '<table>\n<tr><th>System</th><th>Waste (MWh)</th><th>Waste Cost</th>'
            f'<th>Annualized (MWh)</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_investigation_status(self, data: Dict[str, Any]) -> str:
        """Render HTML investigation status table."""
        investigations = data.get("investigation_status", [])
        rows = ""
        for inv in investigations:
            rows += (
                f'<tr><td>{inv.get("anomaly_id", "-")}</td>'
                f'<td>{inv.get("assignee", "-")}</td>'
                f'<td>{inv.get("status", "-")}</td>'
                f'<td>{inv.get("days_open", 0)}</td>'
                f'<td>{inv.get("findings", "-")}</td></tr>\n'
            )
        return (
            '<h2>Investigation Status</h2>\n'
            '<table>\n<tr><th>Anomaly ID</th><th>Assignee</th><th>Status</th>'
            f'<th>Days Open</th><th>Findings</th></tr>\n{rows}</table>'
        )

    def _html_resolution_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML resolution actions table."""
        actions = data.get("resolution_actions", [])
        rows = ""
        for a in actions:
            rows += (
                f'<tr><td>{a.get("anomaly_id", "-")}</td>'
                f'<td>{a.get("action", "-")}</td>'
                f'<td>{a.get("status", "-")}</td>'
                f'<td>{a.get("due_date", "-")}</td>'
                f'<td>{self._fmt(a.get("savings_mwh_yr", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Resolution Actions</h2>\n'
            '<table>\n<tr><th>Anomaly ID</th><th>Action</th><th>Status</th>'
            f'<th>Due Date</th><th>Savings (MWh/yr)</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Prioritize investigation of critical anomalies with highest waste impact",
            "Implement automated fault detection rules for recurring anomaly patterns",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_anomaly_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON anomaly overview."""
        o = data.get("anomaly_overview", {})
        return {
            "total_anomalies": data.get("total_anomalies", 0),
            "critical": o.get("critical", 0),
            "high": o.get("high", 0),
            "medium": o.get("medium", 0),
            "low": o.get("low", 0),
            "total_waste_mwh": o.get("total_waste_mwh", 0),
            "total_waste_cost": o.get("total_waste_cost", 0),
            "open_investigations": o.get("open_investigations", 0),
        }

    def _json_estimated_waste(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON estimated waste summary."""
        waste = data.get("estimated_waste", {})
        return {
            "total_waste_mwh": waste.get("total_waste_mwh", 0),
            "total_waste_cost": waste.get("total_waste_cost", 0),
            "by_system": waste.get("by_system", []),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        anomalies = data.get("anomaly_detail", [])
        causes = data.get("root_cause_analysis", [])
        waste = data.get("estimated_waste", {}).get("by_system", [])
        return {
            "severity_pie": {
                "type": "pie",
                "labels": ["Critical", "High", "Medium", "Low"],
                "values": [
                    data.get("anomaly_overview", {}).get("critical", 0),
                    data.get("anomaly_overview", {}).get("high", 0),
                    data.get("anomaly_overview", {}).get("medium", 0),
                    data.get("anomaly_overview", {}).get("low", 0),
                ],
            },
            "anomaly_timeline": {
                "type": "scatter",
                "items": [
                    {
                        "timestamp": a.get("timestamp", ""),
                        "deviation_pct": a.get("deviation_pct", 0),
                        "severity": a.get("severity", ""),
                    }
                    for a in anomalies
                ],
            },
            "root_cause_bar": {
                "type": "horizontal_bar",
                "labels": [rc.get("cause", "") for rc in causes],
                "values": [rc.get("occurrences", 0) for rc in causes],
            },
            "waste_by_system": {
                "type": "bar",
                "labels": [w.get("system", "") for w in waste],
                "values": [w.get("waste_mwh", 0) for w in waste],
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
            ".severity-high,.severity-critical{color:#dc3545;font-weight:700;}"
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
