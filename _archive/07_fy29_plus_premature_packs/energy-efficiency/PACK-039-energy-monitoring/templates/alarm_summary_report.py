# -*- coding: utf-8 -*-
"""
AlarmSummaryReportTemplate - Alarm management metrics for PACK-039.

Generates comprehensive alarm summary reports showing alarm management
KPIs (MTTA, MTTR, false alarm rate), priority distribution analysis,
standing alarm inventory, top recurring alarm patterns, and operator
response performance tracking.

Sections:
    1. Alarm KPIs
    2. Priority Distribution
    3. Alarm Activity Log
    4. Standing Alarms
    5. Top Recurring Alarms
    6. Operator Performance
    7. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - ISA-18.2 / IEC 62682 (Alarm management)
    - EEMUA 191 (Alarm systems guidelines)
    - ISO 50001:2018 (Operational control monitoring)

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


class AlarmSummaryReportTemplate:
    """
    Alarm summary report template.

    Renders alarm management reports showing KPIs (MTTA, MTTR, false
    alarm rate), priority distribution, standing alarm inventory, top
    recurring patterns, and operator response performance across
    markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AlarmSummaryReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render alarm summary report as Markdown.

        Args:
            data: Alarm summary engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_alarm_kpis(data),
            self._md_priority_distribution(data),
            self._md_alarm_activity(data),
            self._md_standing_alarms(data),
            self._md_top_recurring(data),
            self._md_operator_performance(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render alarm summary report as self-contained HTML.

        Args:
            data: Alarm summary engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_alarm_kpis(data),
            self._html_priority_distribution(data),
            self._html_alarm_activity(data),
            self._html_standing_alarms(data),
            self._html_top_recurring(data),
            self._html_operator_performance(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Alarm Summary Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render alarm summary report as structured JSON.

        Args:
            data: Alarm summary engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "alarm_summary_report",
            "version": "39.0.0",
            "generated_at": self.generated_at.isoformat(),
            "alarm_kpis": self._json_alarm_kpis(data),
            "priority_distribution": data.get("priority_distribution", []),
            "alarm_activity": data.get("alarm_activity", []),
            "standing_alarms": data.get("standing_alarms", []),
            "top_recurring": data.get("top_recurring", []),
            "operator_performance": data.get("operator_performance", []),
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
            f"# Alarm Summary Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Total Alarms:** {data.get('total_alarms', 0)}  \n"
            f"**Standing Alarms:** {data.get('standing_alarm_count', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-039 AlarmSummaryReportTemplate v39.0.0\n\n---"
        )

    def _md_alarm_kpis(self, data: Dict[str, Any]) -> str:
        """Render alarm KPIs section."""
        kpis = data.get("alarm_kpis", {})
        return (
            "## 1. Alarm KPIs\n\n"
            "| KPI | Value | Target | Status |\n|-----|-------|--------|--------|\n"
            f"| Total Alarms | {self._fmt(kpis.get('total_alarms', 0), 0)} "
            f"| {self._fmt(kpis.get('target_total', 0), 0)} "
            f"| {kpis.get('total_status', '-')} |\n"
            f"| Alarms per Day | {self._fmt(kpis.get('alarms_per_day', 0), 1)} "
            f"| {self._fmt(kpis.get('target_per_day', 0), 1)} "
            f"| {kpis.get('per_day_status', '-')} |\n"
            f"| MTTA (min) | {self._fmt(kpis.get('mtta_min', 0), 1)} "
            f"| {self._fmt(kpis.get('target_mtta', 0), 1)} "
            f"| {kpis.get('mtta_status', '-')} |\n"
            f"| MTTR (min) | {self._fmt(kpis.get('mttr_min', 0), 1)} "
            f"| {self._fmt(kpis.get('target_mttr', 0), 1)} "
            f"| {kpis.get('mttr_status', '-')} |\n"
            f"| False Alarm Rate | {self._fmt(kpis.get('false_alarm_rate', 0))}% "
            f"| {self._fmt(kpis.get('target_false_rate', 0))}% "
            f"| {kpis.get('false_rate_status', '-')} |\n"
            f"| Chattering Alarms | {self._fmt(kpis.get('chattering_count', 0), 0)} "
            f"| {self._fmt(kpis.get('target_chattering', 0), 0)} "
            f"| {kpis.get('chattering_status', '-')} |\n"
            f"| Stale Alarms | {self._fmt(kpis.get('stale_count', 0), 0)} "
            f"| {self._fmt(kpis.get('target_stale', 0), 0)} "
            f"| {kpis.get('stale_status', '-')} |"
        )

    def _md_priority_distribution(self, data: Dict[str, Any]) -> str:
        """Render priority distribution section."""
        priorities = data.get("priority_distribution", [])
        if not priorities:
            return "## 2. Priority Distribution\n\n_No priority distribution data available._"
        total = data.get("total_alarms", 1)
        lines = [
            "## 2. Priority Distribution\n",
            "| Priority | Count | % of Total | Avg MTTA (min) | Avg MTTR (min) |",
            "|----------|------:|----------:|---------------:|---------------:|",
        ]
        for p in priorities:
            count = p.get("count", 0)
            lines.append(
                f"| {p.get('priority', '-')} "
                f"| {self._fmt(count, 0)} "
                f"| {self._pct(count, total)} "
                f"| {self._fmt(p.get('avg_mtta_min', 0), 1)} "
                f"| {self._fmt(p.get('avg_mttr_min', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_alarm_activity(self, data: Dict[str, Any]) -> str:
        """Render alarm activity log section."""
        activity = data.get("alarm_activity", [])
        if not activity:
            return "## 3. Alarm Activity Log\n\n_No alarm activity data available._"
        lines = [
            "## 3. Alarm Activity Log\n",
            "| Date | New Alarms | Acknowledged | Resolved | Open End-of-Day |",
            "|------|----------:|------------:|---------:|---------------:|",
        ]
        for a in activity:
            lines.append(
                f"| {a.get('date', '-')} "
                f"| {self._fmt(a.get('new_alarms', 0), 0)} "
                f"| {self._fmt(a.get('acknowledged', 0), 0)} "
                f"| {self._fmt(a.get('resolved', 0), 0)} "
                f"| {self._fmt(a.get('open_eod', 0), 0)} |"
            )
        return "\n".join(lines)

    def _md_standing_alarms(self, data: Dict[str, Any]) -> str:
        """Render standing alarms section."""
        standing = data.get("standing_alarms", [])
        if not standing:
            return "## 4. Standing Alarms\n\n_No standing alarms._"
        lines = [
            "## 4. Standing Alarms\n",
            "| Alarm ID | Tag | Description | Priority | Duration (hrs) | System |",
            "|----------|-----|-------------|----------|---------------:|--------|",
        ]
        for s in standing:
            lines.append(
                f"| {s.get('alarm_id', '-')} "
                f"| {s.get('tag', '-')} "
                f"| {s.get('description', '-')} "
                f"| {s.get('priority', '-')} "
                f"| {self._fmt(s.get('duration_hrs', 0), 1)} "
                f"| {s.get('system', '-')} |"
            )
        return "\n".join(lines)

    def _md_top_recurring(self, data: Dict[str, Any]) -> str:
        """Render top recurring alarms section."""
        recurring = data.get("top_recurring", [])
        if not recurring:
            return "## 5. Top Recurring Alarms\n\n_No recurring alarm data available._"
        lines = [
            "## 5. Top Recurring Alarms\n",
            "| Rank | Tag | Description | Occurrences | Avg Duration (min) | Waste (MWh) |",
            "|-----:|-----|-------------|----------:|-----------------:|-----------:|",
        ]
        for i, r in enumerate(recurring, 1):
            lines.append(
                f"| {i} | {r.get('tag', '-')} "
                f"| {r.get('description', '-')} "
                f"| {self._fmt(r.get('occurrences', 0), 0)} "
                f"| {self._fmt(r.get('avg_duration_min', 0), 1)} "
                f"| {self._fmt(r.get('waste_mwh', 0), 2)} |"
            )
        return "\n".join(lines)

    def _md_operator_performance(self, data: Dict[str, Any]) -> str:
        """Render operator performance section."""
        operators = data.get("operator_performance", [])
        if not operators:
            return "## 6. Operator Performance\n\n_No operator performance data available._"
        lines = [
            "## 6. Operator Performance\n",
            "| Operator | Alarms Handled | MTTA (min) | MTTR (min) | Resolution Rate |",
            "|----------|---------------:|----------:|----------:|---------------:|",
        ]
        for op in operators:
            lines.append(
                f"| {op.get('operator', '-')} "
                f"| {self._fmt(op.get('alarms_handled', 0), 0)} "
                f"| {self._fmt(op.get('mtta_min', 0), 1)} "
                f"| {self._fmt(op.get('mttr_min', 0), 1)} "
                f"| {self._fmt(op.get('resolution_rate', 0))}% |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Rationalize chattering alarms to reduce operator fatigue",
                "Clear standing alarms older than 24 hours or re-classify as maintenance items",
                "Tune alarm setpoints for top recurring alarms to reduce false activations",
                "Implement alarm shelving for known maintenance conditions",
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
            f'<h1>Alarm Summary Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Total Alarms: {data.get("total_alarms", 0)} | '
            f'Period: {data.get("reporting_period", "-")}</p>'
        )

    def _html_alarm_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML alarm KPI cards."""
        kpis = data.get("alarm_kpis", {})
        return (
            '<h2>Alarm KPIs</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Alarms/Day</span>'
            f'<span class="value">{self._fmt(kpis.get("alarms_per_day", 0), 1)}</span></div>\n'
            f'  <div class="card"><span class="label">MTTA</span>'
            f'<span class="value">{self._fmt(kpis.get("mtta_min", 0), 1)} min</span></div>\n'
            f'  <div class="card"><span class="label">MTTR</span>'
            f'<span class="value">{self._fmt(kpis.get("mttr_min", 0), 1)} min</span></div>\n'
            f'  <div class="card"><span class="label">False Alarm Rate</span>'
            f'<span class="value">{self._fmt(kpis.get("false_alarm_rate", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Standing</span>'
            f'<span class="value">{self._fmt(kpis.get("stale_count", 0), 0)}</span></div>\n'
            '</div>'
        )

    def _html_priority_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML priority distribution table."""
        priorities = data.get("priority_distribution", [])
        rows = ""
        for p in priorities:
            prio = p.get("priority", "").lower()
            cls = "severity-high" if prio in ("critical", "1") else (
                "severity-medium" if prio in ("high", "2") else "")
            rows += (
                f'<tr><td class="{cls}">{p.get("priority", "-")}</td>'
                f'<td>{self._fmt(p.get("count", 0), 0)}</td>'
                f'<td>{self._fmt(p.get("avg_mtta_min", 0), 1)}</td>'
                f'<td>{self._fmt(p.get("avg_mttr_min", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Priority Distribution</h2>\n'
            '<table>\n<tr><th>Priority</th><th>Count</th>'
            f'<th>Avg MTTA (min)</th><th>Avg MTTR (min)</th></tr>\n{rows}</table>'
        )

    def _html_alarm_activity(self, data: Dict[str, Any]) -> str:
        """Render HTML alarm activity table."""
        activity = data.get("alarm_activity", [])
        rows = ""
        for a in activity:
            rows += (
                f'<tr><td>{a.get("date", "-")}</td>'
                f'<td>{self._fmt(a.get("new_alarms", 0), 0)}</td>'
                f'<td>{self._fmt(a.get("acknowledged", 0), 0)}</td>'
                f'<td>{self._fmt(a.get("resolved", 0), 0)}</td>'
                f'<td>{self._fmt(a.get("open_eod", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Alarm Activity Log</h2>\n'
            '<table>\n<tr><th>Date</th><th>New</th><th>Acknowledged</th>'
            f'<th>Resolved</th><th>Open EOD</th></tr>\n{rows}</table>'
        )

    def _html_standing_alarms(self, data: Dict[str, Any]) -> str:
        """Render HTML standing alarms table."""
        standing = data.get("standing_alarms", [])
        rows = ""
        for s in standing:
            dur = s.get("duration_hrs", 0)
            cls = "severity-high" if dur > 24 else ""
            rows += (
                f'<tr><td>{s.get("alarm_id", "-")}</td>'
                f'<td>{s.get("tag", "-")}</td>'
                f'<td>{s.get("description", "-")}</td>'
                f'<td>{s.get("priority", "-")}</td>'
                f'<td class="{cls}">{self._fmt(dur, 1)}</td>'
                f'<td>{s.get("system", "-")}</td></tr>\n'
            )
        return (
            '<h2>Standing Alarms</h2>\n'
            '<table>\n<tr><th>Alarm ID</th><th>Tag</th><th>Description</th>'
            f'<th>Priority</th><th>Duration (hrs)</th><th>System</th></tr>\n{rows}</table>'
        )

    def _html_top_recurring(self, data: Dict[str, Any]) -> str:
        """Render HTML top recurring alarms table."""
        recurring = data.get("top_recurring", [])
        rows = ""
        for i, r in enumerate(recurring, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{r.get("tag", "-")}</td>'
                f'<td>{r.get("description", "-")}</td>'
                f'<td>{self._fmt(r.get("occurrences", 0), 0)}</td>'
                f'<td>{self._fmt(r.get("avg_duration_min", 0), 1)}</td>'
                f'<td>{self._fmt(r.get("waste_mwh", 0), 2)}</td></tr>\n'
            )
        return (
            '<h2>Top Recurring Alarms</h2>\n'
            '<table>\n<tr><th>#</th><th>Tag</th><th>Description</th>'
            f'<th>Occurrences</th><th>Avg Duration (min)</th><th>Waste (MWh)</th></tr>\n{rows}</table>'
        )

    def _html_operator_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML operator performance table."""
        operators = data.get("operator_performance", [])
        rows = ""
        for op in operators:
            rows += (
                f'<tr><td>{op.get("operator", "-")}</td>'
                f'<td>{self._fmt(op.get("alarms_handled", 0), 0)}</td>'
                f'<td>{self._fmt(op.get("mtta_min", 0), 1)}</td>'
                f'<td>{self._fmt(op.get("mttr_min", 0), 1)}</td>'
                f'<td>{self._fmt(op.get("resolution_rate", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Operator Performance</h2>\n'
            '<table>\n<tr><th>Operator</th><th>Alarms Handled</th>'
            f'<th>MTTA (min)</th><th>MTTR (min)</th><th>Resolution Rate</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Rationalize chattering alarms to reduce operator fatigue",
            "Clear standing alarms older than 24 hours",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_alarm_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON alarm KPIs."""
        kpis = data.get("alarm_kpis", {})
        return {
            "total_alarms": kpis.get("total_alarms", 0),
            "alarms_per_day": kpis.get("alarms_per_day", 0),
            "mtta_min": kpis.get("mtta_min", 0),
            "mttr_min": kpis.get("mttr_min", 0),
            "false_alarm_rate": kpis.get("false_alarm_rate", 0),
            "chattering_count": kpis.get("chattering_count", 0),
            "stale_count": kpis.get("stale_count", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        priorities = data.get("priority_distribution", [])
        activity = data.get("alarm_activity", [])
        recurring = data.get("top_recurring", [])
        return {
            "priority_pie": {
                "type": "pie",
                "labels": [p.get("priority", "") for p in priorities],
                "values": [p.get("count", 0) for p in priorities],
            },
            "alarm_trend": {
                "type": "stacked_bar",
                "labels": [a.get("date", "") for a in activity],
                "series": {
                    "new": [a.get("new_alarms", 0) for a in activity],
                    "resolved": [a.get("resolved", 0) for a in activity],
                },
            },
            "top_recurring_bar": {
                "type": "horizontal_bar",
                "labels": [r.get("tag", "") for r in recurring],
                "values": [r.get("occurrences", 0) for r in recurring],
            },
            "daily_open_trend": {
                "type": "line",
                "labels": [a.get("date", "") for a in activity],
                "values": [a.get("open_eod", 0) for a in activity],
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
