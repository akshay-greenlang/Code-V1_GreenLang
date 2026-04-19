# -*- coding: utf-8 -*-
"""
EventPerformanceReportTemplate - Post-event analysis for PACK-037.

Generates post-event performance analysis reports for demand response events
with actual vs baseline curtailment comparison, revenue earned, penalties
incurred, performance metrics, and lessons learned for continuous improvement.

Sections:
    1. Performance Summary
    2. Baseline vs Actual Comparison
    3. Revenue & Penalties
    4. Load-Level Performance
    5. DER Contribution Analysis
    6. Lessons Learned

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 745 (performance measurement)
    - ISO/RTO settlement protocols
    - M&V IPMVP Option C/D

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


class EventPerformanceReportTemplate:
    """
    Post-event performance analysis report template.

    Renders DR event performance reports with actual vs baseline curtailment,
    revenue earned, penalties, load-level breakdown, and lessons learned
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EventPerformanceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render event performance report as Markdown.

        Args:
            data: Event performance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_performance_summary(data),
            self._md_baseline_vs_actual(data),
            self._md_revenue_penalties(data),
            self._md_load_level_performance(data),
            self._md_der_contribution(data),
            self._md_lessons_learned(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render event performance report as self-contained HTML.

        Args:
            data: Event performance engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_performance_summary(data),
            self._html_baseline_vs_actual(data),
            self._html_revenue_penalties(data),
            self._html_load_level_performance(data),
            self._html_der_contribution(data),
            self._html_lessons_learned(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Event Performance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render event performance report as structured JSON.

        Args:
            data: Event performance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "event_performance_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "performance_summary": self._json_performance_summary(data),
            "baseline_vs_actual": data.get("baseline_vs_actual", {}),
            "revenue_penalties": data.get("revenue_penalties", {}),
            "load_level_performance": data.get("load_level_performance", []),
            "der_contribution": data.get("der_contribution", []),
            "lessons_learned": data.get("lessons_learned", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# DR Event Performance Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Event ID:** {data.get('event_id', '')}  \n"
            f"**Program:** {data.get('program_name', '')}  \n"
            f"**Event Date:** {data.get('event_date', '')}  \n"
            f"**Event Window:** {data.get('event_start', '')} - {data.get('event_end', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 EventPerformanceReportTemplate v37.0.0\n\n---"
        )

    def _md_performance_summary(self, data: Dict[str, Any]) -> str:
        """Render performance summary section."""
        summary = data.get("performance_summary", {})
        required = summary.get("required_curtailment_kw", 0)
        achieved = summary.get("achieved_curtailment_kw", 0)
        pct = self._pct(achieved, required)
        return (
            "## 1. Performance Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Required Curtailment | {self._format_power(required)} |\n"
            f"| Achieved Curtailment | {self._format_power(achieved)} ({pct}) |\n"
            f"| Performance Ratio | {self._fmt(summary.get('performance_ratio', 0), 2)} |\n"
            f"| Compliance Status | {summary.get('compliance_status', '-')} |\n"
            f"| Baseline Methodology | {summary.get('baseline_methodology', '-')} |\n"
            f"| Event Duration | {summary.get('duration_hours', 0)} hours |\n"
            f"| Response Time | {summary.get('response_time_minutes', 0)} min |\n"
            f"| Net Revenue | {self._format_currency(summary.get('net_revenue', 0))} |\n"
            f"| Overall Grade | {summary.get('overall_grade', '-')} |"
        )

    def _md_baseline_vs_actual(self, data: Dict[str, Any]) -> str:
        """Render baseline vs actual comparison section."""
        comparison = data.get("baseline_vs_actual", {})
        intervals = comparison.get("intervals", [])
        if not intervals:
            return "## 2. Baseline vs Actual\n\n_No interval data available._"
        lines = [
            "## 2. Baseline vs Actual Comparison\n",
            f"**Baseline Method:** {comparison.get('methodology', '-')}  ",
            f"**Adjustment Factor:** {self._fmt(comparison.get('adjustment_factor', 1.0), 3)}\n",
            "| Interval | Baseline (kW) | Actual (kW) | Curtailment (kW) | Performance (%) |",
            "|----------|-------------:|------------:|------------------:|----------------:|",
        ]
        for iv in intervals:
            baseline_kw = iv.get("baseline_kw", 0)
            actual_kw = iv.get("actual_kw", 0)
            curtailment = baseline_kw - actual_kw
            perf_pct = self._pct(curtailment, baseline_kw) if baseline_kw > 0 else "0.0%"
            lines.append(
                f"| {iv.get('interval', '-')} "
                f"| {self._fmt(baseline_kw, 1)} "
                f"| {self._fmt(actual_kw, 1)} "
                f"| {self._fmt(curtailment, 1)} "
                f"| {perf_pct} |"
            )
        return "\n".join(lines)

    def _md_revenue_penalties(self, data: Dict[str, Any]) -> str:
        """Render revenue and penalties section."""
        rp = data.get("revenue_penalties", {})
        return (
            "## 3. Revenue & Penalties\n\n"
            "| Component | Amount |\n|-----------|-------:|\n"
            f"| Capacity Payment | {self._format_currency(rp.get('capacity_payment', 0))} |\n"
            f"| Energy Payment | {self._format_currency(rp.get('energy_payment', 0))} |\n"
            f"| Incentive Bonus | {self._format_currency(rp.get('incentive_bonus', 0))} |\n"
            f"| **Gross Revenue** | **{self._format_currency(rp.get('gross_revenue', 0))}** |\n"
            f"| Non-Performance Penalty | ({self._format_currency(rp.get('non_performance_penalty', 0))}) |\n"
            f"| Under-Delivery Penalty | ({self._format_currency(rp.get('under_delivery_penalty', 0))}) |\n"
            f"| **Net Revenue** | **{self._format_currency(rp.get('net_revenue', 0))}** |\n"
            f"| Revenue per kW | {self._format_currency(rp.get('revenue_per_kw', 0))}/kW |"
        )

    def _md_load_level_performance(self, data: Dict[str, Any]) -> str:
        """Render load-level performance breakdown."""
        loads = data.get("load_level_performance", [])
        if not loads:
            return "## 4. Load-Level Performance\n\n_No load-level data available._"
        lines = [
            "## 4. Load-Level Performance\n",
            "| Load | Planned (kW) | Actual (kW) | Variance (kW) | Score |",
            "|------|------------:|------------:|--------------:|------:|",
        ]
        for ld in loads:
            planned = ld.get("planned_kw", 0)
            actual = ld.get("actual_kw", 0)
            variance = actual - planned
            lines.append(
                f"| {ld.get('load_name', '-')} "
                f"| {self._fmt(planned, 1)} "
                f"| {self._fmt(actual, 1)} "
                f"| {self._fmt(variance, 1)} "
                f"| {self._fmt(ld.get('score', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_der_contribution(self, data: Dict[str, Any]) -> str:
        """Render DER contribution analysis section."""
        ders = data.get("der_contribution", [])
        if not ders:
            return "## 5. DER Contribution Analysis\n\n_No DER contribution data._"
        lines = [
            "## 5. DER Contribution Analysis\n",
            "| DER Asset | Type | Dispatched (kW) | Delivered (kW) | Utilization (%) | Issues |",
            "|-----------|------|----------------:|---------------:|----------------:|--------|",
        ]
        for d in ders:
            dispatched = d.get("dispatched_kw", 0)
            delivered = d.get("delivered_kw", 0)
            utilization = self._pct(delivered, dispatched) if dispatched > 0 else "N/A"
            lines.append(
                f"| {d.get('asset_name', '-')} "
                f"| {d.get('type', '-')} "
                f"| {self._fmt(dispatched, 1)} "
                f"| {self._fmt(delivered, 1)} "
                f"| {utilization} "
                f"| {d.get('issues', 'None')} |"
            )
        return "\n".join(lines)

    def _md_lessons_learned(self, data: Dict[str, Any]) -> str:
        """Render lessons learned section."""
        lessons = data.get("lessons_learned", [])
        if not lessons:
            return "## 6. Lessons Learned\n\n_No lessons documented._"
        lines = ["## 6. Lessons Learned\n"]
        for i, lesson in enumerate(lessons, 1):
            lines.append(
                f"{i}. **{lesson.get('category', 'General')}**: "
                f"{lesson.get('finding', '-')} "
                f"(Action: {lesson.get('action', '-')})"
            )
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
            f'<h1>DR Event Performance Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Event: {data.get("event_id", "-")} | '
            f'Program: {data.get("program_name", "-")}</p>'
        )

    def _html_performance_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML performance summary cards."""
        s = data.get("performance_summary", {})
        grade_cls = "grade-pass" if s.get("compliance_status") == "Compliant" else "grade-fail"
        return (
            '<h2>Performance Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Required</span>'
            f'<span class="value">{self._fmt(s.get("required_curtailment_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Achieved</span>'
            f'<span class="value">{self._fmt(s.get("achieved_curtailment_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Net Revenue</span>'
            f'<span class="value">{self._format_currency(s.get("net_revenue", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Response Time</span>'
            f'<span class="value">{s.get("response_time_minutes", 0)} min</span></div>\n'
            f'  <div class="card {grade_cls}"><span class="label">Grade</span>'
            f'<span class="value">{s.get("overall_grade", "-")}</span></div>\n'
            '</div>'
        )

    def _html_baseline_vs_actual(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline vs actual table."""
        intervals = data.get("baseline_vs_actual", {}).get("intervals", [])
        rows = ""
        for iv in intervals:
            baseline_kw = iv.get("baseline_kw", 0)
            actual_kw = iv.get("actual_kw", 0)
            curtailment = baseline_kw - actual_kw
            rows += (
                f'<tr><td>{iv.get("interval", "-")}</td>'
                f'<td>{self._fmt(baseline_kw, 1)}</td>'
                f'<td>{self._fmt(actual_kw, 1)}</td>'
                f'<td>{self._fmt(curtailment, 1)}</td></tr>\n'
            )
        return (
            '<h2>Baseline vs Actual</h2>\n'
            '<table>\n<tr><th>Interval</th><th>Baseline kW</th>'
            f'<th>Actual kW</th><th>Curtailment kW</th></tr>\n{rows}</table>'
        )

    def _html_revenue_penalties(self, data: Dict[str, Any]) -> str:
        """Render HTML revenue and penalties."""
        rp = data.get("revenue_penalties", {})
        return (
            '<h2>Revenue &amp; Penalties</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-green"><span class="label">Gross Revenue</span>'
            f'<span class="value">{self._format_currency(rp.get("gross_revenue", 0))}</span></div>\n'
            f'  <div class="card card-red"><span class="label">Penalties</span>'
            f'<span class="value">({self._format_currency(rp.get("non_performance_penalty", 0) + rp.get("under_delivery_penalty", 0))})</span></div>\n'
            f'  <div class="card"><span class="label">Net Revenue</span>'
            f'<span class="value">{self._format_currency(rp.get("net_revenue", 0))}</span></div>\n'
            '</div>'
        )

    def _html_load_level_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML load-level performance table."""
        loads = data.get("load_level_performance", [])
        rows = ""
        for ld in loads:
            variance = ld.get("actual_kw", 0) - ld.get("planned_kw", 0)
            cls = "variance-positive" if variance >= 0 else "variance-negative"
            rows += (
                f'<tr><td>{ld.get("load_name", "-")}</td>'
                f'<td>{self._fmt(ld.get("planned_kw", 0), 1)}</td>'
                f'<td>{self._fmt(ld.get("actual_kw", 0), 1)}</td>'
                f'<td class="{cls}">{self._fmt(variance, 1)}</td></tr>\n'
            )
        return (
            '<h2>Load-Level Performance</h2>\n'
            '<table>\n<tr><th>Load</th><th>Planned kW</th>'
            f'<th>Actual kW</th><th>Variance kW</th></tr>\n{rows}</table>'
        )

    def _html_der_contribution(self, data: Dict[str, Any]) -> str:
        """Render HTML DER contribution table."""
        ders = data.get("der_contribution", [])
        rows = ""
        for d in ders:
            rows += (
                f'<tr><td>{d.get("asset_name", "-")}</td>'
                f'<td>{d.get("type", "-")}</td>'
                f'<td>{self._fmt(d.get("dispatched_kw", 0), 1)}</td>'
                f'<td>{self._fmt(d.get("delivered_kw", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>DER Contribution</h2>\n'
            '<table>\n<tr><th>Asset</th><th>Type</th>'
            f'<th>Dispatched kW</th><th>Delivered kW</th></tr>\n{rows}</table>'
        )

    def _html_lessons_learned(self, data: Dict[str, Any]) -> str:
        """Render HTML lessons learned."""
        lessons = data.get("lessons_learned", [])
        items = "".join(
            f'<li><strong>[{l.get("category", "-")}]</strong> {l.get("finding", "-")} '
            f'(Action: {l.get("action", "-")})</li>\n'
            for l in lessons
        )
        return f'<h2>Lessons Learned</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_performance_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON performance summary."""
        s = data.get("performance_summary", {})
        return {
            "required_curtailment_kw": s.get("required_curtailment_kw", 0),
            "achieved_curtailment_kw": s.get("achieved_curtailment_kw", 0),
            "performance_ratio": s.get("performance_ratio", 0),
            "compliance_status": s.get("compliance_status", ""),
            "baseline_methodology": s.get("baseline_methodology", ""),
            "duration_hours": s.get("duration_hours", 0),
            "response_time_minutes": s.get("response_time_minutes", 0),
            "net_revenue": s.get("net_revenue", 0),
            "overall_grade": s.get("overall_grade", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        intervals = data.get("baseline_vs_actual", {}).get("intervals", [])
        loads = data.get("load_level_performance", [])
        ders = data.get("der_contribution", [])
        return {
            "baseline_actual_line": {
                "type": "line",
                "labels": [iv.get("interval", "") for iv in intervals],
                "series": {
                    "baseline": [iv.get("baseline_kw", 0) for iv in intervals],
                    "actual": [iv.get("actual_kw", 0) for iv in intervals],
                },
            },
            "load_performance_bar": {
                "type": "bar",
                "labels": [ld.get("load_name", "") for ld in loads],
                "series": {
                    "planned": [ld.get("planned_kw", 0) for ld in loads],
                    "actual": [ld.get("actual_kw", 0) for ld in loads],
                },
            },
            "der_utilization_bar": {
                "type": "bar",
                "labels": [d.get("asset_name", "") for d in ders],
                "series": {
                    "dispatched": [d.get("dispatched_kw", 0) for d in ders],
                    "delivered": [d.get("delivered_kw", 0) for d in ders],
                },
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
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".grade-pass .value{color:#198754;}"
            ".grade-fail .value{color:#dc3545;}"
            ".variance-positive{color:#198754;font-weight:600;}"
            ".variance-negative{color:#dc3545;font-weight:600;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
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
