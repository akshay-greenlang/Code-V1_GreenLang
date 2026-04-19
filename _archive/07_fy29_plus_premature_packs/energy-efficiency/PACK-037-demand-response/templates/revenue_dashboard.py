# -*- coding: utf-8 -*-
"""
RevenueDashboardTemplate - DR revenue tracking dashboard for PACK-037.

Generates revenue tracking dashboards for demand response programs showing
capacity payments, energy payments, ancillary service payments, penalties,
program-by-program breakdown, monthly trends, and year-to-date totals.

Sections:
    1. Revenue KPIs
    2. Program Revenue Breakdown
    3. Payment Type Analysis
    4. Monthly Revenue Trend
    5. Penalty Tracking
    6. Revenue Forecast

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 745 (compensation at LMP)
    - ISO/RTO settlement timelines
    - Utility tariff schedules

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


class RevenueDashboardTemplate:
    """
    DR revenue tracking dashboard template.

    Renders revenue tracking dashboards with KPI cards, program breakdowns,
    payment type analysis, monthly trends, penalty tracking, and revenue
    forecasts across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RevenueDashboardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render revenue dashboard as Markdown.

        Args:
            data: Revenue tracking engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_revenue_kpis(data),
            self._md_program_breakdown(data),
            self._md_payment_type_analysis(data),
            self._md_monthly_trend(data),
            self._md_penalty_tracking(data),
            self._md_revenue_forecast(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render revenue dashboard as self-contained HTML.

        Args:
            data: Revenue tracking engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_revenue_kpis(data),
            self._html_program_breakdown(data),
            self._html_payment_type_analysis(data),
            self._html_monthly_trend(data),
            self._html_penalty_tracking(data),
            self._html_revenue_forecast(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DR Revenue Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render revenue dashboard as structured JSON.

        Args:
            data: Revenue tracking engine result data.

        Returns:
            Dict with structured dashboard sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "revenue_dashboard",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "revenue_kpis": self._json_revenue_kpis(data),
            "program_breakdown": data.get("program_breakdown", []),
            "payment_type_analysis": data.get("payment_type_analysis", {}),
            "monthly_trend": data.get("monthly_trend", []),
            "penalty_tracking": data.get("penalty_tracking", {}),
            "revenue_forecast": data.get("revenue_forecast", {}),
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
            f"# Demand Response Revenue Dashboard\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Programs Active:** {data.get('programs_active', 0)}  \n"
            f"**Dashboard Generated:** {ts}  \n"
            f"**Template:** PACK-037 RevenueDashboardTemplate v37.0.0\n\n---"
        )

    def _md_revenue_kpis(self, data: Dict[str, Any]) -> str:
        """Render revenue KPI cards section."""
        kpis = data.get("revenue_kpis", {})
        return (
            "## 1. Revenue KPIs\n\n"
            "| KPI | Value | YTD Target | Status |\n|-----|-------|-----------|--------|\n"
            f"| Gross Revenue | {self._format_currency(kpis.get('gross_revenue', 0))} "
            f"| {self._format_currency(kpis.get('gross_revenue_target', 0))} "
            f"| {kpis.get('gross_status', '-')} |\n"
            f"| Net Revenue | {self._format_currency(kpis.get('net_revenue', 0))} "
            f"| {self._format_currency(kpis.get('net_revenue_target', 0))} "
            f"| {kpis.get('net_status', '-')} |\n"
            f"| Total Penalties | ({self._format_currency(kpis.get('total_penalties', 0))}) "
            f"| ({self._format_currency(kpis.get('penalty_budget', 0))}) "
            f"| {kpis.get('penalty_status', '-')} |\n"
            f"| Events Participated | {kpis.get('events_participated', 0)} "
            f"| {kpis.get('events_target', 0)} "
            f"| {kpis.get('events_status', '-')} |\n"
            f"| Avg Performance Ratio | {self._fmt(kpis.get('avg_performance_ratio', 0), 2)} "
            f"| {self._fmt(kpis.get('performance_target', 0), 2)} "
            f"| {kpis.get('performance_status', '-')} |\n"
            f"| Revenue per kW-yr | {self._format_currency(kpis.get('revenue_per_kw_yr', 0))} "
            f"| {self._format_currency(kpis.get('revenue_per_kw_target', 0))} "
            f"| {kpis.get('per_kw_status', '-')} |"
        )

    def _md_program_breakdown(self, data: Dict[str, Any]) -> str:
        """Render program revenue breakdown section."""
        programs = data.get("program_breakdown", [])
        if not programs:
            return "## 2. Program Revenue Breakdown\n\n_No program data available._"
        lines = [
            "## 2. Program Revenue Breakdown\n",
            "| Program | Gross Revenue | Penalties | Net Revenue | Events | Share (%) |",
            "|---------|-------------:|---------:|-----------:|-------:|----------:|",
        ]
        total_net = sum(p.get("net_revenue", 0) for p in programs)
        for p in programs:
            net = p.get("net_revenue", 0)
            share = (net / total_net * 100) if total_net > 0 else 0
            lines.append(
                f"| {p.get('program_name', '-')} "
                f"| {self._format_currency(p.get('gross_revenue', 0))} "
                f"| ({self._format_currency(p.get('penalties', 0))}) "
                f"| {self._format_currency(net)} "
                f"| {p.get('events', 0)} "
                f"| {self._fmt(share)}% |"
            )
        lines.append(
            f"| **TOTAL** | | | **{self._format_currency(total_net)}** | | **100%** |"
        )
        return "\n".join(lines)

    def _md_payment_type_analysis(self, data: Dict[str, Any]) -> str:
        """Render payment type analysis section."""
        analysis = data.get("payment_type_analysis", {})
        return (
            "## 3. Payment Type Analysis\n\n"
            "| Payment Type | Amount | Share (%) |\n|-------------|-------:|----------:|\n"
            f"| Capacity Payments | {self._format_currency(analysis.get('capacity_total', 0))} "
            f"| {self._fmt(analysis.get('capacity_share_pct', 0))}% |\n"
            f"| Energy Payments | {self._format_currency(analysis.get('energy_total', 0))} "
            f"| {self._fmt(analysis.get('energy_share_pct', 0))}% |\n"
            f"| Ancillary Payments | {self._format_currency(analysis.get('ancillary_total', 0))} "
            f"| {self._fmt(analysis.get('ancillary_share_pct', 0))}% |\n"
            f"| Performance Bonuses | {self._format_currency(analysis.get('bonus_total', 0))} "
            f"| {self._fmt(analysis.get('bonus_share_pct', 0))}% |"
        )

    def _md_monthly_trend(self, data: Dict[str, Any]) -> str:
        """Render monthly revenue trend section."""
        trend = data.get("monthly_trend", [])
        if not trend:
            return "## 4. Monthly Revenue Trend\n\n_No trend data available._"
        lines = [
            "## 4. Monthly Revenue Trend\n",
            "| Month | Gross Revenue | Penalties | Net Revenue | Events | Cumulative |",
            "|-------|-------------:|---------:|-----------:|-------:|-----------:|",
        ]
        for t in trend:
            lines.append(
                f"| {t.get('month', '-')} "
                f"| {self._format_currency(t.get('gross_revenue', 0))} "
                f"| ({self._format_currency(t.get('penalties', 0))}) "
                f"| {self._format_currency(t.get('net_revenue', 0))} "
                f"| {t.get('events', 0)} "
                f"| {self._format_currency(t.get('cumulative_revenue', 0))} |"
            )
        return "\n".join(lines)

    def _md_penalty_tracking(self, data: Dict[str, Any]) -> str:
        """Render penalty tracking section."""
        penalties = data.get("penalty_tracking", {})
        events = penalties.get("penalty_events", [])
        lines = [
            "## 5. Penalty Tracking\n",
            f"**Total Penalties YTD:** {self._format_currency(penalties.get('total_penalties_ytd', 0))}  ",
            f"**Penalty Rate:** {self._fmt(penalties.get('penalty_rate_pct', 0))}%  ",
            f"**Penalty Budget Remaining:** {self._format_currency(penalties.get('budget_remaining', 0))}\n",
        ]
        if events:
            lines.extend([
                "| Event Date | Program | Penalty Amount | Reason |",
                "|-----------|---------|---------------:|--------|",
            ])
            for e in events:
                lines.append(
                    f"| {e.get('event_date', '-')} "
                    f"| {e.get('program', '-')} "
                    f"| {self._format_currency(e.get('penalty_amount', 0))} "
                    f"| {e.get('reason', '-')} |"
                )
        else:
            lines.append("_No penalty events recorded._")
        return "\n".join(lines)

    def _md_revenue_forecast(self, data: Dict[str, Any]) -> str:
        """Render revenue forecast section."""
        forecast = data.get("revenue_forecast", {})
        return (
            "## 6. Revenue Forecast\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Projected Annual Revenue | {self._format_currency(forecast.get('projected_annual', 0))} |\n"
            f"| Best Case | {self._format_currency(forecast.get('best_case', 0))} |\n"
            f"| Worst Case | {self._format_currency(forecast.get('worst_case', 0))} |\n"
            f"| Remaining Events (est.) | {forecast.get('remaining_events', 0)} |\n"
            f"| Remaining Revenue (est.) | {self._format_currency(forecast.get('remaining_revenue', 0))} |\n"
            f"| Forecast Confidence | {forecast.get('confidence', '-')} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render dashboard footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>DR Revenue Dashboard</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'Programs: {data.get("programs_active", 0)}</p>'
        )

    def _html_revenue_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML revenue KPI cards."""
        kpis = data.get("revenue_kpis", {})
        return (
            '<h2>Revenue KPIs</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Gross Revenue</span>'
            f'<span class="value">{self._format_currency(kpis.get("gross_revenue", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Net Revenue</span>'
            f'<span class="value">{self._format_currency(kpis.get("net_revenue", 0))}</span></div>\n'
            f'  <div class="card card-red"><span class="label">Penalties</span>'
            f'<span class="value">({self._format_currency(kpis.get("total_penalties", 0))})</span></div>\n'
            f'  <div class="card"><span class="label">Events</span>'
            f'<span class="value">{kpis.get("events_participated", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Rev/kW-yr</span>'
            f'<span class="value">{self._format_currency(kpis.get("revenue_per_kw_yr", 0))}</span></div>\n'
            '</div>'
        )

    def _html_program_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML program breakdown table."""
        programs = data.get("program_breakdown", [])
        rows = ""
        for p in programs:
            rows += (
                f'<tr><td>{p.get("program_name", "-")}</td>'
                f'<td>{self._format_currency(p.get("gross_revenue", 0))}</td>'
                f'<td>{self._format_currency(p.get("net_revenue", 0))}</td>'
                f'<td>{p.get("events", 0)}</td></tr>\n'
            )
        return (
            '<h2>Program Breakdown</h2>\n'
            '<table>\n<tr><th>Program</th><th>Gross</th>'
            f'<th>Net</th><th>Events</th></tr>\n{rows}</table>'
        )

    def _html_payment_type_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML payment type analysis."""
        a = data.get("payment_type_analysis", {})
        return (
            '<h2>Payment Types</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Capacity</span>'
            f'<span class="value">{self._format_currency(a.get("capacity_total", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Energy</span>'
            f'<span class="value">{self._format_currency(a.get("energy_total", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Ancillary</span>'
            f'<span class="value">{self._format_currency(a.get("ancillary_total", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Bonuses</span>'
            f'<span class="value">{self._format_currency(a.get("bonus_total", 0))}</span></div>\n'
            '</div>'
        )

    def _html_monthly_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML monthly trend table."""
        trend = data.get("monthly_trend", [])
        rows = ""
        for t in trend:
            rows += (
                f'<tr><td>{t.get("month", "-")}</td>'
                f'<td>{self._format_currency(t.get("gross_revenue", 0))}</td>'
                f'<td>{self._format_currency(t.get("net_revenue", 0))}</td>'
                f'<td>{self._format_currency(t.get("cumulative_revenue", 0))}</td></tr>\n'
            )
        return (
            '<h2>Monthly Trend</h2>\n'
            '<table>\n<tr><th>Month</th><th>Gross</th>'
            f'<th>Net</th><th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_penalty_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML penalty tracking."""
        penalties = data.get("penalty_tracking", {})
        return (
            '<h2>Penalty Tracking</h2>\n'
            f'<p>Total Penalties YTD: {self._format_currency(penalties.get("total_penalties_ytd", 0))} | '
            f'Penalty Rate: {self._fmt(penalties.get("penalty_rate_pct", 0))}% | '
            f'Budget Remaining: {self._format_currency(penalties.get("budget_remaining", 0))}</p>'
        )

    def _html_revenue_forecast(self, data: Dict[str, Any]) -> str:
        """Render HTML revenue forecast."""
        f = data.get("revenue_forecast", {})
        return (
            '<h2>Revenue Forecast</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Projected Annual</span>'
            f'<span class="value">{self._format_currency(f.get("projected_annual", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Best Case</span>'
            f'<span class="value">{self._format_currency(f.get("best_case", 0))}</span></div>\n'
            f'  <div class="card card-red"><span class="label">Worst Case</span>'
            f'<span class="value">{self._format_currency(f.get("worst_case", 0))}</span></div>\n'
            '</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_revenue_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON revenue KPIs."""
        kpis = data.get("revenue_kpis", {})
        return {
            "gross_revenue": kpis.get("gross_revenue", 0),
            "net_revenue": kpis.get("net_revenue", 0),
            "total_penalties": kpis.get("total_penalties", 0),
            "events_participated": kpis.get("events_participated", 0),
            "avg_performance_ratio": kpis.get("avg_performance_ratio", 0),
            "revenue_per_kw_yr": kpis.get("revenue_per_kw_yr", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        programs = data.get("program_breakdown", [])
        trend = data.get("monthly_trend", [])
        analysis = data.get("payment_type_analysis", {})
        return {
            "program_revenue_bar": {
                "type": "bar",
                "labels": [p.get("program_name", "") for p in programs],
                "series": {
                    "gross": [p.get("gross_revenue", 0) for p in programs],
                    "penalties": [p.get("penalties", 0) for p in programs],
                    "net": [p.get("net_revenue", 0) for p in programs],
                },
            },
            "monthly_trend_line": {
                "type": "line",
                "labels": [t.get("month", "") for t in trend],
                "series": {
                    "net_revenue": [t.get("net_revenue", 0) for t in trend],
                    "cumulative": [t.get("cumulative_revenue", 0) for t in trend],
                },
            },
            "payment_type_pie": {
                "type": "pie",
                "labels": ["Capacity", "Energy", "Ancillary", "Bonuses"],
                "values": [
                    analysis.get("capacity_total", 0),
                    analysis.get("energy_total", 0),
                    analysis.get("ancillary_total", 0),
                    analysis.get("bonus_total", 0),
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
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
