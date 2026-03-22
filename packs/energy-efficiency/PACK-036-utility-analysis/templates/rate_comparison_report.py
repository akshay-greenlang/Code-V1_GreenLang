# -*- coding: utf-8 -*-
"""
RateComparisonReportTemplate - Rate structure comparison report for PACK-036.

Generates rate analysis reports comparing current tariff against available
alternatives, with annual cost projections under each rate, time-of-use
optimization potential, demand charge comparisons, and ranked
recommendations for rate switching or renegotiation.

Sections:
    1. Header & Rate Summary
    2. Current Rate Analysis
    3. Alternative Rate Comparison
    4. Annual Cost Projections
    5. TOU Optimization
    6. Demand Charge Analysis
    7. Optimal Recommendation
    8. Provenance

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash excluding volatile fields."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {
            k: v for k, v in s.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


class RateComparisonReportTemplate:
    """
    Rate structure comparison report template.

    Renders rate analysis results including current tariff breakdown,
    alternative rate simulations, annual cost projections, TOU
    optimization potential, and optimal rate recommendations across
    markdown, HTML, JSON, and CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RateComparisonReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render rate comparison report as Markdown.

        Args:
            data: Rate comparison data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_current_rate(data),
            self._md_alternatives(data),
            self._md_annual_projections(data),
            self._md_tou_optimization(data),
            self._md_demand_charge(data),
            self._md_recommendation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render rate comparison report as self-contained HTML.

        Args:
            data: Rate comparison data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_current_rate(data),
            self._html_alternatives(data),
            self._html_annual_projections(data),
            self._html_tou_optimization(data),
            self._html_demand_charge(data),
            self._html_recommendation(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Rate Comparison Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render rate comparison report as structured JSON.

        Args:
            data: Rate comparison data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "rate_comparison_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "current_rate": data.get("current_rate", {}),
            "alternatives": data.get("alternatives", []),
            "annual_projections": data.get("annual_projections", []),
            "tou_optimization": data.get("tou_optimization", {}),
            "demand_charge_analysis": data.get("demand_charge_analysis", {}),
            "recommendation": data.get("recommendation", {}),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render rate comparison as CSV with one row per alternative.

        Args:
            data: Rate comparison data from engine processing.

        Returns:
            CSV string with rate alternatives and annual costs.
        """
        self.generated_at = _utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Rate Name", "Rate Type", "Energy Charge (per kWh)",
            "Demand Charge (per kW)", "Fixed Charge (monthly)",
            "Annual Energy Cost", "Annual Demand Cost", "Annual Fixed Cost",
            "Annual Total Cost", "Savings vs Current", "Rank",
        ])
        for alt in data.get("alternatives", []):
            writer.writerow([
                alt.get("rate_name", ""),
                alt.get("rate_type", ""),
                self._fmt_raw(alt.get("energy_charge_per_kwh", 0), 4),
                self._fmt_raw(alt.get("demand_charge_per_kw", 0)),
                self._fmt_raw(alt.get("fixed_charge_monthly", 0)),
                self._fmt_raw(alt.get("annual_energy_cost", 0)),
                self._fmt_raw(alt.get("annual_demand_cost", 0)),
                self._fmt_raw(alt.get("annual_fixed_cost", 0)),
                self._fmt_raw(alt.get("annual_total_cost", 0)),
                self._fmt_raw(alt.get("savings_vs_current", 0)),
                alt.get("rank", ""),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with rate summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("rate_summary", {})
        return (
            "# Rate Comparison Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Account:** {data.get('account_number', '-')}  \n"
            f"**Utility:** {data.get('utility_provider', '-')}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '-')}  \n"
            f"**Current Rate:** {summary.get('current_rate_name', '-')}  \n"
            f"**Current Annual Cost:** {self._fmt_currency(summary.get('current_annual_cost', 0))}  \n"
            f"**Best Alternative Savings:** {self._fmt_currency(summary.get('best_savings', 0))}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 RateComparisonReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_current_rate(self, data: Dict[str, Any]) -> str:
        """Render current rate analysis section."""
        rate = data.get("current_rate", {})
        components = rate.get("components", [])
        lines = [
            "## 1. Current Rate Analysis\n",
            f"**Rate Name:** {rate.get('name', '-')}  ",
            f"**Rate Schedule:** {rate.get('schedule', '-')}  ",
            f"**Rate Type:** {rate.get('type', '-')}  ",
            f"**Effective Date:** {rate.get('effective_date', '-')}  ",
            f"**Contract Expiry:** {rate.get('contract_expiry', '-')}\n",
            "### Rate Components\n",
            "| Component | Rate | Units | Monthly Avg | Annual Total |",
            "|-----------|------|-------|------------|-------------|",
        ]
        for c in components:
            lines.append(
                f"| {c.get('component', '-')} "
                f"| {self._fmt(c.get('rate', 0), 4)} "
                f"| {c.get('units', '-')} "
                f"| {self._fmt_currency(c.get('monthly_avg', 0))} "
                f"| {self._fmt_currency(c.get('annual_total', 0))} |"
            )
        return "\n".join(lines)

    def _md_alternatives(self, data: Dict[str, Any]) -> str:
        """Render alternative rate comparison section."""
        alts = data.get("alternatives", [])
        if not alts:
            return "## 2. Alternative Rate Comparison\n\n_No alternatives available._"
        lines = [
            "## 2. Alternative Rate Comparison\n",
            "| Rank | Rate Name | Type | Annual Cost | Savings | Savings (%) |",
            "|------|-----------|------|------------|---------|------------|",
        ]
        for a in alts:
            lines.append(
                f"| {a.get('rank', '-')} "
                f"| {a.get('rate_name', '-')} "
                f"| {a.get('rate_type', '-')} "
                f"| {self._fmt_currency(a.get('annual_total_cost', 0))} "
                f"| {self._fmt_currency(a.get('savings_vs_current', 0))} "
                f"| {self._fmt(a.get('savings_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_annual_projections(self, data: Dict[str, Any]) -> str:
        """Render annual cost projections by month."""
        projections = data.get("annual_projections", [])
        if not projections:
            return "## 3. Annual Cost Projections\n\n_No projections available._"
        rate_names = list(projections[0].get("costs_by_rate", {}).keys()) if projections else []
        header_cols = " | ".join(rate_names)
        separator_cols = " | ".join(["------" for _ in rate_names])
        lines = [
            "## 3. Annual Cost Projections\n",
            f"| Month | {header_cols} |",
            f"|-------|{separator_cols}|",
        ]
        for p in projections:
            costs = p.get("costs_by_rate", {})
            cost_cols = " | ".join(
                self._fmt_currency(costs.get(name, 0)) for name in rate_names
            )
            lines.append(f"| {p.get('month', '-')} | {cost_cols} |")
        return "\n".join(lines)

    def _md_tou_optimization(self, data: Dict[str, Any]) -> str:
        """Render TOU optimization analysis section."""
        tou = data.get("tou_optimization", {})
        periods = tou.get("periods", [])
        lines = [
            "## 4. Time-of-Use Optimization\n",
            f"**Load Shift Potential:** {self._fmt(tou.get('load_shift_potential_kwh', 0), 0)} kWh/yr  ",
            f"**TOU Savings Potential:** {self._fmt_currency(tou.get('tou_savings_potential', 0))}  ",
            f"**Current On-Peak Ratio:** {self._fmt(tou.get('on_peak_ratio_pct', 0))}%  ",
            f"**Optimal On-Peak Ratio:** {self._fmt(tou.get('optimal_on_peak_ratio_pct', 0))}%\n",
        ]
        if periods:
            lines.extend([
                "| Period | Hours | Rate | Current kWh | Share (%) | Cost |",
                "|--------|-------|------|-------------|----------|------|",
            ])
            for p in periods:
                lines.append(
                    f"| {p.get('period', '-')} "
                    f"| {p.get('hours', '-')} "
                    f"| {self._fmt(p.get('rate', 0), 4)} "
                    f"| {self._fmt(p.get('kwh', 0), 0)} "
                    f"| {self._fmt(p.get('share_pct', 0))}% "
                    f"| {self._fmt_currency(p.get('cost', 0))} |"
                )
        return "\n".join(lines)

    def _md_demand_charge(self, data: Dict[str, Any]) -> str:
        """Render demand charge analysis section."""
        dc = data.get("demand_charge_analysis", {})
        monthly = dc.get("monthly_demand", [])
        lines = [
            "## 5. Demand Charge Analysis\n",
            f"**Peak Demand (12-mo):** {self._fmt(dc.get('peak_demand_kw', 0))} kW  ",
            f"**Average Demand:** {self._fmt(dc.get('avg_demand_kw', 0))} kW  ",
            f"**Demand Ratchet Level:** {self._fmt(dc.get('ratchet_kw', 0))} kW  ",
            f"**Annual Demand Cost:** {self._fmt_currency(dc.get('annual_demand_cost', 0))}  ",
            f"**Demand Charge Rate:** {self._fmt(dc.get('demand_rate_per_kw', 0))} /kW\n",
        ]
        if monthly:
            lines.extend([
                "| Month | Peak Demand (kW) | Billed Demand (kW) | Demand Cost |",
                "|-------|-----------------|-------------------|------------|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('peak_kw', 0))} "
                    f"| {self._fmt(m.get('billed_kw', 0))} "
                    f"| {self._fmt_currency(m.get('cost', 0))} |"
                )
        return "\n".join(lines)

    def _md_recommendation(self, data: Dict[str, Any]) -> str:
        """Render optimal recommendation section."""
        rec = data.get("recommendation", {})
        steps = rec.get("implementation_steps", [])
        lines = [
            "## 6. Optimal Recommendation\n",
            f"**Recommended Rate:** {rec.get('recommended_rate', '-')}  ",
            f"**Projected Annual Cost:** {self._fmt_currency(rec.get('projected_annual_cost', 0))}  ",
            f"**Annual Savings:** {self._fmt_currency(rec.get('annual_savings', 0))}  ",
            f"**Savings Percentage:** {self._fmt(rec.get('savings_pct', 0))}%  ",
            f"**Confidence Level:** {rec.get('confidence_level', '-')}  ",
            f"**Switch Date Recommendation:** {rec.get('switch_date', '-')}\n",
        ]
        if rec.get("rationale"):
            lines.append(f"**Rationale:** {rec.get('rationale', '')}\n")
        if steps:
            lines.append("### Implementation Steps\n")
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Rate comparisons are estimates based on historical usage patterns. "
            "Actual costs may vary with future consumption.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("rate_summary", {})
        return (
            f'<h1>Rate Comparison Report</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Account: {data.get("account_number", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Current Rate</span>'
            f'<span class="value">{summary.get("current_rate_name", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Current Annual Cost</span>'
            f'<span class="value">{self._fmt_currency(summary.get("current_annual_cost", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Best Savings</span>'
            f'<span class="value">{self._fmt_currency(summary.get("best_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Alternatives Analyzed</span>'
            f'<span class="value">{summary.get("alternatives_count", 0)}</span></div>\n'
            f'</div>'
        )

    def _html_current_rate(self, data: Dict[str, Any]) -> str:
        """Render HTML current rate analysis."""
        rate = data.get("current_rate", {})
        components = rate.get("components", [])
        rows = ""
        for c in components:
            rows += (
                f'<tr><td>{c.get("component", "-")}</td>'
                f'<td>{self._fmt(c.get("rate", 0), 4)}</td>'
                f'<td>{c.get("units", "-")}</td>'
                f'<td>{self._fmt_currency(c.get("monthly_avg", 0))}</td>'
                f'<td>{self._fmt_currency(c.get("annual_total", 0))}</td></tr>\n'
            )
        return (
            '<h2>Current Rate Analysis</h2>\n'
            f'<div class="info-box"><p><strong>Rate:</strong> {rate.get("name", "-")} | '
            f'<strong>Type:</strong> {rate.get("type", "-")} | '
            f'<strong>Expiry:</strong> {rate.get("contract_expiry", "-")}</p></div>\n'
            '<table>\n<tr><th>Component</th><th>Rate</th><th>Units</th>'
            f'<th>Monthly Avg</th><th>Annual Total</th></tr>\n{rows}</table>'
        )

    def _html_alternatives(self, data: Dict[str, Any]) -> str:
        """Render HTML alternative rate comparison table."""
        alts = data.get("alternatives", [])
        rows = ""
        for a in alts:
            savings = a.get("savings_vs_current", 0)
            cls = "card-green" if savings > 0 else ""
            rows += (
                f'<tr class="{cls}"><td>{a.get("rank", "-")}</td>'
                f'<td>{a.get("rate_name", "-")}</td>'
                f'<td>{a.get("rate_type", "-")}</td>'
                f'<td>{self._fmt_currency(a.get("annual_total_cost", 0))}</td>'
                f'<td>{self._fmt_currency(savings)}</td>'
                f'<td>{self._fmt(a.get("savings_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Alternative Rate Comparison</h2>\n'
            '<table>\n<tr><th>Rank</th><th>Rate Name</th><th>Type</th>'
            '<th>Annual Cost</th><th>Savings</th>'
            f'<th>Savings (%)</th></tr>\n{rows}</table>'
        )

    def _html_annual_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML annual cost projections."""
        projections = data.get("annual_projections", [])
        if not projections:
            return '<h2>Annual Cost Projections</h2>\n<p>No projections available.</p>'
        rate_names = list(projections[0].get("costs_by_rate", {}).keys())
        header = "".join(f'<th>{n}</th>' for n in rate_names)
        rows = ""
        for p in projections:
            costs = p.get("costs_by_rate", {})
            cells = "".join(
                f'<td>{self._fmt_currency(costs.get(n, 0))}</td>' for n in rate_names
            )
            rows += f'<tr><td>{p.get("month", "-")}</td>{cells}</tr>\n'
        return (
            '<h2>Annual Cost Projections</h2>\n'
            f'<table>\n<tr><th>Month</th>{header}</tr>\n{rows}</table>'
        )

    def _html_tou_optimization(self, data: Dict[str, Any]) -> str:
        """Render HTML TOU optimization section."""
        tou = data.get("tou_optimization", {})
        periods = tou.get("periods", [])
        rows = ""
        for p in periods:
            rows += (
                f'<tr><td>{p.get("period", "-")}</td>'
                f'<td>{p.get("hours", "-")}</td>'
                f'<td>{self._fmt(p.get("rate", 0), 4)}</td>'
                f'<td>{self._fmt(p.get("kwh", 0), 0)}</td>'
                f'<td>{self._fmt(p.get("share_pct", 0))}%</td>'
                f'<td>{self._fmt_currency(p.get("cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Time-of-Use Optimization</h2>\n'
            f'<div class="info-box"><p>Load Shift Potential: '
            f'{self._fmt(tou.get("load_shift_potential_kwh", 0), 0)} kWh/yr | '
            f'TOU Savings: {self._fmt_currency(tou.get("tou_savings_potential", 0))}</p></div>\n'
            '<table>\n<tr><th>Period</th><th>Hours</th><th>Rate</th>'
            f'<th>Current kWh</th><th>Share</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_demand_charge(self, data: Dict[str, Any]) -> str:
        """Render HTML demand charge analysis."""
        dc = data.get("demand_charge_analysis", {})
        monthly = dc.get("monthly_demand", [])
        rows = ""
        for m in monthly:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("peak_kw", 0))}</td>'
                f'<td>{self._fmt(m.get("billed_kw", 0))}</td>'
                f'<td>{self._fmt_currency(m.get("cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Demand Charge Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Peak Demand</span>'
            f'<span class="value">{self._fmt(dc.get("peak_demand_kw", 0))} kW</span></div>\n'
            f'  <div class="card"><span class="label">Ratchet Level</span>'
            f'<span class="value">{self._fmt(dc.get("ratchet_kw", 0))} kW</span></div>\n'
            f'  <div class="card"><span class="label">Annual Demand Cost</span>'
            f'<span class="value">{self._fmt_currency(dc.get("annual_demand_cost", 0))}</span></div>\n'
            '</div>\n'
            '<table>\n<tr><th>Month</th><th>Peak (kW)</th>'
            f'<th>Billed (kW)</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_recommendation(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendation section."""
        rec = data.get("recommendation", {})
        steps = rec.get("implementation_steps", [])
        items = "".join(f'<li>{s}</li>\n' for s in steps)
        return (
            '<h2>Optimal Recommendation</h2>\n'
            '<div class="info-box">'
            f'<p><strong>Recommended Rate:</strong> {rec.get("recommended_rate", "-")} | '
            f'<strong>Annual Savings:</strong> {self._fmt_currency(rec.get("annual_savings", 0))} '
            f'({self._fmt(rec.get("savings_pct", 0))}%) | '
            f'<strong>Confidence:</strong> {rec.get("confidence_level", "-")}</p>'
            f'<p>{rec.get("rationale", "")}</p>'
            '</div>\n'
            f'<ol>\n{items}</ol>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        alts = data.get("alternatives", [])
        projections = data.get("annual_projections", [])
        tou = data.get("tou_optimization", {}).get("periods", [])
        rate_names = []
        if projections:
            rate_names = list(projections[0].get("costs_by_rate", {}).keys())
        return {
            "cost_comparison_bar": {
                "type": "bar",
                "labels": [a.get("rate_name", "") for a in alts],
                "values": [a.get("annual_total_cost", 0) for a in alts],
            },
            "monthly_cost_line": {
                "type": "line",
                "labels": [p.get("month", "") for p in projections],
                "series": {
                    name: [p.get("costs_by_rate", {}).get(name, 0) for p in projections]
                    for name in rate_names
                },
            },
            "tou_pie": {
                "type": "pie",
                "labels": [p.get("period", "") for p in tou],
                "values": [p.get("kwh", 0) for p in tou],
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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:160px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _fmt_raw(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value without commas (for CSV)."""
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}"
        return str(val)

    def _fmt_currency(self, val: Any, symbol: str = "") -> str:
        """Format a currency value."""
        sym = symbol or self.config.get("currency_symbol", "EUR")
        if isinstance(val, (int, float)):
            return f"{sym} {val:,.2f}"
        return f"{sym} {val}"

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage."""
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
