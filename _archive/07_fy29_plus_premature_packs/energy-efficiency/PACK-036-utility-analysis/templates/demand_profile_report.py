# -*- coding: utf-8 -*-
"""
DemandProfileReportTemplate - Load profile and demand analysis report for PACK-036.

Generates demand profile reports covering peak demand identification,
load factor analysis, load duration curves, peak event cataloguing,
demand response opportunity assessment, and power factor correction
recommendations. Designed for facility engineers and energy procurement
teams to optimize demand-side management.

Sections:
    1. Header & Peak Summary
    2. Load Profile Overview
    3. Load Duration Curve
    4. Peak Event Analysis
    5. Power Factor Analysis
    6. Demand Response Opportunities
    7. Recommendations
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"

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

class DemandProfileReportTemplate:
    """
    Load profile and demand analysis report template.

    Renders demand profile reports including peak demand summary,
    load duration curves, peak event analysis, power factor metrics,
    and demand response opportunity assessments across markdown,
    HTML, JSON, and CSV formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DemandProfileReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render demand profile report as Markdown.

        Args:
            data: Demand profile data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_load_profile(data),
            self._md_load_duration(data),
            self._md_peak_events(data),
            self._md_power_factor(data),
            self._md_demand_response(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render demand profile report as self-contained HTML.

        Args:
            data: Demand profile data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_load_profile(data),
            self._html_load_duration(data),
            self._html_peak_events(data),
            self._html_power_factor(data),
            self._html_demand_response(data),
            self._html_recommendations(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Demand Profile Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render demand profile report as structured JSON.

        Args:
            data: Demand profile data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "demand_profile_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "peak_summary": data.get("peak_summary", {}),
            "load_profile": data.get("load_profile", {}),
            "load_duration_curve": data.get("load_duration_curve", []),
            "peak_events": data.get("peak_events", []),
            "power_factor": data.get("power_factor", {}),
            "demand_response": data.get("demand_response", {}),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render peak events as CSV.

        Args:
            data: Demand profile data from engine processing.

        Returns:
            CSV string with one row per peak event.
        """
        self.generated_at = utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Event Date", "Event Time", "Peak Demand (kW)",
            "Duration (min)", "Coincident Peak", "Temperature (C)",
            "Demand Charge Impact", "Category",
        ])
        for event in data.get("peak_events", []):
            writer.writerow([
                event.get("date", ""),
                event.get("time", ""),
                self._fmt_raw(event.get("peak_kw", 0)),
                event.get("duration_min", ""),
                event.get("coincident_peak", ""),
                self._fmt_raw(event.get("temperature_c", 0), 1),
                self._fmt_raw(event.get("demand_charge_impact", 0)),
                event.get("category", ""),
            ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with peak summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        ps = data.get("peak_summary", {})
        return (
            "# Demand Profile Report\n\n"
            f"**Facility:** {data.get('facility_name', '-')}  \n"
            f"**Account:** {data.get('account_number', '-')}  \n"
            f"**Analysis Period:** {data.get('analysis_period', '-')}  \n"
            f"**Peak Demand:** {self._fmt(ps.get('peak_demand_kw', 0))} kW  \n"
            f"**Average Demand:** {self._fmt(ps.get('avg_demand_kw', 0))} kW  \n"
            f"**Load Factor:** {self._fmt(ps.get('load_factor_pct', 0))}%  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 DemandProfileReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_load_profile(self, data: Dict[str, Any]) -> str:
        """Render load profile overview section."""
        lp = data.get("load_profile", {})
        monthly = lp.get("monthly_profile", [])
        lines = [
            "## 1. Load Profile Overview\n",
            f"**Total Consumption:** {self._fmt(lp.get('total_kwh', 0), 0)} kWh  ",
            f"**Peak Demand:** {self._fmt(lp.get('peak_kw', 0))} kW  ",
            f"**Minimum Demand:** {self._fmt(lp.get('min_kw', 0))} kW  ",
            f"**Average Demand:** {self._fmt(lp.get('avg_kw', 0))} kW  ",
            f"**Load Factor:** {self._fmt(lp.get('load_factor_pct', 0))}%  ",
            f"**Peak-to-Average Ratio:** {self._fmt(lp.get('peak_to_avg_ratio', 0), 2)}\n",
        ]
        if monthly:
            lines.extend([
                "| Month | Peak (kW) | Avg (kW) | Min (kW) | Load Factor (%) | kWh |",
                "|-------|----------|---------|---------|----------------|-----|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('peak_kw', 0))} "
                    f"| {self._fmt(m.get('avg_kw', 0))} "
                    f"| {self._fmt(m.get('min_kw', 0))} "
                    f"| {self._fmt(m.get('load_factor_pct', 0))}% "
                    f"| {self._fmt(m.get('kwh', 0), 0)} |"
                )
        return "\n".join(lines)

    def _md_load_duration(self, data: Dict[str, Any]) -> str:
        """Render load duration curve section."""
        ldc = data.get("load_duration_curve", [])
        if not ldc:
            return "## 2. Load Duration Curve\n\n_No load duration data available._"
        lines = [
            "## 2. Load Duration Curve\n",
            "| Percentile (%) | Demand (kW) | Hours Above |",
            "|---------------|------------|------------|",
        ]
        for point in ldc:
            lines.append(
                f"| {self._fmt(point.get('percentile', 0), 0)} "
                f"| {self._fmt(point.get('demand_kw', 0))} "
                f"| {self._fmt(point.get('hours_above', 0), 0)} |"
            )
        return "\n".join(lines)

    def _md_peak_events(self, data: Dict[str, Any]) -> str:
        """Render peak event analysis section."""
        events = data.get("peak_events", [])
        if not events:
            return "## 3. Peak Event Analysis\n\n_No peak events recorded._"
        lines = [
            "## 3. Peak Event Analysis\n",
            "| # | Date | Time | Peak (kW) | Duration | Coincident | Temp (C) | Category |",
            "|---|------|------|----------|----------|-----------|----------|----------|",
        ]
        for i, e in enumerate(events, 1):
            lines.append(
                f"| {i} | {e.get('date', '-')} "
                f"| {e.get('time', '-')} "
                f"| {self._fmt(e.get('peak_kw', 0))} "
                f"| {e.get('duration_min', '-')} min "
                f"| {e.get('coincident_peak', '-')} "
                f"| {self._fmt(e.get('temperature_c', 0), 1)} "
                f"| {e.get('category', '-')} |"
            )
        return "\n".join(lines)

    def _md_power_factor(self, data: Dict[str, Any]) -> str:
        """Render power factor analysis section."""
        pf = data.get("power_factor", {})
        monthly_pf = pf.get("monthly", [])
        lines = [
            "## 4. Power Factor Analysis\n",
            f"**Average Power Factor:** {self._fmt(pf.get('avg_power_factor', 0), 3)}  ",
            f"**Minimum Power Factor:** {self._fmt(pf.get('min_power_factor', 0), 3)}  ",
            f"**Target Power Factor:** {self._fmt(pf.get('target_power_factor', 0.95), 3)}  ",
            f"**Power Factor Penalty:** {self._fmt_currency(pf.get('annual_penalty', 0))}  ",
            f"**Correction kVAR Required:** {self._fmt(pf.get('correction_kvar', 0), 0)} kVAR  ",
            f"**Capacitor Bank Cost Est.:** {self._fmt_currency(pf.get('correction_cost_est', 0))}  ",
            f"**Payback Period:** {self._fmt(pf.get('payback_months', 0), 0)} months\n",
        ]
        if monthly_pf:
            lines.extend([
                "| Month | Power Factor | kVAR | Penalty |",
                "|-------|-------------|------|---------|",
            ])
            for m in monthly_pf:
                pf_val = m.get("power_factor", 0)
                marker = " *" if pf_val < pf.get("target_power_factor", 0.95) else ""
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(pf_val, 3)}{marker} "
                    f"| {self._fmt(m.get('kvar', 0), 0)} "
                    f"| {self._fmt_currency(m.get('penalty', 0))} |"
                )
        return "\n".join(lines)

    def _md_demand_response(self, data: Dict[str, Any]) -> str:
        """Render demand response opportunities section."""
        dr = data.get("demand_response", {})
        programs = dr.get("programs", [])
        lines = [
            "## 5. Demand Response Opportunities\n",
            f"**Curtailable Load:** {self._fmt(dr.get('curtailable_kw', 0))} kW  ",
            f"**DR Revenue Potential:** {self._fmt_currency(dr.get('revenue_potential', 0))} /yr  ",
            f"**Peak Shaving Potential:** {self._fmt(dr.get('peak_shaving_kw', 0))} kW  ",
            f"**Annual Savings from DR:** {self._fmt_currency(dr.get('annual_savings', 0))}\n",
        ]
        if programs:
            lines.extend([
                "| Program | Type | Capacity (kW) | Revenue (annual) | Commitment |",
                "|---------|------|-------------|-----------------|-----------|",
            ])
            for p in programs:
                lines.append(
                    f"| {p.get('name', '-')} "
                    f"| {p.get('type', '-')} "
                    f"| {self._fmt(p.get('capacity_kw', 0))} "
                    f"| {self._fmt_currency(p.get('revenue', 0))} "
                    f"| {p.get('commitment', '-')} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 6. Recommendations\n\n_No specific recommendations._"
        lines = [
            "## 6. Recommendations\n",
            "| # | Recommendation | Priority | Savings Potential | Payback |",
            "|---|---------------|----------|------------------|---------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('recommendation', '-')} "
                f"| {r.get('priority', '-')} "
                f"| {self._fmt_currency(r.get('savings_potential', 0))} "
                f"| {r.get('payback', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Demand analysis based on interval meter data. "
            "Power factor values marked with * are below target.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with peak summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        ps = data.get("peak_summary", {})
        lf = ps.get("load_factor_pct", 0)
        lf_cls = "card-green" if lf >= 60 else ("card-red" if lf < 40 else "")
        return (
            f'<h1>Demand Profile Report</h1>\n'
            f'<p class="subtitle">Facility: {data.get("facility_name", "-")} | '
            f'Period: {data.get("analysis_period", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Peak Demand</span>'
            f'<span class="value">{self._fmt(ps.get("peak_demand_kw", 0))} kW</span></div>\n'
            f'  <div class="card"><span class="label">Avg Demand</span>'
            f'<span class="value">{self._fmt(ps.get("avg_demand_kw", 0))} kW</span></div>\n'
            f'  <div class="card {lf_cls}"><span class="label">Load Factor</span>'
            f'<span class="value">{self._fmt(lf)}%</span></div>\n'
            f'  <div class="card"><span class="label">Peak Events</span>'
            f'<span class="value">{len(data.get("peak_events", []))}</span></div>\n'
            f'</div>'
        )

    def _html_load_profile(self, data: Dict[str, Any]) -> str:
        """Render HTML load profile section."""
        lp = data.get("load_profile", {})
        monthly = lp.get("monthly_profile", [])
        rows = ""
        for m in monthly:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("peak_kw", 0))}</td>'
                f'<td>{self._fmt(m.get("avg_kw", 0))}</td>'
                f'<td>{self._fmt(m.get("load_factor_pct", 0))}%</td>'
                f'<td>{self._fmt(m.get("kwh", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Load Profile Overview</h2>\n'
            '<table>\n<tr><th>Month</th><th>Peak (kW)</th><th>Avg (kW)</th>'
            f'<th>Load Factor</th><th>kWh</th></tr>\n{rows}</table>'
        )

    def _html_load_duration(self, data: Dict[str, Any]) -> str:
        """Render HTML load duration curve section."""
        ldc = data.get("load_duration_curve", [])
        rows = ""
        for point in ldc:
            rows += (
                f'<tr><td>{self._fmt(point.get("percentile", 0), 0)}%</td>'
                f'<td>{self._fmt(point.get("demand_kw", 0))}</td>'
                f'<td>{self._fmt(point.get("hours_above", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Load Duration Curve</h2>\n'
            '<table>\n<tr><th>Percentile</th><th>Demand (kW)</th>'
            f'<th>Hours Above</th></tr>\n{rows}</table>'
        )

    def _html_peak_events(self, data: Dict[str, Any]) -> str:
        """Render HTML peak events section."""
        events = data.get("peak_events", [])
        rows = ""
        for e in events:
            rows += (
                f'<tr><td>{e.get("date", "-")}</td>'
                f'<td>{e.get("time", "-")}</td>'
                f'<td>{self._fmt(e.get("peak_kw", 0))}</td>'
                f'<td>{e.get("duration_min", "-")} min</td>'
                f'<td>{e.get("coincident_peak", "-")}</td>'
                f'<td>{e.get("category", "-")}</td></tr>\n'
            )
        return (
            '<h2>Peak Event Analysis</h2>\n'
            '<table>\n<tr><th>Date</th><th>Time</th><th>Peak (kW)</th>'
            '<th>Duration</th><th>Coincident</th>'
            f'<th>Category</th></tr>\n{rows}</table>'
        )

    def _html_power_factor(self, data: Dict[str, Any]) -> str:
        """Render HTML power factor section."""
        pf = data.get("power_factor", {})
        avg_pf = pf.get("avg_power_factor", 0)
        pf_cls = "card-green" if avg_pf >= 0.95 else "card-red"
        return (
            '<h2>Power Factor Analysis</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card {pf_cls}"><span class="label">Avg Power Factor</span>'
            f'<span class="value">{self._fmt(avg_pf, 3)}</span></div>\n'
            f'  <div class="card"><span class="label">Correction kVAR</span>'
            f'<span class="value">{self._fmt(pf.get("correction_kvar", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Penalty</span>'
            f'<span class="value">{self._fmt_currency(pf.get("annual_penalty", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(pf.get("payback_months", 0), 0)} mo</span></div>\n'
            '</div>'
        )

    def _html_demand_response(self, data: Dict[str, Any]) -> str:
        """Render HTML demand response section."""
        dr = data.get("demand_response", {})
        programs = dr.get("programs", [])
        rows = ""
        for p in programs:
            rows += (
                f'<tr><td>{p.get("name", "-")}</td>'
                f'<td>{p.get("type", "-")}</td>'
                f'<td>{self._fmt(p.get("capacity_kw", 0))}</td>'
                f'<td>{self._fmt_currency(p.get("revenue", 0))}</td>'
                f'<td>{p.get("commitment", "-")}</td></tr>\n'
            )
        return (
            '<h2>Demand Response Opportunities</h2>\n'
            f'<div class="info-box"><p>Curtailable Load: '
            f'{self._fmt(dr.get("curtailable_kw", 0))} kW | '
            f'Revenue Potential: {self._fmt_currency(dr.get("revenue_potential", 0))}/yr</p></div>\n'
            '<table>\n<tr><th>Program</th><th>Type</th><th>Capacity (kW)</th>'
            f'<th>Revenue</th><th>Commitment</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations section."""
        recs = data.get("recommendations", [])
        items = "".join(
            f'<li><strong>[{r.get("priority", "-")}]</strong> '
            f'{r.get("recommendation", "-")} '
            f'(Savings: {self._fmt_currency(r.get("savings_potential", 0))} | '
            f'Payback: {r.get("payback", "-")})</li>\n'
            for r in recs
        )
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        monthly = data.get("load_profile", {}).get("monthly_profile", [])
        ldc = data.get("load_duration_curve", [])
        return {
            "monthly_demand_bar": {
                "type": "bar",
                "labels": [m.get("month", "") for m in monthly],
                "series": {
                    "peak_kw": [m.get("peak_kw", 0) for m in monthly],
                    "avg_kw": [m.get("avg_kw", 0) for m in monthly],
                },
            },
            "load_duration_line": {
                "type": "line",
                "labels": [p.get("percentile", 0) for p in ldc],
                "values": [p.get("demand_kw", 0) for p in ldc],
            },
            "load_factor_trend": {
                "type": "line",
                "labels": [m.get("month", "") for m in monthly],
                "values": [m.get("load_factor_pct", 0) for m in monthly],
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
