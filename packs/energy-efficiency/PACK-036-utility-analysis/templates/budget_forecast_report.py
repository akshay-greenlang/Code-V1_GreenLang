# -*- coding: utf-8 -*-
"""
BudgetForecastReportTemplate - Multi-scenario budget forecast report for PACK-036.

Generates budget forecast reports with monthly projections, confidence
intervals, multi-scenario comparisons (base/optimistic/pessimistic),
variance decomposition, weather normalization adjustments, and rate
escalation impacts. Designed for finance and energy management teams
planning annual utility budgets.

Sections:
    1. Header & Forecast Summary
    2. Historical Baseline
    3. Monthly Projections
    4. Scenario Comparison
    5. Confidence Intervals
    6. Variance Decomposition
    7. Budget Recommendation
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

class BudgetForecastReportTemplate:
    """
    Multi-scenario budget forecast report template.

    Renders budget forecast results including monthly projections,
    confidence intervals, scenario analysis, variance decomposition,
    and budget recommendations across markdown, HTML, JSON, and CSV
    formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    SCENARIO_LABELS: Dict[str, str] = {
        "base": "Base Case",
        "optimistic": "Optimistic",
        "pessimistic": "Pessimistic",
        "stretch": "Stretch Target",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BudgetForecastReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render budget forecast report as Markdown.

        Args:
            data: Budget forecast data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_historical_baseline(data),
            self._md_monthly_projections(data),
            self._md_scenario_comparison(data),
            self._md_confidence_intervals(data),
            self._md_variance_decomposition(data),
            self._md_budget_recommendation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render budget forecast report as self-contained HTML.

        Args:
            data: Budget forecast data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_historical_baseline(data),
            self._html_monthly_projections(data),
            self._html_scenario_comparison(data),
            self._html_confidence_intervals(data),
            self._html_variance_decomposition(data),
            self._html_budget_recommendation(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Budget Forecast Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render budget forecast report as structured JSON.

        Args:
            data: Budget forecast data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "budget_forecast_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "forecast_summary": data.get("forecast_summary", {}),
            "historical_baseline": data.get("historical_baseline", {}),
            "monthly_projections": data.get("monthly_projections", []),
            "scenarios": data.get("scenarios", []),
            "confidence_intervals": data.get("confidence_intervals", {}),
            "variance_decomposition": data.get("variance_decomposition", []),
            "recommendation": data.get("recommendation", {}),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    def render_csv(self, data: Dict[str, Any]) -> str:
        """Render monthly projections as CSV.

        Args:
            data: Budget forecast data from engine processing.

        Returns:
            CSV string with one row per month per scenario.
        """
        self.generated_at = utcnow()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Month", "Scenario", "Consumption (kWh)", "Demand (kW)",
            "Energy Cost", "Demand Cost", "Total Cost",
            "Lower Bound (90%)", "Upper Bound (90%)",
        ])
        for proj in data.get("monthly_projections", []):
            month = proj.get("month", "")
            for scenario in proj.get("scenarios", []):
                writer.writerow([
                    month,
                    scenario.get("name", ""),
                    self._fmt_raw(scenario.get("consumption_kwh", 0), 0),
                    self._fmt_raw(scenario.get("demand_kw", 0)),
                    self._fmt_raw(scenario.get("energy_cost", 0)),
                    self._fmt_raw(scenario.get("demand_cost", 0)),
                    self._fmt_raw(scenario.get("total_cost", 0)),
                    self._fmt_raw(scenario.get("lower_bound_90", 0)),
                    self._fmt_raw(scenario.get("upper_bound_90", 0)),
                ])
        return output.getvalue()

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with forecast summary."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("forecast_summary", {})
        return (
            "# Budget Forecast Report\n\n"
            f"**Organization:** {data.get('organization_name', '-')}  \n"
            f"**Forecast Period:** {data.get('forecast_period', '-')}  \n"
            f"**Base Forecast (Annual):** {self._fmt_currency(summary.get('base_annual', 0))}  \n"
            f"**Optimistic:** {self._fmt_currency(summary.get('optimistic_annual', 0))}  \n"
            f"**Pessimistic:** {self._fmt_currency(summary.get('pessimistic_annual', 0))}  \n"
            f"**Confidence Level:** {summary.get('confidence_level', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-036 BudgetForecastReportTemplate v{_MODULE_VERSION}\n\n---"
        )

    def _md_historical_baseline(self, data: Dict[str, Any]) -> str:
        """Render historical baseline section."""
        hb = data.get("historical_baseline", {})
        years = hb.get("annual_history", [])
        lines = [
            "## 1. Historical Baseline\n",
            f"**Baseline Period:** {hb.get('baseline_period', '-')}  ",
            f"**Avg Annual Cost:** {self._fmt_currency(hb.get('avg_annual_cost', 0))}  ",
            f"**Avg Annual Consumption:** {self._fmt(hb.get('avg_annual_kwh', 0), 0)} kWh  ",
            f"**CAGR (Cost):** {self._fmt(hb.get('cost_cagr_pct', 0))}%  ",
            f"**Weather Normalized:** {hb.get('weather_normalized', 'No')}\n",
        ]
        if years:
            lines.extend([
                "| Year | Consumption (kWh) | Cost | vs Previous |",
                "|------|------------------|------|-------------|",
            ])
            for y in years:
                lines.append(
                    f"| {y.get('year', '-')} "
                    f"| {self._fmt(y.get('kwh', 0), 0)} "
                    f"| {self._fmt_currency(y.get('cost', 0))} "
                    f"| {self._fmt(y.get('yoy_change_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_monthly_projections(self, data: Dict[str, Any]) -> str:
        """Render monthly projections section (base case)."""
        projections = data.get("monthly_projections", [])
        if not projections:
            return "## 2. Monthly Projections\n\n_No projections available._"
        lines = [
            "## 2. Monthly Projections (Base Case)\n",
            "| Month | kWh | Demand (kW) | Energy Cost | Demand Cost | Total |",
            "|-------|-----|------------|------------|------------|-------|",
        ]
        annual_total = 0
        for proj in projections:
            base = {}
            for sc in proj.get("scenarios", []):
                if sc.get("name", "").lower() == "base":
                    base = sc
                    break
            if not base and proj.get("scenarios"):
                base = proj["scenarios"][0]
            total = base.get("total_cost", 0)
            annual_total += total
            lines.append(
                f"| {proj.get('month', '-')} "
                f"| {self._fmt(base.get('consumption_kwh', 0), 0)} "
                f"| {self._fmt(base.get('demand_kw', 0))} "
                f"| {self._fmt_currency(base.get('energy_cost', 0))} "
                f"| {self._fmt_currency(base.get('demand_cost', 0))} "
                f"| {self._fmt_currency(total)} |"
            )
        lines.append(
            f"| **TOTAL** | | | | | **{self._fmt_currency(annual_total)}** |"
        )
        return "\n".join(lines)

    def _md_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Render scenario comparison section."""
        scenarios = data.get("scenarios", [])
        if not scenarios:
            return "## 3. Scenario Comparison\n\n_No scenario data available._"
        lines = [
            "## 3. Scenario Comparison\n",
            "| Scenario | Annual Cost | vs Base | Key Assumptions |",
            "|----------|-----------|---------|----------------|",
        ]
        for sc in scenarios:
            label = self.SCENARIO_LABELS.get(
                sc.get("name", "").lower(), sc.get("name", "-")
            )
            lines.append(
                f"| {label} "
                f"| {self._fmt_currency(sc.get('annual_cost', 0))} "
                f"| {self._fmt_currency(sc.get('vs_base', 0))} "
                f"| {sc.get('assumptions', '-')} |"
            )
        return "\n".join(lines)

    def _md_confidence_intervals(self, data: Dict[str, Any]) -> str:
        """Render confidence intervals section."""
        ci = data.get("confidence_intervals", {})
        bands = ci.get("bands", [])
        lines = [
            "## 4. Confidence Intervals\n",
            f"**Forecast Model:** {ci.get('model', '-')}  ",
            f"**Model Accuracy (MAPE):** {self._fmt(ci.get('mape_pct', 0))}%  ",
            f"**R-squared:** {self._fmt(ci.get('r_squared', 0), 3)}\n",
        ]
        if bands:
            lines.extend([
                "| Confidence Level | Lower Bound | Point Estimate | Upper Bound |",
                "|-----------------|------------|---------------|------------|",
            ])
            for b in bands:
                lines.append(
                    f"| {b.get('level', '-')} "
                    f"| {self._fmt_currency(b.get('lower', 0))} "
                    f"| {self._fmt_currency(b.get('point', 0))} "
                    f"| {self._fmt_currency(b.get('upper', 0))} |"
                )
        return "\n".join(lines)

    def _md_variance_decomposition(self, data: Dict[str, Any]) -> str:
        """Render variance decomposition section."""
        components = data.get("variance_decomposition", [])
        if not components:
            return "## 5. Variance Decomposition\n\n_No variance decomposition available._"
        lines = [
            "## 5. Variance Decomposition\n",
            "| Factor | Impact | Share (%) | Direction |",
            "|--------|--------|----------|-----------|",
        ]
        for c in components:
            direction = "UP" if c.get("impact", 0) > 0 else "DOWN"
            lines.append(
                f"| {c.get('factor', '-')} "
                f"| {self._fmt_currency(c.get('impact', 0))} "
                f"| {self._fmt(c.get('share_pct', 0))}% "
                f"| {direction} |"
            )
        return "\n".join(lines)

    def _md_budget_recommendation(self, data: Dict[str, Any]) -> str:
        """Render budget recommendation section."""
        rec = data.get("recommendation", {})
        risks = rec.get("risks", [])
        lines = [
            "## 6. Budget Recommendation\n",
            f"**Recommended Budget:** {self._fmt_currency(rec.get('recommended_budget', 0))}  ",
            f"**Budget Scenario:** {rec.get('budget_scenario', '-')}  ",
            f"**Contingency Reserve:** {self._fmt_currency(rec.get('contingency', 0))} "
            f"({self._fmt(rec.get('contingency_pct', 0))}%)  ",
            f"**Total with Contingency:** {self._fmt_currency(rec.get('total_with_contingency', 0))}  ",
            f"**Confidence:** {rec.get('confidence_level', '-')}\n",
        ]
        if rec.get("rationale"):
            lines.append(f"**Rationale:** {rec.get('rationale', '')}\n")
        if risks:
            lines.append("### Risk Factors\n")
            for r in risks:
                lines.append(
                    f"- **{r.get('risk', '-')}** ({r.get('likelihood', '-')}): "
                    f"{r.get('impact', '-')}"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return (
            "---\n\n"
            "*Generated by GreenLang PACK-036 Utility Analysis Pack*  \n"
            "*Forecast projections are estimates. Actual results may vary "
            "due to weather, rate changes, and operational factors.*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header with forecast summary cards."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        summary = data.get("forecast_summary", {})
        return (
            f'<h1>Budget Forecast Report</h1>\n'
            f'<p class="subtitle">Organization: {data.get("organization_name", "-")} | '
            f'Period: {data.get("forecast_period", "-")} | Generated: {ts}</p>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Base Forecast</span>'
            f'<span class="value">{self._fmt_currency(summary.get("base_annual", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Optimistic</span>'
            f'<span class="value">{self._fmt_currency(summary.get("optimistic_annual", 0))}</span></div>\n'
            f'  <div class="card card-red"><span class="label">Pessimistic</span>'
            f'<span class="value">{self._fmt_currency(summary.get("pessimistic_annual", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Confidence</span>'
            f'<span class="value">{summary.get("confidence_level", "-")}</span></div>\n'
            f'</div>'
        )

    def _html_historical_baseline(self, data: Dict[str, Any]) -> str:
        """Render HTML historical baseline section."""
        hb = data.get("historical_baseline", {})
        years = hb.get("annual_history", [])
        rows = ""
        for y in years:
            rows += (
                f'<tr><td>{y.get("year", "-")}</td>'
                f'<td>{self._fmt(y.get("kwh", 0), 0)}</td>'
                f'<td>{self._fmt_currency(y.get("cost", 0))}</td>'
                f'<td>{self._fmt(y.get("yoy_change_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Historical Baseline</h2>\n'
            f'<div class="info-box"><p>Period: {hb.get("baseline_period", "-")} | '
            f'Avg Annual: {self._fmt_currency(hb.get("avg_annual_cost", 0))} | '
            f'CAGR: {self._fmt(hb.get("cost_cagr_pct", 0))}%</p></div>\n'
            '<table>\n<tr><th>Year</th><th>kWh</th><th>Cost</th>'
            f'<th>YoY Change</th></tr>\n{rows}</table>'
        )

    def _html_monthly_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML monthly projections table."""
        projections = data.get("monthly_projections", [])
        rows = ""
        for proj in projections:
            base = {}
            for sc in proj.get("scenarios", []):
                if sc.get("name", "").lower() == "base":
                    base = sc
                    break
            if not base and proj.get("scenarios"):
                base = proj["scenarios"][0]
            rows += (
                f'<tr><td>{proj.get("month", "-")}</td>'
                f'<td>{self._fmt(base.get("consumption_kwh", 0), 0)}</td>'
                f'<td>{self._fmt_currency(base.get("energy_cost", 0))}</td>'
                f'<td>{self._fmt_currency(base.get("demand_cost", 0))}</td>'
                f'<td>{self._fmt_currency(base.get("total_cost", 0))}</td></tr>\n'
            )
        return (
            '<h2>Monthly Projections (Base Case)</h2>\n'
            '<table>\n<tr><th>Month</th><th>kWh</th><th>Energy Cost</th>'
            f'<th>Demand Cost</th><th>Total</th></tr>\n{rows}</table>'
        )

    def _html_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML scenario comparison section."""
        scenarios = data.get("scenarios", [])
        rows = ""
        for sc in scenarios:
            label = self.SCENARIO_LABELS.get(
                sc.get("name", "").lower(), sc.get("name", "-")
            )
            rows += (
                f'<tr><td>{label}</td>'
                f'<td>{self._fmt_currency(sc.get("annual_cost", 0))}</td>'
                f'<td>{self._fmt_currency(sc.get("vs_base", 0))}</td>'
                f'<td>{sc.get("assumptions", "-")}</td></tr>\n'
            )
        return (
            '<h2>Scenario Comparison</h2>\n'
            '<table>\n<tr><th>Scenario</th><th>Annual Cost</th>'
            f'<th>vs Base</th><th>Assumptions</th></tr>\n{rows}</table>'
        )

    def _html_confidence_intervals(self, data: Dict[str, Any]) -> str:
        """Render HTML confidence intervals section."""
        ci = data.get("confidence_intervals", {})
        bands = ci.get("bands", [])
        rows = ""
        for b in bands:
            rows += (
                f'<tr><td>{b.get("level", "-")}</td>'
                f'<td>{self._fmt_currency(b.get("lower", 0))}</td>'
                f'<td>{self._fmt_currency(b.get("point", 0))}</td>'
                f'<td>{self._fmt_currency(b.get("upper", 0))}</td></tr>\n'
            )
        return (
            '<h2>Confidence Intervals</h2>\n'
            f'<div class="info-box"><p>Model: {ci.get("model", "-")} | '
            f'MAPE: {self._fmt(ci.get("mape_pct", 0))}% | '
            f'R2: {self._fmt(ci.get("r_squared", 0), 3)}</p></div>\n'
            '<table>\n<tr><th>Confidence</th><th>Lower</th>'
            f'<th>Point Estimate</th><th>Upper</th></tr>\n{rows}</table>'
        )

    def _html_variance_decomposition(self, data: Dict[str, Any]) -> str:
        """Render HTML variance decomposition section."""
        components = data.get("variance_decomposition", [])
        rows = ""
        for c in components:
            direction = "UP" if c.get("impact", 0) > 0 else "DOWN"
            cls = "variance-up" if direction == "UP" else "variance-down"
            rows += (
                f'<tr class="{cls}"><td>{c.get("factor", "-")}</td>'
                f'<td>{self._fmt_currency(c.get("impact", 0))}</td>'
                f'<td>{self._fmt(c.get("share_pct", 0))}%</td>'
                f'<td>{direction}</td></tr>\n'
            )
        return (
            '<h2>Variance Decomposition</h2>\n'
            '<table>\n<tr><th>Factor</th><th>Impact</th>'
            f'<th>Share</th><th>Direction</th></tr>\n{rows}</table>'
        )

    def _html_budget_recommendation(self, data: Dict[str, Any]) -> str:
        """Render HTML budget recommendation section."""
        rec = data.get("recommendation", {})
        risks = rec.get("risks", [])
        risk_items = "".join(
            f'<li><strong>{r.get("risk", "-")}</strong> '
            f'({r.get("likelihood", "-")}): {r.get("impact", "-")}</li>\n'
            for r in risks
        )
        return (
            '<h2>Budget Recommendation</h2>\n'
            '<div class="info-box">'
            f'<p><strong>Recommended:</strong> '
            f'{self._fmt_currency(rec.get("recommended_budget", 0))} | '
            f'<strong>Contingency:</strong> '
            f'{self._fmt_currency(rec.get("contingency", 0))} '
            f'({self._fmt(rec.get("contingency_pct", 0))}%) | '
            f'<strong>Total:</strong> '
            f'{self._fmt_currency(rec.get("total_with_contingency", 0))}</p>'
            f'<p>{rec.get("rationale", "")}</p></div>\n'
            f'<h3>Risk Factors</h3>\n<ul>\n{risk_items}</ul>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        projections = data.get("monthly_projections", [])
        scenarios = data.get("scenarios", [])
        decomp = data.get("variance_decomposition", [])
        months = [p.get("month", "") for p in projections]
        scenario_series: Dict[str, List[float]] = {}
        for proj in projections:
            for sc in proj.get("scenarios", []):
                name = sc.get("name", "unknown")
                if name not in scenario_series:
                    scenario_series[name] = []
                scenario_series[name].append(sc.get("total_cost", 0))
        return {
            "monthly_forecast_line": {
                "type": "line",
                "labels": months,
                "series": scenario_series,
            },
            "scenario_comparison_bar": {
                "type": "bar",
                "labels": [sc.get("name", "") for sc in scenarios],
                "values": [sc.get("annual_cost", 0) for sc in scenarios],
            },
            "variance_waterfall": {
                "type": "waterfall",
                "labels": [c.get("factor", "") for c in decomp],
                "values": [c.get("impact", 0) for c in decomp],
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
            "h3{color:#495057;margin-top:20px;}"
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
            ".variance-up{color:#dc3545;}"
            ".variance-down{color:#198754;}"
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
