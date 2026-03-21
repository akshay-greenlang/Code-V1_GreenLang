# -*- coding: utf-8 -*-
"""
SteamSystemReportTemplate - Steam system assessment report for PACK-031.

Generates comprehensive steam system assessment reports with boiler
efficiency analysis, flue gas analysis, steam trap survey results,
insulation assessment, condensate recovery analysis, flash steam recovery
opportunities, and combined heat and power (CHP) feasibility assessment.

Sections:
    1. Executive Summary
    2. Boiler Efficiency Analysis
    3. Flue Gas Analysis
    4. Steam Trap Survey
    5. Insulation Assessment
    6. Condensate Recovery Analysis
    7. Flash Steam Recovery
    8. CHP Feasibility Assessment
    9. Recommendations

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SteamSystemReportTemplate:
    """
    Steam system assessment report template.

    Renders steam system audit data including boiler efficiency, flue gas,
    trap surveys, insulation, condensate recovery, flash steam, and CHP
    assessment across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    BOILER_EFFICIENCY_BENCHMARKS: Dict[str, float] = {
        "best_practice_pct": 94.0,
        "good_pct": 88.0,
        "average_pct": 82.0,
        "poor_pct": 75.0,
    }

    TRAP_FAILURE_BENCHMARKS: Dict[str, float] = {
        "best_practice_pct": 5.0,
        "acceptable_pct": 10.0,
        "poor_pct": 20.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SteamSystemReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render steam system assessment report as Markdown.

        Args:
            data: Steam system engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_boiler_efficiency(data),
            self._md_flue_gas(data),
            self._md_trap_survey(data),
            self._md_insulation(data),
            self._md_condensate_recovery(data),
            self._md_flash_steam(data),
            self._md_chp_assessment(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render steam system assessment report as HTML.

        Args:
            data: Steam system engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_boiler_efficiency(data),
            self._html_trap_survey(data),
            self._html_condensate_recovery(data),
            self._html_chp_assessment(data),
            self._html_recommendations(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Steam System Assessment Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render steam system assessment report as structured JSON.

        Args:
            data: Steam system engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "steam_system_report",
            "version": "31.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "boiler_efficiency": data.get("boiler_efficiency", {}),
            "flue_gas": data.get("flue_gas_analysis", {}),
            "trap_survey": data.get("trap_survey", {}),
            "insulation": data.get("insulation_assessment", {}),
            "condensate_recovery": data.get("condensate_recovery", {}),
            "flash_steam": data.get("flash_steam_recovery", {}),
            "chp_assessment": data.get("chp_assessment", {}),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Industrial Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Steam System Assessment Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Audit Date:** {data.get('audit_date', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 SteamSystemReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        s = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Boiler Capacity | {self._fmt(s.get('total_capacity_mw', 0))} MW |\n"
            f"| Steam Generation | {self._fmt(s.get('steam_generation_tph', 0))} t/h |\n"
            f"| Avg Boiler Efficiency | {self._fmt(s.get('avg_boiler_efficiency_pct', 0))}% |\n"
            f"| Annual Fuel Consumption | {self._fmt(s.get('annual_fuel_mwh', 0))} MWh |\n"
            f"| Annual Fuel Cost | EUR {self._fmt(s.get('annual_fuel_cost_eur', 0))} |\n"
            f"| Trap Failure Rate | {self._fmt(s.get('trap_failure_rate_pct', 0))}% |\n"
            f"| Condensate Return Rate | {self._fmt(s.get('condensate_return_pct', 0))}% |\n"
            f"| Total Savings Potential | {self._fmt(s.get('total_savings_mwh', 0))} MWh/yr "
            f"(EUR {self._fmt(s.get('total_savings_eur', 0))}) |"
        )

    def _md_boiler_efficiency(self, data: Dict[str, Any]) -> str:
        """Render boiler efficiency analysis section."""
        boilers = data.get("boiler_efficiency", {}).get("boilers", [])
        lines = ["## 2. Boiler Efficiency Analysis\n"]
        if not boilers:
            lines.append("_No boiler data available._")
            return "\n".join(lines)
        lines.extend([
            "| Boiler | Fuel | Capacity (MW) | Efficiency (%) | "
            "Benchmark (%) | Excess Air (%) | Stack Temp (C) |",
            "|--------|------|-------------|---------------|"
            "-------------|----------------|---------------|",
        ])
        for b in boilers:
            lines.append(
                f"| {b.get('boiler_id', '-')} "
                f"| {b.get('fuel_type', '-')} "
                f"| {self._fmt(b.get('capacity_mw', 0))} "
                f"| {self._fmt(b.get('efficiency_pct', 0), 1)} "
                f"| {self._fmt(self.BOILER_EFFICIENCY_BENCHMARKS['good_pct'], 1)} "
                f"| {self._fmt(b.get('excess_air_pct', 0), 1)} "
                f"| {self._fmt(b.get('stack_temp_c', 0), 0)} |"
            )
        losses = data.get("boiler_efficiency", {}).get("losses_breakdown", {})
        if losses:
            lines.extend([
                "\n### Heat Loss Breakdown\n",
                "| Loss Category | Percentage |",
                "|--------------|-----------|",
            ])
            for loss_name, loss_val in losses.items():
                lines.append(
                    f"| {loss_name.replace('_', ' ').title()} | {self._fmt(loss_val, 1)}% |"
                )
        return "\n".join(lines)

    def _md_flue_gas(self, data: Dict[str, Any]) -> str:
        """Render flue gas analysis section."""
        fg = data.get("flue_gas_analysis", {})
        measurements = fg.get("measurements", [])
        lines = [
            "## 3. Flue Gas Analysis\n",
            f"**Method:** {fg.get('method', 'Direct measurement with flue gas analyzer')}",
        ]
        if measurements:
            lines.extend([
                "\n| Boiler | O2 (%) | CO2 (%) | CO (ppm) | NOx (ppm) | "
                "Stack Temp (C) | Efficiency (%) |",
                "|--------|--------|---------|----------|-----------|"
                "---------------|---------------|",
            ])
            for m in measurements:
                lines.append(
                    f"| {m.get('boiler_id', '-')} "
                    f"| {self._fmt(m.get('o2_pct', 0), 1)} "
                    f"| {self._fmt(m.get('co2_pct', 0), 1)} "
                    f"| {self._fmt(m.get('co_ppm', 0), 0)} "
                    f"| {self._fmt(m.get('nox_ppm', 0), 0)} "
                    f"| {self._fmt(m.get('stack_temp_c', 0), 0)} "
                    f"| {self._fmt(m.get('efficiency_pct', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_trap_survey(self, data: Dict[str, Any]) -> str:
        """Render steam trap survey section."""
        traps = data.get("trap_survey", {})
        summary = traps.get("summary", {})
        by_type = traps.get("by_type", [])
        lines = [
            "## 4. Steam Trap Survey\n",
            f"**Total Traps Surveyed:** {summary.get('total_surveyed', 0)}  ",
            f"**Failed Traps:** {summary.get('failed', 0)} "
            f"({self._fmt(summary.get('failure_rate_pct', 0))}%)  ",
            f"**Blocked Traps:** {summary.get('blocked', 0)}  ",
            f"**Blowing Through:** {summary.get('blowing_through', 0)}  ",
            f"**Annual Steam Loss:** {self._fmt(summary.get('annual_steam_loss_tonnes', 0))} tonnes  ",
            f"**Annual Loss Cost:** EUR {self._fmt(summary.get('annual_loss_cost_eur', 0))}",
        ]
        if by_type:
            lines.extend([
                "\n### Results by Trap Type\n",
                "| Trap Type | Total | OK | Failed | Blocked | Failure Rate |",
                "|-----------|-------|-----|--------|---------|-------------|",
            ])
            for t in by_type:
                lines.append(
                    f"| {t.get('type', '-')} "
                    f"| {t.get('total', 0)} "
                    f"| {t.get('ok', 0)} "
                    f"| {t.get('failed', 0)} "
                    f"| {t.get('blocked', 0)} "
                    f"| {self._fmt(t.get('failure_rate_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_insulation(self, data: Dict[str, Any]) -> str:
        """Render insulation assessment section."""
        ins = data.get("insulation_assessment", {})
        items = ins.get("items", [])
        lines = [
            "## 5. Insulation Assessment\n",
            f"**Total Items Inspected:** {ins.get('total_inspected', 0)}  ",
            f"**Uninsulated/Damaged:** {ins.get('uninsulated_count', 0)}  ",
            f"**Annual Heat Loss (uninsulated):** {self._fmt(ins.get('annual_loss_mwh', 0))} MWh  ",
            f"**Annual Cost of Losses:** EUR {self._fmt(ins.get('annual_loss_cost_eur', 0))}",
        ]
        if items:
            lines.extend([
                "\n### Insulation Deficiencies\n",
                "| Location | Component | Surface Temp (C) | Loss (kW) | "
                "Annual Cost (EUR) | Priority |",
                "|----------|-----------|-----------------|----------|"
                "-----------------|----------|",
            ])
            for item in items[:20]:
                lines.append(
                    f"| {item.get('location', '-')} "
                    f"| {item.get('component', '-')} "
                    f"| {self._fmt(item.get('surface_temp_c', 0), 0)} "
                    f"| {self._fmt(item.get('loss_kw', 0), 1)} "
                    f"| {self._fmt(item.get('annual_cost_eur', 0))} "
                    f"| {item.get('priority', '-')} |"
                )
        return "\n".join(lines)

    def _md_condensate_recovery(self, data: Dict[str, Any]) -> str:
        """Render condensate recovery analysis section."""
        cr = data.get("condensate_recovery", {})
        lines = [
            "## 6. Condensate Recovery Analysis\n",
            f"**Current Recovery Rate:** {self._fmt(cr.get('current_recovery_pct', 0))}%  ",
            f"**Target Recovery Rate:** {self._fmt(cr.get('target_recovery_pct', 0))}%  ",
            f"**Condensate Temperature:** {self._fmt(cr.get('condensate_temp_c', 0))} C  ",
            f"**Energy Value of Unrecovered:** {self._fmt(cr.get('unrecovered_energy_mwh', 0))} MWh/yr  ",
            f"**Water Cost Savings:** EUR {self._fmt(cr.get('water_savings_eur', 0))}/yr  ",
            f"**Treatment Cost Savings:** EUR {self._fmt(cr.get('treatment_savings_eur', 0))}/yr  ",
            f"**Energy Savings:** {self._fmt(cr.get('energy_savings_mwh', 0))} MWh/yr  ",
            f"**Total Annual Savings:** EUR {self._fmt(cr.get('total_savings_eur', 0))}/yr",
        ]
        barriers = cr.get("barriers", [])
        if barriers:
            lines.append("\n### Recovery Barriers\n")
            for b in barriers:
                lines.append(
                    f"- **{b.get('barrier', '-')}**: {b.get('description', '-')} "
                    f"(Impact: {b.get('impact', '-')})"
                )
        return "\n".join(lines)

    def _md_flash_steam(self, data: Dict[str, Any]) -> str:
        """Render flash steam recovery section."""
        fs = data.get("flash_steam_recovery", {})
        opportunities = fs.get("opportunities", [])
        lines = [
            "## 7. Flash Steam Recovery\n",
            f"**Flash Steam Available:** {self._fmt(fs.get('flash_steam_available_tph', 0))} t/h  ",
            f"**Currently Recovered:** {self._fmt(fs.get('currently_recovered_tph', 0))} t/h  ",
            f"**Recovery Opportunity:** {self._fmt(fs.get('recovery_opportunity_tph', 0))} t/h  ",
            f"**Potential Savings:** {self._fmt(fs.get('savings_mwh', 0))} MWh/yr",
        ]
        if opportunities:
            lines.extend([
                "\n### Recovery Opportunities\n",
                "| Source | Pressure (bar) | Flash (t/h) | Use Case | "
                "Savings (EUR/yr) |",
                "|--------|---------------|-----------|----------|"
                "----------------|",
            ])
            for o in opportunities:
                lines.append(
                    f"| {o.get('source', '-')} "
                    f"| {self._fmt(o.get('pressure_bar', 0), 1)} "
                    f"| {self._fmt(o.get('flash_tph', 0), 2)} "
                    f"| {o.get('use_case', '-')} "
                    f"| {self._fmt(o.get('savings_eur', 0))} |"
                )
        return "\n".join(lines)

    def _md_chp_assessment(self, data: Dict[str, Any]) -> str:
        """Render CHP feasibility assessment section."""
        chp = data.get("chp_assessment", {})
        if not chp:
            return "## 8. CHP Feasibility Assessment\n\n_CHP assessment not applicable._"
        lines = [
            "## 8. CHP Feasibility Assessment\n",
            f"**Feasibility:** {chp.get('feasibility', '-')}  ",
            f"**Recommended Technology:** {chp.get('technology', '-')}  ",
            f"**Electrical Output:** {self._fmt(chp.get('electrical_output_kw', 0))} kW  ",
            f"**Thermal Output:** {self._fmt(chp.get('thermal_output_kw', 0))} kW  ",
            f"**CHP Efficiency:** {self._fmt(chp.get('chp_efficiency_pct', 0))}%  ",
            f"**Annual Savings:** EUR {self._fmt(chp.get('annual_savings_eur', 0))}  ",
            f"**Investment:** EUR {self._fmt(chp.get('investment_eur', 0))}  ",
            f"**Payback:** {self._fmt(chp.get('payback_years', 0), 1)} years  ",
            f"**CO2 Reduction:** {self._fmt(chp.get('co2_reduction_tonnes', 0))} tonnes/yr",
        ]
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 9. Recommendations\n\n_No recommendations._"
        lines = [
            "## 9. Recommendations\n",
            "| # | Recommendation | Category | Savings (MWh/yr) | "
            "Investment (EUR) | Payback (yr) |",
            "|---|--------------|----------|-----------------|"
            "-----------------|-------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('title', '-')} "
                f"| {r.get('category', '-')} "
                f"| {self._fmt(r.get('energy_savings_mwh', 0))} "
                f"| {self._fmt(r.get('investment_eur', 0))} "
                f"| {self._fmt(r.get('payback_years', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Industrial Facility")
        return (
            f'<h1>Steam System Assessment Report</h1>\n'
            f'<p class="subtitle">Facility: {facility}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Boiler Efficiency</span>'
            f'<span class="value">{self._fmt(s.get("avg_boiler_efficiency_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">Trap Failure</span>'
            f'<span class="value">{self._fmt(s.get("trap_failure_rate_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">Condensate Return</span>'
            f'<span class="value">{self._fmt(s.get("condensate_return_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Savings Potential</span>'
            f'<span class="value">{self._fmt(s.get("total_savings_mwh", 0))} MWh</span></div>\n'
            '</div>'
        )

    def _html_boiler_efficiency(self, data: Dict[str, Any]) -> str:
        """Render HTML boiler efficiency."""
        boilers = data.get("boiler_efficiency", {}).get("boilers", [])
        rows = ""
        for b in boilers:
            rows += (
                f'<tr><td>{b.get("boiler_id", "-")}</td>'
                f'<td>{b.get("fuel_type", "-")}</td>'
                f'<td>{self._fmt(b.get("efficiency_pct", 0), 1)}%</td>'
                f'<td>{self._fmt(b.get("stack_temp_c", 0), 0)} C</td></tr>\n'
            )
        return (
            '<h2>Boiler Efficiency</h2>\n<table>\n'
            '<tr><th>Boiler</th><th>Fuel</th><th>Efficiency</th>'
            f'<th>Stack Temp</th></tr>\n{rows}</table>'
        )

    def _html_trap_survey(self, data: Dict[str, Any]) -> str:
        """Render HTML trap survey summary."""
        summary = data.get("trap_survey", {}).get("summary", {})
        return (
            '<h2>Steam Trap Survey</h2>\n'
            f'<p>Surveyed: {summary.get("total_surveyed", 0)} | '
            f'Failed: {summary.get("failed", 0)} '
            f'({self._fmt(summary.get("failure_rate_pct", 0))}%) | '
            f'Loss: EUR {self._fmt(summary.get("annual_loss_cost_eur", 0))}/yr</p>'
        )

    def _html_condensate_recovery(self, data: Dict[str, Any]) -> str:
        """Render HTML condensate recovery."""
        cr = data.get("condensate_recovery", {})
        return (
            '<h2>Condensate Recovery</h2>\n'
            f'<p>Current: {self._fmt(cr.get("current_recovery_pct", 0))}% | '
            f'Target: {self._fmt(cr.get("target_recovery_pct", 0))}% | '
            f'Savings: EUR {self._fmt(cr.get("total_savings_eur", 0))}/yr</p>'
        )

    def _html_chp_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML CHP assessment."""
        chp = data.get("chp_assessment", {})
        return (
            '<h2>CHP Feasibility</h2>\n'
            f'<p>Feasibility: {chp.get("feasibility", "-")} | '
            f'Output: {self._fmt(chp.get("electrical_output_kw", 0))} kWe | '
            f'Payback: {self._fmt(chp.get("payback_years", 0), 1)} yr</p>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        items = "".join(
            f'<li>{r.get("title", "-")} ({r.get("category", "-")}) - '
            f'Savings: {self._fmt(r.get("energy_savings_mwh", 0))} MWh/yr</li>\n'
            for r in recs
        )
        return f'<h2>Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        return data.get("executive_summary", {})

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        boilers = data.get("boiler_efficiency", {}).get("boilers", [])
        traps = data.get("trap_survey", {}).get("by_type", [])
        losses = data.get("boiler_efficiency", {}).get("losses_breakdown", {})
        return {
            "boiler_efficiency_bar": {
                "type": "bar",
                "labels": [b.get("boiler_id", "") for b in boilers],
                "values": [b.get("efficiency_pct", 0) for b in boilers],
                "benchmark": self.BOILER_EFFICIENCY_BENCHMARKS["good_pct"],
            },
            "heat_loss_pie": {
                "type": "pie",
                "labels": [k.replace("_", " ").title() for k in losses.keys()],
                "values": list(losses.values()),
            },
            "trap_status_stacked": {
                "type": "stacked_bar",
                "labels": [t.get("type", "") for t in traps],
                "series": {
                    "ok": [t.get("ok", 0) for t in traps],
                    "failed": [t.get("failed", 0) for t in traps],
                    "blocked": [t.get("blocked", 0) for t in traps],
                },
            },
            "condensate_gauge": {
                "type": "gauge",
                "value": data.get("condensate_recovery", {}).get(
                    "current_recovery_pct", 0
                ),
                "target": data.get("condensate_recovery", {}).get(
                    "target_recovery_pct", 0
                ),
                "max": 100,
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
