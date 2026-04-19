# -*- coding: utf-8 -*-
"""
WasteHeatRecoveryReportTemplate - Waste heat recovery feasibility report for PACK-031.

Generates waste heat recovery feasibility reports with heat source inventory,
pinch analysis results, composite curve data, technology option evaluation,
ROI analysis, and implementation planning. Follows heat integration
methodology per IEA Industrial Energy Technology Roadmap.

Sections:
    1. Executive Summary
    2. Heat Source Inventory
    3. Heat Sink Inventory
    4. Pinch Analysis Results
    5. Composite Curves Data
    6. Technology Options
    7. Financial Analysis (ROI)
    8. Implementation Plan

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WasteHeatRecoveryReportTemplate:
    """
    Waste heat recovery feasibility report template.

    Renders heat integration analysis with source/sink inventory, pinch
    analysis, composite curves, technology options, and financial ROI
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RECOVERY_TECHNOLOGIES: List[str] = [
        "Heat Exchanger (Plate/Shell-and-Tube)",
        "Economizer",
        "Heat Pipe",
        "Organic Rankine Cycle (ORC)",
        "Heat Pump",
        "Thermoelectric Generator",
        "Regenerator",
        "Waste Heat Boiler",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize WasteHeatRecoveryReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render waste heat recovery feasibility report as Markdown.

        Args:
            data: WHR engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_heat_sources(data),
            self._md_heat_sinks(data),
            self._md_pinch_analysis(data),
            self._md_composite_curves(data),
            self._md_technology_options(data),
            self._md_financial_analysis(data),
            self._md_implementation_plan(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render waste heat recovery feasibility report as HTML.

        Args:
            data: WHR engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_heat_sources(data),
            self._html_pinch_analysis(data),
            self._html_technology_options(data),
            self._html_financial_analysis(data),
            self._html_implementation_plan(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Waste Heat Recovery Feasibility Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render waste heat recovery report as structured JSON.

        Args:
            data: WHR engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "waste_heat_recovery_report",
            "version": "31.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "heat_sources": data.get("heat_sources", []),
            "heat_sinks": data.get("heat_sinks", []),
            "pinch_analysis": data.get("pinch_analysis", {}),
            "composite_curves": data.get("composite_curves", {}),
            "technology_options": data.get("technology_options", []),
            "financial_analysis": data.get("financial_analysis", {}),
            "implementation_plan": data.get("implementation_plan", []),
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
            f"# Waste Heat Recovery Feasibility Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Assessment Date:** {data.get('assessment_date', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 WasteHeatRecoveryReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        s = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Waste Heat Available | {self._fmt(s.get('total_waste_heat_kw', 0))} kW |\n"
            f"| Recoverable Heat | {self._fmt(s.get('recoverable_heat_kw', 0))} kW "
            f"({self._fmt(s.get('recovery_potential_pct', 0))}%) |\n"
            f"| Energy Savings Potential | {self._fmt(s.get('energy_savings_mwh', 0))} MWh/yr |\n"
            f"| Cost Savings Potential | EUR {self._fmt(s.get('cost_savings_eur', 0))} /yr |\n"
            f"| CO2 Reduction | {self._fmt(s.get('co2_reduction_tonnes', 0))} tonnes/yr |\n"
            f"| Recommended Investment | EUR {self._fmt(s.get('investment_eur', 0))} |\n"
            f"| Projected Payback | {self._fmt(s.get('payback_years', 0), 1)} years |\n"
            f"| Pinch Temperature | {self._fmt(s.get('pinch_temp_c', 0))} C |\n"
            f"| Heat Sources Identified | {s.get('source_count', 0)} |\n"
            f"| Feasibility Rating | {s.get('feasibility_rating', '-')} |"
        )

    def _md_heat_sources(self, data: Dict[str, Any]) -> str:
        """Render heat source inventory section."""
        sources = data.get("heat_sources", [])
        if not sources:
            return "## 2. Heat Source Inventory\n\n_No heat sources identified._"
        lines = [
            "## 2. Heat Source Inventory\n",
            "| # | Source | Process | Temp (C) | Flow | Heat (kW) | "
            "Availability (hr/yr) | Quality |",
            "|---|--------|---------|---------|------|----------|"
            "--------------------|---------|",
        ]
        for i, src in enumerate(sources, 1):
            lines.append(
                f"| {i} | {src.get('name', '-')} "
                f"| {src.get('process', '-')} "
                f"| {self._fmt(src.get('temperature_c', 0), 0)} "
                f"| {self._fmt(src.get('flow_rate', 0))} {src.get('flow_unit', 'kg/s')} "
                f"| {self._fmt(src.get('heat_kw', 0))} "
                f"| {self._fmt(src.get('availability_hrs', 0), 0)} "
                f"| {src.get('quality', '-')} |"
            )
        total_heat = sum(s.get("heat_kw", 0) for s in sources)
        lines.append(f"\n**Total Waste Heat Available:** {self._fmt(total_heat)} kW")
        return "\n".join(lines)

    def _md_heat_sinks(self, data: Dict[str, Any]) -> str:
        """Render heat sink inventory section."""
        sinks = data.get("heat_sinks", [])
        if not sinks:
            return "## 3. Heat Sink Inventory\n\n_No heat sinks identified._"
        lines = [
            "## 3. Heat Sink Inventory\n",
            "| # | Sink | Process | Required Temp (C) | Heat Demand (kW) | "
            "Current Source | Match Potential |",
            "|---|------|---------|------------------|-----------------|"
            "--------------|----------------|",
        ]
        for i, sink in enumerate(sinks, 1):
            lines.append(
                f"| {i} | {sink.get('name', '-')} "
                f"| {sink.get('process', '-')} "
                f"| {self._fmt(sink.get('required_temp_c', 0), 0)} "
                f"| {self._fmt(sink.get('heat_demand_kw', 0))} "
                f"| {sink.get('current_source', '-')} "
                f"| {sink.get('match_potential', '-')} |"
            )
        return "\n".join(lines)

    def _md_pinch_analysis(self, data: Dict[str, Any]) -> str:
        """Render pinch analysis results section."""
        pinch = data.get("pinch_analysis", {})
        lines = [
            "## 4. Pinch Analysis Results\n",
            f"**Delta T Minimum:** {self._fmt(pinch.get('delta_t_min_c', 10))} C  ",
            f"**Pinch Temperature:** {self._fmt(pinch.get('pinch_temp_c', 0))} C  ",
            f"**Minimum Hot Utility:** {self._fmt(pinch.get('min_hot_utility_kw', 0))} kW  ",
            f"**Minimum Cold Utility:** {self._fmt(pinch.get('min_cold_utility_kw', 0))} kW  ",
            f"**Maximum Heat Recovery:** {self._fmt(pinch.get('max_recovery_kw', 0))} kW  ",
            f"**Current Heat Recovery:** {self._fmt(pinch.get('current_recovery_kw', 0))} kW  ",
            f"**Additional Recovery Potential:** "
            f"{self._fmt(pinch.get('additional_recovery_kw', 0))} kW",
        ]
        violations = pinch.get("pinch_violations", [])
        if violations:
            lines.extend([
                "\n### Pinch Rule Violations\n",
                "| Violation | Description | Energy Penalty (kW) |",
                "|-----------|-------------|-------------------|",
            ])
            for v in violations:
                lines.append(
                    f"| {v.get('type', '-')} "
                    f"| {v.get('description', '-')} "
                    f"| {self._fmt(v.get('penalty_kw', 0))} |"
                )
        return "\n".join(lines)

    def _md_composite_curves(self, data: Dict[str, Any]) -> str:
        """Render composite curves data section."""
        cc = data.get("composite_curves", {})
        hot = cc.get("hot_composite", [])
        cold = cc.get("cold_composite", [])
        lines = [
            "## 5. Composite Curves\n",
            "_Chart data provided for rendering hot and cold composite curves._",
        ]
        if hot:
            lines.extend([
                "\n### Hot Composite Curve Data\n",
                "| Enthalpy (kW) | Temperature (C) |",
                "|-------------|----------------|",
            ])
            for pt in hot[:20]:
                lines.append(
                    f"| {self._fmt(pt.get('enthalpy_kw', 0))} "
                    f"| {self._fmt(pt.get('temperature_c', 0))} |"
                )
        if cold:
            lines.extend([
                "\n### Cold Composite Curve Data\n",
                "| Enthalpy (kW) | Temperature (C) |",
                "|-------------|----------------|",
            ])
            for pt in cold[:20]:
                lines.append(
                    f"| {self._fmt(pt.get('enthalpy_kw', 0))} "
                    f"| {self._fmt(pt.get('temperature_c', 0))} |"
                )
        return "\n".join(lines)

    def _md_technology_options(self, data: Dict[str, Any]) -> str:
        """Render technology options evaluation section."""
        options = data.get("technology_options", [])
        if not options:
            return "## 6. Technology Options\n\n_No technology options evaluated._"
        lines = [
            "## 6. Technology Options\n",
            "| Technology | Applicability | Recovery (kW) | Efficiency (%) "
            "| Investment (EUR) | Annual Savings (EUR) | Payback (yr) |",
            "|-----------|--------------|-------------|---------------|"
            "-----------------|---------------------|-------------|",
        ]
        for opt in options:
            lines.append(
                f"| {opt.get('technology', '-')} "
                f"| {opt.get('applicability', '-')} "
                f"| {self._fmt(opt.get('recovery_kw', 0))} "
                f"| {self._fmt(opt.get('efficiency_pct', 0))}% "
                f"| {self._fmt(opt.get('investment_eur', 0))} "
                f"| {self._fmt(opt.get('annual_savings_eur', 0))} "
                f"| {self._fmt(opt.get('payback_years', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_financial_analysis(self, data: Dict[str, Any]) -> str:
        """Render financial analysis section."""
        fin = data.get("financial_analysis", {})
        scenarios = fin.get("scenarios", [])
        lines = [
            "## 7. Financial Analysis\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Investment | EUR {self._fmt(fin.get('total_investment_eur', 0))} |",
            f"| Annual Energy Savings | EUR {self._fmt(fin.get('annual_savings_eur', 0))} |",
            f"| Annual O&M Cost | EUR {self._fmt(fin.get('annual_om_eur', 0))} |",
            f"| Net Annual Benefit | EUR {self._fmt(fin.get('net_annual_benefit_eur', 0))} |",
            f"| Simple Payback | {self._fmt(fin.get('simple_payback_years', 0), 1)} years |",
            f"| NPV (10 yr, {fin.get('discount_rate_pct', 8)}%) | "
            f"EUR {self._fmt(fin.get('npv_eur', 0))} |",
            f"| IRR | {self._fmt(fin.get('irr_pct', 0))}% |",
            f"| Lifecycle (years) | {fin.get('lifecycle_years', 15)} |",
        ]
        if scenarios:
            lines.extend([
                "\n### Sensitivity Scenarios\n",
                "| Scenario | NPV (EUR) | IRR (%) | Payback (yr) |",
                "|----------|----------|---------|-------------|",
            ])
            for sc in scenarios:
                lines.append(
                    f"| {sc.get('name', '-')} "
                    f"| {self._fmt(sc.get('npv_eur', 0))} "
                    f"| {self._fmt(sc.get('irr_pct', 0))} "
                    f"| {self._fmt(sc.get('payback_years', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_implementation_plan(self, data: Dict[str, Any]) -> str:
        """Render implementation plan section."""
        phases = data.get("implementation_plan", [])
        if not phases:
            return "## 8. Implementation Plan\n\n_No implementation plan defined._"
        lines = ["## 8. Implementation Plan\n"]
        for phase in phases:
            lines.extend([
                f"### {phase.get('phase', 'Phase')}: {phase.get('name', '-')}\n",
                f"- **Timeline:** {phase.get('timeline', '-')}",
                f"- **Budget:** EUR {self._fmt(phase.get('budget_eur', 0))}",
                f"- **Key Activities:**",
            ])
            for act in phase.get("activities", []):
                lines.append(f"  - {act}")
            lines.append(f"- **Dependencies:** {phase.get('dependencies', 'None')}")
            lines.append(f"- **Risks:** {phase.get('risks', 'None')}")
            lines.append("")
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
            f'<h1>Waste Heat Recovery Feasibility Report</h1>\n'
            f'<p class="subtitle">Facility: {facility}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Waste Heat</span>'
            f'<span class="value">{self._fmt(s.get("total_waste_heat_kw", 0))} kW</span></div>\n'
            f'  <div class="card"><span class="label">Recoverable</span>'
            f'<span class="value">{self._fmt(s.get("recoverable_heat_kw", 0))} kW</span></div>\n'
            f'  <div class="card"><span class="label">Savings</span>'
            f'<span class="value">EUR {self._fmt(s.get("cost_savings_eur", 0))}/yr</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(s.get("payback_years", 0), 1)} yr</span></div>\n'
            '</div>'
        )

    def _html_heat_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML heat sources table."""
        sources = data.get("heat_sources", [])
        rows = ""
        for s in sources:
            rows += (
                f'<tr><td>{s.get("name", "-")}</td>'
                f'<td>{self._fmt(s.get("temperature_c", 0), 0)} C</td>'
                f'<td>{self._fmt(s.get("heat_kw", 0))} kW</td>'
                f'<td>{s.get("quality", "-")}</td></tr>\n'
            )
        return (
            '<h2>Heat Source Inventory</h2>\n<table>\n'
            '<tr><th>Source</th><th>Temp</th><th>Heat</th><th>Quality</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_pinch_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML pinch analysis."""
        pinch = data.get("pinch_analysis", {})
        return (
            '<h2>Pinch Analysis</h2>\n'
            f'<p>Pinch: {self._fmt(pinch.get("pinch_temp_c", 0))} C | '
            f'Max Recovery: {self._fmt(pinch.get("max_recovery_kw", 0))} kW | '
            f'Additional: {self._fmt(pinch.get("additional_recovery_kw", 0))} kW</p>\n'
            '<div class="chart-placeholder" data-chart="composite_curves">'
            '[Composite Curves Chart]</div>'
        )

    def _html_technology_options(self, data: Dict[str, Any]) -> str:
        """Render HTML technology options."""
        options = data.get("technology_options", [])
        rows = ""
        for opt in options:
            rows += (
                f'<tr><td>{opt.get("technology", "-")}</td>'
                f'<td>{self._fmt(opt.get("recovery_kw", 0))} kW</td>'
                f'<td>EUR {self._fmt(opt.get("investment_eur", 0))}</td>'
                f'<td>{self._fmt(opt.get("payback_years", 0), 1)} yr</td></tr>\n'
            )
        return (
            '<h2>Technology Options</h2>\n<table>\n'
            '<tr><th>Technology</th><th>Recovery</th><th>Investment</th>'
            f'<th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_financial_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML financial analysis."""
        fin = data.get("financial_analysis", {})
        return (
            '<h2>Financial Analysis</h2>\n'
            f'<p>NPV: EUR {self._fmt(fin.get("npv_eur", 0))} | '
            f'IRR: {self._fmt(fin.get("irr_pct", 0))}% | '
            f'Payback: {self._fmt(fin.get("simple_payback_years", 0), 1)} yr</p>'
        )

    def _html_implementation_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation plan."""
        phases = data.get("implementation_plan", [])
        content = ""
        for phase in phases:
            activities = "".join(
                f'<li>{a}</li>' for a in phase.get("activities", [])
            )
            content += (
                f'<div class="phase"><h3>{phase.get("phase", "")} - '
                f'{phase.get("name", "")}</h3>'
                f'<p>{phase.get("timeline", "")} | '
                f'EUR {self._fmt(phase.get("budget_eur", 0))}</p>'
                f'<ul>{activities}</ul></div>\n'
            )
        return f'<h2>Implementation Plan</h2>\n{content}'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        return data.get("executive_summary", {})

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        hot = data.get("composite_curves", {}).get("hot_composite", [])
        cold = data.get("composite_curves", {}).get("cold_composite", [])
        sources = data.get("heat_sources", [])
        options = data.get("technology_options", [])
        return {
            "composite_curves": {
                "type": "line",
                "series": {
                    "hot": {
                        "x": [p.get("enthalpy_kw", 0) for p in hot],
                        "y": [p.get("temperature_c", 0) for p in hot],
                    },
                    "cold": {
                        "x": [p.get("enthalpy_kw", 0) for p in cold],
                        "y": [p.get("temperature_c", 0) for p in cold],
                    },
                },
                "x_label": "Enthalpy (kW)",
                "y_label": "Temperature (C)",
            },
            "heat_source_bar": {
                "type": "horizontal_bar",
                "labels": [s.get("name", "") for s in sources],
                "values": [s.get("heat_kw", 0) for s in sources],
            },
            "technology_comparison": {
                "type": "grouped_bar",
                "labels": [o.get("technology", "") for o in options],
                "series": {
                    "recovery_kw": [o.get("recovery_kw", 0) for o in options],
                    "investment_keur": [
                        o.get("investment_eur", 0) / 1000 for o in options
                    ],
                },
            },
            "temperature_waterfall": {
                "type": "waterfall",
                "labels": [s.get("name", "") for s in sources],
                "values": [s.get("temperature_c", 0) for s in sources],
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
            "h3{color:#0d6efd;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".phase{border-left:3px solid #0d6efd;padding-left:15px;margin:15px 0;}"
            ".chart-placeholder{background:#f0f0f0;border:2px dashed #ccc;padding:40px;"
            "text-align:center;margin:15px 0;}"
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
