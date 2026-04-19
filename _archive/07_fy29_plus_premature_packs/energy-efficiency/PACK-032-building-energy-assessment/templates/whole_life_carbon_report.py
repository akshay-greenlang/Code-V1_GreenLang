# -*- coding: utf-8 -*-
"""
WholeLifeCarbonReportTemplate - Embodied + operational carbon report for PACK-032.

Generates whole life carbon assessment reports covering lifecycle stage
breakdowns (EN 15978 modules A1-D), material carbon hotspots,
operational carbon projections, whole life carbon per m2 benchmarks,
comparison to RIBA/LETI targets, material substitution opportunities,
biogenic carbon accounting, and sensitivity analysis.

Sections:
    1.  Summary
    2.  Lifecycle Stage Breakdown (A1-D)
    3.  Material Carbon Hotspots
    4.  Operational Carbon Projection
    5.  Whole Life Carbon per m2
    6.  Comparison to RIBA/LETI Targets
    7.  Material Substitution Opportunities
    8.  Biogenic Carbon
    9.  Sensitivity Analysis
    10. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# EN 15978 lifecycle modules
LIFECYCLE_MODULES: List[Dict[str, str]] = [
    {"module": "A1-A3", "name": "Product Stage", "description": "Raw material supply, transport, manufacturing"},
    {"module": "A4", "name": "Transport to Site", "description": "Transport of products to building site"},
    {"module": "A5", "name": "Construction", "description": "Construction/installation processes"},
    {"module": "B1", "name": "Use", "description": "In-use emissions (e.g., carbonation)"},
    {"module": "B2", "name": "Maintenance", "description": "Planned maintenance activities"},
    {"module": "B3", "name": "Repair", "description": "Repair of building components"},
    {"module": "B4", "name": "Replacement", "description": "Component replacement over lifetime"},
    {"module": "B5", "name": "Refurbishment", "description": "Major refurbishment activities"},
    {"module": "B6", "name": "Operational Energy", "description": "Energy used during operation"},
    {"module": "B7", "name": "Operational Water", "description": "Water used during operation"},
    {"module": "C1", "name": "Deconstruction", "description": "Demolition and deconstruction"},
    {"module": "C2", "name": "Waste Transport", "description": "Transport of waste"},
    {"module": "C3", "name": "Waste Processing", "description": "Waste treatment and processing"},
    {"module": "C4", "name": "Disposal", "description": "Final waste disposal"},
    {"module": "D", "name": "Beyond Lifecycle", "description": "Reuse, recovery, recycling potential"},
]


class WholeLifeCarbonReportTemplate:
    """
    Whole life carbon assessment report template.

    Renders whole life carbon reports with EN 15978 lifecycle stage
    analysis, material hotspots, operational projections, RIBA/LETI
    target comparisons, material substitution options, biogenic carbon
    accounting, and sensitivity analysis across markdown, HTML, and JSON.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    WLC_SECTIONS: List[str] = [
        "Summary",
        "Lifecycle Breakdown",
        "Material Hotspots",
        "Operational Projection",
        "WLC per m2",
        "RIBA/LETI Targets",
        "Material Substitution",
        "Biogenic Carbon",
        "Sensitivity Analysis",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize WholeLifeCarbonReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render whole life carbon report as Markdown.

        Args:
            data: WLC assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_lifecycle_breakdown(data),
            self._md_material_hotspots(data),
            self._md_operational_projection(data),
            self._md_wlc_per_m2(data),
            self._md_riba_leti_targets(data),
            self._md_material_substitution(data),
            self._md_biogenic_carbon(data),
            self._md_sensitivity_analysis(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render whole life carbon report as self-contained HTML.

        Args:
            data: WLC assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_summary(data),
            self._html_lifecycle_breakdown(data),
            self._html_material_hotspots(data),
            self._html_operational_projection(data),
            self._html_wlc_per_m2(data),
            self._html_riba_leti_targets(data),
            self._html_material_substitution(data),
            self._html_biogenic_carbon(data),
            self._html_sensitivity_analysis(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Whole Life Carbon Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render whole life carbon report as structured JSON.

        Args:
            data: WLC assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "whole_life_carbon_report",
            "version": "32.0.0",
            "standard": "EN_15978",
            "generated_at": self.generated_at.isoformat(),
            "summary": self._json_summary(data),
            "lifecycle_breakdown": data.get("lifecycle_breakdown", []),
            "material_hotspots": data.get("material_hotspots", []),
            "operational_projection": data.get("operational_projection", {}),
            "wlc_per_m2": data.get("wlc_per_m2", {}),
            "riba_leti_comparison": data.get("riba_leti_comparison", {}),
            "material_substitution": data.get("material_substitution", []),
            "biogenic_carbon": data.get("biogenic_carbon", {}),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
            "lifecycle_modules": LIFECYCLE_MODULES,
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Whole Life Carbon Report\n\n"
            f"**Building:** {name}  \n"
            f"**Address:** {data.get('address', '-')}  \n"
            f"**Reference Period:** {data.get('reference_period_years', 60)} years  \n"
            f"**Assessment Standard:** EN 15978 / RICS WLC  \n"
            f"**GIA:** {self._fmt(data.get('gia_sqm', 0), 0)} m2  \n"
            f"**Assessment Date:** {data.get('assessment_date', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 WholeLifeCarbonReportTemplate v32.0.0\n\n---"
        )

    def _md_summary(self, data: Dict[str, Any]) -> str:
        """Render summary section."""
        s = data.get("summary", {})
        return (
            "## 1. Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total WLC | {self._fmt(s.get('total_wlc_kgco2', 0), 0)} kgCO2e |\n"
            f"| WLC per m2 | {self._fmt(s.get('wlc_per_m2', 0))} kgCO2e/m2 |\n"
            f"| Embodied Carbon (A1-A5) | {self._fmt(s.get('embodied_a1_a5', 0), 0)} kgCO2e |\n"
            f"| Embodied per m2 | {self._fmt(s.get('embodied_per_m2', 0))} kgCO2e/m2 |\n"
            f"| Operational Carbon (B6-B7) | {self._fmt(s.get('operational_b6_b7', 0), 0)} kgCO2e |\n"
            f"| End of Life (C1-C4) | {self._fmt(s.get('end_of_life_c', 0), 0)} kgCO2e |\n"
            f"| Beyond Lifecycle (D) | {self._fmt(s.get('beyond_lifecycle_d', 0), 0)} kgCO2e |\n"
            f"| Embodied Share | {self._fmt(s.get('embodied_share_pct', 0))}% |\n"
            f"| Operational Share | {self._fmt(s.get('operational_share_pct', 0))}% |\n"
            f"| RIBA 2030 Target | {s.get('riba_2030_status', '-')} |\n"
            f"| LETI Target | {s.get('leti_status', '-')} |"
        )

    def _md_lifecycle_breakdown(self, data: Dict[str, Any]) -> str:
        """Render lifecycle stage breakdown (A1-D) section."""
        stages = data.get("lifecycle_breakdown", [])
        if not stages:
            return "## 2. Lifecycle Stage Breakdown (A1-D)\n\n_No lifecycle data._"
        lines = [
            "## 2. Lifecycle Stage Breakdown (A1-D)\n",
            "| Module | Stage | kgCO2e | kgCO2e/m2 | Share (%) |",
            "|--------|-------|--------|-----------|-----------|",
        ]
        for s in stages:
            lines.append(
                f"| {s.get('module', '-')} "
                f"| {s.get('stage_name', '-')} "
                f"| {self._fmt(s.get('kgco2e', 0), 0)} "
                f"| {self._fmt(s.get('kgco2e_m2', 0))} "
                f"| {self._fmt(s.get('share_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_material_hotspots(self, data: Dict[str, Any]) -> str:
        """Render material carbon hotspots section."""
        hotspots = data.get("material_hotspots", [])
        if not hotspots:
            return "## 3. Material Carbon Hotspots\n\n_No material data._"
        lines = [
            "## 3. Material Carbon Hotspots\n",
            "| Material | Building Element | kgCO2e | kgCO2e/m2 | Share (%) | EPD Available |",
            "|----------|-----------------|--------|-----------|-----------|--------------|",
        ]
        for h in hotspots:
            lines.append(
                f"| {h.get('material', '-')} "
                f"| {h.get('element', '-')} "
                f"| {self._fmt(h.get('kgco2e', 0), 0)} "
                f"| {self._fmt(h.get('kgco2e_m2', 0))} "
                f"| {self._fmt(h.get('share_pct', 0))}% "
                f"| {'Yes' if h.get('epd_available', False) else 'No'} |"
            )
        return "\n".join(lines)

    def _md_operational_projection(self, data: Dict[str, Any]) -> str:
        """Render operational carbon projection section."""
        op = data.get("operational_projection", {})
        decades = op.get("decade_breakdown", [])
        lines = [
            "## 4. Operational Carbon Projection\n",
            f"**Grid Decarbonization Scenario:** {op.get('grid_scenario', '-')}  ",
            f"**Annual Operational Carbon (Year 1):** {self._fmt(op.get('year1_kgco2e', 0), 0)} kgCO2e  ",
            f"**60-Year Total (B6):** {self._fmt(op.get('total_b6_kgco2e', 0), 0)} kgCO2e  ",
            f"**60-Year Total (B7):** {self._fmt(op.get('total_b7_kgco2e', 0), 0)} kgCO2e  ",
            f"**Grid Factor Applied:** {self._fmt(op.get('grid_factor', 0), 4)} kgCO2e/kWh",
        ]
        if decades:
            lines.extend([
                "\n### Decade Projections\n",
                "| Decade | B6 (kgCO2e) | B7 (kgCO2e) | Grid Factor | Cumulative |",
                "|--------|------------|------------|------------|-----------|",
            ])
            for d in decades:
                lines.append(
                    f"| {d.get('decade', '-')} "
                    f"| {self._fmt(d.get('b6_kgco2e', 0), 0)} "
                    f"| {self._fmt(d.get('b7_kgco2e', 0), 0)} "
                    f"| {self._fmt(d.get('grid_factor', 0), 4)} "
                    f"| {self._fmt(d.get('cumulative_kgco2e', 0), 0)} |"
                )
        return "\n".join(lines)

    def _md_wlc_per_m2(self, data: Dict[str, Any]) -> str:
        """Render whole life carbon per m2 section."""
        wlc = data.get("wlc_per_m2", {})
        components = wlc.get("components", [])
        lines = [
            "## 5. Whole Life Carbon per m2\n",
            f"**Total WLC/m2:** {self._fmt(wlc.get('total_per_m2', 0))} kgCO2e/m2  ",
            f"**Embodied/m2 (A-C):** {self._fmt(wlc.get('embodied_per_m2', 0))} kgCO2e/m2  ",
            f"**Operational/m2 (B6-B7):** {self._fmt(wlc.get('operational_per_m2', 0))} kgCO2e/m2  ",
            f"**Reference Period:** {wlc.get('reference_period', 60)} years",
        ]
        if components:
            lines.extend([
                "\n### Component Breakdown per m2\n",
                "| Component | kgCO2e/m2 | Share (%) |",
                "|-----------|----------|-----------|",
            ])
            for c in components:
                lines.append(
                    f"| {c.get('component', '-')} "
                    f"| {self._fmt(c.get('kgco2e_m2', 0))} "
                    f"| {self._fmt(c.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_riba_leti_targets(self, data: Dict[str, Any]) -> str:
        """Render comparison to RIBA/LETI targets section."""
        rl = data.get("riba_leti_comparison", {})
        targets = rl.get("targets", [])
        lines = [
            "## 6. Comparison to RIBA/LETI Targets\n",
            f"**Building Type:** {rl.get('building_type', '-')}",
        ]
        if targets:
            lines.extend([
                "\n| Target Framework | Target (kgCO2e/m2) | Actual | Gap | Status |",
                "|-----------------|-------------------|--------|-----|--------|",
            ])
            for t in targets:
                lines.append(
                    f"| {t.get('framework', '-')} "
                    f"| {self._fmt(t.get('target_per_m2', 0))} "
                    f"| {self._fmt(t.get('actual_per_m2', 0))} "
                    f"| {self._fmt(t.get('gap', 0))} "
                    f"| {t.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_material_substitution(self, data: Dict[str, Any]) -> str:
        """Render material substitution opportunities section."""
        subs = data.get("material_substitution", [])
        if not subs:
            return "## 7. Material Substitution Opportunities\n\n_No substitutions identified._"
        lines = [
            "## 7. Material Substitution Opportunities\n",
            "| Current Material | Alternative | Carbon Saving (kgCO2e) | Cost Impact | Feasibility |",
            "|-----------------|-----------|----------------------|------------|------------|",
        ]
        for s in subs:
            lines.append(
                f"| {s.get('current', '-')} "
                f"| {s.get('alternative', '-')} "
                f"| {self._fmt(s.get('carbon_saving_kgco2e', 0), 0)} "
                f"| {s.get('cost_impact', '-')} "
                f"| {s.get('feasibility', '-')} |"
            )
        return "\n".join(lines)

    def _md_biogenic_carbon(self, data: Dict[str, Any]) -> str:
        """Render biogenic carbon section."""
        bio = data.get("biogenic_carbon", {})
        materials = bio.get("materials", [])
        lines = [
            "## 8. Biogenic Carbon\n",
            f"**Total Biogenic Carbon Stored:** {self._fmt(bio.get('total_stored_kgco2', 0), 0)} kgCO2  ",
            f"**Biogenic Carbon per m2:** {self._fmt(bio.get('stored_per_m2', 0))} kgCO2/m2  ",
            f"**Net Biogenic Balance:** {self._fmt(bio.get('net_balance_kgco2', 0), 0)} kgCO2  ",
            f"**Reporting Method:** {bio.get('reporting_method', '-')}",
        ]
        if materials:
            lines.extend([
                "\n### Biogenic Materials\n",
                "| Material | Volume | Carbon Stored (kgCO2) | Source |",
                "|----------|--------|---------------------|--------|",
            ])
            for m in materials:
                lines.append(
                    f"| {m.get('material', '-')} "
                    f"| {m.get('volume', '-')} "
                    f"| {self._fmt(m.get('carbon_stored_kgco2', 0), 0)} "
                    f"| {m.get('source', '-')} |"
                )
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render sensitivity analysis section."""
        scenarios = data.get("sensitivity_analysis", [])
        if not scenarios:
            return "## 9. Sensitivity Analysis\n\n_No sensitivity analysis._"
        lines = [
            "## 9. Sensitivity Analysis\n",
            "| Scenario | Variable | Change | WLC Impact (kgCO2e) | WLC/m2 | % Change |",
            "|----------|---------|--------|--------------------|---------| ---------|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('scenario', '-')} "
                f"| {s.get('variable', '-')} "
                f"| {s.get('change', '-')} "
                f"| {self._fmt(s.get('wlc_impact_kgco2e', 0), 0)} "
                f"| {self._fmt(s.get('wlc_per_m2', 0))} "
                f"| {self._fmt(s.get('pct_change', 0))}% |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 WholeLifeCarbonReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Whole Life Carbon Report</h1>\n'
            f'<p class="subtitle">Building: {name} | Standard: EN 15978 | '
            f'GIA: {self._fmt(data.get("gia_sqm", 0), 0)} m2 | Generated: {ts}</p>'
        )

    def _html_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML summary with KPI cards."""
        s = data.get("summary", {})
        return (
            '<h2>Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">Total WLC</span>'
            f'<span class="value">{self._fmt(s.get("total_wlc_kgco2", 0), 0)}</span>'
            f'<span class="label">kgCO2e</span></div>\n'
            f'<div class="card"><span class="label">WLC/m2</span>'
            f'<span class="value">{self._fmt(s.get("wlc_per_m2", 0))}</span>'
            f'<span class="label">kgCO2e/m2</span></div>\n'
            f'<div class="card"><span class="label">Embodied</span>'
            f'<span class="value">{self._fmt(s.get("embodied_share_pct", 0))}%</span></div>\n'
            f'<div class="card"><span class="label">Operational</span>'
            f'<span class="value">{self._fmt(s.get("operational_share_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_lifecycle_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML lifecycle breakdown."""
        stages = data.get("lifecycle_breakdown", [])
        rows = ""
        for s in stages:
            rows += (
                f'<tr><td>{s.get("module", "-")}</td>'
                f'<td>{s.get("stage_name", "-")}</td>'
                f'<td>{self._fmt(s.get("kgco2e", 0), 0)}</td>'
                f'<td>{self._fmt(s.get("kgco2e_m2", 0))}</td>'
                f'<td>{self._fmt(s.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Lifecycle Stage Breakdown (A1-D)</h2>\n'
            '<table>\n<tr><th>Module</th><th>Stage</th><th>kgCO2e</th>'
            f'<th>kgCO2e/m2</th><th>Share</th></tr>\n{rows}</table>'
        )

    def _html_material_hotspots(self, data: Dict[str, Any]) -> str:
        """Render HTML material hotspots."""
        hotspots = data.get("material_hotspots", [])
        rows = ""
        for h in hotspots:
            rows += (
                f'<tr><td>{h.get("material", "-")}</td>'
                f'<td>{h.get("element", "-")}</td>'
                f'<td>{self._fmt(h.get("kgco2e", 0), 0)}</td>'
                f'<td>{self._fmt(h.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Material Carbon Hotspots</h2>\n'
            '<table>\n<tr><th>Material</th><th>Element</th><th>kgCO2e</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_operational_projection(self, data: Dict[str, Any]) -> str:
        """Render HTML operational projection."""
        op = data.get("operational_projection", {})
        decades = op.get("decade_breakdown", [])
        rows = ""
        for d in decades:
            rows += (
                f'<tr><td>{d.get("decade", "-")}</td>'
                f'<td>{self._fmt(d.get("b6_kgco2e", 0), 0)}</td>'
                f'<td>{self._fmt(d.get("grid_factor", 0), 4)}</td>'
                f'<td>{self._fmt(d.get("cumulative_kgco2e", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Operational Carbon Projection</h2>\n'
            f'<p>Grid Scenario: {op.get("grid_scenario", "-")} | '
            f'Year 1: {self._fmt(op.get("year1_kgco2e", 0), 0)} kgCO2e</p>\n'
            '<table>\n<tr><th>Decade</th><th>B6 (kgCO2e)</th><th>Grid Factor</th>'
            f'<th>Cumulative</th></tr>\n{rows}</table>'
        )

    def _html_wlc_per_m2(self, data: Dict[str, Any]) -> str:
        """Render HTML WLC per m2."""
        wlc = data.get("wlc_per_m2", {})
        components = wlc.get("components", [])
        rows = ""
        for c in components:
            rows += (
                f'<tr><td>{c.get("component", "-")}</td>'
                f'<td>{self._fmt(c.get("kgco2e_m2", 0))}</td>'
                f'<td>{self._fmt(c.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Whole Life Carbon per m2</h2>\n'
            f'<p>Total: {self._fmt(wlc.get("total_per_m2", 0))} kgCO2e/m2 | '
            f'Embodied: {self._fmt(wlc.get("embodied_per_m2", 0))} | '
            f'Operational: {self._fmt(wlc.get("operational_per_m2", 0))}</p>\n'
            '<table>\n<tr><th>Component</th><th>kgCO2e/m2</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_riba_leti_targets(self, data: Dict[str, Any]) -> str:
        """Render HTML RIBA/LETI target comparison."""
        rl = data.get("riba_leti_comparison", {})
        targets = rl.get("targets", [])
        rows = ""
        for t in targets:
            status = t.get("status", "-")
            style = 'color:#198754' if status == "Met" else 'color:#dc3545'
            rows += (
                f'<tr><td>{t.get("framework", "-")}</td>'
                f'<td>{self._fmt(t.get("target_per_m2", 0))}</td>'
                f'<td>{self._fmt(t.get("actual_per_m2", 0))}</td>'
                f'<td style="{style};font-weight:bold">{status}</td></tr>\n'
            )
        return (
            '<h2>Comparison to RIBA/LETI Targets</h2>\n'
            '<table>\n<tr><th>Framework</th><th>Target (kgCO2e/m2)</th>'
            f'<th>Actual</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_material_substitution(self, data: Dict[str, Any]) -> str:
        """Render HTML material substitution."""
        subs = data.get("material_substitution", [])
        rows = ""
        for s in subs:
            rows += (
                f'<tr><td>{s.get("current", "-")}</td>'
                f'<td>{s.get("alternative", "-")}</td>'
                f'<td>{self._fmt(s.get("carbon_saving_kgco2e", 0), 0)}</td>'
                f'<td>{s.get("feasibility", "-")}</td></tr>\n'
            )
        return (
            '<h2>Material Substitution Opportunities</h2>\n'
            '<table>\n<tr><th>Current</th><th>Alternative</th>'
            f'<th>Saving (kgCO2e)</th><th>Feasibility</th></tr>\n{rows}</table>'
        )

    def _html_biogenic_carbon(self, data: Dict[str, Any]) -> str:
        """Render HTML biogenic carbon."""
        bio = data.get("biogenic_carbon", {})
        materials = bio.get("materials", [])
        rows = ""
        for m in materials:
            rows += (
                f'<tr><td>{m.get("material", "-")}</td>'
                f'<td>{m.get("volume", "-")}</td>'
                f'<td>{self._fmt(m.get("carbon_stored_kgco2", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Biogenic Carbon</h2>\n'
            f'<p>Total Stored: {self._fmt(bio.get("total_stored_kgco2", 0), 0)} kgCO2 | '
            f'Per m2: {self._fmt(bio.get("stored_per_m2", 0))} kgCO2/m2</p>\n'
            '<table>\n<tr><th>Material</th><th>Volume</th>'
            f'<th>Carbon Stored (kgCO2)</th></tr>\n{rows}</table>'
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis."""
        scenarios = data.get("sensitivity_analysis", [])
        rows = ""
        for s in scenarios:
            rows += (
                f'<tr><td>{s.get("scenario", "-")}</td>'
                f'<td>{s.get("variable", "-")}</td>'
                f'<td>{s.get("change", "-")}</td>'
                f'<td>{self._fmt(s.get("wlc_per_m2", 0))}</td>'
                f'<td>{self._fmt(s.get("pct_change", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Sensitivity Analysis</h2>\n'
            '<table>\n<tr><th>Scenario</th><th>Variable</th><th>Change</th>'
            f'<th>WLC/m2</th><th>% Change</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON summary."""
        s = data.get("summary", {})
        return {
            "total_wlc_kgco2": s.get("total_wlc_kgco2", 0),
            "wlc_per_m2": s.get("wlc_per_m2", 0),
            "embodied_a1_a5": s.get("embodied_a1_a5", 0),
            "embodied_per_m2": s.get("embodied_per_m2", 0),
            "operational_b6_b7": s.get("operational_b6_b7", 0),
            "end_of_life_c": s.get("end_of_life_c", 0),
            "beyond_lifecycle_d": s.get("beyond_lifecycle_d", 0),
            "embodied_share_pct": s.get("embodied_share_pct", 0),
            "operational_share_pct": s.get("operational_share_pct", 0),
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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
