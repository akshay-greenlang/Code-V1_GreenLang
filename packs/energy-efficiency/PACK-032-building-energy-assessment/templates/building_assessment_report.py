# -*- coding: utf-8 -*-
"""
BuildingAssessmentReportTemplate - Comprehensive building energy assessment for PACK-032.

Generates a comprehensive building energy assessment report covering
envelope performance, HVAC systems, lighting, domestic hot water,
renewable integration, indoor environmental quality, benchmark
comparisons, improvement recommendations, and implementation roadmaps.

Sections:
    1.  Executive Summary
    2.  Building Description
    3.  Envelope Assessment
    4.  HVAC Assessment
    5.  Lighting Assessment
    6.  Domestic Hot Water Assessment
    7.  Renewable Energy Assessment
    8.  Indoor Environmental Quality
    9.  Benchmark Comparison
    10. Improvement Recommendations
    11. Implementation Roadmap
    12. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BuildingAssessmentReportTemplate:
    """
    Comprehensive building energy assessment report template.

    Renders full building energy assessments with envelope, HVAC,
    lighting, DHW, renewables, IEQ, benchmarking, recommendations,
    and implementation roadmaps across markdown, HTML, and JSON.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    ASSESSMENT_SECTIONS: List[str] = [
        "Executive Summary",
        "Building Description",
        "Envelope Assessment",
        "HVAC Assessment",
        "Lighting Assessment",
        "DHW Assessment",
        "Renewable Assessment",
        "Indoor Environment",
        "Benchmark Comparison",
        "Improvement Recommendations",
        "Implementation Roadmap",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BuildingAssessmentReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render comprehensive building assessment as Markdown.

        Args:
            data: Assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_building_description(data),
            self._md_envelope(data),
            self._md_hvac(data),
            self._md_lighting(data),
            self._md_dhw(data),
            self._md_renewables(data),
            self._md_ieq(data),
            self._md_benchmark(data),
            self._md_recommendations(data),
            self._md_roadmap(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render comprehensive building assessment as HTML.

        Args:
            data: Assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_building_description(data),
            self._html_envelope(data),
            self._html_hvac(data),
            self._html_lighting(data),
            self._html_dhw(data),
            self._html_renewables(data),
            self._html_ieq(data),
            self._html_benchmark(data),
            self._html_recommendations(data),
            self._html_roadmap(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Comprehensive Building Energy Assessment</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render comprehensive building assessment as JSON.

        Args:
            data: Assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "building_assessment_report",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "building": data.get("building", {}),
            "envelope": data.get("envelope", {}),
            "hvac": data.get("hvac", {}),
            "lighting": data.get("lighting", {}),
            "dhw": data.get("dhw", {}),
            "renewables": data.get("renewables", {}),
            "indoor_environment": data.get("indoor_environment", {}),
            "benchmark": data.get("benchmark", {}),
            "recommendations": data.get("recommendations", []),
            "roadmap": data.get("roadmap", []),
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
        address = data.get("address", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Comprehensive Building Energy Assessment\n\n"
            f"**Building:** {name}  \n"
            f"**Address:** {address}  \n"
            f"**Assessor:** {data.get('assessor', '-')}  \n"
            f"**Assessment Date:** {data.get('assessment_date', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 BuildingAssessmentReportTemplate v32.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary section."""
        summary = data.get("executive_summary", {})
        eui = summary.get("eui_kwh_m2", 0)
        co2 = summary.get("co2_kg_m2", 0)
        savings_pct = summary.get("savings_potential_pct", 0)
        investment = summary.get("total_investment", 0)
        payback = summary.get("avg_payback_years", 0)
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Energy Use Intensity | {self._fmt(eui)} kWh/m2/yr |\n"
            f"| Carbon Intensity | {self._fmt(co2)} kgCO2/m2/yr |\n"
            f"| EPC Rating | {summary.get('epc_rating', '-')} |\n"
            f"| DEC Rating | {summary.get('dec_rating', '-')} |\n"
            f"| Savings Potential | {self._fmt(savings_pct)}% |\n"
            f"| Estimated Investment | {self._fmt(investment, 0)} |\n"
            f"| Average Payback | {self._fmt(payback, 1)} years |\n"
            f"| Recommendations Count | {summary.get('recommendations_count', 0)} |\n"
            f"| Priority Actions | {summary.get('priority_actions', 0)} |"
        )

    def _md_building_description(self, data: Dict[str, Any]) -> str:
        """Render building description section."""
        bld = data.get("building", {})
        return (
            "## 2. Building Description\n\n"
            f"| Property | Value |\n|----------|-------|\n"
            f"| Type | {bld.get('type', '-')} |\n"
            f"| Use Class | {bld.get('use_class', '-')} |\n"
            f"| Gross Internal Area | {self._fmt(bld.get('gia_sqm', 0), 0)} m2 |\n"
            f"| Net Lettable Area | {self._fmt(bld.get('nla_sqm', 0), 0)} m2 |\n"
            f"| Year Built | {bld.get('year_built', '-')} |\n"
            f"| Last Major Refurb | {bld.get('last_refurb', '-')} |\n"
            f"| Number of Floors | {bld.get('num_floors', '-')} |\n"
            f"| Occupancy Rate | {self._fmt(bld.get('occupancy_pct', 0))}% |\n"
            f"| Operating Hours | {self._fmt(bld.get('operating_hours_yr', 0), 0)} hrs/yr |\n"
            f"| Climate Zone | {bld.get('climate_zone', '-')} |"
        )

    def _md_envelope(self, data: Dict[str, Any]) -> str:
        """Render envelope assessment section."""
        env = data.get("envelope", {})
        components = env.get("components", [])
        lines = [
            "## 3. Envelope Assessment\n",
            f"**Overall U-Value (area-weighted):** {self._fmt(env.get('avg_u_value', 0), 3)} W/m2K  ",
            f"**Air Permeability:** {self._fmt(env.get('air_permeability', 0))} m3/hr/m2 @50Pa  ",
            f"**Thermal Bridging Factor:** {self._fmt(env.get('thermal_bridging', 0), 3)} W/m2K  ",
            f"**Condensation Risk:** {env.get('condensation_risk', '-')}",
        ]
        if components:
            lines.extend([
                "\n### Component Performance\n",
                "| Component | Area (m2) | U-Value (W/m2K) | Target | Gap | Condition |",
                "|-----------|----------|-----------------|--------|-----|-----------|",
            ])
            for c in components:
                lines.append(
                    f"| {c.get('component', '-')} "
                    f"| {self._fmt(c.get('area_sqm', 0), 0)} "
                    f"| {self._fmt(c.get('u_value', 0), 3)} "
                    f"| {self._fmt(c.get('target_u_value', 0), 3)} "
                    f"| {self._fmt(c.get('gap', 0), 3)} "
                    f"| {c.get('condition', '-')} |"
                )
        thermal_images = env.get("thermal_images", [])
        if thermal_images:
            lines.append("\n### Thermal Imaging Findings\n")
            for img in thermal_images:
                lines.append(
                    f"- **{img.get('location', '-')}**: {img.get('finding', '-')} "
                    f"(Delta T: {self._fmt(img.get('delta_t', 0), 1)} C)"
                )
        return "\n".join(lines)

    def _md_hvac(self, data: Dict[str, Any]) -> str:
        """Render HVAC assessment section."""
        hvac = data.get("hvac", {})
        systems = hvac.get("systems", [])
        lines = [
            "## 4. HVAC Assessment\n",
            f"**Total HVAC Energy:** {self._fmt(hvac.get('total_kwh', 0), 0)} kWh/yr  ",
            f"**HVAC Intensity:** {self._fmt(hvac.get('kwh_m2', 0))} kWh/m2/yr  ",
            f"**Share of Total Energy:** {self._fmt(hvac.get('share_pct', 0))}%  ",
            f"**Overall Efficiency Rating:** {hvac.get('efficiency_rating', '-')}",
        ]
        if systems:
            lines.extend([
                "\n### HVAC Systems\n",
                "| System | Type | Capacity | Age (yr) | Efficiency | Condition |",
                "|--------|------|----------|----------|------------|-----------|",
            ])
            for s in systems:
                lines.append(
                    f"| {s.get('name', '-')} "
                    f"| {s.get('type', '-')} "
                    f"| {s.get('capacity', '-')} "
                    f"| {s.get('age_years', '-')} "
                    f"| {s.get('efficiency', '-')} "
                    f"| {s.get('condition', '-')} |"
                )
        controls = hvac.get("controls", {})
        if controls:
            lines.extend([
                "\n### Controls Assessment\n",
                f"- **BMS System:** {controls.get('bms', '-')}",
                f"- **Zoning:** {controls.get('zoning', '-')}",
                f"- **Scheduling:** {controls.get('scheduling', '-')}",
                f"- **Setpoint Optimization:** {controls.get('setpoint_optimization', '-')}",
            ])
        return "\n".join(lines)

    def _md_lighting(self, data: Dict[str, Any]) -> str:
        """Render lighting assessment section."""
        lighting = data.get("lighting", {})
        zones = lighting.get("zones", [])
        lines = [
            "## 5. Lighting Assessment\n",
            f"**Total Lighting Energy:** {self._fmt(lighting.get('total_kwh', 0), 0)} kWh/yr  ",
            f"**Lighting Power Density:** {self._fmt(lighting.get('lpd_w_m2', 0))} W/m2  ",
            f"**Target LPD:** {self._fmt(lighting.get('target_lpd', 0))} W/m2  ",
            f"**LED Share:** {self._fmt(lighting.get('led_share_pct', 0))}%  ",
            f"**Controls Installed:** {lighting.get('controls_installed', '-')}",
        ]
        if zones:
            lines.extend([
                "\n### Zone Analysis\n",
                "| Zone | Area (m2) | LPD (W/m2) | Target | LED% | Controls |",
                "|------|----------|-----------|--------|------|----------|",
            ])
            for z in zones:
                lines.append(
                    f"| {z.get('zone', '-')} "
                    f"| {self._fmt(z.get('area_sqm', 0), 0)} "
                    f"| {self._fmt(z.get('lpd', 0))} "
                    f"| {self._fmt(z.get('target_lpd', 0))} "
                    f"| {self._fmt(z.get('led_pct', 0))}% "
                    f"| {z.get('controls', '-')} |"
                )
        return "\n".join(lines)

    def _md_dhw(self, data: Dict[str, Any]) -> str:
        """Render domestic hot water assessment section."""
        dhw = data.get("dhw", {})
        systems = dhw.get("systems", [])
        lines = [
            "## 6. Domestic Hot Water Assessment\n",
            f"**Total DHW Energy:** {self._fmt(dhw.get('total_kwh', 0), 0)} kWh/yr  ",
            f"**DHW Share of Total:** {self._fmt(dhw.get('share_pct', 0))}%  ",
            f"**Distribution Losses:** {self._fmt(dhw.get('distribution_losses_pct', 0))}%  ",
            f"**Storage Losses:** {self._fmt(dhw.get('storage_losses_kwh', 0), 0)} kWh/yr",
        ]
        if systems:
            lines.extend([
                "\n### DHW Systems\n",
                "| System | Type | Fuel | Capacity | Efficiency | Age (yr) |",
                "|--------|------|------|----------|------------|----------|",
            ])
            for s in systems:
                lines.append(
                    f"| {s.get('name', '-')} "
                    f"| {s.get('type', '-')} "
                    f"| {s.get('fuel', '-')} "
                    f"| {s.get('capacity', '-')} "
                    f"| {s.get('efficiency', '-')} "
                    f"| {s.get('age_years', '-')} |"
                )
        return "\n".join(lines)

    def _md_renewables(self, data: Dict[str, Any]) -> str:
        """Render renewable energy assessment section."""
        ren = data.get("renewables", {})
        systems = ren.get("existing", [])
        potential = ren.get("potential", [])
        lines = [
            "## 7. Renewable Energy Assessment\n",
            f"**Current Renewable Generation:** {self._fmt(ren.get('current_kwh', 0), 0)} kWh/yr  ",
            f"**Renewable Fraction:** {self._fmt(ren.get('fraction_pct', 0))}%  ",
            f"**Roof Area Available:** {self._fmt(ren.get('roof_area_sqm', 0), 0)} m2",
        ]
        if systems:
            lines.extend([
                "\n### Existing Systems\n",
                "| System | Capacity | Generation (kWh/yr) | Performance Ratio |",
                "|--------|----------|--------------------|--------------------|",
            ])
            for s in systems:
                lines.append(
                    f"| {s.get('system', '-')} "
                    f"| {s.get('capacity', '-')} "
                    f"| {self._fmt(s.get('generation_kwh', 0), 0)} "
                    f"| {self._fmt(s.get('performance_ratio', 0))}% |"
                )
        if potential:
            lines.extend([
                "\n### Potential Additions\n",
                "| Technology | Potential (kWh/yr) | Cost | Payback (yr) | CO2 Savings |",
                "|------------|-------------------|------|-------------|------------|",
            ])
            for p in potential:
                lines.append(
                    f"| {p.get('technology', '-')} "
                    f"| {self._fmt(p.get('potential_kwh', 0), 0)} "
                    f"| {p.get('cost', '-')} "
                    f"| {self._fmt(p.get('payback_years', 0), 1)} "
                    f"| {p.get('co2_savings', '-')} |"
                )
        return "\n".join(lines)

    def _md_ieq(self, data: Dict[str, Any]) -> str:
        """Render indoor environmental quality section."""
        ieq = data.get("indoor_environment", {})
        zones = ieq.get("zones", [])
        lines = [
            "## 8. Indoor Environmental Quality\n",
            f"**Overall Comfort Score:** {ieq.get('comfort_score', '-')}/10  ",
            f"**Temperature Compliance:** {self._fmt(ieq.get('temp_compliance_pct', 0))}%  ",
            f"**CO2 Levels (avg):** {self._fmt(ieq.get('avg_co2_ppm', 0), 0)} ppm  ",
            f"**Daylight Factor (avg):** {self._fmt(ieq.get('avg_daylight_factor', 0))}%  ",
            f"**Acoustics:** {ieq.get('acoustics_rating', '-')}",
        ]
        if zones:
            lines.extend([
                "\n### Zone Comfort Analysis\n",
                "| Zone | Temp (C) | Humidity (%) | CO2 (ppm) | Lux | Rating |",
                "|------|----------|-------------|-----------|-----|--------|",
            ])
            for z in zones:
                lines.append(
                    f"| {z.get('zone', '-')} "
                    f"| {self._fmt(z.get('temp_c', 0), 1)} "
                    f"| {self._fmt(z.get('humidity_pct', 0))} "
                    f"| {self._fmt(z.get('co2_ppm', 0), 0)} "
                    f"| {self._fmt(z.get('lux', 0), 0)} "
                    f"| {z.get('rating', '-')} |"
                )
        return "\n".join(lines)

    def _md_benchmark(self, data: Dict[str, Any]) -> str:
        """Render benchmark comparison section."""
        bm = data.get("benchmark", {})
        comparisons = bm.get("comparisons", [])
        lines = [
            "## 9. Benchmark Comparison\n",
            f"**Building EUI:** {self._fmt(bm.get('eui_kwh_m2', 0))} kWh/m2/yr  ",
            f"**Sector Median:** {self._fmt(bm.get('sector_median', 0))} kWh/m2/yr  ",
            f"**Best Practice:** {self._fmt(bm.get('best_practice', 0))} kWh/m2/yr  ",
            f"**Percentile Ranking:** {bm.get('percentile', '-')}th",
        ]
        if comparisons:
            lines.extend([
                "\n### Benchmark Comparisons\n",
                "| Benchmark | Target | Actual | Gap | Status |",
                "|-----------|--------|--------|-----|--------|",
            ])
            for c in comparisons:
                lines.append(
                    f"| {c.get('benchmark', '-')} "
                    f"| {self._fmt(c.get('target', 0))} "
                    f"| {self._fmt(c.get('actual', 0))} "
                    f"| {self._fmt(c.get('gap', 0))} "
                    f"| {c.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render improvement recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 10. Improvement Recommendations\n\n_No recommendations._"
        lines = [
            "## 10. Improvement Recommendations\n",
            "| # | Measure | Category | Savings (kWh) | Cost | Payback (yr) | Priority |",
            "|---|---------|----------|--------------|------|-------------|----------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('measure', '-')} "
                f"| {r.get('category', '-')} "
                f"| {self._fmt(r.get('savings_kwh', 0), 0)} "
                f"| {r.get('cost', '-')} "
                f"| {self._fmt(r.get('payback_years', 0), 1)} "
                f"| {r.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_roadmap(self, data: Dict[str, Any]) -> str:
        """Render implementation roadmap section."""
        phases = data.get("roadmap", [])
        if not phases:
            return "## 11. Implementation Roadmap\n\n_No roadmap defined._"
        lines = ["## 11. Implementation Roadmap\n"]
        for phase in phases:
            lines.extend([
                f"### {phase.get('phase', 'Phase')} - {phase.get('timeframe', '-')}\n",
                f"**Budget:** {phase.get('budget', '-')}  ",
                f"**Expected Savings:** {phase.get('expected_savings', '-')}\n",
            ])
            actions = phase.get("actions", [])
            for a in actions:
                lines.append(
                    f"- {a.get('action', '-')} (Owner: {a.get('owner', '-')})"
                )
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 BuildingAssessmentReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Comprehensive Building Energy Assessment</h1>\n'
            f'<p class="subtitle">Building: {name} | '
            f'Assessor: {data.get("assessor", "-")} | Generated: {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary with KPI cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">EUI</span>'
            f'<span class="value">{self._fmt(s.get("eui_kwh_m2", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'<div class="card"><span class="label">Carbon</span>'
            f'<span class="value">{self._fmt(s.get("co2_kg_m2", 0))}</span>'
            f'<span class="label">kgCO2/m2/yr</span></div>\n'
            f'<div class="card"><span class="label">Savings</span>'
            f'<span class="value">{self._fmt(s.get("savings_potential_pct", 0))}%</span>'
            f'<span class="label">potential</span></div>\n'
            f'<div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(s.get("avg_payback_years", 0), 1)}</span>'
            f'<span class="label">years</span></div>\n'
            '</div>'
        )

    def _html_building_description(self, data: Dict[str, Any]) -> str:
        """Render HTML building description."""
        bld = data.get("building", {})
        fields = [
            ("Type", bld.get("type", "-")),
            ("GIA", f"{self._fmt(bld.get('gia_sqm', 0), 0)} m2"),
            ("Year Built", bld.get("year_built", "-")),
            ("Floors", bld.get("num_floors", "-")),
            ("Climate Zone", bld.get("climate_zone", "-")),
        ]
        rows = "".join(f'<tr><td>{l}</td><td>{v}</td></tr>\n' for l, v in fields)
        return (
            '<h2>Building Description</h2>\n'
            f'<table>\n<tr><th>Property</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_envelope(self, data: Dict[str, Any]) -> str:
        """Render HTML envelope assessment."""
        env = data.get("envelope", {})
        components = env.get("components", [])
        rows = ""
        for c in components:
            rows += (
                f'<tr><td>{c.get("component", "-")}</td>'
                f'<td>{self._fmt(c.get("u_value", 0), 3)}</td>'
                f'<td>{self._fmt(c.get("target_u_value", 0), 3)}</td>'
                f'<td>{c.get("condition", "-")}</td></tr>\n'
            )
        return (
            '<h2>Envelope Assessment</h2>\n'
            f'<p>Avg U-Value: {self._fmt(env.get("avg_u_value", 0), 3)} W/m2K | '
            f'Air Permeability: {self._fmt(env.get("air_permeability", 0))} m3/hr/m2</p>\n'
            '<table>\n<tr><th>Component</th><th>U-Value</th><th>Target</th>'
            f'<th>Condition</th></tr>\n{rows}</table>'
        )

    def _html_hvac(self, data: Dict[str, Any]) -> str:
        """Render HTML HVAC assessment."""
        hvac = data.get("hvac", {})
        systems = hvac.get("systems", [])
        rows = ""
        for s in systems:
            rows += (
                f'<tr><td>{s.get("name", "-")}</td>'
                f'<td>{s.get("type", "-")}</td>'
                f'<td>{s.get("capacity", "-")}</td>'
                f'<td>{s.get("efficiency", "-")}</td>'
                f'<td>{s.get("condition", "-")}</td></tr>\n'
            )
        return (
            '<h2>HVAC Assessment</h2>\n'
            f'<p>Total: {self._fmt(hvac.get("total_kwh", 0), 0)} kWh/yr | '
            f'Intensity: {self._fmt(hvac.get("kwh_m2", 0))} kWh/m2/yr</p>\n'
            '<table>\n<tr><th>System</th><th>Type</th><th>Capacity</th>'
            f'<th>Efficiency</th><th>Condition</th></tr>\n{rows}</table>'
        )

    def _html_lighting(self, data: Dict[str, Any]) -> str:
        """Render HTML lighting assessment."""
        lighting = data.get("lighting", {})
        zones = lighting.get("zones", [])
        rows = ""
        for z in zones:
            rows += (
                f'<tr><td>{z.get("zone", "-")}</td>'
                f'<td>{self._fmt(z.get("lpd", 0))}</td>'
                f'<td>{self._fmt(z.get("target_lpd", 0))}</td>'
                f'<td>{self._fmt(z.get("led_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Lighting Assessment</h2>\n'
            f'<p>LPD: {self._fmt(lighting.get("lpd_w_m2", 0))} W/m2 | '
            f'LED Share: {self._fmt(lighting.get("led_share_pct", 0))}%</p>\n'
            '<table>\n<tr><th>Zone</th><th>LPD (W/m2)</th><th>Target</th>'
            f'<th>LED%</th></tr>\n{rows}</table>'
        )

    def _html_dhw(self, data: Dict[str, Any]) -> str:
        """Render HTML DHW assessment."""
        dhw = data.get("dhw", {})
        systems = dhw.get("systems", [])
        rows = ""
        for s in systems:
            rows += (
                f'<tr><td>{s.get("name", "-")}</td>'
                f'<td>{s.get("type", "-")}</td>'
                f'<td>{s.get("fuel", "-")}</td>'
                f'<td>{s.get("efficiency", "-")}</td></tr>\n'
            )
        return (
            '<h2>Domestic Hot Water Assessment</h2>\n'
            f'<p>Total: {self._fmt(dhw.get("total_kwh", 0), 0)} kWh/yr | '
            f'Share: {self._fmt(dhw.get("share_pct", 0))}%</p>\n'
            '<table>\n<tr><th>System</th><th>Type</th><th>Fuel</th>'
            f'<th>Efficiency</th></tr>\n{rows}</table>'
        )

    def _html_renewables(self, data: Dict[str, Any]) -> str:
        """Render HTML renewables assessment."""
        ren = data.get("renewables", {})
        existing = ren.get("existing", [])
        rows = ""
        for s in existing:
            rows += (
                f'<tr><td>{s.get("system", "-")}</td>'
                f'<td>{s.get("capacity", "-")}</td>'
                f'<td>{self._fmt(s.get("generation_kwh", 0), 0)}</td></tr>\n'
            )
        return (
            '<h2>Renewable Energy Assessment</h2>\n'
            f'<p>Current: {self._fmt(ren.get("current_kwh", 0), 0)} kWh/yr | '
            f'Fraction: {self._fmt(ren.get("fraction_pct", 0))}%</p>\n'
            '<table>\n<tr><th>System</th><th>Capacity</th>'
            f'<th>Generation (kWh/yr)</th></tr>\n{rows}</table>'
        )

    def _html_ieq(self, data: Dict[str, Any]) -> str:
        """Render HTML indoor environment quality."""
        ieq = data.get("indoor_environment", {})
        zones = ieq.get("zones", [])
        rows = ""
        for z in zones:
            rows += (
                f'<tr><td>{z.get("zone", "-")}</td>'
                f'<td>{self._fmt(z.get("temp_c", 0), 1)}</td>'
                f'<td>{self._fmt(z.get("co2_ppm", 0), 0)}</td>'
                f'<td>{z.get("rating", "-")}</td></tr>\n'
            )
        return (
            '<h2>Indoor Environmental Quality</h2>\n'
            f'<p>Comfort Score: {ieq.get("comfort_score", "-")}/10 | '
            f'Temp Compliance: {self._fmt(ieq.get("temp_compliance_pct", 0))}%</p>\n'
            '<table>\n<tr><th>Zone</th><th>Temp (C)</th><th>CO2 (ppm)</th>'
            f'<th>Rating</th></tr>\n{rows}</table>'
        )

    def _html_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark comparison."""
        bm = data.get("benchmark", {})
        comparisons = bm.get("comparisons", [])
        rows = ""
        for c in comparisons:
            rows += (
                f'<tr><td>{c.get("benchmark", "-")}</td>'
                f'<td>{self._fmt(c.get("target", 0))}</td>'
                f'<td>{self._fmt(c.get("actual", 0))}</td>'
                f'<td>{c.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Benchmark Comparison</h2>\n'
            f'<p>EUI: {self._fmt(bm.get("eui_kwh_m2", 0))} kWh/m2/yr | '
            f'Percentile: {bm.get("percentile", "-")}th</p>\n'
            '<table>\n<tr><th>Benchmark</th><th>Target</th><th>Actual</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement recommendations."""
        recs = data.get("recommendations", [])
        rows = ""
        for i, r in enumerate(recs, 1):
            rows += (
                f'<tr><td>{i}</td><td>{r.get("measure", "-")}</td>'
                f'<td>{r.get("category", "-")}</td>'
                f'<td>{self._fmt(r.get("savings_kwh", 0), 0)}</td>'
                f'<td>{r.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Improvement Recommendations</h2>\n'
            '<table>\n<tr><th>#</th><th>Measure</th><th>Category</th>'
            f'<th>Savings (kWh)</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_roadmap(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation roadmap."""
        phases = data.get("roadmap", [])
        content = ""
        for phase in phases:
            actions = "".join(
                f'<li>{a.get("action", "-")} (Owner: {a.get("owner", "-")})</li>'
                for a in phase.get("actions", [])
            )
            content += (
                f'<div class="phase"><h3>{phase.get("phase", "Phase")} - '
                f'{phase.get("timeframe", "-")}</h3>'
                f'<p>Budget: {phase.get("budget", "-")}</p>'
                f'<ul>{actions}</ul></div>\n'
            )
        return f'<h2>Implementation Roadmap</h2>\n{content}'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        s = data.get("executive_summary", {})
        return {
            "eui_kwh_m2": s.get("eui_kwh_m2", 0),
            "co2_kg_m2": s.get("co2_kg_m2", 0),
            "epc_rating": s.get("epc_rating", ""),
            "dec_rating": s.get("dec_rating", ""),
            "savings_potential_pct": s.get("savings_potential_pct", 0),
            "total_investment": s.get("total_investment", 0),
            "avg_payback_years": s.get("avg_payback_years", 0),
            "recommendations_count": s.get("recommendations_count", 0),
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
