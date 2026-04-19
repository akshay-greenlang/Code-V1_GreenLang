# -*- coding: utf-8 -*-
"""
CompressedAirReportTemplate - Compressed air system audit report for PACK-031.

Generates comprehensive compressed air system audit reports with system
inventory, specific power analysis, leak survey results, pressure profile
assessment, VSD (Variable Speed Drive) analysis, and optimization
recommendations following ISO 11011:2013 methodology.

Sections:
    1. Executive Summary
    2. System Inventory
    3. Specific Power Analysis
    4. Leak Survey Results
    5. Pressure Profile Assessment
    6. VSD Opportunity Analysis
    7. Air Quality & Treatment
    8. Optimization Recommendations

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CompressedAirReportTemplate:
    """
    Compressed air system audit report template.

    Renders comprehensive compressed air audit data including compressor
    inventory, specific power, leak detection, pressure profiles, and
    VSD optimization analysis across markdown, HTML, and JSON formats.
    Follows ISO 11011:2013 compressed air energy audit methodology.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    BENCHMARK_SPECIFIC_POWER: Dict[str, float] = {
        "best_practice_kw_per_100cfm": 18.0,
        "good_kw_per_100cfm": 22.0,
        "average_kw_per_100cfm": 25.0,
        "poor_kw_per_100cfm": 30.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CompressedAirReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render compressed air audit report as Markdown.

        Args:
            data: Compressed air engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_system_inventory(data),
            self._md_specific_power(data),
            self._md_leak_survey(data),
            self._md_pressure_profile(data),
            self._md_vsd_analysis(data),
            self._md_air_quality(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render compressed air audit report as HTML.

        Args:
            data: Compressed air engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_system_inventory(data),
            self._html_specific_power(data),
            self._html_leak_survey(data),
            self._html_pressure_profile(data),
            self._html_vsd_analysis(data),
            self._html_recommendations(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Compressed Air System Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render compressed air report as structured JSON.

        Args:
            data: Compressed air engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "compressed_air_report",
            "version": "31.0.0",
            "standard": "ISO 11011:2013",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "system_inventory": data.get("system_inventory", {}),
            "specific_power": data.get("specific_power", {}),
            "leak_survey": data.get("leak_survey", {}),
            "pressure_profile": data.get("pressure_profile", {}),
            "vsd_analysis": data.get("vsd_analysis", {}),
            "air_quality": data.get("air_quality", {}),
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
            f"# Compressed Air System Audit Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Standard:** ISO 11011:2013  \n"
            f"**Audit Date:** {data.get('audit_date', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 CompressedAirReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        summary = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Compressor Capacity | {self._fmt(summary.get('total_capacity_kw', 0))} kW |\n"
            f"| Total Rated Flow | {self._fmt(summary.get('total_flow_cfm', 0))} CFM |\n"
            f"| Annual Energy Consumption | {self._fmt(summary.get('annual_energy_mwh', 0))} MWh |\n"
            f"| Annual Energy Cost | EUR {self._fmt(summary.get('annual_cost_eur', 0))} |\n"
            f"| System Specific Power | {self._fmt(summary.get('system_specific_power', 0))} kW/100CFM |\n"
            f"| Leak Rate | {self._fmt(summary.get('leak_rate_pct', 0))}% |\n"
            f"| Identified Savings | {self._fmt(summary.get('total_savings_mwh', 0))} MWh/yr "
            f"(EUR {self._fmt(summary.get('total_savings_eur', 0))}) |\n"
            f"| Required Investment | EUR {self._fmt(summary.get('total_investment_eur', 0))} |"
        )

    def _md_system_inventory(self, data: Dict[str, Any]) -> str:
        """Render system inventory section."""
        inventory = data.get("system_inventory", {})
        compressors = inventory.get("compressors", [])
        lines = [
            "## 2. System Inventory\n",
            f"**System Pressure:** {self._fmt(inventory.get('system_pressure_bar', 0))} bar(g)  ",
            f"**Total Receivers:** {self._fmt(inventory.get('total_receiver_volume_l', 0))} liters  ",
            f"**Distribution Length:** {self._fmt(inventory.get('distribution_length_m', 0))} m  ",
            f"**Dryers:** {inventory.get('dryer_count', 0)} units",
        ]
        if compressors:
            lines.extend([
                "\n### Compressor Inventory\n",
                "| # | Make/Model | Type | Power (kW) | Flow (CFM) | Age (yr) | "
                "Load (%) | Condition |",
                "|---|-----------|------|-----------|-----------|---------|---------|-----------|",
            ])
            for i, c in enumerate(compressors, 1):
                lines.append(
                    f"| {i} | {c.get('make_model', '-')} "
                    f"| {c.get('type', '-')} "
                    f"| {self._fmt(c.get('rated_power_kw', 0))} "
                    f"| {self._fmt(c.get('rated_flow_cfm', 0))} "
                    f"| {c.get('age_years', '-')} "
                    f"| {self._fmt(c.get('avg_load_pct', 0))}% "
                    f"| {c.get('condition', '-')} |"
                )
        return "\n".join(lines)

    def _md_specific_power(self, data: Dict[str, Any]) -> str:
        """Render specific power analysis section."""
        sp = data.get("specific_power", {})
        by_compressor = sp.get("by_compressor", [])
        lines = [
            "## 3. Specific Power Analysis\n",
            f"**System Specific Power:** {self._fmt(sp.get('system_kw_per_100cfm', 0))} kW/100CFM  ",
            f"**Benchmark (Best Practice):** {self.BENCHMARK_SPECIFIC_POWER['best_practice_kw_per_100cfm']} kW/100CFM  ",
            f"**Rating:** {sp.get('rating', '-')}  ",
            f"**Improvement Potential:** {self._fmt(sp.get('improvement_potential_pct', 0))}%",
        ]
        if by_compressor:
            lines.extend([
                "\n### Specific Power by Compressor\n",
                "| Compressor | Measured (kW/100CFM) | Rated (kW/100CFM) | Efficiency (%) |",
                "|-----------|---------------------|-------------------|---------------|",
            ])
            for c in by_compressor:
                lines.append(
                    f"| {c.get('compressor_id', '-')} "
                    f"| {self._fmt(c.get('measured_sp', 0))} "
                    f"| {self._fmt(c.get('rated_sp', 0))} "
                    f"| {self._fmt(c.get('efficiency_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_leak_survey(self, data: Dict[str, Any]) -> str:
        """Render leak survey results section."""
        leaks = data.get("leak_survey", {})
        leak_list = leaks.get("leaks_found", [])
        lines = [
            "## 4. Leak Survey Results\n",
            f"**Survey Method:** {leaks.get('method', 'Ultrasonic Detection')}  ",
            f"**Total Leaks Found:** {leaks.get('total_leaks', 0)}  ",
            f"**Estimated Leak Rate:** {self._fmt(leaks.get('leak_rate_cfm', 0))} CFM "
            f"({self._fmt(leaks.get('leak_rate_pct', 0))}% of capacity)  ",
            f"**Annual Leak Cost:** EUR {self._fmt(leaks.get('annual_leak_cost_eur', 0))}  ",
            f"**Leak Energy Waste:** {self._fmt(leaks.get('leak_energy_mwh', 0))} MWh/yr",
        ]
        if leak_list:
            lines.extend([
                "\n### Leak Inventory\n",
                "| # | Location | Component | Size | Est. Flow (CFM) | Priority |",
                "|---|----------|-----------|------|-----------------|----------|",
            ])
            for i, lk in enumerate(leak_list[:30], 1):
                lines.append(
                    f"| {i} | {lk.get('location', '-')} "
                    f"| {lk.get('component', '-')} "
                    f"| {lk.get('size', '-')} "
                    f"| {self._fmt(lk.get('est_flow_cfm', 0), 1)} "
                    f"| {lk.get('priority', '-')} |"
                )
        by_severity = leaks.get("by_severity", {})
        if by_severity:
            lines.extend([
                "\n### Leaks by Severity\n",
                "| Severity | Count | Flow (CFM) | Cost (EUR/yr) |",
                "|----------|-------|-----------|--------------|",
            ])
            for sev, info in by_severity.items():
                lines.append(
                    f"| {sev} | {info.get('count', 0)} "
                    f"| {self._fmt(info.get('flow_cfm', 0), 1)} "
                    f"| {self._fmt(info.get('cost_eur', 0))} |"
                )
        return "\n".join(lines)

    def _md_pressure_profile(self, data: Dict[str, Any]) -> str:
        """Render pressure profile assessment section."""
        pressure = data.get("pressure_profile", {})
        zones = pressure.get("zones", [])
        lines = [
            "## 5. Pressure Profile Assessment\n",
            f"**Header Pressure:** {self._fmt(pressure.get('header_pressure_bar', 0))} bar(g)  ",
            f"**Minimum Point-of-Use Pressure:** "
            f"{self._fmt(pressure.get('min_pou_pressure_bar', 0))} bar(g)  ",
            f"**Pressure Drop (Supply to POU):** "
            f"{self._fmt(pressure.get('total_pressure_drop_bar', 0))} bar  ",
            f"**Artificial Demand (est.):** "
            f"{self._fmt(pressure.get('artificial_demand_pct', 0))}%  ",
            f"**Pressure Reduction Opportunity:** "
            f"{self._fmt(pressure.get('reduction_opportunity_bar', 0))} bar "
            f"({self._fmt(pressure.get('reduction_savings_pct', 0))}% energy saving)",
        ]
        if zones:
            lines.extend([
                "\n### Pressure Zones\n",
                "| Zone | Required (bar) | Measured (bar) | Drop (bar) | Status |",
                "|------|---------------|----------------|-----------|--------|",
            ])
            for z in zones:
                lines.append(
                    f"| {z.get('zone_name', '-')} "
                    f"| {self._fmt(z.get('required_bar', 0), 1)} "
                    f"| {self._fmt(z.get('measured_bar', 0), 1)} "
                    f"| {self._fmt(z.get('drop_bar', 0), 1)} "
                    f"| {z.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_vsd_analysis(self, data: Dict[str, Any]) -> str:
        """Render VSD opportunity analysis section."""
        vsd = data.get("vsd_analysis", {})
        candidates = vsd.get("candidates", [])
        lines = [
            "## 6. VSD Opportunity Analysis\n",
            f"**Current VSD Units:** {vsd.get('current_vsd_count', 0)}  ",
            f"**VSD Candidates:** {vsd.get('candidate_count', 0)}  ",
            f"**Total VSD Savings Potential:** {self._fmt(vsd.get('total_savings_mwh', 0))} MWh/yr  ",
            f"**Total VSD Investment:** EUR {self._fmt(vsd.get('total_investment_eur', 0))}",
        ]
        if candidates:
            lines.extend([
                "\n### VSD Candidates\n",
                "| Compressor | Load Variation | Savings (MWh/yr) | "
                "Investment (EUR) | Payback (yr) |",
                "|-----------|---------------|-----------------|-----------------|-------------|",
            ])
            for c in candidates:
                lines.append(
                    f"| {c.get('compressor_id', '-')} "
                    f"| {self._fmt(c.get('load_variation_pct', 0))}% "
                    f"| {self._fmt(c.get('savings_mwh', 0))} "
                    f"| {self._fmt(c.get('investment_eur', 0))} "
                    f"| {self._fmt(c.get('payback_years', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_air_quality(self, data: Dict[str, Any]) -> str:
        """Render air quality and treatment section."""
        aq = data.get("air_quality", {})
        lines = [
            "## 7. Air Quality & Treatment\n",
            f"**ISO 8573-1 Class:** {aq.get('iso_class', '-')}  ",
            f"**Particle Class:** {aq.get('particle_class', '-')}  ",
            f"**Water Class:** {aq.get('water_class', '-')}  ",
            f"**Oil Class:** {aq.get('oil_class', '-')}  ",
            f"**Dew Point:** {aq.get('dew_point_c', '-')} C  ",
            f"**Treatment Energy Cost:** EUR {self._fmt(aq.get('treatment_cost_eur', 0))}/yr",
        ]
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render optimization recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 8. Optimization Recommendations\n\n_No recommendations._"
        lines = [
            "## 8. Optimization Recommendations\n",
            "| # | Recommendation | Savings (MWh/yr) | Cost Savings (EUR/yr) | "
            "Investment (EUR) | Payback (yr) |",
            "|---|--------------|-----------------|---------------------|"
            "-----------------|-------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('title', '-')} "
                f"| {self._fmt(r.get('energy_savings_mwh', 0))} "
                f"| {self._fmt(r.get('cost_savings_eur', 0))} "
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
            f'<h1>Compressed Air System Audit Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | Standard: ISO 11011:2013</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Specific Power</span>'
            f'<span class="value">{self._fmt(s.get("system_specific_power", 0))} kW/100CFM</span></div>\n'
            f'  <div class="card"><span class="label">Leak Rate</span>'
            f'<span class="value">{self._fmt(s.get("leak_rate_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Annual Cost</span>'
            f'<span class="value">EUR {self._fmt(s.get("annual_cost_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Savings Potential</span>'
            f'<span class="value">{self._fmt(s.get("total_savings_mwh", 0))} MWh</span></div>\n'
            '</div>'
        )

    def _html_system_inventory(self, data: Dict[str, Any]) -> str:
        """Render HTML system inventory table."""
        compressors = data.get("system_inventory", {}).get("compressors", [])
        rows = ""
        for c in compressors:
            rows += (
                f'<tr><td>{c.get("make_model", "-")}</td>'
                f'<td>{c.get("type", "-")}</td>'
                f'<td>{self._fmt(c.get("rated_power_kw", 0))}</td>'
                f'<td>{self._fmt(c.get("avg_load_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>System Inventory</h2>\n<table>\n'
            '<tr><th>Make/Model</th><th>Type</th><th>Power (kW)</th>'
            f'<th>Load</th></tr>\n{rows}</table>'
        )

    def _html_specific_power(self, data: Dict[str, Any]) -> str:
        """Render HTML specific power gauge."""
        sp = data.get("specific_power", {})
        val = sp.get("system_kw_per_100cfm", 0)
        rating = sp.get("rating", "-")
        return (
            '<h2>Specific Power Analysis</h2>\n'
            f'<p>System: {self._fmt(val)} kW/100CFM | Rating: <strong>{rating}</strong></p>'
        )

    def _html_leak_survey(self, data: Dict[str, Any]) -> str:
        """Render HTML leak survey summary."""
        leaks = data.get("leak_survey", {})
        return (
            '<h2>Leak Survey Results</h2>\n'
            f'<p>Leaks Found: {leaks.get("total_leaks", 0)} | '
            f'Rate: {self._fmt(leaks.get("leak_rate_pct", 0))}% | '
            f'Annual Cost: EUR {self._fmt(leaks.get("annual_leak_cost_eur", 0))}</p>'
        )

    def _html_pressure_profile(self, data: Dict[str, Any]) -> str:
        """Render HTML pressure profile."""
        pp = data.get("pressure_profile", {})
        return (
            '<h2>Pressure Profile</h2>\n'
            f'<p>Header: {self._fmt(pp.get("header_pressure_bar", 0))} bar | '
            f'Min POU: {self._fmt(pp.get("min_pou_pressure_bar", 0))} bar | '
            f'Drop: {self._fmt(pp.get("total_pressure_drop_bar", 0))} bar</p>'
        )

    def _html_vsd_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML VSD analysis."""
        vsd = data.get("vsd_analysis", {})
        return (
            '<h2>VSD Opportunity Analysis</h2>\n'
            f'<p>Candidates: {vsd.get("candidate_count", 0)} | '
            f'Savings: {self._fmt(vsd.get("total_savings_mwh", 0))} MWh/yr</p>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        items = "".join(
            f'<li>{r.get("title", "-")} - '
            f'Savings: {self._fmt(r.get("energy_savings_mwh", 0))} MWh/yr, '
            f'Payback: {self._fmt(r.get("payback_years", 0), 1)} yr</li>\n'
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
        compressors = data.get("system_inventory", {}).get("compressors", [])
        leaks = data.get("leak_survey", {}).get("by_severity", {})
        sp = data.get("specific_power", {}).get("by_compressor", [])
        return {
            "compressor_load": {
                "type": "bar",
                "labels": [c.get("make_model", "") for c in compressors],
                "values": [c.get("avg_load_pct", 0) for c in compressors],
            },
            "specific_power_comparison": {
                "type": "bar",
                "labels": [c.get("compressor_id", "") for c in sp],
                "series": {
                    "measured": [c.get("measured_sp", 0) for c in sp],
                    "rated": [c.get("rated_sp", 0) for c in sp],
                },
                "benchmark": self.BENCHMARK_SPECIFIC_POWER["best_practice_kw_per_100cfm"],
            },
            "leak_severity_pie": {
                "type": "pie",
                "labels": list(leaks.keys()),
                "values": [v.get("count", 0) for v in leaks.values()],
            },
            "pressure_profile": {
                "type": "line",
                "labels": [
                    z.get("zone_name", "")
                    for z in data.get("pressure_profile", {}).get("zones", [])
                ],
                "series": {
                    "measured": [
                        z.get("measured_bar", 0)
                        for z in data.get("pressure_profile", {}).get("zones", [])
                    ],
                    "required": [
                        z.get("required_bar", 0)
                        for z in data.get("pressure_profile", {}).get("zones", [])
                    ],
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
