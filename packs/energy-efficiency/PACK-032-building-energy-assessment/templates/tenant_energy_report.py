# -*- coding: utf-8 -*-
"""
TenantEnergyReportTemplate - Tenant-facing energy report for PACK-032.

Generates tenant-facing energy reports with energy summaries, cost
breakdowns, comparison to building average, comparison to benchmarks,
monthly trend data, tips for energy reduction, green lease compliance
status, and carbon footprint attribution.

Sections:
    1. Energy Summary
    2. Cost Summary
    3. Comparison to Building Average
    4. Comparison to Benchmark
    5. Monthly Trend
    6. Tips for Reduction
    7. Green Lease Compliance
    8. Carbon Footprint
    9. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TenantEnergyReportTemplate:
    """
    Tenant-facing energy report template.

    Renders tenant energy reports with consumption summaries, cost
    breakdowns, building comparisons, reduction tips, green lease
    compliance, and carbon footprint data across markdown, HTML,
    and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    TENANT_SECTIONS: List[str] = [
        "Energy Summary",
        "Cost Summary",
        "Building Comparison",
        "Benchmark Comparison",
        "Monthly Trend",
        "Reduction Tips",
        "Green Lease Compliance",
        "Carbon Footprint",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TenantEnergyReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render tenant energy report as Markdown.

        Args:
            data: Tenant energy data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_energy_summary(data),
            self._md_cost_summary(data),
            self._md_building_comparison(data),
            self._md_benchmark_comparison(data),
            self._md_monthly_trend(data),
            self._md_reduction_tips(data),
            self._md_green_lease(data),
            self._md_carbon_footprint(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render tenant energy report as self-contained HTML.

        Args:
            data: Tenant energy data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_energy_summary(data),
            self._html_cost_summary(data),
            self._html_building_comparison(data),
            self._html_benchmark_comparison(data),
            self._html_monthly_trend(data),
            self._html_reduction_tips(data),
            self._html_green_lease(data),
            self._html_carbon_footprint(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Tenant Energy Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render tenant energy report as structured JSON.

        Args:
            data: Tenant energy data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "tenant_energy_report",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "tenant": self._json_tenant(data),
            "energy_summary": data.get("energy_summary", {}),
            "cost_summary": data.get("cost_summary", {}),
            "building_comparison": data.get("building_comparison", {}),
            "benchmark_comparison": data.get("benchmark_comparison", {}),
            "monthly_trend": data.get("monthly_trend", []),
            "reduction_tips": data.get("reduction_tips", []),
            "green_lease": data.get("green_lease", {}),
            "carbon_footprint": data.get("carbon_footprint", {}),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with tenant details."""
        tenant = data.get("tenant_name", "Tenant")
        building = data.get("building_name", "Building")
        period = data.get("reporting_period", "-")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Tenant Energy Report\n\n"
            f"**Tenant:** {tenant}  \n"
            f"**Building:** {building}  \n"
            f"**Floor/Suite:** {data.get('floor_suite', '-')}  \n"
            f"**Leased Area:** {self._fmt(data.get('leased_area_sqm', 0), 0)} m2  \n"
            f"**Reporting Period:** {period}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 TenantEnergyReportTemplate v32.0.0\n\n---"
        )

    def _md_energy_summary(self, data: Dict[str, Any]) -> str:
        """Render energy summary section."""
        s = data.get("energy_summary", {})
        return (
            "## 1. Energy Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Electricity | {self._fmt(s.get('electricity_kwh', 0), 0)} kWh |\n"
            f"| Total Gas | {self._fmt(s.get('gas_kwh', 0), 0)} kWh |\n"
            f"| Total Energy | {self._fmt(s.get('total_kwh', 0), 0)} kWh |\n"
            f"| EUI | {self._fmt(s.get('eui_kwh_m2', 0))} kWh/m2 |\n"
            f"| vs Previous Period | {s.get('vs_previous', '-')} |\n"
            f"| vs Same Period Last Year | {s.get('vs_last_year', '-')} |"
        )

    def _md_cost_summary(self, data: Dict[str, Any]) -> str:
        """Render cost summary section."""
        c = data.get("cost_summary", {})
        breakdown = c.get("breakdown", [])
        lines = [
            "## 2. Cost Summary\n",
            f"**Total Energy Cost:** {c.get('total_cost', '-')}  ",
            f"**Cost per m2:** {c.get('cost_per_m2', '-')}  ",
            f"**vs Previous Period:** {c.get('vs_previous', '-')}  ",
            f"**vs Budget:** {c.get('vs_budget', '-')}",
        ]
        if breakdown:
            lines.extend([
                "\n### Cost Breakdown\n",
                "| Item | Amount | Share (%) |",
                "|------|--------|-----------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('item', '-')} "
                    f"| {b.get('amount', '-')} "
                    f"| {self._fmt(b.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_building_comparison(self, data: Dict[str, Any]) -> str:
        """Render comparison to building average section."""
        comp = data.get("building_comparison", {})
        return (
            "## 3. Comparison to Building Average\n\n"
            f"| Metric | Your Space | Building Avg | Difference |\n"
            f"|--------|-----------|-------------|------------|\n"
            f"| EUI (kWh/m2) | {self._fmt(comp.get('tenant_eui', 0))} "
            f"| {self._fmt(comp.get('building_avg_eui', 0))} "
            f"| {comp.get('eui_difference', '-')} |\n"
            f"| Cost/m2 | {comp.get('tenant_cost_m2', '-')} "
            f"| {comp.get('building_avg_cost_m2', '-')} "
            f"| {comp.get('cost_difference', '-')} |\n"
            f"| CO2/m2 | {self._fmt(comp.get('tenant_co2_m2', 0))} "
            f"| {self._fmt(comp.get('building_avg_co2_m2', 0))} "
            f"| {comp.get('co2_difference', '-')} |\n"
            f"| Ranking | {comp.get('ranking', '-')} of "
            f"{comp.get('total_tenants', '-')} tenants | | |"
        )

    def _md_benchmark_comparison(self, data: Dict[str, Any]) -> str:
        """Render comparison to external benchmark section."""
        bm = data.get("benchmark_comparison", {})
        return (
            "## 4. Comparison to Benchmark\n\n"
            f"**Benchmark Source:** {bm.get('source', '-')}  \n"
            f"**Sector:** {bm.get('sector', '-')}\n\n"
            f"| Metric | Your Space | Benchmark | Status |\n"
            f"|--------|-----------|-----------|--------|\n"
            f"| EUI (kWh/m2) | {self._fmt(bm.get('tenant_eui', 0))} "
            f"| {self._fmt(bm.get('benchmark_eui', 0))} "
            f"| {bm.get('eui_status', '-')} |\n"
            f"| CO2 (kgCO2/m2) | {self._fmt(bm.get('tenant_co2', 0))} "
            f"| {self._fmt(bm.get('benchmark_co2', 0))} "
            f"| {bm.get('co2_status', '-')} |"
        )

    def _md_monthly_trend(self, data: Dict[str, Any]) -> str:
        """Render monthly trend section."""
        months = data.get("monthly_trend", [])
        if not months:
            return "## 5. Monthly Trend\n\n_No monthly data available._"
        lines = [
            "## 5. Monthly Trend\n",
            "| Month | kWh | Cost | kWh/m2 | vs Last Year |",
            "|-------|-----|------|--------|-------------|",
        ]
        for m in months:
            lines.append(
                f"| {m.get('month', '-')} "
                f"| {self._fmt(m.get('kwh', 0), 0)} "
                f"| {m.get('cost', '-')} "
                f"| {self._fmt(m.get('kwh_m2', 0))} "
                f"| {m.get('vs_last_year', '-')} |"
            )
        return "\n".join(lines)

    def _md_reduction_tips(self, data: Dict[str, Any]) -> str:
        """Render tips for energy reduction section."""
        tips = data.get("reduction_tips", [])
        if not tips:
            return "## 6. Tips for Reducing Your Energy Use\n\n_No tips available._"
        lines = ["## 6. Tips for Reducing Your Energy Use\n"]
        for i, tip in enumerate(tips, 1):
            lines.extend([
                f"### Tip {i}: {tip.get('title', '-')}\n",
                f"{tip.get('description', '')}  ",
                f"**Potential Saving:** {tip.get('potential_saving', '-')}  ",
                f"**Difficulty:** {tip.get('difficulty', '-')}  ",
                f"**Cost:** {tip.get('cost', '-')}\n",
            ])
        return "\n".join(lines)

    def _md_green_lease(self, data: Dict[str, Any]) -> str:
        """Render green lease compliance section."""
        gl = data.get("green_lease", {})
        clauses = gl.get("clauses", [])
        lines = [
            "## 7. Green Lease Compliance\n",
            f"**Lease Type:** {gl.get('lease_type', '-')}  ",
            f"**Green Lease Active:** {'Yes' if gl.get('active', False) else 'No'}  ",
            f"**Overall Compliance:** {gl.get('overall_compliance', '-')}",
        ]
        if clauses:
            lines.extend([
                "\n### Clause Compliance\n",
                "| Clause | Requirement | Status | Notes |",
                "|--------|------------|--------|-------|",
            ])
            for c in clauses:
                lines.append(
                    f"| {c.get('clause', '-')} "
                    f"| {c.get('requirement', '-')} "
                    f"| {c.get('status', '-')} "
                    f"| {c.get('notes', '-')} |"
                )
        return "\n".join(lines)

    def _md_carbon_footprint(self, data: Dict[str, Any]) -> str:
        """Render carbon footprint section."""
        cf = data.get("carbon_footprint", {})
        breakdown = cf.get("breakdown", [])
        lines = [
            "## 8. Carbon Footprint\n",
            f"**Total CO2:** {self._fmt(cf.get('total_kg_co2', 0), 0)} kgCO2  ",
            f"**CO2 per m2:** {self._fmt(cf.get('kg_co2_m2', 0))} kgCO2/m2  ",
            f"**CO2 per FTE:** {self._fmt(cf.get('kg_co2_fte', 0))} kgCO2/FTE  ",
            f"**vs Previous Period:** {cf.get('vs_previous', '-')}",
        ]
        if breakdown:
            lines.extend([
                "\n### Emissions Breakdown\n",
                "| Source | kgCO2 | Share (%) |",
                "|--------|-------|-----------|",
            ])
            for b in breakdown:
                lines.append(
                    f"| {b.get('source', '-')} "
                    f"| {self._fmt(b.get('kg_co2', 0), 0)} "
                    f"| {self._fmt(b.get('share_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 TenantEnergyReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        tenant = data.get("tenant_name", "Tenant")
        building = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Tenant Energy Report</h1>\n'
            f'<p class="subtitle">Tenant: {tenant} | Building: {building} | '
            f'Generated: {ts}</p>'
        )

    def _html_energy_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML energy summary with KPI cards."""
        s = data.get("energy_summary", {})
        return (
            '<h2>Energy Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">Total Energy</span>'
            f'<span class="value">{self._fmt(s.get("total_kwh", 0), 0)}</span>'
            f'<span class="label">kWh</span></div>\n'
            f'<div class="card"><span class="label">EUI</span>'
            f'<span class="value">{self._fmt(s.get("eui_kwh_m2", 0))}</span>'
            f'<span class="label">kWh/m2</span></div>\n'
            f'<div class="card"><span class="label">vs Last Year</span>'
            f'<span class="value">{s.get("vs_last_year", "-")}</span></div>\n'
            '</div>'
        )

    def _html_cost_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML cost summary."""
        c = data.get("cost_summary", {})
        breakdown = c.get("breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("item", "-")}</td>'
                f'<td>{b.get("amount", "-")}</td>'
                f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Cost Summary</h2>\n'
            f'<p>Total: <strong>{c.get("total_cost", "-")}</strong> | '
            f'Cost/m2: {c.get("cost_per_m2", "-")}</p>\n'
            '<table>\n<tr><th>Item</th><th>Amount</th><th>Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_building_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML building comparison."""
        comp = data.get("building_comparison", {})
        return (
            '<h2>Comparison to Building Average</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Your Space</th><th>Building Avg</th></tr>\n'
            f'<tr><td>EUI (kWh/m2)</td><td>{self._fmt(comp.get("tenant_eui", 0))}</td>'
            f'<td>{self._fmt(comp.get("building_avg_eui", 0))}</td></tr>\n'
            f'<tr><td>Cost/m2</td><td>{comp.get("tenant_cost_m2", "-")}</td>'
            f'<td>{comp.get("building_avg_cost_m2", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_benchmark_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark comparison."""
        bm = data.get("benchmark_comparison", {})
        return (
            '<h2>Comparison to Benchmark</h2>\n'
            f'<p>Source: {bm.get("source", "-")} | Sector: {bm.get("sector", "-")}</p>\n'
            '<table>\n<tr><th>Metric</th><th>Your Space</th><th>Benchmark</th>'
            '<th>Status</th></tr>\n'
            f'<tr><td>EUI</td><td>{self._fmt(bm.get("tenant_eui", 0))}</td>'
            f'<td>{self._fmt(bm.get("benchmark_eui", 0))}</td>'
            f'<td>{bm.get("eui_status", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_monthly_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML monthly trend."""
        months = data.get("monthly_trend", [])
        rows = ""
        for m in months:
            rows += (
                f'<tr><td>{m.get("month", "-")}</td>'
                f'<td>{self._fmt(m.get("kwh", 0), 0)}</td>'
                f'<td>{m.get("cost", "-")}</td>'
                f'<td>{m.get("vs_last_year", "-")}</td></tr>\n'
            )
        return (
            '<h2>Monthly Trend</h2>\n'
            '<table>\n<tr><th>Month</th><th>kWh</th><th>Cost</th>'
            f'<th>vs Last Year</th></tr>\n{rows}</table>'
        )

    def _html_reduction_tips(self, data: Dict[str, Any]) -> str:
        """Render HTML reduction tips."""
        tips = data.get("reduction_tips", [])
        items = ""
        for tip in tips:
            items += (
                f'<div class="tip"><h3>{tip.get("title", "-")}</h3>'
                f'<p>{tip.get("description", "")}</p>'
                f'<p>Potential: {tip.get("potential_saving", "-")} | '
                f'Difficulty: {tip.get("difficulty", "-")}</p></div>\n'
            )
        return f'<h2>Tips for Reducing Energy Use</h2>\n{items}'

    def _html_green_lease(self, data: Dict[str, Any]) -> str:
        """Render HTML green lease compliance."""
        gl = data.get("green_lease", {})
        clauses = gl.get("clauses", [])
        rows = ""
        for c in clauses:
            status = c.get("status", "-")
            style = 'color:#198754' if status == "Compliant" else 'color:#dc3545'
            rows += (
                f'<tr><td>{c.get("clause", "-")}</td>'
                f'<td>{c.get("requirement", "-")}</td>'
                f'<td style="{style};font-weight:bold">{status}</td></tr>\n'
            )
        active = "Yes" if gl.get("active", False) else "No"
        return (
            '<h2>Green Lease Compliance</h2>\n'
            f'<p>Active: {active} | Overall: {gl.get("overall_compliance", "-")}</p>\n'
            '<table>\n<tr><th>Clause</th><th>Requirement</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_carbon_footprint(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon footprint."""
        cf = data.get("carbon_footprint", {})
        breakdown = cf.get("breakdown", [])
        rows = ""
        for b in breakdown:
            rows += (
                f'<tr><td>{b.get("source", "-")}</td>'
                f'<td>{self._fmt(b.get("kg_co2", 0), 0)}</td>'
                f'<td>{self._fmt(b.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Carbon Footprint</h2>\n'
            f'<p>Total: {self._fmt(cf.get("total_kg_co2", 0), 0)} kgCO2 | '
            f'Per m2: {self._fmt(cf.get("kg_co2_m2", 0))} kgCO2/m2</p>\n'
            '<table>\n<tr><th>Source</th><th>kgCO2</th><th>Share</th></tr>\n'
            f'{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_tenant(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON tenant metadata."""
        return {
            "name": data.get("tenant_name", ""),
            "building": data.get("building_name", ""),
            "floor_suite": data.get("floor_suite", ""),
            "leased_area_sqm": data.get("leased_area_sqm", 0),
            "reporting_period": data.get("reporting_period", ""),
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
            ".tip{background:#e8f5e9;border-left:4px solid #198754;padding:12px;margin:10px 0;}"
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
