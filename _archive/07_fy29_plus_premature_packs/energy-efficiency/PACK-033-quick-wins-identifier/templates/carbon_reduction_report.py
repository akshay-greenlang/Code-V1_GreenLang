# -*- coding: utf-8 -*-
"""
CarbonReductionReportTemplate - Emissions reduction report for PACK-033.

Generates carbon reduction reports for quick-win energy efficiency measures,
covering scope breakdowns, measure-level reductions, cumulative projections,
SBTi alignment checks, and location vs market-based comparisons.

Sections:
    1. Carbon Summary
    2. Scope Breakdown (1/2/3)
    3. Measure-Level Reductions
    4. Cumulative Projections
    5. SBTi Alignment
    6. Location vs Market-Based Comparison

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CarbonReductionReportTemplate:
    """
    Carbon emissions reduction report template.

    Renders emissions reduction analysis for quick-win measures with
    scope-level breakdowns, cumulative projections, SBTi alignment,
    and dual-reporting comparisons across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CarbonReductionReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render carbon reduction report as Markdown.

        Args:
            data: Carbon analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_carbon_summary(data),
            self._md_scope_breakdown(data),
            self._md_measure_reductions(data),
            self._md_cumulative_projections(data),
            self._md_sbti_alignment(data),
            self._md_location_vs_market(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render carbon reduction report as self-contained HTML.

        Args:
            data: Carbon analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_carbon_summary(data),
            self._html_scope_breakdown(data),
            self._html_measure_reductions(data),
            self._html_cumulative_projections(data),
            self._html_sbti_alignment(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Carbon Reduction Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render carbon reduction report as structured JSON.

        Args:
            data: Carbon analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "carbon_reduction_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "carbon_summary": self._json_carbon_summary(data),
            "scope_breakdown": data.get("scope_breakdown", {}),
            "measure_reductions": data.get("measure_reductions", []),
            "cumulative_projections": data.get("cumulative_projections", []),
            "sbti_alignment": data.get("sbti_alignment", {}),
            "location_vs_market": data.get("location_vs_market", {}),
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
            f"# Carbon Reduction Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Baseline Year:** {data.get('baseline_year', '')}  \n"
            f"**Reporting Standard:** {data.get('reporting_standard', 'GHG Protocol')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 CarbonReductionReportTemplate v33.0.0\n\n---"
        )

    def _md_carbon_summary(self, data: Dict[str, Any]) -> str:
        """Render carbon summary section."""
        summary = data.get("carbon_summary", {})
        baseline = summary.get("baseline_emissions_tco2e", 0)
        reduction = summary.get("total_reduction_tco2e", 0)
        pct = self._pct(reduction, baseline)
        return (
            "## 1. Carbon Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Baseline Emissions | {self._fmt(baseline)} tCO2e/yr |\n"
            f"| Total Reduction Potential | {self._fmt(reduction)} tCO2e/yr ({pct}) |\n"
            f"| Remaining Emissions | {self._fmt(summary.get('remaining_emissions_tco2e', 0))} tCO2e/yr |\n"
            f"| Scope 1 Reduction | {self._fmt(summary.get('scope1_reduction_tco2e', 0))} tCO2e/yr |\n"
            f"| Scope 2 Reduction | {self._fmt(summary.get('scope2_reduction_tco2e', 0))} tCO2e/yr |\n"
            f"| Scope 3 Reduction | {self._fmt(summary.get('scope3_reduction_tco2e', 0))} tCO2e/yr |\n"
            f"| Marginal Abatement Cost | {self._format_currency(summary.get('mac_eur_per_tco2e', 0))}/tCO2e |"
        )

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        """Render scope breakdown section."""
        breakdown = data.get("scope_breakdown", {})
        lines = ["## 2. Scope Breakdown\n"]
        for scope_key in ["scope_1", "scope_2", "scope_3"]:
            scope = breakdown.get(scope_key, {})
            label = scope_key.replace("_", " ").title()
            lines.extend([
                f"### {label}\n",
                f"- **Baseline:** {self._fmt(scope.get('baseline_tco2e', 0))} tCO2e",
                f"- **Reduction:** {self._fmt(scope.get('reduction_tco2e', 0))} tCO2e",
                f"- **Reduction Share:** {self._fmt(scope.get('reduction_share_pct', 0))}%",
                f"- **Sources:** {', '.join(scope.get('sources', ['-']))}",
                "",
            ])
        return "\n".join(lines)

    def _md_measure_reductions(self, data: Dict[str, Any]) -> str:
        """Render measure-level reductions table."""
        measures = data.get("measure_reductions", [])
        if not measures:
            return "## 3. Measure-Level Reductions\n\n_No measure data available._"
        lines = [
            "## 3. Measure-Level Reductions\n",
            "| # | Measure | Scope | Reduction (tCO2e/yr) | Share (%) | MAC (EUR/tCO2e) |",
            "|---|---------|-------|---------------------|-----------|-----------------|",
        ]
        for i, m in enumerate(measures, 1):
            lines.append(
                f"| {i} | {m.get('measure', '-')} "
                f"| {m.get('scope', '-')} "
                f"| {self._fmt(m.get('reduction_tco2e', 0))} "
                f"| {self._fmt(m.get('share_pct', 0))}% "
                f"| {self._fmt(m.get('mac_eur_per_tco2e', 0))} |"
            )
        total = sum(m.get("reduction_tco2e", 0) for m in measures)
        lines.append(
            f"| | **TOTAL** | | **{self._fmt(total)}** | **100%** | |"
        )
        return "\n".join(lines)

    def _md_cumulative_projections(self, data: Dict[str, Any]) -> str:
        """Render cumulative projections section."""
        projections = data.get("cumulative_projections", [])
        if not projections:
            return "## 4. Cumulative Projections\n\n_No projection data available._"
        lines = [
            "## 4. Cumulative Projections\n",
            "| Year | Annual Reduction (tCO2e) | Cumulative (tCO2e) | Remaining (tCO2e) |",
            "|------|------------------------|--------------------|-------------------|",
        ]
        for p in projections:
            lines.append(
                f"| {p.get('year', '-')} "
                f"| {self._fmt(p.get('annual_reduction', 0))} "
                f"| {self._fmt(p.get('cumulative_reduction', 0))} "
                f"| {self._fmt(p.get('remaining_emissions', 0))} |"
            )
        return "\n".join(lines)

    def _md_sbti_alignment(self, data: Dict[str, Any]) -> str:
        """Render SBTi alignment section."""
        sbti = data.get("sbti_alignment", {})
        lines = [
            "## 5. SBTi Alignment\n",
            f"- **Target Pathway:** {sbti.get('pathway', '1.5C')}",
            f"- **Required Annual Reduction:** {self._fmt(sbti.get('required_annual_pct', 0))}%",
            f"- **Achieved Annual Reduction:** {self._fmt(sbti.get('achieved_annual_pct', 0))}%",
            f"- **Alignment Status:** {sbti.get('alignment_status', 'Not Assessed')}",
            f"- **Gap to Target:** {self._fmt(sbti.get('gap_tco2e', 0))} tCO2e",
            f"- **On Track:** {'Yes' if sbti.get('on_track', False) else 'No'}",
        ]
        return "\n".join(lines)

    def _md_location_vs_market(self, data: Dict[str, Any]) -> str:
        """Render location vs market-based comparison section."""
        comparison = data.get("location_vs_market", {})
        loc = comparison.get("location_based", {})
        mkt = comparison.get("market_based", {})
        return (
            "## 6. Location vs Market-Based Comparison\n\n"
            "| Metric | Location-Based | Market-Based |\n"
            "|--------|---------------|-------------|\n"
            f"| Scope 2 Baseline | {self._fmt(loc.get('baseline_tco2e', 0))} tCO2e "
            f"| {self._fmt(mkt.get('baseline_tco2e', 0))} tCO2e |\n"
            f"| Scope 2 Reduction | {self._fmt(loc.get('reduction_tco2e', 0))} tCO2e "
            f"| {self._fmt(mkt.get('reduction_tco2e', 0))} tCO2e |\n"
            f"| Emission Factor | {self._fmt(loc.get('emission_factor', 0), 4)} tCO2e/MWh "
            f"| {self._fmt(mkt.get('emission_factor', 0), 4)} tCO2e/MWh |\n"
            f"| Remaining | {self._fmt(loc.get('remaining_tco2e', 0))} tCO2e "
            f"| {self._fmt(mkt.get('remaining_tco2e', 0))} tCO2e |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-033 Quick Wins Identifier Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Carbon Reduction Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Standard: {data.get("reporting_standard", "GHG Protocol")}</p>'
        )

    def _html_carbon_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon summary cards."""
        s = data.get("carbon_summary", {})
        return (
            '<h2>Carbon Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Baseline</span>'
            f'<span class="value">{self._fmt(s.get("baseline_emissions_tco2e", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Reduction</span>'
            f'<span class="value">{self._fmt(s.get("total_reduction_tco2e", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Remaining</span>'
            f'<span class="value">{self._fmt(s.get("remaining_emissions_tco2e", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">MAC</span>'
            f'<span class="value">{self._format_currency(s.get("mac_eur_per_tco2e", 0))}/t</span></div>\n'
            '</div>'
        )

    def _html_scope_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML scope breakdown."""
        breakdown = data.get("scope_breakdown", {})
        items = ""
        for scope_key in ["scope_1", "scope_2", "scope_3"]:
            scope = breakdown.get(scope_key, {})
            label = scope_key.replace("_", " ").title()
            items += (
                f'<div class="scope-card"><h3>{label}</h3>'
                f'<p>Baseline: {self._fmt(scope.get("baseline_tco2e", 0))} tCO2e | '
                f'Reduction: {self._fmt(scope.get("reduction_tco2e", 0))} tCO2e</p></div>\n'
            )
        return f'<h2>Scope Breakdown</h2>\n<div class="scope-grid">\n{items}</div>'

    def _html_measure_reductions(self, data: Dict[str, Any]) -> str:
        """Render HTML measure-level reductions table."""
        measures = data.get("measure_reductions", [])
        rows = ""
        for m in measures:
            rows += (
                f'<tr><td>{m.get("measure", "-")}</td>'
                f'<td>{m.get("scope", "-")}</td>'
                f'<td>{self._fmt(m.get("reduction_tco2e", 0))}</td>'
                f'<td>{self._fmt(m.get("mac_eur_per_tco2e", 0))}</td></tr>\n'
            )
        return (
            '<h2>Measure-Level Reductions</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Scope</th>'
            f'<th>Reduction (tCO2e)</th><th>MAC</th></tr>\n{rows}</table>'
        )

    def _html_cumulative_projections(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative projections."""
        projections = data.get("cumulative_projections", [])
        rows = ""
        for p in projections:
            rows += (
                f'<tr><td>{p.get("year", "-")}</td>'
                f'<td>{self._fmt(p.get("annual_reduction", 0))}</td>'
                f'<td>{self._fmt(p.get("cumulative_reduction", 0))}</td></tr>\n'
            )
        return (
            '<h2>Cumulative Projections</h2>\n'
            '<table>\n<tr><th>Year</th><th>Annual (tCO2e)</th>'
            f'<th>Cumulative (tCO2e)</th></tr>\n{rows}</table>'
        )

    def _html_sbti_alignment(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi alignment."""
        sbti = data.get("sbti_alignment", {})
        status = sbti.get("alignment_status", "Not Assessed")
        status_class = "status-pass" if sbti.get("on_track", False) else "status-fail"
        return (
            '<h2>SBTi Alignment</h2>\n'
            f'<div class="{status_class}">'
            f'<strong>Status: {status}</strong> | '
            f'Pathway: {sbti.get("pathway", "1.5C")} | '
            f'Required: {self._fmt(sbti.get("required_annual_pct", 0))}%/yr | '
            f'Achieved: {self._fmt(sbti.get("achieved_annual_pct", 0))}%/yr</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_carbon_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON carbon summary."""
        s = data.get("carbon_summary", {})
        return {
            "baseline_emissions_tco2e": s.get("baseline_emissions_tco2e", 0),
            "total_reduction_tco2e": s.get("total_reduction_tco2e", 0),
            "remaining_emissions_tco2e": s.get("remaining_emissions_tco2e", 0),
            "scope1_reduction_tco2e": s.get("scope1_reduction_tco2e", 0),
            "scope2_reduction_tco2e": s.get("scope2_reduction_tco2e", 0),
            "scope3_reduction_tco2e": s.get("scope3_reduction_tco2e", 0),
            "mac_eur_per_tco2e": s.get("mac_eur_per_tco2e", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        breakdown = data.get("scope_breakdown", {})
        measures = data.get("measure_reductions", [])
        projections = data.get("cumulative_projections", [])
        return {
            "scope_pie": {
                "type": "pie",
                "labels": ["Scope 1", "Scope 2", "Scope 3"],
                "values": [
                    breakdown.get("scope_1", {}).get("reduction_tco2e", 0),
                    breakdown.get("scope_2", {}).get("reduction_tco2e", 0),
                    breakdown.get("scope_3", {}).get("reduction_tco2e", 0),
                ],
            },
            "mac_curve": {
                "type": "bar",
                "labels": [m.get("measure", "") for m in measures],
                "values": [m.get("mac_eur_per_tco2e", 0) for m in measures],
                "widths": [m.get("reduction_tco2e", 0) for m in measures],
            },
            "cumulative_line": {
                "type": "line",
                "labels": [str(p.get("year", "")) for p in projections],
                "series": {
                    "remaining": [p.get("remaining_emissions", 0) for p in projections],
                    "cumulative_reduction": [p.get("cumulative_reduction", 0) for p in projections],
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
            "h3{color:#0d6efd;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".scope-grid{display:flex;gap:15px;margin:15px 0;}"
            ".scope-card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;}"
            ".status-pass{background:#d1e7dd;padding:15px;border-radius:8px;margin:10px 0;}"
            ".status-fail{background:#f8d7da;padding:15px;border-radius:8px;margin:10px 0;}"
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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
