# -*- coding: utf-8 -*-
"""
CarbonImpactReportTemplate - Carbon benefits of DR for PACK-037.

Generates carbon impact reports for demand response participation using
marginal emission factors, avoided generation analysis, SBTi alignment
assessment, and CSRD/ESRS reporting readiness for DR carbon credits.

Sections:
    1. Carbon Impact Summary
    2. Marginal Emission Factor Analysis
    3. Event-Level Carbon Avoidance
    4. Avoided Generation Mix
    5. SBTi Alignment Assessment
    6. CSRD Reporting Integration

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - GHG Protocol Scope 2 Guidance (market-based method)
    - SBTi Corporate Net-Zero Standard
    - EU CSRD / ESRS E1 Climate Change
    - WRI Avoided Emissions Guidance

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class CarbonImpactReportTemplate:
    """
    DR carbon impact report template.

    Renders carbon impact analysis for demand response participation with
    marginal emission factors, avoided generation mix, SBTi alignment,
    and CSRD reporting integration across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CarbonImpactReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render carbon impact report as Markdown.

        Args:
            data: Carbon impact engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_carbon_summary(data),
            self._md_marginal_emission_factors(data),
            self._md_event_carbon_avoidance(data),
            self._md_avoided_generation_mix(data),
            self._md_sbti_alignment(data),
            self._md_csrd_integration(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render carbon impact report as self-contained HTML.

        Args:
            data: Carbon impact engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_carbon_summary(data),
            self._html_marginal_emission_factors(data),
            self._html_event_carbon_avoidance(data),
            self._html_avoided_generation_mix(data),
            self._html_sbti_alignment(data),
            self._html_csrd_integration(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DR Carbon Impact Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render carbon impact report as structured JSON.

        Args:
            data: Carbon impact engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "carbon_impact_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "carbon_summary": self._json_carbon_summary(data),
            "marginal_emission_factors": data.get("marginal_emission_factors", []),
            "event_carbon_avoidance": data.get("event_carbon_avoidance", []),
            "avoided_generation_mix": data.get("avoided_generation_mix", {}),
            "sbti_alignment": data.get("sbti_alignment", {}),
            "csrd_integration": data.get("csrd_integration", {}),
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
            f"# Demand Response Carbon Impact Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Grid Region:** {data.get('grid_region', '')}  \n"
            f"**Reporting Standard:** {data.get('reporting_standard', 'GHG Protocol')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 CarbonImpactReportTemplate v37.0.0\n\n---"
        )

    def _md_carbon_summary(self, data: Dict[str, Any]) -> str:
        """Render carbon impact summary section."""
        summary = data.get("carbon_summary", {})
        return (
            "## 1. Carbon Impact Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total CO2e Avoided | {self._fmt(summary.get('total_co2e_avoided_tonnes', 0))} tonnes |\n"
            f"| DR Events Analyzed | {summary.get('events_analyzed', 0)} |\n"
            f"| Avg Marginal EF | {self._fmt(summary.get('avg_marginal_ef_tco2e_mwh', 0), 4)} tCO2e/MWh |\n"
            f"| Total Energy Curtailed | {self._format_energy(summary.get('total_energy_curtailed_mwh', 0))} |\n"
            f"| Carbon Value (Shadow Price) | {self._format_currency(summary.get('carbon_value', 0))} |\n"
            f"| Equivalent Trees Planted | {self._fmt(summary.get('equivalent_trees', 0), 0)} |\n"
            f"| Equivalent Cars Off Road | {self._fmt(summary.get('equivalent_cars', 0), 0)} |"
        )

    def _md_marginal_emission_factors(self, data: Dict[str, Any]) -> str:
        """Render marginal emission factor analysis section."""
        factors = data.get("marginal_emission_factors", [])
        if not factors:
            return "## 2. Marginal Emission Factor Analysis\n\n_No emission factor data._"
        lines = [
            "## 2. Marginal Emission Factor Analysis\n",
            "| Time Period | Marginal EF (tCO2e/MWh) | Grid EF (tCO2e/MWh) | Marginal Source | Confidence |",
            "|------------|------------------------:|---------------------:|----------------|-----------|",
        ]
        for f in factors:
            lines.append(
                f"| {f.get('time_period', '-')} "
                f"| {self._fmt(f.get('marginal_ef', 0), 4)} "
                f"| {self._fmt(f.get('grid_ef', 0), 4)} "
                f"| {f.get('marginal_source', '-')} "
                f"| {f.get('confidence', '-')} |"
            )
        return "\n".join(lines)

    def _md_event_carbon_avoidance(self, data: Dict[str, Any]) -> str:
        """Render event-level carbon avoidance section."""
        events = data.get("event_carbon_avoidance", [])
        if not events:
            return "## 3. Event-Level Carbon Avoidance\n\n_No event data available._"
        lines = [
            "## 3. Event-Level Carbon Avoidance\n",
            "| Event Date | Energy Curtailed (MWh) | Marginal EF | CO2e Avoided (t) | Carbon Value |",
            "|-----------|----------------------:|------------:|-----------------:|-------------:|",
        ]
        for e in events:
            lines.append(
                f"| {e.get('event_date', '-')} "
                f"| {self._fmt(e.get('energy_curtailed_mwh', 0), 2)} "
                f"| {self._fmt(e.get('marginal_ef', 0), 4)} "
                f"| {self._fmt(e.get('co2e_avoided_tonnes', 0), 3)} "
                f"| {self._format_currency(e.get('carbon_value', 0))} |"
            )
        total_co2e = sum(e.get("co2e_avoided_tonnes", 0) for e in events)
        total_value = sum(e.get("carbon_value", 0) for e in events)
        lines.append(
            f"| **TOTAL** | | | **{self._fmt(total_co2e, 3)}** "
            f"| **{self._format_currency(total_value)}** |"
        )
        return "\n".join(lines)

    def _md_avoided_generation_mix(self, data: Dict[str, Any]) -> str:
        """Render avoided generation mix section."""
        mix = data.get("avoided_generation_mix", {})
        sources = mix.get("sources", [])
        if not sources:
            return "## 4. Avoided Generation Mix\n\n_No generation mix data._"
        lines = [
            "## 4. Avoided Generation Mix\n",
            f"**Analysis Method:** {mix.get('method', 'Marginal Unit Analysis')}  ",
            f"**Data Source:** {mix.get('data_source', '-')}\n",
            "| Generation Source | Share (%) | EF (tCO2e/MWh) | CO2e Avoided (t) |",
            "|-----------------|----------:|----------------:|------------------:|",
        ]
        for s in sources:
            lines.append(
                f"| {s.get('source', '-')} "
                f"| {self._fmt(s.get('share_pct', 0))}% "
                f"| {self._fmt(s.get('emission_factor', 0), 4)} "
                f"| {self._fmt(s.get('co2e_avoided', 0), 3)} |"
            )
        return "\n".join(lines)

    def _md_sbti_alignment(self, data: Dict[str, Any]) -> str:
        """Render SBTi alignment assessment section."""
        sbti = data.get("sbti_alignment", {})
        lines = [
            "## 5. SBTi Alignment Assessment\n",
            f"- **Target Pathway:** {sbti.get('pathway', '1.5C')}",
            f"- **DR Contribution to Target:** {self._fmt(sbti.get('dr_contribution_pct', 0))}%",
            f"- **Annual Target Reduction:** {self._fmt(sbti.get('annual_target_tco2e', 0))} tCO2e",
            f"- **DR Avoided Emissions:** {self._fmt(sbti.get('dr_avoided_tco2e', 0))} tCO2e",
            f"- **Alignment Status:** {sbti.get('alignment_status', 'Not Assessed')}",
            f"- **Classification:** {sbti.get('classification', 'Avoided Emissions (Scope 2)')}",
            f"- **SBTi Eligibility:** {sbti.get('eligibility_note', '-')}",
        ]
        return "\n".join(lines)

    def _md_csrd_integration(self, data: Dict[str, Any]) -> str:
        """Render CSRD reporting integration section."""
        csrd = data.get("csrd_integration", {})
        lines = [
            "## 6. CSRD Reporting Integration\n",
            f"- **ESRS Standard:** {csrd.get('esrs_standard', 'E1 Climate Change')}",
            f"- **Disclosure Requirement:** {csrd.get('disclosure_requirement', 'E1-6 GHG emissions')}",
            f"- **Reporting Category:** {csrd.get('reporting_category', 'Scope 2 avoided emissions')}",
            f"- **Data Quality Level:** {csrd.get('data_quality_level', '-')}",
            f"- **Verification Status:** {csrd.get('verification_status', 'Pending')}",
            f"- **Reporting Period Alignment:** {csrd.get('period_alignment', '-')}",
        ]
        metrics = csrd.get("reportable_metrics", [])
        if metrics:
            lines.append("\n### Reportable Metrics\n")
            lines.append("| Metric | Value | Unit |")
            lines.append("|--------|------:|------|")
            for m in metrics:
                lines.append(
                    f"| {m.get('metric', '-')} "
                    f"| {self._fmt(m.get('value', 0), 2)} "
                    f"| {m.get('unit', '-')} |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>DR Carbon Impact Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Grid Region: {data.get("grid_region", "-")} | '
            f'Standard: {data.get("reporting_standard", "GHG Protocol")}</p>'
        )

    def _html_carbon_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon summary cards."""
        s = data.get("carbon_summary", {})
        return (
            '<h2>Carbon Impact Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-green"><span class="label">CO2e Avoided</span>'
            f'<span class="value">{self._fmt(s.get("total_co2e_avoided_tonnes", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Events Analyzed</span>'
            f'<span class="value">{s.get("events_analyzed", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Avg Marginal EF</span>'
            f'<span class="value">{self._fmt(s.get("avg_marginal_ef_tco2e_mwh", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">Carbon Value</span>'
            f'<span class="value">{self._format_currency(s.get("carbon_value", 0))}</span></div>\n'
            '</div>'
        )

    def _html_marginal_emission_factors(self, data: Dict[str, Any]) -> str:
        """Render HTML marginal emission factors table."""
        factors = data.get("marginal_emission_factors", [])
        rows = ""
        for f in factors:
            rows += (
                f'<tr><td>{f.get("time_period", "-")}</td>'
                f'<td>{self._fmt(f.get("marginal_ef", 0), 4)}</td>'
                f'<td>{self._fmt(f.get("grid_ef", 0), 4)}</td>'
                f'<td>{f.get("marginal_source", "-")}</td></tr>\n'
            )
        return (
            '<h2>Marginal Emission Factors</h2>\n'
            '<table>\n<tr><th>Period</th><th>Marginal EF</th>'
            f'<th>Grid EF</th><th>Source</th></tr>\n{rows}</table>'
        )

    def _html_event_carbon_avoidance(self, data: Dict[str, Any]) -> str:
        """Render HTML event carbon avoidance table."""
        events = data.get("event_carbon_avoidance", [])
        rows = ""
        for e in events:
            rows += (
                f'<tr><td>{e.get("event_date", "-")}</td>'
                f'<td>{self._fmt(e.get("energy_curtailed_mwh", 0), 2)}</td>'
                f'<td>{self._fmt(e.get("co2e_avoided_tonnes", 0), 3)}</td>'
                f'<td>{self._format_currency(e.get("carbon_value", 0))}</td></tr>\n'
            )
        return (
            '<h2>Event Carbon Avoidance</h2>\n'
            '<table>\n<tr><th>Event Date</th><th>Energy MWh</th>'
            f'<th>CO2e Avoided t</th><th>Carbon Value</th></tr>\n{rows}</table>'
        )

    def _html_avoided_generation_mix(self, data: Dict[str, Any]) -> str:
        """Render HTML avoided generation mix."""
        sources = data.get("avoided_generation_mix", {}).get("sources", [])
        rows = ""
        for s in sources:
            rows += (
                f'<tr><td>{s.get("source", "-")}</td>'
                f'<td>{self._fmt(s.get("share_pct", 0))}%</td>'
                f'<td>{self._fmt(s.get("emission_factor", 0), 4)}</td></tr>\n'
            )
        return (
            '<h2>Avoided Generation Mix</h2>\n'
            '<table>\n<tr><th>Source</th><th>Share</th>'
            f'<th>EF (tCO2e/MWh)</th></tr>\n{rows}</table>'
        )

    def _html_sbti_alignment(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi alignment."""
        sbti = data.get("sbti_alignment", {})
        status_cls = "status-pass" if sbti.get("alignment_status") == "Aligned" else "status-fail"
        return (
            '<h2>SBTi Alignment</h2>\n'
            f'<div class="{status_cls}">'
            f'<strong>Status: {sbti.get("alignment_status", "Not Assessed")}</strong> | '
            f'Pathway: {sbti.get("pathway", "1.5C")} | '
            f'DR Contribution: {self._fmt(sbti.get("dr_contribution_pct", 0))}%</div>'
        )

    def _html_csrd_integration(self, data: Dict[str, Any]) -> str:
        """Render HTML CSRD integration."""
        csrd = data.get("csrd_integration", {})
        return (
            '<h2>CSRD Reporting Integration</h2>\n'
            f'<div class="csrd-box">'
            f'<strong>ESRS: {csrd.get("esrs_standard", "E1")}</strong> | '
            f'Disclosure: {csrd.get("disclosure_requirement", "-")} | '
            f'Quality: {csrd.get("data_quality_level", "-")} | '
            f'Verification: {csrd.get("verification_status", "Pending")}</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_carbon_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON carbon summary."""
        s = data.get("carbon_summary", {})
        return {
            "total_co2e_avoided_tonnes": s.get("total_co2e_avoided_tonnes", 0),
            "events_analyzed": s.get("events_analyzed", 0),
            "avg_marginal_ef_tco2e_mwh": s.get("avg_marginal_ef_tco2e_mwh", 0),
            "total_energy_curtailed_mwh": s.get("total_energy_curtailed_mwh", 0),
            "carbon_value": s.get("carbon_value", 0),
            "equivalent_trees": s.get("equivalent_trees", 0),
            "equivalent_cars": s.get("equivalent_cars", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        events = data.get("event_carbon_avoidance", [])
        sources = data.get("avoided_generation_mix", {}).get("sources", [])
        factors = data.get("marginal_emission_factors", [])
        return {
            "event_avoidance_bar": {
                "type": "bar",
                "labels": [e.get("event_date", "") for e in events],
                "values": [e.get("co2e_avoided_tonnes", 0) for e in events],
            },
            "generation_mix_pie": {
                "type": "pie",
                "labels": [s.get("source", "") for s in sources],
                "values": [s.get("share_pct", 0) for s in sources],
            },
            "marginal_ef_line": {
                "type": "line",
                "labels": [f.get("time_period", "") for f in factors],
                "series": {
                    "marginal": [f.get("marginal_ef", 0) for f in factors],
                    "grid_average": [f.get("grid_ef", 0) for f in factors],
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".card-green{background:#d1e7dd;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-pass{background:#d1e7dd;padding:15px;border-radius:8px;margin:10px 0;}"
            ".status-fail{background:#f8d7da;padding:15px;border-radius:8px;margin:10px 0;}"
            ".csrd-box{background:#e2e3f1;padding:15px;border-radius:8px;margin:10px 0;}"
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

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
