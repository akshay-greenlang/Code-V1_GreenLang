# -*- coding: utf-8 -*-
"""
EnergyAuditReportTemplate - EN 16247-compliant energy audit report for PACK-031.

Generates a full industrial energy audit report compliant with EN 16247-1
(General), EN 16247-2 (Buildings), and EN 16247-3 (Processes). Includes
executive summary, facility description, energy consumption analysis,
end-use breakdown, findings with savings and costs, prioritized
recommendations, and implementation roadmap.

Sections:
    1. Executive Summary
    2. Facility Description
    3. Energy Consumption Analysis
    4. End-Use Breakdown
    5. Audit Findings
    6. Savings Opportunities
    7. Prioritized Recommendations
    8. Implementation Roadmap
    9. Appendices & Data Tables

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnergyAuditReportTemplate:
    """
    EN 16247-compliant industrial energy audit report template.

    Renders comprehensive energy audit reports with facility data,
    consumption analysis, end-use breakdowns, savings opportunities,
    and implementation roadmaps across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    EN16247_SECTIONS: List[str] = [
        "Background and Objectives",
        "Facility Description",
        "Energy Sources and Consumption",
        "Energy End-Use Analysis",
        "Findings and Observations",
        "Energy Saving Opportunities",
        "Recommendations",
        "Implementation Plan",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyAuditReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render EN 16247 energy audit report as Markdown.

        Args:
            data: Audit result data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_facility_description(data),
            self._md_energy_consumption(data),
            self._md_end_use_breakdown(data),
            self._md_findings(data),
            self._md_savings_opportunities(data),
            self._md_recommendations(data),
            self._md_implementation_roadmap(data),
            self._md_appendices(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render EN 16247 energy audit report as self-contained HTML.

        Args:
            data: Audit result data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_facility_description(data),
            self._html_energy_consumption(data),
            self._html_end_use_breakdown(data),
            self._html_findings(data),
            self._html_savings_opportunities(data),
            self._html_recommendations(data),
            self._html_implementation_roadmap(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Audit Report (EN 16247)</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render EN 16247 energy audit report as structured JSON.

        Args:
            data: Audit result data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_audit_report",
            "version": "31.0.0",
            "standard": "EN 16247",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "facility": self._json_facility(data),
            "energy_consumption": self._json_energy_consumption(data),
            "end_use_breakdown": data.get("end_use_breakdown", []),
            "findings": data.get("findings", []),
            "savings_opportunities": data.get("savings_opportunities", []),
            "recommendations": data.get("recommendations", []),
            "implementation_roadmap": data.get("implementation_roadmap", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with audit metadata."""
        title = data.get("title", "Industrial Energy Audit Report")
        facility = data.get("facility_name", "Industrial Facility")
        auditor = data.get("auditor_name", "")
        audit_date = data.get("audit_date", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# {title}\n\n"
            f"**Facility:** {facility}  \n"
            f"**Standard:** EN 16247-1:2022 / EN 16247-3 (Processes)  \n"
            f"**Audit Date:** {audit_date}  \n"
            f"**Lead Auditor:** {auditor}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-031 EnergyAuditReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary section."""
        summary = data.get("executive_summary", {})
        total_consumption = summary.get("total_energy_consumption_mwh", 0)
        total_cost = summary.get("total_energy_cost_eur", 0)
        total_savings_potential = summary.get("total_savings_potential_mwh", 0)
        total_cost_savings = summary.get("total_cost_savings_eur", 0)
        total_investment = summary.get("total_investment_eur", 0)
        avg_payback = summary.get("average_payback_years", 0)
        findings_count = summary.get("findings_count", 0)
        co2_reduction = summary.get("co2_reduction_tonnes", 0)

        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Energy Consumption | {self._fmt(total_consumption)} MWh/yr |\n"
            f"| Total Energy Cost | EUR {self._fmt(total_cost)} /yr |\n"
            f"| Identified Savings Potential | {self._fmt(total_savings_potential)} MWh/yr "
            f"({self._pct(total_savings_potential, total_consumption)}) |\n"
            f"| Estimated Cost Savings | EUR {self._fmt(total_cost_savings)} /yr |\n"
            f"| Required Investment | EUR {self._fmt(total_investment)} |\n"
            f"| Average Simple Payback | {self._fmt(avg_payback, 1)} years |\n"
            f"| CO2 Reduction Potential | {self._fmt(co2_reduction)} tonnes/yr |\n"
            f"| Number of Findings | {findings_count} |"
        )

    def _md_facility_description(self, data: Dict[str, Any]) -> str:
        """Render facility description section."""
        facility = data.get("facility", {})
        lines = [
            "## 2. Facility Description\n",
            f"**Name:** {facility.get('name', '-')}  ",
            f"**Address:** {facility.get('address', '-')}  ",
            f"**Sector:** {facility.get('sector', '-')} / {facility.get('sub_sector', '-')}  ",
            f"**Floor Area:** {self._fmt(facility.get('floor_area_sqm', 0), 0)} sqm  ",
            f"**Operating Hours:** {self._fmt(facility.get('operating_hours_yr', 0), 0)} hrs/yr  ",
            f"**Employees:** {facility.get('employees', '-')}  ",
            f"**Production Volume:** {self._fmt(facility.get('production_volume', 0))} "
            f"{facility.get('production_unit', 'units')}/yr  ",
            f"**Year Built / Last Retrofit:** {facility.get('year_built', '-')} / "
            f"{facility.get('last_retrofit', '-')}",
        ]
        processes = facility.get("key_processes", [])
        if processes:
            lines.append("\n### Key Processes\n")
            for p in processes:
                lines.append(
                    f"- **{p.get('name', '-')}**: {p.get('description', '-')} "
                    f"({self._fmt(p.get('energy_share_pct', 0))}% of total)"
                )
        return "\n".join(lines)

    def _md_energy_consumption(self, data: Dict[str, Any]) -> str:
        """Render energy consumption analysis section."""
        consumption = data.get("energy_consumption", {})
        sources = consumption.get("by_source", [])
        lines = [
            "## 3. Energy Consumption Analysis\n",
            f"**Reporting Period:** {consumption.get('period', '-')}  ",
            f"**Total Primary Energy:** {self._fmt(consumption.get('total_primary_mwh', 0))} MWh  ",
            f"**Total Final Energy:** {self._fmt(consumption.get('total_final_mwh', 0))} MWh  ",
            f"**Total Cost:** EUR {self._fmt(consumption.get('total_cost_eur', 0))}",
        ]
        if sources:
            lines.extend([
                "\n### Energy Sources\n",
                "| Source | Consumption (MWh) | Cost (EUR) | Share (%) | Unit Price |",
                "|--------|------------------|------------|-----------|------------|",
            ])
            for s in sources:
                lines.append(
                    f"| {s.get('source_name', '-')} "
                    f"| {self._fmt(s.get('consumption_mwh', 0))} "
                    f"| {self._fmt(s.get('cost_eur', 0))} "
                    f"| {self._fmt(s.get('share_pct', 0))}% "
                    f"| {self._fmt(s.get('unit_price', 0), 4)} |"
                )
        monthly = consumption.get("monthly_profile", [])
        if monthly:
            lines.extend([
                "\n### Monthly Consumption Profile\n",
                "| Month | Electricity (MWh) | Gas (MWh) | Other (MWh) | Total (MWh) |",
                "|-------|-------------------|-----------|-------------|-------------|",
            ])
            for m in monthly:
                lines.append(
                    f"| {m.get('month', '-')} "
                    f"| {self._fmt(m.get('electricity_mwh', 0))} "
                    f"| {self._fmt(m.get('gas_mwh', 0))} "
                    f"| {self._fmt(m.get('other_mwh', 0))} "
                    f"| {self._fmt(m.get('total_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render end-use breakdown section."""
        end_uses = data.get("end_use_breakdown", [])
        if not end_uses:
            return "## 4. End-Use Breakdown\n\n_No end-use data available._"
        lines = [
            "## 4. End-Use Breakdown\n",
            "| End Use | Energy (MWh) | Share (%) | Cost (EUR) | Specific Consumption |",
            "|---------|-------------|-----------|------------|---------------------|",
        ]
        for eu in end_uses:
            lines.append(
                f"| {eu.get('end_use', '-')} "
                f"| {self._fmt(eu.get('energy_mwh', 0))} "
                f"| {self._fmt(eu.get('share_pct', 0))}% "
                f"| {self._fmt(eu.get('cost_eur', 0))} "
                f"| {self._fmt(eu.get('specific_consumption', 0))} "
                f"{eu.get('specific_unit', 'kWh/unit')} |"
            )
        return "\n".join(lines)

    def _md_findings(self, data: Dict[str, Any]) -> str:
        """Render audit findings section."""
        findings = data.get("findings", [])
        if not findings:
            return "## 5. Audit Findings\n\n_No findings recorded._"
        lines = ["## 5. Audit Findings\n"]
        for i, f in enumerate(findings, 1):
            lines.extend([
                f"### Finding {i}: {f.get('title', 'Untitled')}\n",
                f"- **Area:** {f.get('area', '-')}",
                f"- **Category:** {f.get('category', '-')}",
                f"- **Current State:** {f.get('current_state', '-')}",
                f"- **Issue:** {f.get('issue', '-')}",
                f"- **Energy Impact:** {self._fmt(f.get('energy_impact_mwh', 0))} MWh/yr",
                f"- **Priority:** {f.get('priority', '-')}",
                "",
            ])
        return "\n".join(lines)

    def _md_savings_opportunities(self, data: Dict[str, Any]) -> str:
        """Render savings opportunities section."""
        opps = data.get("savings_opportunities", [])
        if not opps:
            return "## 6. Energy Saving Opportunities\n\n_No savings opportunities identified._"
        lines = [
            "## 6. Energy Saving Opportunities\n",
            "| # | Opportunity | Energy Savings (MWh/yr) | Cost Savings (EUR/yr) "
            "| Investment (EUR) | Payback (yr) | CO2 Reduction (t/yr) |",
            "|---|-----------|------------------------|----------------------"
            "|-----------------|-------------|---------------------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('title', '-')} "
                f"| {self._fmt(o.get('energy_savings_mwh', 0))} "
                f"| {self._fmt(o.get('cost_savings_eur', 0))} "
                f"| {self._fmt(o.get('investment_eur', 0))} "
                f"| {self._fmt(o.get('payback_years', 0), 1)} "
                f"| {self._fmt(o.get('co2_reduction_tonnes', 0))} |"
            )
        total_savings = sum(o.get("energy_savings_mwh", 0) for o in opps)
        total_cost_savings = sum(o.get("cost_savings_eur", 0) for o in opps)
        total_investment = sum(o.get("investment_eur", 0) for o in opps)
        total_co2 = sum(o.get("co2_reduction_tonnes", 0) for o in opps)
        lines.append(
            f"| | **TOTAL** | **{self._fmt(total_savings)}** "
            f"| **{self._fmt(total_cost_savings)}** "
            f"| **{self._fmt(total_investment)}** | - "
            f"| **{self._fmt(total_co2)}** |"
        )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render prioritized recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            return "## 7. Prioritized Recommendations\n\n_No recommendations._"
        lines = [
            "## 7. Prioritized Recommendations\n",
            "| Priority | Recommendation | Category | Effort | Impact |",
            "|----------|---------------|----------|--------|--------|",
        ]
        for r in recs:
            lines.append(
                f"| {r.get('priority', '-')} "
                f"| {r.get('title', '-')} "
                f"| {r.get('category', '-')} "
                f"| {r.get('effort', '-')} "
                f"| {r.get('impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_implementation_roadmap(self, data: Dict[str, Any]) -> str:
        """Render implementation roadmap section."""
        phases = data.get("implementation_roadmap", [])
        if not phases:
            return "## 8. Implementation Roadmap\n\n_No roadmap data._"
        lines = ["## 8. Implementation Roadmap\n"]
        for phase in phases:
            lines.extend([
                f"### {phase.get('phase_name', 'Phase')}: "
                f"{phase.get('timeframe', '-')}\n",
                f"**Budget:** EUR {self._fmt(phase.get('budget_eur', 0))}  ",
                f"**Expected Savings:** {self._fmt(phase.get('expected_savings_mwh', 0))} MWh/yr\n",
            ])
            actions = phase.get("actions", [])
            for a in actions:
                lines.append(
                    f"- {a.get('action', '-')} "
                    f"(Owner: {a.get('owner', '-')}, Deadline: {a.get('deadline', '-')})"
                )
            lines.append("")
        return "\n".join(lines)

    def _md_appendices(self, data: Dict[str, Any]) -> str:
        """Render appendices section."""
        return (
            "## 9. Appendices\n\n"
            "- **Appendix A:** Measurement equipment and methodology\n"
            "- **Appendix B:** Detailed calculation worksheets\n"
            "- **Appendix C:** Metering and sub-metering data\n"
            "- **Appendix D:** Photographs and thermal images\n"
            "- **Appendix E:** EN 16247 compliance checklist"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = data.get("title", "Industrial Energy Audit Report")
        facility = data.get("facility_name", "Industrial Facility")
        return (
            f'<h1>{title}</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Standard: EN 16247-1:2022</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Energy</span>'
            f'<span class="value">{self._fmt(s.get("total_energy_consumption_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Energy Cost</span>'
            f'<span class="value">EUR {self._fmt(s.get("total_energy_cost_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Savings Potential</span>'
            f'<span class="value">{self._fmt(s.get("total_savings_potential_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(s.get("average_payback_years", 0), 1)} yrs</span></div>\n'
            '</div>'
        )

    def _html_facility_description(self, data: Dict[str, Any]) -> str:
        """Render HTML facility description."""
        f = data.get("facility", {})
        return (
            '<h2>Facility Description</h2>\n'
            f'<p><strong>{f.get("name", "-")}</strong> - '
            f'{f.get("sector", "-")} | '
            f'{self._fmt(f.get("floor_area_sqm", 0), 0)} sqm | '
            f'{self._fmt(f.get("operating_hours_yr", 0), 0)} hrs/yr</p>'
        )

    def _html_energy_consumption(self, data: Dict[str, Any]) -> str:
        """Render HTML energy consumption table."""
        sources = data.get("energy_consumption", {}).get("by_source", [])
        rows = ""
        for s in sources:
            rows += (
                f'<tr><td>{s.get("source_name", "-")}</td>'
                f'<td>{self._fmt(s.get("consumption_mwh", 0))}</td>'
                f'<td>{self._fmt(s.get("cost_eur", 0))}</td>'
                f'<td>{self._fmt(s.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>Energy Consumption Analysis</h2>\n'
            '<table>\n<tr><th>Source</th><th>MWh</th><th>Cost (EUR)</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_end_use_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML end-use breakdown."""
        end_uses = data.get("end_use_breakdown", [])
        rows = ""
        for eu in end_uses:
            rows += (
                f'<tr><td>{eu.get("end_use", "-")}</td>'
                f'<td>{self._fmt(eu.get("energy_mwh", 0))}</td>'
                f'<td>{self._fmt(eu.get("share_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>End-Use Breakdown</h2>\n'
            '<table>\n<tr><th>End Use</th><th>Energy (MWh)</th>'
            f'<th>Share</th></tr>\n{rows}</table>'
        )

    def _html_findings(self, data: Dict[str, Any]) -> str:
        """Render HTML findings."""
        findings = data.get("findings", [])
        items = ""
        for f in findings:
            items += (
                f'<div class="finding"><h3>{f.get("title", "-")}</h3>'
                f'<p>Area: {f.get("area", "-")} | Priority: {f.get("priority", "-")} | '
                f'Impact: {self._fmt(f.get("energy_impact_mwh", 0))} MWh/yr</p></div>\n'
            )
        return f'<h2>Audit Findings</h2>\n{items}'

    def _html_savings_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML savings opportunities table."""
        opps = data.get("savings_opportunities", [])
        rows = ""
        for o in opps:
            rows += (
                f'<tr><td>{o.get("title", "-")}</td>'
                f'<td>{self._fmt(o.get("energy_savings_mwh", 0))}</td>'
                f'<td>{self._fmt(o.get("cost_savings_eur", 0))}</td>'
                f'<td>{self._fmt(o.get("payback_years", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Savings Opportunities</h2>\n'
            '<table>\n<tr><th>Opportunity</th><th>Savings (MWh)</th>'
            f'<th>Cost Savings (EUR)</th><th>Payback (yr)</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        items = "".join(
            f'<li><strong>[{r.get("priority", "-")}]</strong> {r.get("title", "-")} '
            f'({r.get("category", "-")})</li>\n'
            for r in recs
        )
        return f'<h2>Prioritized Recommendations</h2>\n<ol>\n{items}</ol>'

    def _html_implementation_roadmap(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation roadmap."""
        phases = data.get("implementation_roadmap", [])
        content = ""
        for phase in phases:
            actions = "".join(
                f'<li>{a.get("action", "-")} (Owner: {a.get("owner", "-")})</li>'
                for a in phase.get("actions", [])
            )
            content += (
                f'<div class="phase"><h3>{phase.get("phase_name", "Phase")} - '
                f'{phase.get("timeframe", "-")}</h3>'
                f'<p>Budget: EUR {self._fmt(phase.get("budget_eur", 0))}</p>'
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
            "total_energy_consumption_mwh": s.get("total_energy_consumption_mwh", 0),
            "total_energy_cost_eur": s.get("total_energy_cost_eur", 0),
            "total_savings_potential_mwh": s.get("total_savings_potential_mwh", 0),
            "total_cost_savings_eur": s.get("total_cost_savings_eur", 0),
            "total_investment_eur": s.get("total_investment_eur", 0),
            "average_payback_years": s.get("average_payback_years", 0),
            "co2_reduction_tonnes": s.get("co2_reduction_tonnes", 0),
            "findings_count": s.get("findings_count", 0),
        }

    def _json_facility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON facility description."""
        return data.get("facility", {})

    def _json_energy_consumption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON energy consumption data."""
        return data.get("energy_consumption", {})

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        end_uses = data.get("end_use_breakdown", [])
        sources = data.get("energy_consumption", {}).get("by_source", [])
        monthly = data.get("energy_consumption", {}).get("monthly_profile", [])
        return {
            "end_use_pie": {
                "type": "pie",
                "labels": [eu.get("end_use", "") for eu in end_uses],
                "values": [eu.get("energy_mwh", 0) for eu in end_uses],
            },
            "source_bar": {
                "type": "bar",
                "labels": [s.get("source_name", "") for s in sources],
                "values": [s.get("consumption_mwh", 0) for s in sources],
            },
            "monthly_line": {
                "type": "line",
                "labels": [m.get("month", "") for m in monthly],
                "series": {
                    "electricity": [m.get("electricity_mwh", 0) for m in monthly],
                    "gas": [m.get("gas_mwh", 0) for m in monthly],
                    "total": [m.get("total_mwh", 0) for m in monthly],
                },
            },
            "savings_waterfall": {
                "type": "waterfall",
                "labels": [
                    o.get("title", "")
                    for o in data.get("savings_opportunities", [])
                ],
                "values": [
                    o.get("energy_savings_mwh", 0)
                    for o in data.get("savings_opportunities", [])
                ],
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
            ".finding{background:#fff3cd;border-left:4px solid #ffc107;padding:10px;margin:10px 0;}"
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
