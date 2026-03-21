# -*- coding: utf-8 -*-
"""
AccelerationStrategyReportTemplate - Executive acceleration strategy for PACK-022.

Renders an executive-level net-zero acceleration strategy document that pulls
together scenario analysis, SDA pathways, supplier engagement, climate finance,
progress tracking, temperature alignment, VCMI claims, multi-entity views,
risk assessment, and a 12-month implementation roadmap.

Sections:
    1. Executive Summary
    2. Current Net-Zero Position
    3. Scenario Analysis Summary
    4. SDA Pathway (if applicable)
    5. Supplier Engagement Program
    6. Climate Investment Plan
    7. Progress Against Targets
    8. Temperature Alignment
    9. VCMI Claims Status
   10. Multi-Entity View
   11. Key Risks & Mitigations
   12. Implementation Roadmap (next 12 months)
   13. Board Recommendations

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


class AccelerationStrategyReportTemplate:
    """
    Executive net-zero acceleration strategy report template.

    Consolidates all PACK-022 analysis into a board-level strategy document
    with scenario analysis, SDA pathways, supplier engagement, climate
    finance, temperature alignment, VCMI claims, and implementation roadmap.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_current_position(data),
            self._md_scenario_analysis(data),
            self._md_sda_pathway(data),
            self._md_supplier_engagement(data),
            self._md_climate_investment(data),
            self._md_progress(data),
            self._md_temperature_alignment(data),
            self._md_vcmi_claims(data),
            self._md_multi_entity(data),
            self._md_risks(data),
            self._md_roadmap(data),
            self._md_board_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_current_position(data),
            self._html_scenario_analysis(data),
            self._html_sda_pathway(data),
            self._html_supplier_engagement(data),
            self._html_climate_investment(data),
            self._html_progress(data),
            self._html_temperature_alignment(data),
            self._html_vcmi_claims(data),
            self._html_multi_entity(data),
            self._html_risks(data),
            self._html_roadmap(data),
            self._html_board_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Net Zero Acceleration Strategy</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "acceleration_strategy_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "current_position": data.get("current_position", {}),
            "scenario_analysis": data.get("scenario_analysis", {}),
            "sda_pathway": data.get("sda_pathway", {}),
            "supplier_engagement": data.get("supplier_engagement", {}),
            "climate_investment": data.get("climate_investment", {}),
            "progress": data.get("progress", {}),
            "temperature_alignment": data.get("temperature_alignment", {}),
            "vcmi_claims": data.get("vcmi_claims", {}),
            "multi_entity": data.get("multi_entity", {}),
            "risks": data.get("risks", []),
            "roadmap": data.get("roadmap", []),
            "board_recommendations": data.get("board_recommendations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Net Zero Acceleration Strategy\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Classification:** Board Confidential\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        pos = data.get("current_position", {})
        scenario = data.get("scenario_analysis", {})
        temp = data.get("temperature_alignment", {})
        vcmi = data.get("vcmi_claims", {})
        return (
            "## 1. Executive Summary\n\n"
            f"This acceleration strategy outlines the path for **{data.get('org_name', '')}** "
            f"to achieve net-zero emissions by **{pos.get('net_zero_target_year', 'N/A')}**.\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Current Emissions | {_dec_comma(pos.get('current_tco2e', 0))} tCO2e |\n"
            f"| Base Year Emissions | {_dec_comma(pos.get('base_year_tco2e', 0))} tCO2e |\n"
            f"| Reduction Achieved | {_dec(pos.get('reduction_achieved_pct', 0))}% |\n"
            f"| Recommended Scenario | {scenario.get('recommended_scenario', 'N/A')} |\n"
            f"| Temperature Score | {_dec(temp.get('temperature_c', 0), 1)} C |\n"
            f"| VCMI Tier | {vcmi.get('tier', 'N/A')} |\n"
            f"| Net Zero Target Year | {pos.get('net_zero_target_year', 'N/A')} |"
        )

    def _md_current_position(self, data: Dict[str, Any]) -> str:
        pos = data.get("current_position", {})
        return (
            "## 2. Current Net-Zero Position\n\n"
            f"- **Base Year:** {pos.get('base_year', 'N/A')}\n"
            f"- **Base Year Emissions:** {_dec_comma(pos.get('base_year_tco2e', 0))} tCO2e\n"
            f"- **Current Year Emissions:** {_dec_comma(pos.get('current_tco2e', 0))} tCO2e\n"
            f"- **Absolute Reduction:** {_dec_comma(pos.get('absolute_reduction_tco2e', 0))} tCO2e\n"
            f"- **Reduction Achieved:** {_dec(pos.get('reduction_achieved_pct', 0))}%\n"
            f"- **Required Reduction (target):** {_dec(pos.get('required_reduction_pct', 0))}%\n"
            f"- **Gap to Target:** {_dec(pos.get('gap_pct', 0))}%\n"
            f"- **Scope 1:** {_dec_comma(pos.get('scope1_tco2e', 0))} tCO2e\n"
            f"- **Scope 2:** {_dec_comma(pos.get('scope2_tco2e', 0))} tCO2e\n"
            f"- **Scope 3:** {_dec_comma(pos.get('scope3_tco2e', 0))} tCO2e"
        )

    def _md_scenario_analysis(self, data: Dict[str, Any]) -> str:
        sa = data.get("scenario_analysis", {})
        scenarios = sa.get("scenarios", [])
        lines = [
            "## 3. Scenario Analysis Summary\n",
            f"**Recommended Scenario:** {sa.get('recommended_scenario', 'N/A')}  \n"
            f"**Decision Score:** {_dec(sa.get('decision_score', 0), 1)} / 10.0\n",
            "| Scenario | Target Year | Ambition | Cost Score | Risk Score | Total |",
            "|----------|:-----------:|----------|:---------:|:---------:|:-----:|",
        ]
        for sc in scenarios:
            lines.append(
                f"| {sc.get('name', '-')} "
                f"| {sc.get('target_year', '-')} "
                f"| {sc.get('ambition', '-')} "
                f"| {_dec(sc.get('cost_score', 0), 1)} "
                f"| {_dec(sc.get('risk_score', 0), 1)} "
                f"| {_dec(sc.get('total_score', 0), 1)} |"
            )
        if not scenarios:
            lines.append("| _No scenarios_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_sda_pathway(self, data: Dict[str, Any]) -> str:
        sda = data.get("sda_pathway", {})
        if not sda:
            return "## 4. SDA Pathway\n\n_Not applicable - cross-sector pathway selected._"
        return (
            "## 4. SDA Pathway\n\n"
            f"- **Sector:** {sda.get('sector', 'N/A')}\n"
            f"- **Base Year Intensity:** {_dec(sda.get('base_intensity', 0), 4)}\n"
            f"- **Current Intensity:** {_dec(sda.get('current_intensity', 0), 4)}\n"
            f"- **Convergence Target:** {_dec(sda.get('convergence_target', 0), 4)}\n"
            f"- **Annual Rate:** {_dec(sda.get('annual_rate_pct', 0))}%\n"
            f"- **On Track:** {'Yes' if sda.get('on_track', False) else 'No'}"
        )

    def _md_supplier_engagement(self, data: Dict[str, Any]) -> str:
        se = data.get("supplier_engagement", {})
        return (
            "## 5. Supplier Engagement Program\n\n"
            f"- **Suppliers in Scope:** {se.get('total_suppliers', 0)}\n"
            f"- **Scope 3 Coverage:** {_dec(se.get('scope3_coverage_pct', 0))}%\n"
            f"- **SBTi Adoption Rate:** {_dec(se.get('sbti_adoption_pct', 0))}%\n"
            f"- **Engagement Response Rate:** {_dec(se.get('response_rate_pct', 0))}%\n"
            f"- **Estimated Reduction Potential:** {_dec_comma(se.get('reduction_potential_tco2e', 0))} tCO2e\n"
            f"- **Program Status:** {se.get('status', 'N/A')}"
        )

    def _md_climate_investment(self, data: Dict[str, Any]) -> str:
        ci = data.get("climate_investment", {})
        return (
            "## 6. Climate Investment Plan\n\n"
            f"- **Total Climate CapEx:** EUR {_dec_comma(ci.get('total_capex_eur', 0), 0)}\n"
            f"- **Taxonomy Aligned:** {_dec(ci.get('taxonomy_aligned_pct', 0))}%\n"
            f"- **Weighted IRR:** {_dec(ci.get('weighted_irr_pct', 0), 1)}%\n"
            f"- **Total Abatement:** {_dec_comma(ci.get('total_abatement_tco2e', 0))} tCO2e/yr\n"
            f"- **Avg Cost per tCO2e:** EUR {_dec_comma(ci.get('avg_cost_per_tco2e', 0))}\n"
            f"- **Green Bond Eligible:** EUR {_dec_comma(ci.get('green_bond_eligible_eur', 0), 0)}"
        )

    def _md_progress(self, data: Dict[str, Any]) -> str:
        progress = data.get("progress", {})
        kpis = progress.get("kpis", [])
        lines = [
            "## 7. Progress Against Targets\n",
            "| KPI | Target | Actual | Variance | Status |",
            "|-----|--------|--------|:--------:|:------:|",
        ]
        for kpi in kpis:
            status = kpi.get("status", "AMBER").upper()
            lines.append(
                f"| {kpi.get('name', '-')} "
                f"| {kpi.get('target', '-')} "
                f"| {kpi.get('actual', '-')} "
                f"| {kpi.get('variance', '-')} "
                f"| {status} |"
            )
        if not kpis:
            lines.append("| _No KPI data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_temperature_alignment(self, data: Dict[str, Any]) -> str:
        temp = data.get("temperature_alignment", {})
        return (
            "## 8. Temperature Alignment\n\n"
            f"- **Portfolio Score:** {_dec(temp.get('temperature_c', 0), 1)} C\n"
            f"- **Primary Method:** {temp.get('method', 'WATS')}\n"
            f"- **Alignment:** {'1.5C aligned' if float(Decimal(str(temp.get('temperature_c', 0)))) <= 1.5 else '2C aligned' if float(Decimal(str(temp.get('temperature_c', 0)))) <= 2.0 else 'Above 2C'}\n"
            f"- **Sector Benchmark:** {_dec(temp.get('benchmark_c', 0), 1)} C\n"
            f"- **Data Coverage:** {_dec(temp.get('data_coverage_pct', 0))}%"
        )

    def _md_vcmi_claims(self, data: Dict[str, Any]) -> str:
        vcmi = data.get("vcmi_claims", {})
        return (
            "## 9. VCMI Claims Status\n\n"
            f"- **Tier Achieved:** {vcmi.get('tier', 'None')}\n"
            f"- **Foundational Criteria:** {vcmi.get('criteria_passed', 0)}/{vcmi.get('criteria_total', 4)}\n"
            f"- **Credit Volume Retired:** {_dec_comma(vcmi.get('credits_retired_tco2e', 0))} tCO2e\n"
            f"- **Credit Quality Score:** {_dec(vcmi.get('credit_quality_score', 0), 1)} / 10.0\n"
            f"- **Greenwashing Risks:** {vcmi.get('greenwashing_risk_count', 0)} flags"
        )

    def _md_multi_entity(self, data: Dict[str, Any]) -> str:
        me = data.get("multi_entity", {})
        entities = me.get("entities", [])
        lines = [
            "## 10. Multi-Entity View\n",
            f"**Consolidation Method:** {me.get('consolidation_method', 'Operational Control')}  \n"
            f"**Net Consolidated:** {_dec_comma(me.get('net_consolidated_tco2e', 0))} tCO2e\n",
            "| Entity | Emissions (tCO2e) | Target (tCO2e) | On Track |",
            "|--------|------------------:|---------------:|:--------:|",
        ]
        for e in entities:
            on_track = "YES" if e.get("on_track", False) else "NO"
            lines.append(
                f"| {e.get('name', '-')} "
                f"| {_dec_comma(e.get('emissions_tco2e', 0))} "
                f"| {_dec_comma(e.get('target_tco2e', 0))} "
                f"| {on_track} |"
            )
        if not entities:
            lines.append("| _No entity data_ | - | - | - |")
        return "\n".join(lines)

    def _md_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 11. Key Risks & Mitigations\n",
            "| # | Risk | Category | Likelihood | Impact | Mitigation | Owner |",
            "|---|------|----------|:----------:|:------:|------------|-------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('risk', '-')} "
                f"| {r.get('category', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {r.get('mitigation', '-')} "
                f"| {r.get('owner', '-')} |"
            )
        if not risks:
            lines.append("| - | _No risks identified_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_roadmap(self, data: Dict[str, Any]) -> str:
        roadmap = data.get("roadmap", [])
        lines = [
            "## 12. Implementation Roadmap (Next 12 Months)\n",
            "| Month | Action | Owner | Deliverable | Dependencies | Priority |",
            "|:-----:|--------|-------|-------------|:------------:|:--------:|",
        ]
        for item in roadmap:
            lines.append(
                f"| {item.get('month', '-')} | {item.get('action', '-')} "
                f"| {item.get('owner', '-')} "
                f"| {item.get('deliverable', '-')} "
                f"| {item.get('dependencies', '-')} "
                f"| {item.get('priority', '-')} |"
            )
        if not roadmap:
            lines.append("| - | _No roadmap items_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_board_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("board_recommendations", [])
        lines = ["## 13. Board Recommendations\n"]
        if recs:
            for i, rec in enumerate(recs, 1):
                lines.append(f"### {i}. {rec.get('title', 'Recommendation')}\n")
                lines.append(f"**Priority:** {rec.get('priority', 'N/A')}  ")
                lines.append(f"**Investment Required:** EUR {_dec_comma(rec.get('investment_eur', 0), 0)}  ")
                lines.append(f"**Expected Impact:** {rec.get('impact', 'N/A')}\n")
                lines.append(f"{rec.get('description', '')}\n")
                actions = rec.get("actions", [])
                if actions:
                    lines.append("**Actions:**")
                    for action in actions:
                        lines.append(f"  - {action}")
                lines.append("")
        else:
            lines.append("_No board recommendations at this time._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Board confidential - not for external distribution without approval.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".rag-green{color:#1b5e20;font-weight:700;}"
            ".rag-amber{color:#e65100;font-weight:700;}"
            ".rag-red{color:#c62828;font-weight:700;}"
            ".on-track{color:#1b5e20;font-weight:600;}"
            ".off-track{color:#c62828;font-weight:600;}"
            ".priority-high{color:#c62828;font-weight:600;}"
            ".priority-medium{color:#e65100;font-weight:600;}"
            ".priority-low{color:#1b5e20;font-weight:600;}"
            ".rec-card{margin:16px 0;padding:20px;border:1px solid #c8e6c9;"
            "border-radius:10px;border-left:4px solid #2e7d32;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Net Zero Acceleration Strategy</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts} | '
            f'<strong>Board Confidential</strong></p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        pos = data.get("current_position", {})
        sa = data.get("scenario_analysis", {})
        temp = data.get("temperature_alignment", {})
        vcmi = data.get("vcmi_claims", {})
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Current Emissions</div>'
            f'<div class="card-value">{_dec_comma(pos.get("current_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction Achieved</div>'
            f'<div class="card-value">{_dec(pos.get("reduction_achieved_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Temperature Score</div>'
            f'<div class="card-value">{_dec(temp.get("temperature_c", 0), 1)} C</div></div>\n'
            f'  <div class="card"><div class="card-label">VCMI Tier</div>'
            f'<div class="card-value">{vcmi.get("tier", "None")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Recommended Scenario</div>'
            f'<div class="card-value">{sa.get("recommended_scenario", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Net Zero Year</div>'
            f'<div class="card-value">{pos.get("net_zero_target_year", "N/A")}</div></div>\n'
            f'</div>'
        )

    def _html_current_position(self, data: Dict[str, Any]) -> str:
        pos = data.get("current_position", {})
        gap = float(Decimal(str(pos.get("gap_pct", 0))))
        gap_cls = "rag-red" if gap > 10 else "rag-amber" if gap > 0 else "rag-green"
        return (
            f'<h2>2. Current Net-Zero Position</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Year</div>'
            f'<div class="card-value">{pos.get("base_year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(pos.get("scope1_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(pos.get("scope2_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(pos.get("scope3_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Gap to Target</div>'
            f'<div class="card-value {gap_cls}">{_dec(gap)}%</div></div>\n'
            f'</div>'
        )

    def _html_scenario_analysis(self, data: Dict[str, Any]) -> str:
        sa = data.get("scenario_analysis", {})
        scenarios = sa.get("scenarios", [])
        rows = ""
        for sc in scenarios:
            rows += (
                f'<tr><td><strong>{sc.get("name", "-")}</strong></td>'
                f'<td>{sc.get("target_year", "-")}</td>'
                f'<td>{sc.get("ambition", "-")}</td>'
                f'<td>{_dec(sc.get("cost_score", 0), 1)}</td>'
                f'<td>{_dec(sc.get("risk_score", 0), 1)}</td>'
                f'<td><strong>{_dec(sc.get("total_score", 0), 1)}</strong></td></tr>\n'
            )
        return (
            f'<h2>3. Scenario Analysis Summary</h2>\n'
            f'<p><strong>Recommended:</strong> {sa.get("recommended_scenario", "N/A")} | '
            f'<strong>Score:</strong> {_dec(sa.get("decision_score", 0), 1)} / 10.0</p>\n'
            f'<table>\n'
            f'<tr><th>Scenario</th><th>Target Year</th><th>Ambition</th>'
            f'<th>Cost</th><th>Risk</th><th>Total</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sda_pathway(self, data: Dict[str, Any]) -> str:
        sda = data.get("sda_pathway", {})
        if not sda:
            return '<h2>4. SDA Pathway</h2>\n<p><em>Not applicable - cross-sector pathway selected.</em></p>'
        on_track = sda.get("on_track", False)
        cls = "on-track" if on_track else "off-track"
        return (
            f'<h2>4. SDA Pathway</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Sector</div>'
            f'<div class="card-value">{sda.get("sector", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Current Intensity</div>'
            f'<div class="card-value">{_dec(sda.get("current_intensity", 0), 4)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Target Intensity</div>'
            f'<div class="card-value">{_dec(sda.get("convergence_target", 0), 4)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value {cls}">{"On Track" if on_track else "Off Track"}</div></div>\n'
            f'</div>'
        )

    def _html_supplier_engagement(self, data: Dict[str, Any]) -> str:
        se = data.get("supplier_engagement", {})
        return (
            f'<h2>5. Supplier Engagement</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Suppliers</div>'
            f'<div class="card-value">{se.get("total_suppliers", 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 3 Coverage</div>'
            f'<div class="card-value">{_dec(se.get("scope3_coverage_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">SBTi Adoption</div>'
            f'<div class="card-value">{_dec(se.get("sbti_adoption_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction Potential</div>'
            f'<div class="card-value">{_dec_comma(se.get("reduction_potential_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_climate_investment(self, data: Dict[str, Any]) -> str:
        ci = data.get("climate_investment", {})
        return (
            f'<h2>6. Climate Investment Plan</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Climate CapEx</div>'
            f'<div class="card-value">EUR {_dec_comma(ci.get("total_capex_eur", 0), 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Taxonomy Aligned</div>'
            f'<div class="card-value">{_dec(ci.get("taxonomy_aligned_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Weighted IRR</div>'
            f'<div class="card-value">{_dec(ci.get("weighted_irr_pct", 0), 1)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Total Abatement</div>'
            f'<div class="card-value">{_dec_comma(ci.get("total_abatement_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e/yr</div></div>\n'
            f'</div>'
        )

    def _html_progress(self, data: Dict[str, Any]) -> str:
        progress = data.get("progress", {})
        kpis = progress.get("kpis", [])
        rows = ""
        for kpi in kpis:
            status = kpi.get("status", "AMBER").upper()
            cls = "rag-green" if status == "GREEN" else "rag-red" if status == "RED" else "rag-amber"
            rows += (
                f'<tr><td>{kpi.get("name", "-")}</td>'
                f'<td>{kpi.get("target", "-")}</td>'
                f'<td>{kpi.get("actual", "-")}</td>'
                f'<td>{kpi.get("variance", "-")}</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )
        return (
            f'<h2>7. Progress Against Targets</h2>\n'
            f'<table>\n'
            f'<tr><th>KPI</th><th>Target</th><th>Actual</th>'
            f'<th>Variance</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_temperature_alignment(self, data: Dict[str, Any]) -> str:
        temp = data.get("temperature_alignment", {})
        t = float(Decimal(str(temp.get("temperature_c", 0))))
        alignment = "1.5C Aligned" if t <= 1.5 else "2C Aligned" if t <= 2.0 else "Above 2C"
        cls = "rag-green" if t <= 1.5 else "rag-amber" if t <= 2.0 else "rag-red"
        return (
            f'<h2>8. Temperature Alignment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Temperature</div>'
            f'<div class="card-value {cls}">{_dec(t, 1)} C</div></div>\n'
            f'  <div class="card"><div class="card-label">Alignment</div>'
            f'<div class="card-value">{alignment}</div></div>\n'
            f'  <div class="card"><div class="card-label">Method</div>'
            f'<div class="card-value">{temp.get("method", "WATS")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Benchmark</div>'
            f'<div class="card-value">{_dec(temp.get("benchmark_c", 0), 1)} C</div></div>\n'
            f'</div>'
        )

    def _html_vcmi_claims(self, data: Dict[str, Any]) -> str:
        vcmi = data.get("vcmi_claims", {})
        return (
            f'<h2>9. VCMI Claims Status</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Tier</div>'
            f'<div class="card-value">{vcmi.get("tier", "None")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Criteria Passed</div>'
            f'<div class="card-value">{vcmi.get("criteria_passed", 0)}/{vcmi.get("criteria_total", 4)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Credits Retired</div>'
            f'<div class="card-value">{_dec_comma(vcmi.get("credits_retired_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Quality Score</div>'
            f'<div class="card-value">{_dec(vcmi.get("credit_quality_score", 0), 1)}</div>'
            f'<div class="card-unit">/ 10.0</div></div>\n'
            f'</div>'
        )

    def _html_multi_entity(self, data: Dict[str, Any]) -> str:
        me = data.get("multi_entity", {})
        entities = me.get("entities", [])
        rows = ""
        for e in entities:
            on_track = e.get("on_track", False)
            cls = "on-track" if on_track else "off-track"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f'<tr><td><strong>{e.get("name", "-")}</strong></td>'
                f'<td>{_dec_comma(e.get("emissions_tco2e", 0))}</td>'
                f'<td>{_dec_comma(e.get("target_tco2e", 0))}</td>'
                f'<td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>10. Multi-Entity View</h2>\n'
            f'<p><strong>Method:</strong> {me.get("consolidation_method", "Operational Control")} | '
            f'<strong>Net:</strong> {_dec_comma(me.get("net_consolidated_tco2e", 0))} tCO2e</p>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Emissions (tCO2e)</th><th>Target (tCO2e)</th>'
            f'<th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            impact = r.get("impact", "Medium")
            i_cls = "rag-red" if impact.lower() == "high" else "rag-green" if impact.lower() == "low" else "rag-amber"
            rows += (
                f'<tr><td>{i}</td><td>{r.get("risk", "-")}</td>'
                f'<td>{r.get("category", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td class="{i_cls}">{impact}</td>'
                f'<td>{r.get("mitigation", "-")}</td>'
                f'<td>{r.get("owner", "-")}</td></tr>\n'
            )
        return (
            f'<h2>11. Key Risks & Mitigations</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Risk</th><th>Category</th><th>Likelihood</th>'
            f'<th>Impact</th><th>Mitigation</th><th>Owner</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_roadmap(self, data: Dict[str, Any]) -> str:
        roadmap = data.get("roadmap", [])
        rows = ""
        for item in roadmap:
            priority = item.get("priority", "MEDIUM")
            p_cls = (
                "priority-high" if priority.upper() == "HIGH"
                else "priority-low" if priority.upper() == "LOW"
                else "priority-medium"
            )
            rows += (
                f'<tr><td>{item.get("month", "-")}</td>'
                f'<td>{item.get("action", "-")}</td>'
                f'<td>{item.get("owner", "-")}</td>'
                f'<td>{item.get("deliverable", "-")}</td>'
                f'<td>{item.get("dependencies", "-")}</td>'
                f'<td class="{p_cls}">{priority}</td></tr>\n'
            )
        return (
            f'<h2>12. Implementation Roadmap (Next 12 Months)</h2>\n'
            f'<table>\n'
            f'<tr><th>Month</th><th>Action</th><th>Owner</th>'
            f'<th>Deliverable</th><th>Dependencies</th><th>Priority</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_board_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("board_recommendations", [])
        items = ""
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "MEDIUM")
            p_cls = "priority-high" if priority.upper() == "HIGH" else "priority-low" if priority.upper() == "LOW" else "priority-medium"
            actions_html = ""
            if rec.get("actions"):
                actions_html = "<ul>" + "".join(f"<li>{a}</li>" for a in rec["actions"]) + "</ul>"
            items += (
                f'<div class="rec-card">'
                f'<h3>{i}. {rec.get("title", "Recommendation")}</h3>'
                f'<p><span class="{p_cls}"><strong>Priority:</strong> {priority}</span> | '
                f'<strong>Investment:</strong> EUR {_dec_comma(rec.get("investment_eur", 0), 0)} | '
                f'<strong>Impact:</strong> {rec.get("impact", "N/A")}</p>'
                f'<p>{rec.get("description", "")}</p>'
                f'{actions_html}</div>\n'
            )
        return f'<h2>13. Board Recommendations</h2>\n{items}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Board confidential - not for external distribution.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
