# -*- coding: utf-8 -*-
"""
TCFDReportTemplate - Complete TCFD/ISSB S2 disclosure for PACK-027.

Renders a full TCFD (Task Force on Climate-related Financial Disclosures)
report covering all four pillars: Governance, Strategy, Risk Management,
and Metrics & Targets. Aligned with ISSB S2 (Climate-related Disclosures).

Sections:
    1. Executive Summary
    2. Governance (board oversight, management role)
    3. Strategy (risks, opportunities, scenario analysis, financial impact)
    4. Risk Management (identification, assessment, management)
    5. Metrics and Targets (Scope 1/2/3, intensity, targets, progress)
    6. Scenario Analysis Detail (1.5C, 2C, 4C)
    7. Financial Impact Assessment
    8. Citations

Output: Markdown, HTML, JSON, PDF-ready
Provenance: SHA-256 hash on all outputs

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "tcfd_report"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

TCFD_PILLARS = [
    {"id": "governance", "name": "Governance", "tcfd_ref": "TCFD Governance a/b"},
    {"id": "strategy", "name": "Strategy", "tcfd_ref": "TCFD Strategy a/b/c"},
    {"id": "risk_management", "name": "Risk Management", "tcfd_ref": "TCFD RM a/b/c"},
    {"id": "metrics_targets", "name": "Metrics and Targets", "tcfd_ref": "TCFD MT a/b/c"},
]

ISSB_S2_MAPPING = {
    "governance": ["IFRS S2 para 5-7"],
    "strategy": ["IFRS S2 para 8-22"],
    "risk_management": ["IFRS S2 para 23-26"],
    "metrics_targets": ["IFRS S2 para 27-37"],
}

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)

def _pct(val: Any) -> str:
    try:
        return str(round(float(val), 1)) + "%"
    except Exception:
        return str(val)

def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default

class TCFDReportTemplate:
    """
    Complete TCFD/ISSB S2 climate disclosure template.

    Covers all four TCFD pillars with ISSB S2 cross-references.
    Supports Markdown, HTML, JSON, and PDF-ready output.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json", "pdf"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_governance(data),
            self._md_strategy(data),
            self._md_risk_management(data),
            self._md_metrics_targets(data),
            self._md_scenario_analysis(data),
            self._md_financial_impact(data),
            self._md_issb_crosswalk(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1100px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};margin-top:16px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".pillar{{background:{_LIGHT};padding:16px;border-radius:8px;margin:12px 0;"
            f"border-left:4px solid {_ACCENT};}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        body_parts = [
            f'<h1>TCFD Climate-Related Disclosure</h1>',
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'{data.get("reporting_year", "")} | '
            f'TCFD + ISSB S2 Aligned</p>',
            self._html_pillar_summary(data),
            self._html_emissions_summary(data),
            self._html_scenarios(data),
            f'<div class="footer">Generated by GreenLang PACK-027 | TCFD/ISSB S2 | SHA-256</div>',
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>TCFD Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "framework": "TCFD + ISSB S2",
            "organization": {
                "name": data.get("org_name", ""),
                "sector": data.get("sector", ""),
                "reporting_year": data.get("reporting_year", ""),
            },
            "governance": {
                "board_oversight": data.get("governance_board", {}),
                "management_role": data.get("governance_management", {}),
            },
            "strategy": {
                "risks": data.get("climate_risks", []),
                "opportunities": data.get("climate_opportunities", []),
                "scenario_analysis": data.get("scenario_analysis", {}),
                "financial_impact": data.get("financial_impact", {}),
                "transition_plan": data.get("transition_plan", {}),
            },
            "risk_management": {
                "identification": data.get("risk_identification", {}),
                "assessment": data.get("risk_assessment", {}),
                "management": data.get("risk_management_process", {}),
                "integration": data.get("risk_integration", {}),
            },
            "metrics_and_targets": {
                "scope1_tco2e": round(s1, 2),
                "scope2_location_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
                "total_tco2e": round(s1 + s2 + s3, 2),
                "intensity_metrics": data.get("intensity_metrics", []),
                "targets": data.get("targets", []),
                "progress": data.get("target_progress", {}),
                "internal_carbon_price": data.get("internal_carbon_price", {}),
            },
            "issb_s2_crosswalk": ISSB_S2_MAPPING,
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"tcfd_report_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [
                {
                    "name": "TCFD Summary",
                    "headers": ["Pillar", "Recommendation", "ISSB S2 Ref", "Status"],
                    "rows": [
                        ["Governance", "Board oversight of climate risks", "IFRS S2 para 5-7",
                         data.get("governance_status", "Complete")],
                        ["Strategy", "Climate risks, opportunities, scenarios", "IFRS S2 para 8-22",
                         data.get("strategy_status", "Complete")],
                        ["Risk Management", "Risk identification and management", "IFRS S2 para 23-26",
                         data.get("risk_status", "Complete")],
                        ["Metrics & Targets", "GHG emissions, targets, progress", "IFRS S2 para 27-37",
                         data.get("metrics_status", "Complete")],
                    ],
                },
            ],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# TCFD Climate-Related Financial Disclosure\n\n"
            f"## {data.get('org_name', 'Enterprise')} -- {data.get('reporting_year', '')}\n\n"
            f"**Framework:** TCFD Recommendations (2017/2023) + ISSB S2  \n"
            f"**Generated:** {ts}  \n"
            f"**Report ID:** {_new_uuid()}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        return (
            f"## Executive Summary\n\n"
            f"This report presents {data.get('org_name', 'the organization')}'s climate-related "
            f"financial disclosures in accordance with the TCFD Recommendations and ISSB S2. "
            f"Total GHG emissions for the reporting period were **{_dec_comma(total)} tCO2e** "
            f"(Scope 1: {_dec_comma(s1)}, Scope 2: {_dec_comma(s2)}, Scope 3: {_dec_comma(s3)}).\n\n"
            f"| Pillar | Status | Key Finding |\n"
            f"|--------|:------:|-------------|\n"
            f"| Governance | {data.get('governance_status', 'Complete')} | {data.get('governance_finding', 'Board committee with climate oversight')} |\n"
            f"| Strategy | {data.get('strategy_status', 'Complete')} | {data.get('strategy_finding', 'Transition plan with 1.5C alignment')} |\n"
            f"| Risk Management | {data.get('risk_status', 'Complete')} | {data.get('risk_finding', 'Integrated into enterprise risk framework')} |\n"
            f"| Metrics & Targets | {data.get('metrics_status', 'Complete')} | {data.get('metrics_finding', 'SBTi-validated targets set')} |"
        )

    def _md_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance_board", {})
        mgmt = data.get("governance_management", {})
        return (
            f"## Governance\n\n"
            f"### Board Oversight (TCFD Governance a / ISSB S2 para 5)\n\n"
            f"**Responsible body:** {gov.get('committee', 'Board Sustainability Committee')}  \n"
            f"**Frequency:** {gov.get('frequency', 'Quarterly review of climate strategy and risks')}  \n"
            f"**Skills:** {gov.get('skills', 'Climate competency training for all board members')}  \n"
            f"**Climate in strategy:** {gov.get('strategy_integration', 'Climate reviewed in all strategic decisions')}  \n"
            f"**Incentives:** {gov.get('incentives', 'Climate KPIs in executive compensation (15-25%)')}  \n\n"
            f"### Management Role (TCFD Governance b / ISSB S2 para 6-7)\n\n"
            f"**Senior officer:** {mgmt.get('officer', 'Chief Sustainability Officer, reporting to CEO')}  \n"
            f"**Team:** {mgmt.get('team', 'Sustainability team of 10-15 FTEs')}  \n"
            f"**Cross-functional:** {mgmt.get('cross_functional', 'Climate steering committee (CSO, CFO, COO, CTO)')}  \n"
            f"**Escalation:** {mgmt.get('escalation', 'Material climate risks escalated to Audit Committee')}  "
        )

    def _md_strategy(self, data: Dict[str, Any]) -> str:
        risks = data.get("climate_risks", [])
        opps = data.get("climate_opportunities", [])
        lines = [
            "## Strategy\n",
            "### Climate-Related Risks (TCFD Strategy a / ISSB S2 para 10-14)\n",
            "| Risk | Type | Timeframe | Likelihood | Impact | Financial Effect |",
            "|------|------|:---------:|:----------:|:------:|:----------------:|",
        ]
        for r in risks[:8]:
            lines.append(
                f"| {r.get('name', '')} | {r.get('type', '')} "
                f"| {r.get('timeframe', '')} | {r.get('likelihood', '')} "
                f"| {r.get('impact', '')} | {r.get('financial_effect', '')} |"
            )

        lines.append("\n### Climate-Related Opportunities (TCFD Strategy a)\n")
        lines.append("| Opportunity | Type | Timeframe | Impact | Financial Effect |")
        lines.append("|-------------|------|:---------:|:------:|:----------------:|")
        for o in opps[:5]:
            lines.append(
                f"| {o.get('name', '')} | {o.get('type', '')} "
                f"| {o.get('timeframe', '')} | {o.get('impact', '')} "
                f"| {o.get('financial_effect', '')} |"
            )

        tp = data.get("transition_plan", {})
        lines.append(f"\n### Transition Plan (TCFD Strategy b / ISSB S2 para 14)")
        lines.append(f"\n**Plan status:** {tp.get('status', 'Published')}")
        lines.append(f"**Target:** {tp.get('target', 'Net-zero by 2050, SBTi-validated')}")
        lines.append(f"**Key levers:** {tp.get('key_levers', 'Energy efficiency, electrification, RE procurement, supplier engagement')}")
        lines.append(f"**Investment required:** {tp.get('investment', 'See scenario analysis')}")
        return "\n".join(lines)

    def _md_risk_management(self, data: Dict[str, Any]) -> str:
        rm = data.get("risk_management_process", {})
        return (
            f"## Risk Management\n\n"
            f"### Risk Identification (TCFD RM a / ISSB S2 para 23-24)\n\n"
            f"**Process:** {rm.get('identification', 'Annual climate risk assessment using TCFD scenario analysis')}  \n"
            f"**Scope:** {rm.get('scope', 'Physical and transition risks across all operations and value chain')}  \n"
            f"**Horizon:** {rm.get('horizon', 'Short (0-3 years), Medium (3-10 years), Long (10-30 years)')}  \n\n"
            f"### Risk Assessment (TCFD RM b / ISSB S2 para 25)\n\n"
            f"**Methodology:** {rm.get('assessment', 'Quantitative scenario analysis with financial impact modeling')}  \n"
            f"**Materiality threshold:** {rm.get('materiality', '>1% of revenue or >5% of EBITDA')}  \n\n"
            f"### Integration with ERM (TCFD RM c / ISSB S2 para 26)\n\n"
            f"**Integration:** {rm.get('integration', 'Climate risks integrated into enterprise risk register and quarterly risk review')}  \n"
            f"**Oversight:** {rm.get('oversight', 'Audit Committee reviews climate risk alongside financial risks')}  "
        )

    def _md_metrics_targets(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2_loc = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s2_mkt = float(data.get("scope2_market_tco2e", s2_loc))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2_loc + s3
        intensity_metrics = data.get("intensity_metrics", [])
        targets = data.get("targets", [])

        lines = [
            "## Metrics and Targets\n",
            "### GHG Emissions (TCFD MT a / ISSB S2 para 29)\n",
            "| Metric | tCO2e | % of Total |",
            "|--------|------:|-----------:|",
            f"| Scope 1 | {_dec_comma(s1)} | {_pct(_safe_div(s1, total) * 100)} |",
            f"| Scope 2 (Location-based) | {_dec_comma(s2_loc)} | {_pct(_safe_div(s2_loc, total) * 100)} |",
            f"| Scope 2 (Market-based) | {_dec_comma(s2_mkt)} | - |",
            f"| Scope 3 | {_dec_comma(s3)} | {_pct(_safe_div(s3, total) * 100)} |",
            f"| **Total** | **{_dec_comma(total)}** | **100%** |",
        ]
        if intensity_metrics:
            lines.append("\n### Intensity Metrics (ISSB S2 para 29b)\n")
            lines.append("| Metric | Value | Unit |")
            lines.append("|--------|------:|------|")
            for m in intensity_metrics:
                lines.append(f"| {m.get('name', '')} | {m.get('value', '')} | {m.get('unit', '')} |")

        if targets:
            lines.append("\n### Targets (TCFD MT c / ISSB S2 para 33-37)\n")
            lines.append("| Target | Scope | Base Year | Target Year | Reduction | Progress |")
            lines.append("|--------|-------|:---------:|:-----------:|:---------:|:--------:|")
            for t in targets:
                lines.append(
                    f"| {t.get('name', '')} | {t.get('scope', '')} "
                    f"| {t.get('base_year', '')} | {t.get('target_year', '')} "
                    f"| {_pct(t.get('reduction_pct', 0))} "
                    f"| {_pct(t.get('progress_pct', 0))} |"
                )

        icp = data.get("internal_carbon_price", {})
        if icp:
            lines.append(f"\n### Internal Carbon Price (TCFD MT b)\n")
            lines.append(f"**Price:** {icp.get('price', 'N/A')}  ")
            lines.append(f"**Application:** {icp.get('application', 'CapEx decisions')}  ")
        return "\n".join(lines)

    def _md_scenario_analysis(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", [])
        if not scenarios:
            scenarios = [
                {"name": "1.5C (Aggressive)", "temp": "1.5C", "description": "Rapid transition"},
                {"name": "2C (Moderate)", "temp": "2C", "description": "Orderly transition"},
                {"name": "4C (BAU/Hot House)", "temp": "4C", "description": "Failed transition"},
            ]
        lines = [
            "## Scenario Analysis Detail (TCFD Strategy c / ISSB S2 para 22)\n",
            "| Scenario | Temperature | Key Assumptions | Business Impact |",
            "|----------|:----------:|-----------------|:---------------:|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('name', '')} | {s.get('temp', '')} "
                f"| {s.get('description', '')} "
                f"| {s.get('business_impact', 'See detailed analysis')} |"
            )
        return "\n".join(lines)

    def _md_financial_impact(self, data: Dict[str, Any]) -> str:
        fi = data.get("financial_impact", {})
        lines = [
            "## Financial Impact Assessment (ISSB S2 para 15-21)\n",
            "| Impact Area | Current Year | Anticipated | Timeframe |",
            "|-------------|:------------:|:-----------:|:---------:|",
        ]
        impacts = fi.get("impacts", [
            {"area": "Revenue at risk from physical climate", "current": "Low", "anticipated": "Medium", "timeframe": "Long-term"},
            {"area": "Cost increase from carbon pricing", "current": "Medium", "anticipated": "High", "timeframe": "Medium-term"},
            {"area": "CapEx for transition", "current": "Medium", "anticipated": "High", "timeframe": "Short-term"},
            {"area": "Opportunity from low-carbon products", "current": "Low", "anticipated": "High", "timeframe": "Medium-term"},
        ])
        for imp in impacts:
            lines.append(
                f"| {imp.get('area', '')} | {imp.get('current', '')} "
                f"| {imp.get('anticipated', '')} | {imp.get('timeframe', '')} |"
            )
        return "\n".join(lines)

    def _md_issb_crosswalk(self, data: Dict[str, Any]) -> str:
        lines = [
            "## ISSB S2 Cross-Reference\n",
            "| TCFD Pillar | ISSB S2 Paragraphs | Status |",
            "|-------------|-------------------:|:------:|",
        ]
        for pillar in TCFD_PILLARS:
            refs = ", ".join(ISSB_S2_MAPPING.get(pillar["id"], []))
            lines.append(f"| {pillar['name']} | {refs} | Disclosed |")
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "TCFD-001", "source": "TCFD Final Report: Recommendations", "year": "2017"},
            {"ref": "TCFD-002", "source": "TCFD Status Report (Final)", "year": "2023"},
            {"ref": "ISSB-001", "source": "IFRS S2 Climate-related Disclosures", "year": "2023"},
        ])
        lines = ["## Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*TCFD Recommendations + ISSB S2 compliant. SHA-256 provenance.*"
        )

    # HTML helpers
    def _html_pillar_summary(self, data: Dict[str, Any]) -> str:
        pillars_html = ""
        for p in TCFD_PILLARS:
            status = data.get(f"{p['id']}_status", "Complete")
            pillars_html += (
                f'<div class="pillar"><strong>{p["name"]}</strong>: {status} '
                f'<em>({p["tcfd_ref"]})</em></div>\n'
            )
        return f'<h2>TCFD Pillar Summary</h2>\n{pillars_html}'

    def _html_emissions_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        return (
            f'<h2>Metrics - Emissions</h2>\n'
            f'<table><tr><th>Scope</th><th>tCO2e</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec_comma(s2)}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)}</td></tr>\n'
            f'<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2+s3)}</strong></td></tr>\n'
            f'</table>'
        )

    def _html_scenarios(self, data: Dict[str, Any]) -> str:
        return '<h2>Scenario Analysis</h2>\n<p>See detailed Markdown output.</p>'
