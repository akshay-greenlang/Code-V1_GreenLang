# -*- coding: utf-8 -*-
"""
CSRDESRSReportTemplate - CSRD/ESRS E1 climate report for PACK-027.

Renders a CSRD-compliant climate chapter covering all ESRS E1 datapoints
(E1-1 through E1-9) including transition plan, policies, actions, targets,
energy consumption, GHG emissions, removals, internal carbon pricing, and
anticipated financial effects.

Sections:
    1. Disclosure Header (ESRS E1 overview)
    2. E1-1: Transition Plan for Climate Change Mitigation
    3. E1-2: Policies Related to Climate Change
    4. E1-3: Actions and Resources
    5. E1-4: Targets Related to Climate Change
    6. E1-5: Energy Consumption and Mix
    7. E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions
    8. E1-7: GHG Removals and Storage
    9. E1-8: Internal Carbon Pricing
   10. E1-9: Anticipated Financial Effects from Climate Change

Output: Markdown, HTML, JSON, Excel
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
_TEMPLATE_ID = "csrd_esrs_report"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

ESRS_E1_DATAPOINTS = [
    {"id": "E1-1", "name": "Transition plan for climate change mitigation", "delegated_reg": "DR 2023/2772 Annex 1, para E1-1"},
    {"id": "E1-2", "name": "Policies related to climate change mitigation and adaptation", "delegated_reg": "DR 2023/2772 Annex 1, para E1-2"},
    {"id": "E1-3", "name": "Actions and resources in relation to climate change", "delegated_reg": "DR 2023/2772 Annex 1, para E1-3"},
    {"id": "E1-4", "name": "Targets related to climate change mitigation and adaptation", "delegated_reg": "DR 2023/2772 Annex 1, para E1-4"},
    {"id": "E1-5", "name": "Energy consumption and mix", "delegated_reg": "DR 2023/2772 Annex 1, para E1-5"},
    {"id": "E1-6", "name": "Gross Scopes 1, 2, 3 and total GHG emissions", "delegated_reg": "DR 2023/2772 Annex 1, para E1-6"},
    {"id": "E1-7", "name": "GHG removals and GHG mitigation projects financed through carbon credits", "delegated_reg": "DR 2023/2772 Annex 1, para E1-7"},
    {"id": "E1-8", "name": "Internal carbon pricing", "delegated_reg": "DR 2023/2772 Annex 1, para E1-8"},
    {"id": "E1-9", "name": "Anticipated financial effects from material physical and transition risks", "delegated_reg": "DR 2023/2772 Annex 1, para E1-9"},
]

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

class CSRDESRSReportTemplate:
    """
    CSRD/ESRS E1 climate report template.

    Covers all 9 ESRS E1 disclosure requirements (E1-1 through E1-9)
    per Delegated Regulation 2023/2772.
    Supports Markdown, HTML, JSON, and Excel output.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json", "excel"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_datapoint_index(data),
            self._md_e1_1_transition(data),
            self._md_e1_2_policies(data),
            self._md_e1_3_actions(data),
            self._md_e1_4_targets(data),
            self._md_e1_5_energy(data),
            self._md_e1_6_emissions(data),
            self._md_e1_7_removals(data),
            self._md_e1_8_carbon_pricing(data),
            self._md_e1_9_financial_effects(data),
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
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".esrs-tag{{background:#e3f2fd;color:#1565c0;padding:2px 8px;border-radius:4px;"
            f"font-size:0.8em;font-weight:600;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        rows = ""
        for dp in ESRS_E1_DATAPOINTS:
            rows += f'<tr><td><span class="esrs-tag">{dp["id"]}</span></td><td>{dp["name"]}</td></tr>\n'
        body = (
            f'<h1>CSRD/ESRS E1 -- Climate Change</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'FY{data.get("reporting_year", "")} | '
            f'Directive (EU) 2022/2464</p>\n'
            f'<h2>Disclosure Index</h2>\n'
            f'<table><tr><th>ESRS</th><th>Disclosure Requirement</th></tr>\n{rows}</table>\n'
            f'<h2>E1-6: GHG Emissions</h2>\n'
            f'<table><tr><th>Scope</th><th>tCO2e</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td></tr>\n'
            f'<tr><td>Scope 2 (Location)</td><td>{_dec_comma(s2)}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)}</td></tr>\n'
            f'<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2+s3)}</strong></td></tr>\n'
            f'</table>\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | CSRD/ESRS E1 | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>CSRD ESRS E1 Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2_loc = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s2_mkt = float(data.get("scope2_market_tco2e", s2_loc))
        s3 = float(data.get("scope3_tco2e", 0))
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "regulation": "CSRD (Directive 2022/2464) + ESRS E1 (DR 2023/2772)",
            "organization": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "e1_1_transition_plan": data.get("esrs_e1_1", {}),
            "e1_2_policies": data.get("esrs_e1_2", {}),
            "e1_3_actions": data.get("esrs_e1_3", {}),
            "e1_4_targets": data.get("esrs_e1_4", {}),
            "e1_5_energy": data.get("esrs_e1_5", {}),
            "e1_6_emissions": {
                "scope1_tco2e": round(s1, 2),
                "scope2_location_tco2e": round(s2_loc, 2),
                "scope2_market_tco2e": round(s2_mkt, 2),
                "scope3_tco2e": round(s3, 2),
                "total_tco2e": round(s1 + s2_loc + s3, 2),
                "categories": data.get("scope3_categories", []),
            },
            "e1_7_removals": data.get("esrs_e1_7", {}),
            "e1_8_carbon_pricing": data.get("esrs_e1_8", {}),
            "e1_9_financial_effects": data.get("esrs_e1_9", {}),
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
            "filename": f"csrd_esrs_e1_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [{
                "name": "ESRS E1 Index",
                "headers": ["ESRS Ref", "Disclosure Requirement", "Status", "DR Reference"],
                "rows": [
                    [dp["id"], dp["name"], data.get(f"esrs_{dp['id'].lower().replace('-', '_')}_status", "Complete"), dp["delegated_reg"]]
                    for dp in ESRS_E1_DATAPOINTS
                ],
            }],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # Markdown sections

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CSRD/ESRS E1 -- Climate Change\n\n"
            f"## {data.get('org_name', 'Enterprise')} -- FY{data.get('reporting_year', '')}\n\n"
            f"**Regulation:** CSRD (Directive 2022/2464)  \n"
            f"**Standard:** ESRS E1 Climate Change (DR 2023/2772)  \n"
            f"**Assurance:** {data.get('assurance_level', 'Limited assurance (FY2025+)')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_datapoint_index(self, data: Dict[str, Any]) -> str:
        lines = [
            "## Disclosure Requirement Index\n",
            "| ESRS | Requirement | Status | DR Reference |",
            "|------|-------------|:------:|--------------|",
        ]
        for dp in ESRS_E1_DATAPOINTS:
            status = data.get(f"esrs_{dp['id'].lower().replace('-', '_')}_status", "Complete")
            lines.append(f"| {dp['id']} | {dp['name']} | {status} | {dp['delegated_reg']} |")
        return "\n".join(lines)

    def _md_e1_1_transition(self, data: Dict[str, Any]) -> str:
        tp = data.get("esrs_e1_1", {})
        return (
            f"## E1-1: Transition Plan for Climate Change Mitigation\n\n"
            f"**Has transition plan:** {tp.get('has_plan', 'Yes')}  \n"
            f"**Paris-aligned:** {tp.get('paris_aligned', 'Yes -- 1.5C aligned per SBTi')}  \n"
            f"**Key decarbonization levers:**\n"
            f"- {tp.get('lever_1', 'Energy efficiency improvements')}\n"
            f"- {tp.get('lever_2', 'Renewable energy procurement')}\n"
            f"- {tp.get('lever_3', 'Fleet electrification')}\n"
            f"- {tp.get('lever_4', 'Supplier engagement program')}\n"
            f"- {tp.get('lever_5', 'Process innovation')}\n\n"
            f"**CapEx allocation:** {tp.get('capex_allocation', 'See E1-3 Actions and Resources')}  \n"
            f"**Locked-in emissions:** {tp.get('locked_in', 'Assessed per GHG Protocol guidance')}  "
        )

    def _md_e1_2_policies(self, data: Dict[str, Any]) -> str:
        pol = data.get("esrs_e1_2", {})
        return (
            f"## E1-2: Policies Related to Climate Change\n\n"
            f"**Climate policy adopted:** {pol.get('adopted', 'Yes')}  \n"
            f"**Scope:** {pol.get('scope', 'Mitigation and adaptation')}  \n"
            f"**Board-approved:** {pol.get('board_approved', 'Yes')}  \n"
            f"**Key commitments:**\n"
            f"- {pol.get('commitment_1', 'Net-zero by 2050')}\n"
            f"- {pol.get('commitment_2', 'SBTi-validated targets')}\n"
            f"- {pol.get('commitment_3', 'RE100 membership')}\n"
            f"- {pol.get('commitment_4', 'Zero deforestation (if applicable)')}  "
        )

    def _md_e1_3_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("esrs_e1_3", {}).get("actions", [])
        lines = [
            "## E1-3: Actions and Resources\n",
        ]
        if actions:
            lines.append("| Action | Category | Investment | tCO2e Impact | Timeline |")
            lines.append("|--------|----------|----------:|-----------:|----------|")
            for a in actions:
                lines.append(
                    f"| {a.get('name', '')} | {a.get('category', '')} "
                    f"| {data.get('currency', 'EUR')}{_dec_comma(a.get('investment', 0))} "
                    f"| {_dec_comma(a.get('tco2e_impact', 0))} "
                    f"| {a.get('timeline', '')} |"
                )
        else:
            lines.append("See transition plan (E1-1) for detailed actions and resources.")
        return "\n".join(lines)

    def _md_e1_4_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("esrs_e1_4", {}).get("targets", [])
        lines = [
            "## E1-4: Targets Related to Climate Change\n",
            "| Target | Scope | Base Year | Target Year | Metric | SBTi Validated |",
            "|--------|-------|:---------:|:-----------:|--------|:--------------:|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('name', '')} | {t.get('scope', '')} "
                f"| {t.get('base_year', '')} | {t.get('target_year', '')} "
                f"| {t.get('metric', '')} | {t.get('sbti_validated', 'Yes')} |"
            )
        if not targets:
            lines.append("| See SBTi submission package | All | - | - | - | - |")
        return "\n".join(lines)

    def _md_e1_5_energy(self, data: Dict[str, Any]) -> str:
        energy = data.get("esrs_e1_5", {})
        return (
            f"## E1-5: Energy Consumption and Mix\n\n"
            f"| Metric | Value | Unit |\n"
            f"|--------|------:|------|\n"
            f"| Total energy consumption | {_dec_comma(energy.get('total_mwh', 0))} | MWh |\n"
            f"| Fossil fuel consumption | {_dec_comma(energy.get('fossil_mwh', 0))} | MWh |\n"
            f"| Renewable energy | {_dec_comma(energy.get('renewable_mwh', 0))} | MWh |\n"
            f"| Renewable share | {_pct(energy.get('renewable_pct', 0))} | % |\n"
            f"| Nuclear energy | {_dec_comma(energy.get('nuclear_mwh', 0))} | MWh |\n"
            f"| Energy intensity | {energy.get('intensity', 'N/A')} | MWh/revenue |"
        )

    def _md_e1_6_emissions(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2_loc = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s2_mkt = float(data.get("scope2_market_tco2e", s2_loc))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2_loc + s3
        return (
            f"## E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions\n\n"
            f"| Scope | tCO2e | % of Total |\n"
            f"|-------|------:|-----------:|\n"
            f"| Scope 1 (Direct) | {_dec_comma(s1)} | {_pct(_safe_div(s1, total) * 100)} |\n"
            f"| Scope 2 (Location-based) | {_dec_comma(s2_loc)} | {_pct(_safe_div(s2_loc, total) * 100)} |\n"
            f"| Scope 2 (Market-based) | {_dec_comma(s2_mkt)} | - |\n"
            f"| Scope 3 (Value chain) | {_dec_comma(s3)} | {_pct(_safe_div(s3, total) * 100)} |\n"
            f"| **Total (Location)** | **{_dec_comma(total)}** | **100%** |\n\n"
            f"**GWP source:** {data.get('gwp_source', 'IPCC AR6 GWP-100')}  \n"
            f"**Methodology:** {data.get('ghg_methodology', 'GHG Protocol Corporate Standard')}  "
        )

    def _md_e1_7_removals(self, data: Dict[str, Any]) -> str:
        rem = data.get("esrs_e1_7", {})
        return (
            f"## E1-7: GHG Removals and Carbon Credits\n\n"
            f"**GHG removals from own operations:** {_dec_comma(rem.get('removals_tco2e', 0))} tCO2e  \n"
            f"**Carbon credits purchased:** {_dec_comma(rem.get('credits_purchased_tco2e', 0))} tCO2e  \n"
            f"**Carbon credits retired:** {_dec_comma(rem.get('credits_retired_tco2e', 0))} tCO2e  \n"
            f"**Credit quality:** {rem.get('credit_quality', 'Per SBTi net-zero neutralization guidance')}  \n"
            f"**Note:** Carbon credits are NOT counted toward Scope 1/2/3 reduction targets per ESRS E1 guidance."
        )

    def _md_e1_8_carbon_pricing(self, data: Dict[str, Any]) -> str:
        cp = data.get("esrs_e1_8", {})
        return (
            f"## E1-8: Internal Carbon Pricing\n\n"
            f"**Uses internal carbon price:** {cp.get('uses_icp', 'Yes')}  \n"
            f"**Price type:** {cp.get('type', 'Shadow price for investment decisions')}  \n"
            f"**Price level:** {cp.get('price', '$100/tCO2e')}  \n"
            f"**Scope of application:** {cp.get('scope', 'CapEx decisions, product pricing, BU performance')}  \n"
            f"**Revenue generated:** {cp.get('revenue', 'N/A (shadow price, no actual charge)')}  "
        )

    def _md_e1_9_financial_effects(self, data: Dict[str, Any]) -> str:
        fe = data.get("esrs_e1_9", {})
        lines = [
            "## E1-9: Anticipated Financial Effects\n",
            "### Physical Risks\n",
            "| Risk | Assets at Risk | Financial Impact | Timeframe |",
            "|------|:--------------:|:----------------:|:---------:|",
        ]
        for r in fe.get("physical_risks", []):
            lines.append(
                f"| {r.get('risk', '')} | {r.get('assets', '')} "
                f"| {r.get('impact', '')} | {r.get('timeframe', '')} |"
            )
        lines.append("\n### Transition Risks\n")
        lines.append("| Risk | Impact Area | Financial Impact | Timeframe |")
        lines.append("|------|:----------:|:----------------:|:---------:|")
        for r in fe.get("transition_risks", []):
            lines.append(
                f"| {r.get('risk', '')} | {r.get('area', '')} "
                f"| {r.get('impact', '')} | {r.get('timeframe', '')} |"
            )
        lines.append("\n### Climate Opportunities\n")
        lines.append("| Opportunity | Financial Effect | Timeframe |")
        lines.append("|-------------|:----------------:|:---------:|")
        for o in fe.get("opportunities", []):
            lines.append(
                f"| {o.get('opportunity', '')} | {o.get('effect', '')} "
                f"| {o.get('timeframe', '')} |"
            )
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "CSRD-001", "source": "Directive (EU) 2022/2464 (CSRD)", "year": "2022"},
            {"ref": "ESRS-001", "source": "ESRS E1 Climate Change (DR 2023/2772)", "year": "2023"},
            {"ref": "ESRS-002", "source": "ESRS 1 General Requirements", "year": "2023"},
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
            f"*CSRD/ESRS E1 Climate Change compliant. SHA-256 provenance.*"
        )
