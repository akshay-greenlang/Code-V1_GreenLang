# -*- coding: utf-8 -*-
"""
BoardClimateReportTemplate - Quarterly board climate report for PACK-027.

Renders a 5-10 page quarterly board paper covering emission performance
vs target pathway, key initiatives status, regulatory compliance update,
carbon pricing impact, supply chain engagement progress, upcoming
milestones, and risk assessment.

Sections:
    1. Board Paper Header (meeting date, paper #, classification)
    2. Executive Summary (1 paragraph + key metrics table)
    3. Emission Performance vs Target
    4. Key Initiatives Status (energy, fleet, procurement, digital)
    5. Regulatory Compliance Update
    6. Carbon Pricing Impact
    7. Supply Chain Engagement Progress
    8. Upcoming Milestones (next 90 days)
    9. Risk Assessment Update
   10. Decisions Required
   11. Appendix (supporting data)

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
_TEMPLATE_ID = "board_climate_report"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

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

def _rag_md(status: str) -> str:
    s = status.lower()
    if s == "green":
        return "**[GREEN]**"
    elif s == "amber":
        return "**[AMBER]**"
    elif s == "red":
        return "**[RED]**"
    return f"**[{status.upper()}]**"

class BoardClimateReportTemplate:
    """
    Quarterly board climate report template.

    5-10 page board paper covering emission performance, initiatives,
    regulatory compliance, carbon pricing, and supply chain engagement.
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
            self._md_board_header(data),
            self._md_executive_summary(data),
            self._md_emission_performance(data),
            self._md_initiatives(data),
            self._md_regulatory_update(data),
            self._md_carbon_pricing(data),
            self._md_supply_chain(data),
            self._md_milestones(data),
            self._md_risk_assessment(data),
            self._md_decisions_required(data),
            self._md_appendix(data),
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
            f".report{{max-width:1000px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".decision{{background:#fff3e0;padding:16px;border-radius:8px;margin:12px 0;"
            f"border-left:4px solid #ef6c00;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        body = (
            f'<h1>Board Climate Report</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'Q{data.get("quarter", "?")} {data.get("reporting_year", "")} | '
            f'Paper #{data.get("paper_number", "N/A")} | '
            f'{data.get("classification", "CONFIDENTIAL")}</p>\n'
            f'<h2>Performance Summary</h2>\n'
            f'{self._html_summary_table(data)}\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | Board Report | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>Board Climate Report</title>\n'
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
            "organization": data.get("org_name", ""),
            "quarter": data.get("quarter", ""),
            "reporting_year": data.get("reporting_year", ""),
            "paper_number": data.get("paper_number", ""),
            "classification": data.get("classification", "CONFIDENTIAL"),
            "emissions": {
                "scope1_tco2e": round(s1, 2),
                "scope2_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
                "total_tco2e": round(s1 + s2 + s3, 2),
                "ytd_actual": data.get("ytd_actual_tco2e", 0),
                "ytd_target": data.get("ytd_target_tco2e", 0),
                "on_track": data.get("on_track", True),
            },
            "initiatives": data.get("initiatives", []),
            "regulatory_compliance": data.get("regulatory_compliance", []),
            "carbon_pricing": data.get("carbon_pricing_impact", {}),
            "supply_chain": data.get("supply_chain_progress", {}),
            "milestones_next_90_days": data.get("milestones", []),
            "risks": data.get("climate_risks", []),
            "decisions_required": data.get("decisions_required", []),
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
            "filename": f"board_climate_Q{data.get('quarter', '')}{data.get('reporting_year', '')}.xlsx",
            "worksheets": [{
                "name": "Board Report",
                "headers": ["Section", "Key Metric", "Value", "Status"],
                "rows": [
                    ["Emissions", "Total tCO2e", data.get("scope1_tco2e", 0) + data.get("scope2_tco2e", 0) + data.get("scope3_tco2e", 0), ""],
                    ["Target", "On Track", "Yes" if data.get("on_track", True) else "No", ""],
                ],
            }],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_board_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# BOARD PAPER: Climate & Net Zero Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Meeting Date:** {data.get('meeting_date', ts)}  \n"
            f"**Paper #:** {data.get('paper_number', 'N/A')}  \n"
            f"**Quarter:** Q{data.get('quarter', '?')} {data.get('reporting_year', '')}  \n"
            f"**Classification:** {data.get('classification', 'CONFIDENTIAL')}  \n"
            f"**Author:** {data.get('author', 'Chief Sustainability Officer')}  \n"
            f"**Purpose:** {data.get('purpose', 'For Information and Decision')}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        on_track = data.get("on_track", True)
        status_text = "on track to meet" if on_track else "behind on"

        return (
            f"## 1. Executive Summary\n\n"
            f"Total emissions for Q{data.get('quarter', '?')} {data.get('reporting_year', '')} "
            f"are **{_dec_comma(total)} tCO2e** "
            f"(Scope 1: {_dec_comma(s1)}, Scope 2: {_dec_comma(s2)}, Scope 3: {_dec_comma(s3)}). "
            f"The organization is **{status_text}** the SBTi near-term pathway. "
            f"Key highlights include {data.get('highlights', 'continued progress on decarbonization initiatives')}.\n\n"
            f"| KPI | Value | Status |\n"
            f"|-----|------:|:------:|\n"
            f"| Total Emissions | {_dec_comma(total)} tCO2e | {_rag_md('green' if on_track else 'amber')} |\n"
            f"| SBTi Target Progress | {_pct(data.get('target_progress_pct', 0))} | {_rag_md(data.get('target_rag', 'green'))} |\n"
            f"| Renewable Energy | {_pct(data.get('renewable_energy_pct', 0))} | {_rag_md(data.get('re_rag', 'green'))} |\n"
            f"| Supplier Engagement | {_pct(data.get('supplier_engagement_pct', 0))} | {_rag_md(data.get('supplier_rag', 'amber'))} |\n"
            f"| Data Quality | {data.get('overall_dq_score', 0)}/100 | {_rag_md(data.get('dq_rag', 'green'))} |"
        )

    def _md_emission_performance(self, data: Dict[str, Any]) -> str:
        ytd_actual = float(data.get("ytd_actual_tco2e", 0))
        ytd_target = float(data.get("ytd_target_tco2e", 0))
        delta = ytd_actual - ytd_target
        return (
            f"## 2. Emission Performance vs Target Pathway\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| YTD Actual | {_dec_comma(ytd_actual)} tCO2e |\n"
            f"| YTD Target (SBTi pathway) | {_dec_comma(ytd_target)} tCO2e |\n"
            f"| Variance | {_dec_comma(abs(delta))} tCO2e {'above' if delta > 0 else 'below'} target |\n"
            f"| Status | {'ON TRACK' if delta <= 0 else 'BEHIND TARGET'} |"
        )

    def _md_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        if not initiatives:
            return "## 3. Key Initiatives Status\n\nNo active initiatives to report."
        lines = [
            "## 3. Key Initiatives Status\n",
            "| Initiative | Category | Status | Savings (tCO2e) | Investment | Completion |",
            "|------------|----------|:------:|----------------:|-----------:|:----------:|",
        ]
        for init in initiatives:
            lines.append(
                f"| {init.get('name', '')} | {init.get('category', '')} "
                f"| {_rag_md(init.get('status', 'amber'))} "
                f"| {_dec_comma(init.get('tco2e_saved', 0))} "
                f"| {data.get('currency', '$')}{_dec_comma(init.get('investment', 0))} "
                f"| {_pct(init.get('completion_pct', 0))} |"
            )
        return "\n".join(lines)

    def _md_regulatory_update(self, data: Dict[str, Any]) -> str:
        regs = data.get("regulatory_compliance", [])
        if not regs:
            return "## 4. Regulatory Compliance Update\n\nNo regulatory changes to report."
        lines = [
            "## 4. Regulatory Compliance Update\n",
            "| Framework | Status | Next Deadline | Action Required |",
            "|-----------|:------:|---------------|-----------------|",
        ]
        for reg in regs:
            lines.append(
                f"| {reg.get('framework', '')} | {_rag_md(reg.get('status', 'green'))} "
                f"| {reg.get('deadline', '')} | {reg.get('action', 'None')} |"
            )
        return "\n".join(lines)

    def _md_carbon_pricing(self, data: Dict[str, Any]) -> str:
        cp = data.get("carbon_pricing_impact", {})
        return (
            f"## 5. Carbon Pricing Impact\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| Internal Carbon Price | ${cp.get('price_per_tco2e', 0)}/tCO2e |\n"
            f"| Total Carbon Charge (Q) | {data.get('currency', '$')}{_dec_comma(cp.get('quarterly_charge', 0))} |\n"
            f"| YTD Carbon Charge | {data.get('currency', '$')}{_dec_comma(cp.get('ytd_charge', 0))} |\n"
            f"| Impact on EBITDA | {_pct(cp.get('ebitda_impact_pct', 0))} |\n"
            f"| Highest BU charge | {cp.get('highest_bu', 'N/A')} |\n"
            f"| CapEx decisions influenced | {cp.get('capex_influenced', 0)} projects |"
        )

    def _md_supply_chain(self, data: Dict[str, Any]) -> str:
        sc = data.get("supply_chain_progress", {})
        return (
            f"## 6. Supply Chain Engagement Progress\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| Suppliers engaged | {_dec_comma(sc.get('engaged', 0))} / {_dec_comma(sc.get('total', 0))} |\n"
            f"| CDP disclosure rate | {_pct(sc.get('cdp_rate', 0))} |\n"
            f"| Suppliers with SBTi | {_dec_comma(sc.get('sbti_count', 0))} |\n"
            f"| Scope 3 covered by engagement | {_pct(sc.get('scope3_covered_pct', 0))} |\n"
            f"| YoY Scope 3 change | {sc.get('yoy_change', 'N/A')} |"
        )

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        if not milestones:
            return "## 7. Upcoming Milestones (Next 90 Days)\n\nNo milestones in the next 90 days."
        lines = [
            "## 7. Upcoming Milestones (Next 90 Days)\n",
            "| Date | Milestone | Owner | Status |",
            "|------|-----------|-------|:------:|",
        ]
        for ms in milestones:
            lines.append(
                f"| {ms.get('date', '')} | {ms.get('milestone', '')} "
                f"| {ms.get('owner', '')} | {_rag_md(ms.get('status', 'green'))} |"
            )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("climate_risks", [])
        if not risks:
            return "## 8. Risk Assessment Update\n\nNo new risks identified this quarter."
        lines = [
            "## 8. Risk Assessment Update\n",
            "| Risk | Type | Likelihood | Impact | Trend | Mitigation |",
            "|------|------|:----------:|:------:|:-----:|------------|",
        ]
        for r in risks[:5]:
            lines.append(
                f"| {r.get('name', '')} | {r.get('type', '')} "
                f"| {r.get('likelihood', '')} | {r.get('impact', '')} "
                f"| {r.get('trend', 'Stable')} | {r.get('mitigation', '')} |"
            )
        return "\n".join(lines)

    def _md_decisions_required(self, data: Dict[str, Any]) -> str:
        decisions = data.get("decisions_required", [])
        if not decisions:
            return "## 9. Decisions Required\n\nNo decisions required this quarter. Paper is for information only."
        lines = ["## 9. Decisions Required\n"]
        for i, d in enumerate(decisions, 1):
            lines.append(
                f"**Decision {i}:** {d.get('description', '')}  \n"
                f"**Recommendation:** {d.get('recommendation', '')}  \n"
                f"**Financial Impact:** {d.get('financial_impact', 'N/A')}  \n"
                f"**Deadline:** {d.get('deadline', 'N/A')}\n"
            )
        return "\n".join(lines)

    def _md_appendix(self, data: Dict[str, Any]) -> str:
        return (
            f"## Appendix\n\n"
            f"- Full GHG Inventory Report: Available on request\n"
            f"- SBTi Target Progress Detail: See PACK-027 SBTi template\n"
            f"- Supply Chain Heatmap: See PACK-027 supply chain template\n"
            f"- Scenario Analysis: See PACK-027 scenario comparison template\n"
            f"- Data Quality Matrix: Overall score {data.get('overall_dq_score', 0)}/100"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*Board climate report. {data.get('classification', 'CONFIDENTIAL')}. SHA-256 provenance.*"
        )

    def _html_summary_table(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        return (
            f'<table><tr><th>Scope</th><th>tCO2e</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec_comma(s2)}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)}</td></tr>\n'
            f'<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2+s3)}</strong></td></tr>\n'
            f'</table>'
        )
