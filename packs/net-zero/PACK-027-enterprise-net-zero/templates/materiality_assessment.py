# -*- coding: utf-8 -*-
"""
MaterialityAssessmentTemplate - Financial materiality assessment for PACK-027.

Renders a financial materiality assessment for climate-related topics
covering double materiality (impact and financial), stakeholder analysis,
quantitative thresholds, and disclosure mapping to regulatory frameworks.

Sections:
    1. Assessment Overview
    2. Financial Materiality (outside-in: climate impacts on business)
    3. Impact Materiality (inside-out: business impacts on climate)
    4. Double Materiality Matrix
    5. Stakeholder Analysis
    6. Quantitative Thresholds
    7. Material Topics Identified
    8. Disclosure Mapping (which topics trigger which disclosures)
    9. Methodology & Process
   10. Citations

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
_TEMPLATE_ID = "materiality_assessment"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

MATERIALITY_CATEGORIES = [
    {"id": "ghg_emissions", "name": "GHG Emissions", "esrs": "E1"},
    {"id": "energy_transition", "name": "Energy Transition", "esrs": "E1"},
    {"id": "physical_risk", "name": "Physical Climate Risk", "esrs": "E1"},
    {"id": "transition_risk", "name": "Transition Risk", "esrs": "E1"},
    {"id": "carbon_pricing", "name": "Carbon Pricing Exposure", "esrs": "E1"},
    {"id": "supply_chain_climate", "name": "Supply Chain Climate Risk", "esrs": "E1"},
    {"id": "water_stress", "name": "Water Stress", "esrs": "E3"},
    {"id": "biodiversity", "name": "Biodiversity & Ecosystems", "esrs": "E4"},
    {"id": "circular_economy", "name": "Circular Economy", "esrs": "E5"},
    {"id": "just_transition", "name": "Just Transition", "esrs": "S1"},
]

DISCLOSURE_FRAMEWORKS = [
    {"id": "esrs_e1", "name": "ESRS E1 Climate Change", "trigger": "E1 material"},
    {"id": "sec_climate", "name": "SEC Climate Rule", "trigger": "Material climate risk"},
    {"id": "tcfd", "name": "TCFD/ISSB S2", "trigger": "Climate-related F/O/R"},
    {"id": "cdp", "name": "CDP Climate Change", "trigger": "All companies"},
    {"id": "sbti", "name": "SBTi Targets", "trigger": "Voluntary commitment"},
    {"id": "ca_sb253", "name": "California SB 253", "trigger": ">$1B revenue in CA"},
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

class MaterialityAssessmentTemplate:
    """
    Financial materiality assessment template.

    Double materiality analysis covering impact and financial materiality,
    stakeholder engagement, quantitative thresholds, and disclosure
    framework mapping.
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
            self._md_overview(data),
            self._md_financial_materiality(data),
            self._md_impact_materiality(data),
            self._md_double_materiality_matrix(data),
            self._md_stakeholder_analysis(data),
            self._md_quantitative_thresholds(data),
            self._md_material_topics(data),
            self._md_disclosure_mapping(data),
            self._md_methodology(data),
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
            f".material{{background:#ffcdd2;color:#b71c1c;padding:2px 8px;border-radius:4px;font-size:0.85em;}}"
            f".not-material{{background:#c8e6c9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-size:0.85em;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        topics = data.get("material_topics", [])
        rows = ""
        for t in topics:
            cls = "material" if t.get("material", True) else "not-material"
            label = "Material" if t.get("material", True) else "Not Material"
            rows += (
                f'<tr><td>{t.get("name", "")}</td>'
                f'<td>{t.get("financial_score", "-")}</td>'
                f'<td>{t.get("impact_score", "-")}</td>'
                f'<td><span class="{cls}">{label}</span></td></tr>\n'
            )
        body = (
            f'<h1>Materiality Assessment</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'FY{data.get("reporting_year", "")} | Double Materiality (CSRD)</p>\n'
            f'<h2>Material Topics</h2>\n'
            f'<table><tr><th>Topic</th><th>Financial</th><th>Impact</th><th>Status</th></tr>\n'
            f'{rows}</table>\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | Materiality | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>Materiality Assessment</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "methodology": data.get("materiality_methodology", "CSRD Double Materiality (EFRAG IG-1)"),
            "financial_materiality": {
                "threshold": data.get("financial_threshold", ">1% of revenue or >5% of EBITDA"),
                "topics": data.get("financial_material_topics", []),
            },
            "impact_materiality": {
                "threshold": data.get("impact_threshold", "Significant negative/positive impact on people or environment"),
                "topics": data.get("impact_material_topics", []),
            },
            "material_topics": data.get("material_topics", []),
            "stakeholders_engaged": data.get("stakeholders", []),
            "disclosure_mapping": [
                {
                    "framework": fw["name"],
                    "triggered": data.get(f"fw_{fw['id']}_triggered", True),
                    "trigger_reason": fw["trigger"],
                }
                for fw in DISCLOSURE_FRAMEWORKS
            ],
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        topics = data.get("material_topics", [])
        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"materiality_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [{
                "name": "Material Topics",
                "headers": ["Topic", "Financial Score", "Impact Score", "Material", "ESRS", "Disclosure Required"],
                "rows": [
                    [
                        t.get("name", ""), t.get("financial_score", 0),
                        t.get("impact_score", 0),
                        "Yes" if t.get("material", True) else "No",
                        t.get("esrs", ""), t.get("disclosure", ""),
                    ]
                    for t in topics
                ],
            }],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # Markdown sections

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Financial Materiality Assessment\n\n"
            f"## {data.get('org_name', 'Enterprise')} -- FY{data.get('reporting_year', '')}\n\n"
            f"**Methodology:** CSRD Double Materiality (EFRAG IG-1)  \n"
            f"**Focus:** Climate-related topics (ESRS E1)  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        topics = data.get("material_topics", [])
        material_count = sum(1 for t in topics if t.get("material", True))
        return (
            f"## 1. Assessment Overview\n\n"
            f"This assessment evaluates the financial materiality of climate-related "
            f"topics for {data.get('org_name', 'the organization')} using the CSRD "
            f"double materiality approach.\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| Topics assessed | {len(topics)} |\n"
            f"| Topics determined material | {material_count} |\n"
            f"| Stakeholder groups engaged | {len(data.get('stakeholders', []))} |\n"
            f"| Financial threshold | {data.get('financial_threshold', '>1% revenue or >5% EBITDA')} |\n"
            f"| Assessment date | {data.get('assessment_date', data.get('reporting_year', ''))} |"
        )

    def _md_financial_materiality(self, data: Dict[str, Any]) -> str:
        fm_topics = data.get("financial_material_topics", [])
        lines = [
            "## 2. Financial Materiality (Outside-In)\n",
            "Climate impacts that may affect the organization's financial position, "
            "performance, and cash flows.\n",
            "| Topic | Risk/Opportunity | Financial Impact | Likelihood | Timeframe | Material |",
            "|-------|:----------------:|:----------------:|:----------:|:---------:|:--------:|",
        ]
        for t in fm_topics:
            lines.append(
                f"| {t.get('name', '')} | {t.get('type', '')} "
                f"| {t.get('financial_impact', '')} | {t.get('likelihood', '')} "
                f"| {t.get('timeframe', '')} | {'Yes' if t.get('material', True) else 'No'} |"
            )
        if not fm_topics:
            lines.append("| See material topics section | | | | | |")
        return "\n".join(lines)

    def _md_impact_materiality(self, data: Dict[str, Any]) -> str:
        im_topics = data.get("impact_material_topics", [])
        lines = [
            "## 3. Impact Materiality (Inside-Out)\n",
            "Organization's impacts on climate change and the environment.\n",
            "| Topic | Impact Type | Severity | Scale | Irremediability | Material |",
            "|-------|:----------:|:--------:|:-----:|:--------------:|:--------:|",
        ]
        for t in im_topics:
            lines.append(
                f"| {t.get('name', '')} | {t.get('impact_type', '')} "
                f"| {t.get('severity', '')} | {t.get('scale', '')} "
                f"| {t.get('irremediability', '')} | {'Yes' if t.get('material', True) else 'No'} |"
            )
        if not im_topics:
            lines.append("| See material topics section | | | | | |")
        return "\n".join(lines)

    def _md_double_materiality_matrix(self, data: Dict[str, Any]) -> str:
        topics = data.get("material_topics", [])
        lines = [
            "## 4. Double Materiality Matrix\n",
            "| Topic | Financial Score (1-5) | Impact Score (1-5) | Combined | Material |",
            "|-------|:--------------------:|:------------------:|:--------:|:--------:|",
        ]
        for t in topics:
            fs = t.get("financial_score", 0)
            ims = t.get("impact_score", 0)
            combined = max(fs, ims)
            lines.append(
                f"| {t.get('name', '')} | {fs} | {ims} | {combined} "
                f"| {'**YES**' if t.get('material', True) else 'No'} |"
            )
        return "\n".join(lines)

    def _md_stakeholder_analysis(self, data: Dict[str, Any]) -> str:
        stakeholders = data.get("stakeholders", [])
        if not stakeholders:
            return "## 5. Stakeholder Analysis\n\nStakeholder engagement details not provided."
        lines = [
            "## 5. Stakeholder Analysis\n",
            "| Stakeholder Group | Method | Participants | Key Concerns |",
            "|-------------------|--------|:-----------:|--------------|",
        ]
        for s in stakeholders:
            lines.append(
                f"| {s.get('group', '')} | {s.get('method', '')} "
                f"| {s.get('participants', '')} | {s.get('concerns', '')} |"
            )
        return "\n".join(lines)

    def _md_quantitative_thresholds(self, data: Dict[str, Any]) -> str:
        return (
            f"## 6. Quantitative Thresholds\n\n"
            f"| Dimension | Threshold | Application |\n"
            f"|-----------|-----------|-------------|\n"
            f"| Financial (revenue) | {data.get('threshold_revenue', '>1% of annual revenue')} | Revenue at risk from climate |\n"
            f"| Financial (EBITDA) | {data.get('threshold_ebitda', '>5% of EBITDA')} | Cost impact from carbon pricing |\n"
            f"| Financial (CapEx) | {data.get('threshold_capex', '>10% of annual CapEx')} | Climate transition investment |\n"
            f"| Impact (absolute) | {data.get('threshold_abs_emissions', '>10,000 tCO2e')} | Direct emission significance |\n"
            f"| Impact (intensity) | {data.get('threshold_intensity', 'Above sector median')} | Relative to sector benchmark |\n"
            f"| Stakeholder (count) | {data.get('threshold_stakeholder', '>50% of stakeholders identify')} | Stakeholder salience |"
        )

    def _md_material_topics(self, data: Dict[str, Any]) -> str:
        topics = data.get("material_topics", [])
        material = [t for t in topics if t.get("material", True)]
        if not material:
            return "## 7. Material Topics Identified\n\nNo material topics identified (assessment pending)."
        lines = [
            "## 7. Material Topics Identified\n",
            f"**{len(material)} topics determined material:**\n",
        ]
        for i, t in enumerate(material, 1):
            lines.append(
                f"{i}. **{t.get('name', '')}** "
                f"(Financial: {t.get('financial_score', '-')}/5, "
                f"Impact: {t.get('impact_score', '-')}/5) -- "
                f"{t.get('rationale', '')}"
            )
        return "\n".join(lines)

    def _md_disclosure_mapping(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 8. Disclosure Framework Mapping\n",
            "| Framework | Triggered | Trigger Condition | Disclosures Required |",
            "|-----------|:---------:|-------------------|--------------------|",
        ]
        for fw in DISCLOSURE_FRAMEWORKS:
            triggered = data.get(f"fw_{fw['id']}_triggered", True)
            disclosures = data.get(f"fw_{fw['id']}_disclosures", "See framework requirements")
            lines.append(
                f"| {fw['name']} | {'Yes' if triggered else 'No'} "
                f"| {fw['trigger']} | {disclosures} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        return (
            f"## 9. Methodology & Process\n\n"
            f"**Framework:** CSRD Double Materiality per EFRAG Implementation Guidance 1 (IG-1)  \n"
            f"**Process steps:**\n\n"
            f"1. Topic identification from ESRS topic list and sector-specific standards\n"
            f"2. Stakeholder engagement (surveys, interviews, workshops)\n"
            f"3. Financial materiality assessment (outside-in analysis)\n"
            f"4. Impact materiality assessment (inside-out analysis)\n"
            f"5. Quantitative threshold application\n"
            f"6. Double materiality determination (topic is material if financial OR impact material)\n"
            f"7. Board approval of materiality conclusions\n"
            f"8. Disclosure mapping to applicable frameworks\n\n"
            f"**Board approval date:** {data.get('board_approval_date', 'Pending')}  \n"
            f"**Next review:** {data.get('next_review', 'Annual, or upon significant change in circumstances')}  "
        )

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "DM-001", "source": "EFRAG IG-1: Materiality Assessment Implementation Guidance", "year": "2023"},
            {"ref": "DM-002", "source": "ESRS 1: General Requirements (Chapter 3)", "year": "2023"},
            {"ref": "DM-003", "source": "CSRD Directive (EU) 2022/2464", "year": "2022"},
        ])
        lines = ["## 10. Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*Double materiality assessment per CSRD/EFRAG IG-1. SHA-256 provenance.*"
        )
