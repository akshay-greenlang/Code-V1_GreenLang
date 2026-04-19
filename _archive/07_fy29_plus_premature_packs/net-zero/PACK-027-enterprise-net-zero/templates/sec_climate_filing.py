# -*- coding: utf-8 -*-
"""
SECClimateFilingTemplate - SEC Climate Disclosure Rule filing for PACK-027.

Renders an SEC Climate Disclosure Rule (S-X Article 14) filing template
covering Scope 1+2 disclosure, material Scope 3, attestation requirements,
financial statement footnotes, and transition plan disclosures for large
accelerated filers.

Sections:
    1. Filing Summary (registrant info, filing type)
    2. Governance Disclosures (Reg S-K Item 1501)
    3. Strategy & Risk Management (Reg S-K Item 1502-1503)
    4. GHG Emissions (Reg S-K Item 1504)
    5. Targets and Goals (Reg S-K Item 1505)
    6. Financial Statement Footnotes (Reg S-X Article 14)
    7. Attestation Statement (for LAF)
    8. Safe Harbor & Forward-Looking Statements

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
_TEMPLATE_ID = "sec_climate_filing"

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

class SECClimateFilingTemplate:
    """
    SEC Climate Disclosure Rule filing template.

    Generates SEC-compliant climate disclosures per Reg S-K Items 1501-1505
    and Reg S-X Article 14. Covers governance, strategy, GHG emissions,
    targets, and financial statement effects.
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
            self._md_filing_summary(data),
            self._md_governance(data),
            self._md_strategy_risk(data),
            self._md_ghg_emissions(data),
            self._md_targets_goals(data),
            self._md_financial_footnotes(data),
            self._md_attestation(data),
            self._md_safe_harbor(data),
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
            f".report{{max-width:1000px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".sec-notice{{background:#fff3e0;padding:16px;border-radius:8px;margin:12px 0;"
            f"border:1px solid #ef6c00;font-size:0.9em;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        body = (
            f'<h1>SEC Climate Disclosure</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'FY{data.get("reporting_year", "")} | SEC Final Rule S7-10-22</p>\n'
            f'<h2>GHG Emissions (Reg S-K Item 1504)</h2>\n'
            f'<table><tr><th>Scope</th><th>tCO2e</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec_comma(s2)}</td></tr>\n'
            f'</table>\n'
            f'<div class="sec-notice">This template is for informational purposes. '
            f'Consult legal counsel before filing with the SEC.</div>\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | SEC Filing | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>SEC Climate Filing</title>\n'
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
            "regulation": "SEC Final Rule S7-10-22",
            "registrant": {
                "name": data.get("org_name", ""),
                "cik": data.get("cik", ""),
                "filing_type": data.get("filing_type", "10-K"),
                "filer_category": data.get("filer_category", "Large Accelerated Filer"),
                "fiscal_year": data.get("reporting_year", ""),
            },
            "item_1501_governance": data.get("sec_governance", {}),
            "item_1502_strategy": data.get("sec_strategy", {}),
            "item_1503_risk_management": data.get("sec_risk_management", {}),
            "item_1504_ghg_emissions": {
                "scope1_tco2e": round(s1, 2),
                "scope2_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2) if data.get("scope3_material", False) else None,
                "scope3_material": data.get("scope3_material", False),
                "methodology": data.get("ghg_methodology", "GHG Protocol Corporate Standard"),
                "gwp_source": data.get("gwp_source", "IPCC AR6"),
            },
            "item_1505_targets": data.get("sec_targets", []),
            "article_14_footnotes": data.get("financial_footnotes", {}),
            "attestation_required": data.get("attestation_required", True),
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sec_climate_{data.get('org_name', 'org').replace(' ', '_')}_FY{data.get('reporting_year', '')}.xlsx",
            "worksheets": [{
                "name": "SEC Emissions",
                "headers": ["Item", "Value", "Unit", "Reference"],
                "rows": [
                    ["Scope 1 GHG Emissions", round(s1, 2), "tCO2e", "Reg S-K Item 1504(a)"],
                    ["Scope 2 GHG Emissions", round(s2, 2), "tCO2e", "Reg S-K Item 1504(b)"],
                ],
            }],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # Markdown sections

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# SEC Climate Disclosure\n\n"
            f"## {data.get('org_name', 'Registrant')} -- FY{data.get('reporting_year', '')}\n\n"
            f"**Regulation:** SEC Final Rule S7-10-22 (Climate Disclosure)  \n"
            f"**Filing:** {data.get('filing_type', '10-K')}  \n"
            f"**Filer Category:** {data.get('filer_category', 'Large Accelerated Filer')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_filing_summary(self, data: Dict[str, Any]) -> str:
        return (
            f"## Filing Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Registrant | {data.get('org_name', '')} |\n"
            f"| CIK | {data.get('cik', 'N/A')} |\n"
            f"| Fiscal Year End | {data.get('fiscal_year_end', '')} |\n"
            f"| Filer Category | {data.get('filer_category', 'Large Accelerated Filer')} |\n"
            f"| Attestation Required | {'Yes' if data.get('attestation_required', True) else 'No'} |\n"
            f"| Scope 3 Material | {'Yes' if data.get('scope3_material', False) else 'No'} |"
        )

    def _md_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("sec_governance", {})
        return (
            f"## Governance (Reg S-K Item 1501)\n\n"
            f"**Board oversight:** {gov.get('board_oversight', 'Board Sustainability Committee oversees climate risks')}  \n"
            f"**Management role:** {gov.get('management_role', 'CSO reports to CEO with quarterly board updates')}  \n"
            f"**Climate expertise:** {gov.get('expertise', 'Board members have completed climate governance training')}  \n"
            f"**Integration:** {gov.get('integration', 'Climate risks reviewed in strategic planning and risk management')}  "
        )

    def _md_strategy_risk(self, data: Dict[str, Any]) -> str:
        strategy = data.get("sec_strategy", {})
        rm = data.get("sec_risk_management", {})
        return (
            f"## Strategy & Risk Management (Reg S-K Items 1502-1503)\n\n"
            f"### Climate-Related Risks\n\n"
            f"{strategy.get('risks_description', 'See TCFD report for detailed risk assessment.')}  \n\n"
            f"### Transition Plan\n\n"
            f"{strategy.get('transition_plan', 'SBTi-validated targets with 1.5C pathway.')}  \n\n"
            f"### Risk Management Process\n\n"
            f"{rm.get('process', 'Climate risks integrated into enterprise risk framework with quarterly assessment.')}  "
        )

    def _md_ghg_emissions(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        scope3_material = data.get("scope3_material", False)
        lines = [
            "## GHG Emissions (Reg S-K Item 1504)\n",
            "| Metric | tCO2e | Methodology | Ref |",
            "|--------|------:|-------------|-----|",
            f"| Scope 1 (Direct) | {_dec_comma(s1)} | {data.get('ghg_methodology', 'GHG Protocol')} | Item 1504(a) |",
            f"| Scope 2 (Indirect Energy) | {_dec_comma(s2)} | {data.get('ghg_methodology', 'GHG Protocol')} | Item 1504(b) |",
        ]
        if scope3_material:
            lines.append(
                f"| Scope 3 (Material)* | {_dec_comma(s3)} | "
                f"{data.get('ghg_methodology', 'GHG Protocol')} | Item 1504(c) |"
            )
            lines.append(f"\n*Scope 3 disclosed as material per registrant's assessment.")
        lines.append(f"\n**GWP Source:** {data.get('gwp_source', 'IPCC AR6 GWP-100')}")
        lines.append(f"**Consolidation:** {data.get('consolidation_approach', 'Operational control')}")
        return "\n".join(lines)

    def _md_targets_goals(self, data: Dict[str, Any]) -> str:
        targets = data.get("sec_targets", [])
        if not targets:
            return "## Targets and Goals (Reg S-K Item 1505)\n\nNo climate-related targets publicly disclosed."
        lines = [
            "## Targets and Goals (Reg S-K Item 1505)\n",
            "| Target | Scope | Base Year | Target Year | Metric | Progress |",
            "|--------|-------|:---------:|:-----------:|--------|:--------:|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('name', '')} | {t.get('scope', '')} "
                f"| {t.get('base_year', '')} | {t.get('target_year', '')} "
                f"| {t.get('metric', '')} | {_pct(t.get('progress_pct', 0))} |"
            )
        return "\n".join(lines)

    def _md_financial_footnotes(self, data: Dict[str, Any]) -> str:
        fn = data.get("financial_footnotes", {})
        return (
            f"## Financial Statement Footnotes (Reg S-X Article 14)\n\n"
            f"### Climate-Related Expenditures\n\n"
            f"**Capitalized costs:** {data.get('currency', '$')}{_dec_comma(fn.get('capitalized_costs', 0))}  \n"
            f"**Expensed costs:** {data.get('currency', '$')}{_dec_comma(fn.get('expensed_costs', 0))}  \n"
            f"**Losses from climate events:** {data.get('currency', '$')}{_dec_comma(fn.get('climate_losses', 0))}  \n"
            f"**Insurance recoveries:** {data.get('currency', '$')}{_dec_comma(fn.get('insurance_recoveries', 0))}  \n\n"
            f"### Carbon Offset/Credit Costs\n\n"
            f"**Offset purchases:** {data.get('currency', '$')}{_dec_comma(fn.get('offset_costs', 0))}  \n"
            f"**REC purchases:** {data.get('currency', '$')}{_dec_comma(fn.get('rec_costs', 0))}  "
        )

    def _md_attestation(self, data: Dict[str, Any]) -> str:
        if not data.get("attestation_required", True):
            return "## Attestation\n\nAttestation not required for this filer category."
        return (
            f"## Attestation (Large Accelerated Filer)\n\n"
            f"Scope 1 and Scope 2 GHG emissions have been subject to attestation "
            f"by an independent attestation provider in accordance with the standards "
            f"established by the PCAOB or equivalent body.\n\n"
            f"**Attestation Provider:** {data.get('attestation_provider', '[Provider Name]')}  \n"
            f"**Level:** {data.get('attestation_level', 'Limited assurance')}  \n"
            f"**Standard:** {data.get('attestation_standard', 'PCAOB AS 3000 equivalent')}  "
        )

    def _md_safe_harbor(self, data: Dict[str, Any]) -> str:
        return (
            "## Safe Harbor Statement\n\n"
            "Certain statements in this filing constitute forward-looking statements "
            "within the meaning of Section 27A of the Securities Act of 1933 and "
            "Section 21E of the Securities Exchange Act of 1934. These statements "
            "include, but are not limited to, statements regarding the registrant's "
            "climate-related targets, transition plans, and expected capital expenditures. "
            "These forward-looking statements are based on the registrant's current "
            "expectations and are subject to uncertainty and changes in circumstances."
        )

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "SEC-001", "source": "SEC Final Rule S7-10-22 (Climate Disclosure)", "year": "2024"},
            {"ref": "SEC-002", "source": "Regulation S-K Items 1501-1505", "year": "2024"},
            {"ref": "SEC-003", "source": "Regulation S-X Article 14", "year": "2024"},
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
            f"*SEC Climate Disclosure Rule template. Consult legal counsel before filing.*  \n"
            f"*SHA-256 provenance hashing applied.*"
        )
