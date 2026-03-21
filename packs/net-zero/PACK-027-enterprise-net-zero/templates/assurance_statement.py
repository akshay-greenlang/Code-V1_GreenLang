# -*- coding: utf-8 -*-
"""
AssuranceStatementTemplate - ISO 14064-3 assurance statement for PACK-027.

Renders an ISO 14064-3 / ISAE 3410 assurance statement template with
management assertion, scope of engagement, criteria, findings, and
conclusion. Supports both limited and reasonable assurance versions.

Sections:
    1. Engagement Summary
    2. Management Assertion
    3. Scope of Engagement
    4. Criteria and Standards
    5. Independence and Competence
    6. Work Performed (limited / reasonable)
    7. Findings Summary
    8. Materiality Assessment
    9. Data Quality Evaluation
   10. Conclusion (opinion)
   11. Recommendations
   12. Management Response

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "assurance_statement"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

ASSURANCE_LEVELS = {
    "limited": {
        "label": "Limited Assurance",
        "standard": "ISAE 3410 / ISAE 3000 (Revised)",
        "conclusion_form": "negative form (nothing has come to our attention)",
        "evidence_level": "Primarily analytical procedures and enquiry",
    },
    "reasonable": {
        "label": "Reasonable Assurance",
        "standard": "ISAE 3410 / ISAE 3000 (Revised)",
        "conclusion_form": "positive form (in our opinion)",
        "evidence_level": "Substantive testing, detailed verification, site visits",
    },
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class AssuranceStatementTemplate:
    """
    ISO 14064-3 / ISAE 3410 assurance statement template.

    Generates limited or reasonable assurance statement with management
    assertion, scope, findings, and conclusion. Designed for external
    auditor review and customization.
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
        self.generated_at = _utcnow()
        level = data.get("assurance_level", "limited")
        al = ASSURANCE_LEVELS.get(level, ASSURANCE_LEVELS["limited"])

        sections = [
            self._md_header(data, al),
            self._md_engagement_summary(data, al),
            self._md_management_assertion(data),
            self._md_scope(data, al),
            self._md_criteria(data, al),
            self._md_independence(data),
            self._md_work_performed(data, al),
            self._md_findings(data),
            self._md_materiality(data),
            self._md_data_quality(data),
            self._md_conclusion(data, al),
            self._md_recommendations(data),
            self._md_management_response(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        level = data.get("assurance_level", "limited")
        al = ASSURANCE_LEVELS.get(level, ASSURANCE_LEVELS["limited"])
        css = (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:900px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".conclusion{{background:{_LIGHT};padding:20px;border-radius:8px;margin:16px 0;"
            f"border-left:4px solid {_PRIMARY};font-size:1.05em;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3

        body = (
            f'<h1>Independent {al["label"]} Report</h1>\n'
            f'<p>To the Board of Directors of <strong>{data.get("org_name", "")}</strong></p>\n'
            f'<h2>Scope</h2>\n'
            f'<p>GHG Statement for {data.get("reporting_year", "")} | '
            f'Standard: {al["standard"]} | ISO 14064-3:2019</p>\n'
            f'<h2>Emissions Verified</h2>\n'
            f'<table><tr><th>Scope</th><th>tCO2e</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td></tr>\n'
            f'<tr><td>Scope 2 (Location)</td><td>{_dec_comma(s2)}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)}</td></tr>\n'
            f'<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(total)}</strong></td></tr>\n'
            f'</table>\n'
            f'<div class="conclusion"><strong>Conclusion:</strong> '
            f'{self._conclusion_text(data, al)}</div>\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | Assurance Template | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>Assurance Statement</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        level = data.get("assurance_level", "limited")
        al = ASSURANCE_LEVELS.get(level, ASSURANCE_LEVELS["limited"])
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "assurance_level": al["label"],
            "standard": al["standard"],
            "organization": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "assurance_provider": data.get("assurance_provider", "[Assurance Provider Name]"),
            "emissions_verified": {
                "scope1_tco2e": round(s1, 2),
                "scope2_location_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
                "total_tco2e": round(s1 + s2 + s3, 2),
            },
            "scope_of_engagement": data.get("scope_of_engagement", {}),
            "findings": data.get("findings", []),
            "materiality_threshold": data.get("materiality_threshold", "5%"),
            "conclusion": {
                "type": al["conclusion_form"],
                "opinion": data.get("opinion", "unmodified"),
                "text": self._conclusion_text(data, al),
            },
            "recommendations": data.get("recommendations", []),
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"assurance_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [{
                "name": "Findings",
                "headers": ["Finding #", "Area", "Severity", "Description", "Status"],
                "rows": [
                    [i + 1, f.get("area", ""), f.get("severity", ""),
                     f.get("description", ""), f.get("status", "")]
                    for i, f in enumerate(data.get("findings", []))
                ],
            }],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _conclusion_text(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        org = data.get("org_name", "the organization")
        year = data.get("reporting_year", "")
        opinion = data.get("opinion", "unmodified")
        if al.get("conclusion_form", "").startswith("negative"):
            if opinion == "unmodified":
                return (
                    f"Based on our {al['label'].lower()} procedures, nothing has come to our "
                    f"attention that causes us to believe that {org}'s GHG statement for "
                    f"{year} is not prepared, in all material respects, in accordance with "
                    f"the GHG Protocol Corporate Standard and ISO 14064-1:2018."
                )
            return f"[Modified conclusion - see findings for details]"
        else:
            if opinion == "unmodified":
                return (
                    f"In our opinion, {org}'s GHG statement for {year} is prepared, in all "
                    f"material respects, in accordance with the GHG Protocol Corporate "
                    f"Standard and ISO 14064-1:2018."
                )
            return f"[Modified opinion - see findings for details]"

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Independent {al['label']} Report\n\n"
            f"**To the Board of Directors of {data.get('org_name', 'the Organization')}**\n\n"
            f"**Report ID:** {_new_uuid()}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_engagement_summary(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        return (
            f"## 1. Engagement Summary\n\n"
            f"| Field | Detail |\n"
            f"|-------|--------|\n"
            f"| Organization | {data.get('org_name', '')} |\n"
            f"| Assurance Level | {al['label']} |\n"
            f"| Standard | {al['standard']} + ISO 14064-3:2019 |\n"
            f"| Reporting Year | {data.get('reporting_year', '')} |\n"
            f"| Assurance Provider | {data.get('assurance_provider', '[Provider Name]')} |\n"
            f"| Lead Verifier | {data.get('lead_verifier', '[Name, Qualification]')} |\n"
            f"| Engagement Date | {data.get('engagement_date', '[Date]')} |\n"
            f"| Materiality | {data.get('materiality_threshold', '5%')} |"
        )

    def _md_management_assertion(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        return (
            f"## 2. Management Assertion\n\n"
            f"The management of {data.get('org_name', 'the Organization')} asserts that the "
            f"accompanying GHG statement for the year ended {data.get('reporting_year', '')} "
            f"is prepared in accordance with the GHG Protocol Corporate Accounting and "
            f"Reporting Standard and presents fairly, in all material respects, the "
            f"organization's GHG emissions:\n\n"
            f"| Scope | tCO2e |\n"
            f"|-------|------:|\n"
            f"| Scope 1 (Direct) | {_dec_comma(s1)} |\n"
            f"| Scope 2 (Location-based) | {_dec_comma(s2)} |\n"
            f"| Scope 3 (Value Chain) | {_dec_comma(s3)} |\n"
            f"| **Total** | **{_dec_comma(total)}** |"
        )

    def _md_scope(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        scope = data.get("scope_of_engagement", {})
        return (
            f"## 3. Scope of Engagement\n\n"
            f"**Subject matter:** {scope.get('subject_matter', 'GHG statement (Scope 1, 2, and 3)')}  \n"
            f"**Organizational boundary:** {scope.get('boundary', data.get('consolidation_approach', 'Operational control'))}  \n"
            f"**Gases included:** {scope.get('gases', 'CO2, CH4, N2O, HFCs, PFCs, SF6, NF3')}  \n"
            f"**GWP source:** {scope.get('gwp', 'IPCC AR6 GWP-100')}  \n"
            f"**Entities in scope:** {scope.get('entities', 'All consolidated entities')}  \n"
            f"**Exclusions:** {scope.get('exclusions', 'None')}  "
        )

    def _md_criteria(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        return (
            f"## 4. Criteria and Standards\n\n"
            f"The GHG statement has been prepared in accordance with:\n\n"
            f"- GHG Protocol Corporate Accounting and Reporting Standard (2004, amended 2015)\n"
            f"- GHG Protocol Scope 2 Guidance (2015)\n"
            f"- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)\n"
            f"- ISO 14064-1:2018 Specification for quantification and reporting\n\n"
            f"The assurance engagement was conducted in accordance with:\n\n"
            f"- ISO 14064-3:2019 Specification for verification and validation\n"
            f"- {al['standard']}\n"
            f"- {'ISAE 3410 Assurance Engagements on GHG Statements' if 'ISAE' in al['standard'] else ''}"
        )

    def _md_independence(self, data: Dict[str, Any]) -> str:
        return (
            f"## 5. Independence and Competence\n\n"
            f"**Independence:** {data.get('independence_statement', 'The assurance provider confirms independence from the organization in accordance with IESBA Code of Ethics.')}  \n"
            f"**Competence:** {data.get('competence_statement', 'The verification team has the requisite competence in GHG accounting, assurance standards, and the relevant sector.')}  \n"
            f"**Quality management:** {data.get('quality_statement', 'This engagement was subject to the quality management system of the assurance provider.')}  "
        )

    def _md_work_performed(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        procedures = data.get("procedures_performed", [])
        if not procedures:
            if "limited" in al.get("label", "").lower():
                procedures = [
                    "Enquiry of management and personnel responsible for GHG data",
                    "Analytical procedures on emissions data (trend, ratio, reasonableness)",
                    "Review of GHG data management systems and processes",
                    "Review of emission factor selection and application",
                    "Review of scope and boundary determination",
                    "Comparison of reported data to source documentation (sample basis)",
                    "Assessment of data quality and uncertainty",
                ]
            else:
                procedures = [
                    "Detailed testing of source data to invoices/meter readings",
                    "Site visits to material emission sources",
                    "Re-performance of emission calculations",
                    "Verification of emission factor sources and applicability",
                    "Testing of data management controls",
                    "Detailed review of Scope 3 methodology and data",
                    "Assessment of base year recalculation triggers",
                    "Analytical review of year-over-year trends",
                    "External confirmation of energy data with utilities",
                    "Assessment of data quality scoring accuracy",
                ]
        lines = [f"## 6. Work Performed ({al['label']})\n"]
        for p in procedures:
            lines.append(f"- {p}")
        return "\n".join(lines)

    def _md_findings(self, data: Dict[str, Any]) -> str:
        findings = data.get("findings", [])
        if not findings:
            return "## 7. Findings\n\nNo material findings identified during the engagement."
        lines = [
            "## 7. Findings\n",
            "| # | Area | Severity | Description | Status |",
            "|:-:|------|:--------:|-------------|:------:|",
        ]
        for i, f in enumerate(findings, 1):
            lines.append(
                f"| {i} | {f.get('area', '')} | {f.get('severity', 'Minor')} "
                f"| {f.get('description', '')} | {f.get('status', 'Open')} |"
            )
        return "\n".join(lines)

    def _md_materiality(self, data: Dict[str, Any]) -> str:
        threshold = data.get("materiality_threshold", "5%")
        return (
            f"## 8. Materiality Assessment\n\n"
            f"**Materiality threshold:** {threshold} of total reported emissions  \n"
            f"**Quantitative materiality:** {_dec_comma(data.get('materiality_tco2e', 0))} tCO2e  \n"
            f"**Qualitative considerations:** Omissions, misstatements, aggregation errors  \n"
            f"**Material misstatements identified:** {data.get('material_misstatements', 'None')}  "
        )

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        return (
            f"## 9. Data Quality Evaluation\n\n"
            f"**Overall DQ score:** {data.get('overall_dq_score', 0)}/100  \n"
            f"**Target accuracy:** {data.get('target_accuracy', '+/-3%')}  \n"
            f"**Assessed accuracy:** {data.get('assessed_accuracy', 'Within target')}  \n"
            f"**SHA-256 provenance:** All calculation outputs hashed for audit trail  \n"
            f"**Calculation approach:** Deterministic (zero-hallucination, no LLM in calculation path)  "
        )

    def _md_conclusion(self, data: Dict[str, Any], al: Dict[str, str]) -> str:
        return (
            f"## 10. Conclusion\n\n"
            f"> {self._conclusion_text(data, al)}\n\n"
            f"**[Signature]**  \n"
            f"{data.get('assurance_provider', '[Assurance Provider Name]')}  \n"
            f"{data.get('lead_verifier', '[Lead Verifier Name, Qualification]')}  \n"
            f"{data.get('engagement_date', '[Date]')}  "
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        if not recs:
            return "## 11. Recommendations\n\nNo recommendations at this time."
        lines = ["## 11. Recommendations\n"]
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r.get('description', r) if isinstance(r, dict) else r}")
        return "\n".join(lines)

    def _md_management_response(self, data: Dict[str, Any]) -> str:
        return (
            f"## 12. Management Response\n\n"
            f"{data.get('management_response', '[Management response to findings and recommendations to be inserted here]')}"
        )

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "ISO-001", "source": "ISO 14064-3:2019 Verification and Validation", "year": "2019"},
            {"ref": "ISO-002", "source": "ISO 14064-1:2018 Quantification and Reporting", "year": "2018"},
            {"ref": "ISAE-001", "source": "ISAE 3410 Assurance on GHG Statements", "year": "2012"},
            {"ref": "ISAE-002", "source": "ISAE 3000 (Revised) Assurance Standard", "year": "2013"},
        ])
        lines = ["## Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Template generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*This template is designed for external auditor review and customization.*  \n"
            f"*ISO 14064-3:2019 + ISAE 3410 compliant framework. SHA-256 provenance.*"
        )
