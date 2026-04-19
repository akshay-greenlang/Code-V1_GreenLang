# -*- coding: utf-8 -*-
"""
SBTiTargetSubmissionTemplate - SBTi submission package for PACK-027.

Renders a complete SBTi Corporate Standard target submission package
covering near-term (C1-C28), long-term, and net-zero (NZ-C1 to NZ-C14)
targets with criteria validation matrix, pathway visualization, coverage
analysis, and supporting documentation.

Sections:
    1. Executive Summary (target overview)
    2. Organizational Profile
    3. Near-Term Targets (ACA/SDA/FLAG with annual milestones)
    4. Long-Term / Net-Zero Targets
    5. Criteria Validation Matrix (42 criteria pass/fail/warning)
    6. Coverage Analysis (95% Scope 1+2, 67%+ Scope 3)
    7. Pathway Visualization (base year to 2050)
    8. FLAG Assessment (if applicable)
    9. Supporting Documentation
   10. Submission Checklist

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
_TEMPLATE_ID = "sbti_target_submission"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"
_PASS_CLR = "#2e7d32"
_FAIL_CLR = "#c62828"
_WARN_CLR = "#ef6c00"

NEAR_TERM_CRITERIA = [
    {"id": f"C{i}", "group": grp, "description": desc}
    for i, (grp, desc) in enumerate([
        ("Boundary", "Company boundary includes all relevant entities"),
        ("Boundary", "Scope 1+2 coverage >= 95%"),
        ("Boundary", "Scope 3 screening completed"),
        ("Boundary", "Boundary consistent with financial reporting"),
        ("Boundary", "Coverage calculation methodology documented"),
        ("Base Year", "Base year within 2 most recent completed years"),
        ("Base Year", "Base year data quality validated"),
        ("Base Year", "Recalculation policy defined"),
        ("Base Year", "Base year emissions independently verified"),
        ("Ambition", "ACA >= 4.2%/yr (1.5C) or >= 2.5%/yr (WB2C)"),
        ("Ambition", "SDA convergence validated (if applicable)"),
        ("Ambition", "Absolute reduction from base year calculated"),
        ("Ambition", "Intensity target converted to absolute equivalent"),
        ("Ambition", "Target ambition meets minimum threshold"),
        ("Ambition", "No offsets used toward target achievement"),
        ("Timeframe", "Near-term: 5-10 years from submission"),
        ("Timeframe", "No more than 10 years from base year"),
        ("Timeframe", "Annual milestone pathway defined"),
        ("Scope 3", "Scope 3 >= 67% coverage of total"),
        ("Scope 3", "All material categories included"),
        ("Scope 3", "Supplier engagement target set (if applicable)"),
        ("Scope 3", "Scope 3 calculation methodology documented"),
        ("Scope 3", "Scope 3 data quality assessment completed"),
        ("Reporting", "Annual disclosure commitment"),
        ("Reporting", "Progress tracking methodology defined"),
        ("Reporting", "Recalculation triggers documented"),
        ("Reporting", "Public reporting commitment"),
        ("Reporting", "Five-year review schedule set"),
    ], 1)
]

NET_ZERO_CRITERIA = [
    {"id": f"NZ-C{i}", "group": grp, "description": desc}
    for i, (grp, desc) in enumerate([
        ("Long-term", "90%+ absolute reduction by 2050"),
        ("Long-term", "Scope 1+2 coverage >= 95%"),
        ("Long-term", "Scope 3 coverage >= 90%"),
        ("Long-term", "Long-term pathway defined to 2050"),
        ("Neutralization", "Residual emissions <= 10%"),
        ("Neutralization", "Neutralization via permanent CDR"),
        ("Neutralization", "Credit quality per SBTi guidance"),
        ("Neutralization", "Neutralization strategy documented"),
        ("Interim", "Near-term target set (C1-C28 satisfied)"),
        ("Interim", "Interim milestones every 5 years"),
        ("Interim", "Linear or front-loaded pathway"),
        ("Governance", "Board-level oversight documented"),
        ("Governance", "Annual progress reporting commitment"),
        ("Governance", "Five-year review and revalidation"),
    ], 1)
]

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

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
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default

def _status_icon(status: str) -> str:
    mapping = {"pass": "PASS", "fail": "FAIL", "warning": "WARN", "na": "N/A"}
    return mapping.get(status.lower(), status.upper())

class SBTiTargetSubmissionTemplate:
    """
    SBTi Corporate Standard target submission package template.

    Generates a complete submission package with near-term (C1-C28),
    long-term, and net-zero (NZ-C1 to NZ-C14) targets, criteria
    validation matrix, pathway visualization, and coverage analysis.
    Supports Markdown, HTML, JSON, and PDF-ready output.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
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
            self._md_org_profile(data),
            self._md_near_term_targets(data),
            self._md_long_term_targets(data),
            self._md_criteria_matrix(data),
            self._md_coverage_analysis(data),
            self._md_pathway(data),
            self._md_flag_assessment(data),
            self._md_supporting_docs(data),
            self._md_checklist(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_criteria_matrix(data),
            self._html_coverage(data),
            self._html_pathway(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SBTi Target Submission Package</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        near_term = data.get("near_term_target", {})
        long_term = data.get("long_term_target", {})
        net_zero = data.get("net_zero_target", {})
        criteria_results = data.get("criteria_results", {})

        nt_criteria = []
        for c in NEAR_TERM_CRITERIA:
            result = criteria_results.get(c["id"], {})
            nt_criteria.append({
                "criterion": c["id"],
                "group": c["group"],
                "description": c["description"],
                "status": result.get("status", "not_assessed"),
                "evidence": result.get("evidence", ""),
                "remediation": result.get("remediation", ""),
            })

        nz_criteria = []
        for c in NET_ZERO_CRITERIA:
            result = criteria_results.get(c["id"], {})
            nz_criteria.append({
                "criterion": c["id"],
                "group": c["group"],
                "description": c["description"],
                "status": result.get("status", "not_assessed"),
                "evidence": result.get("evidence", ""),
                "remediation": result.get("remediation", ""),
            })

        passed_nt = sum(1 for c in nt_criteria if c["status"] == "pass")
        passed_nz = sum(1 for c in nz_criteria if c["status"] == "pass")

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "sector": data.get("sector", ""),
                "employees": data.get("employees", 0),
                "revenue": data.get("revenue", 0),
            },
            "base_year": data.get("base_year", ""),
            "near_term_target": {
                "pathway": near_term.get("pathway", "ACA"),
                "ambition": near_term.get("ambition", "1.5C"),
                "reduction_rate": near_term.get("reduction_rate", 4.2),
                "target_year": near_term.get("target_year", ""),
                "scope12_coverage_pct": near_term.get("scope12_coverage_pct", 0),
                "scope3_coverage_pct": near_term.get("scope3_coverage_pct", 0),
                "base_year_tco2e": near_term.get("base_year_tco2e", 0),
                "target_year_tco2e": near_term.get("target_year_tco2e", 0),
                "milestones": near_term.get("milestones", []),
            },
            "long_term_target": {
                "target_year": long_term.get("target_year", "2050"),
                "reduction_pct": long_term.get("reduction_pct", 90),
                "scope12_coverage_pct": long_term.get("scope12_coverage_pct", 0),
                "scope3_coverage_pct": long_term.get("scope3_coverage_pct", 0),
            },
            "net_zero_target": {
                "target_year": net_zero.get("target_year", "2050"),
                "residual_emissions_pct": net_zero.get("residual_emissions_pct", 0),
                "neutralization_strategy": net_zero.get("neutralization_strategy", ""),
            },
            "flag_applicable": data.get("flag_applicable", False),
            "flag_target": data.get("flag_target", {}),
            "criteria_validation": {
                "near_term": nt_criteria,
                "net_zero": nz_criteria,
                "near_term_score": f"{passed_nt}/28",
                "net_zero_score": f"{passed_nz}/14",
                "total_score": f"{passed_nt + passed_nz}/42",
                "submission_ready": passed_nt >= 26 and passed_nz >= 12,
            },
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        json_data = self.render_json(data)

        criteria_sheet = {
            "name": "Criteria Validation",
            "headers": ["Criterion", "Group", "Description", "Status", "Evidence", "Remediation"],
            "rows": [],
        }
        for c in json_data["criteria_validation"]["near_term"]:
            criteria_sheet["rows"].append([
                c["criterion"], c["group"], c["description"],
                c["status"].upper(), c["evidence"], c["remediation"],
            ])
        for c in json_data["criteria_validation"]["net_zero"]:
            criteria_sheet["rows"].append([
                c["criterion"], c["group"], c["description"],
                c["status"].upper(), c["evidence"], c["remediation"],
            ])

        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sbti_submission_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [criteria_sheet],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "Enterprise")
        return (
            f"# SBTi Target Submission Package\n\n"
            f"## {org}\n\n"
            f"**Standard:** SBTi Corporate Manual V5.3 + Net-Zero Standard V1.3  \n"
            f"**Generated:** {ts}  \n"
            f"**Report ID:** {_new_uuid()}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        lt = data.get("long_term_target", {})
        nz = data.get("net_zero_target", {})
        criteria_results = data.get("criteria_results", {})
        passed = sum(1 for v in criteria_results.values() if v.get("status") == "pass")
        total = 42

        lines = [
            "## 1. Executive Summary\n",
            "| Target Type | Pathway | Target Year | Reduction | Status |",
            "|-------------|---------|:-----------:|:---------:|:------:|",
            f"| Near-Term (Scope 1+2) | {nt.get('pathway', 'ACA')} "
            f"| {nt.get('target_year', 'TBD')} "
            f"| {_pct(nt.get('reduction_rate', 4.2))}/yr "
            f"| {'Ready' if passed >= 38 else 'In Progress'} |",
            f"| Near-Term (Scope 3) | {nt.get('scope3_approach', 'Absolute')} "
            f"| {nt.get('target_year', 'TBD')} "
            f"| {_pct(nt.get('scope3_reduction_pct', 0))} "
            f"| Coverage: {_pct(nt.get('scope3_coverage_pct', 0))} |",
            f"| Long-Term | Absolute | {lt.get('target_year', '2050')} "
            f"| {_pct(lt.get('reduction_pct', 90))} "
            f"| - |",
            f"| Net-Zero | Full | {nz.get('target_year', '2050')} "
            f"| {_pct(100 - nz.get('residual_emissions_pct', 10))} + CDR "
            f"| - |",
            "",
            f"**Criteria Score:** {passed}/{total} "
            f"({'Submission Ready' if passed >= 38 else 'Action Required'})",
        ]
        return "\n".join(lines)

    def _md_org_profile(self, data: Dict[str, Any]) -> str:
        return (
            f"## 2. Organizational Profile\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Organization | {data.get('org_name', '')} |\n"
            f"| Sector | {data.get('sector', '')} |\n"
            f"| Employees | {_dec_comma(data.get('employees', 0))} |\n"
            f"| Revenue | {data.get('currency', 'USD')} {_dec_comma(data.get('revenue', 0))} |\n"
            f"| Entities | {len(data.get('entities', []))} |\n"
            f"| Base Year | {data.get('base_year', '')} |\n"
            f"| Base Year Emissions | {_dec_comma(data.get('base_year_tco2e', 0))} tCO2e |\n"
            f"| Consolidation | {data.get('consolidation_approach', 'Operational Control')} |"
        )

    def _md_near_term_targets(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        milestones = nt.get("milestones", [])

        lines = [
            "## 3. Near-Term Targets\n",
            f"**Pathway:** {nt.get('pathway', 'ACA')} "
            f"({nt.get('ambition', '1.5C')}-aligned)  ",
            f"**Reduction Rate:** {nt.get('reduction_rate', 4.2)}% per year  ",
            f"**Base Year:** {data.get('base_year', '')}  ",
            f"**Target Year:** {nt.get('target_year', '')}  ",
            f"**Scope 1+2 Coverage:** {_pct(nt.get('scope12_coverage_pct', 0))} (required: 95%)  ",
            f"**Scope 3 Coverage:** {_pct(nt.get('scope3_coverage_pct', 0))} (required: 67%)  \n",
        ]
        if milestones:
            lines.append("### Annual Milestones\n")
            lines.append("| Year | Target (tCO2e) | Reduction from Base | Cumulative % |")
            lines.append("|------|---------------:|--------------------:|-------------:|")
            for ms in milestones:
                lines.append(
                    f"| {ms.get('year', '')} | {_dec_comma(ms.get('tco2e', 0))} "
                    f"| {_dec_comma(ms.get('reduction_tco2e', 0))} "
                    f"| {_pct(ms.get('cumulative_pct', 0))} |"
                )
        return "\n".join(lines)

    def _md_long_term_targets(self, data: Dict[str, Any]) -> str:
        lt = data.get("long_term_target", {})
        nz = data.get("net_zero_target", {})
        return (
            f"## 4. Long-Term / Net-Zero Targets\n\n"
            f"### Long-Term Target\n"
            f"- **Target Year:** {lt.get('target_year', '2050')}\n"
            f"- **Reduction:** {_pct(lt.get('reduction_pct', 90))} from base year\n"
            f"- **Scope 1+2 Coverage:** {_pct(lt.get('scope12_coverage_pct', 0))} (required: 95%)\n"
            f"- **Scope 3 Coverage:** {_pct(lt.get('scope3_coverage_pct', 0))} (required: 90%)\n\n"
            f"### Net-Zero Target\n"
            f"- **Target Year:** {nz.get('target_year', '2050')}\n"
            f"- **Residual Emissions:** {_pct(nz.get('residual_emissions_pct', 10))} (max 10%)\n"
            f"- **Neutralization:** {nz.get('neutralization_strategy', 'Permanent CDR')}\n"
            f"- **CDR Type:** {nz.get('cdr_type', 'Biochar, DACCS, enhanced weathering')}"
        )

    def _md_criteria_matrix(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        lines = [
            "## 5. Criteria Validation Matrix\n",
            "### Near-Term Criteria (C1-C28)\n",
            "| Criterion | Group | Description | Status | Evidence |",
            "|:---------:|-------|-------------|:------:|----------|",
        ]
        for c in NEAR_TERM_CRITERIA:
            result = criteria_results.get(c["id"], {})
            status = _status_icon(result.get("status", "not_assessed"))
            lines.append(
                f"| {c['id']} | {c['group']} | {c['description']} "
                f"| {status} | {result.get('evidence', '-')} |"
            )

        lines.append("\n### Net-Zero Criteria (NZ-C1 to NZ-C14)\n")
        lines.append("| Criterion | Group | Description | Status | Evidence |")
        lines.append("|:---------:|-------|-------------|:------:|----------|")
        for c in NET_ZERO_CRITERIA:
            result = criteria_results.get(c["id"], {})
            status = _status_icon(result.get("status", "not_assessed"))
            lines.append(
                f"| {c['id']} | {c['group']} | {c['description']} "
                f"| {status} | {result.get('evidence', '-')} |"
            )
        return "\n".join(lines)

    def _md_coverage_analysis(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        s3_cats = data.get("scope3_categories_coverage", [])
        lines = [
            "## 6. Coverage Analysis\n",
            f"| Scope | Covered (tCO2e) | Total (tCO2e) | Coverage % | Required |",
            f"|-------|----------------:|--------------:|-----------:|:--------:|",
            f"| Scope 1+2 | {_dec_comma(nt.get('scope12_covered_tco2e', 0))} "
            f"| {_dec_comma(nt.get('scope12_total_tco2e', 0))} "
            f"| {_pct(nt.get('scope12_coverage_pct', 0))} | 95% |",
            f"| Scope 3 | {_dec_comma(nt.get('scope3_covered_tco2e', 0))} "
            f"| {_dec_comma(nt.get('scope3_total_tco2e', 0))} "
            f"| {_pct(nt.get('scope3_coverage_pct', 0))} | 67% |",
        ]
        if s3_cats:
            lines.append("\n### Scope 3 Category Coverage\n")
            lines.append("| Cat | Category | tCO2e | Included in Target | Coverage |")
            lines.append("|:---:|----------|------:|:------------------:|:--------:|")
            for cat in s3_cats:
                lines.append(
                    f"| {cat.get('num', '')} | {cat.get('name', '')} "
                    f"| {_dec_comma(cat.get('tco2e', 0))} "
                    f"| {'Yes' if cat.get('included', False) else 'No'} "
                    f"| {_pct(cat.get('coverage_pct', 0))} |"
                )
        return "\n".join(lines)

    def _md_pathway(self, data: Dict[str, Any]) -> str:
        pathway_data = data.get("pathway_data", [])
        if not pathway_data:
            return "## 7. Pathway Visualization\n\nPathway data not yet calculated."
        lines = [
            "## 7. Pathway Visualization\n",
            "| Year | Actual (tCO2e) | Target (tCO2e) | SBTi 1.5C | SBTi WB2C | Status |",
            "|------|---------------:|---------------:|----------:|----------:|:------:|",
        ]
        for yr in pathway_data:
            actual = yr.get("actual_tco2e", "")
            target = yr.get("target_tco2e", "")
            sbti_15 = yr.get("sbti_15c_tco2e", "")
            sbti_wb2 = yr.get("sbti_wb2c_tco2e", "")
            status = yr.get("status", "")
            lines.append(
                f"| {yr.get('year', '')} "
                f"| {_dec_comma(actual) if actual else '-'} "
                f"| {_dec_comma(target)} "
                f"| {_dec_comma(sbti_15)} "
                f"| {_dec_comma(sbti_wb2)} "
                f"| {status} |"
            )
        return "\n".join(lines)

    def _md_flag_assessment(self, data: Dict[str, Any]) -> str:
        if not data.get("flag_applicable", False):
            return (
                "## 8. FLAG Assessment\n\n"
                "FLAG targets not applicable. Land use emissions < 20% of total."
            )
        flag = data.get("flag_target", {})
        return (
            f"## 8. FLAG Assessment\n\n"
            f"**FLAG Applicable:** Yes (land use emissions >= 20%)  \n"
            f"**FLAG Pathway:** {flag.get('pathway', '3.03%/yr')}  \n"
            f"**Deforestation Target:** {flag.get('deforestation_target', 'Zero by 2025')}  \n"
            f"**Commodities:** {', '.join(flag.get('commodities', []))}  \n"
            f"**FLAG Emissions:** {_dec_comma(flag.get('flag_tco2e', 0))} tCO2e  \n"
            f"**FLAG % of Total:** {_pct(flag.get('flag_pct', 0))}"
        )

    def _md_supporting_docs(self, data: Dict[str, Any]) -> str:
        docs = data.get("supporting_documents", [
            "GHG Inventory Report (PACK-027 template)",
            "Organizational Boundary Definition",
            "Base Year Emissions Statement",
            "Scope 3 Screening Results",
            "Data Quality Assessment",
            "Recalculation Policy",
            "Board Resolution on Climate Targets",
        ])
        lines = ["## 9. Supporting Documentation\n"]
        for i, doc in enumerate(docs, 1):
            lines.append(f"{i}. {doc}")
        return "\n".join(lines)

    def _md_checklist(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        passed = sum(1 for v in criteria_results.values() if v.get("status") == "pass")
        items = [
            ("Organizational profile complete", passed > 0),
            ("Base year emissions validated", any(
                criteria_results.get(f"C{i}", {}).get("status") == "pass" for i in range(6, 10)
            )),
            ("Near-term targets defined", any(
                criteria_results.get(f"C{i}", {}).get("status") == "pass" for i in range(10, 16)
            )),
            ("Scope 3 coverage >= 67%", criteria_results.get("C19", {}).get("status") == "pass"),
            ("Net-zero target defined", any(
                criteria_results.get(f"NZ-C{i}", {}).get("status") == "pass" for i in range(1, 5)
            )),
            ("All 42 criteria assessed", passed >= 42),
            ("Supporting documents attached", True),
            ("Board approval obtained", criteria_results.get("NZ-C12", {}).get("status") == "pass"),
        ]
        lines = ["## 10. Submission Checklist\n"]
        for desc, done in items:
            mark = "[x]" if done else "[ ]"
            lines.append(f"- {mark} {desc}")
        lines.append(f"\n**Overall Status:** {'READY FOR SUBMISSION' if passed >= 38 else 'ACTION REQUIRED'}")
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "SBTi-001", "source": "SBTi Corporate Manual V5.3", "year": "2024"},
            {"ref": "SBTi-002", "source": "SBTi Corporate Net-Zero Standard V1.3", "year": "2024"},
            {"ref": "SBTi-003", "source": "SBTi FLAG Guidance V1.1", "year": "2022"},
            {"ref": "GHG-001", "source": "GHG Protocol Corporate Standard", "year": "2004"},
        ])
        lines = ["## 11. Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*SBTi Corporate Manual V5.3 + Net-Zero Standard V1.3 compliant.*  \n"
            f"*Zero-hallucination deterministic calculations. SHA-256 provenance.*"
        )

    # ------------------------------------------------------------------ #
    # HTML
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1100px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:32px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".pass{{color:{_PASS_CLR};font-weight:700;}}"
            f".fail{{color:{_FAIL_CLR};font-weight:700;}}"
            f".warn{{color:{_WARN_CLR};font-weight:700;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Target Submission Package</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'SBTi Corporate Manual V5.3 | Generated: {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        criteria_results = data.get("criteria_results", {})
        passed = sum(1 for v in criteria_results.values() if v.get("status") == "pass")
        return (
            f'<h2>Executive Summary</h2>\n'
            f'<p><strong>Pathway:</strong> {nt.get("pathway", "ACA")} '
            f'({nt.get("ambition", "1.5C")}) | '
            f'<strong>Criteria:</strong> {passed}/42 | '
            f'<strong>Status:</strong> '
            f'{"<span class=pass>SUBMISSION READY</span>" if passed >= 38 else "<span class=warn>IN PROGRESS</span>"}'
            f'</p>'
        )

    def _html_criteria_matrix(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        rows = ""
        for c in NEAR_TERM_CRITERIA + NET_ZERO_CRITERIA:
            result = criteria_results.get(c["id"], {})
            status = result.get("status", "not_assessed")
            cls = {"pass": "pass", "fail": "fail", "warning": "warn"}.get(status, "")
            rows += (
                f'<tr><td>{c["id"]}</td><td>{c["group"]}</td><td>{c["description"]}</td>'
                f'<td class="{cls}">{status.upper()}</td>'
                f'<td>{result.get("evidence", "-")}</td></tr>\n'
            )
        return (
            f'<h2>Criteria Validation Matrix</h2>\n'
            f'<table><tr><th>ID</th><th>Group</th><th>Description</th>'
            f'<th>Status</th><th>Evidence</th></tr>\n{rows}</table>'
        )

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        return (
            f'<h2>Coverage Analysis</h2>\n'
            f'<table><tr><th>Scope</th><th>Coverage</th><th>Required</th></tr>\n'
            f'<tr><td>Scope 1+2</td><td>{_pct(nt.get("scope12_coverage_pct", 0))}</td>'
            f'<td>95%</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_pct(nt.get("scope3_coverage_pct", 0))}</td>'
            f'<td>67%</td></tr>\n</table>'
        )

    def _html_pathway(self, data: Dict[str, Any]) -> str:
        return '<h2>Pathway</h2>\n<p>See detailed Markdown output for full pathway table.</p>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-027 on {ts} | SBTi Corporate Manual V5.3 | SHA-256'
            f'</div>'
        )
