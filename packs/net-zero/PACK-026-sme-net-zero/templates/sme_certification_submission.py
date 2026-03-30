# -*- coding: utf-8 -*-
"""
SMECertificationSubmissionTemplate - Certification-specific formats for PACK-026.

Renders pre-filled certification submission documents for SME Climate Hub,
B Corp Climate Collective, ISO 14001, and Carbon Trust certifications.
Includes commitment letters, impact assessments, management system docs,
reduction plans, evidence links, and submission checklists.

Sections:
    1. SME Climate Hub Submission
    2. B Corp Climate Collective Data
    3. ISO 14001 Environmental Management Summary
    4. Carbon Trust Footprint Statement
    5. Supporting Evidence Links
    6. Submission Checklist

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
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

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"
_TEMPLATE_ID = "sme_certification_submission"

_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

# ---------------------------------------------------------------------------
# Certification bodies
# ---------------------------------------------------------------------------
_CERT_BODIES = {
    "sme_climate_hub": {
        "name": "SME Climate Hub",
        "description": "UN Race to Zero-backed commitment for SMEs",
        "requirements": [
            "Halve emissions by 2030",
            "Reach net zero by 2050",
            "Measure and disclose emissions annually",
            "Publish a climate action plan",
        ],
    },
    "b_corp_climate": {
        "name": "B Corp Climate Collective",
        "description": "B Corporation climate impact assessment",
        "requirements": [
            "Complete climate impact assessment",
            "Set science-based targets",
            "Report annual progress",
            "Engage supply chain on climate",
        ],
    },
    "iso_14001": {
        "name": "ISO 14001:2015",
        "description": "Environmental management system standard",
        "requirements": [
            "Environmental policy document",
            "Significant aspects register",
            "Legal compliance register",
            "Objectives and targets",
            "Monitoring and measurement procedures",
            "Management review records",
        ],
    },
    "carbon_trust": {
        "name": "Carbon Trust Standard",
        "description": "Carbon footprint measurement and reduction",
        "requirements": [
            "Verified carbon footprint",
            "Year-on-year reduction achieved",
            "Carbon management plan in place",
            "Senior management commitment",
        ],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

# ===========================================================================
# Template Class
# ===========================================================================

class SMECertificationSubmissionTemplate:
    """
    SME certification submission template for multiple certification bodies.

    Renders pre-filled certification documents for SME Climate Hub,
    B Corp Climate Collective, ISO 14001, and Carbon Trust across
    Markdown, HTML, JSON, and Excel formats with supporting evidence
    links and submission checklists.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json", "excel"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the certification submission as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_sme_climate_hub(data),
            self._md_b_corp(data),
            self._md_iso_14001(data),
            self._md_carbon_trust(data),
            self._md_evidence_links(data),
            self._md_submission_checklist(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the certification submission as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_sme_climate_hub(data),
            self._html_b_corp(data),
            self._html_iso_14001(data),
            self._html_carbon_trust(data),
            self._html_evidence_links(data),
            self._html_submission_checklist(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME Certification Submission</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the certification submission as structured JSON."""
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "registration_number": data.get("registration_number", ""),
                "sector": data.get("sector", ""),
                "employees": data.get("employees", 0),
                "address": data.get("address", ""),
                "signatory": data.get("signatory", ""),
                "signatory_title": data.get("signatory_title", ""),
            },
            "emissions": {
                "baseline_year": data.get("baseline_year", ""),
                "reporting_year": data.get("reporting_year", ""),
                "total_tco2e": round(total, 2),
                "scope1_tco2e": round(s1, 2),
                "scope2_tco2e": round(s2, 2),
                "scope3_tco2e": round(s3, 2),
            },
            "targets": {
                "target_2030_pct": data.get("target_2030_pct", 50),
                "target_2050": data.get("target_2050", "Net Zero"),
                "framework": data.get("target_framework", "SBTi SME"),
            },
            "certifications": {
                "sme_climate_hub": {
                    "eligible": True,
                    "commitment_date": data.get("commitment_date", ""),
                    "fields": self._sme_hub_fields(data),
                },
                "b_corp_climate": {
                    "eligible": True,
                    "fields": self._b_corp_fields(data),
                },
                "iso_14001": {
                    "eligible": True,
                    "fields": self._iso_fields(data),
                },
                "carbon_trust": {
                    "eligible": True,
                    "yoy_reduction": data.get("yoy_reduction_pct", 0),
                    "fields": self._ct_fields(data),
                },
            },
            "evidence": data.get("evidence_links", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Excel-ready data structure for certification submissions."""
        self.generated_at = utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3

        org_sheet = {
            "name": "Organization Profile",
            "headers": ["Field", "Value"],
            "rows": [
                ["Organization Name", data.get("org_name", "")],
                ["Registration Number", data.get("registration_number", "")],
                ["Sector", data.get("sector", "")],
                ["Employees", data.get("employees", 0)],
                ["Address", data.get("address", "")],
                ["Signatory", data.get("signatory", "")],
                ["Title", data.get("signatory_title", "")],
                ["Baseline Year", data.get("baseline_year", "")],
                ["Reporting Year", data.get("reporting_year", "")],
                ["Total Emissions", f"{round(total, 2)} tCO2e"],
                ["Scope 1", f"{round(s1, 2)} tCO2e"],
                ["Scope 2", f"{round(s2, 2)} tCO2e"],
                ["Scope 3", f"{round(s3, 2)} tCO2e"],
                ["2030 Target", f"{data.get('target_2030_pct', 50)}% reduction"],
                ["2050 Target", data.get("target_2050", "Net Zero")],
            ],
        }

        checklist_sheet = {
            "name": "Submission Checklist",
            "headers": ["Certification", "Requirement", "Status", "Evidence"],
            "rows": [],
        }
        for cert_key, cert_info in _CERT_BODIES.items():
            for req in cert_info["requirements"]:
                checklist_sheet["rows"].append([
                    cert_info["name"],
                    req,
                    "Pending",
                    "",
                ])

        evidence_sheet = {
            "name": "Evidence Links",
            "headers": ["Document", "Type", "Location", "Status"],
            "rows": [],
        }
        for ev in data.get("evidence_links", []):
            evidence_sheet["rows"].append([
                ev.get("document", ""),
                ev.get("type", ""),
                ev.get("location", ""),
                ev.get("status", "Available"),
            ])

        actions_sheet = {
            "name": "Action Plan",
            "headers": ["Action", "Category", "Timeline", "Reduction (tCO2e)", "Status"],
            "rows": [],
        }
        for a in data.get("action_plan", []):
            actions_sheet["rows"].append([
                a.get("action", ""),
                a.get("category", ""),
                a.get("timeline", ""),
                a.get("reduction_tco2e", 0),
                a.get("status", "Planned"),
            ])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sme_certification_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [org_sheet, checklist_sheet, evidence_sheet, actions_sheet],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    #  Pre-fill data helpers                                               #
    # ------------------------------------------------------------------ #

    def _sme_hub_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        return {
            "company_name": data.get("org_name", ""),
            "sector": data.get("sector", ""),
            "employee_count": data.get("employees", 0),
            "baseline_year": data.get("baseline_year", ""),
            "scope1_tco2e": round(s1, 2),
            "scope2_tco2e": round(s2, 2),
            "scope3_tco2e": round(s3, 2),
            "target_2030": f"{data.get('target_2030_pct', 50)}% reduction",
            "target_2050": "Net Zero",
            "top_3_actions": [a.get("name", "") for a in data.get("action_plan", [])[:3]],
        }

    def _b_corp_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "company_name": data.get("org_name", ""),
            "ghg_measured": True,
            "targets_set": True,
            "reduction_achieved": data.get("yoy_reduction_pct", 0),
            "supply_chain_engaged": data.get("supply_chain_engaged", False),
            "renewable_energy_pct": data.get("renewable_pct", 0),
        }

    def _iso_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "environmental_policy": data.get("has_env_policy", False),
            "aspects_register": data.get("has_aspects_register", False),
            "legal_register": data.get("has_legal_register", False),
            "objectives_targets": data.get("has_objectives", False),
            "monitoring_procedures": data.get("has_monitoring", False),
            "management_review": data.get("has_mgmt_review", False),
        }

    def _ct_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "company_name": data.get("org_name", ""),
            "footprint_verified": data.get("footprint_verified", False),
            "yoy_reduction": data.get("yoy_reduction_pct", 0),
            "management_plan": data.get("has_management_plan", False),
            "senior_commitment": data.get("has_senior_commitment", False),
        }

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Certification Submission Package\n\n"
            f"**Organization:** {data.get('org_name', 'Your Company')}  \n"
            f"**Prepared By:** {data.get('signatory', '')} ({data.get('signatory_title', '')})  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_sme_climate_hub(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        org = data.get("org_name", "Our Company")
        actions = data.get("action_plan", [])

        lines = [
            "## 1. SME Climate Hub Commitment\n",
            f"### Commitment Letter\n",
            f"As the authorised representative of **{org}**, I commit our organization to:\n",
            f"1. **Halve our greenhouse gas emissions by 2030**",
            f"2. **Achieve net zero emissions by 2050 at the latest**",
            f"3. **Measure and disclose our emissions annually**",
            f"4. **Implement our climate action plan**\n",
            f"### Baseline Statement\n",
            f"| Metric | Value |",
            f"|--------|------:|",
            f"| Baseline Year | {data.get('baseline_year', '')} |",
            f"| Total Emissions | {_dec_comma(total)} tCO2e |",
            f"| Scope 1 | {_dec_comma(s1)} tCO2e |",
            f"| Scope 2 | {_dec_comma(s2)} tCO2e |",
            f"| Scope 3 | {_dec_comma(s3)} tCO2e |",
            f"| Employees | {_dec_comma(data.get('employees', 0))} |",
            f"| Sector | {data.get('sector', '')} |\n",
            f"### Climate Action Plan\n",
        ]
        for idx, a in enumerate(actions[:5], 1):
            lines.append(f"{idx}. **{a.get('action', '')}** - {a.get('timeline', '')} "
                         f"({_dec_comma(a.get('reduction_tco2e', 0))} tCO2e)")

        lines.append(f"\n**Signed:** _________________________  \n"
                     f"**Name:** {data.get('signatory', '')}  \n"
                     f"**Title:** {data.get('signatory_title', '')}  \n"
                     f"**Date:** {data.get('commitment_date', '')}")

        return "\n".join(lines)

    def _md_b_corp(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 2. B Corp Climate Collective\n",
            "### Impact Assessment Data\n",
            f"| Metric | Value | Status |",
            f"|--------|------:|:------:|",
            f"| GHG Emissions Measured | Yes | Complete |",
            f"| Science-Based Targets Set | Yes | Complete |",
            f"| YoY Reduction Achieved | {_pct(data.get('yoy_reduction_pct', 0))} | "
            f"{'Met' if float(data.get('yoy_reduction_pct', 0)) > 0 else 'Pending'} |",
            f"| Supply Chain Engaged | {'Yes' if data.get('supply_chain_engaged') else 'No'} | "
            f"{'Complete' if data.get('supply_chain_engaged') else 'In Progress'} |",
            f"| Renewable Energy | {_pct(data.get('renewable_pct', 0))} | |",
        ]
        return "\n".join(lines)

    def _md_iso_14001(self, data: Dict[str, Any]) -> str:
        checks = [
            ("Environmental Policy", data.get("has_env_policy", False)),
            ("Significant Aspects Register", data.get("has_aspects_register", False)),
            ("Legal Compliance Register", data.get("has_legal_register", False)),
            ("Objectives & Targets", data.get("has_objectives", False)),
            ("Monitoring Procedures", data.get("has_monitoring", False)),
            ("Management Review Records", data.get("has_mgmt_review", False)),
        ]

        lines = [
            "## 3. ISO 14001 Readiness\n",
            "### Environmental Management System Status\n",
            "| Requirement | Status |",
            "|-------------|:------:|",
        ]
        for name, status in checks:
            mark = "[x]" if status else "[ ]"
            lines.append(f"| {name} | {mark} {'Complete' if status else 'Pending'} |")

        completed = sum(1 for _, s in checks if s)
        lines.append(f"\n**Readiness:** {completed}/{len(checks)} requirements met "
                     f"({_pct(_safe_div(completed, len(checks)) * 100)})")

        return "\n".join(lines)

    def _md_carbon_trust(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3

        lines = [
            "## 4. Carbon Trust Standard\n",
            "### Carbon Footprint Statement\n",
            f"| Metric | Value |",
            f"|--------|------:|",
            f"| Reporting Year | {data.get('reporting_year', '')} |",
            f"| Total Footprint | {_dec_comma(total)} tCO2e |",
            f"| Year-on-Year Reduction | {_pct(data.get('yoy_reduction_pct', 0))} |",
            f"| Verification Status | {data.get('verification_status', 'Self-assessed')} |",
            f"| Management Plan | {'Yes' if data.get('has_management_plan') else 'No'} |",
            f"| Senior Commitment | {'Yes' if data.get('has_senior_commitment') else 'No'} |\n",
            f"### Reduction Plan Summary\n",
        ]
        for a in data.get("action_plan", [])[:5]:
            lines.append(f"- {a.get('action', '')}: {_dec_comma(a.get('reduction_tco2e', 0))} tCO2e")

        return "\n".join(lines)

    def _md_evidence_links(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_links", [])
        if not evidence:
            return ""

        lines = [
            "## 5. Supporting Evidence\n",
            "| Document | Type | Location | Status |",
            "|----------|------|----------|:------:|",
        ]
        for ev in evidence:
            lines.append(
                f"| {ev.get('document', '')} "
                f"| {ev.get('type', '')} "
                f"| {ev.get('location', '')} "
                f"| {ev.get('status', 'Available')} |"
            )
        return "\n".join(lines)

    def _md_submission_checklist(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 6. Submission Checklist\n",
            "### SME Climate Hub",
            "- [ ] Commitment letter signed",
            "- [ ] Baseline emissions calculated",
            "- [ ] Action plan documented",
            "- [ ] Registration on smeclimatehub.org\n",
            "### B Corp Climate Collective",
            "- [ ] Impact assessment completed",
            "- [ ] Targets registered",
            "- [ ] Annual data submitted\n",
            "### ISO 14001",
            "- [ ] Environmental policy approved",
            "- [ ] Aspects register complete",
            "- [ ] Internal audit conducted",
            "- [ ] Certification body selected\n",
            "### Carbon Trust",
            "- [ ] Footprint calculated and verified",
            "- [ ] Reduction evidence compiled",
            "- [ ] Management plan documented",
            "- [ ] Application submitted",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Pre-filled certification submission package for SMEs.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:900px;margin:0 auto;background:#fff;padding:32px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:#388e3c;margin-top:16px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".cert-section{{background:{_LIGHTER};border:1px solid {_CARD_BG};"
            f"border-left:5px solid {_ACCENT};border-radius:0 10px 10px 0;"
            f"padding:16px;margin:16px 0;}}"
            f".cert-name{{font-size:1.1em;font-weight:700;color:{_PRIMARY};}}"
            f".cert-desc{{font-size:0.85em;color:#689f38;margin-bottom:8px;}}"
            f".check-item{{padding:4px 0;font-size:0.9em;}}"
            f".check-done{{color:{_PRIMARY};font-weight:600;}}"
            f".check-pending{{color:#ff9800;}}"
            f".sig-block{{background:{_LIGHT};border:2px solid {_CARD_BG};"
            f"border-radius:8px;padding:16px;margin:12px 0;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Certification Submission Package</h1>\n'
            f'<p><strong>{data.get("org_name", "Your Company")}</strong> | '
            f'{data.get("signatory", "")} ({data.get("signatory_title", "")}) | '
            f'Generated: {ts}</p>'
        )

    def _html_sme_climate_hub(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3
        org = data.get("org_name", "Our Company")

        actions_html = ""
        for idx, a in enumerate(data.get("action_plan", [])[:5], 1):
            actions_html += (
                f'<div class="check-item">{idx}. <strong>{a.get("action", "")}</strong> - '
                f'{a.get("timeline", "")} ({_dec_comma(a.get("reduction_tco2e", 0))} tCO2e)</div>\n'
            )

        return (
            f'<div class="cert-section">\n'
            f'<h2>1. SME Climate Hub Commitment</h2>\n'
            f'<div class="cert-name">SME Climate Hub</div>\n'
            f'<div class="cert-desc">UN Race to Zero-backed commitment for SMEs</div>\n'
            f'<h3>Commitment Letter</h3>\n'
            f'<p>As the authorised representative of <strong>{org}</strong>, '
            f'I commit our organization to:</p>\n'
            f'<div class="check-item check-done">1. Halve emissions by 2030</div>\n'
            f'<div class="check-item check-done">2. Achieve net zero by 2050</div>\n'
            f'<div class="check-item check-done">3. Measure and disclose annually</div>\n'
            f'<div class="check-item check-done">4. Implement climate action plan</div>\n'
            f'<h3>Baseline Statement</h3>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Total Emissions</td><td>{_dec_comma(total)} tCO2e</td></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)} tCO2e</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec_comma(s2)} tCO2e</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)} tCO2e</td></tr>\n'
            f'</table>\n'
            f'<h3>Action Plan</h3>\n{actions_html}\n'
            f'<div class="sig-block">\n'
            f'  <p>Signed: _________________________</p>\n'
            f'  <p>Name: {data.get("signatory", "")} | '
            f'Title: {data.get("signatory_title", "")} | '
            f'Date: {data.get("commitment_date", "")}</p>\n'
            f'</div>\n'
            f'</div>'
        )

    def _html_b_corp(self, data: Dict[str, Any]) -> str:
        return (
            f'<div class="cert-section">\n'
            f'<h2>2. B Corp Climate Collective</h2>\n'
            f'<div class="cert-name">B Corp Climate Collective</div>\n'
            f'<div class="cert-desc">B Corporation climate impact assessment</div>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th><th>Status</th></tr>\n'
            f'<tr><td>GHG Measured</td><td>Yes</td><td class="check-done">Complete</td></tr>\n'
            f'<tr><td>Targets Set</td><td>Yes</td><td class="check-done">Complete</td></tr>\n'
            f'<tr><td>YoY Reduction</td><td>{_pct(data.get("yoy_reduction_pct", 0))}</td>'
            f'<td>{"Met" if float(data.get("yoy_reduction_pct", 0)) > 0 else "Pending"}</td></tr>\n'
            f'<tr><td>Supply Chain</td>'
            f'<td>{"Yes" if data.get("supply_chain_engaged") else "No"}</td>'
            f'<td>{"Complete" if data.get("supply_chain_engaged") else "In Progress"}</td></tr>\n'
            f'<tr><td>Renewable Energy</td><td>{_pct(data.get("renewable_pct", 0))}</td>'
            f'<td>-</td></tr>\n'
            f'</table>\n'
            f'</div>'
        )

    def _html_iso_14001(self, data: Dict[str, Any]) -> str:
        checks = [
            ("Environmental Policy", data.get("has_env_policy", False)),
            ("Aspects Register", data.get("has_aspects_register", False)),
            ("Legal Register", data.get("has_legal_register", False)),
            ("Objectives & Targets", data.get("has_objectives", False)),
            ("Monitoring Procedures", data.get("has_monitoring", False)),
            ("Management Review", data.get("has_mgmt_review", False)),
        ]
        completed = sum(1 for _, s in checks if s)
        pct = _safe_div(completed, len(checks)) * 100

        rows = ""
        for name, status in checks:
            cls = "check-done" if status else "check-pending"
            rows += (
                f'<tr><td>{name}</td>'
                f'<td class="{cls}">{"Complete" if status else "Pending"}</td></tr>\n'
            )

        return (
            f'<div class="cert-section">\n'
            f'<h2>3. ISO 14001 Readiness</h2>\n'
            f'<div class="cert-name">ISO 14001:2015</div>\n'
            f'<div class="cert-desc">Readiness: {completed}/{len(checks)} ({_pct(pct)})</div>\n'
            f'<table><tr><th>Requirement</th><th>Status</th></tr>\n{rows}</table>\n'
            f'</div>'
        )

    def _html_carbon_trust(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_tco2e", 0))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2 + s3

        return (
            f'<div class="cert-section">\n'
            f'<h2>4. Carbon Trust Standard</h2>\n'
            f'<div class="cert-name">Carbon Trust Standard</div>\n'
            f'<div class="cert-desc">Carbon footprint measurement and reduction</div>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Total Footprint</td><td>{_dec_comma(total)} tCO2e</td></tr>\n'
            f'<tr><td>YoY Reduction</td><td>{_pct(data.get("yoy_reduction_pct", 0))}</td></tr>\n'
            f'<tr><td>Verified</td><td>{data.get("verification_status", "Self-assessed")}</td></tr>\n'
            f'<tr><td>Management Plan</td>'
            f'<td>{"Yes" if data.get("has_management_plan") else "No"}</td></tr>\n'
            f'<tr><td>Senior Commitment</td>'
            f'<td>{"Yes" if data.get("has_senior_commitment") else "No"}</td></tr>\n'
            f'</table>\n'
            f'</div>'
        )

    def _html_evidence_links(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_links", [])
        if not evidence:
            return ""

        rows = ""
        for ev in evidence:
            rows += (
                f'<tr><td>{ev.get("document", "")}</td>'
                f'<td>{ev.get("type", "")}</td>'
                f'<td>{ev.get("location", "")}</td>'
                f'<td>{ev.get("status", "Available")}</td></tr>\n'
            )

        return (
            f'<h2>5. Supporting Evidence</h2>\n'
            f'<table>\n'
            f'<tr><th>Document</th><th>Type</th><th>Location</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_submission_checklist(self, data: Dict[str, Any]) -> str:
        certs = ["SME Climate Hub", "B Corp Climate Collective", "ISO 14001", "Carbon Trust"]
        items_html = ""
        for cert in certs:
            cert_info = None
            for k, v in _CERT_BODIES.items():
                if v["name"].startswith(cert.split()[0]):
                    cert_info = v
                    break
            if cert_info:
                items_html += f'<h3>{cert_info["name"]}</h3>\n'
                for req in cert_info["requirements"]:
                    items_html += f'<div class="check-item check-pending">{req}</div>\n'

        return f'<h2>6. Submission Checklist</h2>\n{items_html}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'Pre-filled certification submission package for SMEs'
            f'</div>'
        )
