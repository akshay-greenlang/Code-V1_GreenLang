# -*- coding: utf-8 -*-
"""
CampaignSubmissionPackageTemplate - Race to Zero submission bundle for PACK-025.

Renders the complete Race to Zero campaign submission package including
pledge commitment letter, starting line compliance proof, action plan
summary, verification schedule, contact information, and submission
confirmation.

Sections:
    1. Submission Cover Page
    2. Pledge Commitment Summary
    3. Starting Line Compliance Proof
    4. Action Plan Summary
    5. Emissions Baseline Summary
    6. Target Summary
    7. Verification Schedule
    8. Contact Information
    9. Submission Confirmation & Checklist

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "campaign_submission_package"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0


# Submission checklist items
SUBMISSION_CHECKLIST = [
    {"id": "SC-01", "item": "Pledge commitment letter signed by authorized representative",
     "category": "Pledge", "required": True},
    {"id": "SC-02", "item": "Starting line criteria self-assessment completed",
     "category": "Pledge", "required": True},
    {"id": "SC-03", "item": "Partner initiative membership confirmed",
     "category": "Pledge", "required": True},
    {"id": "SC-04", "item": "GHG baseline inventory completed (S1+S2+S3)",
     "category": "Baseline", "required": True},
    {"id": "SC-05", "item": "Base year documentation and methodology",
     "category": "Baseline", "required": True},
    {"id": "SC-06", "item": "Interim target (2030) with pathway alignment",
     "category": "Targets", "required": True},
    {"id": "SC-07", "item": "Long-term net-zero target (by 2050 at latest)",
     "category": "Targets", "required": True},
    {"id": "SC-08", "item": "Action plan / transition plan summary",
     "category": "Plan", "required": True},
    {"id": "SC-09", "item": "Governance structure documentation",
     "category": "Plan", "required": True},
    {"id": "SC-10", "item": "Verification schedule and commitments",
     "category": "Verification", "required": True},
    {"id": "SC-11", "item": "Primary contact for Race to Zero communication",
     "category": "Contact", "required": True},
    {"id": "SC-12", "item": "Authorization to publicly list organization",
     "category": "Authorization", "required": True},
    {"id": "SC-13", "item": "Offset policy statement (no offsets for interim targets)",
     "category": "Policy", "required": False},
    {"id": "SC-14", "item": "Just transition commitment statement",
     "category": "Policy", "required": False},
    {"id": "SC-15", "item": "Previous progress reports (if renewal)",
     "category": "Reporting", "required": False},
]


class CampaignSubmissionPackageTemplate:
    """Race to Zero campaign submission package template for PACK-025.

    Generates the complete submission bundle for Race to Zero campaign
    registration including all required documentation, compliance proof,
    and submission confirmation checklist.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    CHECKLIST = SUBMISSION_CHECKLIST

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the submission package as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_cover(data),
            self._md_pledge_summary(data),
            self._md_starting_line(data),
            self._md_action_plan_summary(data),
            self._md_baseline_summary(data),
            self._md_target_summary(data),
            self._md_verification(data),
            self._md_contact(data),
            self._md_submission_checklist(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the submission package as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_cover(data),
            self._html_pledge(data),
            self._html_starting_line(data),
            self._html_baseline(data),
            self._html_targets(data),
            self._html_verification(data),
            self._html_contact(data),
            self._html_checklist(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Campaign Submission Package</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the submission package as structured JSON."""
        self.generated_at = _utcnow()
        checklist_status = data.get("checklist_status", {})
        completed = sum(1 for v in checklist_status.values() if v.get("completed", False))
        required_items = [c for c in SUBMISSION_CHECKLIST if c["required"]]
        required_completed = sum(
            1 for c in required_items
            if checklist_status.get(c["id"], {}).get("completed", False)
        )

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "submission_id": data.get("submission_id", _new_uuid()),
            "org_name": data.get("org_name", ""),
            "partner_initiative": data.get("partner_initiative", ""),
            "submission_date": data.get("submission_date", ""),
            "submission_type": data.get("submission_type", "Initial Registration"),
            "pledge": {
                "campaign": "Race to Zero",
                "interim_target_year": data.get("interim_target", {}).get("year", 2030),
                "interim_reduction_pct": data.get("interim_target", {}).get("reduction_pct", 50),
                "net_zero_year": data.get("longterm_target", {}).get("year", 2050),
            },
            "baseline": data.get("baseline", {}),
            "checklist": {
                "total_items": len(SUBMISSION_CHECKLIST),
                "completed": completed,
                "required_total": len(required_items),
                "required_completed": required_completed,
                "is_submission_ready": required_completed >= len(required_items),
            },
            "contact": data.get("contact", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = _utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Submission Summary
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        sheets["Submission Summary"] = [
            {"Field": "Organization", "Value": data.get("org_name", "")},
            {"Field": "Partner Initiative", "Value": data.get("partner_initiative", "")},
            {"Field": "Submission Type", "Value": data.get("submission_type", "Initial Registration")},
            {"Field": "Submission Date", "Value": data.get("submission_date", "")},
            {"Field": "Baseline Emissions (tCO2e)", "Value": baseline.get("total_tco2e", 0)},
            {"Field": "Base Year", "Value": baseline.get("year", "")},
            {"Field": "Interim Target", "Value": f"{interim.get('reduction_pct', 50)}% by {interim.get('year', 2030)}"},
            {"Field": "Net-Zero Target", "Value": f"Net-zero by {longterm.get('year', 2050)}"},
        ]

        # Sheet 2: Submission Checklist
        checklist_status = data.get("checklist_status", {})
        cl_rows: List[Dict[str, Any]] = []
        for item in SUBMISSION_CHECKLIST:
            status = checklist_status.get(item["id"], {})
            cl_rows.append({
                "ID": item["id"],
                "Item": item["item"],
                "Category": item["category"],
                "Required": "Yes" if item["required"] else "No",
                "Completed": "Yes" if status.get("completed", False) else "No",
                "Document Ref": status.get("document_ref", ""),
                "Notes": status.get("notes", ""),
            })
        sheets["Submission Checklist"] = cl_rows

        # Sheet 3: Baseline Emissions
        sheets["Baseline Emissions"] = [
            {"Scope": "Scope 1", "Emissions (tCO2e)": baseline.get("scope1_tco2e", 0)},
            {"Scope": "Scope 2", "Emissions (tCO2e)": baseline.get("scope2_tco2e", 0)},
            {"Scope": "Scope 3", "Emissions (tCO2e)": baseline.get("scope3_tco2e", 0)},
            {"Scope": "Total", "Emissions (tCO2e)": baseline.get("total_tco2e", 0)},
        ]

        # Sheet 4: Contact Information
        contact = data.get("contact", {})
        sheets["Contact Information"] = [
            {"Field": "Primary Contact", "Value": contact.get("primary_name", "")},
            {"Field": "Title", "Value": contact.get("primary_title", "")},
            {"Field": "Email", "Value": contact.get("primary_email", "")},
            {"Field": "Phone", "Value": contact.get("primary_phone", "")},
            {"Field": "Secondary Contact", "Value": contact.get("secondary_name", "")},
            {"Field": "Secondary Email", "Value": contact.get("secondary_email", "")},
        ]

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_cover(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        submission_id = data.get("submission_id", _new_uuid())
        return (
            f"# Race to Zero -- Campaign Submission Package\n\n"
            f"**Organization:** {org}  \n"
            f"**Partner Initiative:** {data.get('partner_initiative', '')}  \n"
            f"**Submission Type:** {data.get('submission_type', 'Initial Registration')}  \n"
            f"**Submission ID:** {submission_id}  \n"
            f"**Date:** {ts}\n\n---"
        )

    def _md_pledge_summary(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        baseline = data.get("baseline", {})

        return (
            f"## 1. Pledge Commitment Summary\n\n"
            f"**{org}** hereby commits to the Race to Zero campaign and pledges to:\n\n"
            f"- Reach **net-zero GHG emissions** by **{longterm.get('year', 2050)}** at the latest\n"
            f"- Achieve **{_pct(interim.get('reduction_pct', 50))} reduction** "
            f"by **{interim.get('year', 2030)}** as an interim target\n"
            f"- Report progress annually and submit to independent verification\n"
            f"- Cover all emission scopes (Scope 1, 2, and 3)\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Baseline Emissions | {_dec_comma(baseline.get('total_tco2e', 0))} tCO2e |\n"
            f"| Base Year | {baseline.get('year', 'N/A')} |\n"
            f"| Methodology | {baseline.get('methodology', 'GHG Protocol')} |\n"
            f"| Partner Initiative | {data.get('partner_initiative', 'N/A')} |\n"
            f"| Signatory | {data.get('signatory', 'N/A')} |"
        )

    def _md_starting_line(self, data: Dict[str, Any]) -> str:
        sl = data.get("starting_line", {})
        criteria_met = sl.get("criteria_met", 0)
        total_criteria = sl.get("total_criteria", 20)
        compliance_pct = _safe_div(criteria_met, total_criteria) * 100

        lines = [
            "## 2. Starting Line Compliance Proof\n",
            f"**Compliance Status:** {criteria_met}/{total_criteria} criteria met ({_pct(compliance_pct)})\n",
        ]

        # Category summary
        categories = sl.get("categories", {})
        if categories:
            lines.extend([
                "| Category | Met | Total | Status |",
                "|----------|:---:|:-----:|:------:|",
            ])
            for cat_name, cat_data in categories.items():
                met = cat_data.get("met", 0)
                total = cat_data.get("total", 0)
                status = "PASS" if met >= total else "GAPS"
                lines.append(f"| {cat_name} | {met} | {total} | {status} |")
        else:
            lines.extend([
                "| Category | Status |",
                "|----------|:------:|",
                f"| Pledge | {'PASS' if sl.get('pledge_met', False) else 'PENDING'} |",
                f"| Plan | {'PASS' if sl.get('plan_met', False) else 'PENDING'} |",
                f"| Proceed | {'PASS' if sl.get('proceed_met', False) else 'PENDING'} |",
                f"| Publish | {'PASS' if sl.get('publish_met', False) else 'PENDING'} |",
            ])

        return "\n".join(lines)

    def _md_action_plan_summary(self, data: Dict[str, Any]) -> str:
        plan = data.get("action_plan", {})
        levers = plan.get("key_levers", [])

        lines = [
            "## 3. Action Plan Summary\n",
            f"**Plan Status:** {plan.get('status', 'Developed')}  \n"
            f"**Key Levers:** {len(levers)}  \n"
            f"**Total Investment:** ${_dec_comma(plan.get('total_investment_usd', 0))}\n",
        ]

        if levers:
            lines.extend([
                "### Key Decarbonization Levers\n",
                "| # | Lever | Scope | Reduction | Timeline |",
                "|---|-------|-------|:---------:|:--------:|",
            ])
            for i, lever in enumerate(levers, 1):
                lines.append(
                    f"| {i} | {lever.get('lever', '-')} "
                    f"| {lever.get('scope', '-')} "
                    f"| {_pct(lever.get('reduction_pct', 0))} "
                    f"| {lever.get('timeline', '-')} |"
                )

        return "\n".join(lines)

    def _md_baseline_summary(self, data: Dict[str, Any]) -> str:
        b = data.get("baseline", {})
        total = b.get("total_tco2e", 0)
        return (
            f"## 4. Emissions Baseline Summary\n\n"
            f"| Scope | Emissions (tCO2e) | % of Total |\n|-------|------------------:|:----------:|\n"
            f"| Scope 1 | {_dec_comma(b.get('scope1_tco2e', 0))} "
            f"| {_pct(_safe_div(b.get('scope1_tco2e', 0), max(total, 1)) * 100)} |\n"
            f"| Scope 2 | {_dec_comma(b.get('scope2_tco2e', 0))} "
            f"| {_pct(_safe_div(b.get('scope2_tco2e', 0), max(total, 1)) * 100)} |\n"
            f"| Scope 3 | {_dec_comma(b.get('scope3_tco2e', 0))} "
            f"| {_pct(_safe_div(b.get('scope3_tco2e', 0), max(total, 1)) * 100)} |\n"
            f"| **Total** | **{_dec_comma(total)}** | **100%** |\n\n"
            f"**Base Year:** {b.get('year', 'N/A')}  \n"
            f"**Methodology:** {b.get('methodology', 'GHG Protocol Corporate Standard')}  \n"
            f"**Boundary:** {b.get('boundary', 'Operational control')}"
        )

    def _md_target_summary(self, data: Dict[str, Any]) -> str:
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        return (
            f"## 5. Target Summary\n\n"
            f"| Parameter | Interim Target | Long-Term Target |\n"
            f"|-----------|:--------------:|:----------------:|\n"
            f"| Year | {interim.get('year', 2030)} | {longterm.get('year', 2050)} |\n"
            f"| Reduction | {_pct(interim.get('reduction_pct', 50))} | {_pct(longterm.get('min_reduction_pct', 90))} |\n"
            f"| Scope | {interim.get('scope_coverage', 'S1+S2')} | {longterm.get('scope_coverage', 'S1+S2+S3')} |\n"
            f"| Pathway | {interim.get('pathway', '1.5C aligned')} | {longterm.get('pathway', 'Net-zero')} |\n"
            f"| Validation | {interim.get('validation', 'SBTi')} | {longterm.get('validation', 'SBTi Net-Zero')} |\n"
            f"| Offsets | {interim.get('offset_policy', 'Not permitted')} | {longterm.get('offset_policy', 'Residual only')} |"
        )

    def _md_verification(self, data: Dict[str, Any]) -> str:
        schedule = data.get("verification_schedule", [])
        lines = [
            "## 6. Verification Schedule\n",
        ]
        if schedule:
            lines.extend([
                "| Year | Type | Provider | Standard | Status |",
                "|:----:|------|----------|----------|:------:|",
            ])
            for v in schedule:
                lines.append(
                    f"| {v.get('year', '-')} | {v.get('type', '-')} "
                    f"| {v.get('provider', '-')} | {v.get('standard', '-')} "
                    f"| {v.get('status', 'Planned')} |"
                )
        else:
            lines.append("_Verification schedule to be finalized post-submission._")
        return "\n".join(lines)

    def _md_contact(self, data: Dict[str, Any]) -> str:
        contact = data.get("contact", {})
        return (
            f"## 7. Contact Information\n\n"
            f"### Primary Contact\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Name | {contact.get('primary_name', '')} |\n"
            f"| Title | {contact.get('primary_title', '')} |\n"
            f"| Email | {contact.get('primary_email', '')} |\n"
            f"| Phone | {contact.get('primary_phone', '')} |\n\n"
            f"### Secondary Contact\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Name | {contact.get('secondary_name', '')} |\n"
            f"| Email | {contact.get('secondary_email', '')} |"
        )

    def _md_submission_checklist(self, data: Dict[str, Any]) -> str:
        checklist_status = data.get("checklist_status", {})
        lines = [
            "## 8. Submission Confirmation & Checklist\n",
            "| ID | Item | Category | Required | Complete |",
            "|:--:|------|----------|:--------:|:--------:|",
        ]

        completed_count = 0
        required_completed = 0
        required_count = 0

        for item in SUBMISSION_CHECKLIST:
            status = checklist_status.get(item["id"], {})
            is_complete = status.get("completed", False)
            if is_complete:
                completed_count += 1
            if item["required"]:
                required_count += 1
                if is_complete:
                    required_completed += 1

            lines.append(
                f"| {item['id']} | {item['item']} "
                f"| {item['category']} "
                f"| {'Yes' if item['required'] else 'No'} "
                f"| {'YES' if is_complete else 'NO'} |"
            )

        is_ready = required_completed >= required_count
        status_text = "READY FOR SUBMISSION" if is_ready else "INCOMPLETE -- Required items outstanding"

        lines.extend([
            f"\n**Submission Status:** {status_text}",
            f"- Required items: {required_completed}/{required_count}",
            f"- Total items: {completed_count}/{len(SUBMISSION_CHECKLIST)}",
        ])

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Submit via your Race to Zero partner initiative.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".badge-complete{background:#43a047;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".badge-incomplete{background:#ef5350;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".submission-ready{background:#e8f5e9;border:3px solid #2e7d32;border-radius:12px;"
            "padding:20px;text-align:center;margin:20px 0;}"
            ".submission-not-ready{background:#ffebee;border:3px solid #ef5350;border-radius:12px;"
            "padding:20px;text-align:center;margin:20px 0;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_cover(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Campaign Submission Package</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Type:</strong> {data.get("submission_type", "Initial Registration")} | '
            f'<strong>Date:</strong> {ts}</p>'
        )

    def _html_pledge(self, data: Dict[str, Any]) -> str:
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        baseline = data.get("baseline", {})
        return (
            f'<h2>1. Pledge Summary</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Baseline</div>'
            f'<div class="card-value">{_dec_comma(baseline.get("total_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Interim Target</div>'
            f'<div class="card-value">{_pct(interim.get("reduction_pct", 50))}</div>'
            f'by {interim.get("year", 2030)}</div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero</div>'
            f'<div class="card-value">{longterm.get("year", 2050)}</div></div>\n'
            f'</div>'
        )

    def _html_starting_line(self, data: Dict[str, Any]) -> str:
        sl = data.get("starting_line", {})
        met = sl.get("criteria_met", 0)
        total = sl.get("total_criteria", 20)
        return (
            f'<h2>2. Starting Line Compliance</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Criteria Met</div>'
            f'<div class="card-value">{met}/{total}</div></div>\n'
            f'  <div class="card"><div class="card-label">Compliance</div>'
            f'<div class="card-value">{_pct(_safe_div(met, total) * 100)}</div></div>\n'
            f'</div>'
        )

    def _html_baseline(self, data: Dict[str, Any]) -> str:
        b = data.get("baseline", {})
        return (
            f'<h2>4. Baseline Emissions</h2>\n'
            f'<table><tr><th>Scope</th><th>Emissions (tCO2e)</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(b.get("scope1_tco2e", 0))}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec_comma(b.get("scope2_tco2e", 0))}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(b.get("scope3_tco2e", 0))}</td></tr>\n'
            f'<tr style="font-weight:bold"><td>Total</td><td>{_dec_comma(b.get("total_tco2e", 0))}</td></tr>\n'
            f'</table>'
        )

    def _html_targets(self, data: Dict[str, Any]) -> str:
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        return (
            f'<h2>5. Targets</h2>\n'
            f'<table><tr><th>Parameter</th><th>Interim</th><th>Long-Term</th></tr>\n'
            f'<tr><td>Year</td><td>{interim.get("year", 2030)}</td><td>{longterm.get("year", 2050)}</td></tr>\n'
            f'<tr><td>Reduction</td><td>{_pct(interim.get("reduction_pct", 50))}</td>'
            f'<td>{_pct(longterm.get("min_reduction_pct", 90))}</td></tr>\n'
            f'</table>'
        )

    def _html_verification(self, data: Dict[str, Any]) -> str:
        schedule = data.get("verification_schedule", [])
        rows = ""
        for v in schedule:
            rows += (f'<tr><td>{v.get("year", "-")}</td><td>{v.get("type", "-")}</td>'
                     f'<td>{v.get("status", "Planned")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="3"><em>Schedule pending</em></td></tr>'
        return (
            f'<h2>6. Verification Schedule</h2>\n'
            f'<table><tr><th>Year</th><th>Type</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_contact(self, data: Dict[str, Any]) -> str:
        contact = data.get("contact", {})
        return (
            f'<h2>7. Contact</h2>\n'
            f'<table><tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Primary Contact</td><td>{contact.get("primary_name", "")}</td></tr>\n'
            f'<tr><td>Email</td><td>{contact.get("primary_email", "")}</td></tr>\n'
            f'<tr><td>Phone</td><td>{contact.get("primary_phone", "")}</td></tr>\n'
            f'</table>'
        )

    def _html_checklist(self, data: Dict[str, Any]) -> str:
        checklist_status = data.get("checklist_status", {})
        required_items = [c for c in SUBMISSION_CHECKLIST if c["required"]]
        required_completed = sum(
            1 for c in required_items
            if checklist_status.get(c["id"], {}).get("completed", False)
        )
        is_ready = required_completed >= len(required_items)
        status_class = "submission-ready" if is_ready else "submission-not-ready"
        status_text = "READY FOR SUBMISSION" if is_ready else "INCOMPLETE"

        rows = ""
        for item in SUBMISSION_CHECKLIST:
            status = checklist_status.get(item["id"], {})
            complete = status.get("completed", False)
            badge = "badge-complete" if complete else "badge-incomplete"
            label = "DONE" if complete else "TODO"
            rows += (f'<tr><td>{item["id"]}</td><td>{item["item"]}</td>'
                     f'<td>{"Yes" if item["required"] else "No"}</td>'
                     f'<td><span class="{badge}">{label}</span></td></tr>\n')

        return (
            f'<h2>8. Submission Checklist</h2>\n'
            f'<div class="{status_class}"><strong>{status_text}</strong><br>'
            f'{required_completed}/{len(required_items)} required items complete</div>\n'
            f'<table><tr><th>ID</th><th>Item</th><th>Required</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
