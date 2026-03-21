# -*- coding: utf-8 -*-
"""
StartingLineChecklistTemplate - Race to Zero starting line assessment for PACK-025.

Renders the Race to Zero starting line compliance checklist using the 4P
framework (Pledge/Plan/Proceed/Publish) with 20 sub-criteria, gap analysis,
remediation timeline, and supporting evidence tracking.

Sections:
    1. Assessment Overview
    2. 4P Framework Assessment
    3. Pledge Criteria (5 sub-criteria)
    4. Plan Criteria (5 sub-criteria)
    5. Proceed Criteria (5 sub-criteria)
    6. Publish Criteria (5 sub-criteria)
    7. Gap Analysis
    8. Remediation Timeline
    9. Supporting Evidence Links
    10. Compliance Summary

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
_TEMPLATE_ID = "starting_line_checklist"


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


# ------------------------------------------------------------------ #
#  Default criteria definitions per the 4P framework                   #
# ------------------------------------------------------------------ #

_PLEDGE_CRITERIA = [
    {"id": "P1-01", "category": "Pledge", "criterion": "Public pledge to reach net-zero by 2050 at the latest",
     "guidance": "Executive-level public commitment through partner initiative",
     "weight": 1.0, "critical": True},
    {"id": "P1-02", "category": "Pledge", "criterion": "Pledge covers all scopes (Scope 1, 2, and 3)",
     "guidance": "No material scope exclusions; Scope 3 at minimum 67% coverage",
     "weight": 1.0, "critical": True},
    {"id": "P1-03", "category": "Pledge", "criterion": "Interim target set for 2030 or sooner",
     "guidance": "Near-term target within 5-10 years of pledge date, halving emissions",
     "weight": 1.0, "critical": True},
    {"id": "P1-04", "category": "Pledge", "criterion": "Targets aligned with 1.5C science-based pathways",
     "guidance": "Validated by SBTi or equivalent; no/limited overshoot scenarios",
     "weight": 1.0, "critical": True},
    {"id": "P1-05", "category": "Pledge", "criterion": "No planned use of offsets for interim targets",
     "guidance": "Offsets reserved for neutralization of residual emissions only",
     "weight": 0.8, "critical": False},
]

_PLAN_CRITERIA = [
    {"id": "P2-01", "category": "Plan", "criterion": "Transition plan developed within 12 months of joining",
     "guidance": "Comprehensive plan covering all emission scopes with timeline",
     "weight": 1.0, "critical": True},
    {"id": "P2-02", "category": "Plan", "criterion": "Plan includes specific actions and milestones",
     "guidance": "Quantified reduction levers with responsible owners and dates",
     "weight": 1.0, "critical": True},
    {"id": "P2-03", "category": "Plan", "criterion": "Governance structure for climate action in place",
     "guidance": "Board-level oversight with clear accountability chain",
     "weight": 0.9, "critical": True},
    {"id": "P2-04", "category": "Plan", "criterion": "Budget allocated for decarbonization activities",
     "guidance": "Dedicated CAPEX/OPEX for emissions reduction projects",
     "weight": 0.8, "critical": False},
    {"id": "P2-05", "category": "Plan", "criterion": "Scope 3 engagement strategy documented",
     "guidance": "Supplier and value chain engagement plan with coverage targets",
     "weight": 0.9, "critical": True},
]

_PROCEED_CRITERIA = [
    {"id": "P3-01", "category": "Proceed", "criterion": "Immediate actions initiated within 12 months",
     "guidance": "At least 3 concrete reduction actions started post-pledge",
     "weight": 1.0, "critical": True},
    {"id": "P3-02", "category": "Proceed", "criterion": "Annual emissions inventory conducted",
     "guidance": "Complete S1+S2+S3 inventory per GHG Protocol methodology",
     "weight": 1.0, "critical": True},
    {"id": "P3-03", "category": "Proceed", "criterion": "Reduction actions aligned with transition plan",
     "guidance": "Actions map to plan milestones with measurable outcomes",
     "weight": 0.9, "critical": False},
    {"id": "P3-04", "category": "Proceed", "criterion": "Investment in low-carbon technologies commenced",
     "guidance": "Capex deployed for energy efficiency, renewables, or process changes",
     "weight": 0.8, "critical": False},
    {"id": "P3-05", "category": "Proceed", "criterion": "Supply chain engagement initiated",
     "guidance": "Supplier questionnaires, CDP supply chain, or direct engagement started",
     "weight": 0.8, "critical": False},
]

_PUBLISH_CRITERIA = [
    {"id": "P4-01", "category": "Publish", "criterion": "Annual progress reporting committed",
     "guidance": "Public annual report against Race to Zero requirements",
     "weight": 1.0, "critical": True},
    {"id": "P4-02", "category": "Publish", "criterion": "Emissions data publicly disclosed",
     "guidance": "S1+S2+S3 data in CDP, annual report, or website disclosure",
     "weight": 1.0, "critical": True},
    {"id": "P4-03", "category": "Publish", "criterion": "Progress against targets reported transparently",
     "guidance": "Year-on-year reduction trajectory with explanation of variances",
     "weight": 0.9, "critical": True},
    {"id": "P4-04", "category": "Publish", "criterion": "Independent verification planned or completed",
     "guidance": "Third-party limited or reasonable assurance engagement",
     "weight": 0.8, "critical": False},
    {"id": "P4-05", "category": "Publish", "criterion": "Use of offsets transparently disclosed",
     "guidance": "Offset type, quantity, registry, vintage, and project details",
     "weight": 0.7, "critical": False},
]

ALL_CRITERIA = _PLEDGE_CRITERIA + _PLAN_CRITERIA + _PROCEED_CRITERIA + _PUBLISH_CRITERIA


class StartingLineChecklistTemplate:
    """Race to Zero starting line compliance checklist template for PACK-025.

    Generates the 4P framework assessment (Pledge/Plan/Proceed/Publish)
    with 20 sub-criteria, gap analysis, remediation timeline, and
    supporting evidence links in multiple output formats.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    CRITERIA = ALL_CRITERIA

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the starting line checklist as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_4p_summary(data),
            self._md_pledge_criteria(data),
            self._md_plan_criteria(data),
            self._md_proceed_criteria(data),
            self._md_publish_criteria(data),
            self._md_gap_analysis(data),
            self._md_remediation_timeline(data),
            self._md_evidence_links(data),
            self._md_compliance_summary(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the starting line checklist as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_4p_summary(data),
            self._html_criteria_table(data),
            self._html_gap_analysis(data),
            self._html_remediation(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Starting Line Checklist</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the starting line checklist as structured JSON."""
        self.generated_at = _utcnow()
        assessments = self._build_assessments(data)
        gaps = self._build_gaps(assessments)

        by_category: Dict[str, Dict[str, Any]] = {}
        for cat_name in ["Pledge", "Plan", "Proceed", "Publish"]:
            cat_items = [a for a in assessments if a["category"] == cat_name]
            met = sum(1 for a in cat_items if a["met"])
            by_category[cat_name] = {
                "total": len(cat_items),
                "met": met,
                "compliance_pct": _safe_div(met, len(cat_items)) * 100,
            }

        total_met = sum(1 for a in assessments if a["met"])
        total_all = len(assessments)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "assessment_date": data.get("assessment_date", self.generated_at.strftime("%Y-%m-%d")),
            "overall_compliance": {
                "criteria_met": total_met,
                "total_criteria": total_all,
                "compliance_pct": _safe_div(total_met, total_all) * 100,
                "is_compliant": total_met >= total_all,
                "critical_gaps": sum(1 for g in gaps if g.get("critical", False)),
            },
            "by_category": by_category,
            "assessments": assessments,
            "gaps": gaps,
            "remediation_actions": data.get("remediation_actions", []),
            "evidence_links": data.get("evidence_links", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = _utcnow()
        assessments = self._build_assessments(data)
        gaps = self._build_gaps(assessments)
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Checklist
        checklist_rows: List[Dict[str, Any]] = []
        for a in assessments:
            checklist_rows.append({
                "ID": a["id"],
                "Category": a["category"],
                "Criterion": a["criterion"],
                "Guidance": a.get("guidance", ""),
                "Met": "Yes" if a["met"] else "No",
                "Evidence": a.get("evidence", ""),
                "Notes": a.get("notes", ""),
                "Weight": a.get("weight", 1.0),
                "Critical": "Yes" if a.get("critical", False) else "No",
            })
        sheets["Checklist"] = checklist_rows

        # Sheet 2: Gap Analysis
        gap_rows: List[Dict[str, Any]] = []
        for g in gaps:
            gap_rows.append({
                "ID": g["id"],
                "Category": g["category"],
                "Gap": g["criterion"],
                "Critical": "Yes" if g.get("critical", False) else "No",
                "Remediation": g.get("remediation", ""),
                "Target Date": g.get("target_date", ""),
                "Priority": g.get("priority", "Medium"),
            })
        sheets["Gap Analysis"] = gap_rows

        # Sheet 3: Category Summary
        summary_rows: List[Dict[str, Any]] = []
        for cat_name in ["Pledge", "Plan", "Proceed", "Publish"]:
            cat_items = [a for a in assessments if a["category"] == cat_name]
            met = sum(1 for a in cat_items if a["met"])
            total = len(cat_items)
            summary_rows.append({
                "Category": cat_name,
                "Criteria Met": met,
                "Total Criteria": total,
                "Compliance (%)": round(_safe_div(met, total) * 100, 1),
                "Status": "Compliant" if met >= total else "Gaps Identified",
            })
        total_met = sum(1 for a in assessments if a["met"])
        total_all = len(assessments)
        summary_rows.append({
            "Category": "OVERALL",
            "Criteria Met": total_met,
            "Total Criteria": total_all,
            "Compliance (%)": round(_safe_div(total_met, total_all) * 100, 1),
            "Status": "Compliant" if total_met >= total_all else "Gaps Identified",
        })
        sheets["Category Summary"] = summary_rows

        # Sheet 4: Remediation Timeline
        remediation = data.get("remediation_actions", [])
        rem_rows: List[Dict[str, Any]] = []
        for r in remediation:
            rem_rows.append({
                "Gap ID": r.get("gap_id", ""),
                "Action": r.get("action", ""),
                "Owner": r.get("owner", ""),
                "Start Date": r.get("start_date", ""),
                "Target Date": r.get("target_date", ""),
                "Status": r.get("status", "Not Started"),
            })
        sheets["Remediation Timeline"] = rem_rows

        # Sheet 5: Evidence Links
        evidence = data.get("evidence_links", [])
        ev_rows: List[Dict[str, Any]] = []
        for e in evidence:
            ev_rows.append({
                "Criterion ID": e.get("criterion_id", ""),
                "Document": e.get("document", ""),
                "Type": e.get("type", ""),
                "URL": e.get("url", ""),
                "Date": e.get("date", ""),
            })
        sheets["Evidence Links"] = ev_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_assessments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the complete assessment list, merging user data with defaults."""
        user_items = data.get("checklist_items", [])
        if user_items:
            return user_items

        # Merge user overrides with default criteria
        overrides = data.get("criteria_overrides", {})
        assessments: List[Dict[str, Any]] = []
        for criterion in ALL_CRITERIA:
            cid = criterion["id"]
            override = overrides.get(cid, {})
            assessments.append({
                "id": cid,
                "category": criterion["category"],
                "criterion": criterion["criterion"],
                "guidance": criterion.get("guidance", ""),
                "weight": criterion.get("weight", 1.0),
                "critical": criterion.get("critical", False),
                "met": override.get("met", False),
                "evidence": override.get("evidence", ""),
                "notes": override.get("notes", ""),
            })
        return assessments

    def _build_gaps(self, assessments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract gaps (unmet criteria) from assessments."""
        return [a for a in assessments if not a.get("met", False)]

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Starting Line Checklist\n\n"
            f"**Organization:** {org}  \n"
            f"**Assessment Date:** {data.get('assessment_date', ts)}  \n"
            f"**Framework:** 4P (Pledge / Plan / Proceed / Publish)  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        total_met = sum(1 for a in assessments if a["met"])
        total_all = len(assessments)
        critical_gaps = sum(1 for a in assessments if not a["met"] and a.get("critical", False))
        compliance_pct = _safe_div(total_met, total_all) * 100
        status = "COMPLIANT" if total_met >= total_all else "GAPS IDENTIFIED"

        return (
            f"## 1. Assessment Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Overall Status | **{status}** |\n"
            f"| Criteria Met | {total_met} / {total_all} |\n"
            f"| Compliance | {_pct(compliance_pct)} |\n"
            f"| Critical Gaps | {critical_gaps} |\n"
            f"| Non-Critical Gaps | {total_all - total_met - critical_gaps} |\n"
            f"| Assessment Standard | Race to Zero Starting Line Criteria (2024) |"
        )

    def _md_4p_summary(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        lines = [
            "## 2. 4P Framework Summary\n",
            "| Category | Description | Met | Total | Compliance | Status |",
            "|----------|-------------|:---:|:-----:|:----------:|:------:|",
        ]
        descriptions = {
            "Pledge": "Commit to net-zero through a recognized initiative",
            "Plan": "Develop a transition plan within 12 months",
            "Proceed": "Take immediate action towards decarbonization",
            "Publish": "Report progress annually with transparency",
        }
        for cat_name in ["Pledge", "Plan", "Proceed", "Publish"]:
            cat_items = [a for a in assessments if a["category"] == cat_name]
            met = sum(1 for a in cat_items if a["met"])
            total = len(cat_items)
            pct = _safe_div(met, total) * 100
            status = "PASS" if met >= total else "GAPS"
            lines.append(
                f"| **{cat_name}** | {descriptions.get(cat_name, '')} "
                f"| {met} | {total} | {_pct(pct)} | {status} |"
            )

        total_met = sum(1 for a in assessments if a["met"])
        total_all = len(assessments)
        lines.append(
            f"| **TOTAL** | **All Starting Line Criteria** "
            f"| **{total_met}** | **{total_all}** "
            f"| **{_pct(_safe_div(total_met, total_all) * 100)}** "
            f"| **{'PASS' if total_met >= total_all else 'GAPS'}** |"
        )
        return "\n".join(lines)

    def _md_category_criteria(self, category: str, section_num: int,
                              data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        cat_items = [a for a in assessments if a["category"] == category]

        lines = [
            f"## {section_num}. {category} Criteria\n",
            "| ID | Criterion | Critical | Met | Evidence | Notes |",
            "|:--:|-----------|:--------:|:---:|----------|-------|",
        ]
        for item in cat_items:
            met_text = "YES" if item["met"] else "NO"
            critical_text = "Yes" if item.get("critical", False) else "No"
            lines.append(
                f"| {item['id']} | {item['criterion']} "
                f"| {critical_text} | {met_text} "
                f"| {item.get('evidence', '')} | {item.get('notes', '')} |"
            )

        met_count = sum(1 for i in cat_items if i["met"])
        lines.append(f"\n**{category} Score:** {met_count}/{len(cat_items)} criteria met")
        return "\n".join(lines)

    def _md_pledge_criteria(self, data: Dict[str, Any]) -> str:
        return self._md_category_criteria("Pledge", 3, data)

    def _md_plan_criteria(self, data: Dict[str, Any]) -> str:
        return self._md_category_criteria("Plan", 4, data)

    def _md_proceed_criteria(self, data: Dict[str, Any]) -> str:
        return self._md_category_criteria("Proceed", 5, data)

    def _md_publish_criteria(self, data: Dict[str, Any]) -> str:
        return self._md_category_criteria("Publish", 6, data)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        gaps = self._build_gaps(assessments)

        lines = [
            "## 7. Gap Analysis\n",
        ]
        if not gaps:
            lines.append("**No gaps identified.** All starting line criteria are met.")
            return "\n".join(lines)

        lines.extend([
            f"**{len(gaps)} gap(s) identified** requiring remediation.\n",
            "| # | ID | Category | Gap | Critical | Priority | Remediation |",
            "|---|:--:|:--------:|-----|:--------:|:--------:|-------------|",
        ])
        for i, gap in enumerate(gaps, 1):
            priority = "HIGH" if gap.get("critical", False) else "MEDIUM"
            remediation = gap.get("remediation", "Action plan required")
            lines.append(
                f"| {i} | {gap['id']} | {gap['category']} "
                f"| {gap['criterion']} | {'Yes' if gap.get('critical') else 'No'} "
                f"| {priority} | {remediation} |"
            )
        return "\n".join(lines)

    def _md_remediation_timeline(self, data: Dict[str, Any]) -> str:
        actions = data.get("remediation_actions", [])
        lines = ["## 8. Remediation Timeline\n"]

        if not actions:
            lines.append("_Remediation timeline will be developed based on gap analysis._")
            return "\n".join(lines)

        lines.extend([
            "| # | Gap ID | Action | Owner | Start | Target | Status |",
            "|---|:------:|--------|-------|:-----:|:------:|:------:|",
        ])
        for i, action in enumerate(actions, 1):
            lines.append(
                f"| {i} | {action.get('gap_id', '-')} "
                f"| {action.get('action', '-')} "
                f"| {action.get('owner', '-')} "
                f"| {action.get('start_date', '-')} "
                f"| {action.get('target_date', '-')} "
                f"| {action.get('status', 'Not Started')} |"
            )
        return "\n".join(lines)

    def _md_evidence_links(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_links", [])
        lines = ["## 9. Supporting Evidence Links\n"]

        if not evidence:
            lines.append("_Evidence links to be populated as documentation is compiled._")
            return "\n".join(lines)

        lines.extend([
            "| Criterion | Document | Type | URL | Date |",
            "|:---------:|----------|------|-----|:----:|",
        ])
        for ev in evidence:
            lines.append(
                f"| {ev.get('criterion_id', '-')} "
                f"| {ev.get('document', '-')} "
                f"| {ev.get('type', '-')} "
                f"| {ev.get('url', '-')} "
                f"| {ev.get('date', '-')} |"
            )
        return "\n".join(lines)

    def _md_compliance_summary(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        total_met = sum(1 for a in assessments if a["met"])
        total_all = len(assessments)
        is_compliant = total_met >= total_all

        weighted_score = 0.0
        total_weight = 0.0
        for a in assessments:
            w = a.get("weight", 1.0)
            total_weight += w
            if a["met"]:
                weighted_score += w
        weighted_pct = _safe_div(weighted_score, total_weight) * 100

        status = "COMPLIANT -- Eligible to proceed with Race to Zero submission" if is_compliant else \
                 "NOT COMPLIANT -- Remediation required before submission"

        return (
            f"## 10. Compliance Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Overall Status | **{status}** |\n"
            f"| Simple Compliance | {total_met}/{total_all} ({_pct(_safe_div(total_met, total_all) * 100)}) |\n"
            f"| Weighted Score | {_pct(weighted_pct)} |\n"
            f"| Critical Criteria Met | {sum(1 for a in assessments if a.get('critical') and a['met'])} / "
            f"{sum(1 for a in assessments if a.get('critical'))} |\n"
            f"| Recommendation | {'Proceed with submission' if is_compliant else 'Address gaps before submission'} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Starting Line criteria per Race to Zero campaign requirements.*"
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
            ".cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".badge-met{background:#43a047;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".badge-gap{background:#ef5350;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".badge-critical{background:#ff9800;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".progress-bar{background:#e0e0e0;border-radius:8px;height:24px;overflow:hidden;margin:8px 0;}"
            ".progress-fill{height:100%;border-radius:8px;transition:width 0.3s;}"
            ".progress-fill.high{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".progress-fill.medium{background:linear-gradient(90deg,#ff9800,#ffb74d);}"
            ".progress-fill.low{background:linear-gradient(90deg,#ef5350,#ef9a9a);}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Starting Line Checklist</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Framework:</strong> 4P (Pledge/Plan/Proceed/Publish) | '
            f'<strong>Date:</strong> {ts}</p>'
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        total_met = sum(1 for a in assessments if a["met"])
        total_all = len(assessments)
        compliance_pct = _safe_div(total_met, total_all) * 100
        critical_gaps = sum(1 for a in assessments if not a["met"] and a.get("critical", False))

        bar_class = "high" if compliance_pct >= 80 else ("medium" if compliance_pct >= 50 else "low")

        return (
            f'<h2>Assessment Overview</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Criteria Met</div>'
            f'<div class="card-value">{total_met}/{total_all}</div></div>\n'
            f'  <div class="card"><div class="card-label">Compliance</div>'
            f'<div class="card-value">{_pct(compliance_pct)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Critical Gaps</div>'
            f'<div class="card-value">{critical_gaps}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">{"PASS" if total_met >= total_all else "GAPS"}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar"><div class="progress-fill {bar_class}" '
            f'style="width:{compliance_pct}%"></div></div>'
        )

    def _html_4p_summary(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        rows = ""
        for cat_name in ["Pledge", "Plan", "Proceed", "Publish"]:
            cat_items = [a for a in assessments if a["category"] == cat_name]
            met = sum(1 for a in cat_items if a["met"])
            total = len(cat_items)
            pct = _safe_div(met, total) * 100
            badge = "badge-met" if met >= total else "badge-gap"
            label = "PASS" if met >= total else "GAPS"
            rows += (
                f'<tr><td><strong>{cat_name}</strong></td><td>{met}/{total}</td>'
                f'<td>{_pct(pct)}</td>'
                f'<td><span class="{badge}">{label}</span></td></tr>\n'
            )
        return (
            f'<h2>4P Framework Summary</h2>\n'
            f'<table><tr><th>Category</th><th>Met/Total</th><th>Compliance</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_criteria_table(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        rows = ""
        for a in assessments:
            badge = "badge-met" if a["met"] else "badge-gap"
            label = "MET" if a["met"] else "GAP"
            critical = ' <span class="badge-critical">CRITICAL</span>' if a.get("critical") and not a["met"] else ""
            rows += (
                f'<tr><td>{a["id"]}</td><td>{a["category"]}</td>'
                f'<td>{a["criterion"]}</td>'
                f'<td><span class="{badge}">{label}</span>{critical}</td></tr>\n'
            )
        return (
            f'<h2>Criteria Assessment</h2>\n'
            f'<table><tr><th>ID</th><th>Category</th><th>Criterion</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        assessments = self._build_assessments(data)
        gaps = self._build_gaps(assessments)
        if not gaps:
            return '<h2>Gap Analysis</h2>\n<p>No gaps identified. All criteria met.</p>'

        rows = ""
        for g in gaps:
            priority = "HIGH" if g.get("critical") else "MEDIUM"
            rows += (
                f'<tr><td>{g["id"]}</td><td>{g["category"]}</td>'
                f'<td>{g["criterion"]}</td><td>{priority}</td></tr>\n'
            )
        return (
            f'<h2>Gap Analysis</h2>\n'
            f'<p><strong>{len(gaps)}</strong> gap(s) requiring remediation.</p>\n'
            f'<table><tr><th>ID</th><th>Category</th><th>Gap</th><th>Priority</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_remediation(self, data: Dict[str, Any]) -> str:
        actions = data.get("remediation_actions", [])
        if not actions:
            return '<h2>Remediation</h2>\n<p>Remediation timeline pending gap analysis.</p>'

        rows = ""
        for a in actions:
            rows += (
                f'<tr><td>{a.get("gap_id", "-")}</td><td>{a.get("action", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td><td>{a.get("target_date", "-")}</td>'
                f'<td>{a.get("status", "Not Started")}</td></tr>\n'
            )
        return (
            f'<h2>Remediation Timeline</h2>\n'
            f'<table><tr><th>Gap ID</th><th>Action</th><th>Owner</th>'
            f'<th>Target Date</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
