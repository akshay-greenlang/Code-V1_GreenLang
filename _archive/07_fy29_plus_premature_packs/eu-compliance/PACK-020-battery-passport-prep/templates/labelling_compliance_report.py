# -*- coding: utf-8 -*-
"""
LabellingComplianceReportTemplate - EU Battery Regulation Art 13-14 Labelling Compliance

Renders the labelling and marking compliance assessment report per Articles 13
and 14 of Regulation (EU) 2023/1542. All batteries placed on the EU market must
bear specific labels including the general label (CE marking, manufacturer info,
battery type, chemistry, hazardous substances), the separate collection symbol
(crossed-out wheeled bin), capacity information, QR code linking to the battery
passport (for industrial/EV from 2027), and the carbon footprint performance
class label (for industrial/EV from 2026). This template verifies element-by-
element compliance, identifies missing elements, and recommends corrective actions.

Sections:
    1. Element Checklist - All mandatory labelling elements with status
    2. Compliance Status - Overall and per-category compliance assessment
    3. Missing Elements - Detailed gap analysis
    4. Corrective Actions - Prioritized actions to achieve full compliance

Author: GreenLang Team
Version: 20.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "element_checklist",
    "compliance_status",
    "missing_elements",
    "corrective_actions",
]

# Mandatory labelling elements per Art 13-14 and Annex VI
_LABELLING_ELEMENTS: List[Dict[str, Any]] = [
    # General Information (Art 13(1))
    {"id": "L01", "element": "CE marking", "article": "Art 13(1)(a)",
     "category": "general", "applicable_to": ["all"]},
    {"id": "L02", "element": "Manufacturer name and address", "article": "Art 13(1)(b)",
     "category": "general", "applicable_to": ["all"]},
    {"id": "L03", "element": "Battery type designation", "article": "Art 13(1)(c)",
     "category": "general", "applicable_to": ["all"]},
    {"id": "L04", "element": "Battery model identification", "article": "Art 13(1)(c)",
     "category": "general", "applicable_to": ["all"]},
    {"id": "L05", "element": "Date of manufacture (month/year)", "article": "Art 13(1)(d)",
     "category": "general", "applicable_to": ["all"]},
    {"id": "L06", "element": "Battery weight", "article": "Art 13(1)(e)",
     "category": "general", "applicable_to": ["all"]},
    {"id": "L07", "element": "Battery capacity (Ah and Wh)", "article": "Art 13(1)(f)",
     "category": "capacity", "applicable_to": ["all"]},
    {"id": "L08", "element": "Battery chemistry", "article": "Art 13(1)(g)",
     "category": "general", "applicable_to": ["all"]},
    # Hazardous substances (Art 13(2))
    {"id": "L09", "element": "Hazardous substance symbols (Cd, Pb, Hg)",
     "article": "Art 13(2)", "category": "hazardous",
     "applicable_to": ["all_if_applicable"]},
    {"id": "L10", "element": "Chemical symbol below waste bin (Cd/Pb/Hg)",
     "article": "Art 13(2)", "category": "hazardous",
     "applicable_to": ["all_if_applicable"]},
    # Separate collection (Art 13(3))
    {"id": "L11", "element": "Crossed-out wheeled bin symbol", "article": "Art 13(3)",
     "category": "collection", "applicable_to": ["all"]},
    # Capacity (Art 13(5))
    {"id": "L12", "element": "Rated capacity in Ah", "article": "Art 13(5)(a)",
     "category": "capacity", "applicable_to": ["all"]},
    {"id": "L13", "element": "Nominal voltage", "article": "Art 13(5)(b)",
     "category": "capacity", "applicable_to": ["all"]},
    {"id": "L14", "element": "Energy content in Wh", "article": "Art 13(5)(c)",
     "category": "capacity", "applicable_to": ["rechargeable"]},
    # Carbon footprint (Art 13(6))
    {"id": "L15", "element": "Carbon footprint performance class label",
     "article": "Art 13(6)", "category": "carbon_footprint",
     "applicable_to": ["ev_battery", "industrial_battery", "lmt_battery"]},
    # QR code (Art 13(7) + Art 77)
    {"id": "L16", "element": "QR code linking to battery passport",
     "article": "Art 13(7)", "category": "digital",
     "applicable_to": ["ev_battery", "industrial_battery", "lmt_battery"]},
    {"id": "L17", "element": "QR code linking to EU DoC", "article": "Art 13(7)",
     "category": "digital", "applicable_to": ["all"]},
    # Safety (Art 14)
    {"id": "L18", "element": "Safety instructions for use", "article": "Art 14(1)",
     "category": "safety", "applicable_to": ["all"]},
    {"id": "L19", "element": "Instructions for end-of-life handling",
     "article": "Art 14(2)", "category": "safety", "applicable_to": ["all"]},
    {"id": "L20", "element": "Separate collection instructions",
     "article": "Art 14(3)", "category": "safety", "applicable_to": ["all"]},
]

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class LabellingComplianceReportTemplate:
    """
    Labelling Compliance report template per EU Battery Regulation Art 13-14.

    Assesses labelling and marking compliance for all battery types against
    the 20 mandatory labelling elements defined in Articles 13-14. Generates
    a per-element checklist, overall compliance status, detailed gap analysis
    for missing elements, and prioritized corrective actions.

    Regulatory References:
        - Regulation (EU) 2023/1542, Articles 13-14
        - Regulation (EU) 2023/1542, Annex VI (Labelling and marking)
        - EN IEC 62902 (Secondary cells - marking symbols)
        - Directive 2006/66/EC (repealed, historical reference)

    Example:
        >>> tpl = LabellingComplianceReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LabellingComplianceReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "report_id": _new_uuid(),
            "generated_at": self.generated_at.isoformat(),
        }
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("battery_model"):
            errors.append("battery_model is required")
        if not data.get("battery_type"):
            warnings.append("battery_type not specified; defaulting to 'ev_battery'")
        if "labelling_status" not in data:
            warnings.append("labelling_status dict missing; all elements marked as unchecked")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render labelling compliance report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_element_checklist(data),
            self._md_compliance_status(data),
            self._md_missing_elements(data),
            self._md_corrective_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render labelling compliance report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_element_checklist(data),
            self._html_compliance_status(data),
            self._html_missing_elements(data),
            self._html_corrective_actions(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Labelling Compliance Report - Art 13-14</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render labelling compliance report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "labelling_compliance_report",
            "regulation_reference": "EU Battery Regulation 2023/1542, Art 13-14",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "battery_model": data.get("battery_model", ""),
            "element_checklist": self._section_element_checklist(data),
            "compliance_status": self._section_compliance_status(data),
            "missing_elements": self._section_missing_elements(data),
            "corrective_actions": self._section_corrective_actions(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_element_checklist(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build element-by-element checklist section."""
        battery_type = data.get("battery_type", "ev_battery")
        status_map = data.get("labelling_status", {})
        has_hazardous = data.get("contains_hazardous_substances", False)

        checklist: List[Dict[str, Any]] = []
        for elem in _LABELLING_ELEMENTS:
            applicable = self._is_applicable(elem, battery_type, has_hazardous)
            elem_status = status_map.get(elem["id"], {})
            present = elem_status.get("present", False)
            compliant = elem_status.get("compliant", False) if present else False
            checklist.append({
                "id": elem["id"],
                "element": elem["element"],
                "article": elem["article"],
                "category": elem["category"],
                "applicable": applicable,
                "present": present if applicable else None,
                "compliant": compliant if applicable else None,
                "notes": elem_status.get("notes", ""),
            })

        applicable_count = sum(1 for c in checklist if c["applicable"])
        present_count = sum(1 for c in checklist if c["present"] is True)
        compliant_count = sum(1 for c in checklist if c["compliant"] is True)

        return {
            "title": "Labelling Element Checklist",
            "battery_type": battery_type,
            "total_elements": len(checklist),
            "applicable_elements": applicable_count,
            "present_elements": present_count,
            "compliant_elements": compliant_count,
            "checklist": checklist,
        }

    def _section_compliance_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall compliance status section."""
        checklist = self._section_element_checklist(data)
        applicable = checklist["applicable_elements"]
        compliant = checklist["compliant_elements"]
        compliance_pct = round(compliant / applicable * 100, 1) if applicable > 0 else 0.0

        # Category breakdown
        categories: Dict[str, Dict[str, int]] = {}
        for item in checklist["checklist"]:
            if not item["applicable"]:
                continue
            cat = item["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "compliant": 0, "missing": 0}
            categories[cat]["total"] += 1
            if item["compliant"]:
                categories[cat]["compliant"] += 1
            elif not item["present"]:
                categories[cat]["missing"] += 1

        cat_details: List[Dict[str, Any]] = []
        for cat, counts in categories.items():
            cat_pct = (
                round(counts["compliant"] / counts["total"] * 100, 1)
                if counts["total"] > 0 else 0.0
            )
            cat_details.append({
                "category": cat,
                "category_label": cat.replace("_", " ").title(),
                "total": counts["total"],
                "compliant": counts["compliant"],
                "missing": counts["missing"],
                "compliance_pct": cat_pct,
                "status": "PASS" if cat_pct >= 100.0 else "FAIL",
            })

        return {
            "title": "Compliance Status",
            "overall_compliance_pct": compliance_pct,
            "overall_status": "COMPLIANT" if compliance_pct >= 100.0 else "NON-COMPLIANT",
            "applicable_elements": applicable,
            "compliant_elements": compliant,
            "non_compliant_elements": applicable - compliant,
            "category_breakdown": cat_details,
            "assessment_date": data.get(
                "assessment_date", utcnow().strftime("%Y-%m-%d")
            ),
        }

    def _section_missing_elements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed missing elements gap analysis section."""
        checklist = self._section_element_checklist(data)
        missing: List[Dict[str, Any]] = []
        non_compliant: List[Dict[str, Any]] = []

        for item in checklist["checklist"]:
            if not item["applicable"]:
                continue
            if not item["present"]:
                missing.append({
                    "id": item["id"],
                    "element": item["element"],
                    "article": item["article"],
                    "category": item["category"],
                    "severity": "high",
                    "gap_type": "missing",
                    "description": f"Element '{item['element']}' is not present on label",
                })
            elif not item["compliant"]:
                non_compliant.append({
                    "id": item["id"],
                    "element": item["element"],
                    "article": item["article"],
                    "category": item["category"],
                    "severity": "medium",
                    "gap_type": "non_compliant",
                    "description": (
                        f"Element '{item['element']}' is present but does not meet "
                        f"requirements per {item['article']}"
                    ),
                    "notes": item.get("notes", ""),
                })

        return {
            "title": "Missing & Non-Compliant Elements",
            "total_gaps": len(missing) + len(non_compliant),
            "missing_count": len(missing),
            "non_compliant_count": len(non_compliant),
            "missing_elements": missing,
            "non_compliant_elements": non_compliant,
        }

    def _section_corrective_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build corrective actions section with prioritized remediation."""
        gaps = self._section_missing_elements(data)
        actions: List[Dict[str, Any]] = []
        priority = 0

        # Missing elements get high priority
        for gap in gaps["missing_elements"]:
            priority += 1
            actions.append({
                "rank": priority,
                "priority": "HIGH",
                "element_id": gap["id"],
                "element": gap["element"],
                "article": gap["article"],
                "action": self._generate_corrective_action(gap),
                "estimated_effort": self._estimate_effort(gap),
                "deadline_recommendation": self._recommend_deadline(gap, data),
            })

        # Non-compliant elements get medium priority
        for gap in gaps["non_compliant_elements"]:
            priority += 1
            actions.append({
                "rank": priority,
                "priority": "MEDIUM",
                "element_id": gap["id"],
                "element": gap["element"],
                "article": gap["article"],
                "action": self._generate_corrective_action(gap),
                "estimated_effort": self._estimate_effort(gap),
                "deadline_recommendation": self._recommend_deadline(gap, data),
            })

        return {
            "title": "Corrective Actions",
            "total_actions": len(actions),
            "high_priority_count": sum(1 for a in actions if a["priority"] == "HIGH"),
            "medium_priority_count": sum(1 for a in actions if a["priority"] == "MEDIUM"),
            "actions": actions,
            "estimated_total_effort_days": sum(
                self._effort_to_days(a["estimated_effort"]) for a in actions
            ),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Labelling Compliance Report\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Articles 13-14\n\n"
            f"**Manufacturer:** {data.get('entity_name', '')}  \n"
            f"**Battery Model:** {data.get('battery_model', '')}  \n"
            f"**Battery Type:** {data.get('battery_type', 'ev_battery')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_element_checklist(self, data: Dict[str, Any]) -> str:
        """Render element checklist as markdown."""
        sec = self._section_element_checklist(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Applicable:** {sec['applicable_elements']} of {sec['total_elements']}  \n"
            f"**Present:** {sec['present_elements']}  \n"
            f"**Compliant:** {sec['compliant_elements']}\n",
            "| ID | Element | Article | Present | Compliant |",
            "|----|---------|---------|:-------:|:---------:|",
        ]
        for item in sec["checklist"]:
            if not item["applicable"]:
                continue
            present = "Yes" if item["present"] else "No"
            compliant = "Yes" if item["compliant"] else "No"
            lines.append(
                f"| {item['id']} | {item['element']} | {item['article']} | "
                f"{present} | {compliant} |"
            )
        return "\n".join(lines)

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render compliance status as markdown."""
        sec = self._section_compliance_status(data)
        lines = [
            f"## {sec['title']}\n",
            f"### Overall: {sec['overall_compliance_pct']:.1f}% - "
            f"{sec['overall_status']}\n",
            f"**Assessment Date:** {sec['assessment_date']}\n",
            "### Category Breakdown\n",
            "| Category | Total | Compliant | Missing | Status |",
            "|----------|------:|----------:|--------:|:------:|",
        ]
        for cat in sec["category_breakdown"]:
            lines.append(
                f"| {cat['category_label']} | {cat['total']} | {cat['compliant']} | "
                f"{cat['missing']} | {cat['status']} |"
            )
        return "\n".join(lines)

    def _md_missing_elements(self, data: Dict[str, Any]) -> str:
        """Render missing elements as markdown."""
        sec = self._section_missing_elements(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Gaps:** {sec['total_gaps']}  \n"
            f"**Missing:** {sec['missing_count']}  \n"
            f"**Non-Compliant:** {sec['non_compliant_count']}\n",
        ]
        if sec["missing_elements"]:
            lines.append("### Missing Elements\n")
            for gap in sec["missing_elements"]:
                lines.append(
                    f"- **{gap['id']}** [{gap['article']}]: {gap['element']}"
                )
        if sec["non_compliant_elements"]:
            lines.append("\n### Non-Compliant Elements\n")
            for gap in sec["non_compliant_elements"]:
                lines.append(
                    f"- **{gap['id']}** [{gap['article']}]: {gap['element']} "
                    f"- {gap.get('notes', '')}"
                )
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Render corrective actions as markdown."""
        sec = self._section_corrective_actions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Actions:** {sec['total_actions']}  \n"
            f"**High Priority:** {sec['high_priority_count']}  \n"
            f"**Est. Total Effort:** {sec['estimated_total_effort_days']} days\n",
            "| Rank | Priority | Element | Action | Effort | Deadline |",
            "|-----:|:--------:|---------|--------|--------|----------|",
        ]
        for action in sec["actions"]:
            lines.append(
                f"| {action['rank']} | {action['priority']} | "
                f"{action['element_id']} | {action['action']} | "
                f"{action['estimated_effort']} | {action['deadline_recommendation']} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Report generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Articles 13-14*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1000px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".pass{color:#2e7d32;font-weight:bold}"
            ".fail{color:#c62828;font-weight:bold}"
            ".warn{color:#e65100;font-weight:bold}"
            ".check{color:#2e7d32}.cross{color:#c62828}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Labelling Compliance Report</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Articles 13-14</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('battery_model', '')}</p>"
        )

    def _html_element_checklist(self, data: Dict[str, Any]) -> str:
        """Render element checklist HTML."""
        sec = self._section_element_checklist(data)
        rows = ""
        for item in sec["checklist"]:
            if not item["applicable"]:
                continue
            p_cls = "check" if item["present"] else "cross"
            c_cls = "check" if item["compliant"] else "cross"
            p_sym = "Y" if item["present"] else "N"
            c_sym = "Y" if item["compliant"] else "N"
            rows += (
                f"<tr><td>{item['id']}</td><td>{item['element']}</td>"
                f"<td>{item['article']}</td>"
                f"<td class='{p_cls}'>{p_sym}</td>"
                f"<td class='{c_cls}'>{c_sym}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>ID</th><th>Element</th><th>Article</th>"
            f"<th>Present</th><th>Compliant</th></tr>{rows}</table>"
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render compliance status HTML."""
        sec = self._section_compliance_status(data)
        overall_cls = "pass" if sec["overall_status"] == "COMPLIANT" else "fail"
        cat_rows = "".join(
            f"<tr><td>{c['category_label']}</td><td>{c['total']}</td>"
            f"<td>{c['compliant']}</td>"
            f"<td class='{'pass' if c['status'] == 'PASS' else 'fail'}'>"
            f"{c['status']}</td></tr>"
            for c in sec["category_breakdown"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{overall_cls}'>Overall: {sec['overall_compliance_pct']:.1f}% - "
            f"{sec['overall_status']}</p>\n"
            f"<table><tr><th>Category</th><th>Total</th><th>Compliant</th>"
            f"<th>Status</th></tr>{cat_rows}</table>"
        )

    def _html_missing_elements(self, data: Dict[str, Any]) -> str:
        """Render missing elements HTML."""
        sec = self._section_missing_elements(data)
        items = ""
        for gap in sec["missing_elements"]:
            items += f"<li class='fail'>{gap['id']}: {gap['element']} ({gap['article']})</li>"
        for gap in sec["non_compliant_elements"]:
            items += f"<li class='warn'>{gap['id']}: {gap['element']} ({gap['article']})</li>"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Gaps: {sec['total_gaps']}</p>\n"
            f"<ul>{items}</ul>"
        )

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        """Render corrective actions HTML."""
        sec = self._section_corrective_actions(data)
        rows = "".join(
            f"<tr><td>{a['rank']}</td><td>{a['priority']}</td>"
            f"<td>{a['element_id']}</td><td>{a['action']}</td>"
            f"<td>{a['estimated_effort']}</td></tr>"
            for a in sec["actions"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>#</th><th>Priority</th><th>Element</th>"
            f"<th>Action</th><th>Effort</th></tr>{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_applicable(
        self,
        element: Dict[str, Any],
        battery_type: str,
        has_hazardous: bool,
    ) -> bool:
        """Determine if a labelling element is applicable to this battery type."""
        applicable_to = element.get("applicable_to", [])
        if "all" in applicable_to:
            return True
        if "all_if_applicable" in applicable_to:
            return has_hazardous
        if "rechargeable" in applicable_to and battery_type in (
            "ev_battery", "industrial_battery", "lmt_battery",
        ):
            return True
        return battery_type in applicable_to

    def _generate_corrective_action(self, gap: Dict[str, Any]) -> str:
        """Generate a deterministic corrective action description."""
        if gap["gap_type"] == "missing":
            return f"Add {gap['element']} to battery label per {gap['article']}"
        return f"Update {gap['element']} to meet {gap['article']} requirements"

    def _estimate_effort(self, gap: Dict[str, Any]) -> str:
        """Estimate remediation effort based on element category."""
        effort_map = {
            "general": "1-2 days",
            "capacity": "1 day",
            "hazardous": "2-3 days",
            "collection": "1 day",
            "carbon_footprint": "3-5 days",
            "digital": "5-10 days",
            "safety": "2-3 days",
        }
        return effort_map.get(gap.get("category", ""), "2-3 days")

    def _effort_to_days(self, effort_str: str) -> int:
        """Convert effort string to maximum day estimate."""
        effort_map = {
            "1 day": 1,
            "1-2 days": 2,
            "2-3 days": 3,
            "3-5 days": 5,
            "5-10 days": 10,
        }
        return effort_map.get(effort_str, 3)

    def _recommend_deadline(self, gap: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Recommend a deadline for corrective action completion."""
        target_date = data.get("target_compliance_date", "")
        if target_date:
            return target_date
        if gap.get("category") == "digital":
            return "2027-02-18"
        if gap.get("category") == "carbon_footprint":
            return "2026-08-18"
        return "As soon as practicable"
