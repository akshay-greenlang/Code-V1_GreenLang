# -*- coding: utf-8 -*-
"""
RegulatoryComplianceReportTemplate - EPBD/MEES compliance report for PACK-032.

Generates regulatory compliance reports covering building energy
regulations including EPBD recast, MEES (UK), local planning
conditions, applicable regulations mapping, current ratings vs
minimum requirements, compliance timeline tracking, penalty
exposure analysis, required improvements, and action plans.

Sections:
    1. Compliance Status Summary
    2. Applicable Regulations
    3. Current Ratings
    4. Minimum Requirements
    5. Compliance Timeline
    6. Penalty Exposure
    7. Required Improvements
    8. Action Plan
    9. Provenance

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegulatoryComplianceReportTemplate:
    """
    Building energy regulatory compliance report template.

    Renders compliance reports for EPBD, MEES, and other building
    energy regulations with ratings, timelines, penalties, required
    improvements, and action plans across markdown, HTML, and JSON.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    COMPLIANCE_SECTIONS: List[str] = [
        "Compliance Summary",
        "Applicable Regulations",
        "Current Ratings",
        "Minimum Requirements",
        "Compliance Timeline",
        "Penalty Exposure",
        "Required Improvements",
        "Action Plan",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryComplianceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render compliance report as Markdown.

        Args:
            data: Compliance assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_compliance_summary(data),
            self._md_applicable_regulations(data),
            self._md_current_ratings(data),
            self._md_minimum_requirements(data),
            self._md_compliance_timeline(data),
            self._md_penalty_exposure(data),
            self._md_required_improvements(data),
            self._md_action_plan(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render compliance report as self-contained HTML.

        Args:
            data: Compliance assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_compliance_summary(data),
            self._html_applicable_regulations(data),
            self._html_current_ratings(data),
            self._html_minimum_requirements(data),
            self._html_compliance_timeline(data),
            self._html_penalty_exposure(data),
            self._html_required_improvements(data),
            self._html_action_plan(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Regulatory Compliance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render compliance report as structured JSON.

        Args:
            data: Compliance assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "regulatory_compliance_report",
            "version": "32.0.0",
            "generated_at": self.generated_at.isoformat(),
            "compliance_summary": self._json_compliance_summary(data),
            "applicable_regulations": data.get("applicable_regulations", []),
            "current_ratings": data.get("current_ratings", {}),
            "minimum_requirements": data.get("minimum_requirements", []),
            "compliance_timeline": data.get("compliance_timeline", []),
            "penalty_exposure": data.get("penalty_exposure", {}),
            "required_improvements": data.get("required_improvements", []),
            "action_plan": data.get("action_plan", []),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Regulatory Compliance Report\n\n"
            f"**Building:** {name}  \n"
            f"**Address:** {data.get('address', '-')}  \n"
            f"**Jurisdiction:** {data.get('jurisdiction', '-')}  \n"
            f"**Assessment Date:** {data.get('assessment_date', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-032 RegulatoryComplianceReportTemplate v32.0.0\n\n---"
        )

    def _md_compliance_summary(self, data: Dict[str, Any]) -> str:
        """Render compliance status summary section."""
        s = data.get("compliance_summary", {})
        overall = s.get("overall_status", "Unknown")
        return (
            "## 1. Compliance Status Summary\n\n"
            f"**Overall Status:** {overall}\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Regulations Applicable | {s.get('regulations_applicable', 0)} |\n"
            f"| Currently Compliant | {s.get('currently_compliant', 0)} |\n"
            f"| Non-Compliant | {s.get('non_compliant', 0)} |\n"
            f"| At Risk (Future) | {s.get('at_risk', 0)} |\n"
            f"| Next Deadline | {s.get('next_deadline', '-')} |\n"
            f"| Total Penalty Exposure | {s.get('total_penalty_exposure', '-')} |\n"
            f"| Improvements Required | {s.get('improvements_required', 0)} |"
        )

    def _md_applicable_regulations(self, data: Dict[str, Any]) -> str:
        """Render applicable regulations section."""
        regs = data.get("applicable_regulations", [])
        if not regs:
            return "## 2. Applicable Regulations\n\n_No applicable regulations identified._"
        lines = [
            "## 2. Applicable Regulations\n",
            "| Regulation | Scope | Effective Date | Status | Priority |",
            "|------------|-------|---------------|--------|----------|",
        ]
        for r in regs:
            lines.append(
                f"| {r.get('regulation', '-')} "
                f"| {r.get('scope', '-')} "
                f"| {r.get('effective_date', '-')} "
                f"| {r.get('status', '-')} "
                f"| {r.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_current_ratings(self, data: Dict[str, Any]) -> str:
        """Render current ratings section."""
        ratings = data.get("current_ratings", {})
        certs = ratings.get("certificates", [])
        lines = [
            "## 3. Current Ratings\n",
            f"**EPC Rating:** {ratings.get('epc_rating', '-')} "
            f"(Score: {ratings.get('epc_score', '-')})  ",
            f"**EPC Valid Until:** {ratings.get('epc_valid_until', '-')}  ",
            f"**DEC Rating:** {ratings.get('dec_rating', '-')} "
            f"(Score: {ratings.get('dec_score', '-')})  ",
            f"**DEC Valid Until:** {ratings.get('dec_valid_until', '-')}  ",
            f"**Air Con Inspection:** {ratings.get('ac_inspection_status', '-')}  ",
            f"**AC Inspection Due:** {ratings.get('ac_inspection_due', '-')}",
        ]
        if certs:
            lines.extend([
                "\n### Active Certificates\n",
                "| Certificate | Rating | Valid Until | Status |",
                "|------------|--------|------------|--------|",
            ])
            for c in certs:
                lines.append(
                    f"| {c.get('certificate', '-')} "
                    f"| {c.get('rating', '-')} "
                    f"| {c.get('valid_until', '-')} "
                    f"| {c.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_minimum_requirements(self, data: Dict[str, Any]) -> str:
        """Render minimum requirements section."""
        reqs = data.get("minimum_requirements", [])
        if not reqs:
            return "## 4. Minimum Requirements\n\n_No minimum requirements._"
        lines = [
            "## 4. Minimum Requirements\n",
            "| Regulation | Requirement | Current | Gap | Compliant |",
            "|------------|------------|---------|-----|-----------|",
        ]
        for r in reqs:
            lines.append(
                f"| {r.get('regulation', '-')} "
                f"| {r.get('requirement', '-')} "
                f"| {r.get('current_value', '-')} "
                f"| {r.get('gap', '-')} "
                f"| {r.get('compliant', '-')} |"
            )
        return "\n".join(lines)

    def _md_compliance_timeline(self, data: Dict[str, Any]) -> str:
        """Render compliance timeline section."""
        timeline = data.get("compliance_timeline", [])
        if not timeline:
            return "## 5. Compliance Timeline\n\n_No timeline events._"
        lines = [
            "## 5. Compliance Timeline\n",
            "| Date | Regulation | Event | Impact | Action Required |",
            "|------|-----------|-------|--------|----------------|",
        ]
        for t in timeline:
            lines.append(
                f"| {t.get('date', '-')} "
                f"| {t.get('regulation', '-')} "
                f"| {t.get('event', '-')} "
                f"| {t.get('impact', '-')} "
                f"| {t.get('action_required', '-')} |"
            )
        return "\n".join(lines)

    def _md_penalty_exposure(self, data: Dict[str, Any]) -> str:
        """Render penalty exposure section."""
        pe = data.get("penalty_exposure", {})
        penalties = pe.get("penalties", [])
        lines = [
            "## 6. Penalty Exposure\n",
            f"**Total Annual Exposure:** {pe.get('total_annual', '-')}  ",
            f"**Maximum One-Off Penalty:** {pe.get('max_one_off', '-')}  ",
            f"**Current Penalties Active:** {pe.get('active_penalties', 0)}  ",
            f"**Grace Period Remaining:** {pe.get('grace_period', '-')}",
        ]
        if penalties:
            lines.extend([
                "\n### Penalty Breakdown\n",
                "| Regulation | Penalty Type | Amount | Trigger Date | Avoidable |",
                "|------------|-------------|--------|-------------|-----------|",
            ])
            for p in penalties:
                lines.append(
                    f"| {p.get('regulation', '-')} "
                    f"| {p.get('type', '-')} "
                    f"| {p.get('amount', '-')} "
                    f"| {p.get('trigger_date', '-')} "
                    f"| {p.get('avoidable', '-')} |"
                )
        return "\n".join(lines)

    def _md_required_improvements(self, data: Dict[str, Any]) -> str:
        """Render required improvements section."""
        improvements = data.get("required_improvements", [])
        if not improvements:
            return "## 7. Required Improvements\n\n_No improvements required._"
        lines = [
            "## 7. Required Improvements\n",
            "| # | Improvement | Regulation | Deadline | Cost | Rating Impact |",
            "|---|-----------|-----------|----------|------|--------------|",
        ]
        for i, imp in enumerate(improvements, 1):
            lines.append(
                f"| {i} | {imp.get('improvement', '-')} "
                f"| {imp.get('regulation', '-')} "
                f"| {imp.get('deadline', '-')} "
                f"| {imp.get('cost', '-')} "
                f"| {imp.get('rating_impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_action_plan(self, data: Dict[str, Any]) -> str:
        """Render action plan section."""
        actions = data.get("action_plan", [])
        if not actions:
            return "## 8. Action Plan\n\n_No actions planned._"
        lines = [
            "## 8. Action Plan\n",
            "| # | Action | Regulation | Owner | Start | Complete By | Status |",
            "|---|--------|-----------|-------|-------|-----------|--------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('regulation', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('start', '-')} "
                f"| {a.get('complete_by', '-')} "
                f"| {a.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "---\n\n"
            f"*Report generated by PACK-032 RegulatoryComplianceReportTemplate v32.0.0 on {ts}*"
        )

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        name = data.get("building_name", "Building")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Regulatory Compliance Report</h1>\n'
            f'<p class="subtitle">Building: {name} | '
            f'Jurisdiction: {data.get("jurisdiction", "-")} | Generated: {ts}</p>'
        )

    def _html_compliance_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance summary."""
        s = data.get("compliance_summary", {})
        overall = s.get("overall_status", "Unknown")
        color = "#198754" if overall == "Compliant" else "#dc3545"
        return (
            '<h2>Compliance Status Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'<div class="card"><span class="label">Status</span>'
            f'<span class="value" style="color:{color}">{overall}</span></div>\n'
            f'<div class="card"><span class="label">Compliant</span>'
            f'<span class="value">{s.get("currently_compliant", 0)}</span></div>\n'
            f'<div class="card"><span class="label">Non-Compliant</span>'
            f'<span class="value" style="color:#dc3545">{s.get("non_compliant", 0)}</span></div>\n'
            f'<div class="card"><span class="label">Penalty Exposure</span>'
            f'<span class="value">{s.get("total_penalty_exposure", "-")}</span></div>\n'
            '</div>'
        )

    def _html_applicable_regulations(self, data: Dict[str, Any]) -> str:
        """Render HTML applicable regulations."""
        regs = data.get("applicable_regulations", [])
        rows = ""
        for r in regs:
            rows += (
                f'<tr><td>{r.get("regulation", "-")}</td>'
                f'<td>{r.get("scope", "-")}</td>'
                f'<td>{r.get("effective_date", "-")}</td>'
                f'<td>{r.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Applicable Regulations</h2>\n'
            '<table>\n<tr><th>Regulation</th><th>Scope</th><th>Effective</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_current_ratings(self, data: Dict[str, Any]) -> str:
        """Render HTML current ratings."""
        ratings = data.get("current_ratings", {})
        return (
            '<h2>Current Ratings</h2>\n'
            '<table>\n<tr><th>Certificate</th><th>Rating</th><th>Valid Until</th></tr>\n'
            f'<tr><td>EPC</td><td>{ratings.get("epc_rating", "-")}</td>'
            f'<td>{ratings.get("epc_valid_until", "-")}</td></tr>\n'
            f'<tr><td>DEC</td><td>{ratings.get("dec_rating", "-")}</td>'
            f'<td>{ratings.get("dec_valid_until", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_minimum_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML minimum requirements."""
        reqs = data.get("minimum_requirements", [])
        rows = ""
        for r in reqs:
            compliant = r.get("compliant", "-")
            style = 'color:#198754' if compliant == "Yes" else 'color:#dc3545'
            rows += (
                f'<tr><td>{r.get("regulation", "-")}</td>'
                f'<td>{r.get("requirement", "-")}</td>'
                f'<td>{r.get("current_value", "-")}</td>'
                f'<td style="{style};font-weight:bold">{compliant}</td></tr>\n'
            )
        return (
            '<h2>Minimum Requirements</h2>\n'
            '<table>\n<tr><th>Regulation</th><th>Requirement</th><th>Current</th>'
            f'<th>Compliant</th></tr>\n{rows}</table>'
        )

    def _html_compliance_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance timeline."""
        timeline = data.get("compliance_timeline", [])
        rows = ""
        for t in timeline:
            rows += (
                f'<tr><td>{t.get("date", "-")}</td>'
                f'<td>{t.get("regulation", "-")}</td>'
                f'<td>{t.get("event", "-")}</td>'
                f'<td>{t.get("impact", "-")}</td></tr>\n'
            )
        return (
            '<h2>Compliance Timeline</h2>\n'
            '<table>\n<tr><th>Date</th><th>Regulation</th><th>Event</th>'
            f'<th>Impact</th></tr>\n{rows}</table>'
        )

    def _html_penalty_exposure(self, data: Dict[str, Any]) -> str:
        """Render HTML penalty exposure."""
        pe = data.get("penalty_exposure", {})
        penalties = pe.get("penalties", [])
        rows = ""
        for p in penalties:
            rows += (
                f'<tr><td>{p.get("regulation", "-")}</td>'
                f'<td>{p.get("type", "-")}</td>'
                f'<td>{p.get("amount", "-")}</td>'
                f'<td>{p.get("trigger_date", "-")}</td></tr>\n'
            )
        return (
            '<h2>Penalty Exposure</h2>\n'
            f'<p>Total Annual: {pe.get("total_annual", "-")} | '
            f'Max One-Off: {pe.get("max_one_off", "-")}</p>\n'
            '<table>\n<tr><th>Regulation</th><th>Type</th><th>Amount</th>'
            f'<th>Trigger</th></tr>\n{rows}</table>'
        )

    def _html_required_improvements(self, data: Dict[str, Any]) -> str:
        """Render HTML required improvements."""
        improvements = data.get("required_improvements", [])
        rows = ""
        for i, imp in enumerate(improvements, 1):
            rows += (
                f'<tr><td>{i}</td><td>{imp.get("improvement", "-")}</td>'
                f'<td>{imp.get("regulation", "-")}</td>'
                f'<td>{imp.get("deadline", "-")}</td>'
                f'<td>{imp.get("cost", "-")}</td></tr>\n'
            )
        return (
            '<h2>Required Improvements</h2>\n'
            '<table>\n<tr><th>#</th><th>Improvement</th><th>Regulation</th>'
            f'<th>Deadline</th><th>Cost</th></tr>\n{rows}</table>'
        )

    def _html_action_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML action plan."""
        actions = data.get("action_plan", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            rows += (
                f'<tr><td>{i}</td><td>{a.get("action", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td>'
                f'<td>{a.get("complete_by", "-")}</td>'
                f'<td>{a.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Action Plan</h2>\n'
            '<table>\n<tr><th>#</th><th>Action</th><th>Owner</th>'
            f'<th>Complete By</th><th>Status</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_compliance_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON compliance summary."""
        s = data.get("compliance_summary", {})
        return {
            "overall_status": s.get("overall_status", "Unknown"),
            "regulations_applicable": s.get("regulations_applicable", 0),
            "currently_compliant": s.get("currently_compliant", 0),
            "non_compliant": s.get("non_compliant", 0),
            "at_risk": s.get("at_risk", 0),
            "next_deadline": s.get("next_deadline", ""),
            "total_penalty_exposure": s.get("total_penalty_exposure", ""),
            "improvements_required": s.get("improvements_required", 0),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
