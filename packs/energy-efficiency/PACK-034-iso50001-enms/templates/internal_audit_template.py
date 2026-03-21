# -*- coding: utf-8 -*-
"""
InternalAuditTemplate - ISO 50001 Clause 9.2 Internal Audit for PACK-034.

Generates comprehensive internal audit reports aligned with ISO 50001:2018
Clause 9.2. Covers audit information (scope, criteria, team), audit plan,
clause-by-clause assessment (Clauses 4-10 with sub-clauses), findings
summary table, nonconformity reports, opportunities for improvement,
corrective action register, audit conclusions, and follow-up schedule.

Sections:
    1. Audit Information
    2. Audit Plan
    3. Clause-by-Clause Assessment
    4. Findings Summary
    5. Nonconformity Reports
    6. Opportunities for Improvement
    7. Corrective Action Register
    8. Audit Conclusions
    9. Follow-up Schedule

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InternalAuditTemplate:
    """
    ISO 50001 internal audit report template.

    Renders internal audit reports aligned with ISO 50001:2018 Clause 9.2,
    covering clause-by-clause conformity assessment, nonconformity reports,
    corrective action tracking, and follow-up scheduling across markdown,
    HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    CLAUSE_NAMES: Dict[str, str] = {
        "4": "Context of the Organization",
        "4.1": "Understanding the organization and its context",
        "4.2": "Understanding the needs and expectations of interested parties",
        "4.3": "Determining the scope of the EnMS",
        "4.4": "Energy management system",
        "5": "Leadership",
        "5.1": "Leadership and commitment",
        "5.2": "Energy policy",
        "5.3": "Organization roles, responsibilities and authorities",
        "6": "Planning",
        "6.1": "Actions to address risks and opportunities",
        "6.2": "Objectives, energy targets, and action plans",
        "6.3": "Energy review",
        "6.4": "Energy performance indicators",
        "6.5": "Energy baseline",
        "6.6": "Planning for collection of energy data",
        "7": "Support",
        "7.1": "Resources",
        "7.2": "Competence",
        "7.3": "Awareness",
        "7.4": "Communication",
        "7.5": "Documented information",
        "8": "Operation",
        "8.1": "Operational planning and control",
        "8.2": "Design",
        "8.3": "Procurement",
        "9": "Performance Evaluation",
        "9.1": "Monitoring, measurement, analysis and evaluation",
        "9.2": "Internal audit",
        "9.3": "Management review",
        "10": "Improvement",
        "10.1": "Nonconformity and corrective action",
        "10.2": "Continual improvement",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize InternalAuditTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render internal audit report as Markdown.

        Args:
            data: Audit data including audit_info, clause_assessments,
                  findings, nonconformities, and corrective_actions.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_audit_information(data),
            self._md_audit_plan(data),
            self._md_clause_assessment(data),
            self._md_findings_summary(data),
            self._md_nonconformity_reports(data),
            self._md_opportunities(data),
            self._md_corrective_action_register(data),
            self._md_audit_conclusions(data),
            self._md_followup_schedule(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render internal audit report as self-contained HTML.

        Args:
            data: Audit data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_audit_information(data),
            self._html_clause_assessment(data),
            self._html_findings_summary(data),
            self._html_nonconformity_reports(data),
            self._html_corrective_action_register(data),
            self._html_audit_conclusions(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Internal Audit Report - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render internal audit report as structured JSON.

        Args:
            data: Audit data dict.

        Returns:
            Dict with structured audit sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "internal_audit",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "audit_info": data.get("audit_info", {}),
            "audit_plan": data.get("audit_plan", []),
            "clause_assessments": data.get("clause_assessments", []),
            "findings_summary": self._json_findings_summary(data),
            "nonconformities": data.get("nonconformities", []),
            "opportunities": data.get("opportunities", []),
            "corrective_actions": data.get("corrective_actions", []),
            "conclusions": data.get("conclusions", {}),
            "followup_schedule": data.get("followup_schedule", []),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with audit metadata."""
        info = data.get("audit_info", {})
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Internal Audit Report\n\n"
            f"**Organization:** {info.get('organization', '')}  \n"
            f"**Audit Number:** {info.get('audit_number', '')}  \n"
            f"**Audit Date(s):** {info.get('audit_dates', '')}  \n"
            f"**ISO 50001:2018 Clause:** 9.2  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 InternalAuditTemplate v34.0.0\n\n---"
        )

    def _md_audit_information(self, data: Dict[str, Any]) -> str:
        """Render audit information section."""
        info = data.get("audit_info", {})
        team = info.get("audit_team", [])
        lines = [
            "## 1. Audit Information\n",
            f"**Audit Scope:** {info.get('scope', 'Full EnMS scope')}  ",
            f"**Audit Criteria:** {info.get('criteria', 'ISO 50001:2018')}  ",
            f"**Audit Type:** {info.get('audit_type', 'Surveillance')}  ",
            f"**Lead Auditor:** {info.get('lead_auditor', '-')}  ",
            f"**Auditee Representative:** {info.get('auditee_rep', '-')}\n",
        ]
        if team:
            lines.extend([
                "### Audit Team\n",
                "| Name | Role | Qualifications |",
                "|------|------|---------------|",
            ])
            for member in team:
                lines.append(
                    f"| {member.get('name', '-')} "
                    f"| {member.get('role', '-')} "
                    f"| {member.get('qualifications', '-')} |"
                )
        return "\n".join(lines)

    def _md_audit_plan(self, data: Dict[str, Any]) -> str:
        """Render audit plan section."""
        plan = data.get("audit_plan", [])
        if not plan:
            return "## 2. Audit Plan\n\n_Audit plan not provided._"
        lines = [
            "## 2. Audit Plan\n",
            "| Time | Area / Process | Clauses | Auditor | Auditee |",
            "|------|---------------|---------|---------|---------|",
        ]
        for p in plan:
            clauses = ", ".join(p.get("clauses", []))
            lines.append(
                f"| {p.get('time', '-')} "
                f"| {p.get('area', '-')} "
                f"| {clauses} "
                f"| {p.get('auditor', '-')} "
                f"| {p.get('auditee', '-')} |"
            )
        return "\n".join(lines)

    def _md_clause_assessment(self, data: Dict[str, Any]) -> str:
        """Render clause-by-clause assessment section."""
        assessments = data.get("clause_assessments", [])
        if not assessments:
            return "## 3. Clause-by-Clause Assessment\n\n_No assessments recorded._"
        lines = [
            "## 3. Clause-by-Clause Assessment\n",
            "| Clause | Title | Conformity | Evidence | Findings |",
            "|--------|-------|-----------|----------|----------|",
        ]
        for a in assessments:
            clause = a.get("clause", "")
            title = a.get("title", self.CLAUSE_NAMES.get(clause, ""))
            conformity = a.get("conformity", "Conforming")
            lines.append(
                f"| {clause} | {title} "
                f"| {conformity} "
                f"| {a.get('evidence', '-')} "
                f"| {a.get('findings', '-')} |"
            )
        return "\n".join(lines)

    def _md_findings_summary(self, data: Dict[str, Any]) -> str:
        """Render findings summary table."""
        findings = data.get("findings", [])
        ncs = data.get("nonconformities", [])
        opps = data.get("opportunities", [])
        major_nc = sum(1 for n in ncs if n.get("severity", "").lower() == "major")
        minor_nc = sum(1 for n in ncs if n.get("severity", "").lower() == "minor")
        return (
            "## 4. Findings Summary\n\n"
            "| Category | Count |\n|----------|-------|\n"
            f"| Major Nonconformities | {major_nc} |\n"
            f"| Minor Nonconformities | {minor_nc} |\n"
            f"| Observations | {len(findings)} |\n"
            f"| Opportunities for Improvement | {len(opps)} |\n"
            f"| Total Findings | {major_nc + minor_nc + len(findings) + len(opps)} |"
        )

    def _md_nonconformity_reports(self, data: Dict[str, Any]) -> str:
        """Render nonconformity reports section."""
        ncs = data.get("nonconformities", [])
        if not ncs:
            return "## 5. Nonconformity Reports\n\n_No nonconformities identified._"
        lines = ["## 5. Nonconformity Reports\n"]
        for i, nc in enumerate(ncs, 1):
            lines.extend([
                f"### NC-{nc.get('id', i):03d}: {nc.get('title', 'Nonconformity')}",
                "",
                f"- **Clause Reference:** {nc.get('clause_ref', '-')}",
                f"- **Severity:** {nc.get('severity', '-')}",
                f"- **Area/Process:** {nc.get('area', '-')}",
                f"- **Description:** {nc.get('description', '-')}",
                f"- **Objective Evidence:** {nc.get('evidence', '-')}",
                f"- **Root Cause:** {nc.get('root_cause', 'To be determined')}",
                f"- **Correction:** {nc.get('correction', 'To be determined')}",
                f"- **Corrective Action:** {nc.get('corrective_action', 'To be determined')}",
                f"- **Due Date:** {nc.get('due_date', '-')}",
                f"- **Responsible:** {nc.get('responsible', '-')}",
                f"- **Status:** {nc.get('status', 'Open')}",
                "",
            ])
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        """Render opportunities for improvement section."""
        opps = data.get("opportunities", [])
        if not opps:
            return "## 6. Opportunities for Improvement\n\n_No opportunities identified._"
        lines = [
            "## 6. Opportunities for Improvement\n",
            "| # | Area | Clause | Opportunity | Potential Benefit |",
            "|---|------|--------|------------|------------------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('area', '-')} "
                f"| {o.get('clause_ref', '-')} "
                f"| {o.get('opportunity', '-')} "
                f"| {o.get('potential_benefit', '-')} |"
            )
        return "\n".join(lines)

    def _md_corrective_action_register(self, data: Dict[str, Any]) -> str:
        """Render corrective action register section."""
        actions = data.get("corrective_actions", [])
        if not actions:
            return "## 7. Corrective Action Register\n\n_No corrective actions required._"
        lines = [
            "## 7. Corrective Action Register\n",
            "| NC Ref | Action | Responsible | Due Date | Verified By | Status |",
            "|--------|--------|-------------|----------|------------|--------|",
        ]
        for a in actions:
            lines.append(
                f"| NC-{a.get('nc_ref', '-'):03d} "
                f"| {a.get('action', '-')} "
                f"| {a.get('responsible', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {a.get('verified_by', '-')} "
                f"| {a.get('status', 'Open')} |"
            )
        return "\n".join(lines)

    def _md_audit_conclusions(self, data: Dict[str, Any]) -> str:
        """Render audit conclusions section."""
        conclusions = data.get("conclusions", {})
        lines = [
            "## 8. Audit Conclusions\n",
            f"**Overall Conformity Assessment:** {conclusions.get('overall_assessment', 'To be determined')}  ",
            f"**EnMS Effectiveness:** {conclusions.get('enms_effectiveness', 'To be determined')}  ",
            f"**Energy Performance Improvement:** {conclusions.get('energy_improvement', 'To be determined')}  ",
            f"**Recommendation:** {conclusions.get('recommendation', 'To be determined')}\n",
        ]
        remarks = conclusions.get("remarks", [])
        if remarks:
            lines.append("### Auditor Remarks\n")
            for r in remarks:
                lines.append(f"- {r}")
        return "\n".join(lines)

    def _md_followup_schedule(self, data: Dict[str, Any]) -> str:
        """Render follow-up schedule section."""
        schedule = data.get("followup_schedule", [])
        if not schedule:
            return "## 9. Follow-up Schedule\n\n_Follow-up schedule to be determined._"
        lines = [
            "## 9. Follow-up Schedule\n",
            "| NC Ref | Verification Method | Follow-up Date | Auditor | Status |",
            "|--------|-------------------|---------------|---------|--------|",
        ]
        for s in schedule:
            lines.append(
                f"| NC-{s.get('nc_ref', '-'):03d} "
                f"| {s.get('verification_method', '-')} "
                f"| {s.get('followup_date', '-')} "
                f"| {s.get('auditor', '-')} "
                f"| {s.get('status', 'Pending')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        info = data.get("audit_info", {})
        return (
            f'<h1>Internal Audit Report</h1>\n'
            f'<p class="subtitle">Organization: {info.get("organization", "-")} | '
            f'Audit: {info.get("audit_number", "-")} | '
            f'ISO 50001 Clause 9.2</p>'
        )

    def _html_audit_information(self, data: Dict[str, Any]) -> str:
        """Render HTML audit information."""
        info = data.get("audit_info", {})
        return (
            '<h2>1. Audit Information</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Scope</span>'
            f'<span class="value-sm">{info.get("scope", "Full EnMS")}</span></div>\n'
            f'  <div class="card"><span class="label">Lead Auditor</span>'
            f'<span class="value-sm">{info.get("lead_auditor", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Audit Type</span>'
            f'<span class="value-sm">{info.get("audit_type", "Surveillance")}</span></div>\n'
            f'  <div class="card"><span class="label">Date(s)</span>'
            f'<span class="value-sm">{info.get("audit_dates", "-")}</span></div>\n'
            '</div>'
        )

    def _html_clause_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML clause-by-clause assessment."""
        assessments = data.get("clause_assessments", [])
        rows = ""
        for a in assessments:
            clause = a.get("clause", "")
            title = a.get("title", self.CLAUSE_NAMES.get(clause, ""))
            conformity = a.get("conformity", "Conforming")
            cls = self._conformity_class(conformity)
            rows += (
                f'<tr><td>{clause}</td>'
                f'<td>{title}</td>'
                f'<td class="{cls}">{conformity}</td>'
                f'<td>{a.get("findings", "-")}</td></tr>\n'
            )
        return (
            '<h2>3. Clause-by-Clause Assessment</h2>\n'
            '<table>\n<tr><th>Clause</th><th>Title</th>'
            f'<th>Conformity</th><th>Findings</th></tr>\n{rows}</table>'
        )

    def _html_findings_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML findings summary."""
        ncs = data.get("nonconformities", [])
        opps = data.get("opportunities", [])
        major = sum(1 for n in ncs if n.get("severity", "").lower() == "major")
        minor = sum(1 for n in ncs if n.get("severity", "").lower() == "minor")
        return (
            '<h2>4. Findings Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-danger"><span class="label">Major NC</span>'
            f'<span class="value">{major}</span></div>\n'
            f'  <div class="card card-warning"><span class="label">Minor NC</span>'
            f'<span class="value">{minor}</span></div>\n'
            f'  <div class="card"><span class="label">OFIs</span>'
            f'<span class="value">{len(opps)}</span></div>\n'
            f'  <div class="card"><span class="label">Total</span>'
            f'<span class="value">{major + minor + len(opps)}</span></div>\n'
            '</div>'
        )

    def _html_nonconformity_reports(self, data: Dict[str, Any]) -> str:
        """Render HTML nonconformity reports."""
        ncs = data.get("nonconformities", [])
        parts = ['<h2>5. Nonconformity Reports</h2>\n']
        for i, nc in enumerate(ncs, 1):
            severity_cls = "nc-major" if nc.get("severity", "").lower() == "major" else "nc-minor"
            parts.append(
                f'<div class="nc-report {severity_cls}">\n'
                f'<h3>NC-{nc.get("id", i):03d}: {nc.get("title", "Nonconformity")}</h3>\n'
                f'<p><strong>Clause:</strong> {nc.get("clause_ref", "-")} | '
                f'<strong>Severity:</strong> {nc.get("severity", "-")} | '
                f'<strong>Status:</strong> {nc.get("status", "Open")}</p>\n'
                f'<p><strong>Description:</strong> {nc.get("description", "-")}</p>\n'
                f'<p><strong>Root Cause:</strong> {nc.get("root_cause", "TBD")}</p>\n'
                f'<p><strong>Corrective Action:</strong> {nc.get("corrective_action", "TBD")}</p>\n'
                '</div>\n'
            )
        return "".join(parts)

    def _html_corrective_action_register(self, data: Dict[str, Any]) -> str:
        """Render HTML corrective action register."""
        actions = data.get("corrective_actions", [])
        rows = ""
        for a in actions:
            status_cls = "status-closed" if a.get("status", "").lower() == "closed" else "status-open"
            rows += (
                f'<tr><td>NC-{a.get("nc_ref", "-"):03d}</td>'
                f'<td>{a.get("action", "-")}</td>'
                f'<td>{a.get("responsible", "-")}</td>'
                f'<td>{a.get("due_date", "-")}</td>'
                f'<td class="{status_cls}">{a.get("status", "Open")}</td></tr>\n'
            )
        return (
            '<h2>7. Corrective Action Register</h2>\n'
            '<table>\n<tr><th>NC Ref</th><th>Action</th>'
            f'<th>Responsible</th><th>Due Date</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_audit_conclusions(self, data: Dict[str, Any]) -> str:
        """Render HTML audit conclusions."""
        conclusions = data.get("conclusions", {})
        return (
            '<h2>8. Audit Conclusions</h2>\n'
            '<div class="conclusion-box">\n'
            f'<p><strong>Overall Assessment:</strong> {conclusions.get("overall_assessment", "TBD")}</p>\n'
            f'<p><strong>EnMS Effectiveness:</strong> {conclusions.get("enms_effectiveness", "TBD")}</p>\n'
            f'<p><strong>Recommendation:</strong> {conclusions.get("recommendation", "TBD")}</p>\n'
            '</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_findings_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON findings summary."""
        ncs = data.get("nonconformities", [])
        findings = data.get("findings", [])
        opps = data.get("opportunities", [])
        return {
            "major_nonconformities": sum(
                1 for n in ncs if n.get("severity", "").lower() == "major"
            ),
            "minor_nonconformities": sum(
                1 for n in ncs if n.get("severity", "").lower() == "minor"
            ),
            "observations": len(findings),
            "opportunities": len(opps),
            "total_findings": len(ncs) + len(findings) + len(opps),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _conformity_class(self, conformity: str) -> str:
        """Return CSS class for conformity status."""
        c = conformity.lower()
        if c in ("conforming", "conforms", "pass"):
            return "conforming"
        if c in ("minor nonconformity", "minor nc", "minor"):
            return "minor-nc"
        if c in ("major nonconformity", "major nc", "major"):
            return "major-nc"
        if c in ("not applicable", "n/a"):
            return "not-applicable"
        return ""

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "h3{color:#495057;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:120px;}"
            ".card-danger{border-left:4px solid #dc3545;}"
            ".card-warning{border-left:4px solid #ffc107;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".value-sm{display:block;font-size:1em;font-weight:600;color:#495057;}"
            ".conforming{color:#198754;font-weight:600;}"
            ".minor-nc{color:#fd7e14;font-weight:600;}"
            ".major-nc{color:#dc3545;font-weight:700;}"
            ".not-applicable{color:#6c757d;font-style:italic;}"
            ".nc-report{border:1px solid #dee2e6;border-radius:8px;padding:15px;margin:10px 0;}"
            ".nc-major{border-left:4px solid #dc3545;}"
            ".nc-minor{border-left:4px solid #fd7e14;}"
            ".status-open{color:#dc3545;font-weight:600;}"
            ".status-closed{color:#198754;font-weight:600;}"
            ".conclusion-box{background:#e8f5e9;border-left:4px solid #198754;padding:15px;border-radius:4px;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
