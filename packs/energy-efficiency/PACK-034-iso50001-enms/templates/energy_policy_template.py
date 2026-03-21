# -*- coding: utf-8 -*-
"""
EnergyPolicyTemplate - ISO 50001 Clause 5.2 Energy Policy for PACK-034.

Generates a comprehensive energy policy document aligned with ISO 50001:2018
Clause 5.2 requirements. Covers policy statement, scope, commitments to
continual improvement, legal compliance, EnPI improvement and information
availability, objectives framework, roles and responsibilities,
communication plan, and review schedule.

Sections:
    1. Policy Statement
    2. Scope & Boundaries
    3. Commitments
    4. Objectives Framework
    5. Roles & Responsibilities
    6. Communication Plan
    7. Review Schedule

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnergyPolicyTemplate:
    """
    ISO 50001 energy policy document template.

    Renders energy policy documents aligned with ISO 50001:2018 Clause 5.2,
    including policy commitments, objectives framework, roles, communication
    plan, and review schedule across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyPolicyTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy policy document as Markdown.

        Args:
            data: Energy policy data including organization_name,
                  policy_date, scope, commitments, objectives,
                  responsible_persons, and review_frequency.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_policy_statement(data),
            self._md_scope(data),
            self._md_commitments(data),
            self._md_objectives_framework(data),
            self._md_roles_responsibilities(data),
            self._md_communication_plan(data),
            self._md_review_schedule(data),
            self._md_approval(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy policy document as self-contained HTML.

        Args:
            data: Energy policy data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_policy_statement(data),
            self._html_scope(data),
            self._html_commitments(data),
            self._html_objectives_framework(data),
            self._html_roles_responsibilities(data),
            self._html_communication_plan(data),
            self._html_review_schedule(data),
            self._html_approval(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Policy - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy policy document as structured JSON.

        Args:
            data: Energy policy data dict.

        Returns:
            Dict with structured policy sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "energy_policy",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "organization_name": data.get("organization_name", ""),
            "policy_date": data.get("policy_date", ""),
            "policy_statement": self._json_policy_statement(data),
            "scope": self._json_scope(data),
            "commitments": data.get("commitments", []),
            "objectives_framework": data.get("objectives", []),
            "roles_responsibilities": data.get("responsible_persons", []),
            "communication_plan": data.get("communication_plan", []),
            "review_schedule": self._json_review_schedule(data),
            "approval": data.get("approval", {}),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with policy metadata."""
        org = data.get("organization_name", "Organization")
        policy_date = data.get("policy_date", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        version = data.get("policy_version", "1.0")
        return (
            f"# Energy Policy\n\n"
            f"**Organization:** {org}  \n"
            f"**Policy Date:** {policy_date}  \n"
            f"**Version:** {version}  \n"
            f"**ISO 50001:2018 Clause:** 5.2  \n"
            f"**Document Classification:** Controlled Document  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 EnergyPolicyTemplate v34.0.0\n\n---"
        )

    def _md_policy_statement(self, data: Dict[str, Any]) -> str:
        """Render policy statement section."""
        org = data.get("organization_name", "Organization")
        statement = data.get("policy_statement", "")
        if not statement:
            statement = (
                f"{org} is committed to continual improvement of energy "
                f"performance through an effective Energy Management System "
                f"(EnMS) conforming to ISO 50001:2018. This policy provides "
                f"a framework for setting and reviewing energy objectives "
                f"and energy targets."
            )
        lines = [
            "## 1. Policy Statement\n",
            statement,
            "",
            "This energy policy shall be:",
            "- Appropriate to the nature and scale of the organization's energy use",
            "- A framework for setting and reviewing energy objectives and targets",
            "- Committed to ensuring the availability of information and resources",
            "- Documented, communicated, and available to all relevant parties",
            "- Regularly reviewed and updated as necessary",
        ]
        return "\n".join(lines)

    def _md_scope(self, data: Dict[str, Any]) -> str:
        """Render scope and boundaries section."""
        scope = data.get("scope", {})
        boundaries = scope.get("boundaries", [])
        exclusions = scope.get("exclusions", [])
        lines = [
            "## 2. Scope & Boundaries\n",
            f"**Scope Description:** {scope.get('description', 'All organizational activities and facilities')}  ",
            f"**Organizational Boundary:** {scope.get('organizational_boundary', 'Operational control')}  ",
            f"**Physical Boundary:** {scope.get('physical_boundary', 'All owned and operated facilities')}  ",
        ]
        if boundaries:
            lines.append("\n### Included Boundaries\n")
            for b in boundaries:
                lines.append(f"- {b}")
        if exclusions:
            lines.append("\n### Exclusions\n")
            for e in exclusions:
                lines.append(f"- {e.get('item', e) if isinstance(e, dict) else e}")
                if isinstance(e, dict) and e.get("justification"):
                    lines.append(f"  - *Justification:* {e['justification']}")
        return "\n".join(lines)

    def _md_commitments(self, data: Dict[str, Any]) -> str:
        """Render commitments section aligned with ISO 50001 Clause 5.2."""
        commitments = data.get("commitments", [])
        if not commitments:
            commitments = [
                {
                    "id": "C1",
                    "title": "Continual Improvement of Energy Performance",
                    "description": "Continually improve energy performance, including energy efficiency, energy use, and energy consumption.",
                    "clause_ref": "5.2 a)",
                },
                {
                    "id": "C2",
                    "title": "Legal & Other Requirements Compliance",
                    "description": "Comply with applicable legal requirements and other requirements related to energy efficiency, energy use, energy consumption, and the EnMS.",
                    "clause_ref": "5.2 b)",
                },
                {
                    "id": "C3",
                    "title": "EnPI Improvement",
                    "description": "Support the design and procurement of energy-efficient products and services to improve energy performance indicators.",
                    "clause_ref": "5.2 c)",
                },
                {
                    "id": "C4",
                    "title": "Information & Resource Availability",
                    "description": "Ensure the availability of information and necessary resources to achieve energy objectives and energy targets.",
                    "clause_ref": "5.2 d)",
                },
            ]
        lines = [
            "## 3. Commitments\n",
            "The organization commits to the following in accordance with ISO 50001:2018:\n",
            "| ID | Commitment | ISO 50001 Ref | Description |",
            "|----|-----------|---------------|-------------|",
        ]
        for c in commitments:
            lines.append(
                f"| {c.get('id', '-')} | {c.get('title', '-')} "
                f"| {c.get('clause_ref', '-')} "
                f"| {c.get('description', '-')} |"
            )
        return "\n".join(lines)

    def _md_objectives_framework(self, data: Dict[str, Any]) -> str:
        """Render objectives framework section."""
        objectives = data.get("objectives", [])
        if not objectives:
            return "## 4. Objectives Framework\n\n_No objectives defined yet. Objectives shall be established at relevant functions and levels._"
        lines = [
            "## 4. Objectives Framework\n",
            "Energy objectives shall be consistent with the energy policy, measurable, "
            "and take into account applicable requirements and significant energy uses.\n",
            "| # | Objective | Target | EnPI | Timeline | Owner |",
            "|---|-----------|--------|------|----------|-------|",
        ]
        for i, obj in enumerate(objectives, 1):
            lines.append(
                f"| {i} | {obj.get('objective', '-')} "
                f"| {obj.get('target', '-')} "
                f"| {obj.get('enpi', '-')} "
                f"| {obj.get('timeline', '-')} "
                f"| {obj.get('owner', '-')} |"
            )
        return "\n".join(lines)

    def _md_roles_responsibilities(self, data: Dict[str, Any]) -> str:
        """Render roles and responsibilities section."""
        persons = data.get("responsible_persons", [])
        if not persons:
            persons = [
                {"role": "Top Management", "responsibility": "Overall accountability for EnMS effectiveness and energy policy", "clause_ref": "5.1"},
                {"role": "Energy Manager", "responsibility": "Day-to-day management of EnMS implementation and energy performance improvement", "clause_ref": "5.3"},
                {"role": "Energy Team", "responsibility": "Support energy review, EnPI monitoring, action plan implementation", "clause_ref": "5.3"},
            ]
        lines = [
            "## 5. Roles & Responsibilities\n",
            "| Role | Responsibility | ISO 50001 Ref |",
            "|------|---------------|---------------|",
        ]
        for p in persons:
            lines.append(
                f"| {p.get('role', '-')} "
                f"| {p.get('responsibility', '-')} "
                f"| {p.get('clause_ref', '-')} |"
            )
        return "\n".join(lines)

    def _md_communication_plan(self, data: Dict[str, Any]) -> str:
        """Render communication plan section."""
        plan = data.get("communication_plan", [])
        if not plan:
            plan = [
                {"audience": "All Employees", "method": "Intranet, notice boards, induction training", "frequency": "Ongoing", "content": "Energy policy, targets, performance updates"},
                {"audience": "Management", "method": "Management review meetings", "frequency": "Quarterly", "content": "EnPI performance, objectives status, audit results"},
                {"audience": "Contractors/Visitors", "method": "Site induction, contracts", "frequency": "As required", "content": "Relevant energy policy requirements"},
                {"audience": "External Stakeholders", "method": "Website, annual report", "frequency": "Annual", "content": "Energy policy (if externally communicated)"},
            ]
        lines = [
            "## 6. Communication Plan\n",
            "| Audience | Method | Frequency | Content |",
            "|----------|--------|-----------|---------|",
        ]
        for item in plan:
            lines.append(
                f"| {item.get('audience', '-')} "
                f"| {item.get('method', '-')} "
                f"| {item.get('frequency', '-')} "
                f"| {item.get('content', '-')} |"
            )
        return "\n".join(lines)

    def _md_review_schedule(self, data: Dict[str, Any]) -> str:
        """Render review schedule section."""
        review_freq = data.get("review_frequency", "Annual")
        next_review = data.get("next_review_date", "")
        triggers = data.get("review_triggers", [])
        if not triggers:
            triggers = [
                "Significant changes in energy sources or energy use",
                "Changes in legal or other requirements",
                "Organizational changes (acquisitions, expansions)",
                "Results of energy audits or internal audits",
                "Nonconformities or corrective actions requiring policy update",
                "Planned interval review (at least annually)",
            ]
        lines = [
            "## 7. Review Schedule\n",
            f"**Review Frequency:** {review_freq}  ",
            f"**Next Scheduled Review:** {next_review if next_review else 'To be determined'}  ",
            f"**Review Authority:** {data.get('review_authority', 'Top Management')}\n",
            "### Review Triggers\n",
            "The energy policy shall be reviewed when any of the following occur:\n",
        ]
        for i, trigger in enumerate(triggers, 1):
            lines.append(f"{i}. {trigger}")
        return "\n".join(lines)

    def _md_approval(self, data: Dict[str, Any]) -> str:
        """Render approval section."""
        approval = data.get("approval", {})
        return (
            "## Approval\n\n"
            f"**Approved By:** {approval.get('approved_by', '___________________')}  \n"
            f"**Title:** {approval.get('title', '___________________')}  \n"
            f"**Date:** {approval.get('date', '___________________')}  \n"
            f"**Signature:** {approval.get('signature', '___________________')}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>Energy Policy</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'ISO 50001:2018 Clause 5.2 | '
            f'Date: {data.get("policy_date", "-")}</p>'
        )

    def _html_policy_statement(self, data: Dict[str, Any]) -> str:
        """Render HTML policy statement."""
        org = data.get("organization_name", "Organization")
        statement = data.get("policy_statement", "")
        if not statement:
            statement = (
                f"{org} is committed to continual improvement of energy "
                f"performance through an effective Energy Management System "
                f"(EnMS) conforming to ISO 50001:2018."
            )
        return (
            '<h2>1. Policy Statement</h2>\n'
            f'<div class="policy-box"><p>{statement}</p></div>'
        )

    def _html_scope(self, data: Dict[str, Any]) -> str:
        """Render HTML scope section."""
        scope = data.get("scope", {})
        boundaries = scope.get("boundaries", [])
        items = "".join(f'<li>{b}</li>\n' for b in boundaries)
        return (
            '<h2>2. Scope & Boundaries</h2>\n'
            f'<p><strong>Description:</strong> {scope.get("description", "All facilities")}</p>\n'
            f'<p><strong>Organizational Boundary:</strong> {scope.get("organizational_boundary", "Operational control")}</p>\n'
            f'<ul>\n{items}</ul>'
        )

    def _html_commitments(self, data: Dict[str, Any]) -> str:
        """Render HTML commitments section."""
        commitments = data.get("commitments", [])
        rows = ""
        for c in commitments:
            rows += (
                f'<tr><td>{c.get("id", "-")}</td>'
                f'<td><strong>{c.get("title", "-")}</strong></td>'
                f'<td>{c.get("clause_ref", "-")}</td>'
                f'<td>{c.get("description", "-")}</td></tr>\n'
            )
        return (
            '<h2>3. Commitments</h2>\n'
            '<table>\n<tr><th>ID</th><th>Commitment</th>'
            f'<th>ISO Ref</th><th>Description</th></tr>\n{rows}</table>'
        )

    def _html_objectives_framework(self, data: Dict[str, Any]) -> str:
        """Render HTML objectives framework."""
        objectives = data.get("objectives", [])
        rows = ""
        for obj in objectives:
            rows += (
                f'<tr><td>{obj.get("objective", "-")}</td>'
                f'<td>{obj.get("target", "-")}</td>'
                f'<td>{obj.get("enpi", "-")}</td>'
                f'<td>{obj.get("timeline", "-")}</td>'
                f'<td>{obj.get("owner", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Objectives Framework</h2>\n'
            '<table>\n<tr><th>Objective</th><th>Target</th>'
            f'<th>EnPI</th><th>Timeline</th><th>Owner</th></tr>\n{rows}</table>'
        )

    def _html_roles_responsibilities(self, data: Dict[str, Any]) -> str:
        """Render HTML roles and responsibilities."""
        persons = data.get("responsible_persons", [])
        rows = ""
        for p in persons:
            rows += (
                f'<tr><td><strong>{p.get("role", "-")}</strong></td>'
                f'<td>{p.get("responsibility", "-")}</td>'
                f'<td>{p.get("clause_ref", "-")}</td></tr>\n'
            )
        return (
            '<h2>5. Roles & Responsibilities</h2>\n'
            '<table>\n<tr><th>Role</th><th>Responsibility</th>'
            f'<th>ISO Ref</th></tr>\n{rows}</table>'
        )

    def _html_communication_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML communication plan."""
        plan = data.get("communication_plan", [])
        rows = ""
        for item in plan:
            rows += (
                f'<tr><td>{item.get("audience", "-")}</td>'
                f'<td>{item.get("method", "-")}</td>'
                f'<td>{item.get("frequency", "-")}</td>'
                f'<td>{item.get("content", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Communication Plan</h2>\n'
            '<table>\n<tr><th>Audience</th><th>Method</th>'
            f'<th>Frequency</th><th>Content</th></tr>\n{rows}</table>'
        )

    def _html_review_schedule(self, data: Dict[str, Any]) -> str:
        """Render HTML review schedule."""
        review_freq = data.get("review_frequency", "Annual")
        triggers = data.get("review_triggers", [
            "Significant changes in energy sources or energy use",
            "Changes in legal or other requirements",
            "Organizational changes",
        ])
        items = "".join(f'<li>{t}</li>\n' for t in triggers)
        return (
            '<h2>7. Review Schedule</h2>\n'
            f'<p><strong>Review Frequency:</strong> {review_freq}</p>\n'
            f'<h3>Review Triggers</h3>\n<ol>\n{items}</ol>'
        )

    def _html_approval(self, data: Dict[str, Any]) -> str:
        """Render HTML approval block."""
        approval = data.get("approval", {})
        return (
            '<div class="approval-box">\n'
            '<h2>Approval</h2>\n'
            f'<p><strong>Approved By:</strong> {approval.get("approved_by", "___")}</p>\n'
            f'<p><strong>Title:</strong> {approval.get("title", "___")}</p>\n'
            f'<p><strong>Date:</strong> {approval.get("date", "___")}</p>\n'
            '</div>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_policy_statement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON policy statement section."""
        return {
            "statement": data.get("policy_statement", ""),
            "policy_date": data.get("policy_date", ""),
            "policy_version": data.get("policy_version", "1.0"),
            "iso_clause": "5.2",
        }

    def _json_scope(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON scope section."""
        scope = data.get("scope", {})
        return {
            "description": scope.get("description", ""),
            "organizational_boundary": scope.get("organizational_boundary", ""),
            "physical_boundary": scope.get("physical_boundary", ""),
            "boundaries": scope.get("boundaries", []),
            "exclusions": scope.get("exclusions", []),
        }

    def _json_review_schedule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON review schedule section."""
        return {
            "review_frequency": data.get("review_frequency", "Annual"),
            "next_review_date": data.get("next_review_date", ""),
            "review_authority": data.get("review_authority", "Top Management"),
            "review_triggers": data.get("review_triggers", []),
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
            ".policy-box{background:#e8f5e9;border-left:4px solid #198754;padding:15px 20px;margin:15px 0;border-radius:4px;}"
            ".approval-box{background:#f8f9fa;border:2px solid #dee2e6;padding:20px;margin:30px 0;border-radius:8px;}"
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
