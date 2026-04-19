# -*- coding: utf-8 -*-
"""
EnMSDocumentationTemplate - Comprehensive EnMS Documentation Package for PACK-034.

Generates a complete Energy Management System documentation package aligned
with all ISO 50001:2018 requirements. Covers document control, EnMS scope
and boundaries, energy policy, energy planning, support documentation,
operational controls, performance evaluation, improvement, and appendices
with cross-reference matrices.

Sections:
    1. Document Control (version, date, author, approval status)
    2. EnMS Scope & Boundaries (Clause 4.3)
    3. Energy Policy Statement (Clause 5.2)
    4. Energy Planning Summary (Clause 6)
    5. Support Documentation (Clause 7)
    6. Operational Controls (Clause 8)
    7. Performance Evaluation (Clause 9)
    8. Improvement (Clause 10)
    9. Appendices

Author: GreenLang Team
Version: 34.0.0
"""

__all__ = ["EnMSDocumentationTemplate"]

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnMSDocumentationTemplate:
    """
    Comprehensive EnMS documentation package template.

    Renders a full ISO 50001:2018 Energy Management System documentation
    package including document control, scope/boundaries, energy policy,
    planning, support, operational controls, performance evaluation,
    improvement, and appendices across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    TEMPLATE_ID = "enms_documentation"
    VERSION = "34.0.0"

    # ISO 50001:2018 clause-to-document cross-reference matrix
    CLAUSE_CROSS_REFERENCE: List[Dict[str, str]] = [
        {"clause": "4.1", "title": "Understanding the Organization and its Context", "document_type": "Record"},
        {"clause": "4.2", "title": "Needs and Expectations of Interested Parties", "document_type": "Record"},
        {"clause": "4.3", "title": "Scope of the EnMS", "document_type": "Documented Information"},
        {"clause": "4.4", "title": "Energy Management System", "document_type": "Documented Information"},
        {"clause": "5.1", "title": "Leadership and Commitment", "document_type": "Record"},
        {"clause": "5.2", "title": "Energy Policy", "document_type": "Policy"},
        {"clause": "5.3", "title": "Roles, Responsibilities, and Authorities", "document_type": "Documented Information"},
        {"clause": "6.1", "title": "Actions to Address Risks and Opportunities", "document_type": "Record"},
        {"clause": "6.2", "title": "Objectives, Energy Targets, and Action Plans", "document_type": "Documented Information"},
        {"clause": "6.3", "title": "Energy Review", "document_type": "Record"},
        {"clause": "6.4", "title": "Energy Performance Indicators", "document_type": "Documented Information"},
        {"clause": "6.5", "title": "Energy Baseline", "document_type": "Record"},
        {"clause": "6.6", "title": "Planning for Energy Data Collection", "document_type": "Documented Information"},
        {"clause": "7.1", "title": "Resources", "document_type": "Record"},
        {"clause": "7.2", "title": "Competence", "document_type": "Record"},
        {"clause": "7.3", "title": "Awareness", "document_type": "Record"},
        {"clause": "7.4", "title": "Communication", "document_type": "Documented Information"},
        {"clause": "7.5", "title": "Documented Information", "document_type": "Procedure"},
        {"clause": "8.1", "title": "Operational Planning and Control", "document_type": "Procedure"},
        {"clause": "8.2", "title": "Design", "document_type": "Record"},
        {"clause": "8.3", "title": "Procurement", "document_type": "Documented Information"},
        {"clause": "9.1", "title": "Monitoring, Measurement, Analysis, and Evaluation", "document_type": "Record"},
        {"clause": "9.2", "title": "Internal Audit", "document_type": "Record"},
        {"clause": "9.3", "title": "Management Review", "document_type": "Record"},
        {"clause": "10.1", "title": "Nonconformity and Corrective Action", "document_type": "Record"},
        {"clause": "10.2", "title": "Continual Improvement", "document_type": "Record"},
    ]

    # Default record retention schedule
    DEFAULT_RETENTION_SCHEDULE: List[Dict[str, str]] = [
        {"record_type": "Energy Policy", "retention": "Current + 1 previous version", "location": "EDMS", "disposal": "Secure deletion"},
        {"record_type": "EnMS Scope Document", "retention": "Current + all previous", "location": "EDMS", "disposal": "Archive"},
        {"record_type": "Energy Review Records", "retention": "5 years", "location": "EDMS", "disposal": "Archive then delete"},
        {"record_type": "EnPI/EnB Data", "retention": "Lifetime of baseline + 3 years", "location": "Database", "disposal": "Archive"},
        {"record_type": "Objectives & Action Plans", "retention": "5 years after completion", "location": "EDMS", "disposal": "Archive"},
        {"record_type": "Competence Records", "retention": "Employment + 3 years", "location": "HR System", "disposal": "Secure deletion"},
        {"record_type": "Internal Audit Reports", "retention": "5 years", "location": "EDMS", "disposal": "Archive then delete"},
        {"record_type": "Management Review Minutes", "retention": "5 years", "location": "EDMS", "disposal": "Archive then delete"},
        {"record_type": "NC/CA Records", "retention": "5 years after closure", "location": "EDMS", "disposal": "Archive then delete"},
        {"record_type": "Calibration Records", "retention": "3 years", "location": "CMMS", "disposal": "Delete"},
        {"record_type": "Operational Control Procedures", "retention": "Current + 2 previous", "location": "EDMS", "disposal": "Archive"},
        {"record_type": "Procurement Specifications", "retention": "Contract period + 3 years", "location": "EDMS", "disposal": "Archive"},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnMSDocumentationTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render comprehensive EnMS documentation package as Markdown.

        Args:
            data: EnMS documentation data including organization_name,
                  enms_scope, boundaries, energy_policy, legal_requirements,
                  seu_list, enpi_summary, enb_summary, objectives, targets,
                  action_plans, resources, competence_requirements,
                  operational_controls, monitoring_plan, audit_schedule,
                  management_reviews, nonconformities, improvements,
                  document_register, and metadata.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_document_control(data),
            self._md_enms_scope(data),
            self._md_energy_policy(data),
            self._md_energy_planning(data),
            self._md_support_documentation(data),
            self._md_operational_controls(data),
            self._md_performance_evaluation(data),
            self._md_improvement(data),
            self._md_appendices(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render comprehensive EnMS documentation package as self-contained HTML.

        Args:
            data: EnMS documentation data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_document_control(data),
            self._html_enms_scope(data),
            self._html_energy_policy(data),
            self._html_energy_planning(data),
            self._html_support_documentation(data),
            self._html_operational_controls(data),
            self._html_performance_evaluation(data),
            self._html_improvement(data),
            self._html_appendices(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>EnMS Documentation Package - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render comprehensive EnMS documentation package as structured JSON.

        Args:
            data: EnMS documentation data dict.

        Returns:
            Dict with all documentation sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": self.TEMPLATE_ID,
            "version": self.VERSION,
            "generated_at": self.generated_at.isoformat(),
            "organization_name": data.get("organization_name", ""),
            "document_control": self._json_document_control(data),
            "enms_scope": self._json_enms_scope(data),
            "energy_policy": self._json_energy_policy(data),
            "energy_planning": self._json_energy_planning(data),
            "support_documentation": self._json_support_documentation(data),
            "operational_controls": self._json_operational_controls(data),
            "performance_evaluation": self._json_performance_evaluation(data),
            "improvement": self._json_improvement(data),
            "appendices": self._json_appendices(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with documentation metadata."""
        org = data.get("organization_name", "Organization")
        meta = data.get("metadata", {})
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# EnMS Documentation Package\n\n"
            f"**Organization:** {org}  \n"
            f"**Document Date:** {meta.get('document_date', '')}  \n"
            f"**ISO 50001:2018 Reference:** Full Standard (Clauses 4-10)  \n"
            f"**Document Classification:** Controlled Document  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 EnMSDocumentationTemplate v{self.VERSION}\n\n---"
        )

    def _md_document_control(self, data: Dict[str, Any]) -> str:
        """Render Section 1: Document Control."""
        meta = data.get("metadata", {})
        lines = [
            "## 1. Document Control\n",
            "| Field | Value |",
            "|-------|-------|",
            f"| Document Title | EnMS Documentation Package |",
            f"| Document Number | {meta.get('document_number', 'ENMS-DOC-001')} |",
            f"| Version | {meta.get('version', '1.0')} |",
            f"| Date | {meta.get('document_date', '')} |",
            f"| Author | {meta.get('author', '')} |",
            f"| Reviewed By | {meta.get('reviewed_by', '')} |",
            f"| Approved By | {meta.get('approved_by', '')} |",
            f"| Approval Status | {meta.get('approval_status', 'Draft')} |",
            f"| Next Review Date | {meta.get('next_review_date', '')} |",
            f"| Classification | {meta.get('classification', 'Confidential')} |",
        ]
        # Revision history
        revisions = meta.get("revision_history", [])
        if revisions:
            lines.append("\n### Revision History\n")
            lines.append("| Version | Date | Author | Description |")
            lines.append("|---------|------|--------|-------------|")
            for rev in revisions:
                lines.append(
                    f"| {rev.get('version', '-')} "
                    f"| {rev.get('date', '-')} "
                    f"| {rev.get('author', '-')} "
                    f"| {rev.get('description', '-')} |"
                )
        return "\n".join(lines)

    def _md_enms_scope(self, data: Dict[str, Any]) -> str:
        """Render Section 2: EnMS Scope & Boundaries (Clause 4.3)."""
        scope = data.get("enms_scope", "")
        boundaries = data.get("boundaries", [])
        meta = data.get("metadata", {})
        lines = [
            "## 2. EnMS Scope & Boundaries (Clause 4.3)\n",
            "### 2.1 Scope Statement\n",
            scope if scope else "_Scope not yet defined._",
            "",
            "### 2.2 Organizational Boundaries\n",
        ]
        if boundaries:
            for b in boundaries:
                lines.append(f"- {b}")
        else:
            lines.append("_No boundaries specified._")
        # Exclusions
        exclusions = meta.get("exclusions", [])
        if exclusions:
            lines.append("\n### 2.3 Exclusions\n")
            lines.append("| Exclusion | Justification |")
            lines.append("|-----------|---------------|")
            for exc in exclusions:
                if isinstance(exc, dict):
                    lines.append(
                        f"| {exc.get('item', '-')} "
                        f"| {exc.get('justification', '-')} |"
                    )
                else:
                    lines.append(f"| {exc} | - |")
        # Context references
        lines.extend([
            "\n### 2.4 Context of the Organization\n",
            f"**Internal Issues:** {meta.get('internal_issues', 'See context analysis document')}  ",
            f"**External Issues:** {meta.get('external_issues', 'See context analysis document')}  ",
            f"**Interested Parties:** {meta.get('interested_parties', 'See interested parties register')}",
        ])
        return "\n".join(lines)

    def _md_energy_policy(self, data: Dict[str, Any]) -> str:
        """Render Section 3: Energy Policy Statement (Clause 5.2)."""
        policy = data.get("energy_policy", "")
        lines = [
            "## 3. Energy Policy Statement (Clause 5.2)\n",
        ]
        if policy:
            lines.append(f"> {policy}")
        else:
            lines.append("_Energy policy statement not provided. Refer to standalone energy policy document._")
        lines.extend([
            "",
            "### Policy Commitments\n",
            "The energy policy commits the organization to:",
            "- Continual improvement of energy performance (Clause 5.2 a)",
            "- Compliance with applicable legal and other requirements (Clause 5.2 b)",
            "- Availability of information and resources for energy objectives (Clause 5.2 c)",
            "- Support for procurement of energy-efficient products and services (Clause 5.2 d)",
        ])
        return "\n".join(lines)

    def _md_energy_planning(self, data: Dict[str, Any]) -> str:
        """Render Section 4: Energy Planning Summary (Clause 6)."""
        parts: List[str] = [
            "## 4. Energy Planning Summary (Clause 6)",
            self._md_legal_requirements(data),
            self._md_energy_review_summary(data),
            self._md_seu_list(data),
            self._md_enpi_enb_summary(data),
            self._md_objectives_targets(data),
        ]
        return "\n\n".join(parts)

    def _md_legal_requirements(self, data: Dict[str, Any]) -> str:
        """Render legal and regulatory requirements sub-section (Clause 6.1)."""
        reqs = data.get("legal_requirements", [])
        lines = [
            "### 4.1 Legal & Regulatory Requirements (Clause 6.1)\n",
        ]
        if not reqs:
            lines.append("_No legal requirements registered._")
            return "\n".join(lines)
        lines.extend([
            "| # | Requirement | Authority | Applicability | Compliance Status |",
            "|---|------------|-----------|---------------|-------------------|",
        ])
        for i, req in enumerate(reqs, 1):
            lines.append(
                f"| {i} | {req.get('requirement', '-')} "
                f"| {req.get('authority', '-')} "
                f"| {req.get('applicability', '-')} "
                f"| {req.get('compliance_status', '-')} |"
            )
        return "\n".join(lines)

    def _md_energy_review_summary(self, data: Dict[str, Any]) -> str:
        """Render energy review results summary sub-section (Clause 6.3)."""
        meta = data.get("metadata", {})
        review = meta.get("energy_review_summary", {})
        lines = [
            "### 4.2 Energy Review Results Summary (Clause 6.3)\n",
            f"**Total Energy Consumption:** {review.get('total_consumption', 'N/A')}  ",
            f"**Reporting Period:** {review.get('reporting_period', 'N/A')}  ",
            f"**Primary Energy Sources:** {review.get('primary_sources', 'N/A')}  ",
            f"**Major Energy Uses:** {review.get('major_uses', 'N/A')}  ",
            f"**Trend vs Previous Period:** {review.get('trend', 'N/A')}",
        ]
        return "\n".join(lines)

    def _md_seu_list(self, data: Dict[str, Any]) -> str:
        """Render significant energy uses (SEU) list sub-section (Clause 6.3)."""
        seus = data.get("seu_list", [])
        lines = ["### 4.3 Significant Energy Uses (SEUs)\n"]
        if not seus:
            lines.append("_No significant energy uses identified._")
            return "\n".join(lines)
        lines.extend([
            "| # | SEU | Energy Source | Annual Consumption | % of Total | Relevant Variables | Personnel |",
            "|---|-----|-------------|-------------------|-----------|-------------------|-----------|",
        ])
        for i, seu in enumerate(seus, 1):
            lines.append(
                f"| {i} | {seu.get('name', '-')} "
                f"| {seu.get('energy_source', '-')} "
                f"| {seu.get('annual_consumption', '-')} "
                f"| {seu.get('percentage_of_total', '-')} "
                f"| {seu.get('relevant_variables', '-')} "
                f"| {seu.get('personnel', '-')} |"
            )
        return "\n".join(lines)

    def _md_enpi_enb_summary(self, data: Dict[str, Any]) -> str:
        """Render EnPI and EnB summary sub-section (Clauses 6.4, 6.5)."""
        enpis = data.get("enpi_summary", [])
        enbs = data.get("enb_summary", [])
        lines = ["### 4.4 EnPI & EnB Summary (Clauses 6.4, 6.5)\n"]
        # EnPIs
        lines.append("#### Energy Performance Indicators\n")
        if enpis:
            lines.extend([
                "| # | EnPI | Unit | Current Value | Target Value | Methodology |",
                "|---|------|------|--------------|-------------|-------------|",
            ])
            for i, enpi in enumerate(enpis, 1):
                lines.append(
                    f"| {i} | {enpi.get('name', '-')} "
                    f"| {enpi.get('unit', '-')} "
                    f"| {enpi.get('current_value', '-')} "
                    f"| {enpi.get('target_value', '-')} "
                    f"| {enpi.get('methodology', '-')} |"
                )
        else:
            lines.append("_No EnPIs defined._")
        # EnBs
        lines.append("\n#### Energy Baselines\n")
        if enbs:
            lines.extend([
                "| # | Baseline | Period | Value | Unit | Adjustment Criteria |",
                "|---|----------|--------|-------|------|-------------------|",
            ])
            for i, enb in enumerate(enbs, 1):
                lines.append(
                    f"| {i} | {enb.get('name', '-')} "
                    f"| {enb.get('period', '-')} "
                    f"| {enb.get('value', '-')} "
                    f"| {enb.get('unit', '-')} "
                    f"| {enb.get('adjustment_criteria', '-')} |"
                )
        else:
            lines.append("_No energy baselines defined._")
        return "\n".join(lines)

    def _md_objectives_targets(self, data: Dict[str, Any]) -> str:
        """Render objectives, targets, and action plans sub-section (Clause 6.2)."""
        objectives = data.get("objectives", [])
        targets = data.get("targets", [])
        action_plans = data.get("action_plans", [])
        lines = ["### 4.5 Objectives, Targets, and Action Plans (Clause 6.2)\n"]
        # Objectives
        lines.append("#### Energy Objectives\n")
        if objectives:
            lines.extend([
                "| # | Objective | Relevant SEU | Owner | Timeline | Status |",
                "|---|-----------|-------------|-------|----------|--------|",
            ])
            for i, obj in enumerate(objectives, 1):
                lines.append(
                    f"| {i} | {obj.get('description', '-')} "
                    f"| {obj.get('relevant_seu', '-')} "
                    f"| {obj.get('owner', '-')} "
                    f"| {obj.get('timeline', '-')} "
                    f"| {obj.get('status', '-')} |"
                )
        else:
            lines.append("_No objectives defined._")
        # Targets
        lines.append("\n#### Energy Targets\n")
        if targets:
            lines.extend([
                "| # | Target | Metric | Baseline Value | Target Value | Deadline | Status |",
                "|---|--------|--------|---------------|-------------|----------|--------|",
            ])
            for i, tgt in enumerate(targets, 1):
                lines.append(
                    f"| {i} | {tgt.get('description', '-')} "
                    f"| {tgt.get('metric', '-')} "
                    f"| {tgt.get('baseline_value', '-')} "
                    f"| {tgt.get('target_value', '-')} "
                    f"| {tgt.get('deadline', '-')} "
                    f"| {tgt.get('status', '-')} |"
                )
        else:
            lines.append("_No targets defined._")
        # Action plans
        lines.append("\n#### Action Plans\n")
        if action_plans:
            lines.extend([
                "| # | Action | Objective Ref | Resources | Responsible | Start | End | Status |",
                "|---|--------|-------------|-----------|------------|-------|-----|--------|",
            ])
            for i, ap in enumerate(action_plans, 1):
                lines.append(
                    f"| {i} | {ap.get('action', '-')} "
                    f"| {ap.get('objective_ref', '-')} "
                    f"| {ap.get('resources', '-')} "
                    f"| {ap.get('responsible', '-')} "
                    f"| {ap.get('start_date', '-')} "
                    f"| {ap.get('end_date', '-')} "
                    f"| {ap.get('status', '-')} |"
                )
        else:
            lines.append("_No action plans defined._")
        return "\n".join(lines)

    def _md_support_documentation(self, data: Dict[str, Any]) -> str:
        """Render Section 5: Support Documentation (Clause 7)."""
        parts: List[str] = [
            "## 5. Support Documentation (Clause 7)",
            self._md_resources(data),
            self._md_competence(data),
            self._md_awareness(data),
            self._md_communication(data),
            self._md_documented_info_register(data),
        ]
        return "\n\n".join(parts)

    def _md_resources(self, data: Dict[str, Any]) -> str:
        """Render resources allocated sub-section (Clause 7.1)."""
        resources = data.get("resources", [])
        lines = ["### 5.1 Resources Allocated (Clause 7.1)\n"]
        if not resources:
            lines.append("_No resources documented._")
            return "\n".join(lines)
        lines.extend([
            "| # | Resource | Type | Budget/Allocation | Department | Status |",
            "|---|----------|------|------------------|-----------|--------|",
        ])
        for i, res in enumerate(resources, 1):
            budget = res.get("budget", res.get("allocation", "-"))
            if isinstance(budget, (int, float)):
                budget = self._format_currency(budget)
            lines.append(
                f"| {i} | {res.get('name', '-')} "
                f"| {res.get('type', '-')} "
                f"| {budget} "
                f"| {res.get('department', '-')} "
                f"| {res.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_competence(self, data: Dict[str, Any]) -> str:
        """Render competence requirements sub-section (Clause 7.2)."""
        reqs = data.get("competence_requirements", [])
        lines = ["### 5.2 Competence Requirements (Clause 7.2)\n"]
        if not reqs:
            lines.append("_No competence requirements documented._")
            return "\n".join(lines)
        lines.extend([
            "| # | Role | Competence Required | Training Needed | Evidence | Status |",
            "|---|------|-------------------|----------------|----------|--------|",
        ])
        for i, comp in enumerate(reqs, 1):
            lines.append(
                f"| {i} | {comp.get('role', '-')} "
                f"| {comp.get('competence', '-')} "
                f"| {comp.get('training', '-')} "
                f"| {comp.get('evidence', '-')} "
                f"| {comp.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_awareness(self, data: Dict[str, Any]) -> str:
        """Render awareness program sub-section (Clause 7.3)."""
        meta = data.get("metadata", {})
        awareness = meta.get("awareness_program", {})
        lines = [
            "### 5.3 Awareness Program (Clause 7.3)\n",
            "All persons working under the organization's control shall be aware of:\n",
            "- The energy policy",
            "- Their contribution to the effectiveness of the EnMS",
            "- Their contribution to energy performance improvement",
            "- The implications of not conforming to EnMS requirements\n",
            f"**Training Method:** {awareness.get('method', 'Induction + annual refresher')}  ",
            f"**Frequency:** {awareness.get('frequency', 'Annual')}  ",
            f"**Records Location:** {awareness.get('records_location', 'HR training database')}  ",
            f"**Last Campaign Date:** {awareness.get('last_campaign', 'N/A')}",
        ]
        return "\n".join(lines)

    def _md_communication(self, data: Dict[str, Any]) -> str:
        """Render communication plan sub-section (Clause 7.4)."""
        meta = data.get("metadata", {})
        plan = meta.get("communication_plan", [])
        if not plan:
            plan = [
                {"what": "Energy policy & objectives", "when": "On change / annually", "who": "Energy Manager", "audience": "All employees", "method": "Intranet, notice boards"},
                {"what": "EnPI performance", "when": "Monthly", "who": "Energy Team", "audience": "Management", "method": "Dashboard, reports"},
                {"what": "Energy awareness", "when": "Ongoing", "who": "Energy Team", "audience": "All personnel", "method": "Training, events"},
                {"what": "Audit findings", "when": "Post-audit", "who": "Audit Lead", "audience": "Relevant managers", "method": "Audit reports"},
            ]
        lines = [
            "### 5.4 Communication Plan (Clause 7.4)\n",
            "| What | When | Who | Audience | Method |",
            "|------|------|-----|----------|--------|",
        ]
        for item in plan:
            lines.append(
                f"| {item.get('what', '-')} "
                f"| {item.get('when', '-')} "
                f"| {item.get('who', '-')} "
                f"| {item.get('audience', '-')} "
                f"| {item.get('method', '-')} |"
            )
        return "\n".join(lines)

    def _md_documented_info_register(self, data: Dict[str, Any]) -> str:
        """Render documented information register sub-section (Clause 7.5)."""
        register = data.get("document_register", [])
        lines = ["### 5.5 Documented Information Register (Clause 7.5)\n"]
        if not register:
            lines.append("_No documented information registered. See Appendix A._")
            return "\n".join(lines)
        lines.extend([
            "| # | Doc ID | Title | Type | Clause | Owner | Status |",
            "|---|--------|-------|------|--------|-------|--------|",
        ])
        for i, doc in enumerate(register, 1):
            lines.append(
                f"| {i} | {doc.get('doc_id', '-')} "
                f"| {doc.get('title', '-')} "
                f"| {doc.get('type', '-')} "
                f"| {doc.get('clause', '-')} "
                f"| {doc.get('owner', '-')} "
                f"| {doc.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_operational_controls(self, data: Dict[str, Any]) -> str:
        """Render Section 6: Operational Controls (Clause 8)."""
        controls = data.get("operational_controls", [])
        meta = data.get("metadata", {})
        lines = [
            "## 6. Operational Controls (Clause 8)\n",
            "### 6.1 Operational Criteria (Clause 8.1)\n",
        ]
        if controls:
            lines.extend([
                "| # | Control | Applicable SEU | Criteria | Procedure Ref | Owner |",
                "|---|---------|---------------|----------|--------------|-------|",
            ])
            for i, ctrl in enumerate(controls, 1):
                lines.append(
                    f"| {i} | {ctrl.get('name', '-')} "
                    f"| {ctrl.get('applicable_seu', '-')} "
                    f"| {ctrl.get('criteria', '-')} "
                    f"| {ctrl.get('procedure_ref', '-')} "
                    f"| {ctrl.get('owner', '-')} |"
                )
        else:
            lines.append("_No operational controls documented._")
        # Design considerations
        design = meta.get("design_considerations", [])
        lines.append("\n### 6.2 Design Considerations (Clause 8.2)\n")
        if design:
            for item in design:
                if isinstance(item, dict):
                    lines.append(
                        f"- **{item.get('area', 'N/A')}:** {item.get('consideration', '-')}"
                    )
                else:
                    lines.append(f"- {item}")
        else:
            lines.extend([
                "Energy performance improvement opportunities shall be considered in the design of:",
                "- New, modified, and renovated facilities",
                "- New, modified, and renovated systems",
                "- Equipment and processes affecting energy performance",
                "- Design reviews shall include energy performance evaluation results",
            ])
        # Procurement requirements
        procurement = meta.get("procurement_requirements", [])
        lines.append("\n### 6.3 Procurement Requirements (Clause 8.3)\n")
        if procurement:
            lines.extend([
                "| # | Category | Requirement | Specification |",
                "|---|----------|------------|---------------|",
            ])
            for i, proc in enumerate(procurement, 1):
                lines.append(
                    f"| {i} | {proc.get('category', '-')} "
                    f"| {proc.get('requirement', '-')} "
                    f"| {proc.get('specification', '-')} |"
                )
        else:
            lines.extend([
                "Procurement shall consider energy performance over the planned or expected operational lifetime when:",
                "- Purchasing energy-using products, equipment, and services",
                "- Procuring energy supply services",
                "- Evaluating energy performance criteria in purchasing specifications",
            ])
        return "\n".join(lines)

    def _md_performance_evaluation(self, data: Dict[str, Any]) -> str:
        """Render Section 7: Performance Evaluation (Clause 9)."""
        parts: List[str] = [
            "## 7. Performance Evaluation (Clause 9)",
            self._md_monitoring_plan(data),
            self._md_compliance_evaluation(data),
            self._md_audit_schedule(data),
            self._md_management_review(data),
        ]
        return "\n\n".join(parts)

    def _md_monitoring_plan(self, data: Dict[str, Any]) -> str:
        """Render monitoring and measurement plan sub-section (Clause 9.1)."""
        plan = data.get("monitoring_plan", [])
        lines = ["### 7.1 Monitoring & Measurement Plan (Clause 9.1)\n"]
        if not plan:
            lines.append("_No monitoring plan documented._")
            return "\n".join(lines)
        lines.extend([
            "| # | Parameter | Method | Frequency | Responsibility | Equipment | Accuracy |",
            "|---|-----------|--------|-----------|---------------|-----------|----------|",
        ])
        for i, item in enumerate(plan, 1):
            lines.append(
                f"| {i} | {item.get('parameter', '-')} "
                f"| {item.get('method', '-')} "
                f"| {item.get('frequency', '-')} "
                f"| {item.get('responsibility', '-')} "
                f"| {item.get('equipment', '-')} "
                f"| {item.get('accuracy', '-')} |"
            )
        return "\n".join(lines)

    def _md_compliance_evaluation(self, data: Dict[str, Any]) -> str:
        """Render evaluation of compliance sub-section (Clause 9.1)."""
        meta = data.get("metadata", {})
        compliance = meta.get("compliance_evaluation", {})
        lines = [
            "### 7.2 Evaluation of Compliance (Clause 9.1)\n",
            f"**Evaluation Frequency:** {compliance.get('frequency', 'Annual')}  ",
            f"**Last Evaluation Date:** {compliance.get('last_evaluation', 'N/A')}  ",
            f"**Evaluator:** {compliance.get('evaluator', 'N/A')}  ",
            f"**Overall Compliance Status:** {compliance.get('status', 'N/A')}  ",
            f"**Nonconformities Identified:** {compliance.get('nonconformities_count', '0')}",
        ]
        return "\n".join(lines)

    def _md_audit_schedule(self, data: Dict[str, Any]) -> str:
        """Render internal audit schedule sub-section (Clause 9.2)."""
        schedule = data.get("audit_schedule", [])
        lines = ["### 7.3 Internal Audit Schedule (Clause 9.2)\n"]
        if not schedule:
            lines.append("_No audit schedule documented._")
            return "\n".join(lines)
        lines.extend([
            "| # | Audit | Scope | Clause(s) | Auditor | Planned Date | Status |",
            "|---|-------|-------|-----------|---------|-------------|--------|",
        ])
        for i, audit in enumerate(schedule, 1):
            lines.append(
                f"| {i} | {audit.get('name', '-')} "
                f"| {audit.get('scope', '-')} "
                f"| {audit.get('clauses', '-')} "
                f"| {audit.get('auditor', '-')} "
                f"| {audit.get('planned_date', '-')} "
                f"| {audit.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_management_review(self, data: Dict[str, Any]) -> str:
        """Render management review summary sub-section (Clause 9.3)."""
        reviews = data.get("management_reviews", [])
        lines = ["### 7.4 Management Review Summary (Clause 9.3)\n"]
        if not reviews:
            lines.append("_No management reviews recorded._")
            return "\n".join(lines)
        lines.extend([
            "| # | Review Date | Chair | Key Decisions | Actions | Next Review |",
            "|---|------------|-------|--------------|---------|------------|",
        ])
        for i, rev in enumerate(reviews, 1):
            lines.append(
                f"| {i} | {rev.get('date', '-')} "
                f"| {rev.get('chair', '-')} "
                f"| {rev.get('key_decisions', '-')} "
                f"| {rev.get('actions_count', '-')} "
                f"| {rev.get('next_review', '-')} |"
            )
        return "\n".join(lines)

    def _md_improvement(self, data: Dict[str, Any]) -> str:
        """Render Section 8: Improvement (Clause 10)."""
        parts: List[str] = [
            "## 8. Improvement (Clause 10)",
            self._md_nonconformities(data),
            self._md_continual_improvement(data),
        ]
        return "\n\n".join(parts)

    def _md_nonconformities(self, data: Dict[str, Any]) -> str:
        """Render nonconformity and corrective action register (Clause 10.1)."""
        ncs = data.get("nonconformities", [])
        lines = ["### 8.1 Nonconformity & Corrective Action Register (Clause 10.1)\n"]
        if not ncs:
            lines.append("_No nonconformities recorded._")
            return "\n".join(lines)
        lines.extend([
            "| # | NC ID | Description | Root Cause | Corrective Action | Owner | Due Date | Status |",
            "|---|-------|-----------|-----------|------------------|-------|---------|--------|",
        ])
        for i, nc in enumerate(ncs, 1):
            lines.append(
                f"| {i} | {nc.get('nc_id', '-')} "
                f"| {nc.get('description', '-')} "
                f"| {nc.get('root_cause', '-')} "
                f"| {nc.get('corrective_action', '-')} "
                f"| {nc.get('owner', '-')} "
                f"| {nc.get('due_date', '-')} "
                f"| {nc.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_continual_improvement(self, data: Dict[str, Any]) -> str:
        """Render continual improvement evidence sub-section (Clause 10.2)."""
        improvements = data.get("improvements", [])
        lines = ["### 8.2 Continual Improvement Evidence (Clause 10.2)\n"]
        if not improvements:
            lines.append("_No improvement records documented._")
            return "\n".join(lines)
        lines.extend([
            "| # | Improvement | Category | EnPI Impact | Period | Evidence |",
            "|---|-----------|----------|-----------|--------|---------|",
        ])
        for i, imp in enumerate(improvements, 1):
            lines.append(
                f"| {i} | {imp.get('description', '-')} "
                f"| {imp.get('category', '-')} "
                f"| {imp.get('enpi_impact', '-')} "
                f"| {imp.get('period', '-')} "
                f"| {imp.get('evidence', '-')} |"
            )
        return "\n".join(lines)

    def _md_appendices(self, data: Dict[str, Any]) -> str:
        """Render Section 9: Appendices."""
        parts: List[str] = [
            "## 9. Appendices",
            self._md_appendix_document_register(data),
            self._md_appendix_retention_schedule(data),
            self._md_appendix_cross_reference(data),
        ]
        return "\n\n".join(parts)

    def _md_appendix_document_register(self, data: Dict[str, Any]) -> str:
        """Render Appendix A: Document Register."""
        register = data.get("document_register", [])
        lines = ["### Appendix A: Document Register\n"]
        if not register:
            lines.append("_Document register not populated._")
            return "\n".join(lines)
        lines.extend([
            "| # | Doc ID | Title | Type | Version | Owner | Last Updated | Status |",
            "|---|--------|-------|------|---------|-------|-------------|--------|",
        ])
        for i, doc in enumerate(register, 1):
            lines.append(
                f"| {i} | {doc.get('doc_id', '-')} "
                f"| {doc.get('title', '-')} "
                f"| {doc.get('type', '-')} "
                f"| {doc.get('version', '-')} "
                f"| {doc.get('owner', '-')} "
                f"| {doc.get('last_updated', '-')} "
                f"| {doc.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_appendix_retention_schedule(self, data: Dict[str, Any]) -> str:
        """Render Appendix B: Record Retention Schedule."""
        meta = data.get("metadata", {})
        retention = meta.get("retention_schedule", self.DEFAULT_RETENTION_SCHEDULE)
        lines = [
            "### Appendix B: Record Retention Schedule\n",
            "| Record Type | Retention Period | Storage Location | Disposal Method |",
            "|------------|----------------|-----------------|----------------|",
        ]
        for rec in retention:
            lines.append(
                f"| {rec.get('record_type', '-')} "
                f"| {rec.get('retention', '-')} "
                f"| {rec.get('location', '-')} "
                f"| {rec.get('disposal', '-')} |"
            )
        return "\n".join(lines)

    def _md_appendix_cross_reference(self, data: Dict[str, Any]) -> str:
        """Render Appendix C: Cross-Reference Matrix (clause -> document)."""
        register = data.get("document_register", [])
        doc_by_clause: Dict[str, List[str]] = {}
        for doc in register:
            clause = doc.get("clause", "")
            if clause:
                doc_by_clause.setdefault(clause, []).append(doc.get("doc_id", "-"))
        lines = [
            "### Appendix C: Cross-Reference Matrix (Clause to Document)\n",
            "| ISO 50001 Clause | Clause Title | Document Type | Document Ref |",
            "|-----------------|-------------|---------------|-------------|",
        ]
        for ref in self.CLAUSE_CROSS_REFERENCE:
            clause = ref["clause"]
            doc_refs = ", ".join(doc_by_clause.get(clause, ["Not assigned"]))
            lines.append(
                f"| {clause} "
                f"| {ref['title']} "
                f"| {ref['document_type']} "
                f"| {doc_refs} |"
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
        org = data.get("organization_name", "Organization")
        meta = data.get("metadata", {})
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>EnMS Documentation Package</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'ISO 50001:2018 Clauses 4-10 | '
            f'Date: {meta.get("document_date", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_document_control(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 1: Document Control."""
        meta = data.get("metadata", {})
        rows = ""
        fields = [
            ("Document Number", meta.get("document_number", "ENMS-DOC-001")),
            ("Version", meta.get("version", "1.0")),
            ("Date", meta.get("document_date", "")),
            ("Author", meta.get("author", "")),
            ("Approved By", meta.get("approved_by", "")),
            ("Approval Status", meta.get("approval_status", "Draft")),
            ("Next Review Date", meta.get("next_review_date", "")),
        ]
        for label, value in fields:
            rows += f'<tr><td><strong>{label}</strong></td><td>{value}</td></tr>\n'
        return (
            '<h2>1. Document Control</h2>\n'
            f'<table>\n{rows}</table>'
        )

    def _html_enms_scope(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 2: EnMS Scope & Boundaries."""
        scope = data.get("enms_scope", "")
        boundaries = data.get("boundaries", [])
        items = "".join(f'<li>{b}</li>\n' for b in boundaries)
        return (
            '<h2>2. EnMS Scope & Boundaries (Clause 4.3)</h2>\n'
            f'<div class="policy-box"><p>{scope if scope else "Scope not yet defined."}</p></div>\n'
            f'<h3>Organizational Boundaries</h3>\n<ul>\n{items}</ul>'
        )

    def _html_energy_policy(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 3: Energy Policy Statement."""
        policy = data.get("energy_policy", "")
        return (
            '<h2>3. Energy Policy Statement (Clause 5.2)</h2>\n'
            f'<div class="policy-box"><p>{policy if policy else "Energy policy statement not provided."}</p></div>'
        )

    def _html_energy_planning(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 4: Energy Planning Summary."""
        parts: List[str] = ['<h2>4. Energy Planning Summary (Clause 6)</h2>']
        # Legal requirements
        reqs = data.get("legal_requirements", [])
        rows = ""
        for req in reqs:
            rows += (
                f'<tr><td>{req.get("requirement", "-")}</td>'
                f'<td>{req.get("authority", "-")}</td>'
                f'<td>{req.get("compliance_status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>4.1 Legal & Regulatory Requirements</h3>\n'
            '<table>\n<tr><th>Requirement</th><th>Authority</th>'
            f'<th>Compliance</th></tr>\n{rows}</table>'
        )
        # SEU list
        seus = data.get("seu_list", [])
        rows = ""
        for seu in seus:
            rows += (
                f'<tr><td>{seu.get("name", "-")}</td>'
                f'<td>{seu.get("energy_source", "-")}</td>'
                f'<td>{seu.get("annual_consumption", "-")}</td>'
                f'<td>{seu.get("percentage_of_total", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>4.3 Significant Energy Uses</h3>\n'
            '<table>\n<tr><th>SEU</th><th>Source</th>'
            f'<th>Annual Consumption</th><th>% Total</th></tr>\n{rows}</table>'
        )
        # EnPI summary
        enpis = data.get("enpi_summary", [])
        rows = ""
        for enpi in enpis:
            rows += (
                f'<tr><td>{enpi.get("name", "-")}</td>'
                f'<td>{enpi.get("unit", "-")}</td>'
                f'<td>{enpi.get("current_value", "-")}</td>'
                f'<td>{enpi.get("target_value", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>4.4 EnPI Summary</h3>\n'
            '<table>\n<tr><th>EnPI</th><th>Unit</th>'
            f'<th>Current</th><th>Target</th></tr>\n{rows}</table>'
        )
        # Objectives
        objectives = data.get("objectives", [])
        rows = ""
        for obj in objectives:
            rows += (
                f'<tr><td>{obj.get("description", "-")}</td>'
                f'<td>{obj.get("owner", "-")}</td>'
                f'<td>{obj.get("timeline", "-")}</td>'
                f'<td>{obj.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>4.5 Objectives & Targets</h3>\n'
            '<table>\n<tr><th>Objective</th><th>Owner</th>'
            f'<th>Timeline</th><th>Status</th></tr>\n{rows}</table>'
        )
        return "\n".join(parts)

    def _html_support_documentation(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 5: Support Documentation."""
        parts: List[str] = ['<h2>5. Support Documentation (Clause 7)</h2>']
        # Resources
        resources = data.get("resources", [])
        rows = ""
        for res in resources:
            budget = res.get("budget", res.get("allocation", "-"))
            if isinstance(budget, (int, float)):
                budget = self._format_currency(budget)
            rows += (
                f'<tr><td>{res.get("name", "-")}</td>'
                f'<td>{res.get("type", "-")}</td>'
                f'<td>{budget}</td>'
                f'<td>{res.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>5.1 Resources Allocated</h3>\n'
            '<table>\n<tr><th>Resource</th><th>Type</th>'
            f'<th>Budget</th><th>Status</th></tr>\n{rows}</table>'
        )
        # Competence
        comp_reqs = data.get("competence_requirements", [])
        rows = ""
        for comp in comp_reqs:
            rows += (
                f'<tr><td>{comp.get("role", "-")}</td>'
                f'<td>{comp.get("competence", "-")}</td>'
                f'<td>{comp.get("training", "-")}</td>'
                f'<td>{comp.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>5.2 Competence Requirements</h3>\n'
            '<table>\n<tr><th>Role</th><th>Competence</th>'
            f'<th>Training</th><th>Status</th></tr>\n{rows}</table>'
        )
        # Document register
        register = data.get("document_register", [])
        rows = ""
        for doc in register:
            rows += (
                f'<tr><td>{doc.get("doc_id", "-")}</td>'
                f'<td>{doc.get("title", "-")}</td>'
                f'<td>{doc.get("clause", "-")}</td>'
                f'<td>{doc.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>5.5 Documented Information Register</h3>\n'
            '<table>\n<tr><th>Doc ID</th><th>Title</th>'
            f'<th>Clause</th><th>Status</th></tr>\n{rows}</table>'
        )
        return "\n".join(parts)

    def _html_operational_controls(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 6: Operational Controls."""
        controls = data.get("operational_controls", [])
        rows = ""
        for ctrl in controls:
            rows += (
                f'<tr><td>{ctrl.get("name", "-")}</td>'
                f'<td>{ctrl.get("applicable_seu", "-")}</td>'
                f'<td>{ctrl.get("criteria", "-")}</td>'
                f'<td>{ctrl.get("owner", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Operational Controls (Clause 8)</h2>\n'
            '<table>\n<tr><th>Control</th><th>Applicable SEU</th>'
            f'<th>Criteria</th><th>Owner</th></tr>\n{rows}</table>'
        )

    def _html_performance_evaluation(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 7: Performance Evaluation."""
        parts: List[str] = ['<h2>7. Performance Evaluation (Clause 9)</h2>']
        # Monitoring plan
        plan = data.get("monitoring_plan", [])
        rows = ""
        for item in plan:
            rows += (
                f'<tr><td>{item.get("parameter", "-")}</td>'
                f'<td>{item.get("method", "-")}</td>'
                f'<td>{item.get("frequency", "-")}</td>'
                f'<td>{item.get("responsibility", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>7.1 Monitoring & Measurement Plan</h3>\n'
            '<table>\n<tr><th>Parameter</th><th>Method</th>'
            f'<th>Frequency</th><th>Responsibility</th></tr>\n{rows}</table>'
        )
        # Audit schedule
        schedule = data.get("audit_schedule", [])
        rows = ""
        for audit in schedule:
            rows += (
                f'<tr><td>{audit.get("name", "-")}</td>'
                f'<td>{audit.get("clauses", "-")}</td>'
                f'<td>{audit.get("planned_date", "-")}</td>'
                f'<td>{audit.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>7.3 Internal Audit Schedule</h3>\n'
            '<table>\n<tr><th>Audit</th><th>Clauses</th>'
            f'<th>Planned Date</th><th>Status</th></tr>\n{rows}</table>'
        )
        # Management reviews
        reviews = data.get("management_reviews", [])
        rows = ""
        for rev in reviews:
            rows += (
                f'<tr><td>{rev.get("date", "-")}</td>'
                f'<td>{rev.get("chair", "-")}</td>'
                f'<td>{rev.get("key_decisions", "-")}</td>'
                f'<td>{rev.get("next_review", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>7.4 Management Review Summary</h3>\n'
            '<table>\n<tr><th>Date</th><th>Chair</th>'
            f'<th>Key Decisions</th><th>Next Review</th></tr>\n{rows}</table>'
        )
        return "\n".join(parts)

    def _html_improvement(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 8: Improvement."""
        parts: List[str] = ['<h2>8. Improvement (Clause 10)</h2>']
        # Nonconformities
        ncs = data.get("nonconformities", [])
        rows = ""
        for nc in ncs:
            status = nc.get("status", "-").lower()
            cls = "check-pass" if status in ("closed", "resolved") else "check-fail"
            rows += (
                f'<tr><td>{nc.get("nc_id", "-")}</td>'
                f'<td>{nc.get("description", "-")}</td>'
                f'<td>{nc.get("corrective_action", "-")}</td>'
                f'<td class="{cls}">{nc.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>8.1 Nonconformity & Corrective Action Register</h3>\n'
            '<table>\n<tr><th>NC ID</th><th>Description</th>'
            f'<th>Corrective Action</th><th>Status</th></tr>\n{rows}</table>'
        )
        # Continual improvement
        improvements = data.get("improvements", [])
        rows = ""
        for imp in improvements:
            rows += (
                f'<tr><td>{imp.get("description", "-")}</td>'
                f'<td>{imp.get("category", "-")}</td>'
                f'<td>{imp.get("enpi_impact", "-")}</td>'
                f'<td>{imp.get("evidence", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>8.2 Continual Improvement Evidence</h3>\n'
            '<table>\n<tr><th>Improvement</th><th>Category</th>'
            f'<th>EnPI Impact</th><th>Evidence</th></tr>\n{rows}</table>'
        )
        return "\n".join(parts)

    def _html_appendices(self, data: Dict[str, Any]) -> str:
        """Render HTML Section 9: Appendices."""
        parts: List[str] = ['<h2>9. Appendices</h2>']
        # Appendix A: Document register (full)
        register = data.get("document_register", [])
        rows = ""
        for doc in register:
            rows += (
                f'<tr><td>{doc.get("doc_id", "-")}</td>'
                f'<td>{doc.get("title", "-")}</td>'
                f'<td>{doc.get("type", "-")}</td>'
                f'<td>{doc.get("version", "-")}</td>'
                f'<td>{doc.get("owner", "-")}</td>'
                f'<td>{doc.get("status", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>Appendix A: Document Register</h3>\n'
            '<table>\n<tr><th>Doc ID</th><th>Title</th><th>Type</th>'
            f'<th>Version</th><th>Owner</th><th>Status</th></tr>\n{rows}</table>'
        )
        # Appendix B: Retention schedule
        meta = data.get("metadata", {})
        retention = meta.get("retention_schedule", self.DEFAULT_RETENTION_SCHEDULE)
        rows = ""
        for rec in retention:
            rows += (
                f'<tr><td>{rec.get("record_type", "-")}</td>'
                f'<td>{rec.get("retention", "-")}</td>'
                f'<td>{rec.get("location", "-")}</td>'
                f'<td>{rec.get("disposal", "-")}</td></tr>\n'
            )
        parts.append(
            '<h3>Appendix B: Record Retention Schedule</h3>\n'
            '<table>\n<tr><th>Record Type</th><th>Retention</th>'
            f'<th>Location</th><th>Disposal</th></tr>\n{rows}</table>'
        )
        # Appendix C: Cross-reference matrix
        doc_by_clause: Dict[str, List[str]] = {}
        for doc in register:
            clause = doc.get("clause", "")
            if clause:
                doc_by_clause.setdefault(clause, []).append(doc.get("doc_id", "-"))
        rows = ""
        for ref in self.CLAUSE_CROSS_REFERENCE:
            clause = ref["clause"]
            doc_refs = ", ".join(doc_by_clause.get(clause, ["Not assigned"]))
            rows += (
                f'<tr><td>{clause}</td>'
                f'<td>{ref["title"]}</td>'
                f'<td>{ref["document_type"]}</td>'
                f'<td>{doc_refs}</td></tr>\n'
            )
        parts.append(
            '<h3>Appendix C: Cross-Reference Matrix</h3>\n'
            '<table>\n<tr><th>Clause</th><th>Title</th>'
            f'<th>Doc Type</th><th>Document Ref</th></tr>\n{rows}</table>'
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_document_control(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON document control section."""
        meta = data.get("metadata", {})
        return {
            "document_number": meta.get("document_number", "ENMS-DOC-001"),
            "version": meta.get("version", "1.0"),
            "document_date": meta.get("document_date", ""),
            "author": meta.get("author", ""),
            "reviewed_by": meta.get("reviewed_by", ""),
            "approved_by": meta.get("approved_by", ""),
            "approval_status": meta.get("approval_status", "Draft"),
            "next_review_date": meta.get("next_review_date", ""),
            "classification": meta.get("classification", "Confidential"),
            "revision_history": meta.get("revision_history", []),
        }

    def _json_enms_scope(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON EnMS scope section."""
        meta = data.get("metadata", {})
        return {
            "scope_statement": data.get("enms_scope", ""),
            "boundaries": data.get("boundaries", []),
            "exclusions": meta.get("exclusions", []),
            "internal_issues": meta.get("internal_issues", ""),
            "external_issues": meta.get("external_issues", ""),
            "interested_parties": meta.get("interested_parties", ""),
            "iso_clause": "4.3",
        }

    def _json_energy_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON energy policy section."""
        return {
            "policy_statement": data.get("energy_policy", ""),
            "iso_clause": "5.2",
            "commitments": [
                "Continual improvement of energy performance",
                "Compliance with legal and other requirements",
                "Availability of information and resources",
                "Support for procurement of energy-efficient products",
            ],
        }

    def _json_energy_planning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON energy planning section."""
        return {
            "legal_requirements": data.get("legal_requirements", []),
            "seu_list": data.get("seu_list", []),
            "enpi_summary": data.get("enpi_summary", []),
            "enb_summary": data.get("enb_summary", []),
            "objectives": data.get("objectives", []),
            "targets": data.get("targets", []),
            "action_plans": data.get("action_plans", []),
            "iso_clauses": ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6"],
        }

    def _json_support_documentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON support documentation section."""
        meta = data.get("metadata", {})
        return {
            "resources": data.get("resources", []),
            "competence_requirements": data.get("competence_requirements", []),
            "awareness_program": meta.get("awareness_program", {}),
            "communication_plan": meta.get("communication_plan", []),
            "document_register": data.get("document_register", []),
            "iso_clauses": ["7.1", "7.2", "7.3", "7.4", "7.5"],
        }

    def _json_operational_controls(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON operational controls section."""
        meta = data.get("metadata", {})
        return {
            "controls": data.get("operational_controls", []),
            "design_considerations": meta.get("design_considerations", []),
            "procurement_requirements": meta.get("procurement_requirements", []),
            "iso_clauses": ["8.1", "8.2", "8.3"],
        }

    def _json_performance_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON performance evaluation section."""
        meta = data.get("metadata", {})
        return {
            "monitoring_plan": data.get("monitoring_plan", []),
            "compliance_evaluation": meta.get("compliance_evaluation", {}),
            "audit_schedule": data.get("audit_schedule", []),
            "management_reviews": data.get("management_reviews", []),
            "iso_clauses": ["9.1", "9.2", "9.3"],
        }

    def _json_improvement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON improvement section."""
        return {
            "nonconformities": data.get("nonconformities", []),
            "improvements": data.get("improvements", []),
            "iso_clauses": ["10.1", "10.2"],
        }

    def _json_appendices(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON appendices section."""
        meta = data.get("metadata", {})
        register = data.get("document_register", [])
        doc_by_clause: Dict[str, List[str]] = {}
        for doc in register:
            clause = doc.get("clause", "")
            if clause:
                doc_by_clause.setdefault(clause, []).append(doc.get("doc_id", "-"))
        cross_reference = []
        for ref in self.CLAUSE_CROSS_REFERENCE:
            clause = ref["clause"]
            cross_reference.append({
                "clause": clause,
                "title": ref["title"],
                "document_type": ref["document_type"],
                "document_refs": doc_by_clause.get(clause, []),
            })
        return {
            "document_register": register,
            "retention_schedule": meta.get("retention_schedule", self.DEFAULT_RETENTION_SCHEDULE),
            "cross_reference_matrix": cross_reference,
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
            "h3{color:#495057;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".policy-box{background:#e8f5e9;border-left:4px solid #198754;"
            "padding:15px 20px;margin:15px 0;border-radius:4px;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
            ".check-pass{color:#198754;font-weight:600;}"
            ".check-fail{color:#dc3545;font-weight:600;}"
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
