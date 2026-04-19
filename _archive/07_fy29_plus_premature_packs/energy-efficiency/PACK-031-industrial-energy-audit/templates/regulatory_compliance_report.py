# -*- coding: utf-8 -*-
"""
RegulatoryComplianceReportTemplate - Energy regulatory compliance summary for PACK-031.

Generates regulatory compliance summary reports covering Energy Efficiency
Directive (EED) obligations, audit scheduling, ISO 50001 certification status,
EU ETS obligations, national energy regulation requirements, and deadline
tracking with gap analysis.

Sections:
    1. Executive Summary
    2. EED Compliance Status
    3. Audit Schedule & History
    4. ISO 50001 Status
    5. EU ETS Obligations
    6. National Requirements
    7. Deadline Tracking
    8. Gap Analysis & Action Plan

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegulatoryComplianceReportTemplate:
    """
    Energy regulatory compliance summary report template.

    Renders EED, ISO 50001, EU ETS, and national energy regulation
    compliance status with gap analysis, deadline tracking, and action
    plans across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    EU_REGULATIONS: List[str] = [
        "EED 2023/1791 (Recast)",
        "EU ETS Directive 2003/87/EC",
        "Energy Performance of Buildings Directive (EPBD)",
        "Renewable Energy Directive (RED III)",
        "Corporate Sustainability Reporting Directive (CSRD)",
    ]

    COMPLIANCE_STATES: Dict[str, str] = {
        "compliant": "Fully Compliant",
        "partial": "Partially Compliant",
        "non_compliant": "Non-Compliant",
        "exempt": "Exempt",
        "pending": "Assessment Pending",
    }

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
        """Render regulatory compliance summary as Markdown.

        Args:
            data: Compliance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_eed_status(data),
            self._md_audit_schedule(data),
            self._md_iso50001_status(data),
            self._md_eu_ets(data),
            self._md_national_requirements(data),
            self._md_deadline_tracking(data),
            self._md_gap_analysis(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render regulatory compliance summary as HTML.

        Args:
            data: Compliance engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_eed_status(data),
            self._html_iso50001_status(data),
            self._html_deadline_tracking(data),
            self._html_gap_analysis(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Regulatory Compliance Summary</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render regulatory compliance summary as structured JSON.

        Args:
            data: Compliance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "regulatory_compliance_report",
            "version": "31.0.0",
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "eed_status": data.get("eed_status", {}),
            "audit_schedule": data.get("audit_schedule", {}),
            "iso50001_status": data.get("iso50001_status", {}),
            "eu_ets": data.get("eu_ets", {}),
            "national_requirements": data.get("national_requirements", []),
            "deadlines": data.get("deadlines", []),
            "gap_analysis": data.get("gap_analysis", []),
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
        org = data.get("organization_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Regulatory Compliance Summary\n\n"
            f"**Organization:** {org}  \n"
            f"**Jurisdiction:** {data.get('jurisdiction', 'EU')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 RegulatoryComplianceReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        s = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Regulations Tracked | {s.get('regulations_tracked', 0)} |\n"
            f"| Fully Compliant | {s.get('compliant_count', 0)} |\n"
            f"| Partially Compliant | {s.get('partial_count', 0)} |\n"
            f"| Non-Compliant | {s.get('non_compliant_count', 0)} |\n"
            f"| Overall Compliance Score | {self._fmt(s.get('overall_score_pct', 0))}% |\n"
            f"| Open Action Items | {s.get('open_actions', 0)} |\n"
            f"| Next Deadline | {s.get('next_deadline', '-')} |\n"
            f"| EED Audit Status | {s.get('eed_audit_status', '-')} |\n"
            f"| ISO 50001 Status | {s.get('iso50001_status', '-')} |"
        )

    def _md_eed_status(self, data: Dict[str, Any]) -> str:
        """Render EED compliance status section."""
        eed = data.get("eed_status", {})
        requirements = eed.get("requirements", [])
        lines = [
            "## 2. EED Compliance Status\n",
            f"**EED Version:** EED 2023/1791 (Recast)  ",
            f"**Enterprise Category:** {eed.get('enterprise_category', '-')}  ",
            f"**Energy Consumption Threshold:** {self._fmt(eed.get('threshold_tj', 0))} TJ  ",
            f"**Actual Consumption:** {self._fmt(eed.get('actual_consumption_tj', 0))} TJ  ",
            f"**Audit Obligation:** {eed.get('audit_obligation', '-')}  ",
            f"**EnMS Obligation:** {eed.get('enms_obligation', '-')}  ",
            f"**Overall EED Status:** {eed.get('overall_status', '-')}",
        ]
        if requirements:
            lines.extend([
                "\n### EED Requirements Checklist\n",
                "| Requirement | Status | Due Date | Evidence |",
                "|------------|--------|----------|----------|",
            ])
            for req in requirements:
                lines.append(
                    f"| {req.get('requirement', '-')} "
                    f"| {req.get('status', '-')} "
                    f"| {req.get('due_date', '-')} "
                    f"| {req.get('evidence', '-')} |"
                )
        return "\n".join(lines)

    def _md_audit_schedule(self, data: Dict[str, Any]) -> str:
        """Render audit schedule and history section."""
        schedule = data.get("audit_schedule", {})
        history = schedule.get("history", [])
        upcoming = schedule.get("upcoming", [])
        lines = [
            "## 3. Audit Schedule & History\n",
            f"**Audit Cycle:** {schedule.get('cycle_years', 4)} years  ",
            f"**Last Audit Date:** {schedule.get('last_audit_date', '-')}  ",
            f"**Next Audit Due:** {schedule.get('next_audit_due', '-')}  ",
            f"**Audit Standard:** {schedule.get('audit_standard', 'EN 16247-1')}",
        ]
        if history:
            lines.extend([
                "\n### Audit History\n",
                "| Date | Auditor | Standard | Sites | Status | Findings |",
                "|------|---------|----------|-------|--------|----------|",
            ])
            for h in history:
                lines.append(
                    f"| {h.get('date', '-')} "
                    f"| {h.get('auditor', '-')} "
                    f"| {h.get('standard', '-')} "
                    f"| {h.get('sites_covered', '-')} "
                    f"| {h.get('status', '-')} "
                    f"| {h.get('findings_count', 0)} |"
                )
        if upcoming:
            lines.extend([
                "\n### Upcoming Audits\n",
                "| Site | Planned Date | Scope | Auditor |",
                "|------|-------------|-------|---------|",
            ])
            for u in upcoming:
                lines.append(
                    f"| {u.get('site', '-')} "
                    f"| {u.get('planned_date', '-')} "
                    f"| {u.get('scope', '-')} "
                    f"| {u.get('auditor', '-')} |"
                )
        return "\n".join(lines)

    def _md_iso50001_status(self, data: Dict[str, Any]) -> str:
        """Render ISO 50001 status section."""
        iso = data.get("iso50001_status", {})
        clauses = iso.get("clause_status", [])
        lines = [
            "## 4. ISO 50001 Status\n",
            f"**Certification Status:** {iso.get('certification_status', '-')}  ",
            f"**Certification Body:** {iso.get('certification_body', '-')}  ",
            f"**Valid Until:** {iso.get('valid_until', '-')}  ",
            f"**Scope:** {iso.get('scope', '-')}  ",
            f"**Last Surveillance Audit:** {iso.get('last_surveillance', '-')}  ",
            f"**Next Surveillance Audit:** {iso.get('next_surveillance', '-')}  ",
            f"**Nonconformities Open:** {iso.get('open_nonconformities', 0)}",
        ]
        if clauses:
            lines.extend([
                "\n### Clause Compliance Status\n",
                "| Clause | Title | Status | Notes |",
                "|--------|-------|--------|-------|",
            ])
            for c in clauses:
                lines.append(
                    f"| {c.get('clause', '-')} "
                    f"| {c.get('title', '-')} "
                    f"| {c.get('status', '-')} "
                    f"| {c.get('notes', '-')} |"
                )
        return "\n".join(lines)

    def _md_eu_ets(self, data: Dict[str, Any]) -> str:
        """Render EU ETS obligations section."""
        ets = data.get("eu_ets", {})
        if not ets or not ets.get("applicable", True):
            return "## 5. EU ETS Obligations\n\n_EU ETS not applicable to this facility._"
        lines = [
            "## 5. EU ETS Obligations\n",
            f"**ETS Phase:** {ets.get('phase', 'Phase 4')}  ",
            f"**Installation Category:** {ets.get('category', '-')}  ",
            f"**Annual Emissions:** {self._fmt(ets.get('annual_emissions_tco2', 0))} tCO2  ",
            f"**Free Allocation:** {self._fmt(ets.get('free_allocation_tco2', 0))} tCO2  ",
            f"**Shortfall:** {self._fmt(ets.get('shortfall_tco2', 0))} tCO2  ",
            f"**Carbon Price (est.):** EUR {self._fmt(ets.get('carbon_price_eur', 0))}/tCO2  ",
            f"**Estimated Cost:** EUR {self._fmt(ets.get('estimated_cost_eur', 0))}  ",
            f"**MRV Plan Status:** {ets.get('mrv_plan_status', '-')}  ",
            f"**Verification Status:** {ets.get('verification_status', '-')}",
        ]
        return "\n".join(lines)

    def _md_national_requirements(self, data: Dict[str, Any]) -> str:
        """Render national requirements section."""
        nat = data.get("national_requirements", [])
        if not nat:
            return "## 6. National Requirements\n\n_No national requirements tracked._"
        lines = [
            "## 6. National Requirements\n",
            "| Regulation | Country | Requirement | Status | Deadline | Notes |",
            "|-----------|---------|------------|--------|----------|-------|",
        ]
        for r in nat:
            lines.append(
                f"| {r.get('regulation', '-')} "
                f"| {r.get('country', '-')} "
                f"| {r.get('requirement', '-')} "
                f"| {r.get('status', '-')} "
                f"| {r.get('deadline', '-')} "
                f"| {r.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_deadline_tracking(self, data: Dict[str, Any]) -> str:
        """Render deadline tracking section."""
        deadlines = data.get("deadlines", [])
        if not deadlines:
            return "## 7. Deadline Tracking\n\n_No upcoming deadlines._"
        lines = [
            "## 7. Deadline Tracking\n",
            "| Deadline | Regulation | Action Required | Days Remaining | Status |",
            "|----------|-----------|----------------|---------------|--------|",
        ]
        for d in deadlines:
            urgency = self._urgency_label(d.get("days_remaining", 999))
            lines.append(
                f"| {d.get('deadline', '-')} "
                f"| {d.get('regulation', '-')} "
                f"| {d.get('action_required', '-')} "
                f"| {d.get('days_remaining', '-')} {urgency} "
                f"| {d.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis and action plan section."""
        gaps = data.get("gap_analysis", [])
        actions = data.get("action_plan", [])
        lines = ["## 8. Gap Analysis & Action Plan\n"]
        if gaps:
            lines.extend([
                "### Compliance Gaps\n",
                "| Regulation | Requirement | Current State | Gap | "
                "Risk Level | Effort (weeks) |",
                "|-----------|------------|---------------|-----|"
                "-----------|----------------|",
            ])
            for g in gaps:
                lines.append(
                    f"| {g.get('regulation', '-')} "
                    f"| {g.get('requirement', '-')} "
                    f"| {g.get('current_state', '-')} "
                    f"| {g.get('gap', '-')} "
                    f"| {g.get('risk_level', '-')} "
                    f"| {g.get('effort_weeks', '-')} |"
                )
        else:
            lines.append("### Compliance Gaps\n\n_No gaps identified._")
        if actions:
            lines.extend([
                "\n### Action Plan\n",
                "| # | Action | Regulation | Owner | Deadline | Priority | Status |",
                "|---|--------|-----------|-------|----------|----------|--------|",
            ])
            for i, a in enumerate(actions, 1):
                lines.append(
                    f"| {i} | {a.get('action', '-')} "
                    f"| {a.get('regulation', '-')} "
                    f"| {a.get('owner', '-')} "
                    f"| {a.get('deadline', '-')} "
                    f"| {a.get('priority', '-')} "
                    f"| {a.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>Regulatory Compliance Summary</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'Jurisdiction: {data.get("jurisdiction", "EU")}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Compliance Score</span>'
            f'<span class="value">{self._fmt(s.get("overall_score_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Compliant</span>'
            f'<span class="value">{s.get("compliant_count", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Non-Compliant</span>'
            f'<span class="value" style="color:#dc2626;">'
            f'{s.get("non_compliant_count", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Open Actions</span>'
            f'<span class="value">{s.get("open_actions", 0)}</span></div>\n'
            '</div>'
        )

    def _html_eed_status(self, data: Dict[str, Any]) -> str:
        """Render HTML EED status."""
        eed = data.get("eed_status", {})
        return (
            '<h2>EED Compliance</h2>\n'
            f'<p>Status: <strong>{eed.get("overall_status", "-")}</strong> | '
            f'Consumption: {self._fmt(eed.get("actual_consumption_tj", 0))} TJ | '
            f'Audit: {eed.get("audit_obligation", "-")}</p>'
        )

    def _html_iso50001_status(self, data: Dict[str, Any]) -> str:
        """Render HTML ISO 50001 status."""
        iso = data.get("iso50001_status", {})
        return (
            '<h2>ISO 50001</h2>\n'
            f'<p>Certification: <strong>{iso.get("certification_status", "-")}</strong> | '
            f'Valid Until: {iso.get("valid_until", "-")} | '
            f'Open NCs: {iso.get("open_nonconformities", 0)}</p>'
        )

    def _html_deadline_tracking(self, data: Dict[str, Any]) -> str:
        """Render HTML deadline tracking."""
        deadlines = data.get("deadlines", [])
        rows = ""
        for d in deadlines:
            days = d.get("days_remaining", 999)
            color = "#dc2626" if days < 30 else "#d97706" if days < 90 else "#059669"
            rows += (
                f'<tr><td>{d.get("deadline", "-")}</td>'
                f'<td>{d.get("regulation", "-")}</td>'
                f'<td style="color:{color};font-weight:700;">{days} days</td>'
                f'<td>{d.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>Deadline Tracking</h2>\n<table>\n'
            '<tr><th>Deadline</th><th>Regulation</th><th>Days Left</th>'
            f'<th>Status</th></tr>\n{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gaps = data.get("gap_analysis", [])
        rows = ""
        for g in gaps:
            rows += (
                f'<tr><td>{g.get("regulation", "-")}</td>'
                f'<td>{g.get("requirement", "-")}</td>'
                f'<td>{g.get("gap", "-")}</td>'
                f'<td>{g.get("risk_level", "-")}</td></tr>\n'
            )
        return (
            '<h2>Gap Analysis</h2>\n<table>\n'
            '<tr><th>Regulation</th><th>Requirement</th><th>Gap</th>'
            f'<th>Risk</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        return data.get("executive_summary", {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _urgency_label(self, days: int) -> str:
        """Return urgency label based on remaining days.

        Args:
            days: Days remaining until deadline.

        Returns:
            Urgency label string.
        """
        if days < 0:
            return "[OVERDUE]"
        elif days < 30:
            return "[URGENT]"
        elif days < 90:
            return "[SOON]"
        return ""

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
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
