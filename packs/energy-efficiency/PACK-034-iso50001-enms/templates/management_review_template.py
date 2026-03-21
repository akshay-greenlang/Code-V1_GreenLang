# -*- coding: utf-8 -*-
"""
ManagementReviewTemplate - ISO 50001 Clause 9.3 Management Review for PACK-034.

Generates comprehensive management review reports aligned with ISO 50001:2018
Clause 9.3. Covers meeting information, attendees, review of previous
actions, energy performance summary with EnPI trends, policy review,
objectives status, audit results summary, NC/CA status, resource adequacy,
risks and opportunities, decisions and actions, and next review date.

Sections:
    1. Meeting Information
    2. Attendees
    3. Review of Previous Actions
    4. Energy Performance Summary
    5. Policy Review
    6. Objectives Status
    7. Audit Results Summary
    8. NC/CA Status
    9. Resource Adequacy
    10. Risks & Opportunities
    11. Decisions & Actions
    12. Next Review Date

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ManagementReviewTemplate:
    """
    ISO 50001 management review report template.

    Renders management review reports aligned with ISO 50001:2018
    Clause 9.3, covering all required review inputs and outputs
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ManagementReviewTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render management review report as Markdown.

        Args:
            data: Management review data including meeting_info,
                  previous_actions, enpi_summaries, policy_review,
                  objectives_status, audit_summary, and decisions.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_meeting_info(data),
            self._md_attendees(data),
            self._md_previous_actions(data),
            self._md_energy_performance(data),
            self._md_policy_review(data),
            self._md_objectives_status(data),
            self._md_audit_results(data),
            self._md_nc_ca_status(data),
            self._md_resource_adequacy(data),
            self._md_risks_opportunities(data),
            self._md_decisions_actions(data),
            self._md_next_review(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render management review report as self-contained HTML.

        Args:
            data: Management review data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_meeting_info(data),
            self._html_attendees(data),
            self._html_energy_performance(data),
            self._html_objectives_status(data),
            self._html_audit_results(data),
            self._html_nc_ca_status(data),
            self._html_risks_opportunities(data),
            self._html_decisions_actions(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Management Review - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render management review report as structured JSON.

        Args:
            data: Management review data dict.

        Returns:
            Dict with structured review sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "management_review",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "meeting_info": data.get("meeting_info", {}),
            "attendees": data.get("attendees", []),
            "previous_actions": data.get("previous_actions", []),
            "enpi_summaries": data.get("enpi_summaries", []),
            "policy_review": data.get("policy_review", {}),
            "objectives_status": data.get("objectives_status", []),
            "audit_summary": data.get("audit_summary", {}),
            "nc_ca_status": self._json_nc_ca_status(data),
            "resource_adequacy": data.get("resource_adequacy", {}),
            "risks_opportunities": data.get("risks_opportunities", []),
            "decisions": data.get("decisions", []),
            "next_review_date": data.get("next_review_date", ""),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with review metadata."""
        info = data.get("meeting_info", {})
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Management Review Report\n\n"
            f"**Organization:** {info.get('organization', '')}  \n"
            f"**Meeting Date:** {info.get('meeting_date', '')}  \n"
            f"**Review Period:** {info.get('review_period', '')}  \n"
            f"**ISO 50001:2018 Clause:** 9.3  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 ManagementReviewTemplate v34.0.0\n\n---"
        )

    def _md_meeting_info(self, data: Dict[str, Any]) -> str:
        """Render meeting information section."""
        info = data.get("meeting_info", {})
        return (
            "## 1. Meeting Information\n\n"
            f"- **Meeting Date:** {info.get('meeting_date', '-')}\n"
            f"- **Location:** {info.get('location', '-')}\n"
            f"- **Chairperson:** {info.get('chairperson', '-')}\n"
            f"- **Secretary:** {info.get('secretary', '-')}\n"
            f"- **Review Period:** {info.get('review_period', '-')}\n"
            f"- **Previous Review Date:** {info.get('previous_review_date', '-')}"
        )

    def _md_attendees(self, data: Dict[str, Any]) -> str:
        """Render attendees section."""
        attendees = data.get("attendees", [])
        if not attendees:
            return "## 2. Attendees\n\n_No attendees listed._"
        lines = [
            "## 2. Attendees\n",
            "| Name | Title / Role | Present |",
            "|------|-------------|---------|",
        ]
        for a in attendees:
            present = "Yes" if a.get("present", True) else "Apology"
            lines.append(
                f"| {a.get('name', '-')} "
                f"| {a.get('title', '-')} "
                f"| {present} |"
            )
        return "\n".join(lines)

    def _md_previous_actions(self, data: Dict[str, Any]) -> str:
        """Render review of previous actions section."""
        actions = data.get("previous_actions", [])
        if not actions:
            return "## 3. Review of Previous Actions\n\n_No previous actions to review._"
        lines = [
            "## 3. Review of Previous Actions\n",
            "| # | Action | Owner | Due Date | Status | Remarks |",
            "|---|--------|-------|----------|--------|---------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('owner', '-')} "
                f"| {a.get('due_date', '-')} "
                f"| {a.get('status', '-')} "
                f"| {a.get('remarks', '-')} |"
            )
        return "\n".join(lines)

    def _md_energy_performance(self, data: Dict[str, Any]) -> str:
        """Render energy performance summary section."""
        summaries = data.get("enpi_summaries", [])
        if not summaries:
            return "## 4. Energy Performance Summary\n\n_No EnPI data available._"
        lines = [
            "## 4. Energy Performance Summary\n",
            "### EnPI Trends\n",
            "| EnPI | Baseline | Target | Current | Trend | Status |",
            "|------|----------|--------|---------|-------|--------|",
        ]
        for s in summaries:
            lines.append(
                f"| {s.get('enpi', '-')} "
                f"| {self._fmt(s.get('baseline', 0))} {s.get('unit', '')} "
                f"| {s.get('target', '-')} "
                f"| {self._fmt(s.get('current', 0))} {s.get('unit', '')} "
                f"| {s.get('trend', '-')} "
                f"| {s.get('status', '-')} |"
            )
        perf = data.get("performance_summary", {})
        if perf:
            lines.extend([
                "",
                f"**Total Energy Savings:** {self._format_energy(perf.get('total_savings_mwh', 0))}  ",
                f"**Total Cost Savings:** {self._format_currency(perf.get('total_cost_savings', 0))}  ",
                f"**CO2 Avoided:** {self._fmt(perf.get('co2_avoided_tonnes', 0))} tonnes",
            ])
        return "\n".join(lines)

    def _md_policy_review(self, data: Dict[str, Any]) -> str:
        """Render policy review section."""
        review = data.get("policy_review", {})
        return (
            "## 5. Policy Review\n\n"
            f"**Policy Suitability:** {review.get('suitability', 'Adequate')}  \n"
            f"**Changes Required:** {review.get('changes_required', 'None')}  \n"
            f"**Last Updated:** {review.get('last_updated', '-')}  \n"
            f"**Recommendation:** {review.get('recommendation', 'Maintain current policy')}"
        )

    def _md_objectives_status(self, data: Dict[str, Any]) -> str:
        """Render objectives status section."""
        objectives = data.get("objectives_status", [])
        if not objectives:
            return "## 6. Objectives Status\n\n_No objectives to review._"
        lines = [
            "## 6. Objectives Status\n",
            "| Objective | Target | Progress (%) | On Track | Remarks |",
            "|-----------|--------|-------------|----------|---------|",
        ]
        for o in objectives:
            on_track = "Yes" if o.get("on_track", True) else "No"
            lines.append(
                f"| {o.get('objective', '-')} "
                f"| {o.get('target', '-')} "
                f"| {self._fmt(o.get('progress_pct', 0), 0)}% "
                f"| {on_track} "
                f"| {o.get('remarks', '-')} |"
            )
        return "\n".join(lines)

    def _md_audit_results(self, data: Dict[str, Any]) -> str:
        """Render audit results summary section."""
        audit = data.get("audit_summary", {})
        return (
            "## 7. Audit Results Summary\n\n"
            f"**Last Audit Date:** {audit.get('last_audit_date', '-')}  \n"
            f"**Audit Type:** {audit.get('audit_type', '-')}  \n"
            f"**Overall Finding:** {audit.get('overall_finding', '-')}  \n"
            f"**Major NCs:** {audit.get('major_ncs', 0)}  \n"
            f"**Minor NCs:** {audit.get('minor_ncs', 0)}  \n"
            f"**OFIs:** {audit.get('ofis', 0)}  \n"
            f"**Closure Rate:** {self._fmt(audit.get('closure_rate_pct', 0))}%"
        )

    def _md_nc_ca_status(self, data: Dict[str, Any]) -> str:
        """Render NC/CA status section."""
        ncs = data.get("nc_ca_status", [])
        if not ncs:
            return "## 8. NC/CA Status\n\n_No open nonconformities._"
        lines = [
            "## 8. NC/CA Status\n",
            "| NC Ref | Description | Severity | Corrective Action | Due Date | Status |",
            "|--------|-----------|----------|------------------|----------|--------|",
        ]
        for nc in ncs:
            lines.append(
                f"| {nc.get('nc_ref', '-')} "
                f"| {nc.get('description', '-')} "
                f"| {nc.get('severity', '-')} "
                f"| {nc.get('corrective_action', '-')} "
                f"| {nc.get('due_date', '-')} "
                f"| {nc.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_resource_adequacy(self, data: Dict[str, Any]) -> str:
        """Render resource adequacy section."""
        resources = data.get("resource_adequacy", {})
        return (
            "## 9. Resource Adequacy\n\n"
            f"**Personnel Resources:** {resources.get('personnel', 'Adequate')}  \n"
            f"**Financial Resources:** {resources.get('financial', 'Adequate')}  \n"
            f"**Technology Resources:** {resources.get('technology', 'Adequate')}  \n"
            f"**Training Status:** {resources.get('training', 'Up to date')}  \n"
            f"**Additional Needs:** {resources.get('additional_needs', 'None identified')}"
        )

    def _md_risks_opportunities(self, data: Dict[str, Any]) -> str:
        """Render risks and opportunities section."""
        items = data.get("risks_opportunities", [])
        if not items:
            return "## 10. Risks & Opportunities\n\n_No new risks or opportunities identified._"
        lines = [
            "## 10. Risks & Opportunities\n",
            "| Type | Description | Impact | Likelihood | Action Required |",
            "|------|-----------|--------|-----------|----------------|",
        ]
        for item in items:
            lines.append(
                f"| {item.get('type', '-')} "
                f"| {item.get('description', '-')} "
                f"| {item.get('impact', '-')} "
                f"| {item.get('likelihood', '-')} "
                f"| {item.get('action_required', '-')} |"
            )
        return "\n".join(lines)

    def _md_decisions_actions(self, data: Dict[str, Any]) -> str:
        """Render decisions and actions section."""
        decisions = data.get("decisions", [])
        if not decisions:
            return "## 11. Decisions & Actions\n\n_No decisions recorded._"
        lines = [
            "## 11. Decisions & Actions\n",
            "| # | Decision / Action | Owner | Due Date | Priority |",
            "|---|------------------|-------|----------|----------|",
        ]
        for i, d in enumerate(decisions, 1):
            lines.append(
                f"| {i} | {d.get('decision', '-')} "
                f"| {d.get('owner', '-')} "
                f"| {d.get('due_date', '-')} "
                f"| {d.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_next_review(self, data: Dict[str, Any]) -> str:
        """Render next review date section."""
        return (
            "## 12. Next Review\n\n"
            f"**Next Review Date:** {data.get('next_review_date', 'To be scheduled')}  \n"
            f"**Prepared By:** {data.get('prepared_by', '___________________')}  \n"
            f"**Approved By:** {data.get('approved_by', '___________________')}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        info = data.get("meeting_info", {})
        return (
            f'<h1>Management Review Report</h1>\n'
            f'<p class="subtitle">Organization: {info.get("organization", "-")} | '
            f'Date: {info.get("meeting_date", "-")} | '
            f'ISO 50001 Clause 9.3</p>'
        )

    def _html_meeting_info(self, data: Dict[str, Any]) -> str:
        """Render HTML meeting information."""
        info = data.get("meeting_info", {})
        return (
            '<h2>1. Meeting Information</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Date</span>'
            f'<span class="value-sm">{info.get("meeting_date", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Chairperson</span>'
            f'<span class="value-sm">{info.get("chairperson", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Review Period</span>'
            f'<span class="value-sm">{info.get("review_period", "-")}</span></div>\n'
            '</div>'
        )

    def _html_attendees(self, data: Dict[str, Any]) -> str:
        """Render HTML attendees."""
        attendees = data.get("attendees", [])
        rows = ""
        for a in attendees:
            rows += (
                f'<tr><td>{a.get("name", "-")}</td>'
                f'<td>{a.get("title", "-")}</td>'
                f'<td>{"Present" if a.get("present", True) else "Apology"}</td></tr>\n'
            )
        return (
            '<h2>2. Attendees</h2>\n'
            '<table>\n<tr><th>Name</th><th>Title</th>'
            f'<th>Attendance</th></tr>\n{rows}</table>'
        )

    def _html_energy_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML energy performance summary."""
        summaries = data.get("enpi_summaries", [])
        rows = ""
        for s in summaries:
            status = s.get("status", "").lower()
            cls = "status-improved" if "improv" in status else "status-declined" if "declin" in status else ""
            rows += (
                f'<tr><td>{s.get("enpi", "-")}</td>'
                f'<td>{self._fmt(s.get("baseline", 0))} {s.get("unit", "")}</td>'
                f'<td>{self._fmt(s.get("current", 0))} {s.get("unit", "")}</td>'
                f'<td>{s.get("trend", "-")}</td>'
                f'<td class="{cls}">{s.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Energy Performance Summary</h2>\n'
            '<table>\n<tr><th>EnPI</th><th>Baseline</th>'
            f'<th>Current</th><th>Trend</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_objectives_status(self, data: Dict[str, Any]) -> str:
        """Render HTML objectives status."""
        objectives = data.get("objectives_status", [])
        rows = ""
        for o in objectives:
            on_track = o.get("on_track", True)
            cls = "status-improved" if on_track else "status-declined"
            rows += (
                f'<tr><td>{o.get("objective", "-")}</td>'
                f'<td>{o.get("target", "-")}</td>'
                f'<td>{self._fmt(o.get("progress_pct", 0), 0)}%</td>'
                f'<td class="{cls}">{"On Track" if on_track else "Behind"}</td></tr>\n'
            )
        return (
            '<h2>6. Objectives Status</h2>\n'
            '<table>\n<tr><th>Objective</th><th>Target</th>'
            f'<th>Progress</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_audit_results(self, data: Dict[str, Any]) -> str:
        """Render HTML audit results summary."""
        audit = data.get("audit_summary", {})
        return (
            '<h2>7. Audit Results Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Overall Finding</span>'
            f'<span class="value-sm">{audit.get("overall_finding", "-")}</span></div>\n'
            f'  <div class="card card-danger"><span class="label">Major NCs</span>'
            f'<span class="value">{audit.get("major_ncs", 0)}</span></div>\n'
            f'  <div class="card card-warning"><span class="label">Minor NCs</span>'
            f'<span class="value">{audit.get("minor_ncs", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Closure Rate</span>'
            f'<span class="value">{self._fmt(audit.get("closure_rate_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_nc_ca_status(self, data: Dict[str, Any]) -> str:
        """Render HTML NC/CA status."""
        ncs = data.get("nc_ca_status", [])
        rows = ""
        for nc in ncs:
            status_cls = "status-closed" if nc.get("status", "").lower() == "closed" else "status-open"
            rows += (
                f'<tr><td>{nc.get("nc_ref", "-")}</td>'
                f'<td>{nc.get("description", "-")}</td>'
                f'<td>{nc.get("severity", "-")}</td>'
                f'<td class="{status_cls}">{nc.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>8. NC/CA Status</h2>\n'
            '<table>\n<tr><th>NC Ref</th><th>Description</th>'
            f'<th>Severity</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_risks_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML risks and opportunities."""
        items = data.get("risks_opportunities", [])
        rows = ""
        for item in items:
            type_cls = "risk-item" if item.get("type", "").lower() == "risk" else "opp-item"
            rows += (
                f'<tr class="{type_cls}"><td>{item.get("type", "-")}</td>'
                f'<td>{item.get("description", "-")}</td>'
                f'<td>{item.get("impact", "-")}</td>'
                f'<td>{item.get("action_required", "-")}</td></tr>\n'
            )
        return (
            '<h2>10. Risks & Opportunities</h2>\n'
            '<table>\n<tr><th>Type</th><th>Description</th>'
            f'<th>Impact</th><th>Action</th></tr>\n{rows}</table>'
        )

    def _html_decisions_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML decisions and actions."""
        decisions = data.get("decisions", [])
        rows = ""
        for d in decisions:
            rows += (
                f'<tr><td>{d.get("decision", "-")}</td>'
                f'<td>{d.get("owner", "-")}</td>'
                f'<td>{d.get("due_date", "-")}</td>'
                f'<td>{d.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>11. Decisions & Actions</h2>\n'
            '<table>\n<tr><th>Decision / Action</th><th>Owner</th>'
            f'<th>Due Date</th><th>Priority</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_nc_ca_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON NC/CA status summary."""
        ncs = data.get("nc_ca_status", [])
        return {
            "total": len(ncs),
            "open": sum(1 for n in ncs if n.get("status", "").lower() != "closed"),
            "closed": sum(1 for n in ncs if n.get("status", "").lower() == "closed"),
            "overdue": sum(1 for n in ncs if n.get("status", "").lower() == "overdue"),
            "items": ncs,
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:120px;}"
            ".card-danger{border-left:4px solid #dc3545;}"
            ".card-warning{border-left:4px solid #ffc107;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".value-sm{display:block;font-size:1em;font-weight:600;color:#495057;}"
            ".status-improved{color:#198754;font-weight:600;}"
            ".status-declined{color:#dc3545;font-weight:600;}"
            ".status-open{color:#dc3545;font-weight:600;}"
            ".status-closed{color:#198754;font-weight:600;}"
            ".risk-item{background:#fff3cd;}"
            ".opp-item{background:#d1e7dd;}"
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

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
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
