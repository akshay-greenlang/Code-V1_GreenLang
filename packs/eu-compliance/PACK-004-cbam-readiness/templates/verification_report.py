"""
VerificationReportTemplate - CBAM verification status report template.

This module implements the verification status report for CBAM compliance.
It generates formatted reports covering verification engagement overviews,
timelines, findings summaries, finding detail cards, materiality assessments,
verification statement status, and next verification scheduling.

Example:
    >>> template = VerificationReportTemplate()
    >>> data = {"engagement": {...}, "findings": [...], ...}
    >>> html = template.render_html(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class VerificationReportTemplate:
    """
    CBAM verification status report template.

    Generates formatted verification status reports for CBAM compliance,
    including engagement overviews, milestone timelines, findings summaries,
    materiality assessments, and verification statement tracking.

    Attributes:
        config: Optional configuration dictionary.
        generated_at: Timestamp of report generation.
    """

    FINDING_CATEGORIES: List[str] = [
        "data_accuracy",
        "methodology",
        "completeness",
        "documentation",
        "calculation",
        "reporting",
    ]

    SEVERITY_LEVELS: Dict[str, Dict[str, Any]] = {
        "critical": {"label": "Critical", "color": "#c0392b", "weight": 4},
        "major": {"label": "Major", "color": "#e74c3c", "weight": 3},
        "minor": {"label": "Minor", "color": "#f39c12", "weight": 2},
        "observation": {"label": "Observation", "color": "#3498db", "weight": 1},
    }

    ASSURANCE_LEVELS: List[str] = ["limited", "reasonable"]

    MATERIALITY_THRESHOLD_PCT: float = 5.0

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize VerificationReportTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - materiality_threshold_pct (float): Override 5% threshold.
                - show_finding_details (bool): Whether to show full detail cards.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the verification report as Markdown.

        Args:
            data: Report data dictionary containing:
                - engagement (dict): verifier, scope, assurance_level, status
                - timeline (list[dict]): milestones with dates and status
                - findings (list[dict]): verification findings
                - materiality (dict): materiality assessment data
                - verification_statement (dict): statement status
                - next_verification (dict): next due date and scope

        Returns:
            Formatted Markdown string.
        """
        sections: List[str] = []

        sections.append(self._md_header())
        sections.append(self._md_engagement_overview(data))
        sections.append(self._md_timeline(data))
        sections.append(self._md_findings_summary(data))
        sections.append(self._md_finding_details(data))
        sections.append(self._md_materiality_assessment(data))
        sections.append(self._md_verification_statement(data))
        sections.append(self._md_next_verification(data))
        sections.append(self._md_provenance_footer())

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the verification report as self-contained HTML.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_header())
        sections.append(self._html_engagement_overview(data))
        sections.append(self._html_timeline(data))
        sections.append(self._html_findings_summary(data))
        sections.append(self._html_finding_details(data))
        sections.append(self._html_materiality_assessment(data))
        sections.append(self._html_verification_statement(data))
        sections.append(self._html_next_verification(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title="CBAM Verification Status Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the verification report as a structured dict.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all report sections and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_verification_report",
            "generated_at": self.generated_at,
            "engagement": self._json_engagement_overview(data),
            "timeline": self._json_timeline(data),
            "findings_summary": self._json_findings_summary(data),
            "findings_detail": self._json_finding_details(data),
            "materiality_assessment": self._json_materiality_assessment(data),
            "verification_statement": self._json_verification_statement(data),
            "next_verification": self._json_next_verification(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown header."""
        return (
            "# CBAM Verification Status Report\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_engagement_overview(self, data: Dict[str, Any]) -> str:
        """Build Markdown engagement overview section."""
        eng: Dict[str, Any] = data.get("engagement", {})

        verifier = eng.get("verifier", "N/A")
        scope = eng.get("scope", "N/A")
        assurance = eng.get("assurance_level", "N/A")
        status = eng.get("status", "N/A")
        engagement_ref = eng.get("reference", "N/A")
        start_date = self._format_date(eng.get("start_date", "N/A"))
        end_date = self._format_date(eng.get("end_date", "N/A"))

        return (
            "## Engagement Overview\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Verification Body | {verifier} |\n"
            f"| Engagement Reference | {engagement_ref} |\n"
            f"| Scope | {scope} |\n"
            f"| Assurance Level | {assurance.capitalize()} |\n"
            f"| Status | {status.upper()} |\n"
            f"| Period | {start_date} to {end_date} |"
        )

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Build Markdown timeline section."""
        milestones: List[Dict[str, Any]] = data.get("timeline", [])

        if not milestones:
            return "## Verification Timeline\n\n*No timeline data available.*"

        header = (
            "## Verification Timeline\n\n"
            "| Date | Milestone | Status | Notes |\n"
            "|------|-----------|--------|-------|\n"
        )

        rows: List[str] = []
        for ms in milestones:
            status = ms.get("status", "pending")
            check = "[x]" if status == "completed" else "[ ]"

            rows.append(
                f"| {self._format_date(ms.get('date', ''))} | "
                f"{check} {ms.get('milestone', '')} | "
                f"{status.upper()} | "
                f"{ms.get('notes', '')} |"
            )

        return header + "\n".join(rows)

    def _md_findings_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown findings summary table."""
        findings: List[Dict[str, Any]] = data.get("findings", [])

        if not findings:
            return "## Findings Summary\n\n*No findings recorded.*"

        # Count by severity
        severity_counts: Dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "observation")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Count by status
        status_counts: Dict[str, int] = {}
        for f in findings:
            st = f.get("status", "open")
            status_counts[st] = status_counts.get(st, 0) + 1

        section = (
            "## Findings Summary\n\n"
            f"**Total Findings:** {len(findings)}\n\n"
            "### By Severity\n\n"
            "| Severity | Count |\n"
            "|----------|-------|\n"
        )

        for sev_key in ["critical", "major", "minor", "observation"]:
            count = severity_counts.get(sev_key, 0)
            if count > 0:
                label = self.SEVERITY_LEVELS.get(sev_key, {}).get("label", sev_key)
                section += f"| {label} | {count} |\n"

        section += (
            "\n### By Status\n\n"
            "| Status | Count |\n"
            "|--------|-------|\n"
        )

        for st in sorted(status_counts.keys()):
            section += f"| {st.capitalize()} | {status_counts[st]} |\n"

        # Summary table
        section += (
            "\n### Findings List\n\n"
            "| ID | Category | Severity | Status | Description |\n"
            "|----|----------|----------|--------|-------------|\n"
        )

        for f in findings:
            sev = f.get("severity", "observation")
            section += (
                f"| {f.get('finding_id', '')} | "
                f"{f.get('category', '').replace('_', ' ').title()} | "
                f"{sev.capitalize()} | "
                f"{f.get('status', '').capitalize()} | "
                f"{f.get('description', '')[:80]}{'...' if len(f.get('description', '')) > 80 else ''} |"
                "\n"
            )

        return section

    def _md_finding_details(self, data: Dict[str, Any]) -> str:
        """Build Markdown finding detail cards."""
        show = self.config.get("show_finding_details", True)
        if not show:
            return ""

        findings: List[Dict[str, Any]] = data.get("findings", [])

        if not findings:
            return ""

        section = "## Finding Details\n"

        for f in findings:
            section += (
                f"\n### {f.get('finding_id', 'N/A')}: {f.get('description', '')[:60]}\n\n"
                f"- **Category:** {f.get('category', '').replace('_', ' ').title()}\n"
                f"- **Severity:** {f.get('severity', 'N/A').capitalize()}\n"
                f"- **Status:** {f.get('status', 'N/A').capitalize()}\n\n"
                f"**Description:** {f.get('description', 'N/A')}\n\n"
                f"**Evidence:** {f.get('evidence', 'N/A')}\n\n"
                f"**Response:** {f.get('response', 'N/A')}\n\n"
                f"**Corrective Action:** {f.get('corrective_action', 'N/A')}\n\n"
                f"**Due Date:** {self._format_date(f.get('due_date', 'N/A'))}\n"
            )

        return section

    def _md_materiality_assessment(self, data: Dict[str, Any]) -> str:
        """Build Markdown materiality assessment section."""
        materiality: Dict[str, Any] = data.get("materiality", {})
        threshold = self.config.get(
            "materiality_threshold_pct", self.MATERIALITY_THRESHOLD_PCT
        )

        total_emissions = materiality.get("total_emissions_tco2e", 0.0)
        error_emissions = materiality.get("error_emissions_tco2e", 0.0)
        error_pct = (error_emissions / total_emissions * 100) if total_emissions > 0 else 0.0
        material = error_pct >= threshold

        return (
            "## Materiality Assessment\n\n"
            f"**Materiality Threshold:** {self._format_percentage(threshold)}\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Reported Emissions | {self._format_number(total_emissions)} tCO2e |\n"
            f"| Identified Errors | {self._format_number(error_emissions)} tCO2e |\n"
            f"| Error Rate | {self._format_percentage(error_pct)} |\n"
            f"| Materiality Threshold | {self._format_percentage(threshold)} |\n"
            f"| **Assessment** | **{'MATERIAL' if material else 'NOT MATERIAL'}** |\n\n"
            f"> {'Findings exceed the materiality threshold. Corrections required before statement issuance.' if material else 'Findings are below the materiality threshold. No material misstatements identified.'}"
        )

    def _md_verification_statement(self, data: Dict[str, Any]) -> str:
        """Build Markdown verification statement status section."""
        statement: Dict[str, Any] = data.get("verification_statement", {})

        status = statement.get("status", "pending")
        opinion = statement.get("opinion", "N/A")
        issued_date = self._format_date(statement.get("issued_date", "N/A"))
        valid_until = self._format_date(statement.get("valid_until", "N/A"))
        qualifications: List[str] = statement.get("qualifications", [])

        section = (
            "## Verification Statement Status\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Status | {status.upper()} |\n"
            f"| Opinion | {opinion.capitalize()} |\n"
            f"| Issued Date | {issued_date} |\n"
            f"| Valid Until | {valid_until} |\n"
        )

        if qualifications:
            section += "\n**Qualifications:**\n\n"
            for q in qualifications:
                section += f"- {q}\n"

        return section

    def _md_next_verification(self, data: Dict[str, Any]) -> str:
        """Build Markdown next verification section."""
        nxt: Dict[str, Any] = data.get("next_verification", {})

        due_date = self._format_date(nxt.get("due_date", "N/A"))
        scope = nxt.get("scope", "N/A")
        verifier = nxt.get("verifier", "To be determined")
        preparation_items: List[str] = nxt.get("preparation_items", [])

        section = (
            "## Next Verification\n\n"
            f"- **Due Date:** {due_date}\n"
            f"- **Scope:** {scope}\n"
            f"- **Verifier:** {verifier}\n"
        )

        if preparation_items:
            section += "\n**Preparation Items:**\n\n"
            for item in preparation_items:
                section += f"- [ ] {item}\n"

        return section

    def _md_provenance_footer(self) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: VerificationReportTemplate v1.0*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Verification Status Report</h1>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_engagement_overview(self, data: Dict[str, Any]) -> str:
        """Build HTML engagement overview."""
        eng: Dict[str, Any] = data.get("engagement", {})

        status = eng.get("status", "pending")
        status_color = {
            "completed": "#2ecc71",
            "in_progress": "#3498db",
            "pending": "#f39c12",
            "cancelled": "#e74c3c",
        }.get(status, "#95a5a6")

        return (
            '<div class="section"><h2>Engagement Overview</h2>'
            '<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Verifier</div>'
            f'<div class="kpi-value" style="font-size:18px">{eng.get("verifier", "N/A")}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Assurance Level</div>'
            f'<div class="kpi-value" style="font-size:18px">'
            f'{eng.get("assurance_level", "N/A").capitalize()}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Status</div>'
            f'<div class="kpi-value" style="font-size:18px;color:{status_color}">'
            f'{status.upper()}</div></div>'
            '</div>'
            '<table><tbody>'
            f'<tr><td><strong>Reference</strong></td><td>{eng.get("reference", "N/A")}</td></tr>'
            f'<tr><td><strong>Scope</strong></td><td>{eng.get("scope", "N/A")}</td></tr>'
            f'<tr><td><strong>Period</strong></td>'
            f'<td>{self._format_date(eng.get("start_date", "N/A"))} to '
            f'{self._format_date(eng.get("end_date", "N/A"))}</td></tr>'
            '</tbody></table></div>'
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        """Build HTML timeline section."""
        milestones: List[Dict[str, Any]] = data.get("timeline", [])

        if not milestones:
            return (
                '<div class="section"><h2>Verification Timeline</h2>'
                '<p class="note">No timeline data available.</p></div>'
            )

        items_html = ""
        for ms in milestones:
            status = ms.get("status", "pending")
            completed = status == "completed"
            color = "#2ecc71" if completed else "#f39c12" if status == "in_progress" else "#95a5a6"

            items_html += (
                f'<div class="timeline-item">'
                f'<div class="timeline-dot" style="background:{color}"></div>'
                f'<div class="timeline-content">'
                f'<div class="timeline-date">{self._format_date(ms.get("date", ""))}</div>'
                f'<div class="timeline-milestone">{ms.get("milestone", "")}</div>'
                f'<div class="timeline-status" style="color:{color}">{status.upper()}</div>'
                f'</div></div>'
            )

        return f'<div class="section"><h2>Verification Timeline</h2><div class="timeline">{items_html}</div></div>'

    def _html_findings_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML findings summary."""
        findings: List[Dict[str, Any]] = data.get("findings", [])

        if not findings:
            return (
                '<div class="section"><h2>Findings Summary</h2>'
                '<p class="note">No findings recorded.</p></div>'
            )

        # Count by severity
        severity_counts: Dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "observation")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        badges_html = ""
        for sev_key in ["critical", "major", "minor", "observation"]:
            count = severity_counts.get(sev_key, 0)
            info = self.SEVERITY_LEVELS.get(sev_key, {})
            badges_html += (
                f'<div class="severity-card" style="border-left:4px solid {info.get("color", "#95a5a6")}">'
                f'<div class="severity-count">{count}</div>'
                f'<div class="severity-label">{info.get("label", sev_key)}</div></div>'
            )

        rows_html = ""
        for f in findings:
            sev = f.get("severity", "observation")
            sev_info = self.SEVERITY_LEVELS.get(sev, {})
            color = sev_info.get("color", "#95a5a6")
            st = f.get("status", "open")
            st_color = "#2ecc71" if st == "closed" else "#f39c12" if st == "in_progress" else "#e74c3c"

            rows_html += (
                f'<tr>'
                f'<td class="code">{f.get("finding_id", "")}</td>'
                f'<td>{f.get("category", "").replace("_", " ").title()}</td>'
                f'<td><span class="severity-badge" style="background:{color}">'
                f'{sev.capitalize()}</span></td>'
                f'<td><span class="status-badge" style="background:{st_color}">'
                f'{st.capitalize()}</span></td>'
                f'<td>{f.get("description", "")[:80]}</td></tr>'
            )

        return (
            '<div class="section"><h2>Findings Summary</h2>'
            f'<div class="severity-grid">{badges_html}</div>'
            '<table><thead><tr>'
            '<th>ID</th><th>Category</th><th>Severity</th>'
            '<th>Status</th><th>Description</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_finding_details(self, data: Dict[str, Any]) -> str:
        """Build HTML finding detail cards."""
        show = self.config.get("show_finding_details", True)
        if not show:
            return ""

        findings: List[Dict[str, Any]] = data.get("findings", [])
        if not findings:
            return ""

        cards_html = ""
        for f in findings:
            sev = f.get("severity", "observation")
            sev_info = self.SEVERITY_LEVELS.get(sev, {})
            color = sev_info.get("color", "#95a5a6")

            cards_html += (
                f'<div class="finding-card" style="border-left:4px solid {color}">'
                f'<div class="finding-header">'
                f'<strong>{f.get("finding_id", "")}</strong>'
                f'<span class="severity-badge" style="background:{color}">'
                f'{sev.capitalize()}</span>'
                f'<span>{f.get("status", "").capitalize()}</span></div>'
                f'<div class="finding-body">'
                f'<p><strong>Description:</strong> {f.get("description", "N/A")}</p>'
                f'<p><strong>Evidence:</strong> {f.get("evidence", "N/A")}</p>'
                f'<p><strong>Response:</strong> {f.get("response", "N/A")}</p>'
                f'<p><strong>Corrective Action:</strong> {f.get("corrective_action", "N/A")}</p>'
                f'<p><strong>Due Date:</strong> {self._format_date(f.get("due_date", "N/A"))}</p>'
                f'</div></div>'
            )

        return f'<div class="section"><h2>Finding Details</h2>{cards_html}</div>'

    def _html_materiality_assessment(self, data: Dict[str, Any]) -> str:
        """Build HTML materiality assessment."""
        materiality: Dict[str, Any] = data.get("materiality", {})
        threshold = self.config.get(
            "materiality_threshold_pct", self.MATERIALITY_THRESHOLD_PCT
        )

        total = materiality.get("total_emissions_tco2e", 0.0)
        errors = materiality.get("error_emissions_tco2e", 0.0)
        error_pct = (errors / total * 100) if total > 0 else 0.0
        material = error_pct >= threshold
        color = "#e74c3c" if material else "#2ecc71"

        return (
            '<div class="section"><h2>Materiality Assessment</h2>'
            '<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Total Emissions</div>'
            f'<div class="kpi-value" style="font-size:18px">'
            f'{self._format_number(total)} tCO2e</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Identified Errors</div>'
            f'<div class="kpi-value" style="font-size:18px">'
            f'{self._format_number(errors)} tCO2e</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Error Rate</div>'
            f'<div class="kpi-value" style="font-size:18px;color:{color}">'
            f'{self._format_percentage(error_pct)}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Assessment</div>'
            f'<div class="kpi-value" style="font-size:18px;color:{color}">'
            f'{"MATERIAL" if material else "NOT MATERIAL"}</div></div>'
            '</div>'
            f'<p class="note">Materiality threshold: {self._format_percentage(threshold)}. '
            f'{"Findings exceed threshold - corrections required." if material else "Below threshold - no material misstatements."}'
            '</p></div>'
        )

    def _html_verification_statement(self, data: Dict[str, Any]) -> str:
        """Build HTML verification statement status."""
        statement: Dict[str, Any] = data.get("verification_statement", {})

        status = statement.get("status", "pending")
        opinion = statement.get("opinion", "N/A")
        qualifications: List[str] = statement.get("qualifications", [])

        status_color = {
            "issued": "#2ecc71",
            "pending": "#f39c12",
            "draft": "#3498db",
            "withdrawn": "#e74c3c",
        }.get(status, "#95a5a6")

        qual_html = ""
        if qualifications:
            qual_items = "".join(f"<li>{q}</li>" for q in qualifications)
            qual_html = f'<div class="qualifications"><strong>Qualifications:</strong><ul>{qual_items}</ul></div>'

        return (
            '<div class="section"><h2>Verification Statement</h2>'
            '<table><tbody>'
            f'<tr><td><strong>Status</strong></td>'
            f'<td style="color:{status_color};font-weight:bold">{status.upper()}</td></tr>'
            f'<tr><td><strong>Opinion</strong></td><td>{opinion.capitalize()}</td></tr>'
            f'<tr><td><strong>Issued Date</strong></td>'
            f'<td>{self._format_date(statement.get("issued_date", "N/A"))}</td></tr>'
            f'<tr><td><strong>Valid Until</strong></td>'
            f'<td>{self._format_date(statement.get("valid_until", "N/A"))}</td></tr>'
            f'</tbody></table>{qual_html}</div>'
        )

    def _html_next_verification(self, data: Dict[str, Any]) -> str:
        """Build HTML next verification section."""
        nxt: Dict[str, Any] = data.get("next_verification", {})

        preparation: List[str] = nxt.get("preparation_items", [])
        prep_html = ""
        if preparation:
            items = "".join(
                f'<div class="prep-item"><span class="checkbox">&#9744;</span> {item}</div>'
                for item in preparation
            )
            prep_html = f'<div class="prep-list"><strong>Preparation Items:</strong>{items}</div>'

        return (
            '<div class="section"><h2>Next Verification</h2>'
            '<table><tbody>'
            f'<tr><td><strong>Due Date</strong></td>'
            f'<td>{self._format_date(nxt.get("due_date", "N/A"))}</td></tr>'
            f'<tr><td><strong>Scope</strong></td><td>{nxt.get("scope", "N/A")}</td></tr>'
            f'<tr><td><strong>Verifier</strong></td>'
            f'<td>{nxt.get("verifier", "To be determined")}</td></tr>'
            f'</tbody></table>{prep_html}</div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_engagement_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON engagement overview."""
        eng: Dict[str, Any] = data.get("engagement", {})
        return {
            "verifier": eng.get("verifier", ""),
            "reference": eng.get("reference", ""),
            "scope": eng.get("scope", ""),
            "assurance_level": eng.get("assurance_level", ""),
            "status": eng.get("status", ""),
            "start_date": self._format_date(eng.get("start_date", "")),
            "end_date": self._format_date(eng.get("end_date", "")),
        }

    def _json_timeline(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON timeline."""
        milestones: List[Dict[str, Any]] = data.get("timeline", [])
        return [
            {
                "date": self._format_date(ms.get("date", "")),
                "milestone": ms.get("milestone", ""),
                "status": ms.get("status", "pending"),
                "notes": ms.get("notes", ""),
            }
            for ms in milestones
        ]

    def _json_findings_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON findings summary."""
        findings: List[Dict[str, Any]] = data.get("findings", [])

        severity_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}

        for f in findings:
            sev = f.get("severity", "observation")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            st = f.get("status", "open")
            status_counts[st] = status_counts.get(st, 0) + 1

        return {
            "total_findings": len(findings),
            "by_severity": severity_counts,
            "by_status": status_counts,
        }

    def _json_finding_details(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON finding details."""
        findings: List[Dict[str, Any]] = data.get("findings", [])
        return [
            {
                "finding_id": f.get("finding_id", ""),
                "category": f.get("category", ""),
                "severity": f.get("severity", "observation"),
                "status": f.get("status", "open"),
                "description": f.get("description", ""),
                "evidence": f.get("evidence", ""),
                "response": f.get("response", ""),
                "corrective_action": f.get("corrective_action", ""),
                "due_date": self._format_date(f.get("due_date", "")),
            }
            for f in findings
        ]

    def _json_materiality_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON materiality assessment."""
        materiality: Dict[str, Any] = data.get("materiality", {})
        threshold = self.config.get(
            "materiality_threshold_pct", self.MATERIALITY_THRESHOLD_PCT
        )

        total = materiality.get("total_emissions_tco2e", 0.0)
        errors = materiality.get("error_emissions_tco2e", 0.0)
        error_pct = (errors / total * 100) if total > 0 else 0.0

        return {
            "total_emissions_tco2e": round(total, 2),
            "error_emissions_tco2e": round(errors, 2),
            "error_rate_pct": round(error_pct, 2),
            "threshold_pct": threshold,
            "is_material": error_pct >= threshold,
        }

    def _json_verification_statement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON verification statement."""
        statement: Dict[str, Any] = data.get("verification_statement", {})
        return {
            "status": statement.get("status", "pending"),
            "opinion": statement.get("opinion", ""),
            "issued_date": self._format_date(statement.get("issued_date", "")),
            "valid_until": self._format_date(statement.get("valid_until", "")),
            "qualifications": statement.get("qualifications", []),
        }

    def _json_next_verification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON next verification."""
        nxt: Dict[str, Any] = data.get("next_verification", {})
        return {
            "due_date": self._format_date(nxt.get("due_date", "")),
            "scope": nxt.get("scope", ""),
            "verifier": nxt.get("verifier", "To be determined"),
            "preparation_items": nxt.get("preparation_items", []),
        }

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _format_number(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format a numeric value with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """Format a percentage value."""
        return f"{value:.2f}%"

    def _format_date(self, dt: Union[datetime, str]) -> str:
        """Format a datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if len(dt) >= 10 else dt
        return dt.strftime("%Y-%m-%d")

    def _format_currency(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format a currency value."""
        return f"{currency} {value:,.2f}"

    def _format_cn_code(self, code: str) -> str:
        """Format a CN code to standard XXXX.XX format."""
        clean = code.replace(".", "").replace(" ", "")
        if len(clean) >= 6:
            return f"{clean[:4]}.{clean[4:6]}"
        elif len(clean) == 4:
            return f"{clean}.00"
        return code

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".code{font-family:monospace;font-size:13px}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:24px;font-weight:700;color:#1a5276}"
            ".severity-grid{display:grid;grid-template-columns:repeat(4,1fr);"
            "gap:12px;margin-bottom:16px}"
            ".severity-card{background:#f8f9fa;padding:12px;border-radius:6px;text-align:center}"
            ".severity-count{font-size:28px;font-weight:700}"
            ".severity-label{font-size:12px;color:#7f8c8d}"
            ".severity-badge,.status-badge{display:inline-block;padding:2px 8px;"
            "border-radius:4px;color:#fff;font-size:11px;font-weight:bold}"
            ".timeline{position:relative;padding-left:24px}"
            ".timeline-item{display:flex;align-items:flex-start;margin-bottom:16px;"
            "position:relative}"
            ".timeline-dot{width:12px;height:12px;border-radius:50%;margin-right:12px;"
            "margin-top:4px;flex-shrink:0}"
            ".timeline-content{flex:1}"
            ".timeline-date{font-size:12px;color:#7f8c8d}"
            ".timeline-milestone{font-size:14px;font-weight:600}"
            ".timeline-status{font-size:12px;font-weight:bold}"
            ".finding-card{background:#f8f9fa;padding:16px;border-radius:8px;"
            "margin-bottom:12px}"
            ".finding-header{display:flex;align-items:center;gap:8px;"
            "margin-bottom:8px;flex-wrap:wrap}"
            ".finding-body p{margin:4px 0;font-size:14px}"
            ".qualifications{margin-top:12px;padding:12px;background:#f8f9fa;"
            "border-radius:6px}"
            ".qualifications ul{margin:8px 0;padding-left:20px}"
            ".prep-list{margin-top:12px}"
            ".prep-item{padding:4px 0;font-size:14px}"
            ".checkbox{margin-right:8px;font-size:16px}"
            ".note{color:#7f8c8d;font-style:italic;font-size:13px;margin-top:8px}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )

        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: VerificationReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
